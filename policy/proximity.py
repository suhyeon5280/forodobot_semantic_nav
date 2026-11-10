"""Tell when one surface fills the view straight ahead.

Driving up to something solid -- a long plain skirt, a box, a wall -- the middle
of the picture stops being a scene and becomes one surface: its colour spread
collapses. The sides still show the room or the street around it. So two
numbers, on a horizontal band of the frame:

    center_std   colour spread in the middle of the band
    ratio        center_std / outer_std, the middle against the two side strips

    ┌──────────────────────────────────┐
    │   above the band: sky / ceiling  │   not used: flat on its own
    ├─────┬──────────────────────┬─────┤   band top    (default 0.30)
    │outer│        center        │outer│
    ├─────┴──────────────────────┴─────┤   band bottom (default 0.60)
    │   below the band: floor / road   │   not used: flat on its own
    └──────────────────────────────────┘

A frame counts as full when the ratio is low. The ratio alone decides by
default: on the rover, center_std on its own moved too much with lighting,
exposure and the folds of the fabric to put a fixed number on, while the ratio
compares two parts of the same frame and cancels those out. `max_center_std`
adds center_std as a second condition; 0 leaves it out.

A wall that fills the whole view does not count -- the sides are flat too --
so `max_ratio <= 0` turns the ratio check off for that case, which then needs
`max_center_std` set.

What "spread" means is the part that has to survive real fabric. A fold is the
same colour in shadow, and a sheen is the same colour washed out towards white:
both change brightness, not colour, and a plain RGB spread counts them as a
mix of colours, so a folded, shiny skirt never reads as one surface. The
default `spread="colour"` therefore

  * drops the brightest `trim_bright` and darkest `trim_dark` of the pixels,
    which is where sheen and the deepest fold shadows sit;
  * divides each remaining pixel by its own brightness, so shaded red and lit
    red become the same red, and takes the spread of that;
  * adds back a little of the brightness spread (`brightness_weight`), so a grey
    wall and grey asphalt, which have no colour to compare, still differ.

`spread="rgb"` is the plain standard deviation of the RGB values (0-255). The two
are on different scales, so `max_center_std` has to be read in the units of the
spread in use. The frame is shrunk first, which also smooths away fabric weave
and video compression noise.

A narrower center fills from further away. That distance matters more than the
thresholds: the loop needs about a second to react (video delay plus `ticks`
confirmations), which at 0.3 m/s is 30 cm.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from PIL import Image


@dataclass
class FillConfig:
    band: Tuple[float, float] = (0.30, 0.60)
    center_width: float = 0.20
    outer_width: float = 0.15
    max_center_std: float = 0.0
    max_ratio: float = 0.35
    ticks: int = 2
    size: Tuple[int, int] = (160, 90)
    spread: str = "colour"
    trim_bright: float = 0.25
    trim_dark: float = 0.10
    brightness_weight: float = 0.25

    def validate(self) -> None:
        top, bottom = self.band
        if not 0.0 <= top < bottom <= 1.0:
            raise ValueError(f"band must satisfy 0 <= top < bottom <= 1, got {self.band}")
        if not 0.0 < self.center_width < 1.0 - 2 * self.outer_width:
            raise ValueError(
                f"center_width must be between 0 and {1.0 - 2 * self.outer_width:.2f} "
                f"so it does not overlap the side strips, got {self.center_width}"
            )
        if self.ticks < 1:
            raise ValueError(f"ticks must be at least 1, got {self.ticks}")
        if self.max_center_std <= 0 and self.max_ratio <= 0:
            raise ValueError(
                "the ratio and centre checks are both off, so every frame "
                "would count as full; set at least one above 0"
            )
        if self.spread not in ("colour", "rgb"):
            raise ValueError(f"spread must be 'colour' or 'rgb', got {self.spread!r}")
        if not (0.0 <= self.trim_dark and 0.0 <= self.trim_bright
                and self.trim_dark + self.trim_bright <= 0.8):
            raise ValueError(
                "trim_dark and trim_bright must be at least 0 and leave at "
                f"least 20% of the pixels, got {self.trim_dark} and {self.trim_bright}"
            )


def describe_fill_rule(config: FillConfig) -> str:
    """The full-frame condition in words, for logs and reports."""
    parts = []
    if config.max_ratio > 0:
        parts.append(f"ratio < {config.max_ratio:.2f}")
    if config.max_center_std > 0:
        parts.append(f"centre spread < {config.max_center_std:.1f}")
    rule = " and ".join(parts)
    if config.spread == "colour":
        return (f"{rule} (colour spread, brightest {config.trim_bright:.0%} and "
                f"darkest {config.trim_dark:.0%} dropped)")
    return f"{rule} (RGB spread)"


# Below this the brightness is mostly sensor noise, and dividing by it would
# turn near-black pixels into wild colours.
_MIN_BRIGHTNESS = 20.0


def _spread(region: np.ndarray, config: FillConfig) -> float:
    pixels = region.reshape(-1, 3)
    if config.spread == "rgb":
        return float(np.sqrt(pixels.var(axis=0).mean()))
    brightness = pixels.mean(axis=1)
    low, high = np.quantile(brightness, [config.trim_dark, 1.0 - config.trim_bright])
    keep = (brightness >= low) & (brightness <= high)
    if keep.sum() < 2:
        keep[:] = True
    pixels, brightness = pixels[keep], brightness[keep]
    colour = pixels / np.maximum(brightness, _MIN_BRIGHTNESS)[:, None] * 100.0
    return float(np.sqrt(colour.var(axis=0).mean() + config.brightness_weight * brightness.var()))


def measure(image: Image.Image, config: FillConfig) -> Dict[str, float]:
    """`center_std`, `outer_std`, `ratio`, and `full` for one frame."""
    small = np.asarray(image.convert("RGB").resize(config.size, Image.BILINEAR), np.float32)
    height, width = small.shape[:2]
    top = int(round(config.band[0] * height))
    bottom = max(top + 1, int(round(config.band[1] * height)))
    band = small[top:bottom]

    half = config.center_width / 2.0
    left = int(round((0.5 - half) * width))
    right = max(left + 1, int(round((0.5 + half) * width)))
    side = max(1, int(round(config.outer_width * width)))

    center_std = _spread(band[:, left:right], config)
    outer_std = _spread(
        np.concatenate([band[:, :side], band[:, width - side:]], axis=1), config
    )
    # A frame that is flat everywhere (lens covered, stream blank) comes out
    # at ratio 0 and counts as full: not seeing is a reason to stop.
    ratio = center_std / outer_std if outer_std > 1e-6 else 0.0
    full = (config.max_ratio <= 0 or ratio < config.max_ratio) and (
        config.max_center_std <= 0 or center_std < config.max_center_std
    )
    return {
        "center_std": round(center_std, 2),
        "outer_std": round(outer_std, 2),
        "ratio": round(ratio, 3),
        "full": bool(full),
    }
