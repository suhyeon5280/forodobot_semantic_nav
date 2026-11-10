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

The spread is the standard deviation of the RGB values (0-255), averaged over
the three channels, so a red skirt and a grey wall of the same brightness still
differ. The frame is shrunk first, which also smooths away fabric weave and
video compression noise.

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


def describe_fill_rule(config: FillConfig) -> str:
    """The full-frame condition in words, for logs and reports."""
    parts = []
    if config.max_ratio > 0:
        parts.append(f"ratio < {config.max_ratio:.2f}")
    if config.max_center_std > 0:
        parts.append(f"centre spread < {config.max_center_std:.1f}")
    return " and ".join(parts)


def _spread(region: np.ndarray) -> float:
    return float(np.sqrt(region.reshape(-1, 3).var(axis=0).mean()))


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

    center_std = _spread(band[:, left:right])
    outer_std = _spread(np.concatenate([band[:, :side], band[:, width - side:]], axis=1))
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
