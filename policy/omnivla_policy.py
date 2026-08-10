"""
OmniVLA-edge inference wrapper.

`model_omnivla_edge.py` next to this file is vendored verbatim from
https://github.com/NHirose/OmniVLA (MIT). Everything here mirrors the
preprocessing in that repo's `inference/run_omnivla_edge.py`. Preprocessing that
drifts from training does not raise — it just makes the rover drive badly — so
keep this file in sync with upstream rather than "improving" it.
"""

import math
import os
from typing import List, Optional, Sequence, Tuple

import clip
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

from .model_omnivla_edge import OmniVLA_edge

# Architecture of the shipped omnivla-edge checkpoint. best.pth loads against
# this with strict=True; if it ever doesn't, the checkpoint is a different model.
MODEL_PARAMS = dict(
    context_size=5,
    len_traj_pred=8,
    learn_angle=True,
    obs_encoder="efficientnet-b0",
    obs_encoding_size=1024,
    late_fusion=False,
    mha_num_attention_heads=4,
    mha_num_attention_layers=4,
    mha_ff_dim_factor=4,
)

# The checkpoint is ~415 MB, well past GitHub's 100 MB file limit, so it is
# published as a release asset instead of being committed. Assets live outside
# git history, so cloning stays cheap. The tag pins the code the weights were
# trained against — preprocessing drift degrades driving silently, so the pair
# matters.
DEFAULT_CHECKPOINT = "best.pth"
CHECKPOINT_URL = (
    "https://github.com/minsong0206/frodobot_server/releases/download/"
    "omnivla-v1/best.pth"
)

CLIP_TYPE = "ViT-B/32"
IMG_SIZE = (96, 96)  # obs / goal / satellite tokens
IMG_SIZE_FILM = (224, 224)  # FiLM language branch; fixed by the 2x2x1024=4096 head
CONTEXT_LEN = MODEL_PARAMS["context_size"] + 1  # 6 frames: current + 5 history

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> (1, 3, H, W), ImageNet-normalized.

    Equivalent to upstream `transform_images_PIL_mask` with an all-ones mask:
    that path does to_tensor(img * ones) / 255 -> Normalize, which is the same
    as to_tensor -> Normalize for a uint8 PIL image.
    """
    return _normalize(TF.to_tensor(img)).unsqueeze(0)


def select_modality(
    *, pose: bool, satellite: bool, image: bool, language: bool
) -> int:
    """Modality id -> row of `OmniVLA_edge.all_masks`, from upstream's if-chain.

    Token order is (obs x6, pose, satellite, goal_image, language), and the id
    picks which of the last four get masked out of attention.
    """
    table = {
        (False, True, False, False): 0,  # satellite only
        (True, True, False, False): 1,  # pose + satellite
        (False, True, True, False): 2,  # satellite + goal image
        (True, True, True, False): 3,  # all (no language)
        (True, False, False, False): 4,  # pose only
        (True, False, True, False): 5,  # pose + goal image
        (False, False, True, False): 6,  # goal image only
        (False, False, False, True): 7,  # language only
        (True, False, False, True): 8,  # pose + language
        (False, False, True, True): 9,  # goal image + language
    }
    key = (pose, satellite, image, language)
    if key not in table:
        raise ValueError(
            f"OmniVLA-edge has no mask for modality combination {key} "
            "(pose, satellite, image, language)"
        )
    return table[key]


class OmniVLAEdgePolicy:
    """Loads omnivla-edge + CLIP and turns camera frames into waypoints."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda:0",
        clip_type: str = CLIP_TYPE,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            # OmniVLA_edge.forward calls obs_img.get_device(), which returns -1
            # on CPU and then blows up on .to(-1). Upstream is CUDA-only.
            raise ValueError(
                "OmniVLA_edge only runs on CUDA (its forward pass calls "
                "Tensor.get_device()); pass a cuda device."
            )

        self.model = OmniVLA_edge(**MODEL_PARAMS)
        state_dict = _load_state_dict(ckpt_path)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()

        text_encoder, _ = clip.load(clip_type, device=self.device)
        self.text_encoder = text_encoder.to(torch.float32).to(self.device).eval()
        self._text_cache: dict = {}

        # Satellite imagery is not available on the rover, so both map slots are
        # black. They are still normalized, exactly like upstream's dummy input.
        black = Image.new("RGB", IMG_SIZE, color=(0, 0, 0))
        self._black_map = _to_tensor(black).to(self.device)
        self._black_goal_img = _to_tensor(black).to(self.device)

    # -- text -------------------------------------------------------------

    @torch.no_grad()
    def encode_text(self, prompt: str) -> torch.Tensor:
        """CLIP text feature (1, 512), cached — the prompt rarely changes."""
        if prompt not in self._text_cache:
            tokens = clip.tokenize(prompt, truncate=True).to(self.device)
            self._text_cache[prompt] = self.text_encoder.encode_text(tokens)
        return self._text_cache[prompt]

    # -- inference --------------------------------------------------------

    @torch.no_grad()
    def predict_waypoints(
        self,
        context_frames: Sequence[Image.Image],
        prompt: Optional[str] = None,
        goal_pose: Optional[Sequence[float]] = None,
        goal_image: Optional[Image.Image] = None,
    ) -> Tuple[np.ndarray, int]:
        """Run one forward pass.

        Args:
            context_frames: exactly CONTEXT_LEN frames, oldest first, newest
                last. Any size — they are resized here.
            prompt: language instruction, or None to disable the language token.
            goal_pose: 4-vector [y/0.1, -x/0.1, cos(dyaw), sin(dyaw)] in the
                robot frame, or None to disable the pose token.
            goal_image: egocentric goal image, or None to disable that token.

        Returns:
            (waypoints, modality_id) where waypoints is (8, 4): cumulative
            (dx, dy) in units of 0.1 m plus a normalized (cos, sin) heading,
            in the robot frame (x forward, y left).
        """
        if len(context_frames) != CONTEXT_LEN:
            raise ValueError(
                f"expected {CONTEXT_LEN} context frames, got {len(context_frames)}"
            )

        modality_id = select_modality(
            pose=goal_pose is not None,
            satellite=False,
            image=goal_image is not None,
            language=prompt is not None,
        )

        small = [f.resize(IMG_SIZE) for f in context_frames]
        obs_images = torch.cat([_to_tensor(f) for f in small], dim=1).to(self.device)
        obs_image_cur = obs_images[:, -3:, :, :]

        # (1, 9, 96, 96): satellite_current + satellite_goal + current obs
        map_images = torch.cat(
            (self._black_map, self._black_map, obs_image_cur), dim=1
        )

        cur_large_img = _to_tensor(
            context_frames[-1].resize(IMG_SIZE_FILM)
        ).to(self.device)

        if goal_image is not None:
            goal_image_t = _to_tensor(goal_image.resize(IMG_SIZE)).to(self.device)
        else:
            goal_image_t = self._black_goal_img

        if goal_pose is not None:
            goal_pose_t = torch.tensor(
                list(goal_pose), dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        else:
            goal_pose_t = torch.zeros((1, 4), dtype=torch.float32, device=self.device)

        # Upstream feeds the placeholder "xxxx" when language is masked out.
        feat_text = self.encode_text(prompt if prompt is not None else "xxxx")

        mask_t = torch.tensor([modality_id], dtype=torch.long, device=self.device)

        action_pred, _dist_pred, _mask = self.model(
            obs_images,
            goal_pose_t,
            map_images,
            goal_image_t,
            mask_t,
            feat_text,
            cur_large_img,
        )
        return action_pred[0].float().cpu().numpy(), modality_id


def checkpoint_missing_message(path: str) -> str:
    return (
        f"checkpoint not found: {path}\n"
        f"It is a release asset, not part of the repository. Download it with:\n"
        f"    curl -L -o {path} {CHECKPOINT_URL}"
    )


def _load_state_dict(ckpt_path: str) -> dict:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(checkpoint_missing_message(ckpt_path))
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0
        ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"{ckpt_path} is not a state_dict")
    # Some training runs save the DDP-wrapped module.
    return {k[7:] if k.startswith("module.") else k: v for k, v in ckpt.items()}


def gps_goal_to_pose(
    current_lat: float,
    current_lon: float,
    current_heading_deg: float,
    goal_lat: float,
    goal_lon: float,
    goal_heading_deg: float = 0.0,
    metric_waypoint_spacing: float = 0.1,
    max_distance_m: float = 30.0,
) -> List[float]:
    """Convert a GPS goal into the model's 4-vector pose input.

    Ported from upstream `run_omnivla_edge.py`. EXPERIMENTAL for this rover:
    it assumes `current_heading_deg` is a compass bearing in degrees, which has
    not been verified against the rover's `orientation` field, and it needs a
    real GPS fix (the rover reports lat/lon = 1000 when it has none).
    """
    import utm  # imported lazily so the language-only path needs no GPS deps

    cur = utm.from_latlon(current_lat, current_lon)
    goal = utm.from_latlon(goal_lat, goal_lon)

    cur_compass = -math.radians(current_heading_deg)  # inverted, as upstream
    goal_compass = -math.radians(goal_heading_deg)

    dx, dy = goal[0] - cur[0], goal[1] - cur[1]
    rel_x = dx * math.cos(cur_compass) + dy * math.sin(cur_compass)
    rel_y = -dx * math.sin(cur_compass) + dy * math.cos(cur_compass)

    radius = math.hypot(rel_x, rel_y)
    if radius > max_distance_m:
        rel_x *= max_distance_m / radius
        rel_y *= max_distance_m / radius

    return [
        rel_y / metric_waypoint_spacing,
        -rel_x / metric_waypoint_spacing,
        math.cos(goal_compass - cur_compass),
        math.sin(goal_compass - cur_compass),
    ]


def relative_goal_to_pose(
    forward_m: float,
    left_m: float,
    heading_rad: float = 0.0,
    metric_waypoint_spacing: float = 0.1,
) -> List[float]:
    """Build the pose input from a goal given directly in the robot frame.

    Same semantics as `gps_goal_to_pose`, minus the GPS: a navigation goal
    relative to where the rover is now. Upstream's sample does exactly this to
    test the pose modality without a fix (run_omnivla_edge.py hardcodes a
    relative point over the GPS-derived one), which is the main use here too.
    """
    return [
        left_m / metric_waypoint_spacing,
        -forward_m / metric_waypoint_spacing,
        math.cos(heading_rad),
        math.sin(heading_rad),
    ]
