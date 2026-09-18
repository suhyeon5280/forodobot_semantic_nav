"""arm-4' policy: OmniVLA-edge steered by a heatmap channel.

Interface-compatible with `OmniVLAEdgePolicy`, so `AutonomyLoop` cannot tell the
difference: `predict_waypoints(frames, prompt=, goal_pose=, goal_image=)` still
returns `((8, 4) ndarray, modality_id)`.

Three things differ from the shipped upstream policy:
  (a) the checkpoint is the fine-tuned arm-4' (FiLM first conv is 3 -> 4 ch);
  (b) `current_img` is rebuilt every tick -- RGB zeroed, heatmap as 4th channel;
  (c) the prompt is parsed and grounded before the policy ever sees it.

The policy forward pass and the waypoint -> command conversion are untouched.

Ported from the reference implementations, which live in repositories that are
read-only from here (imported, never modified):

    <edge_vlm>/experiments/omnivla/run_autonomy_ours.py   this class
    <edge_vlm>/experiments/omnivla/d150_e2e.py            the numbers we match
    <edge_vlm>/experiments/omnivla/arm_model.py           ArmModel, the 4ch wrapper
    <edge_vlm>/experiments/omnivla/heatmap.py             HeatmapProducer
    <OmniVLA_edge>/train/vint_train/models/il/il.py       IL_gps_map_mask3_lan2

Runtime: the frodo_lan environment plus <edge_vlm>/.omni_deps, which is where
ultralytics and open_clip are installed. This module appends that directory to
sys.path itself, so exporting PYTHONPATH is optional.

Two divergences from `run_autonomy_ours.py`, both requested and both outside the
path the validation frame exercises (see `check_ours.py`):

  * no candidate survives detection -> upstream falls back to the arm-1 model.
    Here a single synthetic candidate box is placed on the A heatmap peak
    instead, so the field test never silently swaps models underneath itself.
  * the prompt has no relation word -> upstream skips detection and feeds a
    whole-frame heatmap. Here A becomes the whole sentence, B is None, and
    detection plus selection still run with the distance term dropped.
"""

import contextlib
import json
import logging
import os
import re
import sys
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("autonomy.ours")

# ---------------------------------------------------------------------------
# Where the read-only repositories live. Overridable so this is not pinned to
# one machine's home directory.
# ---------------------------------------------------------------------------
EDGE_VLM_ROOT = os.environ.get("EDGE_VLM_ROOT", "/home/shy/suhyeon/edge_vlm")
OMNIVLA_TRAIN_ROOT = os.environ.get(
    "OMNIVLA_TRAIN_ROOT", "/home/shy/suhyeon/OmniVLA_edge/train"
)
OMNI_DEPS = os.environ.get("OMNI_DEPS", os.path.join(EDGE_VLM_ROOT, ".omni_deps"))

# Fine-tuned policy weights. Same architecture as the arm-1 original except the
# FiLM branch's first conv takes 4 input channels instead of 3.
DEFAULT_ARM4_CHECKPOINT = os.path.join(
    EDGE_VLM_ROOT, "results/phase4/omnivla/arm4p_s0/latest.pth"
)
ADAPTER_PATH = os.path.join(
    EDGE_VLM_ROOT, "results/phase4/d79_ladder/act4_cl6159_s2.pt"
)
LOC_HEAD_PATH = os.path.join(
    EDGE_VLM_ROOT, "results/phase4/loc_head_full/full_H1_linear_lr0.001_s0.pt"
)
DETECTOR_WEIGHTS = os.path.join(EDGE_VLM_ROOT, "yolov8n.pt")

# Config files the reference reads its constants from. Read-only, and the single
# source for them -- this module introduces no new tunables of its own.
_CFG_INTEGRATION = os.path.join(
    EDGE_VLM_ROOT, "configs/experiment/omnivla_integration.yaml"
)
_CFG_E2E = os.path.join(EDGE_VLM_ROOT, "configs/experiment/d150_e2e.yaml")
_CFG_EVAL_SETS = os.path.join(
    EDGE_VLM_ROOT, "configs/experiment/omnivla_eval_sets.yaml"
)

# Field log location, as registered in the reference notes (D154 section 6).
DEFAULT_TICK_LOG = os.path.join(
    EDGE_VLM_ROOT, "results/phase4/omnivla/d154_field_log"
)

# CLIP patch grid the heatmap is pooled onto, and the diagonal that normalizes
# the distance term. ViT-B/16 at 224 px gives 14x14 patches.
PATCH_GRID = 14
_DIAG = float(np.sqrt(2.0))

# "A next to B" / beside / near. Everything else is treated as a bare target.
RELATION = re.compile(r"\s+(?:next to|beside|near)\s+")

# Language-only modality: the row of `all_masks` that masks pose, satellite and
# goal image out of attention. arm-4' is only ever driven by language.
MODALITY_LANGUAGE_ONLY = 7

# The spec leaves the synthetic box's size open, so this is a choice, not a
# ported constant: a quarter of the frame per side, centred on the peak, which
# covers roughly 3x3 patches of the 14x14 grid.
PEAK_BOX_SIZE = 0.25

_PATHS_READY = False


@contextlib.contextmanager
def _edge_vlm_cwd():
    """Import the reference modules from the edge_vlm root, then change back.

    Several of them read data files by relative path at import time, so they
    only import cleanly from that directory. The deployment process resolves its
    own paths relative to the repository root, so the change is undone straight
    away rather than left in place.
    """
    previous = os.getcwd()
    os.chdir(EDGE_VLM_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def _bootstrap_paths() -> None:
    """Put the read-only repositories and the extra deps on `sys.path`.

    `.omni_deps` is appended rather than prepended: it holds only ultralytics
    and open_clip, neither of which exists in frodo_lan, so nothing in the
    environment gets shadowed either way, and appending keeps it that way if
    something is installed there later.
    """
    global _PATHS_READY
    if _PATHS_READY:
        return
    for path in (
        os.path.join(EDGE_VLM_ROOT, "experiments/omnivla"),
        EDGE_VLM_ROOT,
        OMNIVLA_TRAIN_ROOT,
    ):
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"{path} not found. arm-4' imports the edge_vlm and OmniVLA_edge "
                "repositories read-only; point EDGE_VLM_ROOT / OMNIVLA_TRAIN_ROOT "
                "at them."
            )
        if path not in sys.path:
            sys.path.insert(0, path)
    if os.path.isdir(OMNI_DEPS) and OMNI_DEPS not in sys.path:
        sys.path.append(OMNI_DEPS)
    _PATHS_READY = True


def parse_prompt(sentence: str) -> Tuple[Optional[str], Optional[str]]:
    """`"A next to B"` -> `(A, B)`; no relation word -> `(whole sentence, None)`.

    Only the leading article is stripped; the phrase is otherwise left alone,
    which is what the reference parser does.
    """
    text = (sentence or "").strip().lower()
    if not text:
        return None, None
    strip_article = lambda t: re.sub(r"^(the|a|an)\s+", "", t.strip())
    match = RELATION.search(text)
    if not match:
        return strip_article(text), None
    head, tail = RELATION.split(text, 1)
    return strip_article(head), strip_article(tail)


def _peak_box(peak_xy: Sequence[float], size: float = PEAK_BOX_SIZE) -> List[float]:
    """A fixed-size box centred on a heatmap peak, clipped to the frame."""
    half = size / 2.0
    x, y = float(peak_xy[0]), float(peak_xy[1])
    return [
        max(0.0, x - half),
        max(0.0, y - half),
        min(1.0, x + half),
        min(1.0, y + half),
    ]


def _center(box: Sequence[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


class OursPolicy:
    """Drop-in replacement for `OmniVLAEdgePolicy` running arm-4'.

    With `arm1_only=True` it loads and runs the unmodified arm-1 policy instead,
    on the same loop and the same frames, which is the A/B control for the field
    test. In that mode no detector, CLIP grounding or heatmap channel is used
    and the policy sees the whole sentence, exactly as the original does.
    """

    CONTEXT_LEN = 6

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        device: str = "cuda:0",
        *,
        arm1_only: bool = False,
        tick_log: Optional[str] = None,
    ) -> None:
        _bootstrap_paths()

        import torch
        import yaml
        from torchvision.transforms import Normalize

        self._torch = torch
        self.device = torch.device(device)
        self.arm1_only = bool(arm1_only)

        integration = yaml.safe_load(open(_CFG_INTEGRATION, encoding="utf-8"))
        e2e = yaml.safe_load(open(_CFG_E2E, encoding="utf-8"))
        eval_sets = yaml.safe_load(open(_CFG_EVAL_SETS, encoding="utf-8"))

        self.img_size = int(integration["image_size"])
        self.current_img_size = int(integration["current_img_size"])
        self.expect_dropped_keys = int(integration["expect_unexpected_keys"])
        self.lambda_d = float(e2e["lambda_d"])
        self.det_conf = float(e2e["detector"]["conf"])
        self.det_imgsz = int(e2e["detector"]["imgsz"])
        model_kwargs = {
            key: integration["model"][key]
            for key in (
                "context_size",
                "len_traj_pred",
                "learn_angle",
                "obs_encoder",
                "obs_encoding_size",
                "late_fusion",
                "mha_num_attention_heads",
                "mha_num_attention_layers",
                "mha_ff_dim_factor",
            )
        }
        if model_kwargs["context_size"] + 1 != self.CONTEXT_LEN:
            raise ValueError(
                f"context_size {model_kwargs['context_size']} does not match "
                f"CONTEXT_LEN {self.CONTEXT_LEN}"
            )

        with _edge_vlm_cwd():
            from build_eval_sets import lemma
            from experiments.context_score import box_mask, patch_coords
            from d150_e2e import head_of

        self._lemma = lemma
        self._box_mask = box_mask
        self._head_of = head_of
        self.patch_xy = patch_coords(PATCH_GRID)

        self.person_words = {lemma(w) for w in eval_sets["person_synonyms"]}
        coco_path = os.path.join(EDGE_VLM_ROOT, eval_sets["coco_instances"])
        self.coco80 = {
            lemma(c["name"])
            for c in json.load(open(coco_path, encoding="utf-8"))["categories"]
        }

        self._normalize = Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        self._black = torch.zeros(
            1, 3, self.img_size, self.img_size, device=self.device
        )

        import clip as openai_clip

        self._clip = openai_clip
        self.text_encoder, _ = openai_clip.load("ViT-B/32", device=self.device)
        self.text_encoder.to(torch.float32).eval()

        with _edge_vlm_cwd():
            from vint_train.models.il.il import IL_gps_map_mask3_lan2

        if self.arm1_only:
            self.model = IL_gps_map_mask3_lan2(**model_kwargs)
            arm1_ckpt = ckpt_path or integration["arm1_ckpt"]
            state = torch.load(arm1_ckpt, map_location="cpu", weights_only=False)
            missing, dropped = self.model.load_state_dict(state, strict=False)
            # The original checkpoint carries 12 dead `lgx.*` tensors that this
            # architecture has no home for. A different count means the
            # checkpoint is not the model we think it is.
            lgx = [k for k in dropped if "lgx" in k.lower()]
            if not (len(lgx) == len(dropped) == self.expect_dropped_keys):
                raise RuntimeError(
                    f"arm-1 checkpoint dropped {len(dropped)} keys "
                    f"({len(lgx)} lgx), expected {self.expect_dropped_keys} lgx: "
                    f"{sorted(dropped)[:5]}"
                )
            if missing:
                raise RuntimeError(f"arm-1 checkpoint is missing {len(missing)} keys")
            self.model.to(self.device).eval()
            self.detector = None
            self.heatmap = None
            self.new_channel_norm = None
            logger.info(
                "arm-1 control loaded from %s (%d lgx keys dropped, as expected)",
                arm1_ckpt,
                len(dropped),
            )
        else:
            with _edge_vlm_cwd():
                import arm_model
                from arm_model import ArmModel, _patch_first_conv

            self._arm_model_module = arm_model
            self.model = ArmModel(**model_kwargs)
            # Widen the FiLM conv to 4 channels *before* loading, because the
            # fine-tuned checkpoint was saved from an already-widened model.
            # Going through `ArmModel.load_state_dict` would re-patch it and
            # re-run the training-time freeze, so call the grandparent's loader.
            self.model._new_conv = _patch_first_conv(self.model, 1)
            arm4_ckpt = ckpt_path or DEFAULT_ARM4_CHECKPOINT
            state = torch.load(arm4_ckpt, map_location="cpu", weights_only=False)
            missing, dropped = IL_gps_map_mask3_lan2.load_state_dict(
                self.model, state, strict=False
            )
            if missing or dropped:
                raise RuntimeError(
                    f"arm-4' checkpoint did not load cleanly: {len(missing)} "
                    f"missing, {len(dropped)} dropped ({sorted(dropped)[:5]})"
                )
            self.model.to(self.device).eval()
            self.new_channel_norm = float(
                self.model._new_conv.weight[:, 3:].norm().item()
            )

            # ArmModel builds the 4th channel by calling this hook inside its
            # forward pass. We have already chosen the channel by then, so the
            # hook just hands the tensor back.
            class _Precomputed:
                def __init__(self) -> None:
                    self.channel = None

                def __call__(self, current_img, tokens):
                    return self.channel.to(current_img.device, current_img.dtype)

            self._precomputed = _Precomputed()
            ArmModel.HEATMAP = self._precomputed
            ArmModel.MASK_P = 0.0  # training-time RGB dropout; we mask explicitly

            with _edge_vlm_cwd():
                from heatmap import HeatmapProducer

            self.heatmap = (
                HeatmapProducer(
                    "adapter_head",
                    "ViT-B-16-quickgelu",
                    "openai",
                    adapter_path=ADAPTER_PATH,
                    head_path=LOC_HEAD_PATH,
                    device=str(self.device),
                )
                .to(self.device)
                .eval()
            )

            from ultralytics import YOLO

            self.detector = YOLO(DETECTOR_WEIGHTS)
            self.detector_names = (
                self.detector.model.names
                if hasattr(self.detector, "model")
                else self.detector.names
            )
            # ultralytics wants a device index, not a torch device string.
            self._det_device = (
                self.device.index if self.device.type == "cuda" else "cpu"
            )
            if self._det_device is None:
                self._det_device = 0
            logger.info(
                "arm-4' loaded from %s (new channel norm %.6f)",
                arm4_ckpt,
                self.new_channel_norm,
            )

        self.ticks: List[dict] = []
        self.tick_log = tick_log
        if self.tick_log:
            os.makedirs(os.path.join(self.tick_log, "thumbs"), exist_ok=True)
        self._warned_unused_goal = False

    # -- preprocessing ------------------------------------------------------

    def _pack(self, frames: Sequence["object"], via_224: bool = True):
        """Frames (oldest first) -> the tensors `IL_gps_map_mask3_lan2` expects.

        `via_224` reproduces the offline reference exactly: every frame is
        resized to 224 first and only then down to 96. That intermediate step
        changes the 96 px pixels slightly, and the numbers this port is checked
        against were produced with it, so it is the default.
        """
        import torch
        from torchvision.transforms import functional as TF

        size = self.img_size
        tensors = []
        for frame in frames:
            t = TF.to_tensor(frame.convert("RGB"))
            if via_224:
                t = TF.resize(t, (self.current_img_size, self.current_img_size))
            tensors.append(TF.resize(t, (size, size)))
        obs_raw = torch.cat(tensors, dim=0).unsqueeze(0).to(self.device)
        per_frame = torch.split(obs_raw, 3, dim=1)
        obs_img = torch.cat([self._normalize(x) for x in per_frame], dim=1)

        # Satellite imagery is not available on the rover, so both map slots are
        # black and the third is the raw (un-normalized) current frame. This is
        # what training fed the 9-channel map encoder; it looks inconsistent and
        # is deliberately left that way.
        black = self._normalize(self._black)
        map_images = torch.cat((black, black, per_frame[-1]), dim=1)

        current = TF.to_tensor(frames[-1].convert("RGB"))
        current = TF.resize(
            current, (self.current_img_size, self.current_img_size)
        )
        current_img = self._normalize(current.unsqueeze(0).to(self.device))
        return obs_img, map_images, current_img, black

    def _grid(self, current_img, text: str) -> np.ndarray:
        """CLIP heatmap for `text`, pooled onto the patch grid and flattened."""
        import torch

        tokens = self._clip.tokenize([text], truncate=True).to(self.device)
        with torch.no_grad():
            pooled = torch.nn.functional.adaptive_avg_pool2d(
                self.heatmap(current_img, tokens), PATCH_GRID
            )
        return pooled[0, 0].cpu().numpy().reshape(-1)

    def _channel(self, grid: np.ndarray, box: Optional[Sequence[float]]):
        """Keep the heatmap inside `box`, normalize, upsample to the FiLM size."""
        import torch

        kept = np.zeros(PATCH_GRID * PATCH_GRID, np.float32)
        if box is None:
            kept = np.clip(grid, 0, None).copy()
        else:
            mask = self._box_mask(box, self.patch_xy)
            kept[mask] = np.clip(grid[mask], 0, None)
        kept = kept.reshape(PATCH_GRID, PATCH_GRID)
        span = max(float(kept.max() - kept.min()), 1e-8)
        kept = (kept - kept.min()) / span
        return torch.nn.functional.interpolate(
            torch.from_numpy(kept)[None, None],
            size=(self.current_img_size, self.current_img_size),
            mode="bilinear",
            align_corners=False,
        )

    def _detect(self, frame) -> List[dict]:
        """YOLOv8n boxes in normalized xyxy, highest confidence first."""
        image = np.array(frame.convert("RGB"))[:, :, ::-1]  # PIL RGB -> BGR
        result = self.detector.predict(
            image,
            imgsz=self.det_imgsz,
            conf=self.det_conf,
            device=self._det_device,
            verbose=False,
        )[0]
        height, width = image.shape[:2]
        boxes = [
            {
                "cat": str(self.detector_names[int(cls)]),
                "conf": float(conf),
                "box": [
                    float(xyxy[0] / width),
                    float(xyxy[1] / height),
                    float(xyxy[2] / width),
                    float(xyxy[3] / height),
                ],
            }
            for xyxy, cls, conf in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
            )
        ]
        boxes.sort(key=lambda d: -d["conf"])
        return boxes

    def _coco_class_for(self, phrase: str) -> Tuple[str, Optional[str]]:
        """Head noun of `phrase` and the COCO class to filter detections by."""
        head = self._head_of(phrase or "")
        if self._lemma(head) in self.person_words:
            return head, "person"
        if self._lemma(head) in self.coco80:
            return head, head
        return head, None

    # -- the interface `AutonomyLoop` calls ---------------------------------

    def predict_waypoints(
        self,
        context_frames: Sequence["object"],
        prompt: Optional[str] = None,
        goal_pose: Optional[Sequence[float]] = None,
        goal_image: Optional["object"] = None,
    ):
        import torch

        started = time.time()
        timing: dict = {}
        frames = list(context_frames)
        if len(frames) != self.CONTEXT_LEN:
            raise ValueError(
                f"expected {self.CONTEXT_LEN} context frames, got {len(frames)}"
            )
        if (goal_pose is not None or goal_image is not None) and not self._warned_unused_goal:
            self._warned_unused_goal = True
            logger.warning(
                "arm-4' is language-only: the goal pose and goal image are "
                "ignored. Drive it with an instruction, not a goal."
            )

        sentence = (prompt or "").strip()
        obs_img, map_images, current_img, black = self._pack(frames)
        goal_pose_t = torch.zeros(1, 4, device=self.device)
        goal_mask = torch.full(
            (1,), MODALITY_LANGUAGE_ONLY, dtype=torch.long, device=self.device
        )
        record = {
            "prompt": sentence,
            "mode": "arm1" if self.arm1_only else "arm4p",
            "A": None,
            "B": None,
            "n_dets": 0,
            "n_candidates": 0,
            "relaxed": False,
            "synthetic_candidate": False,
            "selected": None,
            "scores": [],
            "B_peak": None,
        }
        timing["pack"] = time.time() - started

        if self.arm1_only:
            step = time.time()
            with torch.no_grad():
                features = self.text_encoder.encode_text(
                    self._clip.tokenize([sentence or "xxxx"], truncate=True).to(
                        self.device
                    )
                ).float()
                action, _, _ = self.model(
                    obs_img,
                    goal_pose_t,
                    map_images,
                    black,
                    goal_mask,
                    features,
                    current_img,
                )
            timing["policy"] = time.time() - step
            return self._finish(action, record, timing, channel=None)

        target, anchor = parse_prompt(sentence)
        record["A"], record["B"] = target, anchor
        if target is None:
            # Nothing to ground. Let the whole frame through as the channel so
            # the policy still gets a well-formed input.
            record["fallback"] = "empty_prompt"
            grid = self._grid(current_img, "xxxx")
            channel = self._channel(grid, None)
            return self._run_arm4(
                obs_img, goal_pose_t, map_images, black, goal_mask,
                "xxxx", channel, current_img, record, timing,
            )

        # (2) detection, filtered to the target's COCO class when it maps.
        step = time.time()
        detections = self._detect(frames[-1])
        timing["detector"] = time.time() - step
        record["n_dets"] = len(detections)
        head, want = self._coco_class_for(target)
        record["A_head"] = head
        if want:
            candidates = [
                d
                for d in detections
                if self._lemma(d["cat"]) == self._lemma(want)
            ]
        else:
            candidates = list(detections)
        if want and not candidates and detections:
            # The class mapped but nothing of that class was detected. Relaxing
            # to every box beats giving up, and the reference does the same.
            candidates = list(detections)
            record["relaxed"] = True

        # (3) attribute score per candidate, from the target phrase.
        step = time.time()
        grid_target = self._grid(current_img, target)
        grid_anchor = self._grid(current_img, anchor) if anchor else None
        timing["clip"] = time.time() - step

        peak = None
        if grid_anchor is not None:
            peak = tuple(
                float(v) for v in self.patch_xy[int(np.argmax(grid_anchor))]
            )
            record["B_peak"] = list(peak)

        if not candidates:
            # Divergence from the reference, which would run arm-1 here: place
            # one synthetic box on the target heatmap's own peak so the field
            # test keeps running the model under test.
            target_peak = tuple(
                float(v) for v in self.patch_xy[int(np.argmax(grid_target))]
            )
            candidates = [
                {
                    "cat": "<heatmap peak>",
                    "conf": 0.0,
                    "box": _peak_box(target_peak),
                }
            ]
            record["synthetic_candidate"] = True
            record["fallback"] = "zero_candidates_peak_box"
        record["n_candidates"] = len(candidates)

        # (5) selection. The distance term is dropped when there is no anchor.
        step = time.time()
        scores = []
        for i, candidate in enumerate(candidates):
            mask = self._box_mask(candidate["box"], self.patch_xy)
            score = float(grid_target[mask].max()) if mask.any() else -1e9
            if peak is None:
                distance = 0.0
            else:
                cx, cy = _center(candidate["box"])
                distance = (
                    float(np.hypot(cx - peak[0], cy - peak[1])) / _DIAG
                )
            scores.append(
                {
                    "i": i,
                    "cat": candidate["cat"],
                    "score": score,
                    "dist": distance,
                    "final": score - self.lambda_d * distance,
                }
            )
        selected = int(np.argmax([s["final"] for s in scores]))
        timing["rule"] = time.time() - step
        record["scores"] = scores
        record["selected"] = selected
        record["selected_box"] = candidates[selected]["box"]

        # (6) keep only the selected box's heatmap -> the 4th channel.
        channel = self._channel(grid_target, candidates[selected]["box"])
        return self._run_arm4(
            obs_img, goal_pose_t, map_images, black, goal_mask,
            target, channel, current_img, record, timing,
        )

    def _run_arm4(
        self, obs_img, goal_pose_t, map_images, black, goal_mask,
        text, channel, current_img, record, timing,
    ):
        """(7) zero the RGB, hand over the channel, run the policy."""
        import torch

        step = time.time()
        tokens = self._clip.tokenize([text], truncate=True).to(self.device)
        # ArmModel's forward pass reads the tokens from this module-level stash.
        self._arm_model_module._STASH["tokens"] = tokens
        self._precomputed.channel = channel
        with torch.no_grad():
            features = self.text_encoder.encode_text(tokens).float()
            # Zero in normalized space is the dataset mean, not black. Without
            # this the policy reads the RGB and ignores the heatmap channel.
            masked_current = torch.zeros_like(current_img)
            action, _, _ = self.model(
                obs_img,
                goal_pose_t,
                map_images,
                black,
                goal_mask,
                features,
                masked_current,
            )
        timing["policy"] = time.time() - step
        return self._finish(action, record, timing, channel=channel)

    def _finish(self, action, record, timing, channel):
        waypoints = action[0].float().cpu().numpy()
        record["waypoints"] = waypoints.tolist()
        record["timing_ms"] = {k: round(v * 1000, 3) for k, v in timing.items()}
        record["timing_ms"]["total"] = round(sum(timing.values()) * 1000, 3)
        if self.tick_log:
            index = len(self.ticks)
            if channel is not None:
                from PIL import Image

                thumb = (channel[0, 0].numpy() * 255).astype(np.uint8)
                Image.fromarray(thumb).resize((64, 64)).save(
                    os.path.join(self.tick_log, "thumbs", f"{index:06d}.png")
                )
            with open(
                os.path.join(self.tick_log, "ticks.jsonl"), "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.ticks.append(record)
        return waypoints, MODALITY_LANGUAGE_ONLY
