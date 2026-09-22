"""arm-4' policy: OmniVLA-edge steered by a heatmap channel.

Interface-compatible with `OmniVLAEdgePolicy`, so `AutonomyLoop` cannot tell the
difference: `predict_waypoints(frames, prompt=, goal_pose=, goal_image=)` still
returns `((8, 4) ndarray, modality_id)`.

Three things differ from the shipped upstream policy:
  (a) the checkpoint is the fine-tuned arm-4' (FiLM first conv is 3 -> 4 ch);
  (b) `current_img` is rebuilt every tick -- RGB zeroed, heatmap as 4th channel;
  (c) the prompt is parsed and grounded before the policy ever sees it.

The policy forward pass and the waypoint -> command conversion are untouched.

Ported from the reference implementations in the edge_vlm and OmniVLA_edge
repositories. Those are not modified. The parts this needs are vendored byte for
byte under `policy/refs/`, which is committed, so a clone plus the weights in
`models/` is a complete deployment:

    refs/experiments/omnivla/arm_model.py       ArmModel, the 4-channel wrapper
    refs/experiments/omnivla/heatmap.py         HeatmapProducer
    refs/experiments/omnivla/d150_e2e.py        the numbers check_ours matches
    refs/vint_train/models/il/il.py             IL_gps_map_mask3_lan2
    refs/configs/experiment/*.yaml              every constant used here

The vendored copies are never edited -- an edit is how they would start drifting
from the originals. `refs/` mirrors the original directory layout for the same
reason: several of those modules read data files by relative path at import
time, and the layout is what keeps that working.

Runtime: an environment with torch, plus `models/omni_deps` for ultralytics and
open_clip. This module puts both on sys.path itself, so PYTHONPATH is optional.

Three divergences from `run_autonomy_ours.py`, all requested:

  * no candidate survives detection -> upstream falls back to the arm-1 model.
    Here a single synthetic candidate box is placed on the A heatmap peak
    instead, so the field test never silently swaps models underneath itself.
  * the prompt has no relation word -> upstream skips detection and feeds a
    whole-frame heatmap. Here A becomes the whole sentence, B is None, and
    detection plus selection still run with the distance term dropped.
  * B is placed on a detector box scored by crop cosine, not on the B heatmap's
    peak (D164). The reference measures the distance term to that peak and uses
    a box only to decide what to exclude; both come off the same box here. The
    peak grounds the noun and cannot read the adjective, so a frame of four
    chairs puts it on whichever chair and the distance term then measures from
    the wrong one. `anchor_mode="heatmap_peak"` restores the reference's
    behaviour, and is still the automatic fallback for a B the detector cannot
    box at all.

D157 (2), the exclusion of B's own box from the A candidates, is in the
reference and was missing here until D164. Without it B competes as an A at
distance zero from itself, which is the largest bonus the rule can give.
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
# Everything resolves from inside this repository, so a clone plus the weights
# is the whole deployment.
#
#   policy/refs/  the reference code and config, committed. It mirrors the
#                 edge_vlm layout on purpose: some of those modules read data
#                 files by relative path at import time, so keeping the layout
#                 means the vendored copies need no edits, and edits are what
#                 would let them drift from the originals.
#   models/       the weights, which are too large for git. Empty in a fresh
#                 clone; drop the files in and nothing else needs configuring.
#
# Both are overridable, for a machine that keeps them somewhere else.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
REFS_ROOT = os.environ.get("OMNIVLA_REFS_ROOT", os.path.join(_HERE, "refs"))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(REPO_ROOT, "models"))

# ultralytics and open_clip. They are installed with --no-deps so that they
# cannot pull their own torch over the one the policy needs, which is why they
# are carried as a directory rather than listed as requirements.
OMNI_DEPS = os.environ.get("OMNI_DEPS", os.path.join(MODELS_DIR, "omni_deps"))

# Optional offline caches. If these exist the two CLIP backbones are read from
# here instead of being downloaded, which is what a field laptop needs.
CLIP_CACHE = os.path.join(MODELS_DIR, "clip")
HF_CACHE = os.path.join(MODELS_DIR, "hf")

# Weights, all expected directly in models/ under these names.
DEFAULT_ARM4_CHECKPOINT = os.path.join(MODELS_DIR, "arm4p_s0_latest.pth")
DEFAULT_ARM1_CHECKPOINT = os.path.join(MODELS_DIR, "arm1_latest.pth")
ADAPTER_PATH = os.path.join(MODELS_DIR, "act4_cl6159_s2.pt")
LOC_HEAD_PATH = os.path.join(MODELS_DIR, "full_H1_linear_lr0.001_s0.pt")
DETECTOR_WEIGHTS = os.path.join(MODELS_DIR, "yolov8n.pt")

# What each file is, for the error message when one is missing. arm-1 needs
# only its own checkpoint: it runs no detector and no CLIP grounding.
ARM4_WEIGHTS = (
    (DEFAULT_ARM4_CHECKPOINT, "arm-4' policy", "415M"),
    (ADAPTER_PATH, "CLIP adapter", "55M"),
    (LOC_HEAD_PATH, "localization head", "1.1M"),
    (DETECTOR_WEIGHTS, "YOLOv8n detector", "6.3M"),
)
ARM1_WEIGHTS = ((DEFAULT_ARM1_CHECKPOINT, "arm-1 control policy", "418M"),)

# Config files the reference reads its constants from. This module introduces no
# new tunables of its own.
_CFG_INTEGRATION = os.path.join(
    REFS_ROOT, "configs/experiment/omnivla_integration.yaml"
)
_CFG_E2E = os.path.join(REFS_ROOT, "configs/experiment/d150_e2e.yaml")
_CFG_EVAL_SETS = os.path.join(
    REFS_ROOT, "configs/experiment/omnivla_eval_sets.yaml"
)

# Per-tick log. Defaults inside the repository so a field run writes somewhere
# obvious; gitignored alongside the weights.
DEFAULT_TICK_LOG = os.path.join(REPO_ROOT, "field_log")

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


def required_weights(arm1_only: bool = False):
    return ARM1_WEIGHTS if arm1_only else ARM4_WEIGHTS


def missing_weights(arm1_only: bool = False) -> List[str]:
    """Which of the required weight files are not in `models/` yet."""
    return [
        path
        for path, _, _ in required_weights(arm1_only)
        if not os.path.exists(path)
    ]


def weights_missing_message(arm1_only: bool = False) -> str:
    which = "arm-1" if arm1_only else "arm-4'"
    lines = [
        f"{which} cannot start: weights are missing from {MODELS_DIR}",
        "",
        "They are too large for git, so a fresh clone has an empty models/.",
        "Copy these into it:",
        "",
    ]
    for path, what, size in required_weights(arm1_only):
        mark = "missing" if not os.path.exists(path) else "ok"
        lines.append(
            f"  [{mark:>7s}] {os.path.basename(path):32s} {size:>6s}  {what}"
        )
    lines += ["", "See the README section on putting the models in place."]
    return "\n".join(lines)


@contextlib.contextmanager
def _refs_cwd():
    """Import the reference modules from `policy/refs`, then change back.

    Several of them read data files by relative path at import time, so they
    only import cleanly with that directory as the working directory. The
    deployment process resolves its own paths relative to the repository root,
    so the change is undone straight away rather than left in place.
    """
    previous = os.getcwd()
    os.chdir(REFS_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def _bootstrap_paths(need_omni_deps: bool = True) -> None:
    """Put the vendored reference code and the extra deps on `sys.path`.

    `omni_deps` is appended rather than prepended: it holds only ultralytics
    and open_clip, neither of which exists in the environment, so nothing gets
    shadowed either way, and appending keeps it that way if something is
    installed there later.

    The offline caches are wired up here too, because HF_HOME has to be set
    before huggingface_hub is first imported.
    """
    global _PATHS_READY
    if _PATHS_READY:
        return
    if not os.path.isdir(REFS_ROOT):
        raise FileNotFoundError(
            f"{REFS_ROOT} not found. It is committed to this repository; a "
            "clone should have it. Set OMNIVLA_REFS_ROOT to point elsewhere."
        )
    for path in (os.path.join(REFS_ROOT, "experiments/omnivla"), REFS_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    if os.path.isdir(OMNI_DEPS):
        if OMNI_DEPS not in sys.path:
            sys.path.append(OMNI_DEPS)
    elif need_omni_deps:
        # arm-1 does not need these, which is why this is conditional.
        raise FileNotFoundError(
            f"{OMNI_DEPS} not found. It holds ultralytics and open_clip, which "
            "are installed with --no-deps so they cannot replace torch. Copy "
            "the directory into models/, or set OMNI_DEPS."
        )
    if os.path.isdir(HF_CACHE):
        # Copying the cache in is a statement that this machine should not be
        # reaching out, so stop it reaching out: otherwise it still contacts the
        # hub to revalidate, which stalls on a field network.
        os.environ.setdefault("HF_HOME", HF_CACHE)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        logger.info("using the offline CLIP cache in %s", HF_CACHE)
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
        _bootstrap_paths(need_omni_deps=not arm1_only)

        import torch
        import yaml
        from torchvision.transforms import Normalize

        self._torch = torch
        self.device = torch.device(device)
        self.arm1_only = bool(arm1_only)

        # Fail here, with the list, rather than deep inside a loader. An empty
        # models/ is the normal state of a fresh clone. An explicit --ckpt
        # replaces the policy checkpoint only; the rest still has to be present.
        policy_ckpt = (
            DEFAULT_ARM1_CHECKPOINT if self.arm1_only else DEFAULT_ARM4_CHECKPOINT
        )
        outstanding = [
            path
            for path in missing_weights(self.arm1_only)
            if not (ckpt_path and path == policy_ckpt)
        ]
        if outstanding:
            raise FileNotFoundError(weights_missing_message(self.arm1_only))

        integration = yaml.safe_load(open(_CFG_INTEGRATION, encoding="utf-8"))
        e2e = yaml.safe_load(open(_CFG_E2E, encoding="utf-8"))
        eval_sets = yaml.safe_load(open(_CFG_EVAL_SETS, encoding="utf-8"))

        self.img_size = int(integration["image_size"])
        self.current_img_size = int(integration["current_img_size"])
        self.expect_dropped_keys = int(integration["expect_unexpected_keys"])
        self.lambda_d = float(e2e["lambda_d"])
        self.det_conf = float(e2e["detector"]["conf"])
        self.det_imgsz = int(e2e["detector"]["imgsz"])
        # How a candidate is scored against the target phrase.
        #   crop_cos  crop the box, embed it with the adapter, cosine against
        #             the phrase. This is the path the adapter was judged on,
        #             and the only one that reads adjectives.
        #   heat_max  the localization head's heatmap, maxed inside the box.
        #             It grounds the noun and ignores the adjective, so two
        #             chairs of different colours score the same.
        self.score_mode = e2e.get("score_mode", "crop_cos")
        self.crop_margin = float(e2e.get("crop_margin", 0.10))
        self.score_template = e2e.get("score_template", "a photo of a {}.")
        # How the anchor -- the "B" of "A next to B" -- is placed.
        #   crop_cos      score the detector's boxes for the anchor phrase the
        #                 same way the target's are scored, and take the best
        #                 box's centre. Reads the adjective, so it can tell the
        #                 black chair from the three orange ones.
        #   heatmap_peak  the brightest patch of the anchor's heatmap. That
        #                 runs through the localization head, which grounds the
        #                 noun only, so with four chairs in frame it lands on
        #                 whichever chair. Kept as the fallback for an anchor
        #                 the detector cannot box, and as the way back to the
        #                 pre-D164 behaviour for an A/B run.
        #
        # Selecting the anchor's box is also what makes it excludable from the
        # target's candidates, so this one switch moves both.
        self.anchor_mode = e2e.get("anchor_mode", "crop_cos")
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

        with _refs_cwd():
            from build_eval_sets import lemma
            from experiments.context_score import box_mask, patch_coords
            from d150_e2e import head_of, iou, match_coco_phrase

        self._lemma = lemma
        self._box_mask = box_mask
        self._iou = iou
        self._head_of = head_of
        self._match_coco_phrase = match_coco_phrase
        self.patch_xy = patch_coords(PATCH_GRID)

        self.person_words = {lemma(w) for w in eval_sets["person_synonyms"]}
        coco_path = os.path.join(REFS_ROOT, eval_sets["coco_instances"])
        # The names as written, for multi-word matching ("potted plant"), and
        # lemmatized for the head-noun path.
        self.coco_names = [
            c["name"]
            for c in json.load(open(coco_path, encoding="utf-8"))["categories"]
        ]
        self.coco80 = {lemma(n) for n in self.coco_names}

        self._normalize = Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        self._black = torch.zeros(
            1, 3, self.img_size, self.img_size, device=self.device
        )

        import clip as openai_clip

        self._clip = openai_clip
        # models/clip/ if it was copied over, otherwise ~/.cache/clip and a
        # download on first run.
        clip_kwargs = (
            {"download_root": CLIP_CACHE} if os.path.isdir(CLIP_CACHE) else {}
        )
        self.text_encoder, _ = openai_clip.load(
            "ViT-B/32", device=self.device, **clip_kwargs
        )
        self.text_encoder.to(torch.float32).eval()

        with _refs_cwd():
            from vint_train.models.il.il import IL_gps_map_mask3_lan2

        if self.arm1_only:
            self.model = IL_gps_map_mask3_lan2(**model_kwargs)
            arm1_ckpt = ckpt_path or DEFAULT_ARM1_CHECKPOINT
            if not os.path.exists(arm1_ckpt):
                raise FileNotFoundError(
                    f"the arm-1 control checkpoint is not in models/: {arm1_ckpt}\n"
                    "It is optional -- only --arm1 needs it. Copy it as "
                    f"{os.path.basename(DEFAULT_ARM1_CHECKPOINT)} (418M), or pass "
                    "--ckpt with its path."
                )
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
            with _refs_cwd():
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

            with _refs_cwd():
                from heatmap import (
                    HeatmapProducer,
                    _clip_patch_forward,
                    l2_normalize,
                )

            # `crop_cos`'s own building blocks, so `_crop_embeddings` below is
            # that method's arithmetic rather than a second copy of it.
            self._clip_patch_forward = _clip_patch_forward
            self._l2_normalize = l2_normalize

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
        # Intermediates from the most recent tick, for visualize_pipeline.py.
        self.last_debug: Optional[dict] = None
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

    def _crop_embeddings(self, frame, boxes):
        """The adapter's pooled embedding for each box, in one forward pass.

        This is the first half of `HeatmapProducer.crop_cos`. That method does
        the crop, the forward and the text cosine in one call and returns only
        the scalars, so scoring a second phrase against the same crops through
        it would embed every crop a second time. Keeping the vectors lets the
        target and the anchor be scored on one pass.

        The steps and their order are the reference's own -- its `preprocess`,
        its `_clip_patch_forward`, its `l2_normalize` -- so this stays the path
        the adapter's accuracy was measured on. `check_ours` asserts the
        cosines taken from these vectors match `crop_cos`'s own numbers.
        """
        import torch

        width, height = frame.size
        pixels = []
        for box in boxes:
            x0, y0, x1, y1 = box
            mx, my = (x1 - x0) * self.crop_margin, (y1 - y0) * self.crop_margin
            crop = frame.crop(
                (
                    int(max(0.0, x0 - mx) * width),
                    int(max(0.0, y0 - my) * height),
                    int(min(1.0, x1 + mx) * width),
                    int(min(1.0, y1 + my) * height),
                )
            )
            if min(crop.size) < 8:
                crop = frame
            pixels.append(self.heatmap.preprocess(crop))
        if not pixels:
            return None
        with torch.no_grad():
            patches, _ = self._clip_patch_forward(
                self.heatmap.clip.visual,
                torch.stack(pixels).to(self.device),
                maskclip=False,
            )
            return self._l2_normalize(self._l2_normalize(patches).mean(1))

    def _phrase_vector(self, phrase: str):
        """`phrase` under the score template, as a unit text embedding."""
        import torch

        text = self.score_template.format(phrase)
        with torch.no_grad():
            tokens = self.heatmap.tokenizer([text]).to(self.device)
            vector = self._l2_normalize(
                self.heatmap.clip.encode_text(tokens).float()
            )[0]
        return vector, text

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
        """Head noun of `phrase` and the COCO class to filter detections by.

        A COCO name found anywhere in the phrase is only accepted when it
        contains the head noun. "the orange chair" has to map to `chair`, not
        to `orange` the fruit: an adjective must not take the phrase over.
        """
        head = self._head_of(phrase or "")
        multiword = self._match_coco_phrase(phrase or "", self.coco_names, head)
        if multiword:
            return head, multiword
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
            "fallback": None,
            "B_peak": None,
            "B_box": None,
            "B_mode": None,
            "b_excluded": 0,
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

        # (3) the target heatmap, which builds the 4th channel either way.
        step = time.time()
        grid_target = self._grid(current_img, target)
        timing["clip"] = time.time() - step

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

        # (4a) crop every box once. A crop is cut out of the full-resolution
        # frame and embedded with the adapter, which is the path the adapter's
        # accuracy was measured on and the only one that reads the adjective.
        # The heatmap's in-box max is computed either way and logged next to
        # it, because it is what earlier runs selected on.
        #
        # The anchor's detections are gathered first so its boxes ride along in
        # the same pass. They are filtered by the anchor's own COCO class, not
        # the target's: "the orange chair next to the table" grounds the anchor
        # on tables. When the two share a class -- which is when the relation
        # is doing the work -- the lists coincide and the anchor is scored
        # against the target's own crop vectors, at no extra forward pass.
        step = time.time()
        anchor_pool = []
        if anchor and not record["synthetic_candidate"]:
            anchor_head, anchor_want = self._coco_class_for(anchor)
            record["B_head"] = anchor_head
            if anchor_want:
                anchor_pool = [
                    d
                    for d in detections
                    if self._lemma(d["cat"]) == self._lemma(anchor_want)
                ]

        box_list = [c["box"] for c in candidates]
        anchor_rows = []
        for detection in anchor_pool:
            for i, candidate in enumerate(candidates):
                if candidate is detection:
                    anchor_rows.append(i)
                    break
            else:
                anchor_rows.append(len(box_list))
                box_list.append(detection["box"])

        vectors = None
        crop_scores = None
        crop_text = ""
        if self.score_mode == "crop_cos":
            vectors = self._crop_embeddings(frames[-1].convert("RGB"), box_list)
        if vectors is not None:
            target_vector, crop_text = self._phrase_vector(target)
            crop_scores = [
                float(v) for v in (vectors[: len(candidates)] @ target_vector)
            ]
        timing["score"] = time.time() - step
        record["score_mode"] = self.score_mode
        record["crop_text"] = crop_text

        # (4b) where the anchor is. The same crop cosine as the target, so the
        # adjective is read. The heatmap path this replaces runs through the
        # localization head, which grounds the noun and nothing else: with four
        # chairs in frame its peak lands on whichever chair reads as most
        # chair-like, not on the black one. The heatmap stays as the fallback
        # for an anchor the detector cannot box at all -- a tree, a doorway,
        # anything off COCO -- where a box-based score has nothing to score.
        step = time.time()
        peak = None
        anchor_box = None
        grid_anchor = None
        if anchor:
            if (
                self.anchor_mode == "crop_cos"
                and vectors is not None
                and anchor_rows
            ):
                anchor_vector, anchor_text = self._phrase_vector(anchor)
                anchor_scores = [
                    float(vectors[row] @ anchor_vector) for row in anchor_rows
                ]
                best = int(np.argmax(anchor_scores))
                anchor_box = list(anchor_pool[best]["box"])
                peak = _center(anchor_box)
                record["B_mode"] = "crop_cos"
                record["B_text"] = anchor_text
                record["B_scores"] = anchor_scores
                record["B_cat"] = anchor_pool[best]["cat"]
            else:
                grid_anchor = self._grid(current_img, anchor)
                peak = tuple(
                    float(v) for v in self.patch_xy[int(np.argmax(grid_anchor))]
                )
                record["B_mode"] = "heatmap_peak"
                if record["fallback"] is None and self.anchor_mode == "crop_cos":
                    # Asked for the box path and did not get it, which is the
                    # anchor being off COCO or undetected. Worth a mark in the
                    # log; falling back is not, when it was configured.
                    record["fallback"] = "anchor_not_boxed"
            record["B_peak"] = list(peak)
            record["B_box"] = anchor_box
        timing["anchor"] = time.time() - step

        # (4c) take the anchor's own box out of the target's candidates.
        # Without this the anchor competes as a target while sitting at
        # distance zero from itself, so the distance term hands it the largest
        # bonus on offer and a wrong-coloured box can win on that alone.
        # Overlap rather than identity, because the two lists are filtered
        # separately. Ported from the reference's D157 (2).
        candidates_all = candidates
        excluded = []
        if anchor_box is not None:
            keep = [
                i
                for i, candidate in enumerate(candidates)
                if self._iou(candidate["box"], anchor_box) <= 0.5
            ]
            excluded = [i for i in range(len(candidates)) if i not in keep]
            if keep and excluded:
                candidates = [candidates[i] for i in keep]
                if crop_scores is not None:
                    crop_scores = [crop_scores[i] for i in keep]
            elif excluded:
                # Every candidate overlapped the anchor, which is what "the
                # chair next to the chair" looks like on a single box. The
                # reference calls that a failed frame; the field loop has to
                # return a trajectory, so the exclusion is dropped here and the
                # tick is marked instead.
                excluded = []
                record["fallback"] = record["fallback"] or "b_exclusion_empty"
        record["b_excluded"] = len(excluded)
        record["n_candidates"] = len(candidates)

        # (5) selection. The distance term is dropped when there is no anchor.
        step = time.time()
        scores = []
        for i, candidate in enumerate(candidates):
            mask = self._box_mask(candidate["box"], self.patch_xy)
            heat_max = float(grid_target[mask].max()) if mask.any() else -1e9
            score = heat_max if crop_scores is None else float(crop_scores[i])
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
                    "heat_max": heat_max,
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

        # Everything a picture of this tick would need. Kept for the last tick
        # only, and not written anywhere: the jsonl log holds what is worth
        # keeping, and the grids are two 14x14 arrays.
        self.last_debug = {
            "detections": detections,
            "candidates": candidates,
            # Before the anchor's box was taken out, so a picture can show what
            # was dropped rather than just not drawing it.
            "candidates_all": candidates_all,
            "excluded": excluded,
            "anchor_box": anchor_box,
            "anchor_pool": anchor_pool,
            "grid_target": grid_target,
            "grid_anchor": grid_anchor,
            "peak": peak,
            "channel": channel,
            "selected": selected,
            # The policy's own inputs, so a tool can re-run the forward pass
            # with a different channel and compare the trajectories. Everything
            # here is small: 96x96 obs and one 224x224 frame.
            "policy_inputs": (
                obs_img, goal_pose_t, map_images, black, goal_mask,
                target, current_img,
            ),
        }
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
