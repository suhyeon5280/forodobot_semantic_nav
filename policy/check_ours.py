"""Pre-flight checks for the arm-4' deployment path. Run before the rover.

    conda activate rover
    python -m policy.check_ours

Three checks, all offline and inference-only:

  1. regression -- the upstream OmniVLA-edge path still loads and predicts.
                   Skipped, not failed, when best.pth is not present.
  2. port       -- the two word-order-swapped prompts on the reference frame
                   reproduce the offline reference's endpoint within a tolerance.
                   This is the one that catches a preprocessing mistake.
  3. timing     -- median tick time against the 333 ms control period.

Nothing here drives the rover, and nothing is written outside this repository.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

from .control import (
    DT,
    METRIC_WAYPOINT_SPACING,
    WAYPOINT_INDEX,
    to_control_command,
    waypoint_to_velocity,
)
from .ours_policy import MODELS_DIR, REFS_ROOT, OursPolicy

# The offline reference reports endpoint lateral position at this scale, which
# is not the scale the rover is driven at. Both are printed; see the mismatch
# note in the reference's field-test prep document.
REF_WAYPOINT_SPACING = 0.125
REF_WAYPOINT_INDEX = 7

# The validation frames come from the training dataset, which is far too large
# to carry. `models/frames/` holds the handful this needs; the environment
# variable points at the full dataset on a machine that has it.
DATASET_ROOT = os.environ.get(
    "OMNIVLA_DATASET_ROOT", os.path.join(MODELS_DIR, "frames")
)
# The recorded reference output is small and committed with the rest of refs/.
REF_RESULT = os.path.join(
    REFS_ROOT, "results/phase4/omnivla/d153_e2e_mask.json"
)
REF_FRAME = "episode_0020/00000000"
REF_PROMPTS = (
    "the white van next to the red truck",
    "the red truck next to the white van",
)
TOLERANCE_M = 0.01

# `_crop_embeddings` is `crop_cos` split in two, so the anchor can be scored
# against the crops the target already paid for. The two must keep agreeing to
# floating-point noise; anything larger means the split has drifted from the
# reference and the adapter is no longer being read the way it was measured.
CROP_DRIFT_TOL = 1e-5
LOOP_BUDGET_MS = 1000.0 * DT


def load_context(frame_key, context_len=6):
    """The reference frame plus its history, oldest first, clamped at the start."""
    episode, stem = frame_key.split("/")
    image_dir = os.path.join(DATASET_ROOT, episode, "image")
    stems = sorted(f[:-4] for f in os.listdir(image_dir) if f.endswith(".jpg"))
    k = stems.index(stem)
    picked = [stems[max(0, k - h)] for h in range(context_len - 1, -1, -1)]
    return [
        Image.open(os.path.join(image_dir, s + ".jpg")).convert("RGB")
        for s in picked
    ]


def reference_endpoints(arm="arm4p"):
    """Recorded endpoint lateral positions for one arm, keyed by sentence."""
    with open(REF_RESULT, encoding="utf-8") as handle:
        blob = json.load(handle)
    steps = blob.get("result", blob)["steps"]
    return {
        s["sentence"]: s["endpoint_lateral_m"][arm]
        for s in steps
        if s["frame"] == REF_FRAME and s.get("endpoint_lateral_m")
    }


def endpoint_lateral_m(waypoints):
    return float(waypoints[REF_WAYPOINT_INDEX, 1] * REF_WAYPOINT_SPACING)


def deployment_command(waypoints):
    """What the rover would actually be told, at the deployment scale."""
    v, w = waypoint_to_velocity(waypoints)
    linear, angular = to_control_command(v, w)
    return v, w, linear, angular


def check_regression(device):
    """The shipped path is untouched, so this only has to still work."""
    from .omnivla_policy import CONTEXT_LEN, DEFAULT_CHECKPOINT, OmniVLAEdgePolicy

    print("\n=== 1. regression: upstream OmniVLA-edge path ===")
    if not os.path.exists(DEFAULT_CHECKPOINT):
        print(
            f"  SKIP: {DEFAULT_CHECKPOINT} is not here. It is only needed for "
            "--upstream, and it is a release asset rather than part of the "
            "repository, so this is the normal state of a fresh clone."
        )
        return None
    policy = OmniVLAEdgePolicy(DEFAULT_CHECKPOINT, device=device)
    frames = load_context(REF_FRAME, CONTEXT_LEN)
    waypoints, modality = policy.predict_waypoints(
        frames, prompt="the blue trash bin"
    )
    assert waypoints.shape == (8, 4), waypoints.shape
    v, w, linear, angular = deployment_command(waypoints)
    print(f"  modality_id={modality}  waypoints={waypoints.shape}")
    print(
        "  first 3 (x_m, y_m): "
        + ", ".join(
            f"({x * METRIC_WAYPOINT_SPACING:+.3f}, {y * METRIC_WAYPOINT_SPACING:+.3f})"
            for x, y, _, _ in waypoints[:3]
        )
    )
    print(f"  v={v:.3f} m/s  w={w:.3f} rad/s  linear={linear:+.3f} angular={angular:+.3f}")
    print("  PASS: upstream path loads and predicts")
    del policy
    return True


def check_port(device, tick_log=None):
    print("\n=== 2. port: reference frame, word-order swap ===")
    expected = reference_endpoints()
    if not expected:
        print(f"  SKIP: no recorded reference in {REF_RESULT}")
        return False
    policy = OursPolicy(device=device, tick_log=tick_log)
    frames = load_context(REF_FRAME, policy.CONTEXT_LEN)
    print(f"  frame {REF_FRAME}  new-channel norm {policy.new_channel_norm:.6f}")
    print(f"  tolerance +/-{TOLERANCE_M} m against {os.path.basename(REF_RESULT)}")
    ok = True
    for prompt in REF_PROMPTS:
        waypoints, _ = policy.predict_waypoints(frames, prompt=prompt)
        record = policy.ticks[-1]
        ours = endpoint_lateral_m(waypoints)
        ref = expected.get(prompt)
        delta = abs(ours - ref) if ref is not None else float("nan")
        passed = ref is not None and delta <= TOLERANCE_M
        ok = ok and passed
        print(f"\n  \"{prompt}\"")
        print(f"    A={record['A']!r}  B={record['B']!r}  head={record.get('A_head')!r}")
        print(
            f"    dets={record['n_dets']} cands={record['n_candidates']}"
            f" relaxed={record['relaxed']} synthetic={record['synthetic_candidate']}"
        )
        print(
            f"    B by {record['B_mode']}  at {record['B_peak']}"
            f"  excluded={record['b_excluded']}  selected=c{record['selected']}"
        )
        print(
            "    scores: "
            + " | ".join(
                f"c{s['i']}({s['cat']}) s={s['score']:.3f} d={s['dist']:.3f}"
                f" f={s['final']:.3f}"
                for s in record["scores"]
            )
        )
        print(
            f"    endpoint lateral: ours {ours:+.5f} m   reference {ref:+.5f} m"
            f"   delta {delta:.5f} m   {'PASS' if passed else 'FAIL'}"
        )
        v, w, linear, angular = deployment_command(waypoints)
        print(
            f"    deployment scale (index {WAYPOINT_INDEX}, {METRIC_WAYPOINT_SPACING} m):"
            f" v={v:.3f} m/s w={w:+.3f} rad/s -> linear={linear:+.3f} angular={angular:+.3f}"
        )
        candidates = policy.last_debug["candidates"]
        if record["score_mode"] == "crop_cos" and candidates:
            direct, _ = policy.heatmap.crop_cos(
                frames[-1].convert("RGB"),
                [c["box"] for c in candidates],
                record["A"],
                policy.crop_margin,
                policy.score_template,
            )
            drift = max(
                abs(a - s["score"])
                for a, s in zip(direct, record["scores"])
            )
            drifted = drift > CROP_DRIFT_TOL
            ok = ok and not drifted
            print(
                f"    crop split: max|_crop_embeddings - crop_cos|"
                f" = {drift:.2e}   {'FAIL' if drifted else 'OK'}"
            )
    print(f"\n  {'PASS' if ok else 'FAIL'}: port reproduces the reference")
    return ok


def check_arm1(device, tick_log=None):
    """The A/B control has recorded numbers too, so hold it to them as well."""
    print("\n=== 2b. arm-1 control path ===")
    expected = reference_endpoints("arm1")
    if not expected:
        print(f"  SKIP: no recorded reference in {REF_RESULT}")
        return False
    policy = OursPolicy(device=device, arm1_only=True, tick_log=tick_log)
    frames = load_context(REF_FRAME, policy.CONTEXT_LEN)
    ok = True
    for prompt in REF_PROMPTS:
        waypoints, _ = policy.predict_waypoints(frames, prompt=prompt)
        ours = endpoint_lateral_m(waypoints)
        ref = expected.get(prompt)
        delta = abs(ours - ref) if ref is not None else float("nan")
        passed = ref is not None and delta <= TOLERANCE_M
        ok = ok and passed
        print(
            f"  \"{prompt}\"\n    endpoint lateral: ours {ours:+.5f} m"
            f"   reference {ref:+.5f} m   delta {delta:.5f} m"
            f"   {'PASS' if passed else 'FAIL'}"
        )
    print(f"  {'PASS' if ok else 'FAIL'}: arm-1 control reproduces the reference")
    del policy
    return ok


def check_timing(device, episodes, per_episode, tick_log=None, warmup_ticks=2):
    print("\n=== 3. timing: tick wall time vs the control period ===")
    policy = OursPolicy(device=device, tick_log=tick_log)
    prompts = [
        "the white van next to the red truck",
        "the person next to the car",
        "go to the truck",
    ]
    episode_names = sorted(
        d for d in os.listdir(DATASET_ROOT) if d.startswith("episode_")
    )[:episodes]
    walls, stages, fallbacks = [], {}, {}
    # The first ticks pay for CUDA kernel selection and cuDNN autotuning. They
    # are reported on their own rather than folded into the distribution, which
    # they otherwise dominate.
    warmup = []
    for episode in episode_names:
        image_dir = os.path.join(DATASET_ROOT, episode, "image")
        stems = sorted(f[:-4] for f in os.listdir(image_dir) if f.endswith(".jpg"))
        start = policy.CONTEXT_LEN - 1
        for n, k in enumerate(
            range(start, min(len(stems), start + per_episode))
        ):
            frames = [
                Image.open(
                    os.path.join(image_dir, stems[max(0, k - h)] + ".jpg")
                ).convert("RGB")
                for h in range(policy.CONTEXT_LEN - 1, -1, -1)
            ]
            for frame in frames:
                frame.load()  # PIL decodes lazily; keep JPEG decode off the clock
            began = time.time()
            policy.predict_waypoints(frames, prompt=prompts[n % len(prompts)])
            elapsed = (time.time() - began) * 1000.0
            record = policy.ticks[-1]
            tag = record.get("fallback") or "none"
            fallbacks[tag] = fallbacks.get(tag, 0) + 1
            if len(warmup) < warmup_ticks:
                warmup.append(elapsed)
                continue
            walls.append(elapsed)
            for key, value in record["timing_ms"].items():
                stages.setdefault(key, []).append(value)
    if not walls:
        print("  SKIP: no frames found")
        return False
    walls_arr = np.array(walls)
    print(f"  ticks {len(walls)} over {len(episode_names)} episodes")
    if warmup:
        print(
            "  warm-up (excluded): "
            + ", ".join(f"{v:.0f} ms" for v in warmup)
        )
    print(f"  budget {LOOP_BUDGET_MS:.0f} ms  (rate {1.0 / DT:.1f} Hz)")
    for name, values in sorted(stages.items()):
        arr = np.array(values)
        print(
            f"    {name:9s} median {np.median(arr):7.1f} ms   p90 {np.percentile(arr, 90):7.1f} ms"
        )
    print(
        f"  wall: median {np.median(walls_arr):.1f} ms   mean {walls_arr.mean():.1f} ms"
        f"   p90 {np.percentile(walls_arr, 90):.1f} ms   max {walls_arr.max():.1f} ms"
    )
    over = int((walls_arr > LOOP_BUDGET_MS).sum())
    print(f"  over budget: {over}/{len(walls)} ticks")
    print(f"  fallbacks: {fallbacks}")
    print(
        f"  {'PASS' if over == 0 else 'WARN'}: "
        f"median {np.median(walls_arr):.0f} ms of {LOOP_BUDGET_MS:.0f} ms"
    )
    return over == 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-regression", action="store_true")
    parser.add_argument("--skip-port", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument(
        "--with-arm1",
        action="store_true",
        help="also hold the arm-1 A/B control to its recorded numbers",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--per-episode", type=int, default=10)
    parser.add_argument("--warmup-ticks", type=int, default=2)
    parser.add_argument(
        "--tick-log", default=None, help="write ticks.jsonl and thumbs here"
    )
    args = parser.parse_args(argv)

    results = {}
    if not args.skip_regression:
        results["regression"] = check_regression(args.device)
    if not args.skip_port:
        results["port"] = check_port(args.device, args.tick_log)
    if args.with_arm1:
        results["arm1"] = check_arm1(args.device, args.tick_log)
    if not args.skip_timing:
        results["timing"] = check_timing(
            args.device, args.episodes, args.per_episode, args.tick_log,
            warmup_ticks=args.warmup_ticks,
        )

    print("\n=== summary ===")
    for name, passed in results.items():
        # None means the check could not run, which is not a failure.
        label = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
        print(f"  {name:11s} {label}")
    # Timing is reported, not gating: it is one machine's measurement. A skipped
    # check does not gate either -- only something that ran and failed.
    gating = [
        v for k, v in results.items() if k != "timing" and v is not None
    ]
    return 0 if all(gating) else 1


if __name__ == "__main__":
    sys.exit(main())
