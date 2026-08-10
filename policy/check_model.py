"""
Offline check: does the checkpoint load, does a forward pass run, how fast?

Run this before taking the rover outside — it needs no server and no rover.

    conda activate rover
    python -m policy.check_model --ckpt best.pth
    python -m policy.check_model --image screenshots/front.png
"""

import argparse
import sys
import time

import numpy as np
from PIL import Image

from .control import to_control_command, waypoint_to_velocity
from .omnivla_policy import (
    CONTEXT_LEN,
    DEFAULT_CHECKPOINT,
    OmniVLAEdgePolicy,
    relative_goal_to_pose,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image", help="a real frame; defaults to random noise")
    parser.add_argument("--prompt", default="the blue trash bin")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args(argv)

    print(f"loading {args.ckpt} on {args.device} ...")
    started = time.time()
    try:
        policy = OmniVLAEdgePolicy(args.ckpt, device=args.device)
    except FileNotFoundError as exc:
        # A missing checkpoint is the expected first-run case, not a crash.
        print(exc, file=sys.stderr)
        return 1
    print(f"loaded in {time.time() - started:.1f}s (state_dict matched strict=True)")

    if args.image:
        frame = Image.open(args.image).convert("RGB")
    else:
        rng = np.random.default_rng(0)
        frame = Image.fromarray(
            rng.integers(0, 255, (360, 640, 3), dtype=np.uint8), mode="RGB"
        )
    context = [frame] * CONTEXT_LEN

    cases = [
        ("language only", dict(prompt=args.prompt)),
        ("pose only", dict(goal_pose=relative_goal_to_pose(5.0, 0.0))),
        (
            "pose + language",
            dict(prompt=args.prompt, goal_pose=relative_goal_to_pose(5.0, 2.0)),
        ),
        ("goal image only", dict(goal_image=frame)),
    ]

    for name, kwargs in cases:
        waypoints, modality_id = policy.predict_waypoints(context, **kwargs)
        assert waypoints.shape == (8, 4), waypoints.shape
        v, w = waypoint_to_velocity(waypoints)
        linear, angular = to_control_command(v, w)
        print(
            f"\n{name}: modality_id={modality_id} waypoints={waypoints.shape}"
            f"\n  first 3 (x_m, y_m): "
            + ", ".join(f"({x * 0.1:+.2f}, {y * 0.1:+.2f})" for x, y, _, _ in waypoints[:3])
            + f"\n  v={v:.3f} m/s  omega={w:.3f} rad/s"
            f"  ->  linear={linear:+.3f} angular={angular:+.3f}"
        )

    # Timing on the modality that actually gets used.
    for _ in range(3):
        policy.predict_waypoints(context, prompt=args.prompt)
    started = time.time()
    for _ in range(args.iters):
        policy.predict_waypoints(context, prompt=args.prompt)
    per_iter = (time.time() - started) / args.iters * 1000
    print(f"\ninference: {per_iter:.1f} ms/frame ({1000 / per_iter:.1f} Hz headroom)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
