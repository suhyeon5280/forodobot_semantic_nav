"""One picture of what arm-4' does to a single frame.

    conda activate rover
    python -m policy.visualize_pipeline test.jpg --prompt "the black chair next to the orange chair"

Draws the whole per-tick path side by side: the detector's boxes, the target
and anchor heatmaps, the selection rule's arithmetic, the 4th channel the policy
is actually handed, and the trajectory that comes back.

The drawing helpers are the offline reference's own (`policy/refs/.../d150_e2e.py`),
so a panel here means the same thing it means there.

No robot and no server involved. A single image is fed in as all six context
frames, which is what the loop does when the rover is standing still, so the
trajectory is what the policy would predict from a standstill.
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

from .control import (
    METRIC_WAYPOINT_SPACING,
    WAYPOINT_INDEX,
    to_control_command,
    waypoint_to_velocity,
)
from .ours_policy import (
    PATCH_GRID,
    OursPolicy,
    _bootstrap_paths,
    _refs_cwd,
    parse_prompt,
)

# The reference reports endpoint lateral position at a different scale than the
# rover is driven at. Both are printed; see the two-scales note in the README.
REF_WAYPOINT_SPACING = 0.125


def _reference_drawing():
    """The offline reference's panel painters, imported rather than rewritten."""
    # Drawing needs neither the detector nor open_clip, so do not demand them.
    _bootstrap_paths(need_omni_deps=False)
    with _refs_cwd():
        from d150_e2e import _heat_rgb, panel_heat, panel_left, panel_traj, wrap
    return _heat_rgb, panel_heat, panel_left, panel_traj, wrap


def _grid_to_224(grid):
    """A 14x14 patch grid as a 224x224 map, min-max normalized for display."""
    g = np.clip(np.asarray(grid, np.float32), 0, None).reshape(
        PATCH_GRID, PATCH_GRID
    )
    span = max(float(g.max() - g.min()), 1e-8)
    g = (g - g.min()) / span
    return np.array(
        Image.fromarray((g * 255).astype(np.uint8)).resize(
            (224, 224), Image.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0


def render_compare(image_path, prompt, out_path, device="cuda:0", panel_px=460):
    """The two scoring paths side by side on the same frame.

    Left pair: which box each path picks. The heatmap path reads the
    localization head, which grounds the noun and ignores the adjective, so
    "black chair" and "orange chair" score alike. The crop path embeds the
    candidate's own pixels with the adapter, which is what separates them.
    """
    _, panel_heat, panel_left, panel_traj, wrap = _reference_drawing()

    frame = Image.open(image_path).convert("RGB")
    policy = OursPolicy(device=device)
    waypoints, _ = policy.predict_waypoints([frame] * policy.CONTEXT_LEN, prompt=prompt)
    record = policy.ticks[-1]
    debug = policy.last_debug
    scores = record["scores"]
    candidates = debug["candidates"]

    # Both winners, from the one run: the scores carry each path's number.
    new_sel = record["selected"]
    old_sel = int(
        np.argmax([s["heat_max"] - policy.lambda_d * s["dist"] for s in scores])
    )

    # The trajectory the old score would have produced. Same frame, same policy,
    # only the box that survives into the 4th channel differs, so this is the
    # driving consequence of the score change rather than an argument about it.
    old_channel = policy._channel(debug["grid_target"], candidates[old_sel]["box"])
    if old_sel == new_sel:
        old_waypoints = waypoints
    else:
        obs_img, goal_pose_t, map_images, black, goal_mask, text, current_img = (
            debug["policy_inputs"]
        )
        saved_log, policy.tick_log = policy.tick_log, None
        old_waypoints, _ = policy._run_arm4(
            obs_img, goal_pose_t, map_images, black, goal_mask,
            text, old_channel, current_img, dict(record), dict(record["timing_ms"]),
        )
        policy.tick_log = saved_log
        policy.ticks.pop()  # that re-run is not a tick of its own

    P = panel_px
    before = panel_left(frame, candidates, old_sel, debug["peak"], None, P)
    after = panel_left(frame, candidates, new_sel, debug["peak"], None, P)
    channel_before = panel_heat(
        old_channel[0, 0].numpy(), candidates[old_sel]["box"], P,
        "channel BEFORE",
    )
    channel_after = panel_heat(
        debug["channel"][0, 0].numpy(), record["selected_box"], P,
        "channel AFTER",
    )

    # panel_traj paints its first argument orange and its second blue, and
    # writes its own legend for the arm-1 comparison it was built for. Cover
    # that: here the two lines are two scoring paths, not two policies.
    traj = panel_traj(waypoints[:, :2], old_waypoints[:, :2], [], None, P, (2.0, 4.0))
    td = ImageDraw.Draw(traj)
    td.rectangle([0, 0, P, 60], fill=(0, 0, 0))
    td.text((6, 6), "(7) trajectory, from a standstill", fill=(255, 255, 255))
    td.text((6, 24), "AFTER  crop_cos", fill=(255, 150, 40))
    td.text((6, 40), "BEFORE heat_max", fill=(90, 150, 255))

    for panel, title, colour in (
        (before, "BEFORE  heat_max: heatmap in-box max", (120, 170, 255)),
        (after, "AFTER  crop_cos: adapter crop cosine", (255, 180, 80)),
    ):
        d = ImageDraw.Draw(panel)
        d.rectangle([0, 0, P, 18], fill=(0, 0, 0))
        d.text((6, 3), title, fill=colour)

    lines = [
        f'PROMPT: "{prompt}"    text embedded: {record["crop_text"]!r}',
        "",
        f'{"cand":<6s}{"heat_max":>10s}{"final":>9s}   |{"crop_cos":>10s}{"final":>9s}   dist',
    ]
    for s in scores:
        old_final = s["heat_max"] - policy.lambda_d * s["dist"]
        mark_old = " <-BEFORE" if s["i"] == old_sel else "         "
        mark_new = " <-AFTER" if s["i"] == new_sel else ""
        lines.append(
            f'c{s["i"]:<5d}{s["heat_max"]:>10.4f}{old_final:>9.4f}{mark_old}'
            f'|{s["score"]:>10.4f}{s["final"]:>9.4f}{mark_new}   {s["dist"]:.3f}'
        )
    end_before = float(old_waypoints[7, 1] * REF_WAYPOINT_SPACING)
    end_after = float(waypoints[7, 1] * REF_WAYPOINT_SPACING)
    v_b, w_b = waypoint_to_velocity(old_waypoints)
    v_a, w_a = waypoint_to_velocity(waypoints)
    lines += [
        "",
        (f'selection moved from c{old_sel} to c{new_sel}'
         if old_sel != new_sel else f'both paths pick c{new_sel}'),
        f'endpoint lateral   BEFORE {end_before:+.3f} m     AFTER {end_after:+.3f} m',
        f'rover command      BEFORE v={v_b:.3f} w={w_b:+.3f}     '
        f'AFTER v={v_a:.3f} w={w_a:+.3f}',
    ]

    panels = [before, after, channel_before, channel_after, traj]
    text_h = 30 + 15 * len(lines)
    canvas = Image.new("RGB", (P * len(panels), P + text_h + 26), (12, 12, 14))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 7),
        f"attribute score: heatmap max vs adapter crop cosine  |  "
        f"{os.path.basename(image_path)}  |  same detector boxes, same rule, "
        f"same distance term; only the score changes",
        fill=(255, 255, 255),
    )
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * P, 26))
    wrap(draw, "\n".join(lines), 8, P + 32, 300)
    canvas.save(out_path)

    print(f'\nprompt: {prompt}')
    print(f'  BEFORE (heat_max) -> c{old_sel}  endpoint {end_before:+.3f} m')
    print(f'  AFTER  (crop_cos) -> c{new_sel}  endpoint {end_after:+.3f} m')
    for s in scores:
        print(
            f'   c{s["i"]} heat_max={s["heat_max"]:.4f}  crop_cos={s["score"]:.4f}'
            f'  dist={s["dist"]:.3f}'
        )
    print(f'  saved: {out_path}')
    return out_path


def render(image_path, prompt, out_path, device="cuda:0", panel_px=420):
    heat_rgb, panel_heat, panel_left, panel_traj, wrap = _reference_drawing()

    frame = Image.open(image_path).convert("RGB")
    policy = OursPolicy(device=device)
    # A standing rover sees the same frame six times; that is also what the loop
    # feeds the policy before its history has filled up.
    context = [frame] * policy.CONTEXT_LEN

    waypoints, modality = policy.predict_waypoints(context, prompt=prompt)
    record = policy.ticks[-1]
    debug = policy.last_debug
    target, anchor = parse_prompt(prompt)

    P = panel_px
    detections = debug["detections"] if debug else []
    candidates = debug["candidates"] if debug else []
    selected = record["selected"]
    peak = debug["peak"] if debug else None
    sel_box = record.get("selected_box")

    # (1) detector boxes, the anchor's heatmap peak, and which box won.
    left = panel_left(frame, candidates, selected, peak, None, P)

    # (2)(3) the two heatmaps, before the box mask is applied.
    target_map = panel_heat(
        _grid_to_224(debug["grid_target"]), sel_box, P,
        f"(3) target heatmap: {target!r}",
    )
    if debug.get("grid_anchor") is not None:
        anchor_map = panel_heat(
            _grid_to_224(debug["grid_anchor"]), None, P,
            f"(4) anchor heatmap: {anchor!r}",
        )
    else:
        anchor_map = panel_heat(
            np.zeros((224, 224), np.float32), None, P,
            "(4) no anchor: no relation word",
        )

    # (6) what the policy is actually handed as its 4th channel.
    channel = debug["channel"][0, 0].numpy()
    channel_panel = panel_heat(channel, sel_box, P, "(6) policy input channel")

    # (7) the trajectory. Drawn at the reference's scale so it lines up with the
    # offline pictures; the rover's own scale is printed in the text block.
    traj = panel_traj(
        waypoints[:, :2], None,
        [((d["box"][0] + d["box"][2]) / 2.0) for d in candidates],
        selected, P, (2.0, 4.0),
    )
    head = ImageDraw.Draw(traj)
    head.rectangle([0, 0, P, 18], fill=(0, 0, 0))
    head.text((6, 3), "(7) predicted trajectory (arm-4')", fill=(255, 200, 120))

    scores = record["scores"]
    endpoint_ref = float(waypoints[7, 1] * REF_WAYPOINT_SPACING)
    v, w = waypoint_to_velocity(waypoints)
    linear, angular = to_control_command(v, w)

    lines = [
        f'PROMPT: "{prompt}"',
        f'(1) parse   target={target!r}  anchor={anchor!r}'
        f'   head={record.get("A_head")!r}',
        f'(2) detector: {record["n_dets"]} boxes -> {record["n_candidates"]} candidates'
        + ("  [relaxed to all boxes]" if record["relaxed"] else "")
        + ("  [synthetic box on the target peak]" if record["synthetic_candidate"] else ""),
        f'(5) rule  final = score - {policy.lambda_d} * dist:  '
        + " | ".join(
            f'c{s["i"]}({s["cat"]}) s={s["score"]:.3f} d={s["dist"]:.3f} f={s["final"]:.3f}'
            for s in scores
        ),
        f'(5) selected = c{selected}'
        + (f'  anchor peak=({peak[0]:.3f}, {peak[1]:.3f})' if peak else '  (no anchor: score only)'),
        f'(7) endpoint lateral {endpoint_ref:+.3f} m at the reference scale '
        f'(index 7, {REF_WAYPOINT_SPACING} m)',
        f'    rover command: index {WAYPOINT_INDEX}, {METRIC_WAYPOINT_SPACING} m -> '
        f'v={v:.3f} m/s  w={w:+.3f} rad/s  ->  linear={linear:+.3f}  angular={angular:+.3f}',
        f'    modality_id={modality} (language only)   '
        + "  ".join(f'{k}={v:.0f}ms' for k, v in record["timing_ms"].items()),
    ]

    panels = [left, target_map, anchor_map, channel_panel, traj]
    text_h = 132
    canvas = Image.new("RGB", (P * len(panels), P + text_h + 26), (12, 12, 14))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 7),
        f"arm-4' pipeline on {os.path.basename(image_path)}  |  "
        f"one frame repeated as all {policy.CONTEXT_LEN} context frames  |  "
        "RGB is zeroed before the policy sees it; only the channel carries the target",
        fill=(255, 255, 255),
    )
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * P, 26))
    wrap(draw, "\n".join(lines), 8, P + 32, 300)
    canvas.save(out_path)

    print(f"\nprompt   : {prompt}")
    print(f"parse    : target={target!r} anchor={anchor!r}")
    print(f"detector : {record['n_dets']} boxes -> {record['n_candidates']} candidates")
    for s in scores:
        mark = "<-- selected" if s["i"] == selected else ""
        print(
            f"   c{s['i']:<2d} {s['cat']:<12s} score={s['score']:.3f} "
            f"dist={s['dist']:.3f} final={s['final']:.3f} {mark}"
        )
    print(f"endpoint : {endpoint_ref:+.3f} m lateral (reference scale)")
    print(f"command  : linear={linear:+.3f} angular={angular:+.3f}")
    print(f"saved    : {out_path}")
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="a single frame, e.g. test.jpg")
    parser.add_argument(
        "--prompt", required=True, help='e.g. "the black chair next to the orange chair"'
    )
    parser.add_argument("--out", default=None, help="output PNG")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--panel-px", type=int, default=420)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="the two scoring paths side by side instead of the full pipeline",
    )
    args = parser.parse_args(argv)

    out = args.out or os.path.splitext(os.path.basename(args.image))[0] + "_pipeline.png"
    if args.compare:
        render_compare(args.image, args.prompt, out, args.device, args.panel_px)
    else:
        render(args.image, args.prompt, out, args.device, args.panel_px)
    return 0


if __name__ == "__main__":
    sys.exit(main())
