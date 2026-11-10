"""Read the fill-stop numbers: from a run's log, or from still images.

    python -m policy.fill_report                  # last 60 ticks of field_log/fill.jsonl
    python -m policy.fill_report -n 300 --running # only ticks while driving
    python -m policy.fill_report --image a.jpg b.jpg

Images are measured with the same defaults as the loop; pass the same
--band / --center-width / --center-std / --ratio / --spread / --trim-bright as
the run to compare. `--spread rgb` next to the default shows what folds and
sheen were costing.
"""

import argparse
import json
import os
import sys
import time

from PIL import Image

from .proximity import FillConfig, describe_fill_rule, measure

DEFAULT_LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "field_log", "fill.jsonl"
)


def _row(fill: dict) -> str:
    return "center %5.1f  outer %5.1f  ratio %5.2f  %s" % (
        fill["center_std"],
        fill["outer_std"],
        fill["ratio"],
        "FULL " if fill["full"] else "clear",
    )


def report_log(path: str, last: int, running_only: bool) -> int:
    if not os.path.exists(path):
        print(f"{path} not found. It is written by `python -m policy.run_autonomy` "
              "unless --fill-stop off.")
        return 1
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    if running_only:
        rows = [r for r in rows if r.get("running")]
    shown = rows[-last:]
    print(f"{path}: {len(rows)} ticks, showing the last {len(shown)}")
    print("RUN = driving, --- = paused.  A stop needs FULL on consecutive ticks.\n")
    prompt = object()
    for r in shown:
        if r.get("prompt") != prompt:
            prompt = r.get("prompt")
            print(f'  prompt: "{prompt}"')
        stamp = time.strftime("%H:%M:%S", time.localtime(r["t"])) + ("%.1f" % (r["t"] % 1))[1:]
        print("  %s  %s  %s  streak %d  %s" % (
            stamp,
            "RUN" if r.get("running") else "---",
            _row(r),
            r.get("streak", 0),
            r.get("action") or "",
        ))
    driving = [r for r in rows if r.get("running")]
    if driving:
        print("\nwhile driving: %d ticks, %d FULL, %d stop/would_stop, lowest centre %.1f" % (
            len(driving),
            sum(1 for r in driving if r["full"]),
            sum(1 for r in driving if r.get("action")),
            min(r["center_std"] for r in driving),
        ))
    return 0


def report_images(paths, config: FillConfig) -> int:
    print("band %.2f-%.2f, centre width %.2f, full when %s\n" % (
        config.band[0], config.band[1], config.center_width, describe_fill_rule(config),
    ))
    for path in paths:
        print("  %s  %s" % (_row(measure(Image.open(path), config)), path))
    return 0


def main(argv=None) -> int:
    defaults = FillConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("-n", type=int, default=60, help="how many ticks to show")
    parser.add_argument("--running", action="store_true", help="only ticks while driving")
    parser.add_argument("--image", nargs="+", help="measure still images instead of the log")
    parser.add_argument("--band", type=float, nargs=2, metavar=("TOP", "BOTTOM"), default=defaults.band)
    parser.add_argument("--center-width", type=float, default=defaults.center_width)
    parser.add_argument("--center-std", type=float, default=defaults.max_center_std)
    parser.add_argument("--ratio", type=float, default=defaults.max_ratio)
    parser.add_argument("--spread", choices=("colour", "rgb"), default=defaults.spread)
    parser.add_argument("--trim-bright", type=float, default=defaults.trim_bright)
    args = parser.parse_args(argv)

    if args.image:
        config = FillConfig(
            band=tuple(args.band),
            center_width=args.center_width,
            max_center_std=args.center_std,
            max_ratio=args.ratio,
            spread=args.spread,
            trim_bright=args.trim_bright,
        )
        try:
            config.validate()
        except ValueError as exc:
            parser.error(str(exc))
        return report_images(args.image, config)
    return report_log(args.log, args.n, args.running)


if __name__ == "__main__":
    sys.exit(main())
