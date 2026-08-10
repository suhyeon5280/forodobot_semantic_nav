"""
Drive the rover with OmniVLA-edge.

Runs as its own process against an already-running SDK server:

    conda activate rover-sdk    && hypercorn main:app    # terminal 1, owns the rover
    conda activate rover-policy && python -m policy.run_autonomy --ckpt best.pth

Then open http://localhost:8000/static/autonomy_control.html, type where the
rover should go, and press Start. The instruction is normally typed there rather
than passed on the command line; --prompt only pre-fills it. The rover stays
stopped until you press Start.

The page is served by the SDK server from static/, and talks to this process on
--ui-port over CORS. This process serves the same file too, so
http://localhost:8010 works if the SDK server is not up yet.

Safety notes, because /control latches the last command:
  - the loop always sends a zero command on the way out, including on Ctrl+C;
  - any inference or transport error pauses the loop and stops the rover;
  - --linear-cap / --angular-cap throttle the model's output;
  - the page's E-STOP posts zeros straight to the SDK server rather than going
    through this process, so it still halts the rover if this process dies.
"""

import argparse
import collections
import json
import logging
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

from PIL import Image

from .control import (
    DT,
    MAX_ANG_RPS,
    MAX_LIN_MPS,
    METRIC_WAYPOINT_SPACING,
    WAYPOINT_INDEX,
    load_calibration,
    save_calibration,
    to_control_command,
    waypoint_to_velocity,
)
from .omnivla_policy import (
    CONTEXT_LEN,
    DEFAULT_CHECKPOINT,
    OmniVLAEdgePolicy,
    checkpoint_missing_message,
    gps_goal_to_pose,
    relative_goal_to_pose,
)
from .rover_client import RoverClient

logger = logging.getLogger("autonomy")

HERE = os.path.dirname(os.path.abspath(__file__))
# The operator page lives in static/ so the SDK server serves it alongside the
# manual driving pages. This process serves the same file, so there is only ever
# one copy to keep up to date.
AUTONOMY_PAGE = os.path.join(os.path.dirname(HERE), "static", "autonomy_control.html")


class AutonomyLoop:
    """Capture -> infer -> drive, at a fixed rate, on a background thread."""

    def __init__(
        self,
        policy: OmniVLAEdgePolicy,
        client: RoverClient,
        *,
        prompt: Optional[str],
        goal_image: Optional[Image.Image],
        goal_latlon: Optional[tuple],
        relative_goal: Optional[tuple],
        rate: float,
        context_stride: int,
        waypoint_index: int,
        linear_cap: float,
        angular_cap: float,
        max_lin_mps: float,
        max_ang_rps: float,
        dry_run: bool,
        record: bool,
    ) -> None:
        self.policy = policy
        self.client = client
        self.prompt = prompt
        self.goal_image = goal_image
        self.goal_latlon = goal_latlon
        self.relative_goal = relative_goal
        self.rate = rate
        self.context_stride = context_stride
        self.waypoint_index = waypoint_index
        self.linear_cap = linear_cap
        self.angular_cap = angular_cap
        self.max_lin_mps = max_lin_mps
        self.max_ang_rps = max_ang_rps
        self.dry_run = dry_run
        self.record = record

        # Deep enough to pull CONTEXT_LEN frames spaced `context_stride` apart.
        self._frames = collections.deque(
            maxlen=1 + (CONTEXT_LEN - 1) * max(1, context_stride)
        )

        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._running = False
        self._session_name: Optional[str] = None
        # A timed open-loop move used for speed calibration. Driven from the
        # main loop rather than a thread so it shares the same stop paths.
        self._calibration: Optional[dict] = None
        self.state = {
            "running": False,
            "prompt": prompt,
            "tick": 0,
            "linear": 0.0,
            "angular": 0.0,
            "v_mps": 0.0,
            "w_rps": 0.0,
            "modality_id": None,
            "waypoints": [],
            "frame": None,
            "infer_ms": 0.0,
            "loop_ms": 0.0,
            "error": None,
            "dry_run": dry_run,
            "record": record,
            "session": None,
            "calibrating": None,
            "calibration_left": 0.0,
            "max_lin_mps": max_lin_mps,
            "max_ang_rps": max_ang_rps,
            # Lets the page find the SDK server for its own emergency stop.
            "server": client.base_url,
        }

    # -- operator commands -------------------------------------------------

    def has_goal(self) -> bool:
        return any(
            [self.prompt, self.goal_image, self.goal_latlon, self.relative_goal]
        )

    def start(self) -> None:
        if not self.has_goal():
            raise ValueError("no goal set — type an instruction first")
        if self._calibration:
            raise ValueError("a calibration run is in progress")
        with self._lock:
            if self._running:
                return
            self._running = True
            self.state["running"] = True
            self.state["error"] = None
        if self.record:
            self._begin_recording()
        logger.info("autonomy started (prompt=%r)", self.prompt)

    def stop(self, reason: Optional[str] = None) -> None:
        # A calibration run is also "driving", so Stop and E-STOP must end it.
        calibrating = self._end_calibration()
        with self._lock:
            was_running = self._running or calibrating
            self._running = False
            self.state["running"] = False
            self.state["linear"] = 0.0
            self.state["angular"] = 0.0
            if reason:
                self.state["error"] = reason
        if not was_running:
            return
        # Send the halt more than once: /control latches, so a single dropped
        # packet would leave the rover driving.
        for _ in range(3):
            self.client.stop()
        self._end_recording()
        logger.info("autonomy stopped%s", f" ({reason})" if reason else "")

    def set_prompt(self, prompt: str) -> None:
        with self._lock:
            self.prompt = prompt
            self.state["prompt"] = prompt
        logger.info("prompt set to %r", prompt)

    def set_record(self, on: bool) -> None:
        """Toggle dataset recording from the operator page.

        Recording is driven from here rather than from the page so that frames
        stay 1:1 with the commands the policy actually issued — the page polls
        at its own rate and would duplicate or drop frames.
        """
        with self._lock:
            self.record = on
            self.state["record"] = on
        if on and self._running:
            self._begin_recording()
        elif not on:
            self._end_recording()

    def _begin_recording(self) -> None:
        if self._session_name:
            return
        try:
            session = self.client.dataset_start()
            self._session_name = session.get("session_name")
            self.state["session"] = self._session_name
            logger.info("recording to %s", session.get("session_dir"))
        except Exception as exc:  # noqa: BLE001 - recording is not critical
            logger.error("could not start recording: %s", exc)

    def start_calibration(self, kind: str, value: float, seconds: float) -> None:
        """Drive one fixed open-loop command for a fixed time, then stop.

        This is how the operator measures what `linear=0.5` actually means in
        m/s. It runs inside the main loop, so Stop, E-STOP and process exit all
        end it exactly the way they end normal driving.
        """
        if self._running:
            raise ValueError("stop autonomy before calibrating")
        if self._calibration:
            raise ValueError("a calibration run is already in progress")
        if kind not in ("linear", "angular"):
            raise ValueError(f"unknown calibration kind: {kind}")
        if not 0.0 < abs(value) <= 1.0:
            raise ValueError("calibration value must be within (0, 1]")
        if not 0.0 < seconds <= 60.0:
            raise ValueError("calibration duration must be within (0, 60] seconds")
        if self.dry_run:
            raise ValueError("calibration needs to actually move the rover")

        command = (value, 0.0) if kind == "linear" else (0.0, value)
        with self._lock:
            self._calibration = {
                "command": command,
                "until": time.monotonic() + seconds,
            }
            self.state["calibrating"] = kind
            self.state["calibration_left"] = seconds
            self.state["error"] = None
        logger.info("calibration run: %s=%.2f for %.1fs", kind, value, seconds)

    def _end_calibration(self) -> bool:
        """End any calibration run. Returns whether one was active."""
        with self._lock:
            active = self._calibration is not None
            self._calibration = None
            self.state["calibrating"] = None
            self.state["calibration_left"] = 0.0
            if active:
                self.state["linear"] = 0.0
                self.state["angular"] = 0.0
        if active:
            for _ in range(3):
                self.client.stop()
            logger.info("calibration run ended")
        return active

    def set_calibration(self, max_lin_mps: float, max_ang_rps: float) -> None:
        """Store measured speed constants and use them from the next tick on."""
        if max_lin_mps <= 0 or max_ang_rps <= 0:
            raise ValueError("calibration values must be positive")
        with self._lock:
            self.max_lin_mps = max_lin_mps
            self.max_ang_rps = max_ang_rps
            self.state["max_lin_mps"] = max_lin_mps
            self.state["max_ang_rps"] = max_ang_rps
        save_calibration(max_lin_mps, max_ang_rps)

    def _end_recording(self) -> None:
        if not self._session_name:
            return
        try:
            summary = self.client.dataset_stop()
            logger.info("recording stopped: %s", summary)
        except Exception as exc:  # noqa: BLE001
            logger.error("could not stop recording: %s", exc)
        self._session_name = None
        self.state["session"] = None

    def shutdown(self) -> None:
        self._shutdown.set()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)

    # -- main loop ---------------------------------------------------------

    def run(self) -> None:
        period = 1.0 / self.rate
        while not self._shutdown.is_set():
            started = time.time()
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001 - never let the loop die hot
                logger.exception("tick failed")
                self.stop(reason=f"{type(exc).__name__}: {exc}")
            elapsed = time.time() - started
            with self._lock:
                self.state["loop_ms"] = elapsed * 1000.0
            if elapsed > period:
                logger.warning(
                    "tick overran: %.0f ms > %.0f ms budget",
                    elapsed * 1000,
                    period * 1000,
                )
            self._shutdown.wait(max(0.0, period - elapsed))

    def _tick(self) -> None:
        frame = self.client.front_frame()
        if frame is None:
            return
        self._frames.append(frame)

        if self._calibration:
            self._calibration_tick()
            with self._lock:
                self.state["frame"] = frame.base64
            return

        running = self._running
        # Keep the history warm while paused so pressing Start does not run the
        # policy on frames from minutes ago.
        if not running:
            with self._lock:
                self.state["frame"] = frame.base64
            return

        context = self._context_frames()
        goal_pose = self._goal_pose()

        infer_started = time.time()
        waypoints, modality_id = self.policy.predict_waypoints(
            context,
            prompt=self.prompt,
            goal_pose=goal_pose,
            goal_image=self.goal_image,
        )
        infer_ms = (time.time() - infer_started) * 1000.0

        # dt is the time allotted to reach the chosen waypoint, and upstream
        # sets it equal to its control period (both 1/3 s). Deriving it from
        # the configured rate keeps that relationship if --rate is changed;
        # hardcoding DT would silently alter the effective gain instead.
        v_mps, w_rps = waypoint_to_velocity(
            waypoints, waypoint_index=self.waypoint_index, dt=1.0 / self.rate
        )
        linear, angular = to_control_command(
            v_mps,
            w_rps,
            max_lin_mps=self.max_lin_mps,
            max_ang_rps=self.max_ang_rps,
            linear_cap=self.linear_cap,
            angular_cap=self.angular_cap,
        )

        if not self.dry_run:
            self.client.control(linear, angular)
            if self.record and self._session_name:
                try:
                    self.client.dataset_log_frame(
                        frame.timestamp, linear, angular, frame.base64
                    )
                except Exception as exc:  # noqa: BLE001 - logging must not stop driving
                    logger.error("frame logging failed: %s", exc)

        with self._lock:
            self.state.update(
                {
                    "tick": self.state["tick"] + 1,
                    "linear": linear,
                    "angular": angular,
                    "v_mps": v_mps,
                    "w_rps": w_rps,
                    "modality_id": modality_id,
                    # metres in the robot frame, x forward / y left
                    "waypoints": (
                        waypoints[:, :2] * METRIC_WAYPOINT_SPACING
                    ).tolist(),
                    "frame": frame.base64,
                    "infer_ms": infer_ms,
                    "error": None,
                }
            )

    def _calibration_tick(self) -> None:
        calibration = self._calibration
        if calibration is None:
            return
        remaining = calibration["until"] - time.monotonic()
        if remaining <= 0:
            self._end_calibration()
            return
        linear, angular = calibration["command"]
        self.client.control(linear, angular)
        with self._lock:
            self.state["calibration_left"] = round(remaining, 1)
            self.state["linear"] = linear
            self.state["angular"] = angular

    def _context_frames(self) -> List[Image.Image]:
        """CONTEXT_LEN frames, oldest first, spaced `context_stride` apart.

        Before the buffer is deep enough the oldest frame is repeated, which is
        what upstream's sample does when the robot is standing still.
        """
        buf = list(self._frames)
        last = len(buf) - 1
        indices = [max(0, last - i * self.context_stride) for i in range(CONTEXT_LEN)]
        return [buf[i].image for i in reversed(indices)]

    def _goal_pose(self) -> Optional[List[float]]:
        if self.relative_goal is not None:
            return relative_goal_to_pose(*self.relative_goal)
        if self.goal_latlon is None:
            return None
        data = self.client.data()
        lat, lon = data.get("latitude"), data.get("longitude")
        heading = data.get("orientation")
        # The rover reports 1000 when the GPS has no fix; feeding that in would
        # produce a goal pose thousands of kilometres away.
        if lat is None or lon is None or abs(lat) > 90 or abs(lon) > 180:
            raise RuntimeError(
                f"no usable GPS fix (latitude={lat}, longitude={lon}); "
                "use --prompt or --relative-goal instead"
            )
        return gps_goal_to_pose(
            float(lat), float(lon), float(heading or 0.0), *self.goal_latlon
        )


# ---------------------------------------------------------------------------
# Operator UI
# ---------------------------------------------------------------------------


def make_ui_server(loop: AutonomyLoop, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: A003 - silence per-request logging
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # The page is normally served by the SDK server on another port, so
            # every call here is cross-origin. This process binds localhost-only
            # traffic from an operator's own machine.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):  # noqa: N802 - CORS preflight for the JSON POSTs
            self._send(204, b"", "text/plain")

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path.startswith("/state"):
                body = json.dumps(loop.snapshot()).encode()
                self._send(200, body, "application/json")
            elif self.path.split("?")[0] in ("/", "/index.html"):
                try:
                    with open(AUTONOMY_PAGE, "rb") as handle:
                        self._send(200, handle.read(), "text/html; charset=utf-8")
                except FileNotFoundError:
                    self._send(
                        404,
                        b"static/autonomy_control.html not found; run this from "
                        b"the repository root",
                        "text/plain",
                    )
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            action = payload.get("action")
            if action == "start":
                if payload.get("prompt"):
                    loop.set_prompt(payload["prompt"].strip())
                try:
                    loop.start()
                except ValueError as exc:
                    self._send(
                        400,
                        json.dumps({"error": str(exc)}).encode(),
                        "application/json",
                    )
                    return
            elif action in ("stop", "estop"):
                # No reason string: the PAUSED badge already says what happened,
                # and the error line is for failures the operator did not cause.
                loop.stop()
            elif action == "prompt":
                # An empty instruction would still select the language modality
                # and hand CLIP an empty string, so reject it outright.
                prompt = (payload.get("prompt") or "").strip()
                if not prompt:
                    self._send(
                        400, b'{"error":"prompt is empty"}', "application/json"
                    )
                    return
                loop.set_prompt(prompt)
            elif action == "record":
                loop.set_record(bool(payload.get("on")))
            elif action in ("calibrate_move", "calibrate_set"):
                try:
                    if action == "calibrate_move":
                        loop.start_calibration(
                            payload.get("kind", ""),
                            float(payload.get("value", 0.5)),
                            float(payload.get("seconds", 10)),
                        )
                    else:
                        loop.set_calibration(
                            float(payload["max_lin_mps"]),
                            float(payload["max_ang_rps"]),
                        )
                except (ValueError, KeyError, TypeError) as exc:
                    self._send(
                        400,
                        json.dumps({"error": str(exc)}).encode(),
                        "application/json",
                    )
                    return
            else:
                self._send(400, b'{"error":"unknown action"}', "application/json")
                return
            self._send(200, json.dumps(loop.snapshot()).encode(), "application/json")

    return ThreadingHTTPServer(("0.0.0.0", port), Handler)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _pair(value: str) -> tuple:
    parts = [float(p) for p in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated numbers")
    return tuple(parts)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt", default=DEFAULT_CHECKPOINT, help="OmniVLA-edge checkpoint"
    )
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--device", default="cuda:0")

    goal = parser.add_argument_group(
        "goal (all optional — the instruction is normally typed in the UI)"
    )
    goal.add_argument("--prompt", help='language instruction, e.g. "the red door"')
    goal.add_argument("--goal-image", help="path to an egocentric goal image")
    goal.add_argument(
        "--goal-latlon",
        type=_pair,
        metavar="LAT,LON",
        help="GPS goal; experimental, needs a real fix and a verified compass",
    )
    goal.add_argument(
        "--relative-goal",
        type=_pair,
        metavar="FORWARD_M,LEFT_M",
        help="goal in the robot frame right now, for testing the pose modality",
    )

    loop_group = parser.add_argument_group("loop")
    loop_group.add_argument("--rate", type=float, default=1.0 / DT)
    loop_group.add_argument(
        "--context-stride",
        type=int,
        default=1,
        help="ticks between history frames; 1 means one frame per tick",
    )
    loop_group.add_argument("--waypoint-index", type=int, default=WAYPOINT_INDEX)

    safety = parser.add_argument_group("safety")
    safety.add_argument("--linear-cap", type=float, default=0.5)
    safety.add_argument("--angular-cap", type=float, default=0.5)
    # Default None so calibration.json, written by the page, wins over the
    # placeholder in control.py but still loses to an explicit flag.
    safety.add_argument("--max-lin-mps", type=float, default=None)
    safety.add_argument("--max-ang-rps", type=float, default=None)
    safety.add_argument(
        "--dry-run",
        action="store_true",
        help="run inference and show the result without sending any command",
    )
    safety.add_argument("--autostart", action="store_true")

    parser.add_argument("--record", action="store_true", help="log the run via /dataset/*")
    parser.add_argument("--ui-port", type=int, default=8010)
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)
    has_goal = any(
        [args.prompt, args.goal_image, args.goal_latlon, args.relative_goal]
    )
    if args.autostart and not has_goal:
        parser.error(
            "--autostart needs a goal up front: --prompt, --goal-image, "
            "--goal-latlon or --relative-goal"
        )
    if args.no_ui and not has_goal:
        parser.error("--no-ui needs a goal up front, since nothing can set one")
    if args.goal_latlon and args.relative_goal:
        parser.error("--goal-latlon and --relative-goal are mutually exclusive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not os.path.exists(args.ckpt):
        logger.error("%s", checkpoint_missing_message(args.ckpt))
        return 1

    goal_image = (
        Image.open(args.goal_image).convert("RGB") if args.goal_image else None
    )

    calibration = load_calibration()
    max_lin_mps = args.max_lin_mps or calibration.get("max_lin_mps", MAX_LIN_MPS)
    max_ang_rps = args.max_ang_rps or calibration.get("max_ang_rps", MAX_ANG_RPS)
    if calibration:
        logger.info(
            "using saved calibration: %.3f m/s, %.3f rad/s at full scale",
            max_lin_mps,
            max_ang_rps,
        )
    else:
        logger.info(
            "no saved calibration — using placeholder speed constants. Fine to "
            "drive with; measure them from the page if the steering looks off."
        )

    logger.info("loading %s on %s", args.ckpt, args.device)
    policy = OmniVLAEdgePolicy(args.ckpt, device=args.device)

    client = RoverClient(args.server)
    logger.info("waiting for the rover video stream via %s", args.server)
    if not client.wait_until_ready():
        logger.error("no video stream; is `hypercorn main:app` running?")
        return 1

    loop = AutonomyLoop(
        policy,
        client,
        prompt=args.prompt,
        goal_image=goal_image,
        goal_latlon=args.goal_latlon,
        relative_goal=args.relative_goal,
        rate=args.rate,
        context_stride=args.context_stride,
        waypoint_index=args.waypoint_index,
        linear_cap=args.linear_cap,
        angular_cap=args.angular_cap,
        max_lin_mps=max_lin_mps,
        max_ang_rps=max_ang_rps,
        dry_run=args.dry_run,
        record=args.record,
    )

    server = None
    if not args.no_ui:
        server = make_ui_server(loop, args.ui_port)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        logger.info(
            "operator UI: %s/static/autonomy_control.html "
            "(or http://localhost:%d)",
            args.server.rstrip("/"),
            args.ui_port,
        )

    def handle_signal(signum, _frame):
        logger.info("caught signal %s, stopping", signum)
        loop.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if args.autostart:
        loop.start()
    elif loop.has_goal():
        logger.info("paused — press Start in the UI")
    else:
        logger.info("paused — type an instruction in the UI, then press Start")

    try:
        loop.run()
    finally:
        loop.stop()
        client.stop()  # unconditional: never leave the rover latched on
        if server:
            server.shutdown()
        logger.info("rover stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
