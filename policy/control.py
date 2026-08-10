"""
Waypoint -> rover command.

Two conversions live here, and they are easy to conflate:

  1. waypoint -> (v, omega) in SI units. Ported from upstream
     `run_omnivla_edge.py`; keep it identical so the policy behaves the way it
     was trained/tuned to.
  2. (v, omega) -> the SDK's /control payload, which is NOT SI: `linear` and
     `angular` are normalized to [-1, 1]. That conversion needs the rover's
     actual top speed, which has to be measured. See MAX_LIN_MPS below.
"""

import json
import logging
import math
import os
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Waypoints come out of the model in units of this many metres.
METRIC_WAYPOINT_SPACING = 0.1

# Control period the upstream gains assume (3 Hz). Upstream keeps its control
# loop at the same rate, so dt doubles as "time allotted to reach the chosen
# waypoint" — change one and you change the effective gain.
DT = 1.0 / 3.0

# Which of the 8 predicted waypoints to chase. Upstream uses index 4.
#
# Note what this does and does not do: it picks a single lookahead point and
# steers at it, discarding the other seven. There is no path tracker, and none
# is needed, because the whole trajectory is thrown away and re-predicted from
# a fresh image every tick — the same receding-horizon idea as MPC. The cost is
# that the predicted curvature is ignored, so tight curves can get cut short.
WAYPOINT_INDEX = 4

# Upstream velocity envelope, in SI units.
MAX_V_MPS = 0.3
MAX_W_RPS = 0.3

# ---------------------------------------------------------------------------
# What "1.0" means on this rover.
#
# The model predicts a path in metres and this file turns it into m/s and
# rad/s, but /control takes linear/angular in [-1, 1]. These two numbers bridge
# the two. Manual driving never needed them because the operator was the
# feedback loop; the model has no such loop and has to be told the scale.
#
# 1.0 is a placeholder, and the rover will drive with it. What it costs:
#   - both wrong by the same factor -> the rover drives too fast or too slow,
#     but follows the shape of the predicted path;
#   - wrong by DIFFERENT factors -> the linear/angular ratio is off, so the
#     rover systematically over- or under-steers relative to the prediction.
#
# So this is tuning, not a prerequisite. Drive first; if the steering looks
# systematically off, measure with the Speed calibration panel in the operator
# page, which writes calibration.json next to this file.
# ---------------------------------------------------------------------------
MAX_LIN_MPS = 1.0
MAX_ANG_RPS = 1.0

CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "calibration.json"
)


def load_calibration() -> Dict[str, float]:
    """Measured speed constants, or {} if the rover has not been calibrated."""
    try:
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("ignoring unreadable %s: %s", CALIBRATION_PATH, exc)
        return {}
    return {
        key: float(data[key])
        for key in ("max_lin_mps", "max_ang_rps")
        if key in data
    }


def save_calibration(max_lin_mps: float, max_ang_rps: float) -> None:
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            {"max_lin_mps": max_lin_mps, "max_ang_rps": max_ang_rps},
            handle,
            indent=2,
        )
    logger.info(
        "calibration saved to %s (lin %.3f m/s, ang %.3f rad/s)",
        CALIBRATION_PATH,
        max_lin_mps,
        max_ang_rps,
    )


def clip_angle(theta: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (theta + math.pi) % (2 * math.pi) - math.pi


def waypoint_to_velocity(
    waypoints: np.ndarray,
    waypoint_index: int = WAYPOINT_INDEX,
    dt: float = DT,
    max_v: float = MAX_V_MPS,
    max_w: float = MAX_W_RPS,
    metric_waypoint_spacing: float = METRIC_WAYPOINT_SPACING,
) -> Tuple[float, float]:
    """Pick a waypoint and turn it into (linear m/s, angular rad/s).

    `waypoints` is the (8, 4) array from the policy: cumulative (dx, dy) in
    units of `metric_waypoint_spacing` plus a normalized (cos, sin) heading, in
    the robot frame (x forward, y left). Positive angular means turning left,
    which matches the SDK's `angular` sign.

    Ported verbatim from upstream, including the turn-radius-preserving limiter.
    """
    chosen = waypoints[waypoint_index].copy()
    chosen[:2] *= metric_waypoint_spacing
    dx, dy, hx, hy = chosen

    eps = 1e-8
    if abs(dx) < eps and abs(dy) < eps:
        v = 0.0
        w = clip_angle(math.atan2(hy, hx)) / dt
    elif abs(dx) < eps:
        v = 0.0
        w = float(np.sign(dy)) * math.pi / (2 * dt)
    else:
        v = dx / dt
        w = math.atan(dy / dx) / dt

    v = float(np.clip(v, 0.0, 0.5))
    w = float(np.clip(w, -1.0, 1.0))

    return _limit_velocity(v, w, max_v, max_w)


def _limit_velocity(
    v: float, w: float, max_v: float, max_w: float
) -> Tuple[float, float]:
    """Scale (v, w) into the envelope while preserving the turn radius.

    Returns plain floats, not numpy scalars — these end up in the UI's JSON
    state, and json.dumps cannot serialize np.float64.
    """
    sign_v, sign_w = float(np.sign(v)), float(np.sign(w))

    if abs(v) <= max_v:
        if abs(w) <= max_w:
            return float(v), float(w)
        radius = v / w
        return float(max_w * sign_v * abs(radius)), float(max_w * sign_w)

    if abs(w) <= 0.001:
        return float(max_v * sign_v), 0.0

    radius = v / w
    if abs(radius) >= max_v / max_w:
        return float(max_v * sign_v), float(max_v * sign_w / abs(radius))
    return float(max_w * sign_v * abs(radius)), float(max_w * sign_w)


def to_control_command(
    v_mps: float,
    w_rps: float,
    max_lin_mps: float = MAX_LIN_MPS,
    max_ang_rps: float = MAX_ANG_RPS,
    linear_cap: float = 1.0,
    angular_cap: float = 1.0,
) -> Tuple[float, float]:
    """SI velocities -> the SDK's normalized `linear`/`angular` in [-1, 1].

    `linear_cap` / `angular_cap` are an operator-facing safety throttle on top
    of the model's own envelope, in the same spirit as SPEED_SCALE in the manual
    driving pages.
    """
    linear = float(np.clip(v_mps / max_lin_mps, -linear_cap, linear_cap))
    angular = float(np.clip(w_rps / max_ang_rps, -angular_cap, angular_cap))
    return linear, angular
