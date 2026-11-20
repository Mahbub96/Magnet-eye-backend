"""Swipe format v1 -> solver input (`Observations`): the steps the backend runs on every upload.

Units are fixed before any maths: field µT, position m, time s (from int64 ns), angles rad.
Every stream is brought to one timeline (the magnetometer's), one frame (world, z up) and one
calibration state (hard and soft iron removed) before the baseline is subtracted:

  PR-4  orientation (slerp) and position (linear) interpolated to each magnetometer time
  PR-1  calibrated = soft_iron @ (raw - hard_iron), phone frame, recomputed from raw here
  PR-5  rotated into the world frame with the orientation at the same instant
  PR-3  sensor position = reference point + R @ sensor offset
  PR-2  session baseline subtracted in the world frame, after rotation
  PR-20 saturated readings dropped

Ruler and grid positions between the marks assume constant speed; that is why those modes
carry a larger position jitter (error_sizes, set by the app).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .frames import apply_iron_correction, to_world
from .observations import Observations
from .swipe_format import Swipe


@dataclass
class AlignReport:
    """What happened to one swipe on the way in, for the app and for Phase 3 checks."""

    readings: int
    kept: int
    saturated: int
    outside_orientation: int  # readings outside the orientation stream's time span (clamped)
    max_iron_mismatch_ut: float  # recomputed vs app-calibrated reading, largest difference


def _seconds(t_ns: np.ndarray, t0_ns: int) -> np.ndarray:
    return (np.asarray(t_ns, dtype=np.int64) - t0_ns).astype(float) * 1e-9


def interpolate_orientation(ori_t: np.ndarray, ori_q: np.ndarray, t: np.ndarray, t0: int):
    """Orientation (w x y z) at times t by spherical interpolation; clamped at the ends.
    Returns (quaternions wxyz, number of times outside the stream's span)."""
    ts = _seconds(ori_t, t0)
    # Duplicate timestamps break Slerp; keep the last sample of each.
    keep = np.concatenate([np.diff(ts) > 0, [True]])
    ts, q = ts[keep], ori_q[keep]
    rot = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    tq = _seconds(t, t0)
    outside = int(np.sum((tq < ts[0]) | (tq > ts[-1])))
    tq = np.clip(tq, ts[0], ts[-1])
    if ts.size == 1:
        out = Rotation.concatenate([rot] * tq.size)
    else:
        out = Slerp(ts, rot)(tq)
    return out.as_quat()[:, [3, 0, 1, 2]], outside


def interpolate_position(pos_t: np.ndarray, pos: np.ndarray, t: np.ndarray, t0: int) -> np.ndarray:
    """Reference-point position at times t, linear between rows, held at the ends."""
    ts, tq = _seconds(pos_t, t0), _seconds(t, t0)
    return np.stack([np.interp(tq, ts, pos[:, k]) for k in range(3)], axis=1)


def swipe_observations(
    s: Swipe, baseline_ut: np.ndarray, swipe_id: int, baseline_drift_ut: float = 0.0
) -> tuple[Observations, AlignReport]:
    """Turn one parsed swipe into world-frame, baseline-subtracted observations.

    baseline_ut: the session's background field in the world frame (µT), from the start
    baseline. baseline_drift_ut: how far the end baseline moved from it (µT, 0 if unknown);
    it widens the per-swipe field-offset error, because the background is not known better.
    """
    t0 = int(s.mag_t[0])
    cal = s.calibration
    b_phone = apply_iron_correction(s.mag_raw, cal.hard_iron_ut, cal.soft_iron)
    mismatch = float(np.max(np.abs(b_phone - s.mag_cal))) if len(b_phone) else 0.0

    q, outside = interpolate_orientation(s.ori_t, s.ori_q, s.mag_t, t0)
    ref = interpolate_position(s.pos_t, s.pos, s.mag_t, t0)
    pos, b_world = to_world(b_phone, q, ref, cal.sensor_offset_m)
    b_world = b_world - np.asarray(baseline_ut, dtype=float)

    keep = ~s.mag_saturated
    m = int(keep.sum())
    noise = float(np.sqrt(np.mean(cal.noise_ut**2)))  # per-axis RMS of the measured noise
    e = s.errors
    # Shared field offset per swipe: leftover hard iron and baseline error. Before Phase 5
    # calibration the measured noise and the baseline drift are the best evidence of its size.
    bias_b = float(np.hypot(noise, baseline_drift_ut / 2))
    obs = Observations(
        pos=pos[keep],
        b=b_world[keep],
        swipe=np.full(m, swipe_id),
        sigma_b=np.full(m, noise),
        sigma_pos=np.full(m, e.position_jitter_m),
        sigma_rot=np.full(m, e.orientation_jitter_rad),
        bias_pos=np.full(m, e.position_offset_m),
        bias_rot=np.full(m, e.orientation_offset_rad),
        bias_b=np.full(m, bias_b),
        count=np.ones(m),
    )
    report = AlignReport(
        readings=len(keep),
        kept=m,
        saturated=int((~keep).sum()),
        outside_orientation=outside,
        max_iron_mismatch_ut=mismatch,
    )
    return obs, report


def session_baseline(swipes: list[Swipe]) -> tuple[np.ndarray, float]:
    """Background field (world frame, µT) from swipe 0's start baseline, and the drift to the
    end baseline if a later swipe carries one (else 0)."""
    first = next((s for s in swipes if s.baseline_start_ut is not None), None)
    if first is None:
        raise ValueError("no swipe carries the session's start baseline")
    start = first.baseline_start_ut
    end = next((s.baseline_end_ut for s in swipes if s.baseline_end_ut is not None), None)
    drift = float(np.linalg.norm(end - start)) if end is not None else 0.0
    return start, drift
