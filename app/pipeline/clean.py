"""Intelligent pre-processing for human-recorded phone swipes.

Physics and human motion rules:
1. Lift-Off & Idle Tails: After completing a physical pass, users tilt the phone (> 20°-25°)
   to look at the screen and take several seconds before tapping 'Stop'. We automatically detect
   lift-off and trim trailing dead-air samples. In ruler mode, the interpolated arrival at the
   end mark is mapped to the moment the slide physically ended.
2. Elevation Sanity & Dipole Decay: Outside a magnet, Laplace's equation governs field fall-off.
   A field |B| > 120 µT cannot physically originate from a distance z > 8-10 cm for any handheld
   permanent magnet. When an impossible box height is entered (e.g. 200 mm instead of 20 mm),
   it is scaled to its physical order of magnitude.
3. Multi-Pass / Reversal Segmentation: When users slide back and forth over a magnet,
   multiple major peaks appear separated by a deep valley. We segment the swipe into
   coherent directional passes, reversing the ruler interpolation direction on return passes
   so peaks map to the same physical location.
"""

from __future__ import annotations

import copy

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

from .swipe_format import Swipe


def sanitize_elevation(s: Swipe, max_physical_elevation_m: float = 0.08) -> Swipe:
    """Detect and auto-correct physically impossible box heights.

    For example, entering 200 mm (0.2 m) when |B| > 120 µT violates dipole decay physics
    for handheld magnets; we scale it down by 10x to 20 mm (0.02 m).
    """
    s = copy.deepcopy(s)
    if s.planned_start_m is None or s.planned_end_m is None:
        return s

    z = float(s.planned_start_m[2])
    cal_norm = np.linalg.norm(s.mag_cal, axis=1) if len(s.mag_cal) else np.array([0.0])
    b_max = float(np.max(cal_norm))

    if b_max > 120.0 and z > max_physical_elevation_m:
        scale = 10.0 if z >= 0.15 else (z / 0.025)
        new_z = float(z / scale)
        s.planned_start_m[2] = new_z
        s.planned_end_m[2] = new_z
        if s.pos is not None and len(s.pos):
            s.pos[:, 2] = new_z

    return s


def trim_idle_tail(s: Swipe, baseline_ut: np.ndarray | None = None) -> Swipe:
    """Trim dead air, lift-off tilt, and idle holding at the end of a swipe."""
    if len(s.mag_t) < 30:
        return s

    s = copy.deepcopy(s)
    t_mag = s.mag_t
    t_s = (t_mag - t_mag[0]).astype(float) * 1e-9

    # Phone tilt relative to world z-axis
    rot = Rotation.from_quat(s.ori_q[:, [1, 2, 3, 0]])
    z_w = rot.apply([0, 0, 1])
    tilts_ori = np.arccos(np.clip(z_w[:, 2], -1, 1))
    t_ori_s = (s.ori_t - t_mag[0]).astype(float) * 1e-9
    tilts = np.interp(t_s, t_ori_s, tilts_ori)

    # Signal above baseline
    base_norm = float(np.linalg.norm(baseline_ut)) if baseline_ut is not None else 45.0
    cal_norm = np.linalg.norm(s.mag_cal, axis=1)
    sig = np.abs(cal_norm - base_norm)
    noise = float(np.sqrt(np.mean(s.calibration.noise_ut**2)))

    has_sig = sig > max(3 * noise, 15.0)
    if not np.any(has_sig):
        return s

    last_sig_idx = int(np.where(has_sig)[0][-1])
    last_sig_t = t_s[last_sig_idx]

    # Lift-off tilt (> 20 deg) after the last significant magnetic signal
    lift_candidates = np.where((t_s >= last_sig_t) & (tilts > np.deg2rad(20)))[0]
    if len(lift_candidates):
        trim_idx = int(lift_candidates[0])
    else:
        # If no tilt, but sitting idle at baseline for > 3 seconds
        if t_s[-1] - last_sig_t > 3.0:
            margin = int(1.5 * s.calibration.sampling_rate_hz)
            trim_idx = min(last_sig_idx + margin, len(t_s) - 1)
        else:
            return s

    # Ensure we keep at least 20 samples
    if trim_idx < 20:
        return s

    trim_t_ns = int(t_mag[trim_idx])

    # Slice streams
    m_mask = s.mag_t <= trim_t_ns
    s.mag_t = s.mag_t[m_mask]
    s.mag_raw = s.mag_raw[m_mask]
    s.mag_cal = s.mag_cal[m_mask]
    s.mag_saturated = s.mag_saturated[m_mask]

    o_mask = s.ori_t <= trim_t_ns
    if np.any(o_mask):
        s.ori_t = s.ori_t[o_mask]
        s.ori_q = s.ori_q[o_mask]

    if s.accel_t is not None and s.accel is not None:
        a_mask = s.accel_t <= trim_t_ns
        s.accel_t = s.accel_t[a_mask]
        s.accel = s.accel[a_mask]

    if s.gyro_t is not None and s.gyro is not None:
        g_mask = s.gyro_t <= trim_t_ns
        s.gyro_t = s.gyro_t[g_mask]
        s.gyro = s.gyro[g_mask]

    # For ruler mode: adjust arrival at end mark
    if s.tracking_mode == "ruler" and len(s.pos_t) >= 2:
        s.pos_t[-2:] = trim_t_ns

    return s


def segment_reversals(s: Swipe) -> list[Swipe]:
    """Segment a back-and-forth multi-pass swipe into coherent directional sub-swipes."""
    if s.tracking_mode != "ruler" or s.planned_start_m is None or s.planned_end_m is None:
        return [s]

    if len(s.mag_cal) < 60:
        return [s]

    cal_norm = np.linalg.norm(s.mag_cal, axis=1)
    rate = max(int(s.calibration.sampling_rate_hz), 10)
    window = max(int(0.25 * rate), 3)
    smoothed = np.convolve(cal_norm, np.ones(window) / window, mode="same")

    p_height = 0.5 * float(np.max(smoothed))
    peaks, _ = find_peaks(smoothed, height=p_height, distance=int(3.0 * rate))

    if len(peaks) < 2:
        return [s]

    # Look for a deep valley between the first and last major peaks
    valley_idx = peaks[0] + int(np.argmin(smoothed[peaks[0] : peaks[-1]]))
    valley_val = smoothed[valley_idx]
    peak_min = min(smoothed[peaks[0]], smoothed[peaks[-1]])

    # Valley must drop below 65% of the lower peak to indicate a true reversal
    if valley_val > 0.65 * peak_min:
        return [s]

    split_t_ns = int(s.mag_t[valley_idx])

    # Pass 1: start -> end
    s1 = copy.deepcopy(s)
    m1 = s.mag_t <= split_t_ns
    s1.mag_t = s.mag_t[m1]
    s1.mag_raw = s.mag_raw[m1]
    s1.mag_cal = s.mag_cal[m1]
    s1.mag_saturated = s.mag_saturated[m1]

    o1 = s.ori_t <= split_t_ns
    s1.ori_t = s.ori_t[o1]
    s1.ori_q = s.ori_q[o1]

    s1.pos_t = np.array([s.mag_t[0], split_t_ns], dtype=np.int64)
    s1.pos = np.array([s.planned_start_m, s.planned_end_m], dtype=float)
    s1.pos_state = ["ruler_endpoint", "ruler_endpoint"]
    s1.pos_confidence = np.array([1.0, 1.0])

    # Pass 2: end -> start (direction reversed so positions align!)
    s2 = copy.deepcopy(s)
    s2.swipe_index = s.swipe_index + 1
    m2 = s.mag_t > split_t_ns
    s2.mag_t = s.mag_t[m2]
    s2.mag_raw = s.mag_raw[m2]
    s2.mag_cal = s.mag_cal[m2]
    s2.mag_saturated = s.mag_saturated[m2]

    o2 = s.ori_t > split_t_ns
    s2.ori_t = s.ori_t[o2]
    s2.ori_q = s.ori_q[o2]

    s2.planned_start_m = copy.deepcopy(s.planned_end_m)
    s2.planned_end_m = copy.deepcopy(s.planned_start_m)
    s2.pos_t = np.array([s2.mag_t[0], s2.mag_t[-1]], dtype=np.int64)
    s2.pos = np.array([s2.planned_start_m, s2.planned_end_m], dtype=float)
    s2.pos_state = ["ruler_endpoint", "ruler_endpoint"]
    s2.pos_confidence = np.array([1.0, 1.0])

    return [s1, s2]


def clean_swipe(s: Swipe, baseline_ut: np.ndarray | None = None) -> list[Swipe]:
    """Apply physics-informed cleaning to an uploaded swipe."""
    s_clean = sanitize_elevation(s)
    s_clean = trim_idle_tail(s_clean, baseline_ut)
    return segment_reversals(s_clean)
