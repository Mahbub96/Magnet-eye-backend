"""Simulated phone: turns a planned swipe into what the app would record and upload.

The output deliberately goes through the same steps as real data: raw phone-frame
readings, an orientation quaternion with errors, a tracked reference position with errors,
and a sensor that is not at the tracked point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from app.pipeline.frames import to_world
from app.pipeline.observations import Observations

from .magnets import TestMagnet


@dataclass
class Phone:
    noise_ut: float = 0.8  # per axis
    range_ut: float = 4900.0  # per axis
    lsb_ut: float = 0.15
    rate_hz: float = 100.0
    hard_iron_residual_ut: float = 2.0
    baseline_err_ut: float = 1.0
    pos_jitter_m: float = 0.0015
    pos_bias_m: float = 0.002
    rot_jitter_deg: float = 0.3
    rot_bias_deg: float = 1.0
    sensor_offset_m: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.06, 0.0]))
    speed_mps: float = 0.08
    wobble_m: float = 0.002
    tilt_deg: float = 3.0


NOISE = {
    "ideal": Phone(
        noise_ut=0.3,
        hard_iron_residual_ut=0.0,
        baseline_err_ut=0.0,
        pos_jitter_m=0.0005,
        pos_bias_m=0.0,
        rot_jitter_deg=0.05,
        rot_bias_deg=0.0,
    ),
    "realistic": Phone(),
    "poor": Phone(
        noise_ut=2.0,
        hard_iron_residual_ut=5.0,
        baseline_err_ut=3.0,
        pos_jitter_m=0.003,
        pos_bias_m=0.005,
        rot_jitter_deg=0.6,
        rot_bias_deg=2.0,
    ),
}

EARTH_UT = np.array([18.0, 0.0, -42.0])


def _smooth_noise(rng, n, scale, corr=25):
    """Low-pass random walk-ish noise of the given RMS, like a hand's slow wobble."""
    w = rng.standard_normal(n + corr)
    k = np.ones(corr) / corr
    s = np.convolve(w, k, mode="valid")[:n]
    return scale * s / (s.std() + 1e-12)


def record_swipe(
    magnet: TestMagnet,
    start: np.ndarray,
    end: np.ndarray,
    phone: Phone,
    rng: np.random.Generator,
    swipe_id: int,
) -> Observations:
    """Simulate one swipe of the SENSOR from start to end and return the pipeline input
    (world frame, baseline subtracted, saturated readings dropped)."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(end - start))
    n = max(int(length / phone.speed_mps * phone.rate_hz), 10)
    s = np.linspace(0.0, 1.0, n)
    path = start + s[:, None] * (end - start)
    path += np.stack([_smooth_noise(rng, n, phone.wobble_m) for _ in range(3)], axis=1)

    # True phone orientation: a fixed yaw for the swipe plus small slow tilts.
    yaw = rng.uniform(-15, 15)
    eul = np.stack(
        [
            np.full(n, yaw) + _smooth_noise(rng, n, 1.0),
            _smooth_noise(rng, n, phone.tilt_deg),
            _smooth_noise(rng, n, phone.tilt_deg),
        ],
        axis=1,
    )
    r_true = Rotation.from_euler("zyx", eul, degrees=True)
    ref_true = path - r_true.apply(np.broadcast_to(phone.sensor_offset_m, path.shape))

    # What the magnetometer reports, in the phone frame.
    b_world_true = magnet.field_ut(path) + EARTH_UT
    b_phone = r_true.inv().apply(b_world_true)
    hi = rng.standard_normal(3)
    b_phone += phone.hard_iron_residual_ut * hi / np.linalg.norm(hi)
    b_phone += phone.noise_ut * rng.standard_normal(b_phone.shape)
    b_phone = np.round(b_phone / phone.lsb_ut) * phone.lsb_ut
    saturated = np.any(np.abs(b_phone) >= 0.98 * phone.range_ut, axis=1)
    b_phone = np.clip(b_phone, -phone.range_ut, phone.range_ut)

    # What tracking and the game rotation vector report.
    bias_rot = Rotation.from_rotvec(np.deg2rad(phone.rot_bias_deg) * _unit(rng))
    jit = Rotation.from_rotvec(np.deg2rad(phone.rot_jitter_deg) * rng.standard_normal((n, 3)))
    r_meas = bias_rot * r_true * jit
    q = r_meas.as_quat()[:, [3, 0, 1, 2]]  # wxyz, like Android
    ref_meas = (
        ref_true + phone.pos_bias_m * _unit(rng) + phone.pos_jitter_m * rng.standard_normal((n, 3))
    )

    # Pipeline steps the backend runs on real uploads: frames, then baseline in world frame.
    pos, b_world = to_world(b_phone, q, ref_meas, phone.sensor_offset_m)
    baseline = EARTH_UT + phone.baseline_err_ut * rng.standard_normal(3) / np.sqrt(3)
    b_world -= baseline
    keep = ~saturated
    m = int(keep.sum())
    return Observations(
        pos=pos[keep],
        b=b_world[keep],
        swipe=np.full(m, swipe_id),
        sigma_b=np.full(m, phone.noise_ut),
        sigma_pos=np.full(m, phone.pos_jitter_m),
        sigma_rot=np.full(m, np.deg2rad(phone.rot_jitter_deg)),
        bias_pos=np.full(m, phone.pos_bias_m),
        bias_rot=np.full(m, np.deg2rad(phone.rot_bias_deg)),
        bias_b=np.full(
            m, max(np.hypot(phone.hard_iron_residual_ut, phone.baseline_err_ut), phone.noise_ut)
        ),
        count=np.ones(m),
    )


def _unit(rng):
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)
