"""Rotate phone-frame readings into the fixed world frame and place the sensor.

Rule (Challenges guide 3.2, 8.4): every reading is rotated into the world frame before any
maths. Background subtraction happens AFTER this step, because the Earth's field is only
constant in the world frame.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def to_world(
    b_phone: np.ndarray,
    quat_wxyz: np.ndarray,
    ref_pos: np.ndarray,
    sensor_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sensor positions, B in world frame).

    b_phone: (N, 3) calibrated readings in the phone frame, µT.
    quat_wxyz: (N, 4) phone-to-world orientation, scalar first (Android order).
    ref_pos: (N, 3) tracked position of the phone's reference point, metres.
    sensor_offset: (3,) magnetometer position in the phone frame relative to the
        tracked reference point, metres (measured per phone, never hard-coded).
    """
    q = np.asarray(quat_wxyz, dtype=float)
    rot = Rotation.from_quat(q[:, [1, 2, 3, 0]])  # SciPy wants scalar last
    b_world = rot.apply(np.asarray(b_phone, dtype=float))
    pos = np.asarray(ref_pos, dtype=float) + rot.apply(
        np.broadcast_to(sensor_offset, q[:, :3].shape)
    )
    return pos, b_world


def apply_iron_correction(
    b_raw: np.ndarray, hard_iron: np.ndarray, soft_iron: np.ndarray
) -> np.ndarray:
    """Calibrated = soft_iron @ (raw - hard_iron), all in the phone frame."""
    return (np.asarray(b_raw, dtype=float) - hard_iron) @ np.asarray(soft_iron).T
