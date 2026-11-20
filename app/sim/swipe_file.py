"""Simulated ruler-mode swipes written as swipe format v1 files, exactly what the app uploads.

Used to run whole sessions through the API (Phase 4 "done when") without a phone. The phone
lies flat on a raised surface over the magnet with its top along world +y, slides at constant
speed along the planned line, and records at 100 Hz.

Units as in the format: µT, m, int64 ns, rad. World frame z up, quaternions w x y z.
"""

from __future__ import annotations

import numpy as np

from app.pipeline.swipe_format import (
    FORMAT_NAME,
    FORMAT_VERSION,
    MAG_COLUMNS,
    ORIENTATION_COLUMNS,
    POSITION_COLUMNS,
)

from .magnets import TestMagnet
from .phone import EARTH_UT, Phone

RATE_HZ = 100.0
HARD_IRON_UT = np.array([-70.0, -25.0, -140.0])  # the phone's own field, removed by calibration


def ruler_lines(half: float, height: float, spacing: float, n: int) -> list[tuple[list, list]]:
    """n parallel lines along x, centred on the origin, `spacing` apart, at `height` (m)."""
    ys = (np.arange(n) - (n - 1) / 2) * spacing
    return [([-half, float(y), height], [half, float(y), height]) for y in ys]


def swipe_file(
    magnet: TestMagnet,
    start: list[float],
    end: list[float],
    index: int,
    phone: Phone,
    rng: np.random.Generator,
    session_id: str = "sim-session",
    speed_mps: float = 0.05,
) -> dict:
    """One swipe of the phone's reference point from start to end, as a v1 dict."""
    a, b = np.asarray(start, float), np.asarray(end, float)
    n = max(int(np.linalg.norm(b - a) / speed_mps * RATE_HZ), 20)
    t = (5_000_000_000 + index * 60_000_000_000 + np.arange(n) * int(1e9 / RATE_HZ)).astype(
        np.int64
    )
    path = a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)

    # Phone flat, top along +y: phone frame = world frame. Small orientation jitter only.
    jit = phone.rot_jitter_deg * np.pi / 180 * rng.standard_normal((n, 3))
    half = np.linalg.norm(jit, axis=1)
    axis = jit / np.maximum(half, 1e-12)[:, None]
    q = np.column_stack([np.cos(half / 2), axis * np.sin(half / 2)[:, None]])

    b_world = magnet.field_ut(path) + EARTH_UT
    raw = b_world + HARD_IRON_UT + phone.noise_ut * rng.standard_normal(b_world.shape)
    raw = np.round(raw / phone.lsb_ut) * phone.lsb_ut
    sat = np.any(np.abs(raw) >= 0.98 * phone.range_ut, axis=1)
    raw = np.clip(raw, -phone.range_ut, phone.range_ut)
    cal = raw - HARD_IRON_UT

    mag_rows = [[int(t[i]), *raw[i].tolist(), *cal[i].tolist(), int(sat[i])] for i in range(n)]
    ori_rows = [[int(t[i]), *q[i].tolist()] for i in range(n)]
    pos_rows = [
        [int(t[0]), *a.tolist(), "ruler_endpoint", 1.0],
        [int(t[-1]), *b.tolist(), "ruler_endpoint", 1.0],
    ]
    noise = [phone.noise_ut] * 3
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "app_version": "sim",
        "api_version": "v1",
        "calibration_schema_version": 1,
        "session_id": session_id,
        "swipe_index": index,
        "recorded_at": "2026-11-20T12:00:00Z",
        "device": {"manufacturer": "Sim", "model": "Simulated phone", "android_sdk": 35},
        "tracking_mode": "ruler",
        "reference_point": "top_edge_centre_back",
        "clock": {"source": "elapsedRealtimeNanos", "origin_ns": "1000"},
        "calibration": {
            "source": "android_estimate",
            "profile_id": None,
            "hard_iron_ut": HARD_IRON_UT.tolist(),
            "soft_iron": np.eye(3).tolist(),
            "noise_ut": noise,
            "range_ut": phone.range_ut,
            "saturation_threshold_ut": 0.98 * phone.range_ut,
            "sensor_offset_m": [0.0, 0.0, 0.0],
            "sensor_offset_source": "unknown",
            "sampling_rate_hz": RATE_HZ,
        },
        "session_geometry": {
            "table_z_m": 0.0,
            "baseline_start_ut": EARTH_UT.tolist() if index == 0 else None,
            "baseline_end_ut": None,
            "yaw_alignment_rad": 0.0,
        },
        "error_sizes": {
            "position_jitter_m": 0.003,
            "position_offset_m": 0.002,
            "orientation_jitter_rad": 0.005,
            "orientation_offset_rad": 0.02,
            "source": "mode_default",
        },
        "planned_line": {"start_m": a.tolist(), "end_m": b.tolist()},
        "streams": {
            "mag": {"sensor": "uncalibrated", "columns": MAG_COLUMNS, "rows": mag_rows},
            "orientation": {
                "source": "game_rotation_vector",
                "columns": ORIENTATION_COLUMNS,
                "rows": ori_rows,
            },
            "position": {"columns": POSITION_COLUMNS, "rows": pos_rows},
        },
        "stats": {},
    }
