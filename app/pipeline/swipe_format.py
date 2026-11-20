"""Swipe format v1: load and check one swipe file from the app (SRS 4.4).

The full field-by-field definition lives in the app repo, `docs/swipe-format-v1.md`; the app's
TypeScript builder writes it and `tests/fixtures/swipe-v1-synthetic.json` is a file it produced.
This module only parses and validates. Aligning the streams and turning them into
`Observations` happens in the Phase 3 alignment step.

Units: µT, m, ns, rad. World frame z up. Quaternions w x y z, phone -> world.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_NAME = "magnet-eye-swipe"
FORMAT_VERSION = 1
TRACKING_MODES = {"arcore", "ruler", "grid"}
POSITION_STATES = {"tracking", "limited", "lost", "ruler_endpoint", "grid_mark"}
MAG_COLUMNS = [
    "t_ns",
    "raw_x_ut",
    "raw_y_ut",
    "raw_z_ut",
    "cal_x_ut",
    "cal_y_ut",
    "cal_z_ut",
    "saturated",
]
ORIENTATION_COLUMNS = ["t_ns", "qw", "qx", "qy", "qz"]
POSITION_COLUMNS = ["t_ns", "px_m", "py_m", "pz_m", "state", "confidence"]
VECTOR_COLUMNS = ["t_ns", "x", "y", "z"]


class SwipeFormatError(ValueError):
    """The file is not a usable swipe; `problems` lists every field that failed."""

    def __init__(self, problems: list[str]):
        super().__init__("invalid swipe: " + "; ".join(problems))
        self.problems = problems


@dataclass
class Calibration:
    source: str
    hard_iron_ut: np.ndarray  # (3,)
    soft_iron: np.ndarray  # (3, 3)
    noise_ut: np.ndarray  # (3,)
    range_ut: float
    saturation_threshold_ut: float
    sampling_rate_hz: float
    sensor_offset_m: np.ndarray  # (3,), phone frame, from the reference point
    sensor_offset_known: bool


@dataclass
class ErrorSizes:
    position_jitter_m: float
    position_offset_m: float
    orientation_jitter_rad: float
    orientation_offset_rad: float


@dataclass
class Swipe:
    session_id: str
    swipe_index: int
    tracking_mode: str
    app_version: str
    device_model: str
    clock_origin_ns: int
    calibration: Calibration
    errors: ErrorSizes
    table_z_m: float
    baseline_start_ut: np.ndarray | None
    baseline_end_ut: np.ndarray | None
    planned_start_m: np.ndarray | None
    planned_end_m: np.ndarray | None
    mag_t: np.ndarray  # (N,) int64 ns, relative to clock_origin_ns
    mag_raw: np.ndarray  # (N, 3) µT, phone frame
    mag_cal: np.ndarray  # (N, 3) µT, phone frame, hard/soft iron removed
    mag_saturated: np.ndarray  # (N,) bool
    ori_t: np.ndarray  # (M,) int64 ns
    ori_q: np.ndarray  # (M, 4) w x y z, phone -> world
    ori_source: str
    pos_t: np.ndarray  # (K,) int64 ns
    pos: np.ndarray  # (K, 3) m, reference point, world frame
    pos_state: list[str]
    pos_confidence: np.ndarray  # (K,)
    accel_t: np.ndarray | None = None
    accel: np.ndarray | None = None
    gyro_t: np.ndarray | None = None
    gyro: np.ndarray | None = None


def _get(d: Any, path: str, problems: list[str]) -> Any:
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            problems.append(f"{path}: missing")
            return None
        cur = cur[k]
    return cur


def _vec(v: Any, path: str, problems: list[str], shape=(3,), allow_none=False) -> np.ndarray | None:
    if v is None:
        if not allow_none:
            problems.append(f"{path}: missing")
        return None
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        problems.append(f"{path}: not numeric")
        return None
    if a.shape != shape or not np.all(np.isfinite(a)):
        problems.append(f"{path}: expected finite array of shape {shape}")
        return None
    return a


def _num(v: Any, path: str, problems: list[str], positive=False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
        problems.append(f"{path}: expected a number")
        return float("nan")
    if positive and v <= 0:
        problems.append(f"{path}: must be > 0")
    return float(v)


def _table(stream: Any, name: str, columns: list[str], problems: list[str], min_rows: int):
    """Numeric stream -> (t int64, values float). Returns (None, None) on failure."""
    if not isinstance(stream, dict):
        problems.append(f"streams.{name}: missing")
        return None, None
    if stream.get("columns") != columns:
        problems.append(f"streams.{name}.columns: expected {columns}")
        return None, None
    rows = stream.get("rows")
    if not isinstance(rows, list) or len(rows) < min_rows:
        problems.append(f"streams.{name}.rows: need at least {min_rows}")
        return None, None
    try:
        a = np.asarray(rows, dtype=float)
    except (TypeError, ValueError):
        problems.append(f"streams.{name}.rows: not numeric")
        return None, None
    if a.ndim != 2 or a.shape[1] != len(columns) or not np.all(np.isfinite(a)):
        problems.append(f"streams.{name}.rows: expected finite rows of {len(columns)} values")
        return None, None
    t = a[:, 0].astype(np.int64)
    if np.any(np.diff(t) < 0):
        problems.append(f"streams.{name}: timestamps not in order")
    return t, a[:, 1:]


def parse_swipe(d: dict) -> Swipe:
    """Validate a decoded swipe and return it as arrays. Raises SwipeFormatError listing every
    problem found, not just the first."""
    p: list[str] = []
    if not isinstance(d, dict):
        raise SwipeFormatError(["not a JSON object"])
    if d.get("format") != FORMAT_NAME:
        p.append(f"format: expected {FORMAT_NAME!r}")
    if d.get("format_version") != FORMAT_VERSION:
        raise SwipeFormatError(
            p + [f"format_version: expected {FORMAT_VERSION}, got {d.get('format_version')!r}"]
        )

    session_id = _get(d, "session_id", p)
    index = _get(d, "swipe_index", p)
    if index is not None and (not isinstance(index, int) or index < 0):
        p.append("swipe_index: expected an integer >= 0")
    mode = _get(d, "tracking_mode", p)
    if mode is not None and mode not in TRACKING_MODES:
        p.append(f"tracking_mode: expected one of {sorted(TRACKING_MODES)}")
    origin = _get(d, "clock.origin_ns", p)
    if origin is not None and not (isinstance(origin, str) and origin.isdigit()):
        p.append("clock.origin_ns: expected a decimal string")

    c = _get(d, "calibration", p) or {}
    cal = Calibration(
        source=str(c.get("source")),
        hard_iron_ut=_vec(c.get("hard_iron_ut"), "calibration.hard_iron_ut", p),
        soft_iron=_vec(c.get("soft_iron"), "calibration.soft_iron", p, shape=(3, 3)),
        noise_ut=_vec(c.get("noise_ut"), "calibration.noise_ut", p),
        range_ut=_num(c.get("range_ut"), "calibration.range_ut", p, positive=True),
        saturation_threshold_ut=_num(
            c.get("saturation_threshold_ut"),
            "calibration.saturation_threshold_ut",
            p,
            positive=True,
        ),
        sampling_rate_hz=_num(
            c.get("sampling_rate_hz"), "calibration.sampling_rate_hz", p, positive=True
        ),
        sensor_offset_m=_vec(c.get("sensor_offset_m"), "calibration.sensor_offset_m", p),
        sensor_offset_known=c.get("sensor_offset_source") == "measured",
    )

    e = _get(d, "error_sizes", p) or {}
    errors = ErrorSizes(
        *(
            _num(e.get(k), f"error_sizes.{k}", p)
            for k in (
                "position_jitter_m",
                "position_offset_m",
                "orientation_jitter_rad",
                "orientation_offset_rad",
            )
        )
    )

    g = _get(d, "session_geometry", p) or {}
    table_z = _num(g.get("table_z_m"), "session_geometry.table_z_m", p)
    base0 = _vec(
        g.get("baseline_start_ut"), "session_geometry.baseline_start_ut", p, allow_none=True
    )
    base1 = _vec(g.get("baseline_end_ut"), "session_geometry.baseline_end_ut", p, allow_none=True)
    if index == 0 and base0 is None:
        p.append("session_geometry.baseline_start_ut: required on swipe 0")

    line = d.get("planned_line")
    start = end = None
    if isinstance(line, dict):
        start = _vec(line.get("start_m"), "planned_line.start_m", p)
        end = _vec(line.get("end_m"), "planned_line.end_m", p)
    elif mode in ("ruler", "grid"):
        p.append("planned_line: required in ruler and grid modes")

    s = _get(d, "streams", p) or {}
    mag_t, mag = _table(s.get("mag"), "mag", MAG_COLUMNS, p, 2)
    ori_t, ori = _table(s.get("orientation"), "orientation", ORIENTATION_COLUMNS, p, 2)
    if ori is not None and np.any(np.abs(np.linalg.norm(ori, axis=1) - 1) > 1e-3):
        p.append("streams.orientation: quaternions not unit length")

    pos_t = pos = conf = None
    states: list[str] = []
    ps = s.get("position")
    if not isinstance(ps, dict) or ps.get("columns") != POSITION_COLUMNS:
        p.append(f"streams.position.columns: expected {POSITION_COLUMNS}")
    else:
        rows = ps.get("rows") or []
        if len(rows) < 2:
            p.append("streams.position.rows: need at least 2")
        else:
            try:
                pos_t = np.array([r[0] for r in rows], dtype=np.int64)
                pos = np.array([r[1:4] for r in rows], dtype=float)
                states = [str(r[4]) for r in rows]
                conf = np.array([r[5] for r in rows], dtype=float)
            except (TypeError, ValueError, IndexError):
                p.append("streams.position.rows: expected [t, x, y, z, state, confidence]")
            else:
                if bad := sorted(set(states) - POSITION_STATES):
                    p.append(f"streams.position.state: unknown {bad}")
                if np.any(np.diff(pos_t) < 0):
                    p.append("streams.position: timestamps not in order")
                if not np.all(np.isfinite(pos)):
                    p.append("streams.position: non-finite positions")

    acc_t, acc = (None, None)
    gyr_t, gyr = (None, None)
    if "accel" in s:
        acc_t, acc = _table(s["accel"], "accel", VECTOR_COLUMNS, p, 0)
    if "gyro" in s:
        gyr_t, gyr = _table(s["gyro"], "gyro", VECTOR_COLUMNS, p, 0)

    if p:
        raise SwipeFormatError(p)
    return Swipe(
        session_id=session_id,
        swipe_index=index,
        tracking_mode=mode,
        app_version=str(d.get("app_version")),
        device_model=str((d.get("device") or {}).get("model")),
        clock_origin_ns=int(origin),
        calibration=cal,
        errors=errors,
        table_z_m=table_z,
        baseline_start_ut=base0,
        baseline_end_ut=base1,
        planned_start_m=start,
        planned_end_m=end,
        mag_t=mag_t,
        mag_raw=mag[:, 0:3],
        mag_cal=mag[:, 3:6],
        mag_saturated=mag[:, 6] > 0.5,
        ori_t=ori_t,
        ori_q=ori,
        ori_source=str(s["orientation"].get("source")),
        pos_t=pos_t,
        pos=pos,
        pos_state=states,
        pos_confidence=conf,
        accel_t=acc_t,
        accel=acc,
        gyro_t=gyr_t,
        gyro=gyr,
    )


def load_swipe(path: str | Path) -> Swipe:
    return parse_swipe(json.loads(Path(path).read_text()))
