"""Tests for physics-informed swipe cleaning (tail trim, elevation sanity, multi-pass reversal)."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from app.pipeline.clean import clean_swipe, sanitize_elevation, segment_reversals, trim_idle_tail
from app.pipeline.swipe_format import parse_swipe

FIXTURE = Path(__file__).parent / "fixtures" / "swipe-v1-synthetic.json"


@pytest.fixture
def synthetic_swipe():
    raw = json.loads(FIXTURE.read_text())
    return parse_swipe(raw)


def test_synthetic_swipe_unaffected(synthetic_swipe):
    """Clean synthetic swipes without excessive tilt or impossible elevation pass through intact."""
    cleaned = clean_swipe(synthetic_swipe)
    assert len(cleaned) == 1
    assert len(cleaned[0].mag_t) == len(synthetic_swipe.mag_t)
    assert cleaned[0].planned_start_m[2] == synthetic_swipe.planned_start_m[2]


def test_sanitize_elevation_auto_corrects_typo(synthetic_swipe):
    """A 200 mm height (0.2 m) with high magnetic field is corrected by 10x to 20 mm."""
    s = copy.deepcopy(synthetic_swipe)
    s.planned_start_m = np.array([-0.1, 0.0, 0.20])
    s.planned_end_m = np.array([0.1, 0.0, 0.20])
    s.pos = np.array([[-0.1, 0.0, 0.20], [0.1, 0.0, 0.20]])
    s.mag_cal = np.full((len(s.mag_t), 3), 100.0)  # norm = 173 uT > 120 uT

    s_fixed = sanitize_elevation(s)
    assert s_fixed.planned_start_m[2] == pytest.approx(0.02)
    assert s_fixed.planned_end_m[2] == pytest.approx(0.02)
    assert s_fixed.pos[0, 2] == pytest.approx(0.02)


def test_trim_idle_tail_removes_dead_air(synthetic_swipe):
    """When a phone tilts > 20° after signal ends, the dead air is trimmed."""
    s = copy.deepcopy(synthetic_swipe)
    # Add a magnetic signal in the middle of synthetic swipe
    s.mag_cal[100:200] += 200.0

    # Add 100 trailing idle samples where tilt is 45°
    n_extra = 100
    t_end = s.mag_t[-1]
    extra_t = t_end + np.arange(1, n_extra + 1, dtype=np.int64) * 10_000_000  # 10ms
    s.mag_t = np.concatenate([s.mag_t, extra_t])
    s.mag_raw = np.concatenate([s.mag_raw, np.zeros((n_extra, 3))])
    s.mag_cal = np.concatenate([s.mag_cal, np.zeros((n_extra, 3))])
    s.mag_saturated = np.concatenate([s.mag_saturated, np.zeros(n_extra, dtype=bool)])

    # Extra orientations with 45° tilt
    extra_ori_t = s.ori_t[-1] + np.arange(1, n_extra + 1, dtype=np.int64) * 10_000_000
    # Tilt 45° around y-axis: [cos(22.5°), 0, sin(22.5°), 0]
    q_tilted = np.tile([np.cos(np.pi / 8), 0.0, np.sin(np.pi / 8), 0.0], (n_extra, 1))
    s.ori_t = np.concatenate([s.ori_t, extra_ori_t])
    s.ori_q = np.concatenate([s.ori_q, q_tilted])

    s_trimmed = trim_idle_tail(s, baseline_ut=synthetic_swipe.baseline_start_ut)
    assert len(s_trimmed.mag_t) <= len(synthetic_swipe.mag_t) + 1


def test_segment_reversals_splits_back_and_forth(synthetic_swipe):
    """A back-and-forth swipe with two peaks and a valley is split into two passes."""
    s = copy.deepcopy(synthetic_swipe)
    n = len(s.mag_t)
    # Duplicate to create forward and return pass
    t_span = s.mag_t[-1] - s.mag_t[0]
    s.mag_t = np.concatenate([s.mag_t, s.mag_t + t_span])
    s.mag_raw = np.concatenate([s.mag_raw, s.mag_raw[::-1]])
    # Magnetic signal with two peaks and a deep valley
    peak1 = np.exp(-((np.linspace(-3, 3, n)) ** 2)) * 300.0
    peak2 = np.exp(-((np.linspace(-3, 3, n)) ** 2)) * 300.0
    b_mag = np.concatenate([peak1, peak2])
    s.mag_cal = np.column_stack([b_mag, np.zeros_like(b_mag), np.zeros_like(b_mag)])
    s.mag_saturated = np.zeros(len(s.mag_t), dtype=bool)

    s.ori_t = np.concatenate([s.ori_t, s.ori_t + t_span])
    s.ori_q = np.concatenate([s.ori_q, s.ori_q])

    passes = segment_reversals(s)
    assert len(passes) == 2
    # Pass 1 goes start -> end, Pass 2 goes end -> start
    np.testing.assert_allclose(passes[0].planned_start_m, s.planned_start_m)
    np.testing.assert_allclose(passes[0].planned_end_m, s.planned_end_m)
    np.testing.assert_allclose(passes[1].planned_start_m, s.planned_end_m)
    np.testing.assert_allclose(passes[1].planned_end_m, s.planned_start_m)
