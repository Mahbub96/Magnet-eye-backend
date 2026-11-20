"""Swipe file -> observations, on the app's synthetic fixture (phone flat, Earth field only,
yawed 0.8 rad in the game-rotation-vector frame, sliding x -0.1 -> 0.1 m)."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from app.pipeline.align import (
    interpolate_orientation,
    interpolate_position,
    session_baseline,
    swipe_observations,
)
from app.pipeline.swipe_format import parse_swipe

FIXTURE = Path(__file__).parent / "fixtures" / "swipe-v1-synthetic.json"
EARTH_PAPER = Rotation.from_euler("z", -0.8).apply([0.0, 22.0, -40.0])


@pytest.fixture
def raw():
    return json.loads(FIXTURE.read_text())


def test_earth_field_cancels_after_rotation_and_baseline(raw):
    s = parse_swipe(raw)
    obs, rep = swipe_observations(s, EARTH_PAPER, swipe_id=0)
    assert rep.kept == rep.readings == len(obs) == 300
    assert rep.max_iron_mismatch_ut < 1e-9
    # Only sensor noise (0.3 uT) is left once the field is in the paper frame.
    assert np.abs(obs.b.mean(axis=0)).max() < 0.1
    assert obs.b.std(axis=0).max() < 0.4


def test_positions_follow_the_ruler_marks(raw):
    s = parse_swipe(raw)
    obs, _ = swipe_observations(s, EARTH_PAPER, swipe_id=3)
    assert obs.pos[0, 0] == pytest.approx(-0.1)
    assert obs.pos[-1, 0] == pytest.approx(0.1)
    assert np.all(np.diff(obs.pos[:, 0]) >= -1e-12)  # never moves backwards
    np.testing.assert_allclose(obs.pos[:, 1:], 0.0, atol=1e-12)
    assert set(obs.swipe) == {3}


def test_saturated_readings_are_dropped(raw):
    d = copy.deepcopy(raw)
    for r in d["streams"]["mag"]["rows"][10:15]:
        r[7] = 1
    obs, rep = swipe_observations(parse_swipe(d), EARTH_PAPER, swipe_id=0)
    assert rep.saturated == 5 and rep.kept == 295 == len(obs)


def test_interpolation_helpers():
    t = np.array([0, 1_000_000_000], dtype=np.int64)
    q = np.array([[1.0, 0, 0, 0], [np.cos(0.5), 0, 0, np.sin(0.5)]])  # 0 -> 1 rad about z
    mid, outside = interpolate_orientation(t, q, np.array([500_000_000, 2_000_000_000]), 0)
    assert 2 * np.arctan2(mid[0, 3], mid[0, 0]) == pytest.approx(0.5)
    assert outside == 1
    p = interpolate_position(t, np.array([[0.0, 0, 0], [0.2, 0, 0]]), np.array([250_000_000]), 0)
    assert p[0, 0] == pytest.approx(0.05)


def test_session_baseline_and_drift(raw):
    a = parse_swipe(raw)
    d = copy.deepcopy(raw)
    d["swipe_index"] = 1
    d["session_geometry"]["baseline_start_ut"] = None
    d["session_geometry"]["baseline_end_ut"] = [3.0, 4.0, 0.0]
    b = parse_swipe(d)
    base, drift = session_baseline([a, b])
    np.testing.assert_allclose(base, a.baseline_start_ut)
    assert drift == pytest.approx(5.0)
