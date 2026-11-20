"""Swipe format v1 contract with the app. The fixture was written by the app's TypeScript
builder (magnet-eye-app: WRITE_FIXTURE=... npx jest buildSwipe)."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from app.pipeline.swipe_format import SwipeFormatError, load_swipe, parse_swipe

FIXTURE = Path(__file__).parent / "fixtures" / "swipe-v1-synthetic.json"


@pytest.fixture
def raw():
    return json.loads(FIXTURE.read_text())


def test_app_fixture_loads_with_every_field(raw):
    s = load_swipe(FIXTURE)
    assert s.tracking_mode == "ruler"
    assert s.mag_raw.shape == s.mag_cal.shape == (300, 3)
    assert not s.mag_saturated.any()
    assert s.ori_q.shape == (300, 4)
    assert s.pos.shape == (4, 3)
    assert s.pos_state == ["ruler_endpoint"] * 4
    assert s.calibration.sampling_rate_hz == pytest.approx(100)
    assert s.baseline_start_ut is not None
    assert s.accel is not None and s.gyro is not None
    np.testing.assert_allclose(s.mag_cal, s.mag_raw - s.calibration.hard_iron_ut, atol=1e-9)


def test_orientation_convention_matches_frames_to_world(raw):
    """The app's quaternions rotate calibrated phone readings into the paper frame; SciPy with
    the wxyz -> xyzw reorder used in frames.to_world must give the same world field."""
    s = parse_swipe(raw)
    q = s.ori_q[:, [1, 2, 3, 0]]
    world = Rotation.from_quat(q).apply(s.mag_cal)
    earth = np.array([0.0, 22.0, -40.0])
    paper = Rotation.from_euler("z", -0.8).apply(earth)
    ok = ~s.mag_saturated
    np.testing.assert_allclose(world[ok].mean(axis=0), paper, atol=0.1)


@pytest.mark.parametrize(
    "edit, problem",
    [
        (lambda d: d.pop("session_id"), "session_id"),
        (lambda d: d["calibration"].update(noise_ut=[1, 2]), "calibration.noise_ut"),
        (lambda d: d["session_geometry"].update(baseline_start_ut=None), "baseline_start_ut"),
        (lambda d: d["streams"]["mag"]["rows"].reverse(), "streams.mag: timestamps not in order"),
        (lambda d: d["streams"]["orientation"]["rows"][0].__setitem__(1, 2.0), "unit length"),
        (lambda d: d["streams"]["position"]["rows"][0].__setitem__(4, "teleport"), "unknown"),
        (lambda d: d.update(planned_line=None), "planned_line"),
        (lambda d: d["clock"].update(origin_ns=123), "clock.origin_ns"),
    ],
)
def test_bad_files_are_rejected_with_the_field_named(raw, edit, problem):
    d = copy.deepcopy(raw)
    edit(d)
    with pytest.raises(SwipeFormatError) as e:
        parse_swipe(d)
    assert any(problem in m for m in e.value.problems), e.value.problems


def test_newer_format_version_is_refused(raw):
    raw["format_version"] = 2
    with pytest.raises(SwipeFormatError, match="format_version"):
        parse_swipe(raw)
