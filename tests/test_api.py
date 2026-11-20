"""API (SRS 4.3) end to end with simulated swipe files, plus the service limits (PR-13)."""

import dataclasses
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import store as store_mod
from app.api.config import SETTINGS
from app.api.main import app
from app.pipeline.swipe_format import parse_swipe
from app.sim import magnets
from app.sim.phone import NOISE
from app.sim.session import scan_height
from app.sim.swipe_file import ruler_lines, swipe_file

client = TestClient(app)
NEW = {"app_version": "test", "api_version": "v1", "calibration_schema_version": 1}


def disc_session(n_lines=5, seed=0):
    m = magnets.disc()
    phone = NOISE["realistic"]
    rng = np.random.default_rng(seed)
    h = scan_height(m, phone, m.radius + 0.03)
    z, half = m.top + h, m.radius + 2.5 * h
    lines = ruler_lines(half, z, h, n_lines) + [([0, -half, z], [0, half, z])]
    return m, z, [swipe_file(m, a, b, i, phone, rng) for i, (a, b) in enumerate(lines)]


def wait(job_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/v1/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_health():
    r = client.get("/v1/health").json()
    assert r["status"] == "ok" and r["api_version"] == "v1"
    assert r["queue"]["size"] == SETTINGS.queue_size


def test_whole_session_through_the_api():
    m, z, files = disc_session()
    sid = client.post("/v1/sessions", json=NEW).json()["session_id"]
    for f in files:
        r = client.post(f"/v1/sessions/{sid}/swipes", json=f)
        assert r.status_code == 202, r.text
        j = wait(r.json()["job_id"])
    assert j["status"] == "done", j
    res = client.get(f"/v1/sessions/{sid}/result").json()
    assert res["swipes"] == len(files)
    out = res["result"]
    assert out["fit_seconds"] < 1.0  # NF-1
    assert "Estimate" in out["label"]  # PR-22
    # The returned field matches the true magnet's at the scan height.
    pts = np.array(out["field"]["points_m"])
    near = np.abs(pts[:, 2] - z) < 0.005
    bt = m.field_ut(pts[near])
    be = np.array(out["field"]["b_ut"])[near]
    assert np.linalg.norm(be - bt) / np.linalg.norm(bt) < 0.10
    mom = np.array(out["magnet"]["moment_am2"])
    assert np.linalg.norm(mom - m.moment) / np.linalg.norm(m.moment) < 0.15
    assert out["next_swipe"] is None or len(out["next_swipe"]["start_m"]) == 3
    assert client.delete(f"/v1/sessions/{sid}").status_code == 204
    assert client.get(f"/v1/sessions/{sid}/result").status_code == 404


def test_versions_are_checked():
    r = client.post("/v1/sessions", json={**NEW, "api_version": "v2"})
    assert r.status_code == 409 and r.json()["code"] == "api_version"
    sid = client.post("/v1/sessions", json=NEW).json()["session_id"]
    _, _, files = disc_session(n_lines=1)
    bad = {**files[0], "calibration_schema_version": 2}
    r = client.post(f"/v1/sessions/{sid}/swipes", json=bad)
    assert r.status_code == 409 and r.json()["code"] == "calibration_schema_version"


def test_bad_swipes_are_refused_with_the_reason():
    sid = client.post("/v1/sessions", json=NEW).json()["session_id"]
    _, _, files = disc_session(n_lines=1)
    broken = {**files[0]}
    broken.pop("session_id")
    r = client.post(f"/v1/sessions/{sid}/swipes", json=broken)
    assert r.status_code == 422 and "session_id" in r.json()["message"]
    second = {**files[1], "swipe_index": 1}  # no start baseline, but it is the first upload
    r = client.post(f"/v1/sessions/{sid}/swipes", json=second)
    assert r.status_code == 422 and "start baseline" in r.json()["message"]
    assert client.post("/v1/sessions/nope/swipes", json=files[0]).status_code == 404


def test_full_queue_answers_429_with_retry_after(monkeypatch):
    gate = threading.Event()

    def slow(swipes, state):
        gate.wait(5)
        return {"ok": True}

    monkeypatch.setattr(store_mod, "run_fit", slow)
    s = store_mod.Store(dataclasses.replace(SETTINGS, queue_size=1))
    a, b, c = s.create("a"), s.create("b"), s.create("c")
    sw = parse_swipe(disc_session(n_lines=1)[2][0])
    s.add_swipe(a, sw)  # taken by the worker
    time.sleep(0.1)
    s.add_swipe(b, sw)  # waits in the queue
    with pytest.raises(store_mod.QueueFull) as e:
        s.add_swipe(c, sw)
    assert e.value.retry_after_s >= 1
    assert len(c.swipes) == 0  # the refused swipe is not kept
    gate.set()


def test_a_burst_of_swipes_shares_one_waiting_refit(monkeypatch):
    gate = threading.Event()
    seen = []

    def slow(swipes, state):
        gate.wait(5)
        seen.append(len(swipes))
        return {"ok": True}

    monkeypatch.setattr(store_mod, "run_fit", slow)
    s = store_mod.Store(dataclasses.replace(SETTINGS, queue_size=1))
    sess = s.create("t")
    sw = parse_swipe(disc_session(n_lines=1)[2][0])
    first = s.add_swipe(sess, sw)  # running
    time.sleep(0.1)
    jobs = [s.add_swipe(sess, sw) for _ in range(4)]  # all join one waiting refit
    assert len({j.id for j in jobs}) == 1 and jobs[0].id != first.id
    assert jobs[-1].swipes == 5
    gate.set()
    t0 = time.time()
    while s.job(jobs[0].id).status != "done" and time.time() - t0 < 5:
        time.sleep(0.02)
    assert seen == [1, 5]


def test_idle_sessions_are_forgotten():
    s = store_mod.Store(dataclasses.replace(SETTINGS, session_idle_s=0))
    sess = s.create("test")
    time.sleep(0.01)
    assert s.get(sess.id) is None


def test_moved_magnet_is_flagged_and_steady_scans_are_not():
    from app.api.fit_job import FitState, run_fit

    m, _, files = disc_session()
    swipes = [parse_swipe(f) for f in files]
    state = FitState()
    for k in (1, 3, 4, 5):  # uploads arriving one by one and in bursts
        out = run_fit(swipes[:k], state)
        assert "magnet_may_have_moved" not in out["warnings"], k
    # The magnet slides 3 cm before the last swipe.
    moved = swipe_file(
        m.__class__(m.name, m.sources.copy().move((0.03, 0, 0)), m.radius, m.top),
        *disc_lines_last(files),
        5,
        NOISE["realistic"],
        np.random.default_rng(9),
    )
    out = run_fit(swipes[:5] + [parse_swipe(moved)], state)
    assert "magnet_may_have_moved" in out["warnings"]


def disc_lines_last(files):
    line = files[-1]["planned_line"]
    return line["start_m"], line["end_m"]
