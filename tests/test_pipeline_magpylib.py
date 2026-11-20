"""Accuracy tests against synthetic Magpylib magnets (Challenges guide rule 10).

These encode the Phase 1 pass bar (docs/phase1-pass-bar.md) at realistic noise. The full
multi-seed table is produced by scripts/phase1_report.py; these use one fixed seed so CI is
fast and deterministic.
"""

import numpy as np
import pytest

from app.pipeline.dipole import dipole_kernel
from app.pipeline.uncertainty import posterior_covariance
from app.sim import magnets
from app.sim.phone import NOISE
from app.sim.session import SAFE_FRACTION, run_session

FIELD_ERR = 0.10
MOMENT_ERR = 0.15
PIECE_COS = 0.7
COVERAGE = 0.80
SECONDS = 1.0


@pytest.fixture(scope="module")
def results():
    return {
        name: run_session(magnets.ALL[name](), NOISE["realistic"], "realistic", seed=0)
        for name in ("ring", "bar", "disc", "broken")
    }


@pytest.mark.parametrize("name", ["ring", "bar", "disc", "broken"])
def test_field_error(results, name):
    assert results[name].history[-1].field_err <= FIELD_ERR


@pytest.mark.parametrize("name", ["ring", "bar", "disc", "broken"])
def test_uncertainty_is_honest(results, name):
    assert results[name].history[-1].unc_coverage >= COVERAGE


@pytest.mark.parametrize("name", ["ring", "bar", "disc", "broken"])
def test_fit_is_fast(results, name):
    assert max(s.fit_seconds for s in results[name].history) <= SECONDS


@pytest.mark.parametrize("name", ["ring", "bar", "disc"])
def test_moment_error(results, name):
    assert results[name].history[-1].moment_err <= MOMENT_ERR


def test_broken_magnet_poles(results):
    # Every piece, including the flipped one, points the right way.
    assert min(results["broken"].history[-1].piece_cos) >= PIECE_COS


def test_guided_swipes_help(results):
    for r in results.values():
        assert r.history[-1].field_err <= r.history[0].field_err * 1.05


def test_suggested_swipes_avoid_saturation():
    """Guided swipes are chosen from the current fit; none may drive the sensor into
    saturation (the true field along them stays under the sensor range)."""
    mag = magnets.disc()
    phone = NOISE["ideal"]
    r = run_session(mag, phone, "ideal", guided=2, seed=3)
    guided = r.fit.binned.swipe >= 6
    assert guided.any()
    worst = np.abs(mag.field_ut(r.fit.binned.pos[guided])).max()
    assert worst < phone.range_ut
    assert worst < 1.2 * SAFE_FRACTION * phone.range_ut


def test_posterior_paths_agree(results):
    """The full covariance (guided swipes) and the per-row variance (error bars) describe the
    same posterior, including the unpenalised uniform-magnetisation mode."""
    sol = results["bar"].fit.solution
    pts = sol.grid.centers[:5] + np.array([0.0, 0.0, 0.03])
    op = dipole_kernel(pts, sol.grid.centers)
    full = np.einsum("ij,jk,ik->i", op, posterior_covariance(sol), op)
    np.testing.assert_allclose(full, sol.linear_variance(op), rtol=1e-6)
