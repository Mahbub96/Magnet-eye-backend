"""The coarse-to-fine next-swipe search against scoring every candidate."""

import numpy as np
import pytest

from app.pipeline.dipole import dipole_kernel
from app.pipeline.uncertainty import (
    candidate_lines,
    posterior_covariance,
    region_points,
    suggest_next_swipe,
)
from app.sim import magnets
from app.sim.phone import NOISE
from app.sim.session import run_session


def dense_reference(sol, region, noise, limit, z_low, half_len, step):
    c = posterior_covariance(sol)
    f = dipole_kernel(region, sol.grid.centers)
    fc = f @ c
    total = float(np.trace(fc @ f.T))
    best = None
    for a, b in candidate_lines(
        sol.grid.centers.mean(axis=0), half_len, z_low, sol.grid.standoff / 2
    ):
        n = max(int(np.linalg.norm(b - a) / step), 2)
        pts = a + np.linspace(0, 1, n)[:, None] * (b - a)
        if np.linalg.norm(sol.field(pts) + sol.background, axis=1).max() > limit:
            continue
        h = dipole_kernel(pts, sol.grid.centers)
        s = h @ c @ h.T + noise**2 * np.eye(h.shape[0])
        fch = fc @ h.T
        red = float(np.sum(fch * np.linalg.solve(s, fch.T).T))
        if best is None or red > best[0]:
            best = (red, a, b)
    return best[0] / total, best[1], best[2]


@pytest.mark.parametrize("name", ["disc", "bar", "broken", "two"])
def test_fast_search_matches_the_dense_formula(name):
    r = run_session(magnets.ALL[name](), NOISE["realistic"], "realistic", guided=0, seed=1)
    sol = r.fit.solution
    z_low = float(r.fit.binned.pos[:, 2].min())
    region = region_points(sol, z_low)
    args = (sol, region, 0.8, 0.7 * 4900, z_low, 0.1, r.fit.bin_len)
    fast = suggest_next_swipe(*args)
    red, a, b = dense_reference(*args)
    # The coarse-to-fine search finds the full search's line, or one at least 98 % as good.
    same = np.allclose(fast.start, a) and np.allclose(fast.end, b)
    assert same or fast.variance_reduction >= 0.98 * red
    assert fast.variance_reduction <= red * (1 + 1e-9)
