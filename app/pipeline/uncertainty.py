"""Uncertainty map, stopping rule and the next suggested swipe (PR-10, SC-10, SC-11).

The next swipe is the candidate line that most reduces the predicted field variance over
the region shown to the user (an A-optimal design step), among lines that keep the sensor
below its saturation limit according to the current fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dipole import dipole_kernel
from .inverse import Solution


@dataclass
class SwipeSuggestion:
    start: np.ndarray  # (3,) m, world frame
    end: np.ndarray  # (3,) m
    height: float  # m above the lowest reading of the session
    variance_reduction: float  # fraction of the current region variance it is expected to remove


def posterior_covariance(sol: Solution) -> np.ndarray:
    p = sol.posterior
    n = p.col_scale.size
    vv = (p.v / (p.s**2 + p.lam)) @ p.v.T
    c = vv + (np.eye(n) - p.v @ p.v.T) / p.lam
    if p.u0 is not None:  # y = T z with T = I + (kappa - 1) u0 u0^T
        t = np.eye(n) + (p.kappa - 1.0) * p.u0 @ p.u0.T
        c = t @ c @ t.T
    return p.s2 * (p.col_scale[:, None] * c * p.col_scale[None, :])


def region_points(sol: Solution, z_min: float, n_xy: int = 9, n_z: int = 3) -> np.ndarray:
    """Points where the field is shown and graded: above the voxel footprint, from the scan
    height (z_min) up by one standoff."""
    c = sol.grid.centers
    pad = 0.5 * sol.grid.standoff
    xs = np.linspace(c[:, 0].min() - pad, c[:, 0].max() + pad, n_xy)
    ys = np.linspace(c[:, 1].min() - pad, c[:, 1].max() + pad, n_xy)
    zs = np.linspace(z_min, z_min + sol.grid.standoff, n_z)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def relative_uncertainty(sol: Solution, points: np.ndarray) -> float:
    """RMS field uncertainty over RMS field strength in the region (the stopping number)."""
    sig = sol.field_sigma(points)
    b = sol.field(points)
    return float(np.sqrt(np.mean(sig**2)) / np.sqrt(np.mean(b**2)))


HEIGHTS = (0.0, 0.5, 1.0)
COARSE = 3  # first pass scores every 3rd offset
REFINE = 3  # then the neighbours of the 3 best


def candidate_lines(
    center: np.ndarray,
    half_len: float,
    z_low: float,
    spacing: float,
    heights: tuple[float, ...] = HEIGHTS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Straight swipes across the scan area: 4 directions, several offsets, several heights.

    Heights are multiples of `spacing*2` above the lowest reading so far; offsets step by
    `spacing`. All values come from the current session (standoff), not fixed numbers.
    """
    lines = []
    eff_spacing = max(spacing, half_len / 10.0)
    n_off = min(int(np.floor(half_len / eff_spacing)), 12)
    for ang in np.deg2rad([0.0, 45.0, 90.0, 135.0]):
        u = np.array([np.cos(ang), np.sin(ang), 0.0])
        nrm = np.array([-u[1], u[0], 0.0])
        for k in range(-n_off, n_off + 1):
            for h in heights:
                mid = center + k * eff_spacing * nrm
                mid = np.array([mid[0], mid[1], z_low + h * 2 * eff_spacing])
                lines.append((mid - half_len * u, mid + half_len * u))
    return lines


def suggest_next_swipe(
    sol: Solution,
    region: np.ndarray,
    noise_ut: float,
    b_limit_ut: float,
    z_low: float,
    half_len: float,
    step: float,
) -> SwipeSuggestion | None:
    """Pick the candidate swipe that removes the most field variance in `region`.

    noise_ut: expected per-axis noise of one binned reading (from calibration).
    b_limit_ut: largest field the sensor may see (measured range times a safety factor).
    step: spacing of readings along the candidate line (the bin length).
    """
    c = posterior_covariance(sol)
    f = dipole_kernel(region, sol.grid.centers)
    fc = f @ c
    total = float(np.trace(fc @ f.T))
    m = sol.moments.ravel()
    lines = candidate_lines(sol.grid.centers.mean(axis=0), half_len, z_low, sol.grid.standoff / 2)
    scores: dict[int, float | None] = {}

    def score(i: int) -> float | None:
        """Expected variance removed by candidate i, or None if it would saturate."""
        if i not in scores:
            a, b = lines[i]
            n = min(max(int(np.linalg.norm(b - a) / step), 2), 25)
            pts = a + np.linspace(0, 1, n)[:, None] * (b - a)
            h = dipole_kernel(pts, sol.grid.centers)
            bpred = np.linalg.norm((h @ m).reshape(-1, 3) + sol.background, axis=1)
            if bpred.max() > b_limit_ut:
                scores[i] = None
            else:
                hc = h @ c
                s = hc @ h.T + noise_ut**2 * np.eye(h.shape[0])
                fch = fc @ h.T
                scores[i] = float(np.sum(fch * np.linalg.solve(s, fch.T).T))
        return scores[i]

    # Coarse to fine: the score changes smoothly as a line moves sideways, so score every
    # COARSE-th offset in every direction and height, then the offsets around the best few.
    n_h = len(HEIGHTS)
    n_off = len(lines) // (4 * n_h)
    idx = np.arange(len(lines)).reshape(4, n_off, n_h)
    coarse = idx[:, ::COARSE, :].ravel()
    if n_off % COARSE != 1:
        coarse = np.concatenate([coarse, idx[:, -1, :].ravel()])
    ranked = sorted((i for i in coarse if score(int(i)) is not None), key=lambda i: -scores[int(i)])
    for i in ranked[:REFINE]:
        d, k, hh = np.unravel_index(int(i), idx.shape)
        for kk in range(max(0, k - COARSE + 1), min(n_off, k + COARSE)):
            score(int(idx[d, kk, hh]))
    valid = [(v, i) for i, v in scores.items() if v is not None]
    best = None
    if valid:
        red, i = max(valid)
        best = (red, *lines[i])
    if best is None:
        return None
    red, a, b = best
    return SwipeSuggestion(
        start=a, end=b, height=float(a[2] - z_low), variance_reduction=red / total
    )
