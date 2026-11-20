"""One session fit: locate, size the grid, weight the readings, solve. Pure functions, so the
API worker and the notebooks call exactly the same code."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dipole import dipole_field, dipole_gradient_norm
from .inverse import Solution, SwipeBias, solve
from .locate import DipoleFit, VoxelGrid, fit_dipole, model_field, size_grid
from .observations import Observations, bin_along_swipes, noise_sigma


def swipe_bias_blocks(obs: Observations, dip: DipoleFit) -> list[SwipeBias]:
    """Per-swipe sensitivity of the readings to a shared position shift (3), a shared small
    rotation (3) and a shared constant field offset (3), linearised around the dipole fit.

    The field offset replaces a free background: its size is known from calibration, so it
    cannot soak up part of a large magnet's smooth field (which biased moments low)."""
    src, mom = dip.pos[None, :], dip.moment[None, :]
    b0 = dipole_field(obs.pos, src, mom)
    h = 1e-4
    jpos = np.empty((len(obs), 3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        jpos[:, :, j] = (
            dipole_field(obs.pos + e, src, mom) - dipole_field(obs.pos - e, src, mom)
        ) / (2 * h)
    # rotation theta: b -> b + theta x b  =>  d b / d theta = -[b]x
    jrot = np.zeros((len(obs), 3, 3))
    bx, by, bz = b0[:, 0], b0[:, 1], b0[:, 2]
    jrot[:, 0, 1], jrot[:, 0, 2] = bz, -by
    jrot[:, 1, 0], jrot[:, 1, 2] = -bz, bx
    jrot[:, 2, 0], jrot[:, 2, 1] = by, -bx
    blocks = []
    for s in np.unique(obs.swipe):
        rows = np.flatnonzero(obs.swipe == s)
        joff = np.broadcast_to(np.eye(3), (rows.size, 3, 3))
        jac = np.concatenate([jpos[rows], jrot[rows], joff], axis=2).reshape(3 * rows.size, 9)
        pv = np.concatenate(
            [
                np.full(3, np.mean(obs.bias_pos[rows]) ** 2),
                np.full(3, np.mean(obs.bias_rot[rows]) ** 2),
                np.full(3, np.mean(obs.bias_b[rows]) ** 2),
            ]
        )
        blocks.append(SwipeBias(rows=rows, jac=jac, prior_var=pv))
    return blocks


@dataclass
class SessionFit:
    dipole: DipoleFit
    grid: VoxelGrid
    solution: Solution
    binned: Observations
    sigma: np.ndarray  # per binned row, µT
    bin_len: float


def fit_session(
    raw: Observations,
    method: str = "tikhonov",
    max_voxels: int = 600,
    lam_rule: str = "gcv",
    table_z: float | None = None,
) -> SessionFit:
    # 1. Rough bins (scale from the scanned area) just to locate the magnet cheaply.
    spread = float(np.linalg.norm(raw.pos.std(axis=0)))
    rough = bin_along_swipes(raw, spread / 40)
    dip = fit_dipole(rough)
    # 2. Rebin at a length set by the resolution limit (a sixth of the standoff).
    standoff = float(np.min(np.linalg.norm(raw.pos - dip.pos, axis=1)))
    bin_len = standoff / 6
    binned = bin_along_swipes(raw, bin_len)
    # 3. Weight each reading by sensor noise plus position and orientation error, scaled by
    #    the local field strength and gradient from the dipole fit; then refit the dipole.
    b_dip = model_field(dip, binned.pos)
    grad = dipole_gradient_norm(binned.pos, dip.pos[None, :], dip.moment[None, :])
    sigma = noise_sigma(binned, b_dip, grad)
    dip = fit_dipole(binned, sigma)
    # 4. Voxel grid from the data, then the any-shape solve.
    grid = size_grid(binned, dip, max_voxels, table_z)
    bias = swipe_bias_blocks(binned, dip)
    sol = solve(binned, sigma, grid, method=method, lam_rule=lam_rule, bias=bias)
    return SessionFit(dip, grid, sol, binned, sigma, bin_len)
