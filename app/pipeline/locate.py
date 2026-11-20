"""Coarse first pass: one dipole plus a uniform background, then size the voxel grid.

The fit uses variable projection: for a trial dipole position the moment and the
background enter linearly, so only the 3 position values are searched non-linearly.
Several starting depths are tried, so no magnet size is assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .dipole import dipole_field, dipole_kernel
from .observations import Observations


@dataclass
class DipoleFit:
    pos: np.ndarray  # (3,) m
    moment: np.ndarray  # (3,) A·m²
    background: np.ndarray  # (3,) µT
    chi2_per_dof: float


@dataclass
class VoxelGrid:
    centers: np.ndarray  # (M, 3) m
    size: float  # lateral voxel edge, m
    dz: float  # vertical voxel edge, m
    shape: tuple[int, int, int]
    origin: np.ndarray  # centre of the first voxel
    standoff: float  # smallest sensor-to-source distance, m (sets the resolution limit)


def _linear_part(obs: Observations, w: np.ndarray, p: np.ndarray):
    g = dipole_kernel(obs.pos, p[None, :])  # (3N, 3)
    a = np.hstack([g, np.tile(np.eye(3), (len(obs), 1))])  # moment + background
    aw = a * w[:, None]
    dw = obs.b.ravel() * w
    x, *_ = np.linalg.lstsq(aw, dw, rcond=None)
    return x, aw @ x - dw


def fit_dipole(obs: Observations, sigma: np.ndarray | None = None) -> DipoleFit:
    """Fit one dipole and a uniform background to the observations."""
    s = obs.sigma_b if sigma is None else sigma
    w = np.repeat(1.0 / s, 3)
    bres = obs.b - np.median(obs.b, axis=0)
    k = int(np.argmax(np.linalg.norm(bres, axis=1)))
    top = obs.pos[k]
    # Distance scale from the data itself: spread of the scanned positions.
    spread = float(np.linalg.norm(obs.pos.std(axis=0)))
    best = None
    for depth in (0.3, 0.6, 1.0, 1.6):
        p0 = top - np.array([0.0, 0.0, depth * spread])
        r = least_squares(
            lambda p: _linear_part(obs, w, p)[1],
            p0,
            method="lm",
            x_scale=spread,
        )
        cost = float(np.sum(r.fun**2))
        if best is None or cost < best[0]:
            best = (cost, r.x)
    cost, p = best
    x, _ = _linear_part(obs, w, p)
    dof = max(3 * len(obs) - 9, 1)
    return DipoleFit(pos=p, moment=x[:3], background=x[3:], chi2_per_dof=cost / dof)


# Height of the voxel grid above the dipole depth when the table plane is known, in standoffs.
# Chosen on Phase 1 seeds 0-2 and checked on fresh seeds 3-8 (docs/phase1-results.md).
TABLE_TOP_K = 0.4


def size_grid(
    obs: Observations, fit: DipoleFit, max_voxels: int, table_z: float | None = None
) -> VoxelGrid:
    """Choose the voxel grid extent and resolution from the data (PR-15).

    - Resolution: nothing smaller than about a third of the closest sensor-to-magnet
      distance can be resolved, so the voxel edge starts there.
    - Lateral extent: covers the region where the magnet's own field is strong, plus a
      margin of one standoff, so an elongated or broken magnet is not cut off.
    - Vertical extent: never above the lowest reading. If the table plane is known (ARCore
      plane, or the paper in ruler/grid mode) the grid starts on it, because the magnet cannot
      be below the table; this removes most of the depth-versus-strength ambiguity. Without
      it, the grid spans the dipole depth +- 0.6 standoff. On a table the top is the dipole
      depth + TABLE_TOP_K standoffs: voxels higher than the magnet let the smallest-norm
      solution put moment closer to the phone, where less of it explains the same field, so
      the total moment comes out low (bar: 12-20 % with 0.6).
    """
    rel = obs.pos - fit.pos
    standoff = float(np.min(np.linalg.norm(rel, axis=1)))
    b_mag = np.linalg.norm(obs.b - fit.background, axis=1)
    strong = b_mag > 0.3 * b_mag.max()
    half_xy = np.maximum(np.abs(rel[strong, :2]).max(axis=0), 0.5 * standoff) + 0.5 * standoff
    z_low_sample = float(obs.pos[:, 2].min())
    size = standoff / 3.0
    while True:
        nx, ny = np.ceil(2 * half_xy / size).astype(int) | 1  # odd, centred on the dipole
        if table_z is None:
            z_bot = fit.pos[2] - 0.6 * standoff
            z_top = min(fit.pos[2] + 0.6 * standoff, z_low_sample - 0.5 * size)
        else:
            z_bot = table_z
            z_top = min(
                max(fit.pos[2] + TABLE_TOP_K * standoff, table_z + size),
                z_low_sample - 0.5 * size,
            )
        nz = max(int(round((z_top - z_bot) / size)), 1)
        if nx * ny * nz <= max_voxels:
            break
        size *= 1.15
    dz = (z_top - z_bot) / nz
    origin = np.array(
        [fit.pos[0] - (nx - 1) / 2 * size, fit.pos[1] - (ny - 1) / 2 * size, z_bot + dz / 2]
    )
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    centers = origin + np.stack([size * ix.ravel(), size * iy.ravel(), dz * iz.ravel()], axis=1)
    return VoxelGrid(
        centers=centers, size=size, dz=dz, shape=(nx, ny, nz), origin=origin, standoff=standoff
    )


def model_field(fit: DipoleFit, pos: np.ndarray) -> np.ndarray:
    return dipole_field(pos, fit.pos[None, :], fit.moment[None, :])
