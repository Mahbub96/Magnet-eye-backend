"""Turn a solution into what the app draws: field grid, field lines, magnetisation image."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .inverse import Solution


@dataclass
class Reconstruction:
    grid_points: np.ndarray  # (P, 3) m
    field: np.ndarray  # (P, 3) µT, magnet only
    field_sigma: np.ndarray  # (P, 3) µT
    voxel_centers: np.ndarray  # (M, 3) m
    voxel_size: float
    magnetisation: np.ndarray  # (M, 3) A/m (moment / voxel volume)
    resolution: float  # m: detail finer than this cannot be recovered (NF-12)
    field_lines: list[np.ndarray]


def field_box(sol: Solution, z_min: float, n: int = 12) -> np.ndarray:
    c = sol.grid.centers
    pad = sol.grid.standoff
    xs = np.linspace(c[:, 0].min() - pad, c[:, 0].max() + pad, n)
    ys = np.linspace(c[:, 1].min() - pad, c[:, 1].max() + pad, n)
    zs = np.linspace(z_min, z_min + 2 * sol.grid.standoff, max(n // 2, 2))
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def trace_field_line(
    sol: Solution, seed: np.ndarray, step: float, n_steps: int, z_min: float
) -> np.ndarray:
    """RK4 along the field direction, both ways from the seed, staying in the trusted region
    (at or above z_min, i.e. not extrapolated into the space the phone never reached)."""

    def direction(p):
        b = sol.field(p[None, :])[0]
        n = np.linalg.norm(b)
        return b / n if n > 0 else b

    halves = []
    for sign in (1.0, -1.0):
        pts = [seed]
        p = seed.copy()
        for _ in range(n_steps):
            k1 = direction(p)
            k2 = direction(p + 0.5 * step * sign * k1)
            k3 = direction(p + 0.5 * step * sign * k2)
            k4 = direction(p + step * sign * k3)
            p = p + step * sign * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            if p[2] < z_min:
                break
            pts.append(p.copy())
        halves.append(np.array(pts))
    return np.vstack([halves[1][::-1], halves[0][1:]])


def reconstruct(sol: Solution, z_min: float, n_lines: int = 8) -> Reconstruction:
    pts = field_box(sol, z_min)
    vol = sol.grid.size**2 * sol.grid.dz
    c = sol.grid.centers
    ang = np.linspace(0, 2 * np.pi, n_lines, endpoint=False)
    center = c.mean(axis=0)
    r = 0.5 * (np.ptp(c[:, 0]) + sol.grid.standoff)
    seeds = np.stack(
        [center[0] + r * np.cos(ang), center[1] + r * np.sin(ang), np.full(n_lines, z_min)], axis=1
    )
    lines = [trace_field_line(sol, s, sol.grid.size / 2, 60, z_min) for s in seeds]
    return Reconstruction(
        grid_points=pts,
        field=sol.field(pts),
        field_sigma=sol.field_sigma(pts),
        voxel_centers=c,
        voxel_size=sol.grid.size,
        magnetisation=sol.moments / vol,
        resolution=sol.grid.standoff,
        field_lines=lines,
    )
