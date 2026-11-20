"""A whole simulated scan: starting swipes, guided swipes, and scoring against the truth."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from app.pipeline.fit import SessionFit, fit_session
from app.pipeline.observations import Observations
from app.pipeline.uncertainty import region_points, relative_uncertainty, suggest_next_swipe

from .magnets import TestMagnet
from .phone import Phone, record_swipe

SAFE_FRACTION = 0.7  # stay under 70 % of the sensor range (pass bar doc)


def scan_height(magnet: TestMagnet, phone: Phone, half: float) -> float:
    """Lowest sensor height (m above the magnet's top surface) where the field stays under the safe
    fraction of the range over the scan area. Stands in for the app's live saturation
    warning; never closer than 5 mm (phone back cover to sensor)."""
    xs = np.linspace(-half, half, 41)
    x, y = np.meshgrid(xs, xs)
    for h in np.arange(0.005, 0.2, 0.002):
        pts = np.stack([x.ravel(), y.ravel(), np.full(x.size, magnet.top + h)], axis=1)
        if np.abs(magnet.field_ut(pts)).max() < SAFE_FRACTION * phone.range_ut:
            return float(h)
    raise RuntimeError("magnet too strong to scan")


def starting_swipes(h: float, half: float, top: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Four parallel lines, one crossing line, one higher line (PRD 7)."""
    z1, z2 = top + h, top + 2 * h
    out = []
    for y in (-1.5 * h, -0.5 * h, 0.5 * h, 1.5 * h):
        out.append((np.array([-half, y, z1]), np.array([half, y, z1])))
    out.append((np.array([0.0, -half, z1]), np.array([0.0, half, z1])))
    out.append((np.array([-half, 0.0, z2]), np.array([half, 0.0, z2])))
    return out


@dataclass
class Score:
    field_err: float
    moment_err: float
    unc_coverage: float  # fraction of errors inside the 2-sigma band
    rel_unc: float  # what the app would show as "uncertainty"
    piece_cos: list[float]
    centroid_err_mm: float
    hole_ratio: float  # |M| inside the ring hole / |M| in the ring band (ring cases only)
    fit_seconds: float


@dataclass
class SessionResult:
    magnet: str
    noise: str
    method: str
    scan_height: float
    history: list[Score] = field(default_factory=list)
    fit: SessionFit | None = None


def test_points(h: float, half: float, top: float) -> np.ndarray:
    xs = np.linspace(-0.8 * half, 0.8 * half, 11)
    zs = np.linspace(top + h, top + 1.5 * h, 3)
    x, y, z = np.meshgrid(xs, xs, zs, indexing="ij")
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def score(magnet: TestMagnet, sf: SessionFit, h: float, half: float, secs: float) -> Score:
    sol = sf.solution
    pts = test_points(h, half, magnet.top)
    b_true = magnet.field_ut(pts)
    b_pred = sol.field(pts)
    sig = sol.field_sigma(pts)
    err = b_pred - b_true
    field_err = float(np.linalg.norm(err) / np.linalg.norm(b_true))
    m_true = magnet.moment
    moment_err = float(np.linalg.norm(sol.total_moment - m_true) / np.linalg.norm(m_true))
    coverage = float(np.mean(np.abs(err) <= 2 * sig))
    pieces = magnet.pieces()
    cents = np.array([c for c, _ in pieces])
    owner = np.argmin(np.linalg.norm(sol.grid.centers[:, None, :] - cents[None], axis=2), axis=1)
    cos = []
    for i, (_, m) in enumerate(pieces):
        mm = sol.moments[owner == i].sum(axis=0)
        cos.append(float(mm @ m / (np.linalg.norm(mm) * np.linalg.norm(m) + 1e-30)))
    w = np.linalg.norm(sol.moments, axis=1)
    centroid = (w[:, None] * sol.grid.centers).sum(axis=0) / w.sum()
    true_centroid = np.average(cents, axis=0, weights=[np.linalg.norm(m) for _, m in pieces])
    r = np.linalg.norm(sol.grid.centers[:, :2], axis=1)
    inner, band = w[r < 0.007], w[(r > 0.011) & (r < 0.021)]
    hole = float(inner.mean() / band.mean()) if inner.size and band.size else float("nan")
    region = region_points(sol, magnet.top + h)
    return Score(
        field_err=field_err,
        moment_err=moment_err,
        unc_coverage=coverage,
        rel_unc=relative_uncertainty(sol, region),
        piece_cos=cos,
        centroid_err_mm=float(np.linalg.norm(centroid - true_centroid) * 1e3),
        hole_ratio=hole,
        fit_seconds=secs,
    )


def run_session(
    magnet: TestMagnet,
    phone: Phone,
    noise_name: str,
    method: str = "tikhonov",
    guided: int = 4,
    seed: int = 0,
    max_voxels: int = 600,
    use_table: bool = True,
) -> SessionResult:
    rng = np.random.default_rng(seed)
    half0 = magnet.radius + 0.03
    h = scan_height(magnet, phone, half0)
    half = magnet.radius + 2.5 * h
    swipes = [
        record_swipe(magnet, a, b, phone, rng, i)
        for i, (a, b) in enumerate(starting_swipes(h, half, magnet.top))
    ]
    res = SessionResult(magnet.name, noise_name, method, h)
    for k in range(guided + 1):
        raw = Observations.concat(swipes)
        t0 = time.perf_counter()
        sf = fit_session(
            raw, method=method, max_voxels=max_voxels, table_z=0.0 if use_table else None
        )
        secs = time.perf_counter() - t0
        res.history.append(score(magnet, sf, h, half, secs))
        res.fit = sf
        if k == guided:
            break
        sug = suggest_next_swipe(
            sf.solution,
            region_points(sf.solution, magnet.top + h),
            noise_ut=phone.noise_ut,
            b_limit_ut=SAFE_FRACTION * phone.range_ut,
            z_low=float(raw.pos[:, 2].min()),
            half_len=half,
            step=sf.bin_len,
        )
        if sug is None:
            break
        swipes.append(record_swipe(magnet, sug.start, sug.end, phone, rng, len(swipes)))
    return res
