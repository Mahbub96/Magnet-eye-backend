"""One fit of a session's swipes, and the JSON the app draws from (PR-10, PR-18, PR-19, PR-22).

Units in the result: positions m, field µT, magnetisation A/m, moment A·m², angles none.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from app.pipeline.align import session_baseline, swipe_observations
from app.pipeline.clean import clean_swipe
from app.pipeline.fit import fit_session
from app.pipeline.observations import Observations
from app.pipeline.reconstruct import reconstruct
from app.pipeline.swipe_format import Swipe
from app.pipeline.uncertainty import region_points, relative_uncertainty, suggest_next_swipe

from .config import SETTINGS

SAFE_FRACTION = 0.7  # suggested swipes stay under this share of the sensor range
MOVED_RATIO = 3.0  # misfit per reading this many times the last fit's: magnet may have moved
MIN_SWIPES_TO_COMPARE = 3
HIGH_RESIDUAL = 4.0  # misfit per reading above this with good coverage: other iron or magnets


@dataclass
class FitState:
    """What a session keeps between fits."""

    misfit: float | None = None  # chi² per data point of the last fit
    swipes: int = 0  # how many swipes the last fit used
    history: list[float] = field(default_factory=list)


def _r(a: np.ndarray, nd: int = 5) -> list:
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def run_fit(swipes: list[Swipe], state: FitState) -> dict:
    t0 = time.perf_counter()
    base, drift = session_baseline(swipes)
    cleaned_swipes: list[Swipe] = []
    for s in swipes:
        cleaned_swipes.extend(clean_swipe(s, base))
    parts, reports = [], []
    for i, s in enumerate(cleaned_swipes):
        o, rep = swipe_observations(s, base, i, drift)
        parts.append(o)
        reports.append(rep)
    obs = Observations.concat(parts)
    if len(obs) < 20:
        raise ValueError("too few unsaturated readings to fit")
    table_z = swipes[0].table_z_m
    sf = fit_session(obs, max_voxels=SETTINGS.max_voxels, table_z=table_z)
    sol = sf.solution
    z_low = float(obs.pos[:, 2].min())
    rec = reconstruct(sol, z_low)
    region = region_points(sol, z_low)
    rel_unc = relative_uncertainty(sol, region)
    misfit = sol.chi2 / max(sol.n_data, 1)

    warnings = []
    # PR-19: a jump in misfit when one or two swipes are added to a fit that already had a
    # few. Fits from one or two swipes match their own data too easily to compare against.
    added = len(swipes) - state.swipes
    if (
        state.misfit is not None
        and state.swipes >= MIN_SWIPES_TO_COMPARE
        and 0 < added <= 2
        and misfit > MOVED_RATIO * state.misfit
    ):
        warnings.append("magnet_may_have_moved")
    if len(swipes) >= 4 and misfit > HIGH_RESIDUAL:
        warnings.append("high_residual")  # PR-18
    if any(r.saturated for r in reports):
        warnings.append("saturated_readings_dropped")
    state.misfit = misfit
    state.swipes = len(swipes)
    state.history.append(misfit)

    cal = swipes[-1].calibration
    noise = float(np.sqrt(np.mean(cal.noise_ut**2)))
    half = float(np.max(np.abs(obs.pos[:, :2] - obs.pos[:, :2].mean(axis=0))))
    sug = suggest_next_swipe(
        sol,
        region,
        noise_ut=noise,
        b_limit_ut=SAFE_FRACTION * cal.range_ut,
        z_low=z_low,
        half_len=half,
        step=sf.bin_len,
    )
    moment = sol.total_moment
    m_sigma = np.sqrt(sol.linear_variance(np.tile(np.eye(3), (1, len(sol.moments)))))
    return {
        "label": "Estimate: a magnetisation pattern consistent with the measured field, "
        "not a view inside the magnet.",
        "swipes": len(swipes),
        "readings": int(len(obs)),
        "fit_seconds": round(time.perf_counter() - t0, 3),
        "relative_uncertainty": round(rel_unc, 4),
        "finest_detail_m": round(float(sol.grid.size), 4),  # NF-12
        "misfit_per_reading": round(misfit, 3),
        "warnings": warnings,
        "magnet": {
            "centre_m": _r(sf.dipole.pos),
            "moment_am2": _r(moment),
            "moment_sigma_am2": _r(m_sigma),  # known to be optimistic (Phase 1 open item)
        },
        "field": {
            "points_m": _r(rec.grid_points),
            "b_ut": _r(rec.field, 3),
            "sigma_ut": _r(rec.field_sigma, 3),
        },
        "field_lines_m": [_r(line) for line in rec.field_lines],
        "magnetisation": {
            "voxel_centres_m": _r(rec.voxel_centers),
            "voxel_size_m": round(float(rec.voxel_size), 5),
            "m_a_per_m": _r(rec.magnetisation, 1),
        },
        "next_swipe": None
        if sug is None
        else {
            "start_m": _r(sug.start),
            "end_m": _r(sug.end),
            "height_m": round(float(sug.height), 4),
            "variance_reduction": round(float(sug.variance_reduction), 3),
        },
        "inputs": [
            {"readings": r.readings, "kept": r.kept, "saturated": r.saturated} for r in reports
        ],
    }
