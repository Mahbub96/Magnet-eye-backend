"""Any-shape inverse solve: magnetisation of a voxel grid of dipoles (PR-7, PR-16, PR-17).

Model:  b = G m + e
  G   dipole kernel from every voxel to every reading (div B = curl B = 0 outside the source
      holds by construction, because every voxel is a dipole)
  m   3 free moment components per voxel: no shape, direction or symmetry assumed
  e   random noise (sensor, tracking jitter, orientation jitter; per reading) plus errors
      shared by a whole swipe: a position offset, an orientation offset and a constant field
      offset (leftover hard-iron and baseline error). The shared part is modelled as noise
      correlated along the swipe, C_s = diag(sigma^2) + J_s P J_s^T, and the data are whitened
      with it. This equals fitting a free offset per swipe under a Gaussian prior whose size
      comes from calibration and tracking confidence.

Regularisers, both with sensitivity (depth) weighting so moments are not pulled toward the
voxels nearest the phone:
  "tikhonov"  smallest weighted moment norm, closed form with one SVD (default)
  "sparse"    group sparsity (fewest active voxels), by IRLS

Strength rules: "gcv" (generalised cross-validation, default: estimates the noise level from
the data, on top of the measured per-reading weights), "evidence" (marginal likelihood over
noise and prior scale) and "discrepancy" (misfit equal to the modelled noise). Phase 1 results
are in docs/phase1-results.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, solve_triangular
from scipy.optimize import brentq, minimize

from .dipole import dipole_kernel
from .locate import VoxelGrid
from .observations import Observations


@dataclass
class Posterior:
    """What is needed to predict fields and their uncertainty from a solution.

    The solved variable is y, with moments m = col_scale * (V-space solution). The Gaussian
    posterior covariance of y is V diag(1/(S^2+lam)) V^T + (I - V V^T) / lam, times s2.
    """

    v: np.ndarray  # (n, k)
    s: np.ndarray  # (k,)
    lam: float
    col_scale: np.ndarray  # (n,)
    s2: float  # variance inflation when the fit is worse than the noise model predicts


@dataclass
class Solution:
    method: str
    grid: VoxelGrid
    moments: np.ndarray  # (M, 3) A·m²
    background: np.ndarray  # (3,) µT
    lam: float
    lam_rule: str
    chi2: float
    n_data: int
    posterior: Posterior

    @property
    def total_moment(self) -> np.ndarray:
        return self.moments.sum(axis=0)

    def field(self, points: np.ndarray) -> np.ndarray:
        """Magnet field (no background) at points (T, 3), µT."""
        g = dipole_kernel(points, self.grid.centers)
        return (g @ self.moments.ravel()).reshape(-1, 3)

    def field_sigma(self, points: np.ndarray) -> np.ndarray:
        """One-sigma uncertainty of each field component at points (T, 3), µT."""
        return np.sqrt(self.linear_variance(dipole_kernel(points, self.grid.centers))).reshape(
            -1, 3
        )

    def linear_variance(self, op: np.ndarray) -> np.ndarray:
        """Posterior variance of op @ m.ravel() for each row of op."""
        p = self.posterior
        f = op * p.col_scale[None, :]
        fv = f @ p.v
        in_range = np.sum(fv**2 / (p.s**2 + p.lam)[None, :], axis=1)
        null = (np.sum(f**2, axis=1) - np.sum(fv**2, axis=1)) / p.lam
        return p.s2 * (in_range + np.maximum(null, 0.0))


@dataclass
class SwipeBias:
    """Correlated error of one swipe: rows (indices into obs), sensitivity J (3n, k) of
    those readings to k offset parameters, and the prior variance (k,) of each parameter."""

    rows: np.ndarray
    jac: np.ndarray
    prior_var: np.ndarray


def _whiten(blocks: list[SwipeBias] | None, w: np.ndarray, *mats: np.ndarray):
    if not blocks:
        return mats
    out = [m.copy() for m in mats]
    for blk in blocks:
        r = (3 * blk.rows[:, None] + np.arange(3)[None, :]).ravel()
        jw = blk.jac * w[r][:, None]
        c = np.eye(r.size) + (jw * blk.prior_var[None, :]) @ jw.T
        lower, _ = cho_factor(c, lower=True)
        for o in out:
            o[r] = solve_triangular(lower, o[r], lower=True)
    return out


def _weighted_system(
    obs: Observations,
    sigma: np.ndarray,
    grid: VoxelGrid,
    bias: list[SwipeBias] | None = None,
    free_background: bool = True,
):
    w = np.repeat(1.0 / sigma, 3)
    gw = dipole_kernel(obs.pos, grid.centers) * w[:, None]
    gbw = np.tile(np.eye(3), (len(obs), 1)) * w[:, None]
    dw = obs.b.ravel() * w
    gw, gbw, dw = _whiten(bias, w, gw, gbw, dw)
    if free_background:
        q, _ = np.linalg.qr(gbw)  # background columns, projected out (not penalised)
        gp = gw - q @ (q.T @ gw)
        dp = dw - q @ (q.T @ dw)
    else:
        gp, dp = gw, dw
    # Sensitivity weighting: (sum of squared sensitivities)^(1/4) per voxel ~ depth^-1.5
    sens = np.sqrt(np.sum(gp**2, axis=0).reshape(-1, 3).sum(axis=1) / 3.0)
    wcol = np.repeat(np.sqrt(sens), 3)
    return gw, gbw, dw, gp, dp, wcol


def _evidence(s2: np.ndarray, beta: np.ndarray, r_perp2: float, n_eff: int):
    """Empirical Bayes: maximise the marginal likelihood of the data over the noise scale
    (sigma^2, multiplying the measured noise) and the prior scale (tau^2) together.

    Model in the whitened SVD basis: beta_i ~ N(0, sigma^2 + tau^2 s_i^2) for the k fitted
    directions, and the remaining n_eff - k directions are pure noise. Because both scales come
    from the data, the posterior covariance is consistent with the actual misfit, which is
    what makes the reported uncertainty honest (pass bar item 4).
    """
    k = s2.size
    n_rest = max(n_eff - k, 0)
    b2 = beta**2

    def nll(t):
        sig2, tau2 = np.exp(t)
        v = sig2 + tau2 * s2
        out = np.sum(np.log(v) + b2 / v)
        if n_rest:
            out += n_rest * np.log(sig2) + r_perp2 / sig2
        return float(out)

    best = None
    sig0 = (r_perp2 + b2.sum()) / max(n_eff, 1)
    for tau0 in (1e-6, 1e-3, 1.0):
        t0 = np.log([sig0, tau0 * b2.sum() / max(s2.sum(), 1e-300)])
        r = minimize(nll, t0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6})
        if best is None or r.fun < best.fun:
            best = r
    sig2, tau2 = np.exp(best.x)
    return float(sig2 / tau2), float(sig2)


def _choose_lambda(s: np.ndarray, beta: np.ndarray, r_perp2: float, n_eff: int, rule: str):
    """Returns (lambda, rule actually used, noise scale or None)."""
    s2 = s**2

    def misfit(lam):
        return float(np.sum((lam / (s2 + lam)) ** 2 * beta**2) + r_perp2)

    if rule == "evidence":
        lam, sig2 = _evidence(s2, beta, r_perp2, n_eff)
        return lam, "evidence", sig2
    lo, hi = s2.max() * 1e-12, s2.max() * 1e3
    if rule == "discrepancy" and misfit(lo) < n_eff < misfit(hi):
        lam = float(np.exp(brentq(lambda t: misfit(np.exp(t)) - n_eff, np.log(lo), np.log(hi))))
        return lam, "discrepancy", None
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 200))
    gcv = [misfit(lam) / (n_eff - np.sum(s2 / (s2 + lam))) ** 2 for lam in grid]
    return float(grid[int(np.argmin(gcv))]), "gcv", None


def _tikhonov(a, d, n_eff, rule, lam=None):
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    beta = u.T @ d
    r_perp2 = max(float(d @ d - beta @ beta), 0.0)
    used, sig2 = "given", None
    if lam is None:
        lam, used, sig2 = _choose_lambda(s, beta, r_perp2, n_eff, rule)
    y = vt.T @ (s / (s**2 + lam) * beta)
    return y, s, vt.T, lam, used, sig2


def solve(
    obs: Observations,
    sigma: np.ndarray,
    grid: VoxelGrid,
    method: str = "tikhonov",
    lam_rule: str = "gcv",
    irls_iters: int = 12,
    bias: list[SwipeBias] | None = None,
) -> Solution:
    """Solve for voxel moments. With `bias` (per-swipe offsets including a field offset) the
    background is part of the correlated noise; without it a free uniform background is fitted.
    """
    free_bg = bias is None
    gw, gbw, dw, gp, dp, wcol = _weighted_system(obs, sigma, grid, bias, free_bg)
    n_eff = 3 * len(obs) - (3 if free_bg else 0)
    a = gp / wcol[None, :]

    y, s, v, lam, used, sig2 = _tikhonov(a, dp, n_eff, lam_rule)
    col_scale = 1.0 / wcol

    if method == "sparse":
        # IRLS for  min |a x - d|^2 + lam * sum_j |x_j|  (groups of 3 = one voxel).
        # Each step is a re-weighted Tikhonov problem; the strength is re-chosen each time so
        # the final misfit still matches the noise level.
        x = y
        for _ in range(irls_iters):
            gnorm = np.linalg.norm(x.reshape(-1, 3), axis=1)
            eps = 1e-3 * gnorm.max() + 1e-30
            dscale = np.repeat(np.sqrt(gnorm + eps), 3)  # x = dscale * z
            z, s, v, lam, used, sig2 = _tikhonov(a * dscale[None, :], dp, n_eff, lam_rule)
            x_new = dscale * z
            done = np.linalg.norm(x_new - x) <= 1e-4 * np.linalg.norm(x_new)
            x = x_new
            if done:
                break
        y = z
        col_scale = dscale / wcol
    elif method != "tikhonov":
        raise ValueError(f"unknown method {method!r}")

    m = col_scale * y
    resid = gp @ m - dp
    chi2 = float(resid @ resid)
    k = s.size
    dof_used = float(np.sum(s**2 / (s**2 + lam))) if k else 0.0
    if sig2 is not None:
        s2 = sig2  # noise scale estimated together with the prior (evidence rule)
    else:
        s2 = max(1.0, chi2 / max(n_eff - dof_used, 1.0))
    if free_bg:
        background, *_ = np.linalg.lstsq(gbw, dw - gw @ m, rcond=None)
    else:
        background = np.zeros(3)
    return Solution(
        method=method,
        grid=grid,
        moments=m.reshape(-1, 3),
        background=background,
        lam=lam,
        lam_rule=used,
        chi2=chi2,
        n_data=n_eff,
        posterior=Posterior(v=v, s=s, lam=lam, col_scale=col_scale, s2=s2),
    )
