"""Point-dipole field kernel.

Internal units (the convention fixed at the capture boundary, SRS SC-17):
positions in metres, fields in microtesla, magnetic moments in A·m².
"""

from __future__ import annotations

import numpy as np

# mu0 / (4 pi) in T·m/A, times 1e6 to give microtesla.
K_UT = 1e-7 * 1e6


def dipole_kernel(obs: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Linear operator from dipole moments to field.

    obs: (N, 3) observation points. src: (M, 3) dipole positions.
    Returns G with shape (3N, 3M) so that B.ravel() = G @ m.ravel(),
    where B is (N, 3) in µT and m is (M, 3) in A·m².
    """
    obs = np.asarray(obs, dtype=float)
    src = np.asarray(src, dtype=float)
    r = obs[:, None, :] - src[None, :, :]  # (N, M, 3)
    d2 = np.einsum("nmi,nmi->nm", r, r)
    d = np.sqrt(d2)
    inv_d5 = 1.0 / (d2 * d2 * d)
    inv_d3 = 1.0 / (d2 * d)
    # B_i = K (3 r_i r_j / d^5 - delta_ij / d^3) m_j
    g = 3.0 * r[:, :, :, None] * r[:, :, None, :] * inv_d5[:, :, None, None]
    idx = np.arange(3)
    g[:, :, idx, idx] -= inv_d3[:, :, None]
    g *= K_UT
    # (N, M, 3i, 3j) -> (N, 3i, M, 3j) -> (3N, 3M)
    return g.transpose(0, 2, 1, 3).reshape(3 * obs.shape[0], 3 * src.shape[0])


def dipole_field(obs: np.ndarray, src: np.ndarray, moments: np.ndarray) -> np.ndarray:
    """Field (N, 3) in µT of dipoles at src (M, 3) with moments (M, 3)."""
    obs = np.atleast_2d(obs)
    src = np.atleast_2d(src)
    moments = np.atleast_2d(moments)
    return (dipole_kernel(obs, src) @ moments.ravel()).reshape(-1, 3)


def dipole_gradient_norm(obs: np.ndarray, src: np.ndarray, moments: np.ndarray) -> np.ndarray:
    """Approximate size of the field gradient |dB/dr| in µT/m at each observation point.

    Uses the dipole scaling |grad B| ~ 3 |B| / r from the nearest source, which is all
    the noise model needs.
    """
    obs = np.atleast_2d(obs)
    b = dipole_field(obs, src, moments)
    r = np.linalg.norm(obs[:, None, :] - np.atleast_2d(src)[None, :, :], axis=2).min(axis=1)
    return 3.0 * np.linalg.norm(b, axis=1) / r
