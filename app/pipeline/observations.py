"""World-frame observations and their preparation for the solver.

After `frames.to_world`, a swipe is a list of sensor positions and world-frame field
readings. This module bins them along each swipe (so the solver stays small enough for a
1 GB server) and attaches a per-reading noise level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Observations:
    """Readings in the world frame, baseline already subtracted.

    pos: (N, 3) sensor positions, m. b: (N, 3) field, µT. swipe: (N,) swipe index.
    sigma_b: (N,) magnetometer noise per axis, µT (from calibration).
    sigma_pos: (N,) random position jitter, m (from tracking confidence).
    sigma_rot: (N,) random orientation jitter, rad.
    bias_pos: (N,) expected size of a position offset shared by the whole swipe, m.
    bias_rot: (N,) expected size of an orientation offset shared by the whole swipe, rad.
    bias_b: (N,) expected size of a constant field offset shared by the whole swipe, µT
        (leftover hard-iron and baseline error, from calibration).
    count: (N,) raw readings averaged into each row (1 before binning).

    Random parts average down when readings are binned; shared (bias) parts do not, and are
    handled by the solver as noise that is correlated along each swipe.
    """

    pos: np.ndarray
    b: np.ndarray
    swipe: np.ndarray
    sigma_b: np.ndarray
    sigma_pos: np.ndarray
    sigma_rot: np.ndarray
    bias_pos: np.ndarray
    bias_rot: np.ndarray
    bias_b: np.ndarray
    count: np.ndarray

    def __len__(self) -> int:
        return self.pos.shape[0]

    @staticmethod
    def concat(items: list[Observations]) -> Observations:
        return Observations(
            *(np.concatenate([getattr(o, f) for o in items]) for f in Observations.fields())
        )

    @staticmethod
    def fields() -> tuple[str, ...]:
        return (
            "pos",
            "b",
            "swipe",
            "sigma_b",
            "sigma_pos",
            "sigma_rot",
            "bias_pos",
            "bias_rot",
            "bias_b",
            "count",
        )


def bin_along_swipes(obs: Observations, bin_len: float) -> Observations:
    """Average consecutive readings of each swipe into path-length bins of `bin_len` metres.

    Random noise (sensor, position jitter, orientation jitter) averages down by sqrt(count);
    the shared per-swipe biases do not.
    """
    out = []
    for s in np.unique(obs.swipe):
        idx = np.flatnonzero(obs.swipe == s)
        p = obs.pos[idx]
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        path = np.concatenate([[0.0], np.cumsum(step)])
        key = np.floor(path / bin_len).astype(int)
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        nb = cnt.size

        def mean(a, inv=inv, cnt=cnt, nb=nb):
            acc = np.zeros((nb,) + a.shape[1:])
            np.add.at(acc, inv, a)
            return acc / cnt.reshape((-1,) + (1,) * (a.ndim - 1))

        n_raw = obs.count[idx]
        total = np.zeros(nb)
        np.add.at(total, inv, n_raw)
        out.append(
            Observations(
                pos=mean(p),
                b=mean(obs.b[idx]),
                swipe=np.full(nb, s),
                sigma_b=np.sqrt(mean(obs.sigma_b[idx] ** 2) / cnt),
                sigma_pos=np.sqrt(mean(obs.sigma_pos[idx] ** 2) / cnt),
                sigma_rot=np.sqrt(mean(obs.sigma_rot[idx] ** 2) / cnt),
                bias_pos=mean(obs.bias_pos[idx]),
                bias_rot=mean(obs.bias_rot[idx]),
                bias_b=mean(obs.bias_b[idx]),
                count=total,
            )
        )
    return Observations.concat(out)


def noise_sigma(obs: Observations, b_model: np.ndarray, grad_model: np.ndarray) -> np.ndarray:
    """Per-row standard deviation in µT of the RANDOM (uncorrelated) error.

    sigma^2 = sensor noise^2 + (|grad B| * position jitter)^2 + (|B| * orientation jitter)^2.
    b_model / grad_model come from the coarse dipole fit, so strong-gradient readings close to
    the magnet are trusted less, which is what keeps the solver honest near the magnet.
    """
    bmag = np.linalg.norm(b_model, axis=1)
    return np.sqrt(obs.sigma_b**2 + (grad_model * obs.sigma_pos) ** 2 + (bmag * obs.sigma_rot) ** 2)
