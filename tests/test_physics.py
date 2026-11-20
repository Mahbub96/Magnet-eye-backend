"""Unit tests for the building blocks: dipole kernel, frames, binning."""

import magpylib as magpy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from app.pipeline.dipole import dipole_field, dipole_kernel
from app.pipeline.frames import to_world
from app.pipeline.observations import Observations, bin_along_swipes


def test_dipole_matches_magpylib():
    rng = np.random.default_rng(1)
    src = np.array([0.01, -0.02, 0.003])
    m = np.array([0.3, -1.2, 0.8])
    obs = rng.uniform(-0.1, 0.1, (20, 3)) + np.array([0, 0, 0.15])
    ref = magpy.getB(magpy.misc.Dipole(position=src, moment=m), obs) * 1e6
    np.testing.assert_allclose(dipole_field(obs, src, m), ref, rtol=1e-9)


def test_dipole_field_is_curl_and_divergence_free():
    src, m = np.zeros((1, 3)), np.array([[0.2, 0.5, 1.0]])
    p, h = np.array([0.03, -0.02, 0.05]), 1e-6
    jac = np.empty((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        jac[:, j] = (dipole_field(p + e, src, m) - dipole_field(p - e, src, m))[0] / (2 * h)
    scale = np.abs(jac).max()
    assert abs(np.trace(jac)) < 1e-6 * scale  # div B = 0
    np.testing.assert_allclose(jac, jac.T, atol=1e-6 * scale)  # curl B = 0


def test_kernel_shape_and_linearity():
    obs = np.array([[0, 0, 0.05], [0.02, 0, 0.05]])
    src = np.array([[0, 0, 0], [0.01, 0.01, 0]])
    g = dipole_kernel(obs, src)
    assert g.shape == (6, 6)
    m = np.array([[0, 0, 1.0], [1.0, 0, 0]])
    np.testing.assert_allclose(
        (g @ m.ravel()).reshape(-1, 3),
        dipole_field(obs, src[:1], m[:1]) + dipole_field(obs, src[1:], m[1:]),
    )


def test_to_world_rotation_order_and_sensor_offset():
    # Phone yawed +90 degrees: phone x axis points along world y.
    rot = Rotation.from_euler("z", 90, degrees=True)
    q_xyzw = rot.as_quat()
    q_wxyz = np.array([[q_xyzw[3], *q_xyzw[:3]]])
    pos, b = to_world(
        np.array([[10.0, 0.0, 0.0]]), q_wxyz, np.array([[1.0, 2.0, 3.0]]), np.array([0.0, 0.06, 0])
    )
    np.testing.assert_allclose(b, [[0.0, 10.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(pos, [[1.0 - 0.06, 2.0, 3.0]], atol=1e-12)


def _obs(n, sigma=1.0):
    pos = np.stack([np.linspace(0, 0.1, n), np.zeros(n), np.zeros(n)], axis=1)
    return Observations(
        pos=pos,
        b=np.zeros((n, 3)),
        swipe=np.zeros(n, int),
        sigma_b=np.full(n, sigma),
        sigma_pos=np.full(n, 0.002),
        sigma_rot=np.full(n, 0.01),
        bias_pos=np.full(n, 0.003),
        bias_rot=np.full(n, 0.02),
        bias_b=np.full(n, 2.0),
        count=np.ones(n),
    )


def test_binning_averages_random_noise_but_not_bias():
    obs = _obs(101)
    out = bin_along_swipes(obs, 0.0101)  # ~10 readings per bin
    assert out.count.sum() == 101
    k = np.argmax(out.count)
    n = out.count[k]
    assert out.sigma_b[k] == pytest.approx(1.0 / np.sqrt(n))
    assert out.sigma_pos[k] == pytest.approx(0.002 / np.sqrt(n))
    assert out.bias_pos[k] == pytest.approx(0.003)
    assert out.bias_b[k] == pytest.approx(2.0)
