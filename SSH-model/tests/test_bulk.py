import numpy as np
import pytest

from ssh_model import H_k, bands, k_grid, winding_number, zak_phase


@pytest.mark.parametrize("v,w", [(0.5, 1.0), (1.5, 1.0), (0.2, 0.7), (2.0, 0.3)])
def test_bands_match_diagonalisation(v, w):
    ks = k_grid(64)
    num = np.array([np.linalg.eigvalsh(H_k(k, v, w)) for k in ks])
    em, ep = bands(ks, v, w)
    assert np.allclose(num[:, 0], em) and np.allclose(num[:, 1], ep)


def test_gap_is_2_abs_v_minus_w():
    for v in [0.3, 0.8, 1.4]:
        em, ep = bands(k_grid(2000), v, 1.0)
        assert np.isclose(np.min(ep - em), 2 * abs(v - 1.0), atol=1e-6)


def test_chiral_symmetry_bloch():
    G = np.diag([1.0, -1.0])
    for k in k_grid(17):
        H = H_k(k, 0.6, 1.0)
        assert np.allclose(G @ H @ G, -H)


@pytest.mark.parametrize("v", [0.0, 0.1, 0.5, 0.9, 0.99])
def test_winding_one_for_v_less_than_w(v):
    nu = winding_number(v, 1.0)
    assert abs(nu - 1.0) < 1e-6


@pytest.mark.parametrize("v", [1.01, 1.1, 1.5, 3.0])
def test_winding_zero_for_v_greater_than_w(v):
    assert abs(winding_number(v, 1.0)) < 1e-6


def test_winding_undefined_at_gap_closing():
    assert np.isnan(winding_number(1.0, 1.0))


@pytest.mark.parametrize("v,expected", [(0.3, np.pi), (0.8, np.pi), (1.2, 0.0), (2.5, 0.0)])
def test_zak_phase(v, expected):
    gamma = zak_phase(v, 1.0)
    assert abs(np.angle(np.exp(1j * (gamma - expected)))) < 1e-6


@pytest.mark.parametrize("v", [0.4, 1.6])
def test_zak_phase_gauge_invariant(v):
    rng = np.random.default_rng(1)
    g0 = zak_phase(v, 1.0)
    for _ in range(5):
        g = zak_phase(v, 1.0, random_gauge=rng)
        assert abs(np.angle(np.exp(1j * (g - g0)))) < 1e-10
