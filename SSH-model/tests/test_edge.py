import numpy as np
import pytest

from ssh_model import (
    H_open,
    chiral_operator,
    edge_splitting,
    edge_states,
    fit_xi,
    splitting_prediction,
    xi_exact,
)


def test_open_chain_structure():
    H = H_open(3, 0.5, 1.0)
    assert H.shape == (6, 6)
    assert np.allclose(H, H.T)
    assert np.allclose(np.diag(H, 1), [0.5, 1.0, 0.5, 1.0, 0.5])


def test_open_chain_chiral_symmetry():
    H = H_open(40, 0.7, 1.0)
    G = np.diag(chiral_operator(40))
    E = np.linalg.eigvalsh(H)
    assert np.allclose(G @ H @ G, -H)
    assert np.allclose(np.sort(E), np.sort(-E))


@pytest.mark.parametrize("v", [0.2, 0.3, 0.5])
def test_zero_modes_topological(v):
    assert edge_splitting(60, v, 1.0) < 1e-10


@pytest.mark.parametrize("v", [1.3, 1.5, 2.0])
def test_no_zero_modes_trivial(v):
    eps = edge_splitting(60, v, 1.0)
    assert 0.9 * abs(v - 1.0) < eps < 1.5 * abs(v - 1.0)


def test_edge_states_sublattice_polarised():
    N = 60
    psi_A, psi_B = edge_states(N, 0.5, 1.0)
    assert np.sum(psi_A[1::2] ** 2) < 1e-12
    assert np.sum(psi_B[0::2] ** 2) < 1e-12
    assert np.argmax(np.abs(psi_A)) < 4
    assert np.argmax(np.abs(psi_B)) > 2 * N - 5


@pytest.mark.parametrize("v,w", [(0.3, 1.0), (0.5, 1.0), (0.7, 1.0), (0.6, 1.5)])
def test_xi_matches_analytic(v, w):
    xi, _, _ = fit_xi(100, v, w)
    assert abs(xi / xi_exact(v, w) - 1.0) < 0.02


def test_edge_amplitude_ratio():
    psi_A, _ = edge_states(80, 0.5, 1.0)
    a = psi_A[0::2][:15]
    assert np.allclose(a[1:] / a[:-1], -0.5, rtol=1e-6)


@pytest.mark.parametrize("v,w", [(0.5, 1.0), (0.3, 1.0), (1.0, 2.0)])
def test_splitting_semilog_slope(v, w):
    Ns = np.arange(6, 31)
    eps = np.array([edge_splitting(N, v, w) for N in Ns])
    slope, _ = np.polyfit(Ns, np.log(eps), 1)
    assert abs(slope / np.log(v / w) - 1.0) < 0.01


@pytest.mark.parametrize("v,w", [(0.5, 1.0), (0.3, 1.0), (1.0, 2.0), (0.6, 1.5)])
def test_splitting_prefactor(v, w):
    for N in [10, 20, 30]:
        assert np.isclose(edge_splitting(N, v, w), splitting_prediction(N, v, w), rtol=1e-3)
