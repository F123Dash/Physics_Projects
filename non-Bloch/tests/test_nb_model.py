import numpy as np

from nh_ssh import bloch_energies
from non_bloch import (
    H_obc,
    Laurent,
    circular_radius,
    hatano_nelson,
    hatano_nelson_open_spectrum,
    range2,
    ssh_h1,
    ssh_h2,
    ssh_lambda,
)


def test_laurent_trims_zero_ends():
    """A zero t_{-2} lowers the pole order, so p must drop with it."""
    ch = range2(0.0, 1.0, 0.5, 0.4)
    assert ch.p == 1 and ch.q == 2
    assert np.allclose(ch.coeffs, [1.0, 0.0, 0.5, 0.4])


def test_hatano_nelson_matrix_elements():
    H = H_obc(3, hatano_nelson(1.5, 0.5))
    assert np.allclose(H, [[0, 0.5, 0], [1.5, 0, 0.5], [0, 1.5, 0]])


def test_bulk_rows_reproduce_the_symbol():
    """Rows away from the ends act on beta^n as E(beta) beta^n: fixes H[n, n+k] = t_k."""
    ch = range2(0.3, 1.0, 0.5, 0.4)
    beta = 0.9 * np.exp(0.7j)
    N = 12
    psi = beta ** np.arange(N)
    out = H_obc(N, ch) @ psi
    assert np.allclose(out[2:-2], ch(beta) * psi[2:-2])


def test_gauge_is_a_similarity_transform():
    ch = range2(0.3, 1.0, 0.5, 0.4)
    rho, N = 1.1, 15
    D = np.diag(rho ** np.arange(N))
    assert np.allclose(H_obc(N, ch, rho), np.linalg.inv(D) @ H_obc(N, ch) @ D)
    E1 = np.linalg.eigvals(H_obc(N, ch))
    E2 = np.linalg.eigvals(H_obc(N, ch, rho))
    assert np.max(np.min(np.abs(E1[:, None] - E2[None, :]), axis=1)) < 1e-10


def test_hatano_nelson_open_spectrum_closed_form():
    E = np.linalg.eigvals(H_obc(10, hatano_nelson(1.5, 0.5)))
    Ex = hatano_nelson_open_spectrum(10, 1.5, 0.5)
    assert np.allclose(np.sort(E.real), np.sort(Ex.real), atol=1e-12)
    assert np.max(np.abs(E.imag)) < 1e-12


def test_ssh_lambda_is_h1_h2_and_matches_project_3():
    """E^2 of the nh_ssh Bloch bands equals lambda(e^{ik}): same convention as nh-SSH."""
    t1, t2, g = 0.7, 1.0, 4 / 3
    k = np.linspace(-np.pi, np.pi, 50)
    beta = np.exp(1j * k)
    lam = ssh_lambda(t1, t2, g)
    assert np.allclose(lam(beta), ssh_h1(t1, t2, g)(beta) * ssh_h2(t1, t2, g)(beta))
    assert np.allclose(bloch_energies(k, t1, t2, g)[:50] ** 2, lam(beta))


def test_circular_radius():
    assert np.isclose(circular_radius(hatano_nelson(1.5, 0.5)), np.sqrt(3))
    assert np.isclose(circular_radius(range2(0.3, 1.0, 0.5, 0.3 * 0.25)), np.sqrt(2))
    assert np.isnan(circular_radius(range2(0.3, 1.0, 0.5, 0.4)))
    assert np.isnan(circular_radius(Laurent([1.0, 0.0, 0.5, 0.4], 1)))
