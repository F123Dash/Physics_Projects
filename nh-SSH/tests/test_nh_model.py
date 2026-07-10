import numpy as np
import pytest

from nh_ssh import (
    H_bloch,
    H_open,
    H_partner,
    bloch_energies,
    bloch_gap_closings,
    chiral_operator,
    exact_open_spectrum,
    similarity_diagonal,
    spectrum_error,
)

G = 4.0 / 3.0


def test_open_chain_matrix_elements():
    H = H_open(2, 0.8, 1.0, G)
    up, down = 0.8 + G / 2, 0.8 - G / 2
    expected = np.array([
        [0, up, 0, 0],
        [down, 0, 1, 0],
        [0, 1, 0, up],
        [0, 0, down, 0],
    ])
    assert np.allclose(H, expected)


def test_extra_site_chain_has_odd_length():
    H = H_open(3, 0.8, 1.0, G, extra_site=True)
    assert H.shape == (7, 7)
    assert np.isclose(H[5, 6], 1.0) and np.isclose(H[6, 5], 1.0)


def test_hermitian_limit():
    H = H_open(10, 0.7, 1.0, 0.0)
    assert np.allclose(H, H.T)


def test_chiral_symmetry_and_pm_E_pairs():
    H = H_open(15, 0.9, 1.0, G)
    Gam = np.diag(chiral_operator(30))
    assert np.allclose(Gam @ H @ Gam, -H)
    E = exact_open_spectrum(15, 0.9, 1.0, G)
    assert np.allclose(np.sort(E.real), np.sort(-E.real))


def test_bloch_matches_fourier_transform_of_ring():
    """H(k) from closing the open chain into a ring and Fourier transforming over cells."""
    N, t1 = 8, 0.6
    H = H_open(N, t1, 1.0, G).astype(complex)
    H[0, 2 * N - 1] = H[2 * N - 1, 0] = 1.0
    for m in range(N):
        k = 2 * np.pi * m / N
        phase = np.repeat(np.exp(1j * k * np.arange(N)), 2) / np.sqrt(N)
        U = np.zeros((2 * N, 2), dtype=complex)
        U[0::2, 0] = phase[0::2]
        U[1::2, 1] = phase[1::2]
        assert np.allclose(U.conj().T @ H @ U, H_bloch(k, t1, 1.0, G))


def test_bloch_gap_closings():
    assert np.allclose(bloch_gap_closings(1.0, G), [1 / 3, 5 / 3])
    for t1 in bloch_gap_closings(1.0, G):
        E = bloch_energies(np.linspace(-np.pi, np.pi, 2001), t1, 1.0, G)
        assert np.min(np.abs(E)) < 1e-3


def test_similarity_transform_is_exact():
    N, t1 = 6, 0.9
    d = similarity_diagonal(2 * N, t1, G)
    assert np.allclose(d[:, None] * H_partner(N, t1, 1.0, G) / d[None, :], H_open(N, t1, 1.0, G))


@pytest.mark.parametrize("t1", [0.8, 1.0, 1.4])
def test_open_spectrum_equals_hermitian_ssh(t1):
    """Eigenvalues of the open chain equal SSH(v', t2) to 1e-10 for N <= 20."""
    for N in range(2, 21):
        assert spectrum_error(N, t1, 1.0, G) < 1e-10


def test_imaginary_v_prime_partner_against_mpmath():
    """For t1 < g/2 the partner is complex symmetric; check eigvals against 30-digit mpmath."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 30
    N, t1 = 6, 0.4
    up, down = mp.mpf(t1) + mp.mpf(2) / 3, mp.mpf(t1) - mp.mpf(2) / 3
    H = mp.zeros(2 * N)
    for i in range(2 * N - 1):
        H[i, i + 1], H[i + 1, i] = (up, down) if i % 2 == 0 else (1, 1)
    E_mp = np.array([complex(e) for e in mp.eig(H, left=False, right=False)])
    E = exact_open_spectrum(N, t1, 1.0, G)
    assert np.max(np.min(np.abs(E[:, None] - E_mp[None, :]), axis=1)) < 1e-12
    assert np.max(np.abs(E_mp.imag)) > 0.1
