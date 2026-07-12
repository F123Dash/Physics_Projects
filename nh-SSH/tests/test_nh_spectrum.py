import numpy as np
import pytest

from nh_ssh import (
    H_open,
    biorthogonal_eig,
    chiral_edge_states,
    eigenvector_condition,
    hermitian_edge_states,
    min_abs_energy,
    obc_transitions,
    similarity_diagonal,
    skin_decay_factor,
    skin_factor,
    spectrum_error,
    v_prime,
)

G = 4.0 / 3.0


def test_biorthonormal_eigenvectors():
    H = H_open(10, 0.9, 1.0, G)
    E, R, L = biorthogonal_eig(H)
    assert np.allclose(L.conj().T @ R, np.eye(20), atol=1e-8)
    assert np.allclose(H @ R, R * E)
    assert np.allclose(H.conj().T @ L, L * E.conj())


def test_brief_min_abs_energy_table():
    assert min_abs_energy(40, 0.5, 1.0, G) < 1e-12
    assert np.allclose([min_abs_energy(N, 1.0, 1.0, G) for N in (10, 20, 30)],
                       [2.378e-2, 1.245e-3, 6.588e-5], rtol=1e-3)
    assert np.allclose([min_abs_energy(N, 1.4, 1.0, G) for N in (10, 20, 30)],
                       [0.3367, 0.2703, 0.2515], rtol=1e-3)


@pytest.mark.parametrize("t1", [0.5, 0.9, 1.1])
def test_min_E_decays_exponentially_below_transition(t1):
    """Smallest |E| falls exponentially in N for t1 < sqrt(t2^2 + g^2/4).

    Slope ln(v'/t2). N >= 20 because xi = 7.5 cells at t1 = 1.1; points below 1e-13 are round-off.
    """
    Ns = np.arange(20, 37)
    eps = np.array([min_abs_energy(N, t1, 1.0, G) for N in Ns])
    ok = eps > 1e-13
    slope, _ = np.polyfit(Ns[ok], np.log(eps[ok]), 1)
    expected = np.log(abs(v_prime(t1, G)))
    assert abs(slope / expected - 1) < 0.02


@pytest.mark.parametrize("t1", [1.3, 1.4, 1.6])
def test_min_E_saturates_above_transition(t1):
    eps = [min_abs_energy(N, t1, 1.0, G) for N in (20, 30, 36)]
    gap = abs(v_prime(t1, G).real - 1.0)
    assert np.all(np.array(eps) > 0.9 * gap)
    assert abs(eps[-1] / eps[0] - 1) < 0.2


@pytest.mark.parametrize("g", [0.5, 4 / 3, 2.0, 2.6])
def test_transition_follows_sqrt_formula(g):
    """At t1 = sqrt(t2^2 + g^2/4) min|E| scales as 1/N (critical), not exponentially.

    The partner is then a uniform chain of 2N sites with min|E| = 2 sin(pi / (2(2N + 1))).
    """
    tc = obc_transitions(1.0, g)[-1]
    for N in (10, 20, 30):
        assert np.isclose(min_abs_energy(N, tc, 1.0, g), 2 * np.sin(np.pi / (2 * (2 * N + 1))), rtol=1e-8)
    below = min_abs_energy(30, tc - 0.1, 1.0, g) / min_abs_energy(20, tc - 0.1, 1.0, g)
    above = min_abs_energy(30, tc + 0.1, 1.0, g) / min_abs_energy(20, tc + 0.1, 1.0, g)
    assert below < 0.5 < 41 / 61 < above


def test_bloch_values_are_not_transitions():
    """Negative control: nothing happens to min|E| at the Bloch gap closings 1/3 and 5/3."""
    for t1 in (1 / 3, 5 / 3):
        r = [min_abs_energy(N, t1, 1.0, G) for N in (20, 30)]
        crit = 2 * np.sin(np.pi / (2 * 61)) / (2 * np.sin(np.pi / (2 * 41)))
        assert abs(r[1] / r[0] - crit) > 0.3


@pytest.mark.parametrize("t1", [0.7, 1.0, 1.4])
def test_skin_decay_factor(t1):
    """Bulk right eigenvectors decay by 1/|s| per cell within 3% (N = 12)."""
    s = abs(skin_factor(t1, G))
    assert abs(skin_decay_factor(12, t1, 1.0, G, "right") * s - 1) < 0.03
    assert abs(skin_decay_factor(12, t1, 1.0, G, "left") / s - 1) < 0.03


def test_right_eigenvectors_are_scaled_hermitian_ones():
    """R_n / psi_n = const * D_n exactly, for every eigenvector (N = 12, t1 = 0.7)."""
    N, t1 = 12, 0.7
    _, R, _ = biorthogonal_eig(H_open(N, t1, 1.0, G))
    d = similarity_diagonal(2 * N, t1, G).real
    psi = R / d[:, None]
    psi = psi / np.linalg.norm(psi, axis=0)
    gram = np.abs(psi.conj().T @ psi)
    assert np.allclose(gram, np.eye(2 * N), atol=1e-8)


def test_hermitian_limit_has_no_skin_effect():
    """Negative control: at g = 0, s = 1 and the eigenvector matrix is unitary."""
    assert np.isclose(abs(skin_factor(1.0, 0.0)), 1.0)
    assert eigenvector_condition(30, 0.7, 1.0, 0.0) < 1 + 1e-10


@pytest.mark.parametrize("t1", [0.8, 1.0, 1.1])
def test_biorthogonal_edge_density_matches_hermitian(t1):
    """L^* R of the two edge states equals the Hermitian SSH(v', t2) density to 1e-6."""
    N = 20
    (RA, LA), (RB, LB) = chiral_edge_states(N, t1, 1.0, G)
    pA, pB = hermitian_edge_states(N, t1, 1.0, G)
    assert np.max(np.abs(np.conj(LA) * RA - pA ** 2)) < 1e-6
    assert np.max(np.abs(np.conj(LB) * RB - pB ** 2)) < 1e-6


def test_right_and_left_edge_vectors_sit_at_opposite_ends():
    """For t1 > g/2 both R's pile up on the left and both L's on the right."""
    N = 20
    (RA, LA), (RB, LB) = chiral_edge_states(N, 1.0, 1.0, G)
    for R in (RA, RB):
        assert np.argmax(np.abs(R)) < 4
    for L in (LA, LB):
        assert np.argmax(np.abs(L)) > 2 * N - 5


def test_eig_accurate_below_76_sites():
    for t1 in (0.7, 1.0, 1.4):
        assert spectrum_error(37, t1, 1.0, G) < 1e-11


def test_eig_fails_at_large_N():
    """At N = 60, t1 = 0.7 the eig spectrum is off by > 1e-4."""
    assert spectrum_error(60, 0.7, 1.0, G) > 1e-4
    assert spectrum_error(60, 1.0, 1.0, 0.0) < 1e-12
