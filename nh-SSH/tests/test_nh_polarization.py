import numpy as np
import pytest

from nh_ssh import (
    H_open,
    biorthogonal_density,
    obc_transitions,
    polarization,
    polarization_exact,
    polarization_limit,
    r_factors,
    zero_mode,
)

G = 4.0 / 3.0


def test_r_factors_match_paper():
    rR, rL = r_factors(1.0, 1.0, G)
    assert np.isclose(rR, -(1.0 - G / 2))
    assert np.isclose(rL, -(1.0 + G / 2))


@pytest.mark.parametrize("t1", [0.3, 0.9, 1.4])
def test_zero_mode_is_exact_and_on_A(t1):
    N = 15
    E, R, L = zero_mode(N, t1, 1.0, G)
    H = H_open(N - 1, t1, 1.0, G, extra_site=True)
    assert abs(E) < 1e-10
    assert np.allclose(H @ R, 0, atol=1e-10)
    assert np.sum(np.abs(R[1::2]) ** 2) < 1e-20
    rR, rL = r_factors(t1, 1.0, G)
    a = R[0::2]
    assert np.allclose(a[1:6] / a[:5], rR, rtol=1e-8)


def test_biorthogonal_density_closed_form():
    N, t1 = 20, 1.0
    _, R, L = zero_mode(N, t1, 1.0, G)
    rho = biorthogonal_density(R, L)
    rR, rL = r_factors(t1, 1.0, G)
    x = np.conj(rL) * rR
    n = np.arange(1, N + 1)
    assert np.isclose(np.sum(rho), 1.0)
    assert np.allclose(rho, x ** n / np.sum(x ** n), atol=1e-10)


@pytest.mark.parametrize("g", [4 / 3, 3.0])
def test_numerical_polarization_matches_closed_form(g):
    for N in (11, 21):
        for t1 in np.linspace(0.05, 2.0, 25):
            assert abs(polarization(N, t1, 1.0, g) - polarization_exact(N, t1, 1.0, g)) < 1e-6


def test_polarization_quantised_with_1_over_N_correction():
    """Quantisation with the leading finite-size term, x = r_L^* r_R:

    1 - P = 1 / (N (1 - x)) for |x| < 1 and P = 1 / (N (x - 1)) for x > 1, up to O(x^N).
    """
    N = 1001
    for t1 in [0.1, 0.5, 1.0, 1.1, 1.3, 1.5, 2.0]:
        x = (t1 * t1 - G * G / 4)
        P = polarization_exact(N, t1, 1.0, G)
        if abs(x) < 1:
            assert np.isclose((1 - P) * N, 1 / (1 - x), rtol=1e-9)
            assert polarization_limit(t1, 1.0, G) == 1.0
        else:
            assert np.isclose(P * N, 1 / (x - 1), rtol=1e-9)
            assert polarization_limit(t1, 1.0, G) == 0.0


def test_polarization_jumps_at_obc_point_not_bloch():
    """P changes at sqrt(t2^2 + g^2/4), and not at the Bloch values 1/3, 5/3."""
    tc = obc_transitions(1.0, G)[0]
    assert np.isclose(tc, 1.2019, atol=1e-4)
    N = 1001
    assert polarization_exact(N, tc - 0.02, 1.0, G) > 0.9
    assert polarization_exact(N, tc + 0.02, 1.0, G) < 0.1
    for tb in (1 / 3, 5 / 3):
        left = polarization_exact(N, tb - 0.05, 1.0, G)
        right = polarization_exact(N, tb + 0.05, 1.0, G)
        assert abs(left - right) < 1e-3
    assert np.isnan(polarization_limit(tc, 1.0, G))


def test_second_transition_for_large_gamma():
    """For g > 2 t2 there is a second OBC transition at sqrt(g^2/4 - t2^2)."""
    g = 3.0
    lo, hi = obc_transitions(1.0, g)
    assert np.isclose(lo, np.sqrt(1.25)) and np.isclose(hi, np.sqrt(3.25))
    assert polarization_limit(lo - 0.05, 1.0, g) == 0.0
    assert polarization_limit((lo + hi) / 2, 1.0, g) == 1.0
    assert polarization_limit(hi + 0.05, 1.0, g) == 0.0


def test_even_N_singular_at_x_minus_one():
    """At r_L^* r_R = -1 the norm sum_n x^n vanishes for even N; odd N gives P ~ 1/2."""
    g = 3.0
    t1 = np.sqrt(g * g / 4 - 1.0)
    assert np.isnan(polarization_exact(20, t1, 1.0, g))
    assert np.isclose(polarization_exact(21, t1, 1.0, g), 0.5 - 0.5 / 21)


def test_polarization_hermitian_limit():
    """g = 0: r_L = r_R, P = 1 for t1 < t2 (the SSH topological phase)."""
    assert polarization_limit(0.5, 1.0, 0.0) == 1.0
    assert polarization_limit(1.5, 1.0, 0.0) == 0.0
    assert abs(polarization(21, 0.5, 1.0, 0.0) - polarization_exact(21, 0.5, 1.0, 0.0)) < 1e-12
