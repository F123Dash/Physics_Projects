import numpy as np
import pytest

from berry_phase import (
    H_spin,
    berry_phase,
    berry_phase_cone,
    berry_phase_discrete_exact,
    berry_phase_exact,
    chern_number,
    cone_states,
    discretisation_error,
    spin_state,
    wrap,
)

THETAS = [0.3, 0.9, 1.5, 2.2, 2.9]


def test_hamiltonian_is_hermitian_with_eigenvalues_pm_omega0_over_2():
    H = H_spin(0.7, 1.9, omega0=2.6)
    assert H.shape == (2, 2)
    assert np.allclose(H, H.conj().T)
    assert np.allclose(np.linalg.eigvalsh(H), [-1.3, 1.3])


@pytest.mark.parametrize("aligned,sign", [(True, 1), (False, -1)])
def test_spin_state_is_normalised_eigenvector(aligned, sign):
    u = spin_state(1.1, 0.4, aligned)
    assert np.isclose(np.vdot(u, u).real, 1.0)
    assert np.allclose(H_spin(1.1, 0.4, 2.0) @ u, sign * u)


def test_gauge_is_single_valued_in_phi():
    assert np.allclose(spin_state(0.8, 2 * np.pi), spin_state(0.8, 0.0))


def test_known_matrix_elements():
    assert np.allclose(H_spin(np.pi / 2, 0.0, 2.0), [[0, 1], [1, 0]])
    assert np.allclose(spin_state(0.0, 1.0), [1, 0])


@pytest.mark.parametrize("theta", THETAS)
def test_berry_phase_m400_matches_solid_angle(theta):
    err = wrap(berry_phase_cone(theta, 400) - berry_phase_exact(theta))
    assert abs(err) < 1e-4


@pytest.mark.parametrize("theta", THETAS)
@pytest.mark.parametrize("M", [20, 50, 400])
def test_discrete_formula_has_closed_form(theta, M):
    for aligned in (True, False):
        num = berry_phase_cone(theta, M, aligned)
        assert abs(wrap(num - berry_phase_discrete_exact(theta, M, aligned))) < 1e-12


@pytest.mark.parametrize("theta", [0.3, 0.9, 2.2, 2.9])
def test_discretisation_error_is_order_one_over_m_squared(theta):
    M = 400
    err = wrap(berry_phase_cone(theta, M) - berry_phase_exact(theta))
    assert np.isclose(err, discretisation_error(theta, M), rtol=1e-3)


@pytest.mark.parametrize("theta", THETAS)
def test_random_gauge_changes_nothing(theta):
    rng = np.random.default_rng(7)
    g = berry_phase_cone(theta, 50)
    for _ in range(5):
        assert abs(wrap(berry_phase_cone(theta, 50, rng=rng) - g)) < 1e-12


@pytest.mark.parametrize("theta", THETAS)
def test_reversed_loop_flips_sign(theta):
    u = cone_states(theta, 400)
    assert abs(wrap(berry_phase(u[::-1]) + berry_phase(u))) < 1e-12


@pytest.mark.parametrize("theta", THETAS)
def test_anti_aligned_state_has_opposite_phase(theta):
    g_up = berry_phase_cone(theta, 400, aligned=True)
    g_dn = berry_phase_cone(theta, 400, aligned=False)
    assert abs(wrap(g_up + g_dn)) < 1e-4


def test_berry_phase_vanishes_and_reaches_minus_2pi_at_the_poles():
    assert abs(berry_phase_cone(0.0, 100)) < 1e-14
    assert abs(wrap(berry_phase_cone(np.pi, 100))) < 1e-12


def test_chern_number_is_minus_one_for_aligned_and_plus_one_for_anti_aligned():
    assert abs(chern_number(aligned=True) + 1) < 1e-10
    assert abs(chern_number(aligned=False) - 1) < 1e-10


def test_chern_number_is_gauge_invariant_on_a_coarse_grid():
    rng = np.random.default_rng(3)
    assert abs(chern_number(6, 8, rng=rng) + 1) < 1e-10


def test_orthogonal_neighbours_raise():
    with pytest.raises(ValueError):
        berry_phase_cone(np.pi / 2, 2)
