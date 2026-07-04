import numpy as np
import pytest
from scipy.linalg import expm

from berry_phase import (
    H_spin,
    aharonov_anandan_phase,
    berry_phase_exact,
    geometric_phase,
    magnus4,
    nonadiabatic_slope,
    rotating_frame_angle,
    rotating_frame_exact,
    spin_geometric_phase,
    spin_geometric_phase_exact,
    spin_hamiltonian,
    spin_state,
    wrap,
)


def test_magnus_is_exact_for_constant_hamiltonian():
    H0 = H_spin(0.6, 1.2, 1.7)
    psi0 = np.array([0.6, 0.8j])
    psi = magnus4(lambda t: np.broadcast_to(H0, np.shape(t) + (2, 2)), psi0, 3.0, 7)
    assert np.allclose(psi, expm(-3j * H0) @ psi0, atol=1e-13)


def test_magnus_reproduces_rotating_frame_solution_to_1e8():
    theta, omega0, Omega = 0.9, 1.0, 0.1
    T = 2 * np.pi / Omega
    psi0 = spin_state(theta, 0.0)
    psi = magnus4(spin_hamiltonian(theta, omega0, Omega), psi0, T, 2000)
    assert np.linalg.norm(psi - rotating_frame_exact(theta, omega0, Omega, T, psi0)) < 1e-8
    assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-13)


def test_magnus_is_fourth_order():
    theta, Omega = 0.9, 0.3
    T = 2 * np.pi / Omega
    psi0 = spin_state(theta, 0.0)
    exact = rotating_frame_exact(theta, 1.0, Omega, T, psi0)
    err = [np.linalg.norm(magnus4(spin_hamiltonian(theta, 1.0, Omega), psi0, T, n) - exact)
           for n in (200, 400)]
    assert 14 < err[0] / err[1] < 18


@pytest.mark.parametrize("eps", [0.3, 0.1, 0.03])
@pytest.mark.parametrize("theta", [0.5, 0.9, 2.0])
def test_extracted_phase_matches_rotating_frame_closed_form(theta, eps):
    assert abs(wrap(spin_geometric_phase(theta, eps) - spin_geometric_phase_exact(theta, eps))) < 1e-6


@pytest.mark.parametrize("theta", [0.5, 0.9, 2.0])
def test_extracted_phase_tends_to_berry_phase_linearly(theta):
    err = [wrap(spin_geometric_phase(theta, e) - berry_phase_exact(theta)) for e in (0.004, 0.002)]
    assert np.isclose(err[0] / err[1], 2.0, rtol=0.02)
    assert np.isclose(err[1] / 0.002, nonadiabatic_slope(theta), rtol=0.01)


@pytest.mark.parametrize("eps", [0.05, 0.5, 2.0])
def test_aharonov_anandan_phase_is_exact_at_any_speed(eps):
    theta = 0.9
    T = 2 * np.pi / eps
    H = spin_hamiltonian(theta, 1.0, eps)
    thp = rotating_frame_angle(theta, eps)
    path = magnus4(H, spin_state(thp, 0.0), T, 4000, return_path=True)
    g = aharonov_anandan_phase(H, np.linspace(0, T, 4001), path)
    assert abs(wrap(g - berry_phase_exact(thp))) < 1e-9


def test_geometric_phase_is_nan_when_the_state_has_left():
    assert np.isnan(geometric_phase(np.array([1, 0]), np.array([0, 1]), 0.5, 1.0))
