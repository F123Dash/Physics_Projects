"""Time evolution i d(psi)/dt = H(t) psi, and phases extracted from it (hbar = 1)."""

import numpy as np
from scipy.linalg import expm

from .geometry import wrap
from .spin import SZ, H_spin, spin_state

GAUSS = np.sqrt(3) / 6


def magnus4(H, psi0, T, steps, return_path=False):
    """Fourth-order Magnus integrator with two Gauss nodes per step.

    H: callable mapping an array of times to an array (..., n, n) of Hamiltonians.
    Each step propagator is exp(-i K), K = h/2 (H1 + H2) - i sqrt(3) h^2 / 12 [H2, H1],
    which is exactly unitary, so the norm is conserved to rounding error.
    """
    h = T / steps
    t0 = h * np.arange(steps)
    H1 = H(t0 + (0.5 - GAUSS) * h)
    H2 = H(t0 + (0.5 + GAUSS) * h)
    K = 0.5 * h * (H1 + H2) - 1j * (np.sqrt(3) / 12) * h * h * (H2 @ H1 - H1 @ H2)
    w, V = np.linalg.eigh(K)
    U = (V * np.exp(-1j * w)[:, None, :]) @ np.conj(np.swapaxes(V, 1, 2))
    psi = np.asarray(psi0, dtype=complex)
    path = [psi] if return_path else None
    for u in U:
        psi = u @ psi
        if return_path:
            path.append(psi)
    return np.array(path) if return_path else psi


def spin_hamiltonian(theta, omega0, Omega):
    """Field precessing about z at polar angle theta with angular speed Omega."""
    return lambda t: H_spin(theta, Omega * np.asarray(t), omega0)


def rotating_frame_exact(theta, omega0, Omega, t, psi0):
    """psi(t) = exp(-i Omega t sz/2) exp(-i H_rot t) psi(0),
    H_rot = (omega0/2)(sin theta sx + cos theta sz) - (Omega/2) sz."""
    H_rot = H_spin(theta, 0.0, omega0) - 0.5 * Omega * SZ
    rz = np.diag(np.exp([-0.5j * Omega * t, 0.5j * Omega * t]))
    return rz @ expm(-1j * H_rot * t) @ np.asarray(psi0, dtype=complex)


def geometric_phase(psi0, psiT, energy, T, tol=1e-6):
    """arg <psi(0)|psi(T)> + E T, wrapped; nan if the final state has left psi(0)."""
    ov = np.vdot(psi0, psiT)
    if abs(ov) < tol:
        return np.nan
    return wrap(np.angle(ov) + energy * T)


def spin_geometric_phase(theta, eps, steps_per_period=80, omega0=1.0):
    """Berry-phase estimate for one precession at Omega = eps omega0, from magnus4.

    Starts in the aligned eigenstate of H(0) and removes the adiabatic dynamical
    phase -(omega0/2) T. steps_per_period counts steps per Larmor period 2 pi/omega0.
    """
    Omega = eps * omega0
    T = 2 * np.pi / Omega
    steps = int(np.ceil(steps_per_period / eps))
    psi0 = spin_state(theta, 0.0)
    psiT = magnus4(spin_hamiltonian(theta, omega0, Omega), psi0, T, steps)
    return geometric_phase(psi0, psiT, 0.5 * omega0, T)


def rotating_frame_angle(theta, eps):
    """Polar angle of the effective field (sin theta, 0, cos theta - eps) in H_rot."""
    return np.arctan2(np.sin(theta), np.cos(theta) - eps)


def spin_geometric_phase_exact(theta, eps):
    """Closed form of spin_geometric_phase from the rotating-frame solution."""
    lam = np.sqrt(1 - 2 * eps * np.cos(theta) + eps * eps)
    d = 0.5 * (rotating_frame_angle(theta, eps) - theta)
    x = np.pi * lam / eps
    ov = -(np.cos(d) ** 2 * np.exp(-1j * x) + np.sin(d) ** 2 * np.exp(1j * x))
    return wrap(np.angle(ov) + np.pi / eps)


def nonadiabatic_slope(theta):
    """d gamma / d(Omega/omega0) at Omega -> 0 for the estimate above: -(pi/2) sin^2 theta."""
    return -0.5 * np.pi * np.sin(theta) ** 2


def aharonov_anandan_phase(H, t, path):
    """Total phase minus the true dynamical phase -int <psi|H|psi> dt, for a cyclic path.

    path[k] is the state at time t[k] on a uniform grid; trapezoid rule for the integral.
    """
    e = np.real(np.einsum("ka,kab,kb->k", np.conj(path), H(t), path))
    dyn = np.sum(0.5 * (e[1:] + e[:-1]) * np.diff(t))
    return wrap(np.angle(np.vdot(path[0], path[-1])) + dyn)
