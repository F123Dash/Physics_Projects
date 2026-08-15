"""Photothermal iSCAT."""

import numpy as np

from .scattering import mie_efficiencies

WATER_KAPPA = 0.6  # W/(m K)
WATER_DIFFUSIVITY = 1.43e-7  # m^2/s
WATER_DN_DT = -1.0e-4  # 1/K


def absorption_cross_section(a, m, k):
    """sigma_abs = pi a^2 (Q_ext - Q_sca) (nm^2) from Mie theory."""
    q_ext, q_sca, _ = mie_efficiencies(m, k * a)
    return np.pi * a ** 2 * (q_ext - q_sca)


def thermal_gamma(omega, diffusivity=WATER_DIFFUSIVITY):
    """gamma (1/nm) of the oscillating temperature e^{-gamma r}/r for exp(-i Omega t)."""
    if np.any(np.asarray(omega) < 0):
        raise ValueError("Omega must be >= 0 (its sign is fixed by the exp(-i Omega t) convention)")
    return (1 - 1j) * np.sqrt(np.asarray(omega, dtype=float) / (2 * diffusivity * 1e18))


def temperature(r, power, a, gamma, kappa=WATER_KAPPA):
    """Complex temperature amplitude (K) at r >= a (nm) for absorbed power amplitude (W)."""
    amp = power / (4 * np.pi * kappa * (1 + gamma * a)) * 1e9
    return amp * np.exp(-gamma * (np.asarray(r) - a)) / np.asarray(r)


def lens_form_factor(q2, a, gamma):
    """Radial Born integral of the lens profile e^{-gamma (r-a)}/r at momentum transfer q (nm^2)."""
    q = np.sqrt(np.asarray(q2, dtype=complex))
    qa = q * a
    small = np.abs(qa) < 1e-8
    sinc = np.where(small, a, np.sin(qa) / np.where(small, 1, q))
    return 4 * np.pi * (np.cos(qa) + gamma * sinc) / (q * q + gamma ** 2)


def lens_polarizability(q2, power, a, gamma, dn_dT=WATER_DN_DT, n_m=1.33, kappa=WATER_KAPPA):
    """delta alpha(q) = (2 dn/dT / n_m) int T(r) e^{i q.r} d^3r (nm^3)."""
    amp = power / (4 * np.pi * kappa * (1 + gamma * a)) * 1e9
    return 2 * dn_dT / n_m * amp * lens_form_factor(q2, a, gamma)


def photothermal_contrast(es, ey, plus, minus, er):
    """Lock-in amplitude dC of the contrast, dI/|E_r|^2 = Re(dC e^{-i Omega t})."""
    ex = er + es
    return (np.conj(ex) * plus[0] + ex * np.conj(minus[0])
            + np.conj(ey) * plus[1] + ey * np.conj(minus[1])) / abs(er) ** 2


def lockin_demodulate(counts, omega_t):
    """Lock-in amplitude (2/N) sum n_j e^{i Omega t_j}; noise 2/N per quadrature."""
    counts = np.asarray(counts)
    return 2 * np.sum(counts * np.exp(1j * omega_t), axis=-1) / np.sum(counts, axis=-1)
