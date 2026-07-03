"""Spin-1/2 in a field n(theta, phi): H = (omega0/2) n.sigma, hbar = 1."""

import numpy as np

from .geometry import berry_phase, plaquette_flux, wrap

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SIGMA = np.array([SX, SY, SZ])


def field_direction(theta, phi):
    theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)


def H_spin(theta, phi, omega0=1.0):
    """Shape (..., 2, 2); theta and phi broadcast."""
    return 0.5 * omega0 * np.einsum("...k,kab->...ab", field_direction(theta, phi), SIGMA)


def spin_state(theta, phi, aligned=True):
    """Eigenstate of n.sigma with eigenvalue +1 (aligned) or -1, in a gauge that is
    single-valued in phi and smooth everywhere except the pole where it winds."""
    theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
    c, s, e = np.cos(theta / 2), np.sin(theta / 2), np.exp(1j * phi)
    if aligned:
        return np.stack([c + 0j, e * s], axis=-1)
    return np.stack([s + 0j, -e * c], axis=-1)


def cone_states(theta, M, aligned=True):
    """M states on the cone at polar angle theta, phi_j = 2 pi j / M."""
    return spin_state(theta, 2 * np.pi * np.arange(M) / M, aligned)


def berry_phase_cone(theta, M, aligned=True, rng=None):
    """Discretised Berry phase on the cone; rng multiplies each u_j by a random phase."""
    u = cone_states(theta, M, aligned)
    if rng is not None:
        u = u * np.exp(2j * np.pi * rng.random(M))[:, None]
    return berry_phase(u)


def berry_phase_exact(theta, aligned=True):
    """-(m) x solid angle: -pi (1 - cos theta) aligned, +pi (1 - cos theta) anti-aligned.
    Not wrapped, so it is continuous in theta."""
    g = np.pi * (1 - np.cos(theta))
    return -g if aligned else g


def berry_phase_discrete_exact(theta, M, aligned=True):
    """Closed form of the M-point formula: every link equals c^2 + s^2 e^{2 pi i/M}."""
    a = np.sin(np.asarray(theta) / 2) ** 2
    if not aligned:
        a = 1 - a
    return wrap(-M * np.angle(1 - a + a * np.exp(2j * np.pi / M)))


def discretisation_error(theta, M, aligned=True):
    """Leading term of berry_phase_discrete_exact - berry_phase_exact:
    +-(pi^3/3) sin^2(theta) cos(theta) / M^2 (see Berry-phase.pdf)."""
    e = np.pi ** 3 / 3 * np.sin(theta) ** 2 * np.cos(theta) / np.asarray(M, float) ** 2
    return e if aligned else -e


def sphere_grid(Ntheta, Nphi, aligned=True):
    """(Ntheta + 1, Nphi, 2) states, theta from 0 to pi inclusive, phi periodic."""
    th = np.linspace(0, np.pi, Ntheta + 1)[:, None]
    ph = 2 * np.pi * np.arange(Nphi)[None, :] / Nphi
    return spin_state(th, ph, aligned)


def chern_number(Ntheta=30, Nphi=60, aligned=True, rng=None):
    """Total Berry flux through the sphere of field directions over 2 pi."""
    g = sphere_grid(Ntheta, Nphi, aligned)
    if rng is not None:
        g = g * np.exp(2j * np.pi * rng.random(g.shape[:2]))[..., None]
    return np.sum(plaquette_flux(g)) / (2 * np.pi)
