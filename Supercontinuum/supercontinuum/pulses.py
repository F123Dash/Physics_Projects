"""Input pulses and noise."""

import numpy as np

HBAR = 1.054_571_817e-34  # J s


def sech_pulse(t, P0, T0, chirp=0.0, t_offset=0.0):
    tau = (t - t_offset) / T0
    return np.sqrt(P0) / np.cosh(tau) * np.exp(-0.5j * chirp * tau**2)


def gaussian_pulse(t, P0, T0, chirp=0.0, t_offset=0.0):
    """A = sqrt(P0) exp(-(1 + iC) T^2 / (2 T0^2)); T0 is the 1/e intensity half-width."""
    tau = (t - t_offset) / T0
    return np.sqrt(P0) * np.exp(-0.5 * (1 + 1j * chirp) * tau**2)


def one_photon_noise(grid, rng):
    """Spectral noise: one photon per mode with random phase (Dudley et al., Sec. V).

    With A = fft(Aw), the energy in bin k is window * |Aw_k|^2 (W ps), so one photon
    hbar*omega_k (J) means |Aw_k|^2 = hbar omega_k / window, with the units fixed below.
    """
    w_abs = grid.absolute_omega * 1e12              # rad/s
    energy = HBAR * np.clip(w_abs, 0, None)        # J
    amp = np.sqrt(energy / (grid.window * 1e-12))  # sqrt(W)
    phase = rng.uniform(0, 2 * np.pi, grid.n)
    return amp * np.exp(1j * phase)
