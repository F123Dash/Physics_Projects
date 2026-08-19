"""Time and frequency grids."""

import numpy as np

C_NM_PS = 299_792.458  # speed of light in nm/ps


class Grid:
    """Uniform time grid of ``n`` points spanning ``window`` ps.

    ``t`` is centred on zero (pulse sits in the middle of the window).
    ``omega`` is the offset from the carrier, in FFT order (not shifted).
    """

    def __init__(self, n, window, wavelength0=None):
        self.n = int(n)
        self.window = float(window)
        self.dt = self.window / self.n
        self.t = (np.arange(self.n) - self.n // 2) * self.dt
        self.omega = 2 * np.pi * np.fft.fftfreq(self.n, self.dt)
        self.wavelength0 = wavelength0
        self.omega0 = None if wavelength0 is None else 2 * np.pi * C_NM_PS / wavelength0

    # --- transforms (see package docstring for the convention) ---
    @staticmethod
    def to_freq(A):
        return np.fft.ifft(A, axis=-1)

    @staticmethod
    def to_time(Aw):
        return np.fft.fft(Aw, axis=-1)

    # --- frequency axes ---
    @property
    def omega_shifted(self):
        return np.fft.fftshift(self.omega)

    @property
    def absolute_omega(self):
        """Absolute angular frequency omega0 + Omega (FFT order)."""
        return self.omega0 + self.omega

    @property
    def wavelength(self):
        """Wavelength in nm for each FFT-order bin (nan where omega0+Omega <= 0)."""
        w = self.absolute_omega
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(w > 0, 2 * np.pi * C_NM_PS / w, np.nan)

    def __repr__(self):
        return f"Grid(n={self.n}, window={self.window} ps, dt={self.dt*1e3:.3f} fs)"
