"""Fibre description: dispersion, Kerr and Raman response."""

from dataclasses import dataclass, field, replace
from math import factorial

import numpy as np


def raman_response(t, tau1=0.0122, tau2=0.032):
    """Blow-Wood Raman response h_R(t) (units 1/ps), zero for t < 0."""
    t = np.asarray(t, dtype=float)
    pref = (tau1**2 + tau2**2) / (tau1 * tau2**2)
    h = pref * np.exp(-t / tau2) * np.sin(t / tau1)
    return np.where(t >= 0, h, 0.0)


def raman_time_analytic(fR=0.18, tau1=0.0122, tau2=0.032):
    """First moment T_R = f_R * int t h_R(t) dt = f_R * 2 tau1^2 tau2 / (tau1^2 + tau2^2)."""
    return fR * 2 * tau1**2 * tau2 / (tau1**2 + tau2**2)


@dataclass
class Fibre:
    """Fibre parameters.

    betas : [beta2, beta3, ...] in ps^k/m
    gamma : Kerr coefficient, 1/(W m)
    alpha : power loss, 1/m
    raman, self_steepening : switch those terms on/off
    """

    betas: list = field(default_factory=list)
    gamma: float = 0.0
    length: float = 1.0
    alpha: float = 0.0
    raman: bool = False
    self_steepening: bool = False
    fR: float = 0.18
    tau1: float = 0.0122
    tau2: float = 0.032

    @property
    def beta2(self):
        return self.betas[0] if self.betas else 0.0

    def beta(self, omega):
        """beta(omega0 + Omega) - beta0 - beta1 Omega = sum_k beta_k Omega^k / k!."""
        omega = np.asarray(omega, dtype=float)
        out = np.zeros_like(omega)
        for k, b in enumerate(self.betas, start=2):
            out = out + b * omega**k / factorial(k)
        return out

    def beta1_rel(self, omega):
        """d/dOmega of beta(Omega): inverse group velocity relative to the carrier frame."""
        omega = np.asarray(omega, dtype=float)
        out = np.zeros_like(omega)
        for k, b in enumerate(self.betas, start=2):
            out = out + b * omega ** (k - 1) / factorial(k - 1)
        return out

    def gvd(self, omega):
        """beta2 at offset frequency Omega (ps^2/m)."""
        omega = np.asarray(omega, dtype=float)
        out = np.zeros_like(omega)
        for k, b in enumerate(self.betas, start=2):
            out = out + b * omega ** (k - 2) / factorial(k - 2)
        return out

    def linear_operator(self, omega):
        """Frequency-domain linear operator L(Omega) = i beta(Omega) - alpha/2."""
        return 1j * self.beta(omega) - self.alpha / 2

    @property
    def raman_time(self):
        return raman_time_analytic(self.fR, self.tau1, self.tau2)

    def with_(self, **kw):
        return replace(self, **kw)


def dudley_pcf(raman=True, self_steepening=True, length=0.15):
    """Benchmark PCF of Dudley, Genty & Coen, RMP 78, 1135 (2006), pumped at 835 nm."""
    betas_km = [-11.830, 8.1038e-2, -9.5205e-5, 2.0737e-7, -5.3943e-10,
                1.3486e-12, -2.5495e-15, 3.0524e-18, -1.7140e-21]
    return Fibre(betas=[b * 1e-3 for b in betas_km], gamma=0.11, length=length,
                 raman=raman, self_steepening=self_steepening)


DUDLEY_WAVELENGTH = 835.0     # nm
DUDLEY_T0 = 0.0284            # ps (T_FWHM = 50 fs)
DUDLEY_P0 = 10_000.0          # W
