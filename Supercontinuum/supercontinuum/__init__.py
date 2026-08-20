"""Generalised nonlinear Schrödinger equation (GNLSE) solver for fibre optics.

Units throughout: time in ps, angular frequency in rad/ps, distance in m,
power in W, beta_k in ps^k/m, gamma in 1/(W m), wavelength in nm.

Fourier convention: A(T) = sum_k Aw_k exp(-i Omega_k T), i.e. the spectrum is
``Aw = ifft(A)`` and positive Omega means a frequency above the carrier.
"""

from .grid import Grid, C_NM_PS
from .fibre import Fibre, dudley_pcf, raman_response, raman_time_analytic
from .pulses import sech_pulse, gaussian_pulse, one_photon_noise
from .scales import (dispersion_length, nonlinear_length, soliton_order,
                     fission_length, soliton_period, soliton_power)
from .solver import GNLSE, Result
from . import analysis

__all__ = [
    "Grid", "C_NM_PS", "Fibre", "dudley_pcf", "raman_response",
    "raman_time_analytic", "sech_pulse", "gaussian_pulse", "one_photon_noise",
    "dispersion_length", "nonlinear_length", "soliton_order", "fission_length",
    "soliton_period", "soliton_power", "GNLSE", "Result", "analysis",
]
