"""Characteristic length scales of pulse propagation."""

import numpy as np


def dispersion_length(T0, beta2):
    return T0**2 / abs(beta2)


def nonlinear_length(gamma, P0):
    return 1.0 / (gamma * P0)


def soliton_order(gamma, P0, T0, beta2):
    return np.sqrt(gamma * P0 * T0**2 / abs(beta2))


def soliton_power(N, gamma, T0, beta2):
    """Peak power for soliton order N."""
    return N**2 * abs(beta2) / (gamma * T0**2)


def fission_length(T0, beta2, N):
    return dispersion_length(T0, beta2) / N


def soliton_period(T0, beta2):
    return 0.5 * np.pi * dispersion_length(T0, beta2)
