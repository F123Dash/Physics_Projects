"""Biorthogonal polarisation of Kunst, Edvardsson, Budich, Bergholtz, PRL 121, 026808 (2018).

Odd chain: N A-sites and N - 1 B-sites, cells n = 1..N, the last cell broken.
Eqs. (3)-(4): psi_R ~ r_R^n, psi_L ~ r_L^n on A only, with
r_R = -(t1 - g/2)/t2 and r_L = -(t1 + g/2)/t2 for real parameters.
Eq. (7): P = 1 - lim (1/N) <psi_L| sum_n n Pi_n |psi_R>, with <psi_L|psi_R> = 1.
"""

import numpy as np

from .model import H_open, cell_index, chiral_operator, hoppings
from .spectrum import biorthogonal_eig, cell_weight


def r_factors(t1, t2, g):
    up, down = hoppings(t1, g)
    return -down / t2, -np.conj(up) / np.conj(t2)


def zero_mode(N, t1, t2, g):
    """Right and left zero modes of the odd chain (N A-sites), from scipy eig.

    The mode is picked as the eigenvector with the least weight on B, which is
    exactly zero for the true zero mode, rather than by the size of |E|.
    """
    H = H_open(N - 1, t1, t2, g, extra_site=True)
    E, R, L = biorthogonal_eig(H)
    on_B = chiral_operator(H.shape[0]) < 0

    def weight_on_B(V):
        return np.sum(np.abs(V[on_B]) ** 2, axis=0) / np.sum(np.abs(V) ** 2, axis=0)

    with np.errstate(invalid="ignore", over="ignore"):
        score = weight_on_B(R) + weight_on_B(L)
    if np.all(np.isnan(score)):
        return np.nan, np.full(H.shape[0], np.nan), np.full(H.shape[0], np.nan)
    j = np.nanargmin(score)
    return E[j], R[:, j], L[:, j]


def biorthogonal_density(R, L):
    """<Pi_n> = <L|Pi_n|R> per cell, complex in general."""
    rho = np.zeros((len(R) + 1) // 2, dtype=complex)
    np.add.at(rho, cell_index(len(R)), np.conj(L) * R)
    return rho


def polarization(N, t1, t2, g):
    """Eq. (7) at finite N from the numerical left and right zero modes."""
    _, R, L = zero_mode(N, t1, t2, g)
    rho = biorthogonal_density(R, L)
    n = np.arange(1, N + 1)
    return 1.0 - np.real(np.sum(n * rho)) / N


def polarization_exact(N, t1, t2, g):
    """Eq. (7) at finite N from the closed-form zero modes, <Pi_n> proportional to x^n.

    Evaluated in log space so that |x|^N does not overflow at large N. For x < 0
    and even N the norm sum_n x^n vanishes at x = -1: there L and R are
    biorthogonally orthogonal and P is undefined (nan).
    """
    rR, rL = r_factors(t1, t2, g)
    x = np.conj(rL) * rR
    n = np.arange(1, N + 1)
    if x == 0:
        return 1.0 - 1.0 / N
    if abs(x) == 1.0:
        terms = np.exp(1j * np.angle(x) * n)
    else:
        logw = n * np.log(abs(x))
        terms = np.exp(logw - logw.max() + 1j * np.angle(x) * n)
    norm = np.sum(terms)
    if abs(norm) < 1e-12 * np.max(np.abs(terms)):
        return np.nan
    return 1.0 - np.real(np.sum(n * terms) / norm) / N


def polarization_limit(t1, t2, g, tol=1e-12):
    """N -> infinity: 1 for |r_L^* r_R| < 1, 0 above, nan on the critical line."""
    rR, rL = r_factors(t1, t2, g)
    x = abs(np.conj(rL) * rR)
    if abs(x - 1.0) < tol:
        return np.nan
    return 1.0 if x < 1.0 else 0.0


def obc_transitions(t2, g):
    """t1 >= 0 where |t1^2 - g^2/4| = t2^2 (Eq. 9 of the paper)."""
    cand = [np.sqrt(t2 * t2 + g * g / 4)]
    if g * g / 4 >= t2 * t2:
        cand.append(np.sqrt(g * g / 4 - t2 * t2))
    return np.sort(cand)


def cell_weights_of_zero_mode(N, t1, t2, g):
    """|R|^2 and |L|^2 per cell (each normalised to sum 1) and the biorthogonal density."""
    _, R, L = zero_mode(N, t1, t2, g)
    wR = cell_weight(R)
    wL = cell_weight(L)
    return wR / wR.sum(), wL / wL.sum(), biorthogonal_density(R, L)
