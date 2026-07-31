"""Bloch and non-Bloch winding of the non-reciprocal SSH chain (Yao and Wang 2018).

nu = (W[h2] - W[h1]) / 2 with both windings taken counter-clockwise around |beta| = r.
The sign is fixed by nu = 1 in the Hermitian limit with t1 < t2, the SSH-model convention.
"""

import numpy as np

from nh_ssh import exact_open_spectrum

from .model import ssh_h1, ssh_h2


def wrap(phi):
    """Angle in (-pi, pi]."""
    return np.pi - np.mod(np.pi - phi, 2 * np.pi)


def winding(f, r, n=2001, gap_tol=1e-9):
    """Winding of the Laurent polynomial f around 0 along |beta| = r, from summed phase steps.

    nan if a zero of f lies on the contour. A sampled min|f| would miss it when the
    zero falls between grid points, so the zeros are located exactly.
    """
    if r == 0 or np.any(np.abs(np.abs(zeros(f)) / r - 1) < gap_tol):
        return np.nan
    h = f(r * np.exp(2j * np.pi * np.arange(n) / n))
    return np.sum(wrap(np.diff(np.angle(np.append(h, h[0]))))) / (2 * np.pi)


def zeros(f):
    """Zeros of beta^p f(beta) other than beta = 0."""
    return np.roots(f.coeffs[::-1]) if len(f.coeffs) > 1 else np.array([])


def winding_by_counting(f, r):
    """Argument principle: zeros inside |beta| = r minus the pole order p at beta = 0."""
    return int(np.sum(np.abs(zeros(f)) < r)) - f.p


def gbz_radius_ssh(t1, t2, g):
    """|beta_1| |beta_2| = (t1 - g/2)/(t1 + g/2) at every E, so equal moduli force this radius."""
    return np.sqrt(abs((t1 - g / 2) / (t1 + g / 2)))


def chiral_winding(t1, t2, g, r, n=2001):
    return (winding(ssh_h2(t1, t2, g), r, n) - winding(ssh_h1(t1, t2, g), r, n)) / 2


def bloch_winding(t1, t2, g, n=2001):
    return chiral_winding(t1, t2, g, 1.0, n)


def nonbloch_winding(t1, t2, g, n=2001):
    return chiral_winding(t1, t2, g, gbz_radius_ssh(t1, t2, g), n)


def nonbloch_winding_closed_form(t1, t2, g, tol=1e-12):
    """1 for |t1^2 - g^2/4| < t2^2, 0 above, nan on the boundary. Derivation: README §3.3."""
    x = abs(t1 * t1 - g * g / 4) / (t2 * t2)
    if abs(x - 1) < tol:
        return np.nan
    return 1.0 if x < 1 else 0.0


def edge_pair_count(N, t1, t2, g, tol=1e-8):
    """Number of +-E pairs of the open chain with |E| < tol, from the exact (similar) spectrum."""
    E = exact_open_spectrum(N, t1, t2, g)
    return int(np.sum(np.abs(E) < tol)) // 2
