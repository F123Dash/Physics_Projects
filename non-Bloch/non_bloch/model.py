"""Single-band chains as Laurent polynomials E(beta) = sum_k t_k beta^k, k = -p..q.

Convention: H[n, n+k] = t_k, so (H psi)_n = sum_k t_k psi_{n+k} and psi_n = beta^n
gives E(beta). t_{-k} moves a particle k sites to the right, t_k k sites to the left.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Laurent:
    """coeffs[j] multiplies beta^(j - p)."""

    coeffs: np.ndarray
    p: int

    def __post_init__(self):
        c = np.asarray(self.coeffs, dtype=complex)
        nz = np.flatnonzero(c)
        if nz.size == 0:
            raise ValueError("all coefficients are zero")
        # Zero end coefficients change the number of roots, so p and q must follow them.
        object.__setattr__(self, "coeffs", c[nz[0]:nz[-1] + 1])
        object.__setattr__(self, "p", self.p - nz[0])

    @property
    def q(self):
        return len(self.coeffs) - 1 - self.p

    @property
    def powers(self):
        return np.arange(-self.p, self.q + 1)

    def __call__(self, beta):
        beta = np.asarray(beta, dtype=complex)
        return sum(c * beta ** k for k, c in zip(self.powers, self.coeffs))

    def scaled(self, rho):
        """E(rho beta): the symbol after the imaginary gauge psi_n -> rho^n psi_n."""
        return Laurent(self.coeffs * rho ** self.powers.astype(float), self.p)


def hatano_nelson(t_R, t_L):
    return Laurent([t_R, 0.0, t_L], 1)


def range2(t_m2, t_m1, t_1, t_2):
    return Laurent([t_m2, t_m1, 0.0, t_1, t_2], 2)


def ssh_h1(t1, t2, g):
    """H_AB(beta) = t1 + g/2 + t2/beta, as a Laurent polynomial."""
    return Laurent([t2, t1 + g / 2], 1)


def ssh_h2(t1, t2, g):
    """H_BA(beta) = t1 - g/2 + t2 beta."""
    return Laurent([t1 - g / 2, t2], 0)


def ssh_lambda(t1, t2, g):
    """h1 h2 = E^2. det[H(beta) - E] = E^2 - h1 h2, so the two-band chiral chain
    behaves like one band in the variable lambda = E^2."""
    a, b = t1 + g / 2, t1 - g / 2
    return Laurent([t2 * b, a * b + t2 * t2, t2 * a], 1)


def H_obc(N, chain, rho=1.0):
    """Open N-site Toeplitz matrix, optionally in the gauge psi_n -> rho^n psi_n.

    The gauge leaves the eigenvalues unchanged but, with rho inside the GBZ's
    radial range, removes most of the non-normality (README §2.4).
    """
    sym = chain.scaled(rho) if rho != 1.0 else chain
    H = np.zeros((N, N), dtype=complex)
    for k, c in zip(sym.powers, sym.coeffs):
        if abs(k) < N:
            H += np.diag(np.full(N - abs(k), c), k)
    return H


def hatano_nelson_open_spectrum(N, t_R, t_L):
    """Exact: E_m = 2 sqrt(t_R t_L) cos(pi m / (N + 1)), m = 1..N."""
    m = np.arange(1, N + 1)
    return 2 * np.sqrt(complex(t_R * t_L)) * np.cos(np.pi * m / (N + 1))


def circular_radius(chain, tol=1e-12):
    """Radius rho with t_k rho^k = t_{-k} rho^-k for every k, or nan if none exists.

    Then E(rho beta) = E(rho / beta), roots come in pairs (x, 1/x), and the GBZ is
    exactly the circle |beta| = rho. Only positive ratios t_{-k}/t_k are considered.
    """
    if chain.p != chain.q:
        return np.nan
    c = chain.coeffs
    rho = None
    for k in range(1, chain.p + 1):
        lo, hi = c[chain.p - k], c[chain.p + k]
        if lo == 0 and hi == 0:
            continue
        ratio = lo / hi if hi != 0 else np.inf
        if not np.isfinite(ratio) or abs(ratio.imag) > tol or ratio.real <= 0:
            return np.nan
        r = ratio.real ** (1 / (2 * k))
        if rho is None:
            rho = r
        elif abs(r / rho - 1) > tol:
            return np.nan
    return np.nan if rho is None else rho
