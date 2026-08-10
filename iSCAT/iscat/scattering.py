"""Rayleigh and Mie scattering by a homogeneous sphere."""

import numpy as np


def clausius_mossotti(m):
    """L = (m^2 - 1)/(m^2 + 2)."""
    m2 = np.asarray(m) ** 2
    return (m2 - 1) / (m2 + 2)


def polarizability(a, m):
    """Quasi-static sphere polarizability alpha = 4 pi a^3 L."""
    return 4 * np.pi * np.asarray(a, dtype=float) ** 3 * clausius_mossotti(m)


def polarizability_radiative(a, m, k):
    """alpha / (1 - i k^3 alpha / 6 pi)."""
    alpha = polarizability(a, m)
    return alpha / (1 - 1j * k ** 3 * alpha / (6 * np.pi))


def rayleigh_cross_sections(alpha, k):
    """(sigma_sca, sigma_abs) of a point dipole."""
    alpha = np.asarray(alpha)
    return k ** 4 * np.abs(alpha) ** 2 / (6 * np.pi), k * np.imag(alpha)


def rayleigh_sigma_sca(a, m, k):
    """(8 pi / 3) k^4 a^6 |L|^2, the brief's formula."""
    return 8 * np.pi / 3 * k ** 4 * np.asarray(a, dtype=float) ** 6 * np.abs(clausius_mossotti(m)) ** 2


def mie_nmax(x):
    """Wiscombe's truncation order plus 15 terms."""
    return int(np.round(abs(x) + 4.05 * abs(x) ** (1 / 3) + 2)) + 15


def mie_coefficients(m, x, nmax=None):
    """Mie a_n, b_n for n = 1..nmax (Bohren & Huffman's BHMIE algorithm)."""
    m = complex(m)
    if not x > 0:
        raise ValueError("size parameter x must be positive")
    if nmax is None:
        nmax = mie_nmax(x)
    mx = m * x
    nstart = int(max(nmax, abs(mx))) + 16
    D = np.zeros(nstart + 1, dtype=complex)
    for n in range(nstart, 0, -1):
        D[n - 1] = n / mx - 1 / (D[n] + n / mx)
    n = np.arange(1, nmax + 1)
    psi = np.zeros(nmax + 2)
    chi = np.zeros(nmax + 2)
    psi[0], psi[1] = np.cos(x), np.sin(x)
    chi[0], chi[1] = -np.sin(x), np.cos(x)
    for j in range(2, nmax + 2):
        psi[j] = (2 * j - 3) / x * psi[j - 1] - psi[j - 2]
        chi[j] = (2 * j - 3) / x * chi[j - 1] - chi[j - 2]
    xi = psi - 1j * chi
    psi_n, psi_m1 = psi[2:], psi[1:-1]
    xi_n, xi_m1 = xi[2:], xi[1:-1]
    Dn = D[1:nmax + 1]
    ta = Dn / m + n / x
    tb = m * Dn + n / x
    a = (ta * psi_n - psi_m1) / (ta * xi_n - xi_m1)
    b = (tb * psi_n - psi_m1) / (tb * xi_n - xi_m1)
    return a, b


def mie_efficiencies(m, x):
    """(Q_ext, Q_sca, Q_back) with Q = sigma / (pi a^2)."""
    a, b = mie_coefficients(m, x)
    n = np.arange(1, len(a) + 1)
    q_ext = 2 / x ** 2 * np.sum((2 * n + 1) * (a + b).real)
    q_sca = 2 / x ** 2 * np.sum((2 * n + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2))
    q_back = np.abs(np.sum((2 * n + 1) * (-1.0) ** n * (a - b))) ** 2 / x ** 2
    return q_ext, q_sca, q_back


def mie_amplitudes(m, x, mu):
    """Amplitude functions S1, S2 at mu = cos(theta) (Bohren & Huffman 4.74)."""
    a, b = mie_coefficients(m, x)
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    s1 = np.zeros(mu.shape, dtype=complex)
    s2 = np.zeros(mu.shape, dtype=complex)
    pi_prev, pi_n = np.zeros_like(mu), np.ones_like(mu)
    for n in range(1, len(a) + 1):
        tau_n = n * mu * pi_n - (n + 1) * pi_prev
        w = (2 * n + 1) / (n * (n + 1))
        s1 += w * (a[n - 1] * pi_n + b[n - 1] * tau_n)
        s2 += w * (a[n - 1] * tau_n + b[n - 1] * pi_n)
        pi_prev, pi_n = pi_n, ((2 * n + 1) * mu * pi_n - (n + 1) * pi_prev) / n
    return s1, s2


def mie_backscatter_amplitude(a, m, k):
    """x-component of the backscattered amplitude f (nm) for x-polarised incidence."""
    out = []
    for ai in np.atleast_1d(a):
        an, bn = mie_coefficients(m, k * ai)
        n = np.arange(1, len(an) + 1)
        s1 = np.sum((2 * n + 1) / 2 * (-1.0) ** (n + 1) * (an - bn))
        out.append(s1 / (-1j * k))
    out = np.array(out)
    return out if np.ndim(a) else out[0]


def dipole_amplitude(alpha, k):
    """Scattering amplitude of a point dipole, f = k^2 alpha / 4 pi."""
    return k ** 2 * np.asarray(alpha) / (4 * np.pi)


def rayleigh_x2_coefficient(m):
    """B in sigma_Mie / sigma_Rayleigh = 1 + B x^2 + O(x^3)."""
    m2 = np.asarray(m) ** 2
    return 6 / 5 * np.real((m2 - 2) / (m2 + 2))


def backscatter_x2_coefficient(m):
    """C in f_back/f_dipole = 1 + C x^2 + O(x^3)."""
    m2 = np.asarray(m) ** 2
    return 3 / 5 * (m2 - 2) / (m2 + 2) - (m2 + 2) / 30 - (m2 + 2) / (6 * (2 * m2 + 3))


def protein_polarizability(mass_kda, dn_dc=0.185, n_m=1.33):
    """alpha (nm^3) of a protein from its refractive-index increment dn/dc (mL/g)."""
    grams = np.asarray(mass_kda, dtype=float) * 1e3 / 6.02214076e23
    return 2 * dn_dc * grams / n_m * 1e21


def mie_dipole_polarizability(a, m, k):
    """Exact electric-dipole polarizability 6 pi i a_1 / k^3 of a sphere."""
    out = np.array([6j * np.pi * mie_coefficients(m, k * ai, nmax=max(1, mie_nmax(k * ai)))[0][0] / k ** 3
                    for ai in np.atleast_1d(a)])
    return out if np.ndim(a) else out[0]
