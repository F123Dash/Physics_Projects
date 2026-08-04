"""Paraxial Gaussian and Laguerre-Gauss beams, and grating transmissions.

The waist is at z = 0. With exp(i(kz - wt)) the Gouy phase is a lag:
LG_pl carries exp(i kz - i (2p + |l| + 1) arctan(z / zR)).
"""

import math

import numpy as np
from scipy.special import eval_genlaguerre, gamma


def rayleigh_range(w0, wavelength, n=1.0):
    return np.pi * w0**2 * n / wavelength


def beam_width(z, w0, wavelength, n=1.0):
    """1/e^2 intensity radius."""
    return w0 * np.sqrt(1 + (z / rayleigh_range(w0, wavelength, n)) ** 2)


def curvature(z, w0, wavelength, n=1.0):
    """1/R(z), which is finite (zero) at the waist."""
    zR = rayleigh_range(w0, wavelength, n)
    return z / (z**2 + zR**2)


def gouy(z, w0, wavelength, n=1.0):
    return np.arctan(z / rayleigh_range(w0, wavelength, n))


def lg(x, y, p, l, w0, wavelength, z=0.0, n=1.0, with_gouy=True):
    """LG_pl(x, y, z), normalised to unit power; x, y broadcast against each other."""
    k = 2 * np.pi * n / wavelength
    w = beam_width(z, w0, wavelength, n)
    r2 = x**2 + y**2
    s = 2 * r2 / w**2
    C = np.sqrt(2 * math.factorial(p) / (np.pi * math.factorial(p + abs(l))))
    amp = C / w * s ** (abs(l) / 2) * eval_genlaguerre(p, abs(l), s) * np.exp(-r2 / w**2)
    phase = l * np.arctan2(y, x) + k * z + k * r2 * curvature(z, w0, wavelength, n) / 2
    if with_gouy:
        phase = phase - (2 * p + abs(l) + 1) * gouy(z, w0, wavelength, n)
    return amp * np.exp(1j * phase)


def gaussian(x, y, w0, wavelength, z=0.0, n=1.0, with_gouy=True):
    return lg(x, y, 0, 0, w0, wavelength, z, n, with_gouy)


def laguerre_moment(p, l, q):
    """<s^q> in |LG_pl|^2 with s = 2 r^2 / w^2, i.e. in the weight s^|l| e^-s L_p^|l|(s)^2.
    Gauss-Laguerre with this many nodes is exact for the polynomial integrand."""
    a = abs(l)
    s, wt = np.polynomial.laguerre.laggauss(p + a // 2 + q + 2)
    f = s**a * eval_genlaguerre(p, a, s) ** 2
    return np.sum(wt * s**q * f) / np.sum(wt * f)


def width_nonparaxial(z, w0, wavelength, n=1.0):
    """Second-moment radius of a Gaussian under exact propagation, to second order in
    eps = 1/(k w0)^2: w^2 = w0^2 + 4 z^2 <kx^2 / kz^2> = w0^2 [1 + (z/zR)^2 (1 + 4 eps + 24 eps^2)]."""
    eps = 1 / (2 * np.pi * n / wavelength * w0) ** 2
    zeta = z / rayleigh_range(w0, wavelength, n)
    return w0 * np.sqrt(1 + zeta**2 * (1 + 4 * eps + 24 * eps**2))


def gouy_on_axis_nonparaxial(z, w0, wavelength, n=1.0):
    """kz - arg E(0, z) for a Gaussian, to first order in eps = 1/(k w0)^2:
    arctan(zeta) + 2 eps zeta (1 - zeta^2) / (1 + zeta^2)^2."""
    eps = 1 / (2 * np.pi * n / wavelength * w0) ** 2
    zeta = z / rayleigh_range(w0, wavelength, n)
    return np.arctan(zeta) + 2 * eps * zeta * (1 - zeta**2) / (1 + zeta**2) ** 2


def lg_nonparaxial_phase(z, p, l, w0, wavelength, n=1.0):
    """First-order phase of <LG_pl(z)|exact E(z)> beyond the Gouy lag, -zeta eps <s^2> / 4,
    from kz ~ k - k_rho^2/(2k) - k_rho^4/(8k^3) averaged over the mode's angular spectrum."""
    eps = 1 / (2 * np.pi * n / wavelength * w0) ** 2
    zeta = z / rayleigh_range(w0, wavelength, n)
    return -zeta * eps * laguerre_moment(p, l, 2) / 4


def lg_nonparaxial_infidelity(z, p, l, w0, wavelength, n=1.0):
    """Leading 1 - |<LG_pl(z)|exact E(z)>|^2 = (zeta eps / 4)^2 Var(s^2)."""
    eps = 1 / (2 * np.pi * n / wavelength * w0) ** 2
    zeta = z / rayleigh_range(w0, wavelength, n)
    var = laguerre_moment(p, l, 4) - laguerre_moment(p, l, 2) ** 2
    return (zeta * eps / 4) ** 2 * var


def gaussian_on_axis_exact(z, w0, wavelength, n=1.0, nodes=2000):
    """E(0, z) / E(0, 0) for the field exp(-r^2 / w0^2) at z = 0 under the scalar Helmholtz
    equation: the Hankel integral of its angular spectrum, 2 int exp(-s^2 + i kz z) s ds with
    k_rho = 2 s / w0, by Gauss-Legendre quadrature on s in [0, 7] (the tail is below e^-49)."""
    k = 2 * np.pi * n / wavelength
    t, wt = np.polynomial.legendre.leggauss(nodes)
    s, wt = 3.5 * (t + 1), 3.5 * wt
    kz = np.sqrt((k**2 - (2 * s / w0) ** 2).astype(complex))
    z = np.atleast_1d(z)
    out = 2 * np.exp(1j * np.outer(z, kz) - s**2) @ (s * wt)
    return out if out.size > 1 else out[0]


def binary_grating(x, d, duty=0.5):
    """Amplitude transmission 1 on the first duty * d of each period, 0 elsewhere."""
    return (np.mod(x, d) < duty * d).astype(float)


def binary_grating_series(x, d, wavelength, duty=0.5):
    """The propagating part of binary_grating: its Fourier series c_m = exp(-i pi m duty) sin(pi m duty) / (pi m)
    summed over |m| < d / wavelength."""
    m = np.arange(-int(d / wavelength), int(d / wavelength) + 1)
    m = m[np.abs(m) * wavelength / d < 1]
    c = np.where(m == 0, duty, np.exp(-1j * np.pi * m * duty) * np.sin(np.pi * m * duty) / (np.pi * np.where(m == 0, 1, m)))
    return np.exp(2j * np.pi * np.outer(x, m) / d) @ c


def cosine_grating(x, d, m=0.5):
    """1 + m cos(2 pi x / d): only the harmonics 0 and +-1."""
    return 1 + m * np.cos(2 * np.pi * x / d)


def talbot_paraxial(d, wavelength):
    return 2 * d**2 / wavelength


def talbot_exact(d, wavelength, m=1):
    """Distance at which harmonic m first rephases with harmonic 0: (k - kz_m) z = 2 pi.
    Written as lambda (1 + sqrt(1 - u)) / u, since 1 - sqrt(1 - u) cancels for d >> lambda."""
    u = (m * wavelength / d) ** 2
    return wavelength * (1 + np.sqrt(1 - u)) / u


def talbot_defect_asymptote(d, wavelength):
    """Leading 1 - F at z_T for a 50% binary grating as d / lambda -> infinity:
    (4 / pi^2) Re J sqrt(lambda / d), Re J = (pi/2)^(1/4) (1/4) (-Gamma(-1/4)) cos(pi/8)."""
    re_J = (np.pi / 2) ** 0.25 * 0.25 * (-gamma(-0.25)) * np.cos(np.pi / 8)
    return 4 * re_J / np.pi**2 * np.sqrt(wavelength / d)


def talbot_fidelity_series(d, wavelength, z, duty=0.5):
    """Field fidelity |<t|E(z)>|^2 / (<t|t><E|E>) of a binary grating's propagating orders,
    summed over its Fourier series c_m = sin(pi m duty) / (pi m)."""
    m = np.arange(1, int(d / wavelength) + 1)
    m = m[m * wavelength / d < 1]
    P = np.concatenate([[duty**2], 2 * (np.sin(np.pi * m * duty) / (np.pi * m)) ** 2])
    u = (np.concatenate([[0], m]) * wavelength / d) ** 2
    phase = 2 * np.pi * z / wavelength * u / (1 + np.sqrt(1 - u))  # (k - kz_m) z
    return abs(np.sum(P * np.exp(1j * phase))) ** 2 / np.sum(P) ** 2
