"""Observables of sampled fields: widths, phases, winding, OAM, image fidelity."""

import numpy as np

from .propagate import frequencies


def wrap(phi):
    """Map angles to (-pi, pi]."""
    return np.pi - np.mod(np.pi - phi, 2 * np.pi)


def overlap(a, b, dx):
    """<a|b> = integral of conj(a) b."""
    return np.vdot(a, b) * dx**np.ndim(a)


def fidelity(a, b):
    """|<a|b>|^2 / (<a|a><b|b>): 1 iff b is a multiple of a."""
    return abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)


def second_moment_width(E, x, axis=0):
    """2 sigma of |E|^2 along one axis, which is the 1/e^2 radius w for a Gaussian (D4sigma / 2)."""
    I = np.abs(E) ** 2
    other = tuple(a for a in range(I.ndim) if a != axis)
    Ix = I.sum(axis=other) if other else I
    xc = np.sum(x * Ix) / Ix.sum()
    return 2 * np.sqrt(np.sum((x - xc) ** 2 * Ix) / Ix.sum())


def fit_curvature(E_line, x, k, w):
    """1/R from a weighted fit phi = a + k x^2 / (2R) to the phase of a centred line cut, |x| < w."""
    sel = np.abs(x) < w
    phi = np.unwrap(np.angle(E_line[sel]))
    weights = np.abs(E_line[sel])
    A = np.stack([np.ones(sel.sum()), x[sel] ** 2], axis=1)
    coef = np.linalg.lstsq(A * weights[:, None], phi * weights, rcond=None)[0]
    return 2 * coef[1] / k


def square_loop(i0, j0, h):
    """Pixel indices of the square of half-width h around (i0, j0), counter-clockwise in (x, y)
    when axis 0 is x and axis 1 is y."""
    s = np.arange(-h, h)
    i = np.concatenate([i0 + s, np.full(2 * h, i0 + h), i0 - s, np.full(2 * h, i0 - h)])
    j = np.concatenate([np.full(2 * h, j0 - h), j0 + s, np.full(2 * h, j0 + h), j0 - s])
    return i, j


def winding_number(E, i0, j0, h):
    """Sum of wrapped phase increments around a closed pixel loop, divided by 2 pi."""
    i, j = square_loop(i0, j0, h)
    phi = np.angle(E[i, j])
    return np.sum(wrap(np.diff(np.append(phi, phi[0])))) / (2 * np.pi)


def oam_per_photon(E, dx, x, y):
    """<L_z> / hbar = <E| -i (x d_y - y d_x) |E> / <E|E>, with spectral derivatives.
    x, y are broadcastable coordinate arrays matching axes 0 and 1."""
    fx, fy = frequencies(E.shape, dx)
    Ek = np.fft.fft2(E)
    dEx = np.fft.ifft2(2j * np.pi * fx * Ek)
    dEy = np.fft.ifft2(2j * np.pi * fy * Ek)
    Lz = -1j * (x * dEy - y * dEx)
    return (np.vdot(E, Lz) / np.vdot(E, E)).real


def azimuthal_angle(I, x, y, m):
    """Orientation theta of the cos(m (phi - theta)) component of an intensity pattern.

    The weight is (x + iy)^m rather than exp(i m phi): a smooth weight keeps the grid sum
    spectrally accurate, while exp(i m phi) is singular at r = 0 and aliases with m + 4 on a
    square grid (an error of 6e-4 rad for the beams here)."""
    return np.angle(np.sum(I * (x + 1j * y) ** m)) / m


def harmonic_shift(I, d, dx):
    """Lateral shift (in units of d, in [-1/2, 1/2)) of a d-periodic pattern sampled over whole
    periods, from the phase of its first Fourier harmonic."""
    N = I.size
    c = np.fft.fft(I)[round(N * dx / d)]
    return -np.angle(c) / (2 * np.pi)


def parabolic_peak(z, F):
    """Vertex (z*, F*) of the parabola through the largest sample and its two neighbours."""
    i = int(np.clip(np.argmax(F), 1, len(F) - 2))
    a, b, c = F[i - 1], F[i], F[i + 1]
    h = z[1] - z[0]
    t = 0.5 * (a - c) / (a - 2 * b + c)
    return z[i] + t * h, b - 0.25 * (a - c) * t
