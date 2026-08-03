"""Angular-spectrum propagation of scalar fields sampled on 1D or 2D grids.

Convention: fields carry exp(i(kz - wt)), so free propagation multiplies each
plane wave exp(2 pi i f.x) by exp(i kz z). Frequencies f are in cycles per unit
length, as returned by numpy.fft.fftfreq.
"""

import numpy as np


def grid(N, dx):
    """Sample positions with x = 0 on sample N // 2."""
    return (np.arange(N) - N // 2) * dx


def frequencies(shape, dx):
    """FFT-ordered spatial frequencies, one broadcastable array per axis."""
    return np.meshgrid(*[np.fft.fftfreq(n, dx) for n in shape], indexing="ij", sparse=True)


def band_limit(L, wavelength, z, paraxial=False):
    """Largest |f| along an axis of window length L whose transfer-function phase is sampled
    at Nyquist or better (Matsushima and Shimobaba 2009). wavelength is the in-medium one."""
    z = abs(z)
    if paraxial:
        return np.inf if z == 0 else L / (2 * wavelength * z)
    return 1 / (wavelength * np.sqrt((2 * z / L) ** 2 + 1))


def critical_distance(L, dx, wavelength):
    """Distance beyond which the band limit cuts below the grid Nyquist frequency 1/(2 dx), i.e. where
    an unlimited transfer function starts to alias. Zero when dx <= wavelength / 2."""
    r = 2 * dx / wavelength
    return 0.0 if r <= 1 else L / 2 * np.sqrt(r**2 - 1)


def transfer_function(shape, dx, wavelength, z, n=1.0, bandlimit=True, paraxial=False):
    """exp(i kz z) on the FFT frequency grid.

    Evanescent waves decay as exp(-kappa |z|) in both directions, so back-propagation stays bounded.
    """
    lam = wavelength / n
    f = frequencies(shape, dx)
    f2 = sum(fi**2 for fi in f)
    if paraxial:
        H = np.exp(2j * np.pi * z / lam) * np.exp(-1j * np.pi * lam * z * f2)
    else:
        arg = 1 / lam**2 - f2
        kz = 2 * np.pi * np.sqrt(np.abs(arg))
        H = np.where(arg >= 0, np.exp(1j * kz * z), np.exp(-kz * abs(z)))
    if bandlimit:
        for fi, N in zip(f, shape):
            H = H * (np.abs(fi) <= band_limit(N * dx, lam, z, paraxial))
    return H


def propagate(E0, dx, wavelength, z, n=1.0, bandlimit=True, paraxial=False, pad=False):
    """Field after a distance z of homogeneous medium of index n.

    No fftshift is needed: the transfer function depends only on f, and fft/ifft are applied
    on the same grid, so where x = 0 sits does not matter.
    pad=True doubles the window with zeros (as in Matsushima and Shimobaba) and crops back,
    which stops light leaving one side from re-entering on the other.
    """
    E0 = np.asarray(E0, dtype=complex)
    if pad:
        widths = [(N // 2, N - N // 2) for N in E0.shape]
        E = propagate(np.pad(E0, widths), dx, wavelength, z, n, bandlimit, paraxial)
        return E[tuple(slice(a, a + N) for (a, _), N in zip(widths, E0.shape))]
    H = transfer_function(E0.shape, dx, wavelength, z, n, bandlimit, paraxial)
    return np.fft.ifftn(np.fft.fftn(E0) * H)


def power(E, dx):
    """Integral of |E|^2 over the window."""
    return np.sum(np.abs(E) ** 2) * dx**np.ndim(E)


def propagating_power(E, dx, wavelength, n=1.0):
    """Power carried by the plane waves with |f| <= n / wavelength (Parseval)."""
    f2 = sum(fi**2 for fi in frequencies(np.shape(E), dx))
    Ek = np.fft.fftn(E)
    return np.sum(np.abs(Ek[f2 <= (n / wavelength) ** 2]) ** 2) / np.size(E) * dx**np.ndim(E)
