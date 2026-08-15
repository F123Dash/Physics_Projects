"""Static speckle background from coverslip roughness."""

import numpy as np

from .imaging import _pupil_ab, _pupil_nodes, fresnel, vectorial_reference, vectorial_transfer


def gaussian_psd(kappa, sigma, ell):
    """Gaussian-correlated surface spectrum pi ell^2 sigma^2 exp(-kappa^2 ell^2 / 4)."""
    return np.pi * ell ** 2 * sigma ** 2 * np.exp(-np.asarray(kappa) ** 2 * ell ** 2 / 4)


def _grid_k(n, dx):
    k = 2 * np.pi * np.fft.fftfreq(n, dx)
    return np.meshgrid(k, k, indexing="xy")


def rough_surface(rng, n, dx, sigma, ell):
    """Periodic n x n Gaussian random surface (nm)."""
    kx, ky = _grid_k(n, dx)
    w = np.fft.fft2(rng.normal(size=(n, n)))
    return np.fft.ifft2(w * np.sqrt(gaussian_psd(np.hypot(kx, ky), sigma, ell)) / dx).real


def roughness_field(zeta, dx, lam0, n1, n2, NA, zf=0.0):
    """Camera field E_x/E0 scattered by the height map zeta (periodic, spacing dx)."""
    kx, ky = _grid_k(zeta.shape[0], dx)
    Fx, _ = vectorial_transfer(kx, ky, 0.0, lam0, n1, n2, NA, zf)
    return np.fft.ifft2(np.fft.fft2(zeta) * ((n1 / n2) ** 2 - 1) * Fx)


def speckle_contrast_std(sigma, ell, lam0, n1, n2, NA, zf=0.0, order=400):
    """Predicted rms of the linear speckle contrast 2 Re(E_rough / E_r)."""
    k0 = 2 * np.pi / lam0
    k1, k2 = n1 * k0, n2 * k0
    kap, kz1, kz2, wt = _pupil_nodes(k1, k2, NA * k0, order)
    a, b = _pupil_ab(kap, kz1, kz2, 0.0, zf, k1, k2, n1, n2)
    P = ((n1 / n2) ** 2 - 1) * 2 * np.pi * 1j * k2 ** 2 * fresnel(n1, n2)[1] / (8 * np.pi)
    P = P / vectorial_reference(lam0, n1, n2, zf)
    ang = 2 * np.pi * (P * a).real ** 2 + np.pi * (P * b).real ** 2
    S = gaussian_psd(kap, sigma, ell)
    return np.sqrt(4 * np.sum(wt * S * ang) / (2 * np.pi) ** 2)
