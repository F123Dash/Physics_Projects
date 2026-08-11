"""Shot-noise-limited iSCAT images, contrast estimators and detection limits."""

import numpy as np
from scipy.optimize import brentq


def camera_grid(npix, pixel):
    """Pixel-centre coordinates (object space), with the particle at pixel npix // 2."""
    c = (np.arange(npix) - npix // 2) * pixel
    return np.meshgrid(c, c, indexing="xy")


def expected_counts(intensity, n_photons):
    """Mean photons per pixel."""
    return n_photons * np.asarray(intensity)


def ratiometric_contrast(rng, intensity, background, n_photons, n_frames=1, n_bg_frames=None):
    """Contrast image (sum of signal frames)/(sum of particle-free frames) - 1."""
    background = np.broadcast_to(background, np.shape(intensity))
    sig = rng.poisson(expected_counts(intensity, n_photons * n_frames))
    if n_bg_frames is None:
        bg = expected_counts(background, n_photons * n_frames)
    else:
        bg = rng.poisson(expected_counts(background, n_photons * n_bg_frames)) * (n_frames / n_bg_frames)
    return sig / bg - 1


def contrast_noise(intensity, background, n_photons, n_frames=1, n_bg_frames=None):
    """Predicted per-pixel standard deviation of ratiometric_contrast (delta method)."""
    I, B = np.asarray(intensity), np.asarray(background)
    var = I / (n_photons * n_frames * B ** 2)
    if n_bg_frames is not None:
        var = var + I ** 2 / (n_photons * n_bg_frames * B ** 3)
    return np.sqrt(var)


def matched_filter(image, template, sigma=None):
    """Least-squares amplitude of image = A template, and its standard error if sigma is given."""
    t = np.asarray(template)
    norm = np.sum(t ** 2)
    amp = np.sum(t * image, axis=(-2, -1)) / norm
    if sigma is None:
        return amp
    s2 = np.broadcast_to(np.asarray(sigma) ** 2, t.shape)
    return amp, np.sqrt(np.sum(t ** 2 * s2)) / norm


def detection_limit(contrast_of_radius, threshold, a_lo=0.1, a_hi=500.0):
    """Smallest radius with |contrast(a)| = threshold, by bisection in log a."""
    g = lambda la: np.log(abs(contrast_of_radius(np.exp(la)))) - np.log(threshold)
    if g(np.log(a_lo)) * g(np.log(a_hi)) > 0:
        raise ValueError("threshold not crossed between a_lo and a_hi")
    return float(np.exp(brentq(g, np.log(a_lo), np.log(a_hi), xtol=1e-12)))


def snr_single_pixel(s_over_e0, r, n_incident):
    """Exact single-pixel SNR of the contrast for n_incident photons per pixel at the sample."""
    i_tot = np.abs(r + s_over_e0) ** 2
    return np.abs(i_tot - np.abs(r) ** 2) * np.sqrt(n_incident) / np.sqrt(i_tot)
