"""3D localisation, Cramer-Rao bounds and MSD analysis."""

import numpy as np
from scipy.optimize import least_squares

from .imaging import contrast, vectorial_reference

class ParticleModel:
    """Camera intensity (units of |r E0|^2) of a dipole s * alpha at (x0, y0, h)."""

    def __init__(self, imager, x, y, alpha, lam0, n1, n2, zf=0.0):
        self.imager, self.x, self.y, self.alpha = imager, x, y, alpha
        self.er = vectorial_reference(lam0, n1, n2, zf)

    def __call__(self, p):
        x0, y0, h, s = p
        ex, ey = self.imager.field(self.x - x0, self.y - y0, h, s * self.alpha)
        return 1 + contrast(ex, self.er, ey)


def fisher_information(model, p, n_photons, steps=(0.5, 0.5, 0.5, 1e-3), free=None):
    """Poisson Fisher information F_jk = N sum dI/dp_j dI/dp_k / I."""
    p = np.asarray(p, dtype=float)
    free = range(len(p)) if free is None else free
    I = model(p)
    grads = []
    for j in free:
        dp = np.zeros_like(p)
        dp[j] = steps[j]
        grads.append(((model(p + dp) - model(p - dp)) / (2 * steps[j])).ravel())
    G = np.array(grads)
    return n_photons * (G / I.ravel()) @ G.T


def crlb(F):
    """Cramer-Rao lower bounds on the standard deviations."""
    return np.sqrt(np.diag(np.linalg.inv(F)))


def localise(image, model, guess, n_photons, free=(0, 1, 2, 3)):
    """Weighted least-squares fit of a contrast image; returns the full parameter vector."""
    guess = np.asarray(guess, dtype=float)
    free = list(free)
    w = np.sqrt(n_photons / model(guess))

    def full(q):
        p = guess.copy()
        p[free] = q
        return p

    res = lambda q: ((image + 1 - model(full(q))) * w).ravel()
    scale = np.array([50.0, 50.0, 50.0, 0.1])[free]
    sol = least_squares(res, guess[free], x_scale=scale, diff_step=1e-6, method="lm")
    return full(sol.x)


def brownian_track(rng, n_steps, D, dt, dim=2):
    """Positions of a free Brownian particle sampled every dt."""
    steps = rng.normal(0.0, np.sqrt(2 * D * dt), size=(n_steps, dim))
    return np.vstack([np.zeros((1, dim)), np.cumsum(steps, axis=0)])


def msd(track, max_lag):
    """Mean-squared displacement per axis for lags 1..max_lag."""
    track = np.asarray(track)
    return np.array([np.mean((track[..., lag:, :] - track[..., :-lag, :]) ** 2, axis=(-2, -1))
                     for lag in range(1, max_lag + 1)]).T


def fit_msd(msd_values, dt, n_lags=2):
    """(D, sigma^2) from MSD(n dt) = 2 D n dt + 2 sigma^2 over the first n_lags lags."""
    if n_lags < 2:
        raise ValueError("a slope and an intercept need at least two lags")
    msd_values = np.asarray(msd_values)
    lags = np.arange(1, n_lags + 1) * dt
    slope, icpt = np.polyfit(lags, msd_values[..., :n_lags].T, 1)
    return slope / 2, icpt / 2
