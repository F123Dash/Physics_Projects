"""Diagnostics: widths, moments, conserved quantities, peaks, phase matching,
spectrograms and coherence."""

import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks

from .grid import C_NM_PS

fft = np.fft.fft
ifft = np.fft.ifft


# ---------------------------------------------------------------- moments
def energy(grid, A):
    """Pulse energy sum |A|^2 dt (W ps = pJ)."""
    return np.sum(np.abs(A) ** 2, axis=-1) * grid.dt


def photon_number(grid, Aw):
    """Quantity proportional to photon number: sum |Aw|^2 / (omega0 + Omega).

    Conserved by the full GNLSE (Kerr + self-steepening + Raman) without loss,
    while the energy is not (Raman transfers energy to the phonons).
    """
    w = grid.absolute_omega
    mask = w > 0
    return np.sum(np.abs(Aw[..., mask]) ** 2 / w[mask], axis=-1)


def rms_width(t, A):
    I = np.abs(A) ** 2
    m = np.sum(t * I) / np.sum(I)
    return np.sqrt(np.sum((t - m) ** 2 * I) / np.sum(I))


def mean_frequency(omega, Aw):
    S = np.abs(Aw) ** 2
    return np.sum(omega * S, axis=-1) / np.sum(S, axis=-1)


def fwhm(x, y):
    """Full width at half maximum of a single-peaked y(x), with linear interpolation."""
    y = np.asarray(y)
    i = int(np.argmax(y))
    half = y[i] / 2
    l = i
    while l > 0 and y[l] > half:
        l -= 1
    r = i
    while r < len(y) - 1 and y[r] > half:
        r += 1
    xl = np.interp(half, [y[l], y[l + 1]], [x[l], x[l + 1]])
    xr = np.interp(half, [y[r], y[r - 1]], [x[r], x[r - 1]])
    return xr - xl


def count_peaks(y, rel_height=0.01):
    """Number of local maxima above rel_height * max(y)."""
    peaks, _ = find_peaks(y, height=rel_height * np.max(y))
    return len(peaks)


# ---------------------------------------------------------------- spectra
def spectrum_wavelength(grid, Aw):
    """Spectral density per unit wavelength, |A(lambda)|^2 ∝ |A(omega)|^2 omega^2 / (2 pi c).

    Returns (wavelength_nm, density), both sorted by increasing wavelength,
    restricted to positive absolute frequency.
    """
    w = grid.absolute_omega
    mask = w > 0
    lam = 2 * np.pi * C_NM_PS / w[mask]
    S = np.abs(Aw[..., mask]) ** 2 * w[mask] ** 2 / (2 * np.pi * C_NM_PS)
    order = np.argsort(lam)
    return lam[order], S[..., order]


def to_db(S, ref=None):
    ref = np.max(S) if ref is None else ref
    with np.errstate(divide="ignore"):
        return 10 * np.log10(np.maximum(S, 1e-300) / ref)


def spectral_edges(lam, S, level_db=-40.0):
    """Shortest and longest wavelength where the density is above level_db (relative to max)."""
    above = np.nonzero(to_db(S) >= level_db)[0]
    return lam[above[0]], lam[above[-1]]


def edge_fraction(Aw, frac=0.05):
    """Fraction of spectral energy within ``frac`` of either end of the frequency window."""
    S = np.abs(np.fft.fftshift(Aw)) ** 2
    m = max(1, int(frac * len(S)))
    return (S[:m].sum() + S[-m:].sum()) / S.sum()


def time_edge_fraction(A, frac=0.05):
    I = np.abs(A) ** 2
    m = max(1, int(frac * len(I)))
    return (I[:m].sum() + I[-m:].sum()) / I.sum()


# ---------------------------------------------------------------- physics
def dispersive_wave_offset(fibre, P_s, omega_s=0.0, bracket=None, side=+1):
    """Solve the phase-matching condition for Cherenkov radiation from a soliton.

    beta(Omega) - beta(Omega_s) - beta1(Omega_s) (Omega - Omega_s) = gamma P_s / 2

    (Akhmediev & Karlsson 1995), with Omega offsets from the carrier. ``side=+1``
    looks for the blue-shifted root. Returns Omega in rad/ps.
    """
    b_s = float(fibre.beta(omega_s))
    b1_s = float(fibre.beta1_rel(omega_s))

    def f(w):
        return float(fibre.beta(w)) - b_s - b1_s * (w - omega_s) - fibre.gamma * P_s / 2

    if bracket is None:
        # march away from the soliton until the sign changes
        step = 0.01 * (1 + abs(omega_s))
        w_prev, f_prev = omega_s + side * step, f(omega_s + side * step)
        for _ in range(200000):
            w = w_prev + side * step
            fw = f(w)
            if np.sign(fw) != np.sign(f_prev):
                bracket = (min(w_prev, w), max(w_prev, w))
                break
            w_prev, f_prev = w, fw
            step *= 1.01
        else:
            raise RuntimeError("no phase-matched frequency found")
    return brentq(f, *bracket)


def omega_to_wavelength(grid, Omega):
    return 2 * np.pi * C_NM_PS / (grid.omega0 + Omega)


def wavelength_to_omega(grid, lam):
    return 2 * np.pi * C_NM_PS / lam - grid.omega0


def peak_power_evolution(At):
    return np.max(np.abs(At) ** 2, axis=-1)


def find_solitons(t, A, rel_height=0.05, min_sep=None):
    """Return (times, peak powers) of temporal peaks above rel_height * max."""
    I = np.abs(A) ** 2
    dist = None if min_sep is None else max(1, int(min_sep / (t[1] - t[0])))
    peaks, props = find_peaks(I, height=rel_height * I.max(), distance=dist)
    return t[peaks], props["peak_heights"]


def spectrogram(grid, A, gate_fwhm, n_delays=256, t_range=None):
    """Gated spectrogram S(tau, Omega) = |FT{A(t) g(t - tau)}|^2 with a Gaussian gate.

    Returns (delays, Omega_shifted, S) with S of shape (n_delays, n).
    """
    t = grid.t
    if t_range is None:
        t_range = (t[0], t[-1])
    delays = np.linspace(*t_range, n_delays)
    sigma = gate_fwhm / (2 * np.sqrt(2 * np.log(2)))
    gates = np.exp(-((t[None, :] - delays[:, None]) ** 2) / (2 * sigma**2))
    S = np.abs(np.fft.fftshift(ifft(A[None, :] * gates, axis=-1), axes=-1)) ** 2
    return delays, grid.omega_shifted, S


def coherence(Aws):
    """Modulus of the complex degree of first-order coherence at zero delay.

    |g12(omega)| = |<A_i*(omega) A_j(omega)>_{i != j}| / <|A(omega)|^2>
    computed over all distinct pairs of an ensemble of spectra (Dudley RMP eq. 9).
    """
    Aws = np.asarray(Aws)
    M = Aws.shape[0]
    s = Aws.sum(axis=0)
    # sum over i != j of A_i* A_j = |sum A|^2 - sum |A_i|^2
    p2 = np.sum(np.abs(Aws) ** 2, axis=0)
    cross = (np.abs(s) ** 2 - p2) / (M * (M - 1))
    mean_I = p2 / M
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mean_I > 0, np.abs(cross) / mean_I, 0.0)


def classify_solitons(grid, fibre, A, rel_height=0.01, min_sep=0.05, n_range=(0.6, 1.6)):
    """Identify fundamental solitons among the temporal peaks of A.

    For each peak: local duration T0 = FWHM/1.763 of the isolated peak, local
    carrier Omega_c from the spectrum of the field gated over +-2 FWHM, and local
    soliton number N_loc = sqrt(gamma P T0^2 / |beta2(Omega_c)|). A peak counts as
    a soliton when beta2(Omega_c) < 0, N_loc is within ``n_range`` and the peak
    is resolved (T0 >= 4 dt; narrower spikes are interference fringes).
    Returns a list of dicts (time, power, T0, wavelength, N_loc, soliton).
    """
    t = grid.t
    I = np.abs(A) ** 2
    dist = max(1, int(min_sep / grid.dt))
    peaks, props = find_peaks(I, height=rel_height * I.max(), distance=dist)
    out = []
    for p, P in zip(peaks, props["peak_heights"]):
        lo, hi = p, p
        while lo > 0 and I[lo] > P / 2:
            lo -= 1
        while hi < len(I) - 1 and I[hi] > P / 2:
            hi += 1
        width = (hi - lo) * grid.dt
        gate = np.abs(t - t[p]) < 2 * width
        S = np.abs(ifft(np.where(gate, A, 0))) ** 2
        w_c = np.sum(grid.omega * S) / np.sum(S)
        b2 = float(fibre.gvd(w_c))
        T0 = width / 1.7627
        N_loc = np.sqrt(fibre.gamma * P * T0**2 / abs(b2)) if b2 != 0 else np.inf
        out.append(dict(time=t[p], power=P, T0=T0, wavelength=omega_to_wavelength(grid, w_c),
                        N_loc=N_loc,
                        soliton=bool(b2 < 0 and n_range[0] <= N_loc <= n_range[1]
                                     and T0 >= 4 * grid.dt)))
    # a weaker candidate within 4 T0 of a stronger soliton is its shoulder
    sols = sorted((d for d in out if d["soliton"]), key=lambda d: -d["power"])
    for i, d in enumerate(sols):
        if any(abs(d["time"] - s["time"]) < 4 * s["T0"] for s in sols[:i] if s["soliton"]):
            d["soliton"] = False
    return out
