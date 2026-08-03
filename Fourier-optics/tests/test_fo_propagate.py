import numpy as np
import pytest

from fourier_optics import (
    band_limit,
    frequencies,
    gaussian,
    grid,
    power,
    propagate,
    propagating_power,
    transfer_function,
)

LAM = 0.5


def lowpass_random(shape, dx, fmax, seed):
    """Random complex field whose spectrum is confined to |f| < fmax."""
    rng = np.random.default_rng(seed)
    Ek = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    f2 = sum(fi**2 for fi in frequencies(shape, dx))
    return np.fft.ifftn(Ek * (f2 < fmax**2))


@pytest.mark.parametrize("bandlimit", [False, True])
@pytest.mark.parametrize("z", [10.0, 1e3, 1e5])
def test_power_conserved_for_propagating_field(z, bandlimit):
    """Total power conserved to 1e-10 when every component propagates and passes the band limit."""
    dx, N = 0.25, 128
    fmax = 0.9 * min(1 / LAM, band_limit(N * dx, LAM, z)) if bandlimit else 0.9 / LAM
    E0 = lowpass_random((N, N), dx, fmax, seed=1)
    E = propagate(E0, dx, LAM, z, bandlimit=bandlimit)
    assert abs(power(E, dx) / power(E0, dx) - 1) < 1e-10


def test_evanescent_power_decays_and_propagating_power_survives():
    """A field with evanescent content keeps exactly its propagating power after many decay lengths."""
    dx, N = 0.1, 128  # Nyquist 5/um, well beyond 1/lambda = 2/um
    E0 = lowpass_random((N, N), dx, 5.0, seed=2)
    P_prop = propagating_power(E0, dx, LAM)
    assert P_prop < 0.5 * power(E0, dx)
    E = propagate(E0, dx, LAM, 50.0, bandlimit=False)
    assert abs(power(E, dx) / P_prop - 1) < 1e-10


@pytest.mark.parametrize("n", [1.0, 1.5])
@pytest.mark.parametrize("m", [(0, 0), (3, -5), (24, 0)])  # (24, 0) is at 70 degrees
def test_plane_wave_stays_plane_wave(m, n):
    """A grid-harmonic plane wave only acquires exp(i kz z), kz = 2 pi sqrt((n/lam)^2 - f^2)."""
    dx, N, z = 0.2, 64, 123.4
    x = grid(N, dx)
    f = np.array(m) / (N * dx)
    X, Y = np.meshgrid(x, x, indexing="ij")
    E0 = np.exp(2j * np.pi * (f[0] * X + f[1] * Y))
    kz = 2 * np.pi * np.sqrt((n / LAM) ** 2 - f @ f)
    E = propagate(E0, dx, LAM, z, n=n, bandlimit=False)
    assert np.max(np.abs(E - E0 * np.exp(1j * kz * z))) < 1e-10


def test_forward_then_backward_is_identity():
    dx, N = 0.25, 128
    E0 = lowpass_random((N, N), dx, 1.5, seed=3)
    E = propagate(propagate(E0, dx, LAM, 500.0, bandlimit=False), dx, LAM, -500.0, bandlimit=False)
    assert np.max(np.abs(E - E0)) < 1e-12 * np.max(np.abs(E0))


def test_steps_compose():
    """Without band limiting, H(z1) H(z2) = H(z1 + z2) exactly."""
    dx, N = 0.25, 128
    E0 = lowpass_random((N, N), dx, 1.9, seed=4)
    one = propagate(E0, dx, LAM, 300.0, bandlimit=False)
    two = propagate(propagate(E0, dx, LAM, 100.0, bandlimit=False), dx, LAM, 200.0, bandlimit=False)
    assert np.max(np.abs(one - two)) < 1e-12 * np.max(np.abs(E0))


def test_translation_equivariance_needs_no_fftshift():
    """Shifting the input by whole samples shifts the output: the origin of the grid is irrelevant."""
    dx, N = 0.5, 256
    x = grid(N, dx)
    X, Y = np.meshgrid(x, x, indexing="ij")
    E0 = gaussian(X - 7.0, Y + 3.0, 5.0, LAM)
    a = np.roll(propagate(E0, dx, LAM, 200.0), (10, -4), axis=(0, 1))
    b = propagate(np.roll(E0, (10, -4), axis=(0, 1)), dx, LAM, 200.0)
    assert np.max(np.abs(a - b)) < 1e-14


def test_fftshifted_transfer_function_gives_phase_ramp():
    """Negative control for the brief's pitfall: applying an fftshift-ordered H to an unshifted spectrum is wrong."""
    dx, N = 0.5, 256
    x = grid(N, dx)
    E0 = gaussian(x, 0.0, 5.0, LAM)
    good = propagate(E0, dx, LAM, 100.0)
    H = np.fft.fftshift(transfer_function(E0.shape, dx, LAM, 100.0))
    bad = np.fft.ifft(np.fft.fft(E0) * H)
    assert np.max(np.abs(bad - good)) > 0.1 * np.max(np.abs(good))


def test_one_dimensional_field_is_y_invariant_slice_of_two_dimensional():
    dx, N = 0.5, 128
    x = grid(N, dx)
    e1 = gaussian(x, 0.0, 3.0, LAM)
    E2 = np.repeat(e1[:, None], 8, axis=1)
    a = propagate(e1, dx, LAM, 80.0, bandlimit=False)
    b = propagate(E2, dx, LAM, 80.0, bandlimit=False)
    assert np.max(np.abs(b - a[:, None])) < 1e-13


def test_paraxial_transfer_function_matches_exact_for_small_angles():
    """Fresnel TF = small-angle limit: the error is the quartic term, z k_rho^4 / (8 k^3)."""
    dx, N, z = 0.5, 256, 1000.0
    x = grid(N, dx)
    X, Y = np.meshgrid(x, x, indexing="ij")
    E0 = gaussian(X, Y, 20.0, LAM)
    a = propagate(E0, dx, LAM, z, bandlimit=False)
    b = propagate(E0, dx, LAM, z, bandlimit=False, paraxial=True)
    k = 2 * np.pi / LAM
    quartic = z * 8 / 20.0**4 / (8 * k**3)  # z <k_rho^4> / (8 k^3), <k_rho^4> = 8 / w0^4
    assert np.linalg.norm(a - b) / np.linalg.norm(a) < 3 * quartic


def test_band_limit_formula():
    """f_limit -> 1/lambda at z -> 0 and -> L / (2 lambda z) far away, the paraxial form."""
    L = 100.0
    assert band_limit(L, LAM, 0.0) == pytest.approx(1 / LAM)
    assert band_limit(L, LAM, 1e7) == pytest.approx(band_limit(L, LAM, 1e7, paraxial=True), rel=1e-8)
