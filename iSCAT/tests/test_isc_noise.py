import numpy as np
import pytest

from iscat import (
    camera_grid,
    contrast,
    contrast_noise,
    detection_limit,
    matched_filter,
    polarizability,
    ratiometric_contrast,
    snr_single_pixel,
    vectorial_psf,
    vectorial_reference,
)

LAM0, N1, N2, NA = 532.0, 1.52, 1.33, 1.4
M_PS = 1.59 / N2
ER = vectorial_reference(LAM0, N1, N2)


def psf_maps(a=20.0, h=20.0, npix=21, pixel=60.0):
    x, y = camera_grid(npix, pixel)
    ex, ey = vectorial_psf(x, y, h, polarizability(a, M_PS), LAM0, N1, N2, NA)
    C = contrast(ex, ER, ey)
    return C, 1 + C


def within(value, target, se, n_se=3.5):
    return abs(value - target) < n_se * se


def test_background_noise_is_one_over_sqrt_n():
    rng = np.random.default_rng(1)
    N = 1e4
    img = ratiometric_contrast(rng, np.ones((300, 300)), 1.0, N)
    n = img.size
    assert within(img.std(), 1 / np.sqrt(N), 1 / np.sqrt(N) / np.sqrt(2 * n))
    assert within(img.mean(), 0.0, 1 / np.sqrt(N * n))


@pytest.mark.parametrize("n_bg", [None, 1, 4])
def test_simulated_peak_contrast_and_noise(n_bg):
    """Centre-pixel contrast over 1000 noisy images."""
    rng = np.random.default_rng(2)
    C, I = psf_maps()
    c = C.shape[0] // 2
    N, M = 1e5, 1000
    samples = np.array([ratiometric_contrast(rng, I, 1.0, N, n_bg_frames=n_bg)[c, c] for _ in range(M)])
    sigma = contrast_noise(I[c, c], 1.0, N, n_bg_frames=n_bg)
    assert within(samples.mean(), C[c, c], sigma / np.sqrt(M))
    assert within(samples.std(ddof=1), sigma, sigma / np.sqrt(2 * (M - 1)))


@pytest.mark.parametrize("n_bg", [None, 1])
def test_matched_filter_amplitude_and_error(n_bg):
    """Least-squares fit of the known PSF shape."""
    rng = np.random.default_rng(3)
    C, I = psf_maps()
    c = C.shape[0] // 2
    T = C / C[c, c]
    N, M = 1e5, 1000
    sig = contrast_noise(I, 1.0, N, n_bg_frames=n_bg)
    fits = [matched_filter(ratiometric_contrast(rng, I, 1.0, N, n_bg_frames=n_bg), T, sig) for _ in range(M)]
    amps = np.array([f[0] for f in fits])
    err = fits[0][1]
    assert within(amps.mean(), C[c, c], err / np.sqrt(M))
    assert within(amps.std(ddof=1), err, err / np.sqrt(2 * (M - 1)))
    white = 1 / np.sqrt(N * np.sum(T**2)) * (1 if n_bg is None else np.sqrt(2))
    assert err == pytest.approx(white, rel=0.05)
    assert err < 0.5 / np.sqrt(N)


def test_detection_limit_scales_as_n_to_minus_one_sixth():
    """Shot-noise threshold 3/sqrt(N) on the full contrast (interference + |s|^2)."""
    def peak(a):
        ex, ey = vectorial_psf(0.0, 0.0, 20.0, polarizability(a, M_PS), LAM0, N1, N2, NA)
        return contrast(ex, ER, ey)

    Ns = np.logspace(4, 10, 7)
    amin = np.array([detection_limit(peak, 3 / np.sqrt(N)) for N in Ns])
    slope = np.polyfit(np.log(Ns), np.log(amin), 1)[0]
    assert slope == pytest.approx(-1 / 6, abs=2e-3)
    assert abs(peak(amin[2])) == pytest.approx(3e-3, rel=1e-9)


def sample_snr(rng, r, s, n_photons_per_pixel, M):
    """Mean contrast and its SNR over M centre pixels, background counts n_photons."""
    C = ratiometric_contrast(rng, np.full(M, abs(1 + s / r) ** 2), 1.0, n_photons_per_pixel)
    snr = C.mean() / C.std(ddof=1)
    return C.mean(), snr, np.sqrt((1 + snr**2 / 2) / M)


def test_signal_to_noise_independent_of_reference_at_fixed_illumination():
    """A 3x weaker reference triples the contrast but not the SNR at fixed illumination."""
    rng = np.random.default_rng(4)
    s = -2e-4
    N_inc, M = 1e9, 4000
    (c1, snr1, se1), (c2, snr2, se2) = [sample_snr(rng, r, s, N_inc * r**2, M) for r in (0.0667, 0.0667 / 3)]
    assert c2 / c1 == pytest.approx(3, rel=0.05)
    assert within(snr1 - snr2, 0.0, np.hypot(se1, se2))
    assert within(abs(snr1), snr_single_pixel(s, 0.0667, N_inc), se1)
    assert snr_single_pixel(s, 0.0667, N_inc) == pytest.approx(2 * abs(s) * np.sqrt(N_inc), rel=4e-3)


def test_strong_attenuation_approaches_dark_field():
    """With |r| << |s| the SNR falls from 2|s| sqrt(N) to the dark-field |s| sqrt(N)."""
    s, N = -2e-4, 1e9
    assert snr_single_pixel(s, 1e-7, N) == pytest.approx(abs(s) * np.sqrt(N), rel=1e-3)
    rng = np.random.default_rng(6)
    _, snr, se = sample_snr(rng, 0.0667 / 30, s, N * (0.0667 / 30) ** 2, 4000)
    assert within(abs(snr), snr_single_pixel(s, 0.0667 / 30, N), se)
    assert not within(abs(snr), 2 * abs(s) * np.sqrt(N), se)


def test_signal_to_noise_grows_as_reference_falls_at_fixed_counts():
    """At a fixed number of detected photons per pixel (camera full well), SNR ~ 1/|r|."""
    rng = np.random.default_rng(5)
    s, N, M = -2e-4, 1e6, 4000
    (_, snr1, se1), (_, snr2, se2) = [sample_snr(rng, r, s, N, M) for r in (0.0667, 0.0667 / 3)]
    assert within(snr2 / snr1, 3.0, 3 * np.hypot(se1 / snr1, se2 / snr2))


def test_fixed_focus_breaks_one_sixth_law():
    """Particle resting on the glass (h = a), focus on the interface."""
    from iscat import best_focus, dipole_amplitude, scalar_peak, scalar_reference

    k = 2 * np.pi * N2 / LAM0

    def scalar(a):
        s = scalar_peak(a, dipole_amplitude(polarizability(a, M_PS), k), LAM0, N2, 1.2, N1)
        return contrast(s, scalar_reference(LAM0, N2, N1, N2))

    Ns = np.array([1e12, 1e14])
    amin = [detection_limit(scalar, 3 / np.sqrt(N), a_lo=0.01) for N in Ns]
    assert np.diff(np.log(amin))[0] / np.diff(np.log(Ns))[0] == pytest.approx(-1 / 8, abs=3e-3)

    def vec(a, zf):
        ex, ey = vectorial_psf(0.0, 0.0, a, polarizability(a, M_PS), LAM0, N1, N2, NA, zf=zf)
        return contrast(ex, vectorial_reference(LAM0, N1, N2, zf), ey)

    Ns = np.array([1e6, 1e8])
    zf = best_focus(3.0, LAM0, N1, N2, NA)
    assert 150 < zf < 300
    for focus, lo, hi in ((0.0, -0.13, -0.11), (zf, -1 / 6 - 0.005, -1 / 6 + 0.005)):
        amin = [detection_limit(lambda a: vec(a, focus), 3 / np.sqrt(N), a_lo=0.05) for N in Ns]
        slope = np.diff(np.log(amin))[0] / np.diff(np.log(Ns))[0]
        assert lo < slope < hi
