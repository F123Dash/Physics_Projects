import numpy as np
import pytest

from iscat import (
    DipoleImager,
    ParticleModel,
    brownian_track,
    camera_grid,
    crlb,
    fisher_information,
    fit_msd,
    localise,
    mie_dipole_polarizability,
    msd,
    ratiometric_contrast,
    vectorial_psf,
)

LAM0, N1, N2, NA = 532.0, 1.52, 1.33, 1.4
K = 2 * np.pi * N2 / LAM0
M_AU = (0.5439 + 2.2309j) / N2
ALPHA = mie_dipole_polarizability(10.0, M_AU, K)


@pytest.fixture(scope="module")
def model():
    x, y = camera_grid(15, 60.0)
    imager = DipoleImager(LAM0, N1, N2, NA, rho_max=900.0, d_rho=1.0, order=100)
    return ParticleModel(imager, x, y, ALPHA, LAM0, N1, N2)


@pytest.mark.parametrize("zf", [0.0, 150.0])
def test_fast_imager_matches_vectorial_psf(zf):
    """Tabulated Bessel functions + linear interpolation vs direct quadrature (same nodes)."""
    x, y = np.meshgrid(np.linspace(-900, 900, 19), np.linspace(-893, 907, 19))
    imager = DipoleImager(LAM0, N1, N2, NA, zf=zf, rho_max=1300.0, d_rho=0.5, order=300)
    for h in (5.0, 60.0, 250.0):
        ex, ey = vectorial_psf(x, y, h, 1.0, LAM0, N1, N2, NA, zf=zf)
        fx, fy = imager.field(x, y, h, 1.0)
        scale = np.abs(ex).max()
        assert np.abs(fx - ex).max() < 3.4e-5 * scale
        assert np.abs(fy - ey).max() < 3.4e-5 * scale


def test_fisher_information_of_a_poisson_rate():
    """Toy model I_i = p0 c_i on 50 pixels."""
    c = np.linspace(0.2, 3.0, 50)
    F = fisher_information(lambda p: p[0] * c, [1.7], 1e4, steps=(1e-3,))
    assert F[0, 0] == pytest.approx(1e4 * c.sum() / 1.7, rel=1e-10)


def test_fit_reaches_cramer_rao_bound(model):
    """150 shot-noise images (N = 1e4) of gold a = 10 nm at h = 60 nm."""
    rng = np.random.default_rng(3)
    p, N, M = np.array([13.0, -7.0, 60.0, 1.0]), 1e4, 150
    I = model(p)
    est = np.array([localise(ratiometric_contrast(rng, I, 1.0, N), model,
                             p + rng.normal(0, [5, 5, 3, 0.03]), N) for _ in range(M)])
    bound = crlb(fisher_information(model, p, N))
    std = est.std(axis=0, ddof=1)
    assert np.all(np.abs(est.mean(axis=0) - p) < 3.5 * bound / np.sqrt(M))
    assert np.all(np.abs(std - bound) < 3.5 * bound / np.sqrt(2 * (M - 1)))


def test_axial_best_and_lateral_worst_at_contrast_zero(model):
    """At a contrast zero sigma_z is smallest and sigma_x largest."""
    N = 1e4
    hs = np.arange(40.0, 142.0, 2.0)
    c0 = np.array([model([0.0, 0.0, h, 1.0])[7, 7] - 1 for h in hs])
    j = np.nonzero(np.diff(np.sign(c0)))[0][0]
    h0 = hs[j] - c0[j] * (hs[j + 1] - hs[j]) / (c0[j + 1] - c0[j])
    b = np.array([crlb(fisher_information(model, [13.0, -7.0, h, 1.0], N)) for h in hs])
    assert abs(hs[np.argmin(b[:, 2])] - h0) < 10
    assert abs(hs[np.argmax(b[:, 0])] - h0) < 10
    assert b[:, 2].max() > 3 * b[:, 2].min()


def test_known_amplitude_tightens_axial_bound(model):
    """Fixing the amplitude can only lower each bound (Schur complement)."""
    ratio = {}
    for h in (25.0, 90.0):
        p = [13.0, -7.0, h, 1.0]
        b4 = crlb(fisher_information(model, p, 1e4))
        b3 = crlb(fisher_information(model, p, 1e4, free=(0, 1, 2)))
        assert np.all(b3 <= b4[:3] * (1 + 1e-9))
        ratio[h] = b3[2] / b4[2]
    assert ratio[25.0] < 0.7
    assert ratio[90.0] > 0.97


def test_msd_fit_recovers_diffusion_and_localisation_error():
    """400 tracks of 200 steps with Gaussian localisation noise sigma^2 = D dt."""
    rng = np.random.default_rng(4)
    D, dt, sig = 5e5, 2e-5, np.sqrt(5e5 * 2e-5)
    tracks = np.array([brownian_track(rng, 200, D, dt) for _ in range(400)])
    noisy = tracks + rng.normal(0.0, sig, tracks.shape)
    d_est, s2_est = fit_msd(msd(noisy, 2), dt)
    n = len(d_est)
    assert abs(d_est.mean() - D) < 3.5 * d_est.std(ddof=1) / np.sqrt(n)
    assert abs(s2_est.mean() - sig**2) < 3.5 * s2_est.std(ddof=1) / np.sqrt(n)
    d_clean, s2_clean = fit_msd(msd(tracks, 2), dt)
    assert abs(s2_clean.mean()) < 3.5 * s2_clean.std(ddof=1) / np.sqrt(n)


def test_particle_free_images_give_zero_amplitude(model):
    """Negative control."""
    rng = np.random.default_rng(14)
    p, N, M = np.array([13.0, -7.0, 60.0, 0.0]), 1e4, 200
    est = np.array([localise(ratiometric_contrast(rng, np.ones(model.x.shape), 1.0, N), model, p, N, free=(3,))[3]
                    for _ in range(M)])
    bound = crlb(fisher_information(model, p, N, free=(3,)))[0]
    assert abs(est.mean()) < 3.5 * bound / np.sqrt(M)
    assert abs(est.std(ddof=1) - bound) < 3.5 * bound / np.sqrt(2 * (M - 1))


def test_bounds_invariant_under_whole_pixel_translation(model):
    """Bounds are unchanged when particle and camera move together by whole pixels."""
    shifted = ParticleModel(model.imager, model.x + 120.0, model.y - 60.0, ALPHA, LAM0, N1, N2)
    b0 = crlb(fisher_information(model, [13.0, -7.0, 60.0, 1.0], 1e4))
    b1 = crlb(fisher_information(shifted, [133.0, -67.0, 60.0, 1.0], 1e4))
    np.testing.assert_allclose(b1, b0, rtol=1e-4)


def test_msd_fit_needs_two_lags():
    with pytest.raises(ValueError):
        fit_msd(np.ones(5), 1e-3, n_lags=1)
