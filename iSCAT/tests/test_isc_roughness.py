import numpy as np
import pytest

from iscat import (
    DipoleImager,
    camera_grid,
    contrast,
    fresnel,
    gaussian_psd,
    polarizability,
    rough_surface,
    roughness_field,
    speckle_contrast_std,
    vectorial_psf,
    vectorial_reference,
)

LAM0, N1, N2, NA = 532.0, 1.52, 1.33, 1.4
K1 = 2 * np.pi * N1 / LAM0
R = fresnel(N1, N2)[0]


@pytest.mark.parametrize("zf", [0.0, 150.0])
def test_uniform_height_is_a_shifted_interface(zf):
    """A uniform glass layer reproduces the phase of the shifted-interface reflection."""
    zeta0 = 0.2
    E = roughness_field(np.full((32, 32), zeta0), 20.0, LAM0, N1, N2, NA, zf)
    er = vectorial_reference(LAM0, N1, N2, zf)
    np.testing.assert_allclose(E, 2j * K1 * zeta0 * er, rtol=1e-12)


@pytest.mark.parametrize("zf", [0.0, 150.0])
def test_single_bump_is_a_dipole_image(zf):
    """One raised pixel images as a dipole (m^2 - 1) zeta dx^2 at h = 0."""
    n, dx = 1024, 40.0
    zeta = np.zeros((n, n))
    zeta[0, 0] = 1.0
    E = roughness_field(zeta, dx, LAM0, N1, N2, NA, zf)
    i = np.arange(0, 20)
    sub = E[(4 * i) % n, (3 * i) % n]
    ref = vectorial_psf(3 * i * dx, 4 * i * dx, 0.0, ((N1 / N2) ** 2 - 1) * dx**2, LAM0, N1, N2, NA, zf)[0]
    assert np.abs(sub - ref).max() < 1e-3 * np.abs(ref[0])


def test_rough_surface_statistics():
    """Generated heights."""
    rng = np.random.default_rng(5)
    sigma, ell, dx = 0.3, 20.0, 2.0
    z = rough_surface(rng, 1024, dx, sigma, ell)
    assert abs(z.std() / sigma - 1) < 0.08
    lag = int(ell / dx)
    corr = np.mean(z * np.roll(z, lag, axis=1)) / z.var()
    assert abs(corr - np.exp(-1)) < 0.08 * np.exp(-1) + 0.02


@pytest.mark.parametrize("ell,zf,scatter", [(20.0, 0.0, 0.0142), (20.0, 200.0, 0.0047), (150.0, 100.0, 0.0053)])
def test_speckle_std_matches_spectral_integral(ell, zf, scatter):
    """Monte Carlo speckle (one 41 um map) vs the radial integral 4 int S (Re G)^2."""
    rng = np.random.default_rng(6)
    z = rough_surface(rng, 1024, 40.0, 0.3, ell)
    E = roughness_field(z, 40.0, LAM0, N1, N2, NA, zf)
    c = 2 * (E / vectorial_reference(LAM0, N1, N2, zf)).real
    assert abs(c.std() / speckle_contrast_std(0.3, ell, LAM0, N1, N2, NA, zf) - 1) < 3.5 * scatter


def test_focus_on_interface_speckle_needs_supercritical_light():
    """With zf = 0 the subcritical transfer (kappa < k2) is purely imaginary relative to E_r."""
    for na in (0.8, 1.2, 1.33):
        assert speckle_contrast_std(0.3, 20.0, LAM0, N1, N2, na, 0.0) < 1e-15
    assert speckle_contrast_std(0.3, 20.0, LAM0, N1, N2, 1.4, 0.0) > 1e-4
    assert speckle_contrast_std(0.3, 20.0, LAM0, N1, N2, 1.2, 100.0) > 1e-4


def test_speckle_asymptotes():
    """sigma_C ∝ ell for ell << lambda (flat PSD, S = pi ell^2 sigma^2)."""
    s = lambda ell, na, zf: speckle_contrast_std(0.3, ell, LAM0, N1, N2, na, zf)
    assert abs(np.log(s(2.0, 1.4, 0.0) / s(1.0, 1.4, 0.0)) / np.log(2) - 1) < 1e-3
    assert abs(np.log(s(8000.0, 1.2, 200.0) / s(4000.0, 1.2, 200.0)) / np.log(2) + 2) < 1e-2
    assert gaussian_psd(0.0, 0.3, 20.0) == pytest.approx(np.pi * 400 * 0.09)


def test_ratiometric_image_cancels_static_speckle():
    """Dividing by a particle-free frame removes static speckle to second order."""
    rng = np.random.default_rng(8)
    n, dx = 128, 20.0
    z = rough_surface(rng, n, dx, 0.5, 60.0)
    e = roughness_field(z, dx, LAM0, N1, N2, NA, 0.0)
    x, y = camera_grid(n, dx)
    imager = DipoleImager(LAM0, N1, N2, NA, rho_max=1900.0, d_rho=0.5)
    es, ey = imager.field(x, y, 20.0, polarizability(20.0, 1.59 / N2))
    er = vectorial_reference(LAM0, N1, N2)
    smooth = contrast(es, er, ey)
    rough_ratio = (np.abs(er + e + es) ** 2 + np.abs(ey) ** 2) / np.abs(er + e) ** 2 - 1
    resid = np.abs(rough_ratio - smooth)
    u, v = np.abs(es / er).max(), np.abs(e / er).max()
    assert resid.max() < 2.2 * u * v
    assert resid.std() < 0.2 * (2 * (e / er).real).std()


def test_speckle_statistics_independent_of_seed():
    """Invariance."""
    out = []
    for seed in (11, 12):
        z = rough_surface(np.random.default_rng(seed), 1024, 40.0, 0.3, 20.0)
        e = roughness_field(z, 40.0, LAM0, N1, N2, NA, 200.0)
        out.append((2 * (e / vectorial_reference(LAM0, N1, N2, 200.0)).real).std())
    assert abs(out[0] / out[1] - 1) < 3.5 * np.sqrt(2) * 0.0047


def test_flat_surface_and_index_matched_glass_give_no_speckle():
    """Negative controls."""
    z = rough_surface(np.random.default_rng(1), 64, 20.0, 0.3, 20.0)
    assert np.all(roughness_field(np.zeros((64, 64)), 20.0, LAM0, N1, N2, NA) == 0)
    assert np.abs(roughness_field(z, 20.0, LAM0, N2, N2, 1.3)).max() == 0
