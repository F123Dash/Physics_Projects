import numpy as np
import pytest
from scipy.special import j1

from iscat.imaging import _pupil_ab, _pupil_nodes
from iscat import (
    DipoleImager,
    detection_limit,
    mie_efficiencies,
    contrast,
    dipole_amplitude,
    fresnel,
    fresnel_angular,
    polarizability,
    scalar_peak,
    scalar_psf,
    scalar_reference,
    vectorial_psf,
    vectorial_reference,
    vectorial_window_signal,
    wrap,
)

LAM0, N1, N2 = 532.0, 1.52, 1.33
K2 = 2 * np.pi * N2 / LAM0
LAM_M = LAM0 / N2
ALPHA_PS = polarizability(20.0, 1.59 / N2)
F_PS = dipole_amplitude(ALPHA_PS, K2)


def phase_period(h, field):
    """Fringe period from a straight-line fit to the unwrapped phase."""
    slope = np.polyfit(h, np.unwrap(np.angle(field)), 1)[0]
    return 2 * np.pi / slope


def fwhm(r, profile):
    p = profile / profile[0]
    i = np.argmax(p < 0.5)
    return 2 * np.interp(0.5, [p[i], p[i - 1]], [r[i], r[i - 1]])


def test_reference_has_no_pi_shift_and_brief_reflectivity():
    """Glass -> water reflection seen from the glass."""
    r, t, tp = fresnel(N1, N2)
    assert r > 0
    assert r**2 == pytest.approx(0.004444, rel=1e-3)
    assert 1 + r == pytest.approx(t)


@pytest.mark.parametrize("sin_t", [0.0, 0.3, 0.8])
def test_fresnel_energy_conservation(sin_t):
    k0 = 2 * np.pi / LAM0
    kap = sin_t * N2 * k0
    kz2, kz1 = np.sqrt((N2 * k0) ** 2 - kap**2), np.sqrt((N1 * k0) ** 2 - kap**2)
    r_s, r_p, t_s, t_p = fresnel_angular(kz2, kz1, N2, N1)
    assert abs(r_s) ** 2 + kz1 / kz2 * abs(t_s) ** 2 == pytest.approx(1, abs=1e-14)
    assert abs(r_p) ** 2 + kz1 / kz2 * abs(t_p) ** 2 == pytest.approx(1, abs=1e-14)
    if sin_t == 0:
        assert t_p == pytest.approx(t_s)


@pytest.mark.parametrize("NA", [0.3, 1.0, 1.33])
@pytest.mark.parametrize("h,zf", [(0.0, 0.0), (35.0, 0.0), (120.0, 60.0), (40.0, 40.0)])
def test_scalar_closed_form_matches_quadrature(NA, h, zf):
    assert scalar_psf(0.0, h, F_PS, LAM0, N2, NA, N1, zf) == pytest.approx(
        scalar_peak(h, F_PS, LAM0, N2, NA, N1, zf), rel=1e-12)


def test_scalar_low_na_psf_is_airy():
    NA = 0.05
    rho = np.linspace(1, 15000, 50)
    v = 2 * np.pi / LAM0 * NA * rho
    psf = scalar_psf(rho, 0.0, 1.0, LAM0, N2, NA, N1)
    psf0 = scalar_psf(0.0, 0.0, 1.0, LAM0, N2, NA, N1)
    assert np.max(np.abs(psf / psf0 - 2 * j1(v) / v)) < (NA / N2) ** 2


def test_gouy_phase_quarter_wave_in_focus():
    """At h = zf = 0 the image of a lossless dipole is i times the reference."""
    s = scalar_peak(0.0, F_PS, LAM0, N2, 1.0, N1) / scalar_reference(LAM0, N2, N1, N2)
    assert abs(wrap(np.angle(s) - np.pi / 2)) < 1e-14
    ex, _ = vectorial_psf(0.0, 0.0, 0.0, ALPHA_PS, LAM0, N1, N2, 1.2)
    assert abs(wrap(np.angle(ex / vectorial_reference(LAM0, N1, N2)) - np.pi / 2)) < 1e-12


def test_particle_on_glass_is_dark():
    for NA in (0.5, 1.4):
        ex, ey = vectorial_psf(0.0, 0.0, 20.0, ALPHA_PS, LAM0, N1, N2, NA)
        assert contrast(ex, vectorial_reference(LAM0, N1, N2), ey) < 0
    s = scalar_peak(20.0, F_PS, LAM0, N2, 1.0, N1)
    assert contrast(s, scalar_reference(LAM0, N2, N1, N2)) < 0


@pytest.mark.parametrize("NA", [0.3, 0.9, 1.2, 1.33])
def test_scalar_fixed_focus_period(NA):
    """Focus on the interface."""
    c = np.sqrt(1 - (NA / N2) ** 2)
    h = np.linspace(1, 150, 60)
    s = scalar_peak(h, F_PS, LAM0, N2, NA, N1)
    assert phase_period(h, s) == pytest.approx(2 * LAM_M / (3 + c), rel=1e-10)


def test_brief_period_bar_at_low_na():
    """The brief's lambda_m / 2 fringe period: within 1% at NA 0.3, more than 10% off at NA 1.4."""
    h = np.linspace(0, 400, 201)
    sc = scalar_peak(h, F_PS, LAM0, N2, 0.3, N1)
    assert phase_period(h, sc) == pytest.approx(LAM_M / 2, rel=0.01)
    for NA, ok in ((0.3, True), (1.4, False)):
        ex = np.array([vectorial_psf(0.0, 0.0, hh, ALPHA_PS, LAM0, N1, N2, NA)[0] for hh in h])
        dev = phase_period(h, ex) / (LAM_M / 2) - 1
        assert (abs(dev) < 0.01) == ok
        if not ok:
            assert dev > 0.10


def test_scalar_focus_tracking_period_exact():
    """With the focus on the particle (zf = h) the scalar phase is pi/2 + 2kh exactly."""
    h = np.linspace(0, 400, 50)
    s = scalar_peak(h, F_PS, LAM0, N2, 1.2, N1, zf=h) / scalar_reference(LAM0, N2, N1, N2, zf=h)
    assert phase_period(h, s) == pytest.approx(LAM_M / 2, rel=1e-12)


@pytest.mark.parametrize("W", [1000.0, 3000.0])
def test_window_integrated_period(W):
    """Signal summed over a Gaussian window of radius W."""
    h = np.linspace(0, 800, 161)
    s = np.array([vectorial_window_signal(hh, ALPHA_PS, LAM0, N1, N2, 1.4, W) for hh in h])
    pred = LAM_M / 2 * (1 + 1 / (K2 * W) ** 2)
    assert phase_period(h, s) == pytest.approx(pred, rel=1e-4)


def test_scalar_defocus_matches_fourier_optics_propagator():
    """The defocused scalar image equals the in-focus image propagated with the Fourier-optics propagator."""
    fo = pytest.importorskip("fourier_optics")
    NA, h, N, dx = 0.8, 150.0, 512, 20.0
    fx, fy = fo.frequencies((N, N), dx)
    kap = 2 * np.pi * np.sqrt(fx**2 + fy**2)
    kzm = K2 * np.sqrt(1 - (NA / N2) ** 2)
    kz = np.sqrt(K2**2 - kap**2 + 0j)
    spec = np.where(kz.real > kzm, 1j / (2 * np.pi) / np.where(kz == 0, 1, kz), 0) * (2 * np.pi / (N * dx)) ** 2
    focus = np.fft.ifft2(spec) * N**2
    E = fo.propagate(focus, dx, LAM_M, h, bandlimit=False)
    _, t, tp = fresnel(N1, N2)
    E = E * t * tp * np.exp(1j * K2 * h)
    rho = np.arange(40) * dx
    ref = scalar_psf(rho, h, 1.0, LAM0, N2, NA, N1)
    got = E[:40, 0]
    assert np.max(np.abs(got - ref)) / np.abs(ref[0]) < 1e-2


@pytest.mark.parametrize("NA", [0.4, 1.0, 1.3])
def test_vectorial_center_closed_form_without_interface(NA):
    """n1 = n2, h = zf = 0."""
    n = 1.33
    k = 2 * np.pi * n / LAM0
    c = np.sqrt(1 - (NA / n) ** 2)
    ex, ey = vectorial_psf(0.0, 0.0, 0.0, 1.0, LAM0, n, n, NA)
    pred = 1j * k**3 / (8 * np.pi) * (2 / 3 * (1 - c**1.5) + 2 / 5 * (1 - c**2.5))
    assert ex == pytest.approx(pred, rel=1e-10)
    assert ey == 0


def test_vectorial_reduces_to_scalar_at_low_na():
    """n1 = n2."""
    n = 1.33
    k = 2 * np.pi * n / LAM0
    dev = []
    for NA in (0.2, 0.1):
        ex, _ = vectorial_psf(0.0, 0.0, 40.0, 1.0, LAM0, n, n, NA)
        sc = scalar_psf(0.0, 40.0, dipole_amplitude(1.0, k), LAM0, n, NA, n)
        dev.append(abs(ex / sc - 1))
    assert dev[0] / dev[1] == pytest.approx(4, rel=0.02)


def test_vectorial_symmetry():
    xs = np.array([150.0, -150.0, 150.0, -150.0, 200.0, 0.0])
    ys = np.array([90.0, 90.0, -90.0, -90.0, 0.0, 200.0])
    ex, ey = vectorial_psf(xs, ys, 30.0, ALPHA_PS, LAM0, N1, N2, 1.4)
    assert np.allclose(ex[:4], ex[0], rtol=1e-12)
    assert np.allclose(ey[:4] * [1, -1, -1, 1], ey[0], rtol=1e-12)
    assert np.allclose(ey[4:], 0, atol=1e-16)


def test_vectorial_psf_elongated_along_polarisation():
    """The x-dipole image is wider along x than along y at NA 1.4."""
    r = np.linspace(0, 800, 1601)
    ex_x, _ = vectorial_psf(r, 0 * r, 20.0, 1.0, LAM0, N1, N2, 1.4)
    ex_y, _ = vectorial_psf(0 * r, r, 20.0, 1.0, LAM0, N1, N2, 1.4)
    wx, wy = fwhm(r, abs(ex_x) ** 2), fwhm(r, abs(ex_y) ** 2)
    assert wx / wy > 1.2
    r = np.linspace(0, 6000, 3001)
    ex_x, _ = vectorial_psf(r, 0 * r, 20.0, 1.0, LAM0, N1, N2, 0.2)
    ex_y, _ = vectorial_psf(0 * r, r, 20.0, 1.0, LAM0, N1, N2, 0.2)
    assert fwhm(r, abs(ex_x) ** 2) / fwhm(r, abs(ex_y) ** 2) == pytest.approx(1, abs=0.01)


def test_quadrature_converged():
    ex1, ey1 = vectorial_psf(120.0, 80.0, 25.0, 1.0, LAM0, N1, N2, 1.45, order=200)
    ex2, ey2 = vectorial_psf(120.0, 80.0, 25.0, 1.0, LAM0, N1, N2, 1.45, order=400)
    assert ex1 == pytest.approx(ex2, rel=1e-10)
    assert ey1 == pytest.approx(ey2, rel=1e-10)


NA_HI = 1.4


def test_wrap_maps_to_half_open_interval():
    np.testing.assert_allclose(wrap([3 * np.pi / 2, -3 * np.pi / 2, np.pi, -np.pi, 7.0]),
                               [-np.pi / 2, np.pi / 2, np.pi, np.pi, 7.0 - 2 * np.pi], atol=1e-15)


def test_pupil_terms_are_fresnel_coefficients_over_kz():
    """The kz2 = 0-safe pupil terms equal t_s/kz2 and t_p/kz2 from fresnel_angular."""
    k0 = 2 * np.pi / LAM0
    k1, k2 = N1 * k0, N2 * k0
    kap, kz1, kz2, _ = _pupil_nodes(k1, k2, NA_HI * k0, 40)
    a, b = _pupil_ab(kap, kz1, kz2, 0.0, 0.0, k1, k2, N1, N2)
    _, _, t_s, t_p = fresnel_angular(kz2, kz1, N2, N1)
    g = np.sqrt(kz1.real / k1)
    np.testing.assert_allclose(a, g * (t_s + kz2 / k2 * t_p) / kz2, rtol=1e-12)
    np.testing.assert_allclose(b, g * (t_s - kz2 / k2 * t_p) / kz2, rtol=1e-12, atol=1e-12 * np.abs(a).max())


def test_tabulated_imager_converges_second_order():
    """Linear interpolation of the Bessel tables."""
    x, y = np.meshgrid(np.linspace(-800, 800, 25), np.linspace(-790, 810, 25))
    ex = vectorial_psf(x, y, 30.0, 1.0, LAM0, N1, N2, NA_HI)[0]
    err = [np.abs(DipoleImager(LAM0, N1, N2, NA_HI, rho_max=1200.0, d_rho=d).field(x, y, 30.0, 1.0)[0] - ex).max()
           for d in (2.0, 1.0, 0.5)]
    order = np.polyfit(np.log([2.0, 1.0, 0.5]), np.log(err), 1)[0]
    assert order == pytest.approx(2.0, abs=0.1)


def test_invalid_inputs_raise():
    """Undefined cases raise instead of returning a misleading number."""
    with pytest.raises(ValueError):
        vectorial_psf(0.0, 0.0, 10.0, 1.0, LAM0, N1, N2, 1.52)
    with pytest.raises(ValueError):
        DipoleImager(LAM0, N1, N2, 1.6)
    with pytest.raises(ValueError):
        DipoleImager(LAM0, N1, N2, NA_HI, rho_max=500.0).field(np.array([600.0]), np.array([0.0]), 10.0, 1.0)
    with pytest.raises(ValueError):
        mie_efficiencies(1.2, 0.0)
    with pytest.raises(ValueError):
        detection_limit(lambda a: 1e-3 * a**3, 1e9, a_lo=1.0, a_hi=10.0)
