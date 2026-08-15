import numpy as np
import pytest
from scipy.integrate import quad

from iscat import (
    absorption_cross_section,
    contrast,
    lens_form_factor,
    lens_polarizability,
    lockin_demodulate,
    momentum_transfer_sq,
    mie_dipole_polarizability,
    photothermal_contrast,
    polarizability_radiative,
    rayleigh_cross_sections,
    temperature,
    thermal_gamma,
    vectorial_psf,
    vectorial_reference,
)

LAM0, N1, N2, NA = 532.0, 1.52, 1.33, 1.4
K = 2 * np.pi * N2 / LAM0
M_AU = (0.5439 + 2.2309j) / N2


def test_absorption_cross_section_dipole_limit():
    """Mie sigma_abs -> sigma_ext - sigma_sca of the radiative dipole for a = 2 nm (x = 0.03)."""
    a = 2.0
    alpha = polarizability_radiative(a, M_AU, K)
    sca, _ = rayleigh_cross_sections(alpha, K)
    assert absorption_cross_section(a, M_AU, K) == pytest.approx(K * alpha.imag - sca, rel=1e-2)


@pytest.mark.parametrize("omega", [0.0, 2 * np.pi * 1e6])
def test_temperature_solves_heat_equation(omega):
    """T solves the oscillating heat equation and carries the absorbed power across r = a."""
    a, P = 10.0, 1e-6
    g = thermal_gamma(omega)
    assert g**2 == pytest.approx(-1j * omega / 1.43e-7 / 1e18)
    r, d = np.array([15.0, 60.0, 300.0]), 0.5
    T = lambda r: temperature(r, P, a, g)
    lap = ((r + d) * T(r + d) - 2 * r * T(r) + (r - d) * T(r - d)) / d**2 / r
    bound = 1.5 * (abs(g) * d) ** 2 / 12 * abs(g) ** 2 * np.abs(T(r)) + 1e-9 * np.abs(T(r)) / r**2
    assert np.all(np.abs(lap - g**2 * T(r)) < bound)
    d = 1e-3
    dTdr = (-3 * T(a) + 4 * T(a + d) - T(a + 2 * d)) / (2 * d)
    assert -0.6 * dTdr * 1e9 * 4 * np.pi * (a * 1e-9) ** 2 == pytest.approx(P, rel=1e-5)


@pytest.mark.parametrize("q", [0.0, K, 2 * K])
@pytest.mark.parametrize("r_th", [150.0, 600.0])
def test_lens_form_factor_closed_form(q, r_th):
    """Closed form vs direct radial quadrature of 4 pi r e^{-gamma(r-a)} sin(qr)/q."""
    a, g = 10.0, (1 - 1j) / r_th
    f = lambda r: 4 * np.pi * r * np.exp(-g * (r - a)) * (np.sin(q * r) / (q * r) if q else 1.0)
    upper = a + 60 * r_th
    re = quad(lambda r: f(r).real, a, upper, limit=800, epsabs=0, epsrel=1e-11)[0]
    im = quad(lambda r: f(r).imag, a, upper, limit=800, epsabs=0, epsrel=1e-11)[0]
    assert lens_form_factor(q**2, a, g) == pytest.approx(re + 1j * im, rel=1e-8)


def test_steady_lens_is_finite_in_reflection_and_diverges_forward():
    """At q = 2k the lens converges as Omega -> 0 to 4 pi cos(2ka)/(2k)^2."""
    a = 10.0
    lo = [lens_form_factor(4 * K**2, a, thermal_gamma(2 * np.pi * f)) for f in (1.0, 10.0)]
    assert lo[0] == pytest.approx(4 * np.pi * np.cos(2 * K * a) / (4 * K**2), rel=1e-3)
    assert abs(lo[1] / lo[0] - 1) < 1e-3
    fw = [abs(lens_form_factor(0.0, a, thermal_gamma(2 * np.pi * f))) for f in (1.0, 10.0)]
    assert np.log10(fw[1] / fw[0]) == pytest.approx(-1, abs=1e-3)


def test_constant_form_factor_is_a_point_dipole():
    """form_factor = const c reproduces a dipole of polarizability c (pupil plumbing)."""
    x = np.array([0.0, 120.0, 300.0])
    a = vectorial_psf(x, 0.5 * x, 30.0, 1.0, LAM0, N1, N2, NA, 80.0, form_factor=lambda q2: 7.0 - 2j + 0 * q2)
    b = vectorial_psf(x, 0.5 * x, 30.0, 7.0 - 2j, LAM0, N1, N2, NA, 80.0)
    np.testing.assert_allclose(a, b, rtol=1e-13)


def test_sideband_formula_matches_time_domain_demodulation():
    """Point lens with a lagging delta alpha_+ (complex)."""
    zf, h = 100.0, 10.0
    alpha = mie_dipole_polarizability(10.0, M_AU, K)
    er = vectorial_reference(LAM0, N1, N2, zf)
    es, ey = vectorial_psf(0.0, 50.0, h, alpha, LAM0, N1, N2, NA, zf)
    gx, gy = vectorial_psf(0.0, 50.0, h, 1.0, LAM0, N1, N2, NA, zf)
    da = 1e-3 * abs(alpha) * np.exp(-0.7j)
    wt = 2 * np.pi * np.arange(256) / 256
    dat = (da * np.exp(-1j * wt)).real
    I = np.abs(er + es + gx * dat) ** 2 + np.abs(ey + gy * dat) ** 2
    num = 2 * np.mean(I * np.exp(1j * wt)) / abs(er) ** 2
    ana = photothermal_contrast(es, ey, (gx * da, gy * da), (gx * np.conj(da), gy * np.conj(da)), er)
    assert abs(num - ana) < 1e-6 * abs(ana)
    eps = 1e-3
    fd = (contrast(es + eps * gx, er, ey + eps * gy) - contrast(es - eps * gx, er, ey - eps * gy)) / (2 * eps)
    assert photothermal_contrast(es, ey, (gx, gy), (gx, gy), er) == pytest.approx(fd, rel=1e-9)


def test_lockin_shot_noise_is_sqrt_two_over_n():
    """Poisson counts in 64 phase bins with rate ∝ 1 + Re(dC e^{-i W t})."""
    rng = np.random.default_rng(9)
    N, dC, M = 1e6, 3e-3 * np.exp(0.4j), 2000
    wt = 2 * np.pi * (np.arange(64) + 0.5) / 64
    rate = N / 64 * (1 + (dC * np.exp(-1j * wt)).real)
    est = lockin_demodulate(rng.poisson(rate, size=(M, 64)), wt)
    s = np.sqrt(2 / N)
    assert abs(est.mean().real - dC.real) < 3.5 * s / np.sqrt(M)
    assert abs(est.mean().imag - dC.imag) < 3.5 * s / np.sqrt(M)
    for part in (est.real, est.imag):
        assert abs(part.std(ddof=1) - s) < 3.5 * s / np.sqrt(2 * (M - 1))


def test_photothermal_signal_scales_as_a_cubed():
    """sigma_abs ∝ a^3 and Delta T ∝ sigma_abs / a ∝ a^2."""
    zf, I_pump, g = 100.0, 1e-9, thermal_gamma(2 * np.pi * 1e5)
    er = vectorial_reference(LAM0, N1, N2, zf)
    out = []
    for a in (2.0, 5.0):
        P = absorption_cross_section(a, M_AU, K) * I_pump
        es, ey = vectorial_psf(0.0, 0.0, 6.0, mie_dipole_polarizability(a, M_AU, K), LAM0, N1, N2, NA, zf)
        plus = vectorial_psf(0.0, 0.0, 6.0, 1.0, LAM0, N1, N2, NA, zf, form_factor=lambda q2: lens_polarizability(q2, P, a, g))
        minus = vectorial_psf(0.0, 0.0, 6.0, 1.0, LAM0, N1, N2, NA, zf,
                              form_factor=lambda q2: lens_polarizability(q2, np.conj(P), a, np.conj(g)))
        out.append(abs(photothermal_contrast(es, ey, plus, minus, er)))
    assert np.log(out[1] / out[0]) / np.log(2.5) == pytest.approx(3, abs=0.02)


def test_lossless_sphere_absorbs_nothing():
    """sigma_abs = sigma_ext - sigma_sca vanishes for real m at any size (energy balance)."""
    for a in (5.0, 50.0, 150.0):
        assert abs(absorption_cross_section(a, 1.2, K)) < 1e-12 * np.pi * a**2


def test_momentum_transfer_geometry():
    """q^2 = |k_in - k_out|^2 for k_in = k2 z_hat and k_out = (kappa, 0, -kz2)."""
    k2 = K
    for kap in (0.0, 0.3 * k2, 0.9 * k2, k2):
        kz2 = np.sqrt(k2**2 - kap**2)
        q2 = kap**2 + (k2 + kz2) ** 2
        assert momentum_transfer_sq(kz2, k2) == pytest.approx(q2, rel=1e-14)
    assert momentum_transfer_sq(k2, k2) == pytest.approx(4 * k2**2)
    assert momentum_transfer_sq(0.0, k2) == pytest.approx(2 * k2**2)


@pytest.mark.parametrize("q", [0.5 * K, 2 * K])
def test_lens_polarizability_is_born_integral_of_index_profile(q):
    """Lens polarizability equals the Born integral of the explicit index profile."""
    a, P, n_m, dndT = 10.0, 2e-7, 1.33, -1e-4
    T = lambda r: temperature(r, P, a, 1 / 400.0)
    g = lambda r: ((n_m + dndT * T(r)) ** 2 - n_m**2) / n_m**2 * 4 * np.pi * r / q
    born = quad(g, a, a + 400.0 * 60, weight="sin", wvar=q, limit=2000)[0]
    tol = abs(dndT * T(a)) / (2 * n_m)
    assert tol < 1.1e-4
    assert lens_polarizability(q**2, P, a, 1 / 400.0, dndT, n_m) == pytest.approx(born, rel=tol)


def test_no_absorption_or_no_dn_dT_gives_no_photothermal_signal():
    """Negative controls."""
    zf, a = 200.0, 10.0
    er = vectorial_reference(LAM0, N1, N2, zf)
    es, ey = vectorial_psf(0.0, 0.0, a, 1e4, LAM0, N1, N2, NA, zf)
    g = thermal_gamma(2 * np.pi * 1e5)
    for P, dndT in ((absorption_cross_section(a, 1.59 / N2, K) * 1e-9, -1e-4), (1e-7, 0.0)):
        lens = lambda P_, g_: vectorial_psf(0.0, 0.0, a, 1.0, LAM0, N1, N2, NA, zf,
                                            form_factor=lambda q2: lens_polarizability(q2, P_, a, g_, dndT))
        dC = photothermal_contrast(es, ey, lens(P, g), lens(np.conj(P), np.conj(g)), er)
        assert abs(dC) < 1e-12 * abs(contrast(es, er, ey))


def test_form_factor_small_q_branch_is_continuous():
    """The q -> 0 branch (|qa| < 1e-8) joins the general formula continuously."""
    a, g = 10.0, (1 - 1j) / 300.0
    assert lens_form_factor(1e-22, a, g) == pytest.approx(lens_form_factor(1e-17, a, g), rel=1e-10)
    assert lens_form_factor(0.0, a, g) == pytest.approx(4 * np.pi * (1 + g * a) / g**2, rel=1e-12)


def test_negative_modulation_frequency_raises():
    with pytest.raises(ValueError):
        thermal_gamma(-1.0)
