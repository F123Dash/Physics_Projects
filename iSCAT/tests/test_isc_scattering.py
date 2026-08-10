import numpy as np
import pytest
from scipy.optimize import brentq

from iscat import (
    backscatter_x2_coefficient,
    clausius_mossotti,
    dipole_amplitude,
    mie_amplitudes,
    mie_backscatter_amplitude,
    mie_coefficients,
    mie_dipole_polarizability,
    mie_efficiencies,
    polarizability,
    polarizability_radiative,
    protein_polarizability,
    rayleigh_cross_sections,
    rayleigh_sigma_sca,
    rayleigh_x2_coefficient,
)

LAM0, N_W = 532.0, 1.33
K = 2 * np.pi * N_W / LAM0
M_PS = 1.59 / N_W
M_AU = (0.5439 + 2.2309j) / N_W


def test_bohren_huffman_example():
    """BHMIE listing (Bohren & Huffman appendix A)."""
    x = 2 * np.pi * 0.525 / 0.6328
    q_ext, q_sca, q_back = mie_efficiencies(1.55, x)
    assert q_sca == pytest.approx(3.10543, abs=5e-5)
    assert q_ext == pytest.approx(3.10543, abs=5e-5)
    assert q_back == pytest.approx(2.92534, abs=5e-5)


@pytest.mark.parametrize("m", [M_PS, M_AU, 1.5 + 0.01j, 2.0 + 1.0j])
@pytest.mark.parametrize("x", [0.05, 0.6, 3.0, 25.0])
def test_mie_matches_miepython(m, x):
    """Agreement to 1e-6."""
    miepython = pytest.importorskip("miepython")
    q_ext, q_sca, q_back = mie_efficiencies(m, x)
    ref = miepython.efficiencies_mx(np.conj(m), x)
    assert q_ext == pytest.approx(ref[0], rel=1e-6)
    assert q_sca == pytest.approx(ref[1], rel=1e-6)
    assert q_back == pytest.approx(ref[2], rel=1e-6)


@pytest.mark.parametrize("x", [0.1, 1.0, 10.0, 80.0])
def test_lossless_sphere_ext_equals_sca(x):
    q_ext, q_sca, _ = mie_efficiencies(1.4, x)
    assert q_ext == pytest.approx(q_sca, rel=1e-12)


@pytest.mark.parametrize("m", [M_PS, M_AU])
def test_optical_theorem(m):
    x = 1.7
    s1, s2 = mie_amplitudes(m, x, [1.0])
    assert s1[0] == pytest.approx(s2[0], rel=1e-12)
    assert mie_efficiencies(m, x)[0] == pytest.approx(4 * s1[0].real / x**2, rel=1e-12)


def test_backscatter_amplitudes_are_opposite():
    s1, s2 = mie_amplitudes(M_AU, 2.3, [-1.0])
    assert s1[0] == pytest.approx(-s2[0], rel=1e-12)


def test_extinction_paradox():
    """Q_ext -> 2 for x >> 1."""
    dev = [mie_efficiencies(1.5, x)[0] - 2 for x in (200.0, 2000.0)]
    assert abs(dev[1]) < 0.015 and abs(dev[1]) < abs(dev[0])


@pytest.mark.parametrize("m", [1.5 + 0.01j, M_AU])
def test_mie_truncation_converged(m):
    """Our default order agrees with 60 extra terms to 1e-10 for a weakly absorbing x = 50."""
    x = 50.0
    a, b = mie_coefficients(m, x)
    a2, b2 = mie_coefficients(m, x, nmax=len(a) + 60)
    n, n2 = np.arange(1, len(a) + 1), np.arange(1, len(a2) + 1)
    q = 2 / x**2 * np.sum((2 * n + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2))
    q2 = 2 / x**2 * np.sum((2 * n2 + 1) * (np.abs(a2) ** 2 + np.abs(b2) ** 2))
    assert q == pytest.approx(q2, rel=1e-10)


def test_mie_converged_in_nmax():
    a, b = mie_coefficients(M_AU, 8.0)
    a2, b2 = mie_coefficients(M_AU, 8.0, nmax=len(a) + 10)
    q = lambda a, b: np.sum((2 * np.arange(1, len(a) + 1) + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2))
    assert q(a, b) == pytest.approx(q(a2, b2), rel=1e-12)


def test_rayleigh_sigma_formula_equals_dipole():
    a = 12.0
    s_dip, _ = rayleigh_cross_sections(polarizability(a, M_PS), K)
    assert s_dip == pytest.approx(rayleigh_sigma_sca(a, M_PS, K), rel=1e-14)


def test_radiative_correction_restores_optical_theorem():
    """sigma_ext = sigma_sca for a lossless dipole only with radiation reaction."""
    a = 30.0
    alpha = polarizability_radiative(a, M_PS, K)
    s_sca, s_ext = rayleigh_cross_sections(alpha, K)
    assert s_ext == pytest.approx(s_sca, rel=1e-12)
    assert rayleigh_cross_sections(polarizability(a, M_PS), K)[1] == 0


@pytest.mark.parametrize("m", [M_PS, M_AU])
def test_mie_dipole_polarizability_limit(m):
    """Mie dipole polarizability and backscatter amplitude follow the x^2 law."""
    a = 0.5
    x = K * a
    m2 = m**2
    ratio = mie_dipole_polarizability(a, m, K) / polarizability(a, m)
    assert abs(ratio - (1 + 0.6 * x**2 * (m2 - 2) / (m2 + 2))) < 10 * x**3
    f = mie_backscatter_amplitude(a, m, K)
    C = backscatter_x2_coefficient(m)
    assert abs(f / dipole_amplitude(polarizability(a, m), K) - (1 + C * x**2)) < 10 * x**3


def test_rayleigh_x2_coefficient_lossless():
    """sigma_Mie/sigma_Rayleigh - 1 = B x^2 + O(x^4) for real m."""
    B = rayleigh_x2_coefficient(M_PS)
    for x in (0.02, 0.04):
        a = x / K
        ratio = mie_efficiencies(M_PS, x)[1] * np.pi * a**2 / rayleigh_sigma_sca(a, M_PS, K)
        assert abs((ratio - 1) / x**2 - B) < 0.5 * x**2


def test_backscatter_x2_coefficient():
    """f_back/f_dipole - 1 = C x^2 with C from a_1, b_1 and a_2."""
    C = backscatter_x2_coefficient(M_PS)
    x = 0.02
    a = x / K
    ratio = mie_backscatter_amplitude(a, M_PS, K) / dipole_amplitude(polarizability(a, M_PS), K)
    assert abs((ratio.real - 1) / x**2 - C) < x
    assert backscatter_x2_coefficient(1 + 1e-9) == pytest.approx(-0.4, abs=1e-8)


def test_brief_a6_law_within_2pct_needs_medium_wavelength():
    """The a^6 law holds to 2% up to a = lambda / (20 n_water); at lambda / 20 it is off by > 3.5%."""
    def dev(a):
        return mie_efficiencies(M_PS, K * a)[1] * np.pi * a**2 / rayleigh_sigma_sca(a, M_PS, K) - 1

    a2 = brentq(lambda a: abs(dev(a)) - 0.02, 1, 60)
    assert a2 == pytest.approx(LAM0 / N_W / 20, rel=1e-3)
    assert abs(dev(LAM0 / 20)) > 0.035


def test_divergence_radius_ps_predicted_by_x2_law_less_so_gold():
    def dev(a, m):
        return mie_efficiencies(m, K * a)[1] * np.pi * a**2 / rayleigh_sigma_sca(a, m, K) - 1

    a10_ps = brentq(lambda a: abs(dev(a, M_PS)) - 0.1, 1, 80)
    pred_ps = np.sqrt(0.1 / abs(rayleigh_x2_coefficient(M_PS))) / K
    assert a10_ps == pytest.approx(pred_ps, rel=0.05)
    a10_au = brentq(lambda a: abs(dev(a, M_AU)) - 0.1, 1, 20)
    assert a10_au < 0.5 * a10_ps
    pred_au = np.sqrt(0.1 / abs(rayleigh_x2_coefficient(M_AU))) / K
    assert abs(a10_au / pred_au - 1) > 0.1


def test_protein_polarizability_matches_sphere_to_first_order():
    """dn/dc and equivalent-sphere polarizabilities agree to 1 - delta/6 - 2 delta^2/9."""
    rho = 1.35
    for dndc in (0.002, 0.01):
        mass = 100.0
        volume = mass * 1e3 / 6.02214076e23 / rho * 1e21
        delta = rho * dndc / N_W
        ratio = 3 * volume * clausius_mossotti(1 + delta) / protein_polarizability(mass, dndc, N_W)
        assert ratio == pytest.approx(1 - delta / 6 - 2 * delta**2 / 9, abs=delta**3)


def loglog_slope(a, y):
    return np.polyfit(np.log(a), np.log(np.abs(y)), 1)[0]


def test_brief_contrast_slopes_10_to_40_nm():
    """Interference term ~ |f_back|, scattering term ~ |f_back|^2."""
    a = np.geomspace(10, 40, 25)
    f_ray = dipole_amplitude(polarizability(a, M_PS), K)
    assert loglog_slope(a, f_ray) == pytest.approx(3, abs=1e-12)
    assert loglog_slope(a, f_ray**2) == pytest.approx(6, abs=1e-12)
    f_mie = mie_backscatter_amplitude(a, M_PS, K)
    s3, s6 = loglog_slope(a, f_mie), loglog_slope(a, np.abs(f_mie) ** 2)
    assert s3 == pytest.approx(3, abs=0.1)
    assert abs(s6 - 6) > 0.1
    series = np.abs(1 + backscatter_x2_coefficient(M_PS) * (K * a) ** 2) * a**3
    assert s3 == pytest.approx(loglog_slope(a, series), abs=0.02)
