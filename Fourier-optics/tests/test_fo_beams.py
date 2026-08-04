from functools import lru_cache

import numpy as np
import pytest

from fourier_optics import (
    azimuthal_angle,
    beam_width,
    curvature,
    fidelity,
    fit_curvature,
    gaussian,
    gaussian_on_axis_exact,
    gouy,
    gouy_on_axis_nonparaxial,
    grid,
    laguerre_moment,
    lg,
    lg_nonparaxial_infidelity,
    lg_nonparaxial_phase,
    oam_per_photon,
    overlap,
    power,
    propagate,
    rayleigh_range,
    second_moment_width,
    width_nonparaxial,
    winding_number,
    wrap,
)

LAM, W0 = 0.5, 10.0
ZR = rayleigh_range(W0, LAM)
K = 2 * np.pi / LAM
EPS = 1 / (K * W0) ** 2
ZETAS = [0.0, 0.5, 1.0, 2.0, 3.0]
MODES = [(0, 1), (0, 3), (1, 1), (2, -2), (1, -3)]


@lru_cache
def mesh(N, dx):
    x = grid(N, dx)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return x, X, Y


@lru_cache
def gaussian_at(zeta):
    x, X, Y = mesh(512, 0.5)
    return propagate(gaussian(X, Y, W0, LAM), 0.5, LAM, zeta * ZR)


@lru_cache
def lg_at(p, l, zeta):
    x, X, Y = mesh(1024, 0.5)
    return propagate(lg(X, Y, p, l, W0, LAM), 0.5, LAM, zeta * ZR)


# ---------- Gaussian ----------

@pytest.mark.parametrize("zeta", ZETAS)
def test_gaussian_width(zeta):
    """w(z) = w0 sqrt(1 + (z/zR)^2) to 0.5% for 0 <= z <= 3 zR."""
    x = mesh(512, 0.5)[0]
    w = second_moment_width(gaussian_at(zeta), x)
    assert abs(w / beam_width(zeta * ZR, W0, LAM) - 1) < 5e-3


@pytest.mark.parametrize("zeta", ZETAS)
def test_gaussian_width_nonparaxial_term(zeta):
    """The 1e-4 deviation from the paraxial width is the derived (4 eps + 24 eps^2) term, not noise."""
    x = mesh(512, 0.5)[0]
    w = second_moment_width(gaussian_at(zeta), x)
    assert abs(w / width_nonparaxial(zeta * ZR, W0, LAM) - 1) < 1e-10


@pytest.mark.parametrize("zeta", [0.5, 1.0, 2.0, 3.0])
def test_gaussian_curvature(zeta):
    """1/R from a phase fit over |x| < w matches z / (z^2 + zR^2) to 0.1%."""
    x = mesh(512, 0.5)[0]
    E = gaussian_at(zeta)
    z = zeta * ZR
    c = fit_curvature(E[:, 256], x, K, beam_width(z, W0, LAM))
    assert abs(c / curvature(z, W0, LAM) - 1) < 1e-3


@pytest.mark.parametrize("zeta", ZETAS)
def test_gaussian_gouy(zeta):
    """kz - arg E(0, z) = arctan(z / zR) to 0.01 rad (a lag with exp(i(kz - wt)))."""
    z = zeta * ZR
    g = -wrap(np.angle(gaussian_at(zeta)[256, 256]) - K * z)
    assert abs(g - gouy(z, W0, LAM)) < 0.01


@pytest.mark.parametrize("zeta", ZETAS)
def test_gaussian_gouy_nonparaxial_term(zeta):
    """Residual after the derived 2 eps zeta (1 - zeta^2)/(1 + zeta^2)^2 term is O(eps^2)."""
    z = zeta * ZR
    g = -wrap(np.angle(gaussian_at(zeta)[256, 256]) - K * z)
    assert abs(g - gouy_on_axis_nonparaxial(z, W0, LAM)) < 10 * EPS**2


@pytest.mark.parametrize("zeta", ZETAS)
def test_gaussian_matches_exact_helmholtz_on_axis(zeta):
    """The grid propagator equals the Hankel integral of the angular spectrum: no approximation."""
    E = gaussian_at(zeta)[256, 256] / gaussian(0.0, 0.0, W0, LAM)
    assert abs(E - gaussian_on_axis_exact(zeta * ZR, W0, LAM)) < 1e-8


# ---------- Laguerre-Gauss ----------

@pytest.mark.parametrize("p, l", MODES + [(0, 0), (3, 2)])
def test_lg_orthonormal(p, l):
    x, X, Y = mesh(1024, 0.5)
    E = lg(X, Y, p, l, W0, LAM)
    assert abs(power(E, 0.5) - 1) < 1e-12
    for q, m in MODES:
        if (q, m) != (p, l):
            assert abs(overlap(lg(X, Y, q, m, W0, LAM), E, 0.5)) < 1e-12


@pytest.mark.parametrize("p, l", MODES)
def test_lg_formula_solves_paraxial_equation(p, l):
    """The analytic LG_pl(z), Gouy factor and curvature included, is the paraxial-TF image of LG_pl(0)."""
    x, X, Y = mesh(1024, 0.5)
    z = 1.7 * ZR
    E = propagate(lg(X, Y, p, l, W0, LAM), 0.5, LAM, z, paraxial=True)
    assert np.sqrt(power(E - lg(X, Y, p, l, W0, LAM, z), 0.5)) < 1e-10


@pytest.mark.parametrize("p, l", MODES)
@pytest.mark.parametrize("zeta", [0.0, 1.0, 3.0])
def test_lg_phase_winding(p, l, zeta):
    """The summed phase increments around the core are 2 pi l, at the waist and after propagation.
    The loop (corner at 0.7 w) stays inside the first dark ring at r = w sqrt(s_1 / 2)."""
    w = beam_width(zeta * ZR, W0, LAM)
    h = int(0.5 * w / 0.5)
    assert winding_number(lg_at(p, l, zeta), 512, 512, h) == pytest.approx(l, abs=1e-9)


def test_winding_negative_controls():
    """A loop that misses the core gives 0, a Gaussian gives 0, and a loop wholly outside the dark ring
    of LG_1^1 (r = w, where L_1^1 changes sign) still gives l: the sign flip is the same all the way round."""
    E = lg_at(1, 1, 0.0)
    assert winding_number(E, 512 + 60, 512, 10) == pytest.approx(0, abs=1e-9)
    assert winding_number(lg_at(0, 0, 1.0), 512, 512, 20) == pytest.approx(0, abs=1e-9)
    assert winding_number(E, 512, 512, 30) == pytest.approx(1, abs=1e-9)


@pytest.mark.parametrize("p, l", MODES)
@pytest.mark.parametrize("zeta", [1.0, 3.0])
def test_lg_gouy(p, l, zeta):
    """arg <LG(z) without Gouy | E(z)> = -(2p + |l| + 1) arctan(z / zR) to 0.01 rad."""
    x, X, Y = mesh(1024, 0.5)
    z = zeta * ZR
    ref = lg(X, Y, p, l, W0, LAM, z, with_gouy=False)
    ph = np.angle(overlap(ref, lg_at(p, l, zeta), 0.5))
    assert abs(wrap(ph + (2 * p + abs(l) + 1) * gouy(z, W0, LAM))) < 0.01


@pytest.mark.parametrize("p, l", MODES)
@pytest.mark.parametrize("zeta", [1.0, 3.0])
def test_lg_nonparaxial_phase_and_shape(p, l, zeta):
    """The 1e-3 Gouy residual is -zeta eps <s^2>/4 (next term zeta eps^2 <s^3>/4), and the only change
    of shape is the predicted infidelity (zeta eps / 4)^2 Var(s^2)."""
    x, X, Y = mesh(1024, 0.5)
    z = zeta * ZR
    E = lg_at(p, l, zeta)
    ref = lg(X, Y, p, l, W0, LAM, z, with_gouy=False)
    ph = np.angle(overlap(ref, E, 0.5))
    pred = -(2 * p + abs(l) + 1) * gouy(z, W0, LAM) + lg_nonparaxial_phase(z, p, l, W0, LAM)
    assert abs(wrap(ph - pred)) < 0.5 * zeta * EPS**2 * laguerre_moment(p, l, 3)
    assert 1 - fidelity(ref, E) == pytest.approx(lg_nonparaxial_infidelity(z, p, l, W0, LAM), rel=0.01)


@pytest.mark.parametrize("p, l", [(0, 0), (0, 3), (2, -2)])
def test_laguerre_moments_closed_form(p, l):
    """<s> = 2p + |l| + 1 and <s^2> = 6p^2 + 6p|l| + l^2 + 6p + 3|l| + 2."""
    a = abs(l)
    assert laguerre_moment(p, l, 1) == pytest.approx(2 * p + a + 1, rel=1e-12)
    assert laguerre_moment(p, l, 2) == pytest.approx(6 * p * p + 6 * p * a + a * a + 6 * p + 3 * a + 2, rel=1e-12)


@pytest.mark.parametrize("p, l", MODES)
def test_oam_per_photon_is_l_and_conserved(p, l):
    """<L_z> = l hbar per photon at the waist and after 3 zR (L_z commutes with free propagation)."""
    x, X, Y = mesh(1024, 0.5)
    assert oam_per_photon(lg(X, Y, p, l, W0, LAM), 0.5, X, Y) == pytest.approx(l, abs=1e-10)
    assert oam_per_photon(lg_at(p, l, 3.0), 0.5, X, Y) == pytest.approx(l, abs=1e-10)


def test_oam_of_superposition_is_weighted_mean():
    """Negative control on winding: |a|^2 l1 + |b|^2 l2, not an integer, for LG_0^1 and LG_0^-2 mixed."""
    x, X, Y = mesh(1024, 0.5)
    E = np.sqrt(0.7) * lg(X, Y, 0, 1, W0, LAM) + np.sqrt(0.3) * lg(X, Y, 0, -2, W0, LAM)
    assert oam_per_photon(E, 0.5, X, Y) == pytest.approx(0.7 - 0.6, abs=1e-10)


@lru_cache
def rotating(zeta, paraxial):
    x, X, Y = mesh(1024, 0.5)
    E0 = (lg(X, Y, 0, 0, W0, LAM) + lg(X, Y, 0, 2, W0, LAM)) / np.sqrt(2)
    E = propagate(E0, 0.5, LAM, zeta * ZR, paraxial=paraxial)
    return azimuthal_angle(np.abs(E) ** 2, X, Y, 2)


@pytest.mark.parametrize("zeta", [0.5, 1.0, 3.0])
def test_gouy_phase_rotates_mode_superposition(zeta):
    """LG_0^0 + LG_0^2 dephase by 2 arctan(z/zR), so the two-lobed pattern turns by arctan(z/zR).
    Exact to round-off with the paraxial TF; O(eps) = 1e-4 rad with the exact one."""
    g = gouy(zeta * ZR, W0, LAM)
    assert abs(rotating(zeta, True) - g) < 1e-10
    assert abs(rotating(zeta, False) - g) < 3 * EPS
