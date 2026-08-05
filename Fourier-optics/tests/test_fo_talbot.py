import numpy as np
import pytest

from fourier_optics import (
    binary_grating,
    binary_grating_series,
    cosine_grating,
    fidelity,
    harmonic_shift,
    parabolic_peak,
    propagate,
    talbot_defect_asymptote,
    talbot_exact,
    talbot_fidelity_series,
    talbot_paraxial,
    wrap,
)

LAM = 0.5
M = 512  # samples per period; the window is one period, so the grating is exactly infinite


def period_grid(d, M=M):
    return np.arange(M) * d / M


def prop(t, d, z, **kw):
    # A periodic field has a discrete spectrum sampled exactly by the DFT: no band limit applies.
    return propagate(t, d / t.size, LAM, z, bandlimit=False, **kw)


def test_talbot_distances_quoted_in_brief():
    assert talbot_exact(10.0, LAM) == pytest.approx(399.7498, abs=1e-4)
    assert talbot_paraxial(10.0, LAM) == 400.0


@pytest.mark.parametrize("d", [10.0, 2.0])
def test_cosine_grating_self_image_at_exact_talbot_distance(d):
    """The revival of a grating with harmonics 0, +-1 peaks at the exact z_T.
    Resolution: 11 samples spaced 0.002 z_T around the paraxial guess, parabolic vertex; the peak is within
    1% of the paraxial miss (lambda/2 = 0.25 um) of z_T, i.e. 2.5 nm."""
    t = cosine_grating(period_grid(d), d)
    zT, zp = talbot_exact(d, LAM), talbot_paraxial(d, LAM)
    zs = zp * np.linspace(0.99, 1.01, 11)
    F = np.array([fidelity(t, prop(t, d, z)) for z in zs])
    z_peak, F_peak = parabolic_peak(zs, F)
    assert abs(z_peak - zT) < 0.01 * abs(zp - zT)
    assert fidelity(t, prop(t, d, zT)) == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("d", [10.0, 2.0])
@pytest.mark.parametrize("kind", ["cosine", "binary"])
def test_half_talbot_image_shifted_by_half_period(d, kind):
    """At z_T / 2 the first harmonic of the intensity is shifted by d/2 for both gratings."""
    x = period_grid(d)
    t = cosine_grating(x, d) if kind == "cosine" else binary_grating(x, d)
    I = np.abs(prop(t, d, talbot_exact(d, LAM) / 2)) ** 2
    shift = harmonic_shift(I, d, d / M) - harmonic_shift(np.abs(t) ** 2, d, d / M)
    assert abs(wrap(2 * np.pi * (shift - 0.5))) < 1e-10
    if kind == "cosine":
        assert fidelity(np.roll(t, M // 2), prop(t, d, talbot_exact(d, LAM) / 2)) == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("d", [10.0, 2.0])
def test_binary_grating_self_images_only_paraxially(d):
    """The paraxial TF rephases every order m at 2 d^2 / lambda (phase 2 pi m^2); the exact TF does not,
    so a binary grating never reproduces itself (brief's claim holds for its first harmonic only)."""
    t = binary_grating(period_grid(d), d)
    assert fidelity(t, prop(t, d, talbot_paraxial(d, LAM), paraxial=True)) == pytest.approx(1, abs=1e-12)
    assert fidelity(t, prop(t, d, talbot_exact(d, LAM))) < 0.9


@pytest.mark.parametrize("z_over_zT", [0.25, 0.5, 1.0, 1.37])
def test_fft_matches_fourier_series_of_binary_grating(z_over_zT):
    """Closed form of the discrete problem: the grid propagator equals the phase sum over orders m."""
    d = 10.0
    t = binary_grating_series(period_grid(d, 1024), d, LAM)
    z = z_over_zT * talbot_exact(d, LAM)
    assert fidelity(t, prop(t, d, z)) == pytest.approx(talbot_fidelity_series(d, LAM, z), abs=1e-12)


@pytest.mark.parametrize("d_over_lam", [40960, 163840, 655360])
def test_binary_talbot_defect_scales_as_sqrt_lambda_over_d(d_over_lam):
    """1 - F(z_T) -> (4/pi^2) Re J sqrt(lambda/d) = 0.514 sqrt(lambda/d), from the 1/m^2 orders whose
    phase error at z_T grows as (pi/2) m^4 (lambda/d)^2; derived in Fourier-optics.pdf."""
    d = d_over_lam * LAM
    defect = 1 - talbot_fidelity_series(d, LAM, talbot_exact(d, LAM))
    assert defect / talbot_defect_asymptote(d, LAM) == pytest.approx(1, abs=0.02)


def test_talbot_exact_is_stable_for_large_periods():
    """lambda (1 + sqrt(1-u)) / u has no cancellation: it matches 2 d^2/lambda - lambda/2 for d >> lambda."""
    d = 1e5 * LAM
    assert talbot_exact(d, LAM) == pytest.approx(talbot_paraxial(d, LAM) - 0.5 * LAM, rel=1e-14)
