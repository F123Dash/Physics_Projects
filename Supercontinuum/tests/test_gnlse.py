"""Pass-the-bar tests. The benchmark, photon-number and convergence tests take a
few minutes; deselect them with  pytest -m "not slow"."""

import numpy as np
import pytest

from supercontinuum import (Grid, Fibre, GNLSE, sech_pulse, raman_time_analytic,
                            raman_response, dudley_pcf, soliton_order,
                            dispersion_length, fission_length)
from supercontinuum import checks as ck, analysis as an
from supercontinuum.fibre import DUDLEY_P0, DUDLEY_T0, DUDLEY_WAVELENGTH


# ------------------------------------------------------------- unit checks
def test_fourier_convention_positive_omega_is_blue():
    g = Grid(1024, 10.0, 800.0)
    w = 5.0
    Aw = g.to_freq(np.exp(-1j * w * g.t))
    assert abs(g.omega[np.argmax(np.abs(Aw))] - w) < g.omega[1]
    assert g.wavelength[np.argmax(np.abs(Aw))] < 800.0


def test_raman_time():
    t = np.linspace(0, 2.0, 400001)
    h = raman_response(t)
    assert np.trapezoid(h, t) == pytest.approx(1.0, rel=1e-6)
    TR = 0.18 * np.trapezoid(t * h, t)
    assert TR == pytest.approx(raman_time_analytic(), rel=1e-6)
    assert TR * 1e3 == pytest.approx(1.462, abs=1e-3)   # fs


def test_benchmark_scales_and_zero_dispersion():
    fib = dudley_pcf()
    N = soliton_order(fib.gamma, DUDLEY_P0, DUDLEY_T0, fib.beta2)
    assert dispersion_length(DUDLEY_T0, fib.beta2) == pytest.approx(0.0682, abs=2e-4)
    assert N == pytest.approx(8.66, abs=0.01)
    assert fission_length(DUDLEY_T0, fib.beta2, N) == pytest.approx(0.00787, abs=1e-4)
    g = Grid(2**13, 12.5, DUDLEY_WAVELENGTH)
    Om = np.linspace(0, 400, 400001)
    i = np.nonzero(np.diff(np.sign(fib.gvd(Om))))[0][0]
    zdw = an.omega_to_wavelength(g, Om[i])
    assert zdw == pytest.approx(780, abs=5)


def test_methods_agree():
    g = Grid(2**12, 6.0, DUDLEY_WAVELENGTH)
    fib = dudley_pcf(length=0.003)
    A0 = sech_pulse(g.t, DUDLEY_P0, DUDLEY_T0)
    a = GNLSE(g, fib).propagate(A0, method="rk4ip", nsteps=600).Aw[-1]
    b = GNLSE(g, fib).propagate(A0, method="ssfm", nsteps=600).Aw[-1]
    c = GNLSE(g, fib).propagate(A0, method="rk4ip", tol=1e-8).Aw[-1]
    assert np.linalg.norm(a - b) / np.linalg.norm(a) < 1e-3
    assert np.linalg.norm(a - c) / np.linalg.norm(a) < 1e-5


def test_batch_matches_serial_solver():
    """The batched (GPU when available) integrator reproduces the serial one: mixed
    fibres, lengths, tolerances and integrators in one call."""
    from supercontinuum.batch import Run, run_batch
    g = Grid(2**11, 6.0, DUDLEY_WAVELENGTH)
    A0 = sech_pulse(g.t, DUDLEY_P0, DUDLEY_T0)
    specs = [dict(fibre=dudley_pcf(length=0.002), tol=1e-8),
             dict(fibre=dudley_pcf(raman=False, length=0.001), tol=1e-7),
             dict(fibre=dudley_pcf(length=0.002), nsteps=200),
             dict(fibre=dudley_pcf(length=0.002), nsteps=200, method="ssfm"),
             dict(fibre=dudley_pcf(raman=False, self_steepening=False, length=0.002),
                  nsteps=200, method="ssfm")]
    out = run_batch([Run(g, A0=A0, nsave=4, **s) for s in specs], verbose=False)
    for s, r in zip(specs, out):
        kw = {k: v for k, v in s.items() if k != "fibre"}
        c = GNLSE(g, s["fibre"]).propagate(A0, nsave=4, **kw)
        assert r.steps == c.steps
        assert np.linalg.norm(r.Aw - c.Aw) / np.linalg.norm(c.Aw) < 1e-11


# ------------------------------------------------------------ pass the bar
@pytest.mark.parametrize("fn", [ck.check_linear, ck.check_spm, ck.check_fundamental_soliton,
                                ck.check_conservation, ck.check_raman_shift,
                                ck.check_dispersive_wave],
                         ids=lambda f: f.__name__)
def test_bar(fn):
    r = fn()
    assert r["passed"], (r["name"], r["value"])


@pytest.mark.parametrize("N", [2, 3])
def test_higher_order_soliton(N):
    r = ck.check_higher_soliton(N)
    assert r["passed"], r["value"]


@pytest.fixture(scope="module")
def bench():
    return ck.run_benchmark(nsave=101)


@pytest.mark.slow
def test_benchmark(bench):
    r = ck.check_benchmark(bench)
    assert r["passed"], r["value"]


@pytest.mark.slow
def test_photon_number(bench):
    r = ck.check_photon_number(bench)
    assert r["passed"], r["value"]


@pytest.mark.slow
def test_convergence_order():
    r = ck.check_convergence(ck.convergence_study(steps=(1024, 2048, 4096), ref_steps=32768))
    assert r["passed"], r["value"]
