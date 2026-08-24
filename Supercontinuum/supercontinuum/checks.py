"""The "pass the bar" checks. Each returns a dict with the measured value,
the target, a pass flag and the data needed for plotting. The pytest suite
asserts on ``passed``; ``scripts/run_checks.py`` turns them into the README table.

Every check is split into a plan, ``plan_x(...) -> (runs, finish)``, and
``check_x(...) = execute(plan_x(...))``. A plan only lists its propagations, so a
script can hand the plans of many checks to one ``execute`` call and all their runs
are integrated together in GPU batches (see ``batch.py``).

Plot data is stored compactly (float32 intensity maps, cropped) so cached results
stay small."""

import numpy as np

from . import analysis as an
from .fibre import Fibre, dudley_pcf, DUDLEY_P0, DUDLEY_T0, DUDLEY_WAVELENGTH
from .grid import Grid
from .pulses import sech_pulse, gaussian_pulse
from .scales import (dispersion_length, soliton_order, soliton_period,
                     soliton_power, fission_length)
from .batch import Run, execute


def _check(name, value, target, passed, **data):
    return dict(name=name, value=value, target=target, passed=bool(passed), data=data)


def _one(plan):
    return execute(plan, verbose=False)[0]


def _intensity_map(t, At, t_max=4.0, every=1):
    """|A|^2 on |t| < t_max as float32, every ``every``-th row (for plotting)."""
    m = np.abs(t) < t_max
    return t[m], (np.abs(At[::every, m]) ** 2).astype(np.float32)


# --------------------------------------------------------------------- linear
def plan_linear(method="rk4ip"):
    """Unchirped Gaussian, dispersion only: T(z) = T0 sqrt(1 + (z/L_D)^2)."""
    T0, beta2 = 1.0, -1.0
    g = Grid(2**12, 80.0)
    fib = Fibre(betas=[beta2], gamma=0.0, length=4.0)
    LD = dispersion_length(T0, beta2)
    run = Run(g, fib, gaussian_pulse(g.t, 1.0, T0), nsave=41, method=method, nsteps=40)
    return [run], lambda r: _finish_linear(g, T0, LD, r[0])


def check_linear(method="rk4ip"):
    return _one(plan_linear(method))


def _finish_linear(g, T0, LD, res):
    w0 = an.rms_width(g.t, res.At[0])
    widths = np.array([an.rms_width(g.t, A) for A in res.At]) / w0 * T0
    theory = T0 * np.sqrt(1 + (res.z / LD) ** 2)
    err = np.max(np.abs(widths / theory - 1))
    return _check("Linear Gaussian broadening", err, "< 1e-3 relative", err < 1e-3,
                  z=res.z / LD, width=widths, theory=theory)


# ------------------------------------------------------------------------ SPM
def plan_spm(phis=(1.5, 2.5, 3.5, 4.5)):
    """Kerr only, Gaussian: phi_max = gamma P0 L = (M - 1/2) pi gives M spectral peaks."""
    g = Grid(2**12, 40.0)
    A0 = gaussian_pulse(g.t, 1.0, 1.0)
    runs = [Run(g, Fibre(betas=[], gamma=1.0, length=phi * np.pi), A0, method="ssfm", nsteps=1)
            for phi in phis]
    return runs, lambda r: _finish_spm(g, phis, r)


def check_spm(phis=(1.5, 2.5, 3.5, 4.5)):
    return _one(plan_spm(phis))


def _finish_spm(g, phis, results):
    counts, spectra = [], []
    for res in results:
        S = np.abs(np.fft.fftshift(res.Aw[-1])) ** 2
        counts.append(an.count_peaks(S, 0.01))
        spectra.append(S / S.max())
    expected = [int(round(p + 0.5)) for p in phis]
    return _check("SPM spectral peaks", counts, expected, counts == expected,
                  phis=list(phis), omega=g.omega_shifted, spectra=spectra)


# -------------------------------------------------------------------- solitons
def _soliton_setup(N, n=2**11, window=60.0):
    g = Grid(n, window)
    fib = Fibre(betas=[-1.0], gamma=1.0)
    return g, fib, sech_pulse(g.t, N**2, 1.0)


def plan_fundamental_soliton(n_LD=20, nsteps=4000):
    g, fib, A0 = _soliton_setup(1)
    run = Run(g, fib, A0, length=n_LD * 1.0, nsave=81, nsteps=nsteps)
    return [run], lambda r: _finish_fundamental(A0, r[0])


def check_fundamental_soliton(n_LD=20, nsteps=4000):
    return _one(plan_fundamental_soliton(n_LD, nsteps))


def _finish_fundamental(A0, res):
    err_z = np.max(np.abs(np.abs(res.At) - np.abs(A0)), axis=-1) / np.max(np.abs(A0))
    err = err_z[-1]
    return _check("Fundamental soliton over 20 L_D", float(np.max(err_z)), "< 1e-4",
                  np.max(err_z) < 1e-4, z=res.z, err=err_z)


def plan_higher_soliton(N, nsteps=8000):
    """N = 2, 3: the intensity profile is periodic with period z0 = pi/2 L_D."""
    g, fib, A0 = _soliton_setup(N)
    z0 = soliton_period(1.0, -1.0)
    run = Run(g, fib, A0, length=1.25 * z0, nsave=501, nsteps=nsteps)
    return [run], lambda r: _finish_higher(N, g, A0, z0, r[0])


def check_higher_soliton(N, nsteps=8000):
    return _one(plan_higher_soliton(N, nsteps))


def _finish_higher(N, g, A0, z0, res):
    At = res.At
    I0 = np.abs(A0) ** 2
    dist = np.max(np.abs(np.abs(At) ** 2 - I0), axis=-1) / I0.max()
    # recurrence: minimum distance for z > z0/2, refined with a parabola
    mask = res.z > 0.5 * z0
    i = np.nonzero(mask)[0][np.argmin(dist[mask])]
    y0, y1, y2 = dist[i - 1], dist[i], dist[i + 1]
    dz = res.z[1] - res.z[0]
    z_rec = res.z[i] + 0.5 * dz * (y0 - y2) / (y0 - 2 * y1 + y2)
    i_z0 = np.argmin(np.abs(res.z - z0))
    shape_err = dist[i_z0]
    period_err = abs(z_rec / z0 - 1)
    ok = period_err < 5e-3 and shape_err < 1e-3 and dist.max() > 0.5
    return _check(f"N = {N} soliton period", dict(period_err=period_err, shape_err=shape_err),
                  "period within 0.5%, |A|^2 error at z0 < 1e-3", ok,
                  z=res.z / z0, dist=dist, z_rec=z_rec / z0,
                  map=_intensity_map(g.t, At, every=4), z_map=res.z[::4] / z0)


# --------------------------------------------------------------- conservation
def plan_conservation(length=0.02, tol=1e-10, nsteps=2000):
    """Kerr + full dispersion (no Raman, no shock, no loss): energy conserved.

    Benchmark pulse (N = 8.7) through soliton fission. RK4IP is not unitary, so
    its drift is its truncation error (adaptive, tol); the symmetric split step
    with an exact Kerr sub-step is unitary and conserves energy to rounding.
    """
    g = Grid(2**13, 12.5, DUDLEY_WAVELENGTH)
    fib = dudley_pcf(raman=False, self_steepening=False, length=length)
    A0 = sech_pulse(g.t, DUDLEY_P0, DUDLEY_T0)
    runs = [Run(g, fib, A0, nsave=51, tol=tol),
            Run(g, fib, A0, nsave=51, method="ssfm", nsteps=nsteps)]
    return runs, lambda r: _finish_conservation(g, r)


def check_conservation(length=0.02, tol=1e-10, nsteps=2000):
    return _one(plan_conservation(length, tol, nsteps))


def _finish_conservation(g, results):
    out = {}
    for method, res in zip(("rk4ip", "ssfm"), results):
        E = an.energy(g, res.At)
        out[method] = np.abs(E / E[0] - 1)
    worst = max(out["rk4ip"].max(), out["ssfm"].max())
    return _check("Energy conservation (Kerr + dispersion)",
                  {k: float(v.max()) for k, v in out.items()}, "< 1e-8", worst < 1e-8,
                  z=res.z, drift=out)


def check_photon_number(bench=None):
    """Extension: full GNLSE without loss conserves photon number but not energy."""
    g, fib, res = run_benchmark() if bench is None else bench
    E = an.energy(g, res.At)
    Nph = an.photon_number(g, res.Aw)
    dN = np.abs(Nph / Nph[0] - 1)
    dE = np.abs(E / E[0] - 1)
    return _check("Photon number (full GNLSE)", dict(photon=float(dN.max()), energy=float(dE.max())),
                  "photon drift < 1e-6 while energy changes", dN.max() < 1e-6 and dE.max() > 1e-2,
                  z=res.z, dN=dN, dE=dE, result=res)


# ---------------------------------------------------------------------- Raman
def plan_raman_shift(T0=0.1, n_LD=5.0):
    """Soliton self-frequency shift vs Gordon: dOmega/dz = -8 T_R |beta2| / (15 T0^4)."""
    beta2, gamma = -11.830e-3, 0.11
    g = Grid(2**12, 6.0, DUDLEY_WAVELENGTH)
    fib = Fibre(betas=[beta2], gamma=gamma, raman=True)
    LD = dispersion_length(T0, beta2)
    P0 = soliton_power(1, gamma, T0, beta2)
    run = Run(g, fib, sech_pulse(g.t, P0, T0), length=n_LD * LD, nsave=101, tol=1e-8)
    return [run], lambda r: _finish_raman(g, fib, T0, beta2, LD, r[0])


def check_raman_shift(T0=0.1, n_LD=5.0):
    return _one(plan_raman_shift(T0, n_LD))


def _finish_raman(g, fib, T0, beta2, LD, res):
    TR = fib.raman_time
    w_mean = an.mean_frequency(g.omega, res.Aw)
    # fit after one dispersion length (the pulse first sheds a little radiation)
    m = res.z >= LD
    slope = np.polyfit(res.z[m], w_mean[m], 1)[0]
    gordon = -8 * TR * abs(beta2) / (15 * T0**4)
    rel = abs(slope / gordon - 1)
    return _check("Raman soliton self-frequency shift", dict(measured=slope, gordon=gordon, rel=rel),
                  "within 20% of Gordon", rel < 0.2, z=res.z, w=w_mean, LD=LD, TR=TR)


# ------------------------------------------------------------ dispersive wave
def plan_dispersive_wave(beta3=0.5, N=1.0, n_LD=10.0):
    """Soliton + third-order dispersion: radiation peak at the phase-matched frequency."""
    g = Grid(2**13, 200.0)
    fib = Fibre(betas=[-1.0, beta3], gamma=1.0)
    run = Run(g, fib, sech_pulse(g.t, N**2, 1.0), length=n_LD, tol=1e-8)
    return [run], lambda r: _finish_dw(g, fib, N, r[0])


def check_dispersive_wave(beta3=0.5, N=1.0, n_LD=10.0):
    return _one(plan_dispersive_wave(beta3, N, n_LD))


def _finish_dw(g, fib, N, res):
    Om = g.omega_shifted
    S = np.abs(np.fft.fftshift(res.Aw[-1])) ** 2
    # soliton centre after propagation (it recoils slightly) and its peak power
    w_s = Om[np.argmax(S)]
    P_s = np.max(np.abs(res.final) ** 2)
    w_pm = an.dispersive_wave_offset(fib, P_s, omega_s=w_s)
    w_pm_simple = an.dispersive_wave_offset(fib, N**2, omega_s=0.0)
    # radiation peak: largest local maximum on the blue side beyond the spectral dip
    blue = Om > w_s + 1.0
    Sb, Ob = S[blue], Om[blue]
    i_dip = np.argmin(Sb[: np.searchsorted(Ob, w_pm)])
    i_pk = i_dip + np.argmax(Sb[i_dip:])
    w_dw = Ob[i_pk]
    width = an.fwhm(Ob[i_dip:], Sb[i_dip:])
    ok = abs(w_dw - w_pm) < width / 2
    return _check("Dispersive-wave frequency", dict(sim=w_dw, phase_match=w_pm, fwhm=width,
                                                    phase_match_no_recoil=w_pm_simple),
                  "|Omega_sim - Omega_pm| < FWHM/2", ok, omega=Om, S=S)


# ------------------------------------------------------------------ benchmark
def plan_benchmark(n=2**13, window=12.5, tol=1e-8, nsave=201, fibre=None, P0=DUDLEY_P0,
                   T0=DUDLEY_T0, **kw):
    """Benchmark run; finishes as (grid, fibre, result). ``kw`` goes to ``Run``."""
    g = Grid(n, window, DUDLEY_WAVELENGTH)
    fib = dudley_pcf() if fibre is None else fibre
    run = Run(g, fib, sech_pulse(g.t, P0, T0), nsave=nsave, tol=tol, **kw)
    return [run], lambda r: (g, fib, r[0])


def run_benchmark(**kw):
    return _one(plan_benchmark(**kw))


def check_benchmark(bench=None):
    g, fib, res = run_benchmark() if bench is None else bench
    beta2 = fib.beta2
    N = soliton_order(fib.gamma, DUDLEY_P0, DUDLEY_T0, beta2)
    LD = dispersion_length(DUDLEY_T0, beta2)
    Lf = fission_length(DUDLEY_T0, beta2, N)
    P = an.peak_power_evolution(res.At)
    z_fiss = res.z[np.argmax(P)]
    lam, S = an.spectrum_wavelength(g, res.Aw[-1])
    lo, hi = an.spectral_edges(lam, S, -40.0)
    ratio = z_fiss / Lf
    sol = [s for s in an.classify_solitons(g, fib, res.final) if s["soliton"]]
    ok = (1 / 1.5 < ratio < 1.5) and lo < 500 and hi > 1300
    return _check("Benchmark (Dudley et al.)",
                  dict(N=N, L_D=LD, L_fiss_theory=Lf, z_fiss=z_fiss, ratio=ratio,
                       lam_min=lo, lam_max=hi, solitons=len(sol)),
                  "z_fiss within x1.5 of L_D/N; -40 dB span < 500 nm to > 1300 nm", ok,
                  grid=g, fibre=fib, result=res, solitons=sol)


# ---------------------------------------------------------------- convergence
CONV_STEPS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def _conv_setup(length, n):
    g = Grid(n, 12.5, DUDLEY_WAVELENGTH)
    return g, dudley_pcf(length=length), sech_pulse(g.t, DUDLEY_P0, DUDLEY_T0)


def plan_convergence(length=0.01, n=2**13, steps=CONV_STEPS, ref_steps=32768):
    """Global error at z = length for the full GNLSE benchmark (through fission).

    Error: ||Aw - Aw_ref|| / ||Aw_ref||, reference = RK4IP with ref_steps.
    Both methods cost four nonlinear evaluations per step here; the cost axis is
    the number of evaluations (machine independent).
    """
    g, fib, A0 = _conv_setup(length, n)
    methods = ("rk4ip", "ssfm")
    runs = [Run(g, fib, A0, nsteps=ref_steps)]
    runs += [Run(g, fib, A0, method=m, nsteps=ns) for m in methods for ns in steps]

    def finish(results):
        ref = results[0].Aw[-1]
        out = {}
        for k, m in enumerate(methods):
            part = results[1 + k * len(steps): 1 + (k + 1) * len(steps)]
            finals = np.array([r.Aw[-1] for r in part])
            err = np.linalg.norm(finals - ref, axis=-1) / np.linalg.norm(ref)
            out[m] = dict(h=length / np.array(steps), err=err,
                          evals=np.array([r.evaluations for r in part]))
        return out
    return runs, finish


def convergence_study(**kw):
    return _one(plan_convergence(**kw))


def _asymptotic_slope(h, err, floor=1e-12, max_err=1e-4):
    """Fit log(err) vs log(h) where the error is small (asymptotic regime)."""
    m = (err > floor) & (err < max_err)
    return np.polyfit(np.log(h[m]), np.log(err[m]), 1)[0]


def check_convergence(study=None):
    study = convergence_study() if study is None else study
    p4 = _asymptotic_slope(study["rk4ip"]["h"], study["rk4ip"]["err"])
    p2 = _asymptotic_slope(study["ssfm"]["h"], study["ssfm"]["err"])
    ok = abs(p4 - 4) < 0.5 and abs(p2 - 2) < 0.3
    return _check("Convergence order", dict(rk4ip=p4, ssfm=p2),
                  "RK4IP ~ h^4, split step ~ h^2", ok, study=study)


def plan_window_study(length=0.15, dt=12.5 / 2**13, windows=(1.5625, 3.125, 6.25, 12.5, 25.0),
                      n_fixed_window=12.5, ns=(2**11, 2**12, 2**13, 2**14, 2**15), tol=1e-8):
    """Discretisation error from the time window (fixed dt) and from the number of
    points (fixed window, so the frequency span changes).

    Window study: fields at the same sampling times are compared on the time range
    of the smallest window, against the widest window.
    Points study: spectra on the same frequency bins (dOmega fixed by the window)
    are compared on the span of the coarsest grid, against the finest grid.
    """
    win_jobs = [(int(round(W / dt)), W) for W in windows]
    n_jobs = [(n, n_fixed_window) for n in ns]
    jobs = sorted(set(win_jobs + n_jobs))
    grids = [Grid(n, W, DUDLEY_WAVELENGTH) for n, W in jobs]
    fib = dudley_pcf(length=length)
    runs = [Run(g, fib, sech_pulse(g.t, DUDLEY_P0, DUDLEY_T0), tol=tol, keep="ends")
            for g in grids]

    def finish(results):
        done = {j: (g.t, r.final, g.omega, r.Aw[-1]) for j, g, r in zip(jobs, grids, results)}
        return _finish_window(windows, ns, [done[j] for j in win_jobs], [done[j] for j in n_jobs])
    return runs, finish


def window_study(**kw):
    return _one(plan_window_study(**kw))


def _finish_window(windows, ns, win_runs, n_runs):
    ref_t, ref_A = win_runs[-1][0], win_runs[-1][1]
    t_lo = win_runs[0][0]
    sel = lambda t: (t >= t_lo[0] - 1e-12) & (t <= t_lo[-1] + 1e-12)
    ref_c = ref_A[sel(ref_t)]
    win_err = [np.linalg.norm(A[sel(t)] - ref_c) / np.linalg.norm(ref_c)
               for t, A, _, _ in win_runs]
    win_edge = [an.time_edge_fraction(A) for _, A, _, _ in win_runs]

    ref_w, ref_Aw = n_runs[-1][2], n_runs[-1][3]
    w_hi = np.abs(n_runs[0][2]).max()
    keep = lambda w: np.abs(w) < w_hi
    # same frequency bins on every grid; compare on the span of the coarsest one
    span = lambda w, Aw: Aw[keep(w)][np.argsort(w[keep(w)])]
    ref_s = span(ref_w, ref_Aw)
    n_err = [np.linalg.norm(span(w, Aw) - ref_s) / np.linalg.norm(ref_s)
             for _, _, w, Aw in n_runs]
    n_edge = [an.edge_fraction(Aw) for _, _, _, Aw in n_runs]
    return dict(windows=np.array(windows), win_err=np.array(win_err), win_edge=np.array(win_edge),
                ns=np.array(ns), n_err=np.array(n_err), n_edge=np.array(n_edge))
