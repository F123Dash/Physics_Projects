"""Extension studies.

E1  Photon-number conservation of the full GNLSE (energy is not conserved: Raman).
E2  Validity of the Gordon formula: SSFS rate / Gordon vs soliton duration T0.
E3  Higher-order soliton compression and fission vs N: pure NLS against the
    empirical laws z_opt/z0 = 0.32/N + 1.1/N^2 and F_c = 4.1 N, and the full GNLSE
    against L_fiss = L_D / N.
E4  Coherence of the supercontinuum vs input pulse duration (one-photon-per-mode
    noise ensembles, |g12(lambda)|).
E5  Ablation: the benchmark with each higher-order term switched on in turn.

All propagations go to the GPU as batches in one call (supercontinuum/batch.py).
--small runs a reduced smoke test of the same pipeline into results/small/.

    python scripts/run_extensions.py [--fresh] [--redo a,b] [--small]
"""

import json
import pickle
import sys
import time

import numpy as np

from style import plt, save, map_axes, RES, SEQ, SMALL, C1, C2, C3, C4, C5, INK2, SERIES

from supercontinuum import (Grid, Fibre, sech_pulse, one_photon_noise, dudley_pcf,
                            dispersion_length, soliton_order, soliton_power, soliton_period)
from supercontinuum import analysis as an, checks as ck
from supercontinuum.batch import Run, combine, execute
from supercontinuum.fibre import DUDLEY_P0, DUDLEY_T0, DUDLEY_WAVELENGTH

N_SEEDS = 16
FWHM_LIST = (0.050, 0.100, 0.150, 0.200)   # ps
N_LIST = (2, 3, 4, 5, 6, 8, 10, 12, 14, 16)
T0_LIST = (0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3)
N_WIDE = 2**14   # broad spectra (long pulses, no shock term) reach 440 nm: 2^13 aliases
LENGTH = 0.15    # m, coherence and ablation
if SMALL:        # smoke test of the same pipeline
    N_SEEDS, FWHM_LIST, N_LIST, T0_LIST, LENGTH = 4, (0.05, 0.1), (2, 4, 8), (0.05, 0.1, 0.2), 0.03


# ------------------------------------------------------------------ plans
# each plan is (runs, finish); all of them go to the GPU in one execute() call
def plan_raman():
    def finish(checks):
        return [(T0, c["value"]["measured"], c["value"]["gordon"])
                for T0, c in zip(T0_LIST, checks)]
    runs, fin = combine(*[ck.plan_raman_shift(T0=T0) for T0 in T0_LIST])
    return runs, lambda r: finish(fin(r))


def plan_nls_compression(N):
    """Pure NLS (beta2, Kerr), dimensionless units T0 = |beta2| = gamma = 1."""
    g = Grid(2**14, 40.0)
    fib = Fibre(betas=[-1.0], gamma=1.0)
    z0 = soliton_period(1.0, -1.0)
    zmax = min(1.0, 2.5 * (0.32 / N + 1.1 / N**2)) * z0
    best = dict(P=-1.0)

    def track(z, Aw):   # keep only the most compressed pulse, not the history
        I = np.abs(g.to_time(Aw)) ** 2
        if I.max() > best["P"]:
            best.update(P=I.max(), z=z, I=I)

    A0 = sech_pulse(g.t, N**2, 1.0)
    run = Run(g, fib, A0, length=zmax, nsave=801, tol=1e-9, keep="ends", callback=track)

    def finish(_):
        f_in = an.fwhm(g.t, np.abs(A0) ** 2)
        return N, best["z"] / z0, f_in / an.fwhm(g.t, best["I"]), best["P"] / N**2
    return [run], finish


def plan_gnlse_fission(N):
    """Benchmark fibre and pulse duration, power set for soliton order N."""
    fib = dudley_pcf()
    P0 = soliton_power(N, fib.gamma, DUDLEY_T0, fib.beta2)
    LD = dispersion_length(DUDLEY_T0, fib.beta2)
    g = Grid(2**13, 12.5, DUDLEY_WAVELENGTH)
    zmax = 2.5 * LD / N if N > 2 else 1.5 * LD
    peak = []
    run = Run(g, fib, sech_pulse(g.t, P0, DUDLEY_T0), length=zmax, nsave=401, tol=1e-7,
              keep="ends",
              callback=lambda z, Aw: peak.append((z, np.max(np.abs(g.to_time(Aw)) ** 2))))

    def finish(_):
        z, P = np.array(peak).T
        return N, z[np.argmax(P)] / (LD / N)
    return [run], finish


def coherence_grid(fw):
    """Pulses >= 100 fs shed Raman solitons that walk off more than the 12.5 ps
    window holds (16-19% of the energy wrapped round at 150-200 fs): they get a
    25 ps window at the same dt (2^15 points)."""
    return (Grid(N_WIDE, 12.5, DUDLEY_WAVELENGTH) if fw < 0.075
            else Grid(2 * N_WIDE, 25.0, DUDLEY_WAVELENGTH))


def plan_coherence():
    """One-photon-noise ensembles: N_SEEDS shots for each pulse duration."""
    fib = dudley_pcf(length=LENGTH)
    runs = []
    for k, fw in enumerate(FWHM_LIST):
        g = coherence_grid(fw)
        for s in range(N_SEEDS):
            rng = np.random.default_rng(1000 * k + s)
            A0 = sech_pulse(g.t, DUDLEY_P0, fw / 1.7627) + g.to_time(one_photon_noise(g, rng))
            runs.append(Run(g, fib, A0, tol=1e-7, keep="ends"))

    def finish(results):
        coh, edges = {}, {}
        for k, fw in enumerate(FWHM_LIST):
            part = results[k * N_SEEDS:(k + 1) * N_SEEDS]
            coh[fw] = np.array([r.Aw[-1] for r in part])
            edges[fw] = np.max([(an.edge_fraction(r.Aw[-1]), an.time_edge_fraction(r.final))
                                for r in part], axis=0)
        return coh, edges
    return runs, finish


ABLATION = (("dispersion + Kerr", False, False), ("+ self-steepening", False, True),
            ("+ Raman", True, False), ("full GNLSE", True, True))


def plan_ablation():
    plans = [ck.plan_benchmark(n=N_WIDE, fibre=dudley_pcf(raman=raman, self_steepening=ss,
                                                          length=LENGTH), nsave=2, tol=1e-7)
             for _, raman, ss in ABLATION]
    runs, fin = combine(*plans)

    def finish(results):
        return [(label, g, res.Aw[-1], res.steps)
                for (label, _, _), (g, _, res) in zip(ABLATION, fin(results))]
    return runs, finish


def compute(keys):
    """Compute the studies named in ``keys``: every run goes into one batched call."""
    makers = dict(ablation=plan_ablation, coherence=plan_coherence,
                  nls=lambda: combine(*[plan_nls_compression(N) for N in N_LIST]),
                  fission=lambda: combine(*[plan_gnlse_fission(N) for N in N_LIST]),
                  raman=plan_raman)
    keys = [k for k in makers if k in keys]
    out = dict(zip(keys, execute(*[makers[k]() for k in keys])))
    if "coherence" in out:
        out["coherence"], out["coherence_edges"] = out["coherence"]
    return out


# ------------------------------------------------------------------ figures
def fig_photon(bench):
    c = ck.check_photon_number(bench)
    d = c["data"]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.semilogy(d["z"] * 100, np.maximum(d["dE"], 1e-16), color=C2, label="energy $|E/E_0-1|$")
    ax.semilogy(d["z"] * 100, np.maximum(d["dN"], 1e-16), color=C1,
                label="photon number $|N_{ph}/N_{ph,0}-1|$")
    ax.set(xlabel="z (cm)", ylabel="relative change", ylim=(1e-14, 1),
           title="E1: Invariants of the full GNLSE (benchmark)")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig6_photon_number.png")
    return c


def fig_scaling(raman, nls, fission):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
    a = ax[0]
    T0 = np.array([r[0] for r in raman]) * 1e3
    ratio = np.array([r[1] / r[2] for r in raman])
    a.loglog(T0, ratio - 1, "o", ms=5, color=C1, label="simulation")
    # slope fitted on the long-pulse half
    m = T0 >= 70
    p = np.polyfit(np.log(T0[m]), np.log(ratio[m] - 1), 1)
    a.loglog(T0, np.exp(np.polyval(p, np.log(T0))), color=C2, ls="--",
             label=f"fit $\\propto T_0^{{{p[0]:.2f}}}$")
    a.set(xlabel="$T_0$ (fs)", ylabel="rate / Gordon $-$ 1",
          title="E2: Gordon formula validity")
    a.set_xticks([30, 50, 100, 200, 300], ["30", "50", "100", "200", "300"])
    a.xaxis.set_minor_formatter(plt.NullFormatter())
    a.legend()

    a = ax[1]
    N = np.array([r[0] for r in nls])
    zopt = np.array([r[1] for r in nls])
    Fc = np.array([r[2] for r in nls])
    Nf = np.linspace(1.8, 17, 200)
    a.plot(N, zopt, "o", ms=5, color=C1, label="NLS: $z_{opt}/z_0$")
    a.plot(Nf, 0.32 / Nf + 1.1 / Nf**2, color=C1, lw=1, ls="--", label="$0.32/N + 1.1/N^2$")
    a.set(xlabel="soliton order N", ylabel="$z_{opt}/z_0$", title="E3a: Compression point (NLS)")
    a2 = a.twinx()
    a2.plot(N, Fc, "s", ms=4, color=C2, label="compression factor $F_c$")
    a2.plot(Nf, 4.1 * Nf, color=C2, lw=1, ls="--", label="$4.1N$")
    a2.set_ylabel("$F_c$", color=C2)
    a2.grid(False)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper center")

    a = ax[2]
    Nfis = np.array([r[0] for r in fission])
    rat = np.array([r[1] for r in fission])
    a.plot(Nfis, rat, "o-", ms=5, color=C1, label="full GNLSE, benchmark fibre")
    zth = (0.32 / Nfis + 1.1 / Nfis**2) * (np.pi / 2) * Nfis
    a.plot(Nfis, zth, color=C2, ls="--", lw=1, label="NLS empirical law")
    a.axhline(1, color=INK2, lw=0.8)
    a.set(xlabel="soliton order N", ylabel="$z_{fiss} / (L_D/N)$",
          title="E3b: Fission length vs N")
    a.legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig7_scaling.png")
    return dict(gordon_exponent=float(p[0]), raman=[(float(a), float(b / c)) for a, b, c in raman],
                nls=[tuple(map(float, r)) for r in nls], fission=[tuple(map(float, r)) for r in fission])


def fig_coherence(coh):
    fig, ax = plt.subplots(len(coh), 1, figsize=(7.5, 2.1 * len(coh)), sharex=True)
    summary = {}
    for a, (fw, Aws) in zip(ax, sorted(coh.items())):
        g = coherence_grid(fw)
        lam, S = an.spectrum_wavelength(g, Aws)
        w = g.absolute_omega
        mask = w > 0
        o = np.argsort(2 * np.pi / w[mask])
        g12 = an.coherence(Aws[:, mask][:, o])
        mean_S = S.mean(axis=0)
        sdb = an.to_db(mean_S)
        sel = sdb > -40
        avg = np.sum(g12[sel] * mean_S[sel]) / np.sum(mean_S[sel])
        T0 = fw / 1.7627
        N = soliton_order(0.11, DUDLEY_P0, T0, -11.830e-3)
        summary[fw] = dict(N=float(N), mean_g12=float(avg))
        for k in range(min(5, len(S))):
            a.plot(lam, an.to_db(S[k], mean_S.max()), color=INK2, lw=0.4, alpha=0.4)
        a.plot(lam, sdb, color=C1, lw=1.0, label="ensemble mean (dB)")
        a.set(ylim=(-60, 5), xlim=(400, 1400), ylabel="dB")
        a.set_title(f"$T_{{FWHM}}$ = {fw*1e3:.0f} fs,  N = {N:.1f},  "
                    f"spectrally averaged $\\langle|g_{{12}}|\\rangle$ = {avg:.2f}", fontsize=9)
        a2 = a.twinx()
        a2.plot(lam, np.where(sdb > -40, g12, np.nan), color=C2, lw=1.0, label="$|g_{12}|$")
        a2.set(ylim=(0, 1.05))
        a2.set_ylabel("$|g_{12}|$", color=C2)
        a2.grid(False)
    ax[-1].set_xlabel("wavelength (nm)")
    ax[0].legend(loc="lower right", fontsize=7.5)
    fig.suptitle(f"E4: Coherence vs pulse duration ({N_SEEDS} noise seeds each, "
                 f"P0 = 10 kW, 15 cm; grey: single shots)", fontsize=10)
    fig.tight_layout()
    save(fig, "fig8_coherence.png")
    return summary


def fig_ablation(abl):
    fig, ax = plt.subplots(len(abl), 1, figsize=(7, 1.7 * len(abl)), sharex=True)
    summary = {}
    for k, (a, (label, g, Aw, steps)) in enumerate(zip(ax, abl)):
        lam, S = an.spectrum_wavelength(g, Aw)
        lo, hi = an.spectral_edges(lam, S, -40)
        summary[label] = dict(lam_min=float(lo), lam_max=float(hi), steps=int(steps),
                              spectral_edge=float(an.edge_fraction(Aw)))
        a.plot(lam, an.to_db(S), color=SERIES[k], lw=0.9)
        a.set(ylim=(-60, 5), xlim=(400, 1400), ylabel="spectrum (dB)")
        a.text(1390, -10, f"{label}\n-40 dB: {lo:.0f}-{hi:.0f} nm", ha="right", va="top", fontsize=8.5)
    ax[-1].set_xlabel("wavelength (nm)")
    ax[0].set_title("E5: Benchmark output with each term switched on")
    fig.tight_layout()
    save(fig, "fig9_ablation.png")
    return summary


def main():
    fresh = "--fresh" in sys.argv
    path = RES / "extensions.pkl"
    data = {} if fresh or not path.exists() else pickle.loads(path.read_bytes())
    # --redo name[,name] recomputes selected studies
    redo = sys.argv[sys.argv.index("--redo") + 1].split(",") if "--redo" in sys.argv else []
    missing = [k for k in ("raman", "nls", "fission", "ablation", "coherence")
               if k not in data or k in redo]
    if missing:
        t = time.perf_counter()
        data.update(compute(missing))
        print(f"  extensions {missing}: {time.perf_counter() - t:.1f} s")
        path.write_bytes(pickle.dumps(data))
    bench = pickle.loads((RES / "checks.pkl").read_bytes())["run_benchmark"]
    photon = fig_photon(bench)
    summary = dict(photon=dict(photon=photon["value"]["photon"], energy=photon["value"]["energy"]))
    summary.update(fig_scaling(data["raman"], data["nls"], data["fission"]))
    summary["coherence"] = fig_coherence(data["coherence"])
    for fw, (e_w, e_t) in data["coherence_edges"].items():
        summary["coherence"][fw].update(max_spectral_edge=float(e_w), max_time_edge=float(e_t))
    summary["ablation"] = fig_ablation(data["ablation"])
    (RES / "extensions.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
