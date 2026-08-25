"""Run the pass-the-bar checks, the benchmark and the convergence studies.

Writes results/checks.json, results/checks_table.md and figures 1-5.
All propagations run as GPU batches in one call (see supercontinuum/batch.py) and
are cached in results/checks.pkl; pass --fresh to recompute. --small runs a reduced
smoke test of the same pipeline into results/small/.

    python scripts/run_checks.py [--fresh] [--small]
"""

import json
import pickle
import sys
import time

import numpy as np

from style import plt, save, map_axes, RES, SEQ, SMALL, C1, C2, C3, C4, C5, INK2

from supercontinuum import checks as ck, analysis as an
from supercontinuum.grid import C_NM_PS
from supercontinuum.batch import combine, execute


def plans():
    """Every propagation of the checks, as plans for one batched ``execute`` call."""
    if SMALL:   # smoke test: same code path, small sizes (checks may not pass)
        conv = ck.plan_convergence(steps=(64, 128, 256, 512), ref_steps=2048, length=0.002)
        bench = ck.plan_benchmark(nsave=51)
        window = ck.plan_window_study(length=0.03, windows=(3.125, 6.25, 12.5),
                                      ns=(2**11, 2**12, 2**13))
    else:
        conv, bench, window = ck.plan_convergence(), ck.plan_benchmark(), ck.plan_window_study()
    return dict(
        quick=combine(ck.plan_linear(), ck.plan_spm(), ck.plan_fundamental_soliton(),
                      ck.plan_higher_soliton(2), ck.plan_higher_soliton(3)),
        check_conservation=ck.plan_conservation(),
        check_raman_shift=ck.plan_raman_shift(),
        check_dispersive_wave=ck.plan_dispersive_wave(),
        run_benchmark=bench, convergence_study=conv, window_study=window)


def run_all(fresh):
    """All results, cached in results/checks.pkl (--fresh recomputes)."""
    path = RES / "checks.pkl"
    if path.exists() and not fresh:
        return pickle.loads(path.read_bytes())
    t = time.perf_counter()
    todo = plans()
    data = dict(zip(todo, execute(*todo.values())))
    print(f"  all propagations: {time.perf_counter() - t:.1f} s")
    path.write_bytes(pickle.dumps(data))
    return data


# ---------------------------------------------------------------- figures
def fig_limits(quick):
    lin, spm, sol1, sol2, sol3 = quick
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.2))

    a = ax[0, 0]
    d = lin["data"]
    a.plot(d["z"], d["theory"], color=INK2, lw=3, alpha=0.35, label="$T_0\\sqrt{1+(z/L_D)^2}$")
    a.plot(d["z"], d["width"], "o", ms=4, color=C1, label="simulated (RMS)")
    a.set(xlabel="$z/L_D$", ylabel="width / $T_0$", title="(a) Linear Gaussian broadening")
    a.legend()

    a = ax[0, 1]
    d = spm["data"]
    for k, (phi, S) in enumerate(zip(d["phis"], d["spectra"])):
        a.plot(d["omega"], S + 1.15 * k, color=[C1, C2, C3, C5][k], lw=1.2,
               label=f"$\\phi_{{max}}={phi}\\pi$: M = {spm['value'][k]}")
    a.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7.5, reverse=True)
    a.set(xlim=(-10, 10), xlabel="$\\Omega T_0$", ylabel="spectrum (offset)",
          title="(b) SPM: $\\phi_{max}=(M-\\frac{1}{2})\\pi$", yticks=[])

    a = ax[0, 2]
    d = sol1["data"]
    a.semilogy(d["z"], np.maximum(d["err"], 1e-16), color=C1)
    a.set(xlabel="$z/L_D$", ylabel="max $||A|-|A_0||/|A_0|_{max}$",
          title="(c) Fundamental soliton, 20 $L_D$", ylim=(1e-16, 1e-2))

    for a, s, N in ((ax[1, 0], sol2, 2), (ax[1, 1], sol3, 3)):
        d = s["data"]
        t, I = d["map"]
        im = a.pcolormesh(t, d["z_map"], I / I.max(), cmap=SEQ, shading="auto",
                          rasterized=True)
        map_axes(a)
        a.axhline(1.0, color=C2, lw=1, ls="--")
        a.set(xlabel="$T/T_0$", ylabel="$z/z_0$",
              title=f"({'de'[N - 2]}) N = {N}: recurrence at {d['z_rec']:.6f} $z_0$")
        fig.colorbar(im, ax=a, label="$|A|^2/\\max$", pad=0.02)

    a = ax[1, 2]
    for s, N, c in ((sol2, 2, C1), (sol3, 3, C2)):
        d = s["data"]
        a.semilogy(d["z"], np.maximum(d["dist"], 1e-14), color=c, label=f"N = {N}")
    a.axvline(1.0, color=INK2, lw=0.8, ls=":")
    a.set(xlabel="$z/z_0$", ylabel="max $||A|^2-|A_0|^2| / P_{max}$",
          title="(f) Distance from the input shape")
    a.legend()
    fig.tight_layout()
    save(fig, "fig1_limits.png")


def fig_raman_dw(cons, raman, dw, sweep=None):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))

    a = ax[0]
    d = cons["data"]
    a.semilogy(d["z"] * 100, np.maximum(d["drift"]["rk4ip"], 1e-17), color=C1, label="RK4IP (adaptive, tol $10^{-10}$)")
    a.semilogy(d["z"] * 100, np.maximum(d["drift"]["ssfm"], 1e-17), color=C2, label="split step (exact Kerr)")
    a.set(xlabel="z (cm)", ylabel="$|E/E_0-1|$", title="(a) Energy, Kerr + $\\beta_2..\\beta_{10}$",
          ylim=(1e-17, 1e-6))
    a.legend(loc="lower right")

    a = ax[1]
    d, v = raman["data"], raman["value"]
    z = d["z"] / d["LD"]
    a.plot(z, d["w"], color=C1, label=f"simulated: {v['measured']:.4f} rad/ps/m")
    a.plot(z, v["gordon"] * d["z"], color=C2, ls="--", label=f"Gordon: {v['gordon']:.4f} rad/ps/m")
    a.set(xlabel="$z/L_D$", ylabel="$\\langle\\Omega\\rangle$ (rad/ps)",
          title=f"(b) Soliton self-frequency shift, $T_0$ = 100 fs\n$T_R$ = {d['TR']*1e3:.3f} fs")
    a.legend()

    a = ax[2]
    d, v = dw["data"], dw["value"]
    a.semilogy(d["omega"], d["S"] / d["S"].max(), color=C1, lw=1.2)
    a.axvline(v["phase_match"], color=C2, ls="--", lw=1.2, label=f"phase matching {v['phase_match']:.3f}")
    a.axvline(v["phase_match_no_recoil"], color=C4, ls=":", lw=1.2,
              label=f"no recoil, $P_s=P_0$: {v['phase_match_no_recoil']:.3f}")
    a.set(xlim=(-4, 9), ylim=(1e-10, 5), xlabel="$\\Omega T_0$", ylabel="spectrum (norm.)",
          title=f"(c) Dispersive wave ($\\beta_3$ = 0.5): peak at {v['sim']:.3f}")
    a.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig2_conservation_raman_dw.png")


def fig_benchmark(bench_check):
    d = bench_check["data"]
    g, res, v = d["grid"], d["result"], bench_check["value"]
    lam, S = an.spectrum_wavelength(g, res.Aw)
    Sdb = an.to_db(S)
    I = np.abs(res.At) ** 2
    Idb = an.to_db(I)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a = ax[0]
    m = (lam > 400) & (lam < 1400)
    im = a.pcolormesh(lam[m], res.z * 100, Sdb[:, m], vmin=-40, vmax=0, cmap=SEQ,
                      shading="auto", rasterized=True)
    map_axes(a)
    a.axhline(v["L_fiss_theory"] * 100, color=C2, ls="--", lw=1)
    a.text(1380, v["L_fiss_theory"] * 100 + 0.3, "$L_D/N$", color=C2, ha="right", fontsize=8.5)
    a.set(xlabel="wavelength (nm)", ylabel="z (cm)", title="(a) Spectral evolution (dB)")
    fig.colorbar(im, ax=a, pad=0.02, label="dB")
    a = ax[1]
    m = (g.t > -0.5) & (g.t < 5)
    im = a.pcolormesh(g.t[m], res.z * 100, Idb[:, m], vmin=-40, vmax=0, cmap=SEQ,
                      shading="auto", rasterized=True)
    map_axes(a)
    a.axhline(v["z_fiss"] * 100, color=C2, ls="--", lw=1)
    a.text(4.9, v["z_fiss"] * 100 + 0.3, "max. compression", color=C2, ha="right", fontsize=8.5)
    a.set(xlabel="delay (ps)", ylabel="z (cm)", title="(b) Temporal evolution (dB)")
    fig.colorbar(im, ax=a, pad=0.02, label="dB")
    fig.tight_layout()
    save(fig, "fig3_benchmark_evolution.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    a = ax[0]
    sdb = an.to_db(S[-1])
    a.plot(lam, sdb, color=C1, lw=1.0, label="output, z = 15 cm")
    a.plot(lam, an.to_db(S[0], S[-1].max()), color=INK2, lw=1.0, alpha=0.6, label="input")
    a.axhline(-40, color=C2, ls="--", lw=1)
    for x in (v["lam_min"], v["lam_max"]):
        a.axvline(x, color=C2, ls=":", lw=1)
    a.text(v["lam_min"] + 10, -55, f"{v['lam_min']:.0f} nm", color=INK2, fontsize=8)
    a.text(v["lam_max"] - 10, -55, f"{v['lam_max']:.0f} nm", color=INK2, fontsize=8, ha="right")
    a.set(xlim=(400, 1400), ylim=(-70, 5), xlabel="wavelength (nm)", ylabel="spectral density (dB)",
          title="(a) Output spectrum")
    a.legend(loc="upper right")
    a = ax[1]
    delays, Om, Sg = an.spectrogram(g, res.final, gate_fwhm=0.05, n_delays=300, t_range=(-0.5, 5.0))
    lam_g = 2 * np.pi * C_NM_PS / (g.omega0 + Om)
    m = (lam_g > 400) & (lam_g < 1400)
    o = np.argsort(lam_g[m])
    im = a.pcolormesh(delays, lam_g[m][o], an.to_db(Sg[:, m][:, o]).T, vmin=-40, vmax=0,
                      cmap=SEQ, shading="auto", rasterized=True)
    map_axes(a)
    a.set(xlabel="delay (ps)", ylabel="wavelength (nm)",
          title="(b) Output spectrogram (50 fs Gaussian gate, dB)")
    fig.colorbar(im, ax=a, pad=0.02, label="dB")
    fig.tight_layout()
    save(fig, "fig4_benchmark_output.png")


def fig_convergence(conv_check, window):
    st = conv_check["study"] if "study" in conv_check else conv_check["data"]["study"]
    p = conv_check["value"]
    fig, ax = plt.subplots(2, 2, figsize=(9.5, 6.4))
    a = ax[0, 0]
    for m, c, lab, order in (("rk4ip", C1, "RK4IP", 4), ("ssfm", C2, "split step", 2)):
        h, e = st[m]["h"] * 1e6, st[m]["err"]
        a.loglog(h, e, "o-", ms=4, color=c, label=f"{lab}: slope {p[m]:.2f}")
        ref = e[-1] * (h / h[-1]) ** order
        a.loglog(h, ref, color=c, lw=0.8, ls=":")
    a.set(xlabel="step h (µm)", ylabel="relative error at z = 1 cm",
          title="(a) Error vs step (full GNLSE, through fission)")
    a.legend()
    a = ax[0, 1]
    for m, c, lab in (("rk4ip", C1, "RK4IP"), ("ssfm", C2, "split step")):
        a.loglog(st[m]["evals"], st[m]["err"], "o-", ms=4, color=c, label=lab)
    a.set(xlabel="nonlinear-operator evaluations", ylabel="relative error",
          title="(b) Error vs cost")
    a.legend()
    a = ax[1, 0]
    W, e = window["windows"][:-1], window["win_err"][:-1]
    a.plot(W, e, "o-", ms=4, color=C1)
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set(xlabel="time window (ps), dt = 1.53 fs", ylabel="relative error at 15 cm",
          title="(c) Time window (reference: 25 ps)")
    a.set_xticks(W, [f"{w:g}" for w in W])
    a = ax[1, 1]
    n, e = window["ns"][:-1], window["n_err"][:-1]
    a.plot(n, e, "s-", ms=4, color=C2)
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set(xlabel="number of points (window 12.5 ps)", ylabel="relative spectral error at 15 cm",
          title="(d) Grid size (reference: $2^{15}$)")
    a.set_xticks(n, [f"$2^{{{int(np.log2(k))}}}$" for k in n])
    fig.tight_layout()
    save(fig, "fig5_convergence.png")


# ---------------------------------------------------------------- table
def fmt(v):
    if isinstance(v, dict):
        return ", ".join(f"{k} = {fmt(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    if isinstance(v, (float, np.floating)):
        return f"{v:.3g}"
    return str(v)


def main():
    fresh = "--fresh" in sys.argv
    data = run_all(fresh)
    window = data["window_study"]
    quick = data["quick"]
    bench = ck.check_benchmark(data["run_benchmark"])
    photon = ck.check_photon_number(data["run_benchmark"])
    conv = ck.check_convergence(data["convergence_study"])
    checks = quick + [data["check_conservation"], data["check_raman_shift"],
                      data["check_dispersive_wave"], bench, conv, photon]

    rows = ["| Test | Measured |", "|---|---|"]
    summary = []
    for c in checks:
        rows.append(f"| {c['name']} | {fmt(c['value'])} |")
        summary.append(dict(name=c["name"], value=fmt(c["value"]), target=c["target"],
                            passed=c["passed"]))
    table = "\n".join(rows)
    print(table)
    (RES / "checks_table.md").write_text(table + "\n")
    (RES / "checks.json").write_text(json.dumps(summary, indent=2))
    w = {k: v.tolist() for k, v in window.items()}
    (RES / "window_study.json").write_text(json.dumps(w, indent=2))
    print("window study:", json.dumps(w))
    print("convergence:", {m: data["convergence_study"][m]["err"].tolist()
                           for m in ("rk4ip", "ssfm")})

    fig_limits(quick)
    fig_raman_dw(data["check_conservation"], data["check_raman_shift"], data["check_dispersive_wave"])
    fig_benchmark(bench)
    fig_convergence(dict(study=data["convergence_study"], value=conv["value"]), window)
    return all(c["passed"] for c in checks)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
