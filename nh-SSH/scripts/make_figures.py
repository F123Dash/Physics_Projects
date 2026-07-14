"""Generate figures/fig1-fig6 and figures/results.txt.

Usage: python nh-SSH/scripts/make_figures.py [--skip-mpmath]
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import contextlib
import io
import pathlib
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nh_ssh import (
    H_open,
    biorthogonal_eig,
    biorthogonal_density,
    bloch_energies,
    bloch_gap_closings,
    cell_weight,
    cell_weights_of_zero_mode,
    chiral_edge_states,
    eigenvector_condition,
    exact_open_spectrum,
    hermitian_edge_states,
    min_abs_energy,
    obc_transitions,
    polarization,
    polarization_exact,
    r_factors,
    skin_decay_factor,
    skin_factor,
    spectrum_error,
    v_prime,
)

FIG = ROOT / "figures"
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2"
GREY = "#9a9994"
SINGLE = (4.6, 3.3)
DOUBLE = (9.2, 3.5)
T2, G0 = 1.0, 4.0 / 3.0
# At v' = t2 the partner is a uniform 2N-site chain, min|E| = 2 t2 sin(pi / (2(2N + 1))) ~ 1/N
CRIT_RATIO = np.log10(np.sin(np.pi / 122) / np.sin(np.pi / 82))

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.labelsize": 12,
    "axes.linewidth": 0.8,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

results = []


def log(msg):
    print(msg)
    results.append(msg)


def panel(ax, label):
    ax.text(-0.01, 1.015, f"({label})", transform=ax.transAxes, va="bottom", ha="right", fontsize=12)


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)


def mark_transitions(ax, g, bloch=True, ymin=0, ymax=1):
    for x in obc_transitions(T2, g):
        ax.axvline(x, color="k", ls="--", lw=0.9, ymin=ymin, ymax=ymax)
    if bloch:
        for x in bloch_gap_closings(T2, g):
            ax.axvline(x, color=GREY, ls=":", lw=1.1, ymin=ymin, ymax=ymax)


def environment_header():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(Exception):
        np.show_config()
    blas = next((ln.split(":", 1)[1].strip().strip('",') for ln in buf.getvalue().splitlines()
                 if "openblas configuration" in ln), "unknown")
    log(f"# python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}")
    log(f"# BLAS/LAPACK: {blas}")


def fig_spectra(N=30):
    t1 = 0.7
    k = np.linspace(-np.pi, np.pi, 801)
    Eb = bloch_energies(k, t1, T2, G0)
    Eo = np.linalg.eigvals(H_open(N, t1, T2, G0))
    log(f"[spectra] t1={t1}, g=4/3, N={N}: max|Im E_open| = {np.abs(Eo.imag).max():.1e}, "
        f"max|Im E_bloch| = {np.abs(Eb.imag).max():.3f}")

    t1s = np.linspace(0.0, 2.0, 401)
    absE = np.array([np.sort(np.abs(np.linalg.eigvals(H_open(N, t, T2, G0)))) for t in t1s])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE, gridspec_kw={"width_ratios": [1, 1.2]})
    a1.plot(Eb.real[:801], Eb.imag[:801], color=GREY, lw=1.2, label="periodic (Bloch)")
    a1.plot(Eb.real[801:], Eb.imag[801:], color=GREY, lw=1.2)
    a1.plot(Eo.real, Eo.imag, "o", color=BLUE, ms=3.5, mfc="none", mew=0.9, label=f"open, $N={N}$")
    a1.axhline(0, color=GREY, lw=0.4, zorder=0)
    a1.axvline(0, color=GREY, lw=0.4, zorder=0)
    a1.set(xlabel=r"$\mathrm{Re}\,E$", ylabel=r"$\mathrm{Im}\,E$", xlim=(-2.9, 2.9), ylim=(-1.75, 1.75))
    a1.legend(loc="upper center", ncol=2, columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))
    a1.text(0.03, 0.05, rf"$t_1={t1},\ \gamma=4/3$", transform=a1.transAxes)

    a2.plot(t1s, absE[:, 2:], color=GREY, lw=0.4)
    a2.plot(t1s, absE[:, :2], color=ORANGE, lw=1.6)
    mark_transitions(a2, G0)
    a2.axvline(G0 / 2, color=PURPLE, lw=0.8, ls="-.")
    a2.set(xlabel=r"$t_1$", ylabel=r"$|E|$ (open chain)", xlim=(0, 2), ylim=(0, 2.6))
    a2.text(1.2019, 2.3, r"$\sqrt{t_2^2+\gamma^2/4}$", ha="center", fontsize=9,
            bbox={"fc": "white", "ec": "none", "pad": 0.5})
    for x in bloch_gap_closings(T2, G0):
        a2.text(x, 2.1, "Bloch", ha="center", fontsize=9, color="#6b6a66",
                bbox={"fc": "white", "ec": "none", "pad": 0.5})
    a2.text(G0 / 2 + 0.03, 1.65, "EP\n" r"$t_1=\gamma/2$", fontsize=9, color=PURPLE)
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig1_spectra.png")


def fig_skin(N=20, t1=1.0):
    E, R, L = biorthogonal_eig(H_open(N, t1, T2, G0))
    s = abs(skin_factor(t1, G0))
    order = np.argsort(E.real)
    picks = order[[N // 2 + 2, N + 4, 2 * N - 3]]
    n = np.arange(N)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE, sharey=True)
    for ax, V, sign, name in [(a1, R, -1, "R"), (a2, L, +1, "L")]:
        for j in range(2 * N):
            w = cell_weight(V[:, j] / np.linalg.norm(V[:, j]))
            ax.semilogy(n, w, color=GREY, lw=0.4, alpha=0.6)
        for c, m, j in zip([BLUE, ORANGE, GREEN], "o^s", picks):
            w = cell_weight(V[:, j] / np.linalg.norm(V[:, j]))
            ax.semilogy(n, w, m + "-", color=c, ms=3.5, lw=0.9, mfc="white", mew=0.9,
                        label=rf"$E={E[j].real:+.3f}$")
        ref = s ** (2 * sign * (n - (0 if sign < 0 else N - 1)))
        ax.semilogy(n, 0.6 * ref, "k--", lw=1.0)
        ax.set(xlabel=r"cell $n$", xlim=(-0.5, N - 0.5), ylim=(1e-15, 3))
        ax.text(0.5, 0.06,
                rf"dashed: $|s|^{{{2 * sign:+d}n}}$, $|s| = {s:.3f}$", transform=ax.transAxes,
                ha="center", fontsize=9)
    a1.set_ylabel(r"$|R(n)|^2$, $|L(n)|^2$ per cell")
    a1.legend(loc="upper right", fontsize=8.5)
    a1.text(0.05, 0.2, "right eigenvectors", transform=a1.transAxes)
    a2.text(0.95, 0.2, "left eigenvectors", transform=a2.transAxes, ha="right")
    for ax in (a1, a2):
        ax.set_xticks(range(0, N, 4))
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig2_skin.png")

    for t in [0.7, 1.0, 1.4]:
        for Nn in [12, 20]:
            fr = skin_decay_factor(Nn, t, T2, G0, "right")
            fl = skin_decay_factor(Nn, t, T2, G0, "left")
            si = abs(skin_factor(t, G0))
            log(f"[skin] t1={t}, N={Nn}: right decay/cell={fr:.4f}, 1/|s|={1 / si:.4f} "
                f"(rel.err {fr * si - 1:+.3f}); left growth/cell={fl:.4f}, |s|={si:.4f} "
                f"(rel.err {fl / si - 1:+.3f})")


def splitting_ssh(N, t1, g):
    """Edge splitting of the Hermitian partner, from the SSH project: (t2^2 - v'^2)/t2 (v'/t2)^N."""
    vp = v_prime(t1, g).real
    return (T2 ** 2 - vp ** 2) / T2 * (vp / T2) ** N


def fig_exact(Ns=np.arange(4, 37)):
    for t1 in [0.8, 1.0, 1.4]:
        errs = [spectrum_error(N, t1, T2, G0) for N in range(2, 21)]
        log(f"[exact] t1={t1}: max over N=2..20 of |E_eig - E_SSH(v',t2)| = {max(errs):.1e}")
    for t1, Nlist in [(0.5, [40]), (1.0, [10, 20, 30]), (1.4, [10, 20, 30])]:
        vals = [min_abs_energy(N, t1, T2, G0) for N in Nlist]
        log(f"[brief table] t1={t1}: min|E| at N={Nlist} = {[f'{v:.2e}' for v in vals]}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE)
    t1s = [0.5, 1.0, 1.1, 1.2019, 1.3, 1.4]
    cols = [BLUE, BLUE, BLUE, "k", ORANGE, ORANGE]
    marks = ["o", "^", "s", "D", "v", "x"]
    for t1, c, m in zip(t1s, cols, marks):
        eps = np.array([min_abs_energy(N, t1, T2, G0) for N in Ns])
        ok = eps > 1e-15
        a1.semilogy(Ns[ok], eps[ok], m, color=c, ms=3.5, mfc="none" if m != "x" else c, mew=0.9,
                    label=rf"$t_1={t1}$")
        if 0.75 < t1 < 1.2:
            a1.semilogy(Ns, splitting_ssh(Ns, t1, G0), color=c, lw=0.9)
    a1.set(xlabel=r"$N$ (cells)", ylabel=r"$\min|E|$", xlim=(2, 38), ylim=(1e-16, 30))
    a1.legend(loc="lower left", ncol=2, fontsize=8.5, columnspacing=0.8)
    a1.text(0.97, 0.42, r"$\gamma=4/3$, $t_2=1$" "\n" r"lines: $\frac{t_2^2-v'^2}{t_2}\left(\frac{v'}{t_2}\right)^N$",
            transform=a1.transAxes, ha="right", va="bottom", fontsize=9.5)

    for t1 in [0.5, 1.0, 1.1, 1.3, 1.4]:
        eps = [min_abs_energy(N, t1, T2, G0) for N in (10, 20, 30)]
        log(f"[minE] t1={t1}: N=10,20,30 -> {[f'{e:.2e}' for e in eps]}")
    for t1 in [1.0, 1.1]:
        fit = Ns[Ns >= 20]
        slope, _ = np.polyfit(fit, np.log([min_abs_energy(N, t1, T2, G0) for N in fit]), 1)
        log(f"[minE] t1={t1}: slope of ln min|E| vs N (N=20..36) = {slope:.5f}, ln(v'/t2) = {np.log(v_prime(t1, G0).real):.5f}")

    for c, gam in zip([BLUE, GREEN, ORANGE, PURPLE], [0.5, 4 / 3, 2.0, 2.6]):
        tc = obc_transitions(T2, gam)[-1]
        dts = np.linspace(-0.3, 0.3, 61)
        e20 = np.array([min_abs_energy(20, tc + d, T2, gam) for d in dts])
        e30 = np.array([min_abs_energy(30, tc + d, T2, gam) for d in dts])
        ratio = np.log10(e30 / e20)
        a2.plot(dts, ratio, color=c, lw=1.4, label=rf"$\gamma={gam:.3g}$, $t_1^c={tc:.4f}$")
        i = np.where(np.diff(np.sign(ratio - CRIT_RATIO)) != 0)[0][0]
        cross = dts[i] - (ratio[i] - CRIT_RATIO) * (dts[i + 1] - dts[i]) / (ratio[i + 1] - ratio[i])
        log(f"[transition] g={gam:.4f}: N=20/30 ratio crosses the critical value at t1 = {tc + cross:.4f}; "
            f"sqrt(t2^2+g^2/4) = {tc:.4f}; Bloch closings {np.round(bloch_gap_closings(T2, gam), 4).tolist()}")
    a2.axvline(0, color="k", ls="--", lw=0.9)
    a2.axhline(CRIT_RATIO, color=GREY, lw=0.8, ls=":")
    a2.text(-0.29, CRIT_RATIO + 0.15, r"critical: $\log_{10}(41/61)$", fontsize=8.5, color="#6b6a66")
    a2.set(xlabel=r"$t_1 - \sqrt{t_2^2+\gamma^2/4}$",
           ylabel=r"$\log_{10}[\min|E|_{N=30} / \min|E|_{N=20}]$", xlim=(-0.3, 0.3))
    a2.legend(loc="lower right", fontsize=8.5)
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig3_open_vs_exact.png")


def phase_row(args):
    g, t1s, N = args
    minE = np.array([min_abs_energy(N - 1, t, T2, g) for t in t1s])
    exact = np.array([min_abs_energy(N - 1, t, T2, g, exact=True) for t in t1s])
    err = np.array([spectrum_error(N - 1, t, T2, g) for t in t1s])
    P = np.array([polarization(N, t, T2, g) for t in t1s])
    return minE, exact, err, P


def fig_phase(N=31):
    t1s = np.linspace(0.0, 2.0, 201)
    gs = np.linspace(0.0, 3.0, 151)
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(phase_row, [(g, t1s, N) for g in gs]))
    minE, exact, err, P = (np.array([r[i] for r in rows]) for i in range(4))
    with np.errstate(divide="ignore"):
        wrong = (exact > 1e-12) & (np.abs(np.log10(minE / exact)) > np.log10(2))
    log(f"[phase] min|E| from the even chain, N={N - 1}; P from the odd chain, N={N} A-sites")
    log(f"[phase] grid {len(t1s)}x{len(gs)}: max |E_eig - E_exact| = {err.max():.1e}, but min|E| is off "
        f"by more than 2x at {wrong.sum()} of {np.sum(exact > 1e-12)} points with exact min|E| > 1e-12 "
        f"(worst: eig {minE[wrong].min():.0e} vs exact {exact[wrong][np.argmin(minE[wrong])]:.0e})")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    im = a1.pcolormesh(t1s, gs, np.log10(np.maximum(minE, 1e-14)), cmap="Greys_r", vmin=-14, vmax=0,
                       shading="auto", rasterized=True)
    cb = fig.colorbar(im, ax=a1, pad=0.02)
    cb.set_label(rf"$\log_{{10}}\min|E|$ (eig, $N={N - 1}$)")
    T, Gm = np.meshgrid(t1s, gs)
    a1.plot(T[wrong], Gm[wrong], "x", color="#f2c200", ms=2.5, mew=0.7)
    im2 = a2.pcolormesh(t1s, gs, P, cmap="Blues", vmin=0, vmax=1, shading="auto", rasterized=True)
    cb2 = fig.colorbar(im2, ax=a2, pad=0.02)
    cb2.set_label(rf"$P$ (odd chain, $N={N}$)")
    gg = np.linspace(0, 3, 400)
    for ax in (a1, a2):
        ax.plot(np.sqrt(T2 ** 2 + gg ** 2 / 4), gg, color=ORANGE, lw=1.6)
        sel = gg >= 2 * T2
        ax.plot(np.sqrt(gg[sel] ** 2 / 4 - T2 ** 2), gg[sel], color=ORANGE, lw=1.6)
        for line in [np.abs(T2 - gg / 2), gg / 2 + T2]:
            ax.plot(line, gg, color=GREEN, lw=1.1, ls="--")
        ax.plot(gg / 2, gg, color=PURPLE, lw=0.9, ls="-.")
        ax.set(xlabel=r"$t_1$", ylabel=r"$\gamma$", xlim=(0, 2), ylim=(0, 3))
        ax.minorticks_on()
    a1.text(0.12, 0.25, "edge\nstates", color="white", fontsize=10, transform=a1.transAxes)
    a1.text(0.80, 0.25, "no edge\nstates", color="k", fontsize=10, transform=a1.transAxes)
    a2.text(1.42, 0.35, "Bloch", color=GREEN, fontsize=9, rotation=55)
    a2.text(1.45, 1.55, "OBC", color=ORANGE, fontsize=9, rotation=48,
            bbox={"fc": "white", "ec": "none", "pad": 0.3, "alpha": 0.8})
    a2.text(1.05, 2.25, "EP", color=PURPLE, fontsize=9)
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig4_phase_diagram.png")

    for g in [0.5, 4 / 3, 2.0, 2.6]:
        i = np.argmin(np.abs(gs - g))
        row = P[i] - 0.5
        ok = ~np.isnan(row)
        tt, rr = t1s[ok], row[ok]
        j = np.where(np.diff(np.sign(rr)) != 0)[0]
        cross = tt[j] - rr[j] * (tt[j + 1] - tt[j]) / (rr[j + 1] - rr[j])
        log(f"[phase] g={gs[i]:.3f}: P(N={N}) = 1/2 at t1 = {np.round(cross, 3).tolist()}; "
            f"OBC prediction {np.round(obc_transitions(T2, gs[i]), 3).tolist()}")


def fig_densities(N=20, t1=1.0):
    wR, wL, rho = cell_weights_of_zero_mode(N, t1, T2, G0)
    rR, rL = r_factors(t1, T2, G0)
    n = np.arange(1, N + 1)
    x = np.conj(rL) * rR
    rho_exact = x ** n / np.sum(x ** n)
    log(f"[density] odd chain N={N}, t1={t1}: r_R={rR:.4f}, r_L={rL:.4f}, r_L*r_R={x:.4f}; "
        f"max|<Pi_n> - x^n/sum| = {np.max(np.abs(rho - rho_exact)):.1e}, max|Im<Pi_n>| = {np.abs(rho.imag).max():.1e}")
    (RA, LA), (RB, LB) = chiral_edge_states(N, t1, T2, G0)
    pA, pB = hermitian_edge_states(N, t1, T2, G0)
    dA = np.max(np.abs(np.conj(LA) * RA - pA ** 2))
    dB = np.max(np.abs(np.conj(LB) * RB - pB ** 2))
    log(f"[density] even chain N={N}, t1={t1}: max|L*R - psi_SSH^2| = {dA:.1e} (A edge), {dB:.1e} (B edge)")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE)
    a1.semilogy(n, wR, "o-", color=BLUE, ms=3.5, lw=0.9, mfc="white", label=r"$|\psi_R(n)|^2$")
    a1.semilogy(n, wL, "s-", color=ORANGE, ms=3.5, lw=0.9, mfc="white", label=r"$|\psi_L(n)|^2$")
    a1.semilogy(n, np.abs(rho), "D", color="k", ms=4, label=r"$\langle\Pi_n\rangle$, biorthogonal")
    a1.semilogy(n, np.abs(rho_exact), color="k", lw=1.0, label=r"$x^n/\sum_m x^m$, $x=r_L^*r_R$")
    a1.set(xlabel=r"cell $n$", ylabel="weight per cell", xlim=(0, N + 1), ylim=(1e-12, 3))
    a1.legend(loc="lower right", fontsize=8.5)

    cells = np.arange(N)
    for (R, L), psi, c, m, lab in [((RA, LA), pA, BLUE, "o", "A end"), ((RB, LB), pB, ORANGE, "s", "B end")]:
        bi = biorthogonal_density(R, L).real
        a2.semilogy(cells, bi, m, color=c, ms=4.5, mfc="none", mew=1.0, label=rf"$L^*R$, {lab}")
        a2.semilogy(cells, cell_weight(psi), color=c, lw=1.0)
        a2.semilogy(cells, cell_weight(R), ":", color=c, lw=1.1)
    a2.set(xlabel=r"cell $n$", ylabel="weight per cell", xlim=(-1, N), ylim=(1e-12, 3))
    a2.legend(loc="lower center", fontsize=8.5, ncol=2)
    a2.text(0.5, 0.92, r"lines: Hermitian SSH$(v',t_2)$; dotted: $|R|^2$ alone", transform=a2.transAxes,
            ha="center", fontsize=9)
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig5_densities.png")


def fig_polarization():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE)
    sizes = [11, 21, 41, 161, 1001]
    shades = ["#c6d9f0", "#93b9e6", "#5c96d9", "#2a78d6", "#123f7a"]
    for ax, g, t1s in [(a1, G0, np.linspace(0, 2, 401)), (a2, 3.0, np.linspace(0, 2.6, 521))]:
        for N, c in zip(sizes, shades):
            ax.plot(t1s, [polarization_exact(N, t, T2, g) for t in t1s], color=c, lw=1.2, label=rf"$N={N}$")
        tn = t1s[::16]
        ax.plot(tn, [polarization(21, t, T2, g) for t in tn], "o", color=ORANGE, ms=3.8, mfc="none",
                mew=1.0, label=r"eig, $N=21$")
        ax.set(xlabel=r"$t_1$", ylabel=r"biorthogonal polarisation $P$", ylim=(-0.06, 1.12),
               xlim=(t1s[0], t1s[-1]))
        mark_transitions(ax, g)
        ax.text(0.97, 0.5, r"$\gamma=4/3$" if g == G0 else rf"$\gamma={g:g}$", transform=ax.transAxes, ha="right")
    a1.legend(loc="center left", fontsize=8.5, bbox_to_anchor=(0.0, 0.45))
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig6_polarization.png")

    for g in [G0, 3.0]:
        tcs = obc_transitions(T2, g)
        tbs = bloch_gap_closings(T2, g)
        log(f"[P] g={g:.4f}: OBC transitions {np.round(tcs, 4).tolist()}, Bloch gap closings {np.round(tbs, 4).tolist()}")
        for N in [21, 41, 161, 1001]:
            at_c = [polarization_exact(N, t, T2, g) for t in tcs]
            log(f"    N={N}: P at OBC points = {np.round(at_c, 4).tolist()}; "
                f"P at Bloch points = {np.round([polarization_exact(N, t, T2, g) for t in tbs], 4).tolist()}")
        diffs = [abs(polarization(N, t, T2, g) - polarization_exact(N, t, T2, g))
                 for N in (11, 21, 31) for t in np.linspace(0.05, 2.0, 40)]
        log(f"    max |P_eig - P_closed form| for N=11,21,31: {max(diffs):.1e}")
    near = [polarization_exact(N, np.sqrt(3.0 ** 2 / 4 - T2 ** 2), T2, 3.0) for N in (20, 21)]
    log(f"[P] g=3 at x = r_L* r_R = -1: P(N=20) = {near[0]}, P(N=21) = {near[1]:.4f} "
        f"(even N: <L|R> = 0, P undefined)")


def mp_spot_check(N, t1, dps=60):
    import mpmath as mp

    mp.mp.dps = dps
    third = mp.mpf(2) / 3
    up, down = mp.mpf(t1) + third, mp.mpf(t1) - third
    H = mp.zeros(2 * N)
    for i in range(2 * N - 1):
        H[i, i + 1], H[i + 1, i] = (up, down) if i % 2 == 0 else (1, 1)
    E = np.array([complex(e) for e in mp.eig(H, left=False, right=False)])
    Hd = H_open(N, t1, T2, G0)
    Ex = exact_open_spectrum(N, t1, T2, G0)
    Ed = np.linalg.eigvals(Hd)
    dist = lambda A, B: np.max(np.min(np.abs(A[:, None] - B[None, :]), axis=1))
    return dist(E, Ex), dist(Ed, E), np.abs(E.imag).max()


def fig_accuracy(skip_mpmath):
    Ns = np.arange(4, 101, 2)
    cases = [(1.4, ORANGE, "v"), (1.0, BLUE, "o"), (0.7, GREEN, "s")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE)
    for t1, c, m in cases:
        err = np.array([spectrum_error(N, t1, T2, G0) for N in Ns])
        cond = np.array([eigenvector_condition(N, t1, T2, G0) for N in Ns])
        s = abs(skin_factor(t1, G0))
        a1.semilogy(Ns, np.maximum(err, 1e-16), m + "-", color=c, ms=3.2, lw=0.8, mfc="white", mew=0.9,
                    label=rf"$t_1={t1}$, $|s|={s:.2f}$")
        a2.semilogy(Ns, cond, m, color=c, ms=3.2, mfc="white", mew=0.9)
        a2.semilogy(Ns, s ** Ns, color=c, lw=1.0)
        bad = Ns[err > 1e-10]
        log(f"[accuracy] t1={t1}, |s|={s:.4f}: first N with eig error > 1e-10: {bad.min() if bad.size else None}; "
            + ", ".join(f"N={N}: {e:.1e}" for N, e in zip(Ns, err) if N in (30, 36, 38, 40, 60, 80, 100)))
        log(f"    cond(R): " + ", ".join(f"N={N}: {k:.1e} (|s|^N={s ** N:.1e})"
                                     for N, k in zip(Ns, cond) if N in (10, 20, 30, 40, 60)))
    for t1 in (1.0, 0.7):
        ims = {N: np.abs(np.linalg.eigvals(H_open(N, t1, T2, G0)).imag).max() for N in (40, 60, 70, 80, 100)}
        log(f"[accuracy] t1={t1}: max|Im E_eig| " + ", ".join(f"N={N}: {v:.0e}" for N, v in ims.items()))
    herm = np.array([spectrum_error(N, 1.0, T2, 0.0) for N in Ns])
    a1.semilogy(Ns, np.maximum(herm, 1e-16), "-", color=GREY, lw=1.0, label=r"$\gamma=0$ (Hermitian)")
    log(f"[accuracy] negative control g=0, t1=1: max eig error N=4..100 = {herm.max():.1e}")
    step = [(N, spectrum_error(N, 0.7, T2, G0)) for N in (37, 38)]
    log(f"[accuracy] t1=0.7: N=37 (74 sites) error {step[0][1]:.1e}; N=38 (76 sites) error {step[1][1]:.1e}")
    for ax in (a1,):
        ax.axvline(37.5, color="k", ls="--", lw=0.8)
        ax.text(36.5, 1e-9, "76 sites:\nLAPACK switches\nQR algorithm", fontsize=8.5, ha="right")
    if not skip_mpmath:
        t0 = time.time()
        d_exact, d_eig, im = mp_spot_check(40, 0.7)
        log(f"[mpmath] N=40, t1=0.7, 60 digits ({time.time() - t0:.0f} s): "
            f"|E_mp - E_exact| = {d_exact:.1e}, max|Im E_mp| = {im:.0e}, |E_eig - E_mp| = {d_eig:.1e}")
        a1.semilogy([40], [max(d_exact, 1e-16)], "*", color="k", ms=9, label="mpmath vs exact")
    a1.axhline(1e-10, color=GREY, lw=0.6, ls=":")
    a1.set(xlabel=r"$N$ (cells)", ylabel=r"$\max|E_{\mathrm{eig}} - E_{\mathrm{exact}}|$",
           xlim=(0, 102), ylim=(1e-16, 1))
    a1.legend(loc="upper left", fontsize=8.5)
    a2.set(xlabel=r"$N$ (cells)", ylabel=r"$\mathrm{cond}(R)$", xlim=(0, 102), ylim=(1, 1e60))
    a2.text(0.04, 0.92, r"symbols: eig;  lines: $|s|^N$", transform=a2.transAxes, fontsize=9)
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig7_accuracy.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-mpmath", action="store_true", help="skip the 60-digit check (~20 s)")
    args = parser.parse_args()
    FIG.mkdir(exist_ok=True)
    t0 = time.time()
    environment_header()
    fig_spectra()
    fig_skin()
    fig_exact()
    fig_phase()
    fig_densities()
    fig_polarization()
    fig_accuracy(args.skip_mpmath)
    print(f"done in {time.time() - t0:.0f} s")
    (FIG / "results.txt").write_text("\n".join(results) + "\n")


if __name__ == "__main__":
    main()
