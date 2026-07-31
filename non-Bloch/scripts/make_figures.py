"""Generate figures/fig1-fig7 and figures/results.txt.

Usage: python non-Bloch/scripts/make_figures.py [--skip-mpmath]
The mpmath references are cached in figures/mpmath_ref.npz (git-ignored).
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
for dep in ("nh-SSH", "SSH-model"):
    sys.path.insert(0, str(ROOT.parent / dep))

from nh_ssh import bloch_gap_closings, exact_open_spectrum, obc_transitions, skin_decay_factor
from ssh_model import winding_number
from non_bloch import (
    H_obc,
    bloch_winding,
    boundary_log_ratio,
    char_roots,
    circular_radius,
    distance_to_curve,
    edge_pair_count,
    gbz_points,
    gbz_radius_ssh,
    growth_factor,
    hatano_nelson,
    hatano_nelson_open_spectrum,
    nonbloch_winding,
    radial_range,
    range2,
    root_gap,
    ssh_h1,
    ssh_h2,
    ssh_lambda,
    third_root_separation,
    zeros,
)

FIG = ROOT / "figures"
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2"
GREY = "#9a9994"
SINGLE = (4.6, 3.3)
DOUBLE = (9.2, 3.5)
T2, G0 = 1.0, 4.0 / 3.0
TR, TL = 1.5, 0.5
R2_PARAMS = (0.3, 1.0, 0.5, 0.4)  # t_-2, t_-1, t_1, t_2
R2 = range2(*R2_PARAMS)

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


def circle(r, n=400):
    return r * np.exp(2j * np.pi * np.arange(n + 1) / n)


def max_dist(A, B):
    return np.max(np.min(np.abs(A[:, None] - B[None, :]), axis=1))


def environment_header():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(Exception):
        np.show_config()
    blas = next((ln.split(":", 1)[1].strip().strip('",') for ln in buf.getvalue().splitlines()
                 if "openblas configuration" in ln), "unknown")
    log(f"# python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}")
    log(f"# BLAS/LAPACK: {blas}")


def fig_hatano_nelson(N=30):
    ch = hatano_nelson(TR, TL)
    r = np.sqrt(TR / TL)
    for tR, tL in [(TR, TL), (0.4, 1.0), (2.0, 0.1)]:
        lo, hi = radial_range(gbz_points(hatano_nelson(tR, tL), 1000))
        rr = np.sqrt(tR / tL)
        log(f"[HN] t_R={tR}, t_L={tL}: theta-scan |beta| in [{lo:.15f}, {hi:.15f}], sqrt(t_R/t_L) = {rr:.15f}, "
            f"max rel. dev {max(abs(lo / rr - 1), abs(hi / rr - 1)):.0e}")
    E = np.linalg.eigvals(H_obc(N, ch))
    Ex = hatano_nelson_open_spectrum(N, TR, TL)
    log(f"[HN] N={N}: max|E_eig - 2 sqrt(t_R t_L) cos(pi m/(N+1))| = {max_dist(E, Ex):.0e}, "
        f"max root gap of exact E_m = {max(root_gap(ch, e) for e in Ex):.0e}")
    for Nn in (12, 20, 30):
        log(f"[HN] N={Nn}: per-site growth of summed eigenvector weight = {growth_factor(H_obc(Nn, ch)):.4f}, "
            f"r = {r:.4f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE)
    Ebz = ch(circle(1.0))
    Egbz = ch(circle(r))
    a1.plot(Ebz.real, Ebz.imag, color=GREY, lw=1.2, label=r"$E(|\beta|=1)$, periodic")
    a1.plot(Egbz.real, Egbz.imag, color=ORANGE, lw=1.4, label=r"$E(|\beta|=r)$, $r=\sqrt{3}$")
    a1.plot(E.real, E.imag, "o", color=BLUE, ms=3.5, mfc="none", mew=0.9, label=f"open, $N={N}$")
    a1.set(xlabel=r"$\mathrm{Re}\,E$", ylabel=r"$\mathrm{Im}\,E$", xlim=(-2.3, 2.3), ylim=(-1.45, 1.45))
    a1.legend(loc="upper center", fontsize=8.5, ncol=2, columnspacing=0.9, bbox_to_anchor=(0.5, 1.02))
    a1.text(0.03, 0.05, rf"$t_R={TR},\ t_L={TL}$", transform=a1.transAxes)

    Nv = 20
    _, V = np.linalg.eig(H_obc(Nv, ch))
    n = np.arange(Nv)
    for j in range(Nv):
        a2.semilogy(n, np.abs(V[:, j]) ** 2 / np.sum(np.abs(V[:, j]) ** 2), color=GREY, lw=0.5, alpha=0.7)
    a2.semilogy(n, 0.5 * r ** (2.0 * (n - Nv + 1)), "k--", lw=1.0)
    a2.text(0.96, 0.06, r"dashed: $r^{2n}$, $r=\sqrt{t_R/t_L}$", transform=a2.transAxes, ha="right", fontsize=9.5)
    a2.set_xticks(range(0, Nv, 4))
    a2.text(0.04, 0.9, rf"all {Nv} right eigenvectors, open chain", transform=a2.transAxes, fontsize=9.5)
    a2.set(xlabel=r"site $n$", ylabel=r"$|\psi_n|^2$", xlim=(-0.5, Nv - 0.5), ylim=(1e-11, 3))
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig1_hatano_nelson.png")


def continuous_sqrt(lam, r, sign, n=1600):
    """Both branches of E = sqrt(lambda) along |beta| = r, without the jump at the branch cut."""
    l = lam(r * np.exp(1j * np.linspace(0, 4 * np.pi, n)))
    E = sign * np.sqrt(np.abs(l)) * np.exp(0.5j * np.unwrap(np.angle(l)))
    return E.real, E.imag


def fig_ssh_gbz(N=16):
    t1 = 0.7
    lam = ssh_lambda(t1, T2, G0)
    r = gbz_radius_ssh(t1, T2, G0)
    for t in (0.3, 0.7, 1.0, 1.4, 2.0):
        lo, hi = radial_range(gbz_points(ssh_lambda(t, T2, G0), 1000))
        rr = gbz_radius_ssh(t, T2, G0)
        log(f"[SSH] t1={t}: theta-scan |beta| in [{lo:.12f}, {hi:.12f}], sqrt|(t1-g/2)/(t1+g/2)| = {rr:.12f}")
    for t in (0.7, 1.0, 1.4):
        Eo = exact_open_spectrum(40, t, T2, G0)
        bulk = Eo[np.abs(Eo) > 0.1]
        mods = np.array([np.abs(char_roots(ssh_lambda(t, T2, G0), e * e)) for e in bulk])
        log(f"[SSH] t1={t}, N=40: E-scan over {len(bulk)} bulk OBC eigenvalues, max | |beta_i|/r - 1 | = "
            f"{np.max(np.abs(mods / gbz_radius_ssh(t, T2, G0) - 1)):.0e}")
    log(f"[SSH] t1=0.7: decay factor per cell of bulk right eigenvectors (project 3, N=12/20) = "
        f"{skin_decay_factor(12, 0.7, T2, G0):.4f}/{skin_decay_factor(20, 0.7, T2, G0):.4f}; r = {r:.4f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE)
    Eo = exact_open_spectrum(N, t1, T2, G0)
    for rr, c, lw, lab in [(1.0, GREY, 1.3, "periodic"), (r, ORANGE, 5.0, r"$E(\mathrm{GBZ})$")]:
        for s in (1, -1):
            a1.plot(*continuous_sqrt(lam, rr, s), color=c, lw=lw, label=lab if s > 0 else None)
    a1.plot(Eo.real, Eo.imag, "o", color=BLUE, ms=2.8, mfc="white", mew=0.8, label=f"open, $N={N}$")
    a1.set(xlabel=r"$\mathrm{Re}\,E$", ylabel=r"$\mathrm{Im}\,E$", xlim=(-1.8, 1.8), ylim=(-0.95, 0.95))
    a1.legend(loc="upper center", ncol=3, fontsize=8.5, columnspacing=0.9)
    a1.text(0.03, 0.05, rf"$t_1={t1},\ \gamma=4/3$", transform=a1.transAxes)

    t1s = np.linspace(0.02, 2.5, 300)
    a2.plot(t1s, [gbz_radius_ssh(t, T2, G0) for t in t1s], color="k", lw=1.2,
            label=r"$\sqrt{|t_1-\gamma/2|/|t_1+\gamma/2|}$")
    tsc = np.linspace(0.05, 2.45, 25)
    tsc = tsc[np.abs(tsc - G0 / 2) > 0.02]
    rs = np.array([radial_range(gbz_points(ssh_lambda(t, T2, G0), 200)) for t in tsc])
    a2.plot(tsc, rs[:, 1], "o", color=ORANGE, ms=4, mfc="none", mew=1.0, label=r"$\theta$-scan")
    tfit = np.array([0.8, 1.0, 1.2, 1.4, 1.7, 2.0, 2.4])
    a2.plot(tfit, [skin_decay_factor(20, t, T2, G0) for t in tfit], "^", color=BLUE, ms=4.5, mfc="none",
            mew=1.0, label="eigenvector decay, $N=20$")
    a2.axhline(1, color=GREY, ls=":", lw=1.0)
    a2.axvline(G0 / 2, color=PURPLE, ls="-.", lw=0.8)
    a2.text(G0 / 2 + 0.04, 0.85, r"EP, $t_1=\gamma/2$", color=PURPLE, fontsize=9)
    a2.text(2.45, 1.03, "Brillouin zone", ha="right", fontsize=9, color="#6b6a66")
    a2.set(xlabel=r"$t_1$", ylabel=r"GBZ radius $r$", xlim=(0, 2.5), ylim=(0, 1.15))
    a2.legend(loc="lower right", fontsize=8.5)
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig2_ssh_gbz.png")


def winding_row(args):
    t1, g = args
    return bloch_winding(t1, T2, g), nonbloch_winding(t1, T2, g), edge_pair_count(200, t1, T2, g)


def fig_winding():
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE)
    for ax, g in zip(axes, (G0, 3.0)):
        t1s = np.linspace(0.0, 2.5, 501)
        with ProcessPoolExecutor() as pool:
            rows = np.array(list(pool.map(winding_row, [(t, g) for t in t1s], chunksize=20)))
        wb, wnb, cnt = rows.T
        tc = obc_transitions(T2, g)
        near = np.min(np.abs(t1s[:, None] - np.append(tc, g / 2)[None, :]), axis=1)
        ok = ~np.isnan(wnb)
        bad = ok & (np.abs(cnt - wnb) > 1e-6)
        log(f"[winding] g={g:.4f}: nu_nonBloch values {np.unique(np.round(wnb[ok], 8)).tolist()}, "
            f"nu_Bloch values {np.unique(np.round(wb[~np.isnan(wb)], 8) + 0.0).tolist()}; edge count (N=200, |E|<1e-8) "
            f"!= nu_nonBloch at {bad.sum()} of {ok.sum()} points, all within {near[bad].max() if bad.any() else 0:.3f} "
            f"of a transition; OBC transitions {np.round(tc, 4).tolist()}")
        log(f"    Bloch winding != nu_nonBloch at {np.sum(ok & (np.abs(wb - wnb) > 1e-6))} points")
        for t0 in sorted(set(np.round(list(bloch_gap_closings(T2, g)), 6))):
            ax.axvline(t0, color=GREY, ls=":", lw=1.1)
        for t0 in tc:
            ax.axvline(t0, color="k", ls="--", lw=0.9)
        ax.plot(t1s, wb, color=GREY, lw=2.2, label="Bloch")
        ax.plot(t1s, wnb, color=ORANGE, lw=1.4, label="non-Bloch")
        sel = slice(0, None, 10)
        ax.plot(t1s[sel], cnt[sel], "o", color=BLUE, ms=4, mfc="none", mew=1.0,
                label="open, $N=200$")
        ax.set(xlabel=r"$t_1$", ylabel=r"winding $\nu$", xlim=(0, 2.5), ylim=(-0.15, 1.3))
        ax.text(0.97, 0.88, r"$\gamma=4/3$" if g == G0 else rf"$\gamma={g:g}$", transform=ax.transAxes, ha="right")
    axes[1].legend(loc="upper left", fontsize=8.5)
    panel(axes[0], "a")
    panel(axes[1], "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig3_winding.png")

    log("[winding] spot table (t1, g): nu_Bloch, nu_nonBloch, edge pairs N=200, min|E| N=200")
    for t1, g in [(0.2, G0), (0.7, G0), (1.0, G0), (1.3, G0), (2.0, G0), (0.9, 3.0), (1.4, 3.0), (2.2, 3.0)]:
        Eo = exact_open_spectrum(200, t1, T2, g)
        log(f"    ({t1}, {g:.4f}): {bloch_winding(t1, T2, g):+.6f}, {nonbloch_winding(t1, T2, g):+.6f}, "
            f"{edge_pair_count(200, t1, T2, g)}, {np.min(np.abs(Eo)):.1e}")
    for t1 in (0.5, 1.5):
        log(f"[winding] g=0, t1={t1}: nu_nonBloch = {nonbloch_winding(t1, T2, 0.0):+.6f}, project-1 winding = "
            f"{winding_number(t1, T2):+.6f}, GBZ radius = {gbz_radius_ssh(t1, T2, 0.0)}")
    t1 = 1.0
    z1, z2 = zeros(ssh_h1(t1, T2, G0)), zeros(ssh_h2(t1, T2, G0))
    log(f"[winding] t1=1, g=4/3: zero of h1 at |beta| = {abs(z1[0]):.4f}, zero of h2 at {abs(z2[0]):.4f}, "
        f"GBZ r = {gbz_radius_ssh(t1, T2, G0):.4f}, BZ r = 1")


def fig_range2(N=80):
    pts = gbz_points(R2, 6000)
    lo, hi = radial_range(pts)
    rho = np.sqrt(lo * hi)
    log(f"[range2] t_-2, t_-1, t_1, t_2 = {R2_PARAMS}: GBZ |beta| in [{lo:.4f}, {hi:.4f}], max/min = {hi / lo:.4f}; "
        f"{len(pts)} points from 6000 theta values; gauge rho = sqrt(min max) = {rho:.4f}")
    cusp = pts[np.argmax(np.abs(pts))]
    log(f"[range2] cusp at beta = {cusp:.4f} (largest |beta|): E = {R2(cusp).real:.4f}, root moduli there "
        f"{np.round(np.abs(char_roots(R2, R2(cusp))), 4).tolist()} (beta_4 within 0.1%: the cusp sits at the junction of the arcs)")
    E = np.linalg.eigvals(H_obc(N, R2, rho))
    curve = R2(pts)
    log(f"[range2] N={N}: max root gap {max(root_gap(R2, e) for e in E):.2e}, max distance to E(GBZ) "
        f"{np.max(distance_to_curve(E, curve)):.1e}, to E(unit circle) {np.max(distance_to_curve(E, R2(circle(1.0, 4000)))):.2f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE, gridspec_kw={"width_ratios": [1, 1.25]})
    for rr, ls in [(lo, "--"), (hi, "--")]:
        c = circle(rr)
        a1.plot(c.real, c.imag, color=GREY, lw=0.7, ls=ls)
    c = circle(1.0)
    a1.plot(c.real, c.imag, color=GREY, lw=1.1, ls=":")
    closed = np.append(pts, pts[0])
    a1.plot(closed.real, closed.imag, color=ORANGE, lw=1.6)
    a1.set_aspect("equal")
    a1.set(xlabel=r"$\mathrm{Re}\,\beta$", ylabel=r"$\mathrm{Im}\,\beta$", xlim=(-1.55, 1.55), ylim=(-1.45, 1.45))
    a1.text(0.0, 0.0, "GBZ", color=ORANGE, ha="center", va="center", fontsize=10)
    a1.text(0.97, 0.03, rf"$|\beta|\in[{lo:.3f},\,{hi:.3f}]$", transform=a1.transAxes, ha="right", fontsize=9)
    a1.text(0.62, 0.62, "BZ", color="#6b6a66", fontsize=9)

    Ebz = R2(circle(1.0, 2000))
    a2.plot(Ebz.real, Ebz.imag, color=GREY, lw=1.1, label="periodic, $E(|\\beta|=1)$")
    a2.plot(curve.real, curve.imag, color=ORANGE, lw=1.6, label=r"$E(\mathrm{GBZ})$")
    a2.plot(E.real, E.imag, "o", color=BLUE, ms=3.2, mfc="none", mew=0.8, label=f"open, $N={N}$")
    a2.set(xlabel=r"$\mathrm{Re}\,E$", ylabel=r"$\mathrm{Im}\,E$", xlim=(-1.3, 2.35), ylim=(-0.7, 0.7))
    a2.legend(loc="upper right", fontsize=8.5)
    a2.text(0.03, 0.05, r"$(t_{-2},t_{-1},t_1,t_2)=(0.3,1,0.5,0.4)$", transform=a2.transAxes, fontsize=9)
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig4_range2.png")
    return pts, rho


def finite_size_row(args):
    N, rho = args
    E = np.linalg.eigvals(H_obc(N, R2, rho))
    gap = np.array([root_gap(R2, e) for e in E])
    C = np.array([boundary_log_ratio(R2, e) for e in E])
    sep = np.array([third_root_separation(R2, e) for e in E]) * N
    return E, gap, C, sep


def fig_finite_size(pts, rho):
    Ns = np.array([10, 14, 20, 28, 40, 56, 80, 113, 160])
    curve = R2(pts)
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(finite_size_row, [(N, rho) for N in Ns]))
    gmax_far, gmax_all, dist_far, dist_all, Cmax = [], [], [], [], []
    for N, (E, gap, C, sep) in zip(Ns, rows):
        far = sep > 3
        gmax_far.append(gap[far].max())
        gmax_all.append(gap.max())
        dist_far.append(np.max(distance_to_curve(E[far], curve)))
        dist_all.append(np.max(distance_to_curve(E, curve)))
        Cmax.append(np.abs(C[far]).max())
        err = np.abs((N + 1) * gap - C)
        log(f"[finite size] N={N}: {far.sum()}/{N} eigenvalues with N*sep > 3; max gap there {gap[far].max():.3e} "
            f"(ln|C|/(N+1) max {np.abs(C[far]).max() / (N + 1):.3e}); median |(N+1) gap - ln|C|| = {np.median(err[far]):.0e}; "
            f"max E-distance {dist_far[-1]:.1e}; over all eigenvalues: max gap {gap.max():.2e}, "
            f"max E-distance {dist_all[-1]:.1e}")
    big = Ns >= 40
    fit = lambda y: np.polyfit(np.log(Ns[big]), np.log(np.array(y)[big]), 1)[0]
    log(f"[finite size] fitted exponents for N >= 40: max gap (N*sep>3) {fit(gmax_far):.3f}, "
        f"max E-distance (N*sep>3) {fit(dist_far):.3f}, max gap over all eigenvalues {fit(gmax_all):.3f}, "
        f"max E-distance over all {fit(dist_all):.3f}; prediction -1")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE)
    a1.loglog(Ns, gmax_far, "o", color=BLUE, mfc="none", mew=1.0, label=r"max root gap, $Nd>3$")
    a1.loglog(Ns, np.array(Cmax) / (Ns + 1), color=BLUE, lw=1.0, label=r"$\max\,\ln|C(E)|/(N+1)$")
    a1.loglog(Ns, dist_far, "s", color=GREEN, mfc="none", mew=1.0, label=r"max distance to $E(\mathrm{GBZ})$")
    a1.loglog(Ns, gmax_all, "x", color=GREY, mew=1.0, label="max root gap, all eigenvalues")
    a1.set(xlabel=r"$N$ (sites)", ylabel=r"$\ln|\beta_3/\beta_2|$, $|\Delta E|$", xlim=(8, 200))
    a1.legend(loc="lower left", fontsize=8.5)

    for N, (E, gap, C, sep), c, m in zip(Ns[[2, 4, 6]], [rows[i] for i in (2, 4, 6)], [BLUE, GREEN, ORANGE], "os^"):
        err = np.abs((N + 1) * gap - C)
        a2.semilogy(sep, np.maximum(err, 1e-16), m, color=c, ms=3.5, mfc="none", mew=0.9, label=f"$N={N}$")
    x = np.linspace(0, 60, 200)
    a2.semilogy(x, 2 * np.exp(-x), "k--", lw=1.0, label=r"$2e^{-Nd}$")
    a2.set(xlabel=r"$N d$,  $d=\min(\ln|\beta_2/\beta_1|,\ln|\beta_4/\beta_3|)$",
           ylabel=r"$|(N+1)\ln|\beta_3/\beta_2| - \ln|C(E)||$", xlim=(0, 60), ylim=(1e-16, 10))
    a2.legend(loc="upper right", fontsize=8.5)
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig5_finite_size.png")


def deform_row(t2):
    ch = range2(R2_PARAMS[0], R2_PARAMS[1], R2_PARAMS[2], t2)
    lo, hi = radial_range(gbz_points(ch, 1500))
    return lo, hi


def fig_deformation():
    t_m2, t_m1, t_1, _ = R2_PARAMS
    t2_circ = t_m2 * (t_1 / t_m1) ** 2
    t2s = np.unique(np.concatenate([np.linspace(0.0, 1.0, 51), [t2_circ]]))
    with ProcessPoolExecutor() as pool:
        rr = np.array(list(pool.map(deform_row, t2s)))
    ratio = rr[:, 1] / rr[:, 0]
    i = np.argmin(ratio)
    log(f"[deform] t_2 scan: min of max/min |beta| = {ratio[i]:.12f} at t_2 = {t2s[i]:.4f}; predicted circle at "
        f"t_2 = t_-2 (t_1/t_-1)^2 = {t2_circ}, radius {circular_radius(range2(t_m2, t_m1, t_1, t2_circ)):.6f} "
        f"(GBZ there: [{rr[i, 0]:.10f}, {rr[i, 1]:.10f}])")
    for t in (0.0, 0.2, 0.4, 0.8, 1.0):
        j = np.argmin(np.abs(t2s - t))
        log(f"    t_2={t2s[j]:.2f}: |beta| in [{rr[j, 0]:.4f}, {rr[j, 1]:.4f}], ratio {ratio[j]:.4f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE, gridspec_kw={"width_ratios": [1, 1.25]})
    shades = ["#c6d9f0", "#93b9e6", "#5c96d9", "#2a78d6", "#123f7a"]
    for t, c in zip([0.0, t2_circ, 0.2, 0.4, 0.8], [GREEN] + shades[1:]):
        pts = gbz_points(range2(t_m2, t_m1, t_1, t), 3000)
        closed = np.append(pts, pts[0])
        lab = rf"$t_2={t:.3g}$" + (" (circle)" if t == t2_circ else "")
        a1.plot(closed.real, closed.imag, color=ORANGE if t == t2_circ else c, lw=1.3, label=lab)
    c = circle(1.0)
    a1.plot(c.real, c.imag, color=GREY, lw=1.0, ls=":")
    a1.set_aspect("equal")
    a1.set(xlabel=r"$\mathrm{Re}\,\beta$", ylabel=r"$\mathrm{Im}\,\beta$", xlim=(-1.9, 3.4), ylim=(-2.0, 2.0))
    a1.legend(loc="center right", fontsize=8, handlelength=1.2, borderaxespad=0.2)

    a2.plot(t2s, ratio, "o", color=BLUE, ms=3.5, mfc="none", mew=0.9, label=r"$\theta$-scan")
    a2.axvline(t2_circ, color=ORANGE, ls="--", lw=1.0)
    a2.text(t2_circ + 0.02, 1.35, r"$t_2=t_{-2}(t_1/t_{-1})^2$", color=ORANGE, fontsize=9.5)
    a2.axhline(1, color=GREY, lw=0.8, ls=":")
    a2.set(xlabel=r"$t_2$", ylabel=r"$\max|\beta| / \min|\beta|$ on the GBZ", xlim=(-0.02, 1.02))
    a2.text(0.97, 0.92, r"$t_{-2}=0.3,\ t_{-1}=1,\ t_1=0.5$", transform=a2.transAxes, ha="right", fontsize=9)
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig6_deformation.png")


def mp_eigs(args):
    N, dps = args
    import mpmath as mp

    mp.mp.dps = dps
    H = mp.zeros(N)
    for k, c in zip(R2.powers, R2.coeffs.real):
        for n in range(max(0, -k), min(N, N - k)):
            H[n, n + k] = mp.mpf(c)
    return np.array([complex(e) for e in mp.eig(H, left=False, right=False)])


def fig_accuracy(rho, skip_mpmath):
    Ns = np.arange(10, 201, 5)
    plain = np.array([max_dist(np.linalg.eigvals(H_obc(N, R2)), np.linalg.eigvals(H_obc(N, R2, rho))) for N in Ns])
    gauge2 = np.array([max_dist(np.linalg.eigvals(H_obc(N, R2, rho * 1.03)), np.linalg.eigvals(H_obc(N, R2, rho)))
                       for N in Ns])
    for N, a, b in zip(Ns, plain, gauge2):
        if N in (40, 60, 75, 80, 100, 120, 160, 200):
            log(f"[accuracy] N={N}: |E_plain - E_gauged| = {a:.1e}, |E_gauged(1.03 rho) - E_gauged(rho)| = {b:.1e}")
    log(f"[accuracy] first N (step 5) with |E_plain - E_gauged| > 1e-10: {Ns[plain > 1e-10].min()}; "
        f"with gauge-gauge > 1e-10: {Ns[gauge2 > 1e-10].min()}")

    cache = FIG / "mpmath_ref.npz"
    mpN = [40, 60, 80, 100]
    ref = {}
    if not skip_mpmath:
        if cache.exists():
            ref = {int(k[1:]): v for k, v in np.load(cache).items()}
        todo = [N for N in mpN if N not in ref]
        if todo:
            t0 = time.time()
            with ProcessPoolExecutor() as pool:
                for N, E in zip(todo, pool.map(mp_eigs, [(N, 40) for N in todo])):
                    ref[N] = E
            print(f"[mpmath] 40-digit eig for N={todo}: {time.time() - t0:.0f} s (cached to figures/mpmath_ref.npz)")
            np.savez(cache, **{f"N{N}": ref[N] for N in ref})
        for N in mpN:
            ep = max_dist(np.linalg.eigvals(H_obc(N, R2)), ref[N])
            eg = max_dist(np.linalg.eigvals(H_obc(N, R2, rho)), ref[N])
            log(f"[mpmath] N={N}, 40 digits: plain eig error {ep:.1e}, gauged eig error {eg:.1e}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE)
    a1.semilogy(Ns, np.maximum(plain, 1e-16), "o-", color=GREY, ms=3.2, lw=0.8, mfc="white", mew=0.9,
                label=r"plain $-$ gauged")
    a1.semilogy(Ns, np.maximum(gauge2, 1e-16), "s-", color=BLUE, ms=3.2, lw=0.8, mfc="white", mew=0.9,
                label=r"gauge $1.03\rho$ $-$ gauge $\rho$")
    if ref:
        a1.semilogy(mpN, [max_dist(np.linalg.eigvals(H_obc(N, R2)), ref[N]) for N in mpN], "*", color="k", ms=8,
                    label="plain vs mpmath")
        a1.semilogy(mpN, [max(max_dist(np.linalg.eigvals(H_obc(N, R2, rho)), ref[N]), 1e-16) for N in mpN], "*",
                    color=ORANGE, ms=8, label="gauged vs mpmath")
    a1.axhline(1e-10, color=GREY, lw=0.6, ls=":")
    a1.set(xlabel=r"$N$ (sites)", ylabel=r"$\max|\Delta E|$", xlim=(0, 205), ylim=(1e-16, 1))
    a1.legend(loc="upper left", fontsize=8.5)

    N = 160
    Ep = np.linalg.eigvals(H_obc(N, R2))
    Eg = np.linalg.eigvals(H_obc(N, R2, rho))
    curve = R2(gbz_points(R2, 3000))
    a2.plot(curve.real, curve.imag, color=ORANGE, lw=1.2, label=r"$E(\mathrm{GBZ})$")
    a2.plot(Ep.real, Ep.imag, "x", color=GREY, ms=3.5, mew=0.8, label=f"plain eig, $N={N}$")
    a2.plot(Eg.real, Eg.imag, "o", color=BLUE, ms=2.8, mfc="none", mew=0.7, label=rf"gauged, $\rho={rho:.3f}$")
    a2.set(xlabel=r"$\mathrm{Re}\,E$", ylabel=r"$\mathrm{Im}\,E$", xlim=(-1.3, 2.35), ylim=(-0.7, 0.7))
    a2.legend(loc="upper right", fontsize=8.5)
    log(f"[accuracy] N={N}: plain eig max distance to E(GBZ) {np.max(distance_to_curve(Ep, curve)):.1e}, "
        f"gauged {np.max(distance_to_curve(Eg, curve)):.1e}")
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig7_accuracy.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-mpmath", action="store_true", help="skip the 40-digit references (~1 min uncached)")
    args = parser.parse_args()
    FIG.mkdir(exist_ok=True)
    t0 = time.time()
    environment_header()
    fig_hatano_nelson()
    fig_ssh_gbz()
    fig_winding()
    pts, rho = fig_range2()
    fig_finite_size(pts, rho)
    fig_deformation()
    fig_accuracy(rho, args.skip_mpmath)
    print(f"done in {time.time() - t0:.0f} s")
    (FIG / "results.txt").write_text("\n".join(results) + "\n")


if __name__ == "__main__":
    main()
