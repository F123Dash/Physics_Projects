"""Generate figures/fig1-fig8 and figures/results.txt.

Usage: python Fourier-optics/scripts/make_figures.py
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import contextlib
import io
import pathlib
import platform
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.special

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fourier_optics import (
    azimuthal_angle,
    beam_width,
    binary_grating,
    binary_grating_series,
    cosine_grating,
    critical_distance,
    curvature,
    fidelity,
    fit_curvature,
    gaussian,
    gaussian_on_axis_exact,
    gouy,
    gouy_on_axis_nonparaxial,
    grid,
    harmonic_shift,
    laguerre_moment,
    lg,
    lg_nonparaxial_infidelity,
    lg_nonparaxial_phase,
    oam_per_photon,
    overlap,
    parabolic_peak,
    power,
    propagate,
    rayleigh_range,
    second_moment_width,
    talbot_defect_asymptote,
    talbot_exact,
    talbot_fidelity_series,
    talbot_paraxial,
    transfer_function,
    width_nonparaxial,
    winding_number,
    wrap,
)

FIG = ROOT / "figures"
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2"
GREY = "#9a9994"
SINGLE = (4.6, 3.3)
DOUBLE = (9.2, 3.5)

LAM, W0 = 0.5, 10.0  # um
ZR = rayleigh_range(W0, LAM)
K = 2 * np.pi / LAM
EPS = 1 / (K * W0) ** 2
MODES = [(0, 1), (0, 3), (1, 1), (2, -2)]
MODE_COLOURS = [BLUE, ORANGE, GREEN, PURPLE]

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


def environment_header():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(Exception):
        np.show_config()
    blas = next((ln.split(":", 1)[1].strip().strip('",') for ln in buf.getvalue().splitlines()
                 if "openblas configuration" in ln), "unknown")
    log(f"# python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}")
    log(f"# BLAS/LAPACK: {blas}")
    log(f"# lambda = {LAM} um, w0 = {W0} um, zR = {ZR:.4f} um, eps = 1/(k w0)^2 = {EPS:.4e}")


def mesh(N, dx):
    x = grid(N, dx)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return x, X, Y


# ---------------------------------------------------------------- propagator


def check_propagator():
    rng = np.random.default_rng(1)
    dx, N = 0.25, 128
    f = np.fft.fftfreq(N, dx)
    f2 = f[:, None] ** 2 + f[None, :] ** 2
    E0 = np.fft.ifft2((rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) * (f2 < (0.9 / LAM) ** 2))
    worst = max(abs(power(propagate(E0, dx, LAM, z, bandlimit=False), dx) / power(E0, dx) - 1)
                for z in (10.0, 1e3, 1e5))
    log(f"[prop] random field with |f| < 0.9/lambda, z = 10, 1e3, 1e5 um: max |P(z)/P(0) - 1| = {worst:.0e} (bound 1e-10)")
    x = grid(64, 0.2)
    X, Y = np.meshgrid(x, x, indexing="ij")
    fx = 24 / (64 * 0.2)
    E = propagate(np.exp(2j * np.pi * fx * X), 0.2, LAM, 123.4, bandlimit=False)
    kz = 2 * np.pi * np.sqrt(1 / LAM**2 - fx**2)
    log(f"[prop] plane wave at {np.degrees(np.arcsin(fx * LAM)):.1f} deg: max |E - E0 exp(i kz z)| = "
        f"{np.max(np.abs(E - np.exp(2j * np.pi * fx * X) * np.exp(1j * kz * 123.4))):.0e}")


# ---------------------------------------------------------------- Gaussian


def fig_gaussian():
    x, X, Y = mesh(512, 0.5)
    E0 = gaussian(X, Y, W0, LAM)
    zeta = np.linspace(-3, 3, 25)
    w, inv_R, g, exact = [], [], [], []
    for s in zeta:
        z = s * ZR
        E = propagate(E0, 0.5, LAM, z)
        w.append(second_moment_width(E, x))
        inv_R.append(fit_curvature(E[:, 256], x, K, beam_width(z, W0, LAM)))
        g.append(-wrap(np.angle(E[256, 256]) - K * z))
        exact.append(abs(E[256, 256] / E0[256, 256] - gaussian_on_axis_exact(z, W0, LAM)))
    w, inv_R, g = map(np.array, (w, inv_R, g))
    pos = zeta >= 0
    w_err = np.abs(w / beam_width(zeta * ZR, W0, LAM) - 1)
    g_err = np.abs(g - gouy(zeta * ZR, W0, LAM))
    nz = zeta != 0
    R_err = np.abs(inv_R[nz] / curvature(zeta[nz] * ZR, W0, LAM) - 1)
    log(f"[gauss] N = 512, dx = 0.5 um, window 256 um; z in [-3, 3] zR, 25 points")
    log(f"[gauss] max |w / w0 sqrt(1 + zeta^2) - 1| on 0..3 zR = {w_err[pos].max():.2e}")
    log(f"[gauss] max |w / w_nonparaxial - 1| = {np.max(np.abs(w / width_nonparaxial(zeta * ZR, W0, LAM) - 1)):.0e}")
    log(f"[gauss] max |1/R_fit / (z/(z^2 + zR^2)) - 1| = {R_err.max():.2e}")
    log(f"[gauss] max |(kz - arg E(0,z)) - arctan(z/zR)| = {g_err.max():.2e} rad")
    log(f"[gauss] max |(kz - arg E(0,z)) - Gouy_nonparaxial| = "
        f"{np.max(np.abs(g - gouy_on_axis_nonparaxial(zeta * ZR, W0, LAM))):.0e} rad")
    log(f"[gauss] max |E(0,z)/E(0,0) - Hankel integral| = {max(exact):.0e}")
    log(f"[gauss] Gouy phase from z = -3 zR to +3 zR: {g[-1] - g[0]:.6f} rad = {(g[-1] - g[0]) / np.pi:.6f} pi "
        f"(2 arctan 3 = {2 * np.arctan(3) / np.pi:.6f} pi; pi as zeta -> infinity)")

    # LG Gouy, via overlaps with the analytic mode without its Gouy factor
    x2, X2, Y2 = mesh(1024, 0.5)
    lg_g = {}
    for p, l in MODES:
        E0l = lg(X2, Y2, p, l, W0, LAM)
        vals = []
        for s in zeta:  # steps of 0.25 zR keep N * delta(Gouy) < pi, so np.unwrap is unambiguous
            z = s * ZR
            ph = np.angle(overlap(lg(X2, Y2, p, l, W0, LAM, z, with_gouy=False), propagate(E0l, 0.5, LAM, z), 0.5))
            vals.append(ph)
        vals = -np.unwrap(np.array(vals))
        vals -= vals[len(vals) // 2]  # zeta = 0 sample, where the phase is 0
        lg_g[(p, l)] = vals
        dev = np.max(np.abs(vals - (2 * p + abs(l) + 1) * gouy(zeta * ZR, W0, LAM)))
        log(f"[LG] (p, l) = ({p}, {l:+d}): max |Gouy - (2p+|l|+1) arctan(z/zR)| on [-3, 3] zR = {dev:.2e} rad")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE)
    zz = np.linspace(-3, 3, 400)
    a1.plot(zz, beam_width(zz * ZR, W0, LAM) / W0, color=GREY, lw=1.2)
    a1.plot(zeta, w / W0, "o", color=BLUE, mfc="none", label=r"$w$ (second moment)")
    a1.plot(zz, -beam_width(zz * ZR, W0, LAM) / W0, color=GREY, lw=1.2)
    a1.fill_between(zz, -beam_width(zz * ZR, W0, LAM) / W0, beam_width(zz * ZR, W0, LAM) / W0, color=GREY, alpha=0.12, lw=0)
    a1.set_xlabel(r"$z/z_R$")
    a1.set_ylabel(r"$\pm w(z)/w_0$")
    a1b = a1.twinx()
    a1b.plot(zz, curvature(zz * ZR, W0, LAM) * ZR, color=ORANGE, lw=1.2, ls="--")
    a1b.plot(zeta, inv_R * ZR, "s", color=ORANGE, mfc="none", ms=3.5)
    a1b.set_ylabel(r"$z_R/R(z)$", color=ORANGE)
    a1b.tick_params(axis="y", colors=ORANGE, which="both")
    a1.tick_params(axis="y", which="both", right=False)
    a1.text(0.30, 0.92, r"$w$: circles", transform=a1.transAxes, color=BLUE)
    a1.text(0.30, 0.84, r"$1/R$: squares", transform=a1.transAxes, color=ORANGE)
    a1.set_xlim(-3.2, 3.2)

    for (p, l), c in zip(MODES, MODE_COLOURS):
        Nmode = 2 * p + abs(l) + 1
        a2.plot(zz, Nmode * gouy(zz * ZR, W0, LAM) / np.pi, color=c, lw=1.0)
        a2.plot(zeta[::2], lg_g[(p, l)][::2] / np.pi, "o", color=c, mfc="none", ms=4,
                label=rf"LG$_{{{p}}}^{{{l:+d}}}$, $N={Nmode}$")
    a2.plot(zz, gouy(zz * ZR, W0, LAM) / np.pi, color="k", lw=1.0)
    a2.plot(zeta, g / np.pi, "o", color="k", ms=3.5, label=r"Gaussian, on axis")
    a2.set_xlabel(r"$z/z_R$")
    a2.set_ylabel(r"Gouy phase $/\pi$")
    a2.legend(loc="upper left", ncol=1)
    for a, lab in ((a1, "a"), (a2, "b")):
        panel(a, lab)
    fig.tight_layout()
    save(fig, "fig1_gaussian.png")


def fig_nonparaxial():
    """Residuals from the paraxial formulas, against the derived first-order terms."""
    x, X, Y = mesh(512, 0.5)
    E0 = gaussian(X, Y, W0, LAM)
    zeta = np.linspace(0.25, 3, 12)
    dw, dg = [], []
    for s in zeta:
        E = propagate(E0, 0.5, LAM, s * ZR)
        dw.append(second_moment_width(E, x) / beam_width(s * ZR, W0, LAM) - 1)
        dg.append(-wrap(np.angle(E[256, 256]) - K * s * ZR) - gouy(s * ZR, W0, LAM))
    dw, dg = np.array(dw), np.array(dg)

    x2, X2, Y2 = mesh(1024, 0.5)
    lg_dev = {}
    log(f"[LG] overlap phase beyond Gouy vs -zeta eps <s^2>/4, and 1 - F vs (zeta eps/4)^2 Var(s^2), at zeta = 1, 3:")
    for p, l in MODES:
        E0l = lg(X2, Y2, p, l, W0, LAM)
        out = []
        for s in (1.0, 2.0, 3.0):
            z = s * ZR
            ref = lg(X2, Y2, p, l, W0, LAM, z, with_gouy=False)
            E = propagate(E0l, 0.5, LAM, z)
            ph = wrap(np.angle(overlap(ref, E, 0.5)) + (2 * p + abs(l) + 1) * gouy(z, W0, LAM))
            out.append(ph)
            if s != 2.0:
                log(f"    ({p},{l:+d}) zeta={s:.0f}: phase {ph:.4e} (pred {lg_nonparaxial_phase(z, p, l, W0, LAM):.4e}); "
                    f"1-F {1 - fidelity(ref, E):.4e} (pred {lg_nonparaxial_infidelity(z, p, l, W0, LAM):.4e}); "
                    f"<s^2> = {laguerre_moment(p, l, 2):.0f}")
        lg_dev[(p, l)] = np.array(out)
    log(f"[gauss] width residual at 3 zR = {dw[-1]:.4e}; first-order prediction (9/10)(2 eps) = {0.9 * 2 * EPS:.4e}")
    log(f"[gauss] on-axis Gouy residual at 0.5 zR from the Hankel integral = "
        f"{gouy_on_axis_nonparaxial(0.5 * ZR, W0, LAM) - gouy(0.5 * ZR, W0, LAM):.4e} rad")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE)
    zz = np.linspace(0, 3.1, 300)
    a1.plot(zz, (width_nonparaxial(zz * ZR, W0, LAM) / beam_width(zz * ZR, W0, LAM) - 1) / EPS, color=BLUE, lw=1.2,
            label=r"$2\zeta^2/(1+\zeta^2)$")
    a1.plot(zeta, dw / EPS, "o", color=BLUE, mfc="none", label=r"$w$: numerics")
    a1.plot(zz, (gouy_on_axis_nonparaxial(zz * ZR, W0, LAM) - gouy(zz * ZR, W0, LAM)) / EPS, color=ORANGE, lw=1.2,
            label=r"$2\zeta(1-\zeta^2)/(1+\zeta^2)^2$")
    a1.plot(zeta, dg / EPS, "s", color=ORANGE, mfc="none", ms=4, label=r"Gouy: numerics")
    a1.axhline(0, color=GREY, lw=0.6)
    a1.set_xlabel(r"$\zeta = z/z_R$")
    a1.set_ylabel(r"(numerics $-$ paraxial)$\;/\;\varepsilon$")
    a1.legend(loc="lower left")
    a1.set_xlim(0, 3.1)

    s_pts = np.array([1.0, 2.0, 3.0])
    for (p, l), c in zip(MODES, MODE_COLOURS):
        m2 = laguerre_moment(p, l, 2)
        a2.plot(zz, -zz * m2 / 4, color=c, lw=1.0)
        a2.plot(s_pts, lg_dev[(p, l)] / EPS, "o", color=c, mfc="none", label=rf"LG$_{{{p}}}^{{{l:+d}}}$, $\langle s^2\rangle={m2:.0f}$")
    a2.set_xlabel(r"$\zeta = z/z_R$")
    a2.set_ylabel(r"overlap phase beyond Gouy $/\;\varepsilon$")
    a2.legend(loc="lower left")
    a2.set_xlim(0, 3.1)
    for a, lab in ((a1, "a"), (a2, "b")):
        panel(a, lab)
    fig.tight_layout()
    save(fig, "fig2_nonparaxial.png")


# ---------------------------------------------------------------- LG modes


def fig_lg():
    N, dx = 1024, 0.5
    x, X, Y = mesh(N, dx)
    z = ZR
    w = beam_width(z, W0, LAM)
    fields = {}
    log(f"[LG] N = 1024, dx = 0.5 um; winding = sum of wrapped phase steps / 2 pi on a square loop of half-width 0.5 w(z)")
    for p, l in MODES:
        E0 = lg(X, Y, p, l, W0, LAM)
        E = propagate(E0, dx, LAM, z)
        fields[(p, l)] = E
        wn = [winding_number(propagate(E0, dx, LAM, s * ZR), N // 2, N // 2, int(0.5 * beam_width(s * ZR, W0, LAM) / dx))
              for s in (0.0, 1.0, 3.0)]
        oam = [oam_per_photon(E0, dx, X, Y), oam_per_photon(propagate(E0, dx, LAM, 3 * ZR), dx, X, Y)]
        log(f"    ({p},{l:+d}): winding at z = 0, zR, 3zR = {wn[0]:+.12f}, {wn[1]:+.12f}, {wn[2]:+.12f}; "
            f"<Lz>/hbar at 0, 3zR = {oam[0]:+.12f}, {oam[1]:+.12f}; 1 - F(zR) = "
            f"{1 - fidelity(lg(X, Y, p, l, W0, LAM, z), E):.1e}")
    E = fields[(1, 1)]
    log(f"[LG] negative controls: loop off the core (centre +30 um) on LG_1^+1: "
        f"{winding_number(E, N // 2 + 60, N // 2, 10):+.1e}; Gaussian: "
        f"{winding_number(propagate(gaussian(X, Y, W0, LAM), dx, LAM, z), N // 2, N // 2, 20):+.1e}")
    mix = np.sqrt(0.7) * lg(X, Y, 0, 1, W0, LAM) + np.sqrt(0.3) * lg(X, Y, 0, -2, W0, LAM)
    log(f"[LG] <Lz>/hbar of sqrt(.7) LG_0^1 + sqrt(.3) LG_0^-2 = {oam_per_photon(mix, dx, X, Y):+.12f} (0.7 - 0.6 = 0.1)")

    half = 3.2 * w
    sel = np.abs(x) <= half
    ext = [-half / w, half / w, -half / w, half / w]
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.7))
    for col, ((p, l), c) in enumerate(zip(MODES, MODE_COLOURS)):
        E = fields[(p, l)][np.ix_(sel, sel)]
        I = np.abs(E) ** 2
        ai, ap = axes[0, col], axes[1, col]
        ai.imshow(I.T / I.max(), origin="lower", extent=ext, cmap="magma", vmin=0, vmax=1)
        phase = np.angle(E * np.exp(-1j * K * z)).T
        ap.imshow(np.where(I.T > 1e-3 * I.max(), phase, np.nan), origin="lower", extent=ext, cmap="twilight",
                  vmin=-np.pi, vmax=np.pi, interpolation="nearest")
        ai.set_title(rf"LG$_{{{p}}}^{{{l:+d}}}$", fontsize=11)
        for a in (ai, ap):
            a.set_xticks([-2, 0, 2])
            a.set_yticks([-2, 0, 2])
            a.tick_params(which="both", color="white", top=True, right=True)
            a.minorticks_off()
        hh = 0.5
        sq = np.array([[-hh, -hh], [hh, -hh], [hh, hh], [-hh, hh], [-hh, -hh]])
        ap.plot(sq[:, 0], sq[:, 1], color="k", lw=0.8)
        ap.text(0.04, 0.04, rf"$W={winding_number(fields[(p, l)], N // 2, N // 2, int(0.5 * w / dx)):+.0f}$",
                transform=ap.transAxes, fontsize=10)
        ap.set_xlabel(r"$x/w$")
        if col == 0:
            ai.set_ylabel(r"$y/w$")
            ap.set_ylabel(r"$y/w$")
        else:
            ai.set_yticklabels([])
            ap.set_yticklabels([])
        ai.set_xticklabels([])
    panel(axes[0, 0], "a")
    panel(axes[1, 0], "b")
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, "fig3_lg_modes.png")


def fig_rotation():
    N, dx = 1024, 0.5
    x, X, Y = mesh(N, dx)
    E0 = (lg(X, Y, 0, 0, W0, LAM) + lg(X, Y, 0, 2, W0, LAM)) / np.sqrt(2)
    zeta = np.linspace(-3, 3, 25)
    th, th_par = [], []
    snaps = {}
    for s in zeta:
        E = propagate(E0, dx, LAM, s * ZR)
        th.append(azimuthal_angle(np.abs(E) ** 2, X, Y, 2))
        th_par.append(azimuthal_angle(np.abs(propagate(E0, dx, LAM, s * ZR, paraxial=True)) ** 2, X, Y, 2))
        if s in (0.0, 1.0, 3.0):
            snaps[s] = E
    th, th_par = np.array(th), np.array(th_par)
    ref = gouy(zeta * ZR, W0, LAM)
    log(f"[rot] (LG_0^0 + LG_0^2)/sqrt2: orientation of the cos 2(phi - theta) lobes vs arctan(z/zR), z in [-3, 3] zR")
    log(f"[rot]   paraxial TF: max |theta - arctan| = {np.max(np.abs(th_par - ref)):.0e}; exact TF: {np.max(np.abs(th - ref)):.1e} rad "
        f"(eps = {EPS:.1e}); total turn -3..3 zR = {np.degrees(th[-1] - th[0]):.3f} deg")

    fig = plt.figure(figsize=DOUBLE)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.9], wspace=0.08)
    for i, s in enumerate((0.0, 1.0, 3.0)):
        a = fig.add_subplot(gs[0, i])
        w = beam_width(s * ZR, W0, LAM)
        sel = np.abs(x) <= 2.5 * w
        I = np.abs(snaps[s][np.ix_(sel, sel)]) ** 2
        a.imshow(I.T / I.max(), origin="lower", extent=[-2.5, 2.5, -2.5, 2.5], cmap="magma")
        t = gouy(s * ZR, W0, LAM)
        a.plot([-2.2 * np.cos(t), 2.2 * np.cos(t)], [-2.2 * np.sin(t), 2.2 * np.sin(t)], color="white", lw=0.8, ls="--")
        a.set_xticks([])
        a.set_yticks([])
        a.minorticks_off()
        a.set_xlabel(rf"$z = {s:.0f}\,z_R$")
        if i == 0:
            panel(a, "a")
    a = fig.add_subplot(gs[0, 3])
    zz = np.linspace(-3, 3, 300)
    a.plot(zz, np.degrees(gouy(zz * ZR, W0, LAM)), color=GREY, lw=1.2, label=r"$\arctan(z/z_R)$")
    a.plot(zeta, np.degrees(th), "o", color=BLUE, mfc="none", label="lobe orientation")
    a.set_xlabel(r"$z/z_R$")
    a.set_ylabel(r"$\theta$ (deg)")
    a.yaxis.set_label_position("right")
    a.tick_params(axis="y", which="both", labelleft=False, labelright=True)
    a.legend(loc="upper left")
    panel(a, "b")
    save(fig, "fig4_rotation.png")


# ---------------------------------------------------------------- Talbot


def fig_talbot():
    d, M = 10.0, 512
    xg = np.arange(M) * d / M
    t = binary_grating(xg, d)
    zT, zp = talbot_exact(d, LAM), talbot_paraxial(d, LAM)
    log(f"[talbot] d = {d} um: z_T exact = {zT:.4f} um, paraxial 2d^2/lambda = {zp:.4f} um")
    zs = np.linspace(0, zT, 801)
    Ek = np.fft.fft(t)
    carpet = np.abs(np.fft.ifft(Ek[None, :] * np.stack([transfer_function((M,), d / M, LAM, z, bandlimit=False) for z in zs]),
                                axis=1)) ** 2

    for dd in (10.0, 2.0):
        zT_, zp_ = talbot_exact(dd, LAM), talbot_paraxial(dd, LAM)
        xs = np.arange(M) * dd / M
        tc = cosine_grating(xs, dd)
        zz = zp_ * np.linspace(0.99, 1.01, 11)
        z_pk, _ = parabolic_peak(zz, np.array([fidelity(tc, propagate(tc, dd / M, LAM, z, bandlimit=False)) for z in zz]))
        tb = binary_grating(xs, dd)
        Fb_ex = fidelity(tb, propagate(tb, dd / M, LAM, zT_, bandlimit=False))
        Fb_par = fidelity(tb, propagate(tb, dd / M, LAM, zp_, bandlimit=False, paraxial=True))
        Ih = np.abs(propagate(tb, dd / M, LAM, zT_ / 2, bandlimit=False)) ** 2
        sh = harmonic_shift(Ih, dd, dd / M) - harmonic_shift(tb, dd, dd / M)
        zfine = np.linspace(0.9 * zT_, 1.1 * zT_, 2001)
        Fb = [fidelity(tb, propagate(tb, dd / M, LAM, z, bandlimit=False)) for z in zfine]
        zb, Fbm = parabolic_peak(zfine, np.array(Fb))
        log(f"[talbot] d = {dd} um (d/lambda = {dd / LAM:.0f}): exact {zT_:.5f}, paraxial {zp_:.5f} um; cosine grating revival "
            f"peak (11 samples, 0.002 z_T apart, around 2d^2/lambda) at {z_pk:.6f} um, |peak - z_T| = {abs(z_pk - zT_):.0e} um, "
            f"F(z_T) = {fidelity(tc, propagate(tc, dd / M, LAM, zT_, bandlimit=False)):.15f}, "
            f"F(2d^2/lambda) = {fidelity(tc, propagate(tc, dd / M, LAM, zp_, bandlimit=False)):.8f}")
        log(f"[talbot]   binary 50% grating, M = {M} samples per period: F(z_T exact) = {Fb_ex:.4f}; paraxial TF at 2d^2/lambda "
            f"F = {Fb_par:.15f}; best revival in 0.9..1.1 z_T at {zb:.3f} um, F = {Fbm:.4f}; "
            f"half-Talbot first-harmonic shift = {sh % 1:.12f} d")
        Ftk = np.abs(np.fft.fft(tb)) ** 2
        evan = Ftk[np.abs(np.fft.fftfreq(M, dd / M)) > 1 / LAM].sum() / Ftk.sum()
        log(f"[talbot]   binary: fidelity at z_T/2 with the grating shifted by d/2 = "
            f"{fidelity(np.roll(tb, M // 2), propagate(tb, dd / M, LAM, zT_ / 2, bandlimit=False)):.4f}; "
            f"power in evanescent orders = {evan:.4f}")

    zfine = np.linspace(0.97 * zT, 1.03 * zT, 601)
    tc = cosine_grating(xg, d)
    Fc = [fidelity(tc, propagate(tc, d / M, LAM, z, bandlimit=False)) for z in zfine]
    Fb = [fidelity(t, propagate(t, d / M, LAM, z, bandlimit=False)) for z in zfine]
    Fbp = [fidelity(t, propagate(t, d / M, LAM, z, bandlimit=False, paraxial=True)) for z in zfine]

    fig = plt.figure(figsize=(9.2, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.28)
    a1 = fig.add_subplot(gs[0, 0])
    tiled = np.tile(carpet, (1, 2))
    a1.imshow(tiled, origin="lower", aspect="auto", extent=[0, 2, 0, 1], cmap="magma", vmin=0, vmax=np.percentile(carpet, 99.5))
    for frac, lab in ((0.5, r"$z_T/2$"), (0.25, r"$z_T/4$")):
        a1.axhline(frac, color="white", lw=0.5, ls=":")
        a1.text(2.02, frac, lab, va="center", fontsize=9, transform=a1.transData, clip_on=False)
    a1.set_xlabel(r"$x/d$")
    a1.set_ylabel(r"$z/z_T$")
    a1.minorticks_off()
    a1.tick_params(color="white", which="both")
    panel(a1, "a")
    a2 = fig.add_subplot(gs[0, 1])
    a2.plot(zfine, Fc, color=BLUE, label="cosine grating")
    a2.plot(zfine, Fb, color=ORANGE, label="binary, exact TF")
    a2.plot(zfine, Fbp, color=GREY, lw=1.2, ls="--", label="binary, paraxial TF")
    a2.axvline(zT, color="k", lw=0.7)
    a2.axvline(zp, color="k", lw=0.7, ls=":")
    a2.text(zT - 0.4, 0.72, r"$z_T$", ha="right", fontsize=10)
    a2.text(zp + 0.4, 0.72, r"$2d^2/\lambda$", ha="left", fontsize=10)
    a2.set_xlabel(r"$z$ ($\mu$m)")
    a2.set_ylabel(r"field fidelity $F(z)$")
    a2.set_ylim(0.55, 1.02)
    a2.legend(loc="lower left")
    panel(a2, "b")
    save(fig, "fig5_talbot.png")


def fig_talbot_defect():
    ratios = np.unique(np.round(np.logspace(np.log10(5), 6, 60)).astype(int))
    defect = np.array([1 - talbot_fidelity_series(r * LAM, LAM, talbot_exact(r * LAM, LAM)) for r in ratios])
    fft_r = np.array([5, 10, 20, 40, 80, 160, 320])
    fft_defect = []
    for r in fft_r:
        d, M = r * LAM, 4096
        t = binary_grating_series(np.arange(M) * d / M, d, LAM)
        fft_defect.append(1 - fidelity(t, propagate(t, d / M, LAM, talbot_exact(d, LAM), bandlimit=False)))
    fft_defect = np.array(fft_defect)
    series_at_fft = np.array([1 - talbot_fidelity_series(r * LAM, LAM, talbot_exact(r * LAM, LAM)) for r in fft_r])
    log(f"[defect] binary grating 1 - F(z_T), FFT on propagating orders vs series sum, d/lambda = 5..320: "
        f"max diff = {np.max(np.abs(fft_defect - series_at_fft)):.0e}")
    for r in (4, 20, 100, 1000, 10**4, 10**5, 10**6):
        dfc = 1 - talbot_fidelity_series(r * LAM, LAM, talbot_exact(r * LAM, LAM))
        log(f"[defect]   d/lambda = {r:>7d}: 1 - F = {dfc:.4e}, asymptote 0.5137 sqrt(lambda/d) = "
            f"{talbot_defect_asymptote(r * LAM, LAM):.4e}, ratio {dfc / talbot_defect_asymptote(r * LAM, LAM):.4f}")
    big = ratios >= 1000
    slope = np.polyfit(np.log(ratios[big]), np.log(defect[big]), 1)[0]
    log(f"[defect] fitted exponent of 1 - F vs d/lambda for d/lambda >= 1000: {slope:.4f} (predicted -1/2)")

    fig, a = plt.subplots(figsize=SINGLE)
    rr = np.logspace(np.log10(5), 6, 200)
    a.loglog(rr, talbot_defect_asymptote(rr * LAM, LAM), color=GREY, lw=1.2, label=r"$0.514\,(\lambda/d)^{1/2}$")
    a.loglog(ratios, defect, "-", color=ORANGE, lw=1.0, label="phase sum over orders")
    a.loglog(fft_r, fft_defect, "o", color=BLUE, mfc="none", label="FFT propagator")
    a.set_xlabel(r"$d/\lambda$")
    a.set_ylabel(r"$1-F(z_T)$, binary grating")
    a.legend(loc="lower left")
    save(fig, "fig6_talbot_defect.png")


# ---------------------------------------------------------------- sampling


def fig_sampling():
    N, dx, a_slit, N_ref = 1024, 0.5, 20.0, 2**16
    L = N * dx
    slit = lambda n: (np.abs(grid(n, dx)) < a_slit / 2).astype(complex)
    zs = np.unique(np.concatenate([np.logspace(np.log10(30), np.log10(8000), 40), [4000.0]]))
    c = np.abs(grid(N, dx)) < L / 4
    variants = {(False, False): [], (True, False): [], (False, True): [], (True, True): []}
    refs = {}
    for z in zs:
        ref = propagate(slit(N_ref), dx, LAM, z, bandlimit=False)[N_ref // 2 - N // 2:N_ref // 2 + N // 2]
        refs[z] = ref
        for (bl, pad), lst in variants.items():
            E = propagate(slit(N), dx, LAM, z, bandlimit=bl, pad=pad)
            lst.append(np.linalg.norm((E - ref)[c]) / np.linalg.norm(ref[c]))
    zc1, zc2 = critical_distance(L, dx, LAM), critical_distance(2 * L, dx, LAM)
    log(f"[sampling] 1D slit a = {a_slit} um, N = {N}, dx = {dx} um (= lambda), L = {L} um; reference: N = {N_ref} "
        f"(alias-free to {critical_distance(N_ref * dx, dx, LAM):.0f} um); error = rel. L2 on |x| < L/4")
    log(f"[sampling] z_c = (L/2) sqrt((2dx/lambda)^2 - 1) = {zc1:.1f} um unpadded, {zc2:.1f} um padded")
    i4 = int(np.argmin(np.abs(zs - 4000)))
    for pad in (False, True):
        plain, limited = np.array(variants[(False, pad)]), np.array(variants[(True, pad)])
        log(f"[sampling] pad={pad!s:5}: first scanned z where the unlimited error exceeds 1.5x the band-limited one: "
            f"{zs[np.argmax(plain > 1.5 * limited)]:.0f} um (previous sample {zs[np.argmax(plain > 1.5 * limited) - 1]:.0f} um)")
    for (bl, pad), lst in variants.items():
        lst = np.array(lst)
        log(f"[sampling]   bandlimit={bl!s:5} pad={pad!s:5}: error at z = 300 / 4000 / 8000 um = "
            f"{np.interp(300, zs, lst):.1e} / {lst[i4]:.1e} / {lst[-1]:.1e}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE)
    style = {(False, False): (GREY, "--", "plain"), (True, False): (ORANGE, "-", "band-limited"),
             (False, True): (GREY, "-", "padded"), (True, True): (BLUE, "-", "padded + band-limited")}
    for key, lst in variants.items():
        col, ls, lab = style[key]
        a1.loglog(zs, lst, ls, color=col, label=lab, lw=1.4)
    for zc, lab in ((zc1, r"$z_c(L)$"), (zc2, r"$z_c(2L)$")):
        a1.axvline(zc, color="k", lw=0.6, ls=":")
        a1.text(zc * 1.05, 2e-5, lab, fontsize=9)
    a1.set_xlabel(r"$z$ ($\mu$m)")
    a1.set_ylabel(r"relative error on $|x|<L/4$")
    a1.set_ylim(1e-5, 1)
    a1.legend(loc="upper left")
    panel(a1, "a")

    z = 4000.0
    x = grid(N, dx)
    a2.plot(x, np.abs(refs[zs[i4]]) ** 2, color="k", lw=2.2, alpha=0.35, label="reference (64$\\times$ window)")
    a2.plot(x, np.abs(propagate(slit(N), dx, LAM, z, bandlimit=False, pad=True)) ** 2, color=ORANGE, lw=0.7, label="padded")
    a2.plot(x, np.abs(propagate(slit(N), dx, LAM, z, bandlimit=True, pad=True)) ** 2, color=BLUE, lw=0.9, ls="--",
            label="padded + band-limited")
    a2.axvspan(-L / 4, L / 4, color=GREY, alpha=0.08, lw=0)
    a2.set_xlim(-L / 2, L / 2)
    a2.set_xlabel(r"$x$ ($\mu$m)")
    a2.set_ylabel(r"$|E(x, z = 4\,\mathrm{mm})|^2$")
    a2.legend(loc="upper right")
    panel(a2, "b")
    fig.tight_layout()
    save(fig, "fig7_sampling.png")


def fig_grid():
    """Where a Gaussian (w0 = 10 um) breaks: window too small at 3 zR, or samples too coarse."""
    z = 3 * ZR
    wz = beam_width(z, W0, LAM)
    dx = 0.5
    Ns = np.array([48, 56, 64, 80, 96, 112, 128, 160, 192, 256, 384, 512])
    werr = {}
    for pad in (False, True):
        out = []
        for N in Ns:
            x, X, Y = mesh(N, dx)
            out.append(abs(second_moment_width(propagate(gaussian(X, Y, W0, LAM), dx, LAM, z, pad=pad), x)
                           / width_nonparaxial(z, W0, LAM) - 1))
        werr[pad] = np.array(out)
    ratio = Ns * dx / (2 * wz)
    log(f"[grid] window scan at z = 3 zR (w = {wz:.2f} um), dx = 0.5 um: |w_num / w_nonparaxial - 1| vs (L/2)/w")
    for r, e0, e1 in zip(ratio, werr[False], werr[True]):
        log(f"    (L/2)/w = {r:5.2f}: unpadded {e0:.1e}, padded {e1:.1e}")

    L = 512.0
    dxs = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10])
    ferr = []
    for d_ in dxs:
        x, X, Y = mesh(int(round(L / d_)), d_)
        E = propagate(gaussian(X, Y, W0, LAM), d_, LAM, ZR, paraxial=True)
        ref = gaussian(X, Y, W0, LAM, ZR)
        ferr.append(np.sqrt(power(E - ref, d_) / power(ref, d_)))
    ferr = np.array(ferr)
    # The DFT sees the spectrum folded at the Nyquist frequency; power beyond it along either axis:
    q = scipy.special.erfc(np.pi * W0 / (np.sqrt(2) * dxs))
    beyond = q * (2 - q)  # 1 - erf^2 without cancellation
    log(f"[grid] sampling scan, L = {L} um, paraxial TF, Gaussian at z = zR: rel. L2 field error vs dx/w0 "
        f"(and sqrt of the power fraction beyond Nyquist)")
    for d_, e, b in zip(dxs, ferr, beyond):
        log(f"    dx/w0 = {d_ / W0:.2f}: error {e:.1e}, sqrt(P_beyond) {np.sqrt(b):.1e}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE)
    a1.semilogy(ratio, werr[False], "o-", color=ORANGE, mfc="none", lw=1.0, label="unpadded")
    a1.semilogy(ratio, werr[True], "s-", color=BLUE, mfc="none", lw=1.0, ms=4, label="padded")
    a1.set_xlabel(r"half-window $/\,w(3z_R)$")
    a1.set_ylabel(r"$|w/w_\mathrm{exact}-1|$ at $z=3z_R$")
    a1.legend(loc="upper right")
    panel(a1, "a")
    a2.semilogy(dxs / W0, ferr, "o", color=BLUE, mfc="none", label="field error at $z_R$")
    a2.semilogy(dxs / W0, np.sqrt(beyond), color=GREY, lw=1.2, label=r"$\sqrt{P(|f_i|>1/2\Delta x)}$")
    a2.set_xlabel(r"$\Delta x / w_0$")
    a2.set_ylabel("relative error")
    a2.set_ylim(1e-17, 3)
    a2.legend(loc="lower right")
    panel(a2, "b")
    fig.tight_layout()
    save(fig, "fig8_grid.png")


def main():
    t0 = time.time()
    FIG.mkdir(exist_ok=True)
    environment_header()
    check_propagator()
    fig_gaussian()
    fig_nonparaxial()
    fig_lg()
    fig_rotation()
    fig_talbot()
    fig_talbot_defect()
    fig_sampling()
    fig_grid()
    print(f"total {time.time() - t0:.1f} s")
    (FIG / "results.txt").write_text("\n".join(results) + "\n")


if __name__ == "__main__":
    main()
