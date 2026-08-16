"""Generate figures/fig1-fig10 and figures/results.txt."""

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
from scipy.optimize import brentq
from scipy.special import j1

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Fourier-optics"))

from fourier_optics import gaussian
from iscat import (
    DipoleImager,
    ParticleModel,
    absorption_cross_section,
    backscatter_x2_coefficient,
    best_focus,
    brownian_track,
    camera_grid,
    clausius_mossotti,
    contrast,
    contrast_noise,
    crlb,
    detection_limit,
    dipole_amplitude,
    fisher_information,
    fit_msd,
    fresnel,
    gaussian_psd,
    lens_polarizability,
    localise,
    matched_filter,
    mie_backscatter_amplitude,
    mie_coefficients,
    mie_dipole_polarizability,
    mie_efficiencies,
    mie_nmax,
    msd,
    photothermal_contrast,
    polarizability,
    protein_polarizability,
    ratiometric_contrast,
    rayleigh_sigma_sca,
    rayleigh_x2_coefficient,
    rough_surface,
    roughness_field,
    scalar_peak,
    scalar_psf,
    scalar_reference,
    snr_single_pixel,
    speckle_contrast_std,
    thermal_gamma,
    vectorial_psf,
    vectorial_reference,
    vectorial_transfer,
    vectorial_window_signal,
)

FIG = ROOT / "figures"
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2"
GREY = "#9a9994"
SINGLE = (4.6, 3.3)
DOUBLE = (9.2, 3.5)

LAM0 = 532.0  # nm
N1, N2 = 1.52, 1.33
NA = 1.4
N_PS = 1.59
N_AU = 0.5439 + 2.2309j
M_PS, M_AU = N_PS / N2, N_AU / N2
K = 2 * np.pi * N2 / LAM0
LAM_M = LAM0 / N2
R, T, TP = fresnel(N1, N2)
ER = vectorial_reference(LAM0, N1, N2)
PIXEL = 60.0  # nm
DNDC = 0.185  # mL/g
SIGMAS = 3.0
ROUGH_SIGMA, ROUGH_ELL = 0.3, 20.0  # nm
PUMP = 1e-9  # W/nm^2 = 100 kW/cm^2
F_MOD = 1e5  # Hz
PS_COL, AU_COL = BLUE, ORANGE

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


def bound(v):
    """Round-off-level numbers as an upper bound (they change between BLAS builds)."""
    return "< 1e-15" if v < 1e-15 else f"< 1e{int(np.floor(np.log10(v))) + 1}"


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
    log(f"# lambda0 = {LAM0} nm, n_glass = {N1}, n_water = {N2}, NA = {NA}, k = {K:.6f} /nm, "
        f"lambda_m = {LAM_M:.2f} nm")
    log(f"# polystyrene n = {N_PS}, gold n = {N_AU.real} + {N_AU.imag}i (Johnson & Christy, interpolated); "
        f"L_PS = {clausius_mossotti(M_PS):.5f}, L_Au = {clausius_mossotti(M_AU):.4f}")
    log(f"# reference r = {R:.5f}, |r|^2 = {R**2:.5f}; t = {T:.5f}, t' = {TP:.5f}; pixel = {PIXEL} nm")


def peak_field(h, alpha, zf=0.0, na=NA):
    return vectorial_psf(0.0, 0.0, h, alpha, LAM0, N1, N2, na, zf=zf)


def peak_contrast(h, alpha, zf=0.0):
    ex, ey = peak_field(h, alpha, zf)
    return contrast(ex, vectorial_reference(LAM0, N1, N2, zf), ey)


def sigma_ratio(a, m):
    return mie_efficiencies(m, K * a)[1] * np.pi * a**2 / rayleigh_sigma_sca(a, m, K)


def f_ratio(a, m):
    return mie_backscatter_amplitude(a, m, K) / dipole_amplitude(polarizability(a, m), K)


def check_mie():
    x = 2 * np.pi * 0.525 / 0.6328
    q = mie_efficiencies(1.55, x)
    log(f"[mie] Bohren-Huffman example m = 1.55, x = {x:.5f}: Q_ext = Q_sca = {q[1]:.5f}, "
        f"Q_back = {q[2]:.5f} (BHMIE listing 3.10543, 2.92534)")
    try:
        import miepython
    except ImportError:
        log("[mie] miepython not installed; cross-check skipped")
        return
    worst, trunc = 0.0, 0.0
    for m in (M_PS, M_AU, 1.5 + 0.01j, 2 + 1j):
        for x in np.geomspace(0.01, 50, 40):
            ours = np.array(mie_efficiencies(m, x))
            ref = np.array(miepython.efficiencies_mx(np.conj(m), x)[:3])
            worst = max(worst, np.max(np.abs(ours / ref - 1)))
            a, b = mie_coefficients(m, x, nmax=mie_nmax(x) + 60)
            n = np.arange(1, len(a) + 1)
            q_conv = 2 / x**2 * np.sum((2 * n + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2))
            trunc = max(trunc, abs(ours[1] / q_conv - 1))
    log(f"[mie] max relative difference from miepython {miepython.__version__} (Q_ext, Q_sca, Q_back; "
        f"4 indices x 40 sizes, x = 0.01..50): {worst:.1e}; ours vs 60 extra terms (Q_sca): "
        f"{bound(trunc)}")


def first_crossing(fn, level, grid=np.geomspace(0.5, 200, 200)):
    """Smallest radius on the grid where |fn - 1| reaches level, refined by bisection."""
    g = lambda ai: abs(fn(ai) - 1) - level
    vals = np.array([g(ai) for ai in grid])
    i = np.argmax(vals > 0)
    return brentq(g, grid[i - 1], grid[i])


def fig_mie():
    a = np.geomspace(1, 150, 90)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE)
    for m, col, lab in ((M_PS, PS_COL, "polystyrene"), (M_AU, AU_COL, "gold")):
        q_mie = np.array([mie_efficiencies(m, K * ai)[1] for ai in a])
        q_ray = rayleigh_sigma_sca(a, m, K) / (np.pi * a**2)
        ax1.plot(a, q_ray, color=col, lw=1.2)
        ax1.plot(a[::3], q_mie[::3], "o", color=col, mfc="none", label=lab)
        sr = np.array([sigma_ratio(ai, m) for ai in a])
        fr = np.abs([f_ratio(ai, m) for ai in a])
        ax2.plot(a, sr, "o", color=col, mfc="none", ms=3.5)
        ax2.plot(a, fr, "s", color=col, ms=2.5)
        x = K * a
        sig_law = 1 + rayleigh_x2_coefficient(m) * x**2
        f_law = np.abs(1 + backscatter_x2_coefficient(m) * x**2)
        ax2.plot(a[sig_law > 0.3], sig_law[sig_law > 0.3], "--", color=col, lw=1)
        keep = (1 + backscatter_x2_coefficient(m) * x**2).real > 0.3
        ax2.plot(a[keep], f_law[keep], ":", color=col, lw=1.2)
        for name, fn in (("sigma_sca", lambda ai: sigma_ratio(ai, m)), ("|f_back|", lambda ai: abs(f_ratio(ai, m)))):
            r2, r10 = first_crossing(fn, 0.02), first_crossing(fn, 0.10)
            log(f"[mie] {lab}: {name} departs from Rayleigh by 2% at a = {r2:.2f} nm (x = {K * r2:.4f}), "
                f"by 10% at a = {r10:.2f} nm (x = {K * r10:.4f})")
        log(f"[mie] {lab}: B = {rayleigh_x2_coefficient(m):+.5f} (sigma), C = "
            f"{complex(backscatter_x2_coefficient(m)):.5f} (f_back); x^2-law 10% radius for sigma = "
            f"{np.sqrt(0.1 / abs(rayleigh_x2_coefficient(m))) / K:.2f} nm")
    aa = np.linspace(5, 40, 351)
    gr = np.array([sigma_ratio(ai, M_AU) for ai in aa])
    log(f"[mie] gold: sigma_Mie/sigma_Rayleigh peaks at {gr.max():.4f} at a = {aa[np.argmax(gr)]:.1f} nm")
    log(f"[mie] |L_Au / L_PS| = {abs(clausius_mossotti(M_AU) / clausius_mossotti(M_PS)):.2f}; "
        f"|f_back| Mie/Rayleigh for polystyrene at a = 20 nm: {abs(f_ratio(20.0, M_PS)):.4f}")
    dev_m = sigma_ratio(LAM_M / 20, M_PS) - 1
    dev_0 = sigma_ratio(LAM0 / 20, M_PS) - 1
    log(f"[mie] Rayleigh a^6 law error, polystyrene: at lambda_m/20 = {LAM_M / 20:.2f} nm "
        f"error {dev_m:+.4%}; at lambda0/20 = {LAM0 / 20:.2f} nm error {dev_0:+.4%}")
    ax1.set(xscale="log", yscale="log", xlabel=r"radius $a$ (nm)", ylabel=r"$Q_{\rm sca}=\sigma_{\rm sca}/\pi a^2$")
    ax1.set_ylim(1e-9, 20)
    ax1.legend(loc="lower right")
    ax1.text(0.05, 0.92, "lines: Rayleigh\nsymbols: Mie", transform=ax1.transAxes, va="top")
    ax2.axvline(LAM_M / 20, color=GREY, lw=0.8, ls="--")
    ax2.axvline(LAM0 / 20, color=GREY, lw=0.8, ls=":")
    ax2.text(LAM_M / 20 * 0.97, 1.24, r"$\lambda_m/20$", ha="right", color="k")
    ax2.text(LAM0 / 20 * 1.03, 1.24, r"$\lambda_0/20$", ha="left", color="k")
    ax2.set(xscale="log", xlabel=r"radius $a$ (nm)", ylabel="Mie / Rayleigh", ylim=(0.3, 1.3), xlim=(1, 150))
    ax2.plot([], [], "o", color="k", mfc="none", ms=3.5, label=r"$\sigma_{\rm sca}$")
    ax2.plot([], [], "s", color="k", ms=2.5, label=r"$|f_{\rm back}|$")
    ax2.plot([], [], "--", color="k", lw=1, label=r"$1+Bx^2$")
    ax2.plot([], [], ":", color="k", lw=1.2, label=r"$|1+Cx^2|$")
    ax2.legend(loc="lower left", ncol=2)
    panel(ax1, "a")
    panel(ax2, "b")
    save(fig, "fig1_mie.png")


def interference_and_scattering(a, m, mie=True, h=20.0):
    """|s/r| at the PSF centre for a dipole with the Mie backscatter amplitude."""
    f = mie_backscatter_amplitude(a, m, K) if mie else dipole_amplitude(polarizability(a, m), K)
    out = []
    for fi in np.atleast_1d(f):
        ex, _ = peak_field(h, 4 * np.pi * fi / K**2)
        out.append(ex / ER)
    s = np.array(out)
    return 2 * np.abs(s), np.abs(s) ** 2


def fig_radius():
    a = np.geomspace(2, 150, 60)
    fit = (a >= 10) & (a <= 40)
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE, sharey=True)
    for ax, m, col, lab in ((axes[0], M_PS, PS_COL, "polystyrene"), (axes[1], M_AU, AU_COL, "gold")):
        i_ray, s_ray = interference_and_scattering(a, m, mie=False)
        i_mie, s_mie = interference_and_scattering(a, m)
        ax.plot(a, i_ray, color=col, lw=1.2)
        ax.plot(a, s_ray, color=col, lw=1.2, ls="--")
        ax.plot(a[::2], i_mie[::2], "o", color=col, mfc="none")
        ax.plot(a[::2], s_mie[::2], "s", color=col, ms=3)
        ax.axvspan(10, 40, color=GREY, alpha=0.15, lw=0)
        slopes = []
        for y in (i_ray, s_ray, i_mie, s_mie):
            slopes.append(np.polyfit(np.log(a[fit]), np.log(y[fit]), 1)[0])
        log(f"[radius] {lab}, h = 20 nm, focus on interface: log-log slope on a = 10-40 nm: "
            f"Rayleigh interference {slopes[0]:.4f}, scattering {slopes[1]:.4f}; "
            f"Mie interference {slopes[2]:.4f}, scattering {slopes[3]:.4f}")
        a_x = brentq(lambda ai: interference_and_scattering(ai, m)[1][0] - interference_and_scattering(ai, m)[0][0], 2, 150)
        log(f"[radius] {lab}: |s|^2 overtakes 2|s| (|s/r| = 2) at a = {a_x:.1f} nm (Mie)")
        if a_x < 100:
            ax.axvline(a_x, color=GREY, lw=0.8, ls=":")
            ax.text(a_x * 1.06, 2e-7, rf"$|s/r|=2$" + "\n" + rf"$a={a_x:.0f}$ nm", fontsize=9)
        ax.set(xscale="log", yscale="log", xlabel=r"radius $a$ (nm)", xlim=(2, 150), ylim=(1e-8, 1e3))
        ax.text(0.04, 0.95, lab, transform=ax.transAxes, va="top", fontsize=11)
        ax.text(2.4, i_ray[np.searchsorted(a, 2.4)] * 15, r"$2|s/r|\propto a^3$", fontsize=9)
        ax.text(5.0, s_ray[np.searchsorted(a, 5.0)] * 0.08, r"$|s/r|^2\propto a^6$", fontsize=9)
    axes[0].set_ylabel("contrast term at PSF centre")
    axes[1].plot([], [], color="k", lw=1.2, label="Rayleigh")
    axes[1].plot([], [], "o", color="k", mfc="none", label="Mie")
    axes[1].legend(loc="upper left", bbox_to_anchor=(0.0, 0.9))
    panel(axes[0], "a")
    panel(axes[1], "b")
    fig.subplots_adjust(wspace=0.06)
    save(fig, "fig2_radius.png")


def phase_period(h, field):
    return 2 * np.pi / np.polyfit(h, np.unwrap(np.angle(field)), 1)[0]


def fig_height():
    a = 20.0
    h = np.linspace(a, 600, 291)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE)
    alpha_ps = mie_dipole_polarizability(a, M_PS, K)
    alpha_au = mie_dipole_polarizability(a, M_AU, K)
    for alpha, col, lab in ((alpha_ps, PS_COL, "polystyrene"), (alpha_au, AU_COL, "gold")):
        C = np.array([peak_contrast(hh, alpha) for hh in h])
        ax1.plot(h, C / np.max(np.abs(C)), color=col, label=lab)
        hmin = h[np.argmin(C[:60])]
        log(f"[height] {lab} a = {a} nm, NA {NA}, focus on interface: first dark fringe at h = {hmin:.0f} nm, "
            f"C(h = a) = {C[0]:+.4e}, max |C| = {np.max(np.abs(C)):.4e}")
    shift = np.angle(alpha_au / alpha_ps)
    log(f"[height] arg(alpha_Au/alpha_PS) = {shift:.4f} rad: gold fringes lead by this phase")
    sc = scalar_peak(h, 1.0, LAM0, N2, NA, N1)
    Csc = 2 * (sc / scalar_reference(LAM0, N2, N1, N2)).real
    ax1.plot(h, Csc / np.max(np.abs(Csc)), color=GREY, lw=1.1, ls="--", label="scalar")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set(xlabel=r"particle height $h$ (nm)", ylabel=r"$C(h)/\max|C|$", xlim=(0, 600), ylim=(-1.1, 1.25))
    ax1.legend(loc="upper right", ncol=3, handlelength=1.4, columnspacing=1.0)

    nas = np.array([0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2, 1.25, 1.3, 1.33, 1.36, 1.4, 1.45])
    hh = np.linspace(20, 220, 101)
    vec = []
    for na in nas:
        ex = np.array([peak_field(x, 1.0, na=na)[0] for x in hh])
        vec.append(phase_period(hh, ex) / (LAM_M / 2))
    vec = np.array(vec)
    nn = np.linspace(0.01, N2, 300)
    c = np.sqrt(1 - (nn / N2) ** 2)
    ax2.plot(nn, 4 / (3 + c), color=GREY, lw=1.2, label=r"scalar $4/(3+\cos\theta_m)$")
    ax2.plot(nas, vec, "o", color=PS_COL, label="vectorial, peak")
    W = 3000.0
    win = []
    for na in nas[::3]:
        s = np.array([vectorial_window_signal(x, 1.0, LAM0, N1, N2, na, W) for x in np.linspace(0, 800, 161)])
        win.append(phase_period(np.linspace(0, 800, 161), s) / (LAM_M / 2))
    ax2.plot(nas[::3], win, "D", color=GREEN, ms=4, label=rf"summed over $W={W / 1000:.0f}\,\mu$m")
    ax2.axvline(N2, color=GREY, lw=0.8, ls=":")
    ax2.text(N2 - 0.02, 1.27, r"NA$=n_{\rm water}$", ha="right", fontsize=9)
    ax2.set(xlabel="numerical aperture", ylabel=r"fringe period / $(\lambda_0/2n_{\rm water})$",
            xlim=(0, 1.5), ylim=(0.97, 1.36))
    ax2.legend(loc="upper left")
    for na, v in zip(nas, vec):
        log(f"[height] NA = {na:.2f}: vectorial peak period (fit h = 20-220 nm) = {v * LAM_M / 2:.2f} nm "
            f"= {v:.4f} lambda_m/2" + (f"; scalar 2 lambda_m/(3+c) = {2 * LAM_M / (3 + np.sqrt(1 - (na / N2) ** 2)):.2f} nm"
                                      if na <= N2 else ""))
    for na, v in zip(nas[::3], win):
        log(f"[height] NA = {na:.2f}: window-summed (W = {W:.0f} nm) period = {v:.6f} lambda_m/2; "
            f"predicted 1 + 1/(k W)^2 = {1 + 1 / (K * W) ** 2:.6f}")
    log(f"[height] scalar, focus tracking the particle: period = lambda_m/2 = {LAM_M / 2:.2f} nm exactly")
    c1 = 4 / 1.01 - 3
    log(f"[height] period within 1% of lambda_m/2 only below NA = n_water sin(arccos({c1:.5f})) = {N2 * np.sqrt(1 - c1**2):.4f} "
        f"(scalar fixed-focus formula)")
    panel(ax1, "a")
    panel(ax2, "b")
    save(fig, "fig3_height.png")


def fwhm(r, p):
    p = p / p[0]
    i = np.argmax(p < 0.5)
    return 2 * np.interp(0.5, [p[i], p[i - 1]], [r[i], r[i - 1]])


def fig_psf():
    a = 30.0
    alpha = mie_dipole_polarizability(a, M_PS, K)
    zf = best_focus(a, LAM0, N1, N2, NA)
    er = vectorial_reference(LAM0, N1, N2, zf)
    ext = 900.0
    x = np.linspace(-ext, ext, 181)
    X, Y = np.meshgrid(x, x)
    ex, ey = vectorial_psf(X, Y, a, alpha, LAM0, N1, N2, NA, zf=zf)
    C = contrast(ex, er, ey)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE, gridspec_kw={"width_ratios": [1, 1.25]})
    vmax = np.max(np.abs(C))
    im = ax1.imshow(C, extent=[-ext, ext, -ext, ext], origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label("contrast $C$")
    ax1.set(xlabel=r"$x$ (nm)", ylabel=r"$y$ (nm)")
    ax1.annotate("", xy=(700, -750), xytext=(400, -750), arrowprops=dict(arrowstyle="->", color="k"))
    ax1.text(550, -700, r"$\mathbf{E}_0$", ha="center", va="bottom")
    log(f"[psf] polystyrene a = {a} nm on glass (h = a), best focus zf = {zf:.1f} nm: peak C = {C[90, 90]:+.4e}; "
        f"at zf = 0: {peak_contrast(a, alpha):+.4e}")

    r = np.linspace(0, 900, 1801)
    vx, _ = vectorial_psf(r, 0 * r, a, 1.0, LAM0, N1, N2, NA)
    vy, _ = vectorial_psf(0 * r, r, a, 1.0, LAM0, N1, N2, NA)
    sc = scalar_psf(r, a, 1.0, LAM0, N2, NA, N1)
    v = 2 * np.pi / LAM0 * NA * np.maximum(r, 1e-9)
    airy = (2 * j1(v) / v) ** 2
    curves = ((np.abs(vx) ** 2, PS_COL, "-", r"vectorial, along $x$ ($\parallel \mathbf{E}_0$)", "vectorial along x"),
              (np.abs(vy) ** 2, PS_COL, "--", r"vectorial, along $y$", "vectorial along y"),
              (np.abs(sc) ** 2, GREY, "-", "scalar dipole", "scalar dipole"),
              (airy, "k", ":", r"Airy, NA 1.4", "Airy NA 1.4"))
    for p, col, ls, lab, name in curves:
        ax2.plot(r, p / p[0], color=col, ls=ls, label=lab)
        log(f"[psf] |E_s|^2 FWHM, {name}: {fwhm(r, p):.1f} nm")
    log(f"[psf] Airy 0.514 lambda0/NA = {0.5145 * LAM0 / NA:.1f} nm; vectorial x/y FWHM ratio = "
        f"{fwhm(r, np.abs(vx) ** 2) / fwhm(r, np.abs(vy) ** 2):.3f}")
    ix = (2 * (vx / ER).real)
    log(f"[psf] interference term Re(E_x) FWHM along x at focus zf = 0: "
        f"{fwhm(r, -ix) if ix[0] < 0 else fwhm(r, ix):.1f} nm")
    ax2.set(xlabel=r"$\rho$ (nm)", ylabel=r"$|E_s(\rho)|^2/|E_s(0)|^2$", xlim=(0, 600), ylim=(0, 1.05))
    ax2.legend(loc="upper right")
    panel(ax1, "a")
    panel(ax2, "b")
    fig.subplots_adjust(wspace=0.6)
    save(fig, "fig4_psf.png")


def fig_image():
    rng = np.random.default_rng(2024)
    a, npix = 15.0, 33
    N = 1e5
    alpha = mie_dipole_polarizability(a, M_PS, K)
    zf = best_focus(a, LAM0, N1, N2, NA)
    er = vectorial_reference(LAM0, N1, N2, zf)
    X, Y = camera_grid(npix, PIXEL)
    w0, xp, yp = 4000.0, 600.0, -300.0
    g = gaussian(X, Y, w0, LAM0, n=N1)
    g = g / gaussian(0.0, 0.0, w0, LAM0, n=N1)
    gp = gaussian(xp, yp, w0, LAM0, n=N1) / gaussian(0.0, 0.0, w0, LAM0, n=N1)
    xs, ys = X - xp, Y - yp
    ex, ey = vectorial_psf(xs, ys, a, alpha, LAM0, N1, N2, NA, zf=zf)
    I = np.abs(g + gp * ex / er) ** 2 + np.abs(gp * ey / er) ** 2
    B = np.abs(g) ** 2
    model = I / B - 1
    img = ratiometric_contrast(rng, I, B, N, n_bg_frames=1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE, gridspec_kw={"width_ratios": [1, 1.3]})
    ext = npix // 2 * PIXEL / 1000
    vmax = 4 / np.sqrt(N)
    im = ax1.imshow(img, extent=[-ext, ext, -ext, ext], origin="lower", cmap="gray", vmin=-vmax, vmax=vmax)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label("measured contrast")
    ax1.set(xlabel=r"$x$ ($\mu$m)", ylabel=r"$y$ ($\mu$m)")

    xc, yc = camera_grid(21, PIXEL)
    exc, eyc = vectorial_psf(xc, yc, a, alpha, LAM0, N1, N2, NA, zf=zf)
    Cc = contrast(exc, er, eyc)
    Ic = 1 + Cc
    c = 10
    T = Cc / Cc[c, c]
    M = 2000
    sig = contrast_noise(Ic, 1.0, N, n_bg_frames=1)
    stack = np.array([ratiometric_contrast(rng, Ic, 1.0, N, n_bg_frames=1) for _ in range(M)])
    px = stack[:, c, c]
    mf, mf_err = matched_filter(stack, T, sig)
    pred_px = sig[c, c]
    bins = np.linspace(Cc[c, c] - 5 * pred_px, Cc[c, c] + 5 * pred_px, 50)
    ax2.hist(px, bins=bins, density=True, color=GREY, alpha=0.6, label="centre pixel")
    ax2.hist(mf, bins=bins, density=True, histtype="step", color=PS_COL, lw=1.4, label="PSF fit")
    xx = np.linspace(bins[0], bins[-1], 400)
    for s, col in ((pred_px, "k"), (mf_err, PS_COL)):
        ax2.plot(xx, np.exp(-(xx - Cc[c, c]) ** 2 / (2 * s**2)) / np.sqrt(2 * np.pi) / s, color=col, lw=1, ls="--")
    ax2.axvline(Cc[c, c], color="k", lw=0.6)
    ax2.set(xlabel="contrast estimate", ylabel="probability density")
    ax2.legend(loc="upper left")
    log(f"[image] polystyrene a = {a} nm, h = a, best focus zf = {zf:.1f} nm, N = {N:.0e} photons/pixel, "
        f"one background frame, Gaussian illumination w0 = {w0 / 1000:.0f} um, particle at ({xp:.0f}, {yp:.0f}) nm")
    log(f"[image] model peak contrast {model[npix // 2 + int(yp / PIXEL), npix // 2 + int(xp / PIXEL)]:+.4e} "
        f"(off-centre, Gaussian beam); centred {Cc[c, c]:+.4e}")
    log(f"[image] {M} images, centre pixel: mean {px.mean():+.4e} +- {px.std(ddof=1) / np.sqrt(M):.1e} "
        f"(model {Cc[c, c]:+.4e}); std {px.std(ddof=1):.4e} +- {px.std(ddof=1) / np.sqrt(2 * (M - 1)):.1e} "
        f"(predicted {pred_px:.4e}; 1/sqrt(N) = {1 / np.sqrt(N):.4e})")
    log(f"[image] PSF fit: mean {mf.mean():+.4e} +- {mf.std(ddof=1) / np.sqrt(M):.1e}; std {mf.std(ddof=1):.4e} "
        f"+- {mf.std(ddof=1) / np.sqrt(2 * (M - 1)):.1e} (predicted {mf_err:.4e}); "
        f"gain over one pixel {pred_px / mf_err:.2f} = sqrt(sum T^2) = {np.sqrt(np.sum(T**2)):.2f}")
    panel(ax1, "a")
    panel(ax2, "b")
    fig.subplots_adjust(wspace=0.55)
    save(fig, "fig5_image.png")
    return T


def fig_detection(T):
    Ns = np.logspace(3, 12, 19)
    gain = np.sqrt(np.sum(T**2))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE)
    zfs = {}
    for m, col, lab in ((M_PS, PS_COL, "polystyrene"), (M_AU, AU_COL, "gold")):
        phase = np.angle(clausius_mossotti(m))
        zf = best_focus(5.0, LAM0, N1, N2, NA, alpha_phase=phase)
        zfs[lab] = zf

        def peak(a, zf=zf):
            ex, ey = vectorial_psf(0.0, 0.0, a, mie_dipole_polarizability(a, m, K), LAM0, N1, N2, NA, zf=zf)
            return contrast(ex, vectorial_reference(LAM0, N1, N2, zf), ey)

        for g, mk, ls, what in ((1.0, "o", "-", "pixel"), (gain, "s", "--", "PSF fit")):
            amin = np.array([detection_limit(peak, SIGMAS / np.sqrt(N) / g, a_lo=0.05, a_hi=200) for N in Ns])
            ax1.plot(Ns, amin, mk, color=col, mfc="none" if g == 1 else col, ms=3.5)
            j = np.searchsorted(Ns, 1e6)
            sl = np.polyfit(np.log(Ns[j:]), np.log(amin[j:]), 1)[0]
            log(f"[detect] {lab}, best focus zf = {zf:.1f} nm, h = a, {SIGMAS:.0f} sigma, {what}: "
                f"a_min(N = 1e6) = {amin[j]:.2f} nm; slope d ln a_min / d ln N on 1e6-1e12 = {sl:.4f}")
            ax1.plot(Ns, amin[j] * (Ns / 1e6) ** (-1 / 6), color=col, lw=0.8, ls=ls)
        amin0 = [detection_limit(lambda a: peak(a, 0.0), SIGMAS / np.sqrt(N), a_lo=0.05, a_hi=200) for N in (1e6, 1e8)]
        log(f"[detect] {lab}, focus on interface, pixel: a_min(1e6) = {amin0[0]:.2f} nm, "
            f"slope 1e6-1e8 = {np.log(amin0[1] / amin0[0]) / np.log(100):.4f}")
    ax1.set(xscale="log", yscale="log", xlabel=r"photons per pixel $N$", ylabel=r"smallest radius $a_{\min}$ (nm)")
    ax1.plot([], [], "o", color="k", mfc="none", label="single pixel")
    ax1.plot([], [], "s", color="k", label="PSF fit")
    ax1.plot([], [], color="k", lw=0.8, label=r"$\propto N^{-1/6}$")
    ax1.legend(loc="lower left")
    ax1.text(1e9, 5.0, "polystyrene", color=PS_COL)
    ax1.text(3e3, 1.6, "gold", color=AU_COL)
    ax1.axvline(1e6, color=GREY, lw=0.8, ls=":")

    h_p = 3.0
    zf = best_focus(h_p, LAM0, N1, N2, NA)
    er = vectorial_reference(LAM0, N1, N2, zf)
    ex1, ey1 = vectorial_psf(0.0, 0.0, h_p, protein_polarizability(1000.0, DNDC, N2), LAM0, N1, N2, NA, zf=zf)
    c_mda = contrast(ex1, er, ey1)
    ex0, ey0 = vectorial_psf(0.0, 0.0, h_p, protein_polarizability(1000.0, DNDC, N2), LAM0, N1, N2, NA)
    c_mda0 = contrast(ex0, ER, ey0)
    log(f"[protein] dn/dc = {DNDC} mL/g, h = {h_p} nm: alpha = {protein_polarizability(1000.0, DNDC, N2):.1f} nm^3/MDa; "
        f"contrast per MDa {c_mda:+.4e} at best focus zf = {zf:.1f} nm, {c_mda0:+.4e} with focus on the interface")
    rho_p = 1.35
    delta = rho_p * DNDC / N2
    ratio = 3 * clausius_mossotti(1 + delta) / (2 * delta)
    log(f"[protein] equivalent sphere (density {rho_p} g/cm^3, n = {N2 * (1 + delta):.4f}): alpha_sphere/alpha_dndc = "
        f"{ratio:.4f}; 1 - d/6 - 2d^2/9 = {1 - delta / 6 - 2 * delta**2 / 9:.4f} (d = {delta:.4f})")
    for M in (66.0, 150.0):
        n_need = (SIGMAS / gain / (abs(c_mda) * M / 1000)) ** 2
        log(f"[protein] {M:.0f} kDa at 3 sigma with a PSF fit needs N = {n_need:.2e} photons per pixel")
    for g, mk, what in ((1.0, "o", "pixel"), (gain, "s", "PSF fit")):
        mmin = SIGMAS / np.sqrt(Ns) / g / abs(c_mda) * 1000
        ax2.plot(Ns, mmin, mk, color=GREEN, mfc="none" if g == 1 else GREEN, ms=3.5, label=what)
        for N in (1e6, 1e8, 1e10):
            log(f"[protein] {what}, N = {N:.0e}: smallest detectable mass {SIGMAS / np.sqrt(N) / g / abs(c_mda) * 1000:.1f} kDa")
    ax2.plot(Ns, SIGMAS / np.sqrt(Ns) / gain / abs(c_mda) * 1000, color=GREEN, lw=0.8)
    for M, lab in ((66, "BSA 66 kDa"), (150, "IgG 150 kDa")):
        ax2.axhline(M, color=GREY, lw=0.8, ls="--")
        ax2.text(1.5e3, M * 1.12, lab, fontsize=9)
    ax2.set(xscale="log", yscale="log", xlabel=r"photons per pixel $N$", ylabel="smallest mass (kDa)")
    ax2.legend(loc="upper right")
    panel(ax1, "a")
    panel(ax2, "b")
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig6_detection.png")


def fig_focus_reference():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE)
    zf = np.linspace(-400, 500, 361)
    for h, col in ((3.0, GREEN), (30.0, PS_COL), (100.0, PURPLE)):
        s = np.array([vectorial_psf(0.0, 0.0, h, 1.0, LAM0, N1, N2, NA, zf=z)[0] / vectorial_reference(LAM0, N1, N2, z)
                      for z in zf])
        ax1.plot(zf, 2 * s.real / np.max(np.abs(2 * s.real)), color=col, label=rf"$h={h:.0f}$ nm")
        zb = best_focus(h, LAM0, N1, N2, NA)
        i0 = np.argmin(np.abs(zf))
        log(f"[focus] h = {h:.0f} nm: best focus zf = {zb:.1f} nm; |interference| at best / at zf = 0: "
            f"{abs((vectorial_psf(0.0, 0.0, h, 1.0, LAM0, N1, N2, NA, zf=zb)[0] / vectorial_reference(LAM0, N1, N2, zb)).real / s[i0].real):.2f}")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.axvline(0, color=GREY, lw=0.8, ls=":")
    ax1.set(xlabel=r"focal plane $z_f$ (nm)", ylabel="interference term (normalised)", xlim=(-400, 500))
    ax1.legend(loc="upper right")

    rng = np.random.default_rng(7)
    s = -2e-4
    n_inc = 1e9
    n_full = n_inc * R**2
    att = np.geomspace(1, 30, 8)
    aa = np.geomspace(1, 30, 100)
    M = 3000
    cases = ((False, PS_COL, "o", "fixed illumination"), (True, ORANGE, "s", "fixed counts (full well)"))
    for full_well, col, mk, lab in cases:
        theo_fn = lambda A: snr_single_pixel(s, R / A, n_full / (R / A) ** 2 if full_well else n_inc)
        snr, err = [], []
        for A in att:
            r = R / A
            n_bg = n_full if full_well else n_inc * r**2
            Cs = ratiometric_contrast(rng, np.full(M, abs(1 + s / r) ** 2), 1.0, n_bg)
            v = abs(Cs.mean() / Cs.std(ddof=1))
            snr.append(v)
            err.append(np.sqrt((1 + v**2 / 2) / M))
        ax2.errorbar(att, snr, err, fmt=mk, color=col, mfc="none", capsize=2, label=lab)
        ax2.plot(aa, theo_fn(aa), color=col, lw=1)
        theo = theo_fn(att)
        log(f"[reference] {lab}: SNR at attenuation 1, {att[-1]:.0f}: {snr[0]:.2f} +- {err[0]:.2f}, "
            f"{snr[-1]:.2f} +- {err[-1]:.2f} (exact theory {theo[0]:.2f}, {theo[-1]:.2f}; "
            f"weak-scatterer limit 2|s|sqrt(N_inc) = {2 * abs(s) * np.sqrt(n_inc):.2f} at attenuation 1)")
    ax2.plot(aa, 2 * abs(s) * np.sqrt(n_inc) * np.ones_like(aa), color=GREY, lw=0.8, ls=":")
    ax2.text(1.05, 2 * abs(s) * np.sqrt(n_inc) * 0.93, r"$2|s|\sqrt{N_{\rm inc}}$", ha="left", va="top", fontsize=9)
    ax2.set(xscale="log", yscale="log", xlabel=r"reference attenuation $|r_0/r|$", ylabel="single-pixel SNR",
            ylim=(8, 600))
    ax2.legend(loc="upper left")
    panel(ax1, "a")
    panel(ax2, "b")
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig7_focus_reference.png")


def check_convergence():
    """Measured convergence orders of the extension discretisations, next to the predicted ones."""
    x, y = np.meshgrid(np.linspace(-800, 800, 25), np.linspace(-790, 810, 25))
    ex = vectorial_psf(x, y, 30.0, 1.0, LAM0, N1, N2, NA)[0]
    steps = np.array([4.0, 2.0, 1.0, 0.5])
    err = [np.abs(DipoleImager(LAM0, N1, N2, NA, rho_max=1200.0, d_rho=d).field(x, y, 30.0, 1.0)[0] - ex).max()
           / np.abs(ex).max() for d in steps]
    order = np.polyfit(np.log(steps[1:]), np.log(err[1:]), 1)[0]
    log(f"[conv] DipoleImager interpolation, max error / peak at d_rho = 4, 2, 1, 0.5 nm: "
        + ", ".join(f"{e:.1e}" for e in err) + f"; order (2-0.5 nm) {order:.2f}, predicted 2")
    dx, alpha = 80.0, ((N1 / N2) ** 2 - 1) * 80.0**2
    direct = vectorial_psf(0.0, 0.0, 0.0, alpha, LAM0, N1, N2, NA, order=600)[0]
    ns = np.array([64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024])
    werr = []
    for n in ns:
        z = np.zeros((n, n))
        z[0, 0] = 1.0
        werr.append(abs(roughness_field(z, dx, LAM0, N1, N2, NA)[0, 0] - direct) / abs(direct))
    env = [max(werr[i:]) for i in range(len(werr))]
    slope = np.polyfit(np.log(ns * dx), np.log(env), 1)[0]
    log(f"[conv] roughness FFT box L = 5-82 um: wrap error at the bump {werr[0]:.1e} -> {werr[-1]:.1e} (not monotonic: "
        f"the image lattice sum converges only by oscillation); envelope slope {slope:.2f}, predicted -1.5")
    zf = 215.5
    er = vectorial_reference(LAM0, N1, N2, zf)
    exact = speckle_contrast_std(ROUGH_SIGMA, ROUGH_ELL, LAM0, N1, N2, NA, zf) ** 2
    ns = np.array([64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048])
    serr = []
    for n in ns:
        k = 2 * np.pi * np.fft.fftfreq(n, dx)
        kx, ky = np.meshgrid(k, k)
        G = ((N1 / N2) ** 2 - 1) * vectorial_transfer(kx, ky, 0.0, LAM0, N1, N2, NA, zf)[0] / er
        var = 4 * np.sum(gaussian_psd(np.hypot(kx, ky), ROUGH_SIGMA, ROUGH_ELL) * G.real**2) / (n * dx) ** 2
        serr.append(abs(var / exact - 1))
    dk = 2 * np.pi / (ns * dx)
    env = [max(serr[i:]) for i in range(len(serr))]
    slope = np.polyfit(np.log(dk), np.log(env), 1)[0]
    log(f"[conv] speckle variance, FFT-grid sum vs radial integral, dkappa/kappa_NA = {dk[0] / (NA * 2 * np.pi / LAM0):.3f}"
        f"-{dk[-1] / (NA * 2 * np.pi / LAM0):.4f}: relative error {serr[0]:.1e} -> {serr[-1]:.1e}; envelope slope "
        f"{slope:.2f}, predicted 1.5 (lattice points in a hard-edged disc); radial quadrature (400 nodes) vs 2000: "
        f"{bound(abs(speckle_contrast_std(ROUGH_SIGMA, ROUGH_ELL, LAM0, N1, N2, NA, zf) ** 2 / speckle_contrast_std(ROUGH_SIGMA, ROUGH_ELL, LAM0, N1, N2, NA, zf, order=2000) ** 2 - 1))}")


def fig_tracking():
    """3D localisation of a 10 nm gold particle."""
    a, N = 10.0, 1e4
    alpha = mie_dipole_polarizability(a, M_AU, K)
    x, y = camera_grid(15, PIXEL)
    imager = DipoleImager(LAM0, N1, N2, NA, rho_max=900.0, d_rho=1.0, order=100)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE)

    model = ParticleModel(imager, x, y, alpha, LAM0, N1, N2)
    hs = np.arange(12.0, 322.0, 2.0)
    b = np.array([crlb(fisher_information(model, [13.0, -7.0, h, 1.0], N)) for h in hs])
    b3 = np.array([crlb(fisher_information(model, [13.0, -7.0, h, 1.0], N, free=(0, 1, 2))) for h in hs])
    c0 = np.array([model([0.0, 0.0, h, 1.0])[7, 7] - 1 for h in hs])
    for v, col, lab in ((b[:, 0], BLUE, r"$\sigma_x$"), (b[:, 1], GREEN, r"$\sigma_y$"), (b[:, 2], ORANGE, r"$\sigma_z$")):
        ax1.plot(hs, v, color=col, label=lab)
    ax1.plot(hs, b3[:, 2], color=ORANGE, ls="--", lw=1, label=r"$\sigma_z$, amplitude known")
    zeros = [hs[j] - c0[j] * (hs[j + 1] - hs[j]) / (c0[j + 1] - c0[j]) for j in np.nonzero(np.diff(np.sign(c0)))[0]]
    for z0 in zeros:
        ax1.axvline(z0, color=GREY, lw=0.8, ls=":")
    ax1.set(yscale="log", xlabel=r"height $h$ (nm)", ylabel=r"Cramér–Rao bound (nm)", xlim=(0, 320), ylim=(0.4, 80))
    ax1.legend(loc="upper right", ncol=2)
    ax1.text(0.03, 0.05, rf"gold $a={a:.0f}$ nm, $N=10^4$", transform=ax1.transAxes, fontsize=9)
    for h in (25.0, 90.0):
        i = np.argmin(np.abs(hs - h))
        log(f"[track] gold a = {a} nm, N = {N:.0e}, zf = 0, 15x15 pixels: CRLB at h = {hs[i]:.0f} nm: "
            f"x {b[i, 0]:.2f}, y {b[i, 1]:.2f}, z {b[i, 2]:.2f} nm (z with amplitude known {b3[i, 2]:.2f}); "
            f"centre contrast {c0[i]:+.4f}")
    log(f"[track] contrast zero crossings at h = " + ", ".join(f"{z:.1f}" for z in zeros)
        + f" nm; sigma_z smallest at h = {hs[np.argmin(b[:, 2])]:.0f} nm ({b[:, 2].min():.2f} nm), "
        f"sigma_x largest at h = {hs[np.argmax(b[:, 0])]:.0f} nm ({b[:, 0].max():.2f} nm)")

    rng = np.random.default_rng(21)
    p = np.array([13.0, -7.0, 60.0, 1.0])
    img = ratiometric_contrast(rng, model(p), 1.0, N)
    chi2 = lambda q: float(np.sum((img + 1 - model(q)) ** 2 * N / model(q)))
    good = localise(img, model, p, N, free=(0, 1, 2))
    wrong = localise(img, model, p + [0, 0, 240.0, 0], N, free=(0, 1, 2))
    log(f"[track] fringe ambiguity at h = 60 nm: fit started at h + 240 nm converges to h = {wrong[2]:.1f} nm, "
        f"chi^2 = {chi2(wrong):.0f} vs {chi2(good):.0f} at the true fringe ({x.size} pixels)")

    for h_test in (25.0, 60.0):
        pt_ = np.array([13.0, -7.0, h_test, 1.0])
        I_ = model(pt_)
        fits = np.array([localise(ratiometric_contrast(rng, I_, 1.0, N), model, pt_ + rng.normal(0, [5, 5, 3, 0.03]), N)
                         for _ in range(200)])
        bnd = crlb(fisher_information(model, pt_, N))
        log(f"[track] 200 fits at h = {h_test:.0f} nm, all four parameters free: std x, y, z = "
            f"{fits[:, 0].std(ddof=1):.2f}, {fits[:, 1].std(ddof=1):.2f}, {fits[:, 2].std(ddof=1):.2f} nm "
            f"(CRLB {bnd[0]:.2f}, {bnd[1]:.2f}, {bnd[2]:.2f}; standard error of a std 5%); "
            f"bias in h {fits[:, 2].mean() - h_test:+.2f} nm (standard error {fits[:, 2].std(ddof=1) / np.sqrt(200):.2f})")

    D, dt, n = 5e5, 2e-5, 400
    steps = np.diff(brownian_track(rng, n, D, dt, dim=3), axis=0)
    pos = [np.array([0.0, 0.0, 100.0])]
    for st in steps:
        q = pos[-1] + st
        q[2] = q[2] if q[2] > a else 2 * a - q[2]
        pos.append(q)
    pos = np.array(pos)
    est, bounds, guess = [], [], pos[0].copy()
    for q in pos:
        cx, cy = np.round(guess[:2] / PIXEL) * PIXEL
        mdl = ParticleModel(imager, x + cx, y + cy, alpha, LAM0, N1, N2)
        img = ratiometric_contrast(rng, mdl([*q, 1.0]), 1.0, N)
        guess = localise(img, mdl, [*guess, 1.0], N, free=(0, 1, 2))[:3]
        est.append(guess)
        bounds.append(crlb(fisher_information(mdl, [*q, 1.0], N, free=(0, 1, 2))))
    est, bounds = np.array(est), np.array(bounds)
    err = est - pos
    rms_b = np.sqrt(np.mean(bounds**2, axis=0))
    log(f"[track] diffusing gold a = {a} nm, D = {D / 1e6:.2f} um^2/s, dt = {dt * 1e6:.0f} us, {n} frames, "
        f"N = {N:.0e}, amplitude known: z range {pos[:, 2].min():.0f}-{pos[:, 2].max():.0f} nm")
    log(f"[track] error std x, y, z = {err[:, 0].std():.2f}, {err[:, 1].std():.2f}, {err[:, 2].std():.2f} nm; "
        f"rms CRLB {rms_b[0]:.2f}, {rms_b[1]:.2f}, {rms_b[2]:.2f} nm; max |error| {np.abs(err).max():.1f} nm")
    sims = np.array([brownian_track(rng, n, D, dt, dim=1) for _ in range(2000)])
    d_sim, _ = fit_msd(msd(sims + rng.normal(0, rms_b[2], sims.shape), 2), dt)
    msd_sd = {}
    for axes, sig in (((0, 1), np.sqrt(np.mean(rms_b[:2] ** 2))), ((2,), rms_b[2])):
        sim = np.array([brownian_track(rng, n, D, dt, dim=len(axes)) for _ in range(500)])
        msd_sd[axes] = msd(sim + rng.normal(0, sig, sim.shape), 10).std(axis=0, ddof=1)
    log(f"[track] expected scatter of a 2-lag D estimate from {n} frames, one axis (2000 simulated tracks): "
        f"{d_sim.std() / D * 100:.0f}%")
    lags = np.arange(1, 11)
    for axes, col, lab in (((0, 1), BLUE, "lateral"), ((2,), ORANGE, "axial")):
        m_est = msd(est[:, axes], 10)
        m_true = msd(pos[:, axes], 10)
        d_est, s2 = fit_msd(m_est, dt)
        d_true, _ = fit_msd(m_true, dt)
        ax2.errorbar(lags * dt * 1e3, m_est, msd_sd[axes], fmt="o", color=col, mfc="none", capsize=2,
                     label=f"{lab}, tracked")
        ax2.plot(lags * dt * 1e3, m_true, "-", color=col, lw=1, label=f"{lab}, true")
        tt = np.linspace(0, 10, 50) * dt
        ax2.plot(tt * 1e3, 2 * d_est * tt + 2 * s2, color=col, lw=0.8, ls="--")
        exp_s2 = np.mean(bounds[:, axes] ** 2)
        log(f"[track] {lab} MSD fit (2 lags): D = {d_est / 1e6:.3f} um^2/s (true track {d_true / 1e6:.3f}, input "
            f"{D / 1e6:.3f}); sigma = {np.sqrt(max(s2, 0)):.2f} nm (rms CRLB {np.sqrt(exp_s2):.2f})")
    ax2.set(xlabel=r"lag $\tau$ (ms)", ylabel=r"MSD per axis (nm$^2$)", xlim=(0, 0.21), ylim=(0, None))
    ax2.legend(loc="upper left")
    panel(ax1, "a")
    panel(ax2, "b")
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig8_tracking.png")


def peak_contrast_mie(a, m, zf):
    ex, ey = vectorial_psf(0.0, 0.0, a, mie_dipole_polarizability(a, m, K), LAM0, N1, N2, NA, zf=zf)
    return contrast(ex, vectorial_reference(LAM0, N1, N2, zf), ey)


def fig_roughness():
    """Static speckle from the coverslip."""
    fig = plt.figure(figsize=(9.2, 3.3))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.06, 0.62, 1.3], wspace=0.08)
    ax1, ax2, cax, ax3 = (fig.add_subplot(gs[0, i]) for i in (0, 1, 2, 4))
    zf = best_focus(5.0, LAM0, N1, N2, NA, alpha_phase=np.angle(clausius_mossotti(M_PS)))
    er = vectorial_reference(LAM0, N1, N2, zf)
    rng = np.random.default_rng(31)
    n, dx = 1024, 20.0
    z = rough_surface(rng, n, dx, ROUGH_SIGMA, ROUGH_ELL)
    e = roughness_field(z, dx, LAM0, N1, N2, NA, zf)
    sig_th = speckle_contrast_std(ROUGH_SIGMA, ROUGH_ELL, LAM0, N1, N2, NA, zf)
    sig_mc = (2 * (e / er).real).std()
    log(f"[rough] sigma = {ROUGH_SIGMA} nm, ell = {ROUGH_ELL} nm, focus zf = {zf:.1f} nm (polystyrene best): "
        f"speckle contrast rms {sig_mc:.3e} (20 um map), spectral integral {sig_th:.3e}")
    a = 12.0
    x, y = camera_grid(n, dx)
    c = slice(n // 2 - 75, n // 2 + 75)
    imager = DipoleImager(LAM0, N1, N2, NA, zf=zf, rho_max=2200.0, d_rho=0.5)
    es, ey = imager.field(x[c, c], y[c, c], a, mie_dipole_polarizability(a, M_PS, K))
    ec = e[c, c]
    raw = (np.abs(er + ec + es) ** 2 + np.abs(ey) ** 2) / abs(er) ** 2 - 1
    ratio = (np.abs(er + ec + es) ** 2 + np.abs(ey) ** 2) / np.abs(er + ec) ** 2 - 1
    smooth = contrast(es, er, ey)
    log(f"[rough] polystyrene a = {a} nm on the rough glass: peak contrast {smooth[75, 75]:+.3e}; raw image rms "
        f"{raw.std():.3e}; ratiometric image max |deviation from smooth glass| {np.abs(ratio - smooth).max():.1e}")
    ext = [-1.5, 1.5, -1.5, 1.5]
    lim = 3 * sig_mc * 1e3
    for ax, im, note in ((ax1, raw, "flat field only"), (ax2, ratio, "ratio to particle-free frame")):
        h_im = ax.imshow(im * 1e3, extent=ext, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set(xlabel=r"$x$ ($\mu$m)", yticks=[-1, 0, 1])
        ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top", fontsize=8.5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax1.set_ylabel(r"$y$ ($\mu$m)")
    ax2.set_yticklabels([])
    cb = fig.colorbar(h_im, cax=cax)
    cb.set_label(r"contrast ($10^{-3}$)")
    for ax in (ax1, ax2):
        ax.plot([0.9, 1.4], [-1.3, -1.3], color="k", lw=2)
    ax1.text(1.15, -1.22, "500 nm", ha="center", fontsize=8)

    ells = np.geomspace(1, 3000, 80)
    for zz, col, lab in ((0.0, PS_COL, r"$z_f=0$"), (zf, ORANGE, rf"$z_f={zf:.0f}$ nm")):
        ax3.plot(ells, [speckle_contrast_std(ROUGH_SIGMA, ell, LAM0, N1, N2, NA, zz) for ell in ells], color=col, label=lab)
        mc, se = [], []
        for ell in (5.0, 20.0, 100.0, 400.0):
            vals = []
            for _ in range(3):
                zz_map = rough_surface(rng, 1024, 40.0, ROUGH_SIGMA, ell)
                field = roughness_field(zz_map, 40.0, LAM0, N1, N2, NA, zz) / vectorial_reference(LAM0, N1, N2, zz)
                vals.append((2 * field.real).std())
            mc.append(np.mean(vals))
            se.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ax3.errorbar([5, 20, 100, 400], mc, se, fmt="o", color=col, mfc="none", capsize=2)
        th = [speckle_contrast_std(ROUGH_SIGMA, ell, LAM0, N1, N2, NA, zz) for ell in (5.0, 20.0, 100.0, 400.0)]
        log(f"[rough] zf = {zz:.1f} nm, Monte Carlo (3 maps of 41 um) / integral at ell = 5, 20, 100, 400 nm: "
            + ", ".join(f"{m / t:.4f}({s / t * 1e4:.0f})" for m, s, t in zip(mc, se, th))
            + f"; largest |deviation| {max(abs(m - t) / s for m, s, t in zip(mc, se, th)):.2f} standard errors")
    s12 = [speckle_contrast_std(ROUGH_SIGMA, ell, LAM0, N1, N2, 1.2, zf) for ell in ells]
    ax3.plot(ells, s12, color=GREEN, lw=1, ls="--", label=rf"NA 1.2, $z_f={zf:.0f}$ nm")
    ax3.set(xscale="log", yscale="log", xlabel=r"correlation length $\ell$ (nm)", ylabel=r"speckle contrast rms",
            ylim=(1e-5, 4e-2))
    ax3.legend(loc="lower center", fontsize=8.5, bbox_to_anchor=(0.42, 0.0))
    ax3.text(0.03, 0.93, rf"$\sigma_\zeta={ROUGH_SIGMA}$ nm", transform=ax3.transAxes, fontsize=9)
    for zz in (0.0, zf):
        sm = [speckle_contrast_std(ROUGH_SIGMA, ell, LAM0, N1, N2, NA, zz) for ell in ells]
        log(f"[rough] zf = {zz:.1f} nm: speckle rms peaks at ell = {ells[np.argmax(sm)]:.0f} nm ({max(sm):.3e}); "
            f"NA <= n_water at zf = 0 gives {bound(speckle_contrast_std(ROUGH_SIGMA, ROUGH_ELL, LAM0, N1, N2, 1.33, 0.0))}")

    for m, lab in ((M_PS, "polystyrene"), (M_AU, "gold")):
        zb = best_focus(5.0, LAM0, N1, N2, NA, alpha_phase=np.angle(clausius_mossotti(m)))
        for zz, what in ((0.0, "focus on interface"), (zb, f"best focus zf = {zb:.1f} nm")):
            sb = speckle_contrast_std(ROUGH_SIGMA, ROUGH_ELL, LAM0, N1, N2, NA, zz)
            floor = detection_limit(lambda a, zz=zz: peak_contrast_mie(a, m, zz), SIGMAS * sb, 0.05, 200)
            log(f"[rough] {lab}, {what}: speckle rms {sb:.3e}; 3-sigma floor without background removal a = {floor:.2f} nm")
    panel(ax1, "a")
    panel(ax2, "b")
    panel(ax3, "c")
    save(fig, "fig9_roughness.png")


def lens_fields(a, zf, power, gamma, point=False):
    """(plus, minus) sideband fields of the thermal lens at the PSF centre."""
    if point:
        da = lambda P, g: lens_polarizability(4 * K**2, P, a, g)
        return (peak_field(a, da(power, gamma), zf), peak_field(a, da(np.conj(power), np.conj(gamma)), zf))
    f = lambda P, g: vectorial_psf(0.0, 0.0, a, 1.0, LAM0, N1, N2, NA, zf, form_factor=lambda q2: lens_polarizability(q2, P, a, g))
    return f(power, gamma), f(np.conj(power), np.conj(gamma))


def pt_contrast(a, zf, pump=PUMP, f_mod=F_MOD, point=False):
    P = absorption_cross_section(a, M_AU, K) * pump
    g = thermal_gamma(2 * np.pi * f_mod)
    es, ey = peak_field(a, mie_dipole_polarizability(a, M_AU, K), zf)
    plus, minus = lens_fields(a, zf, P, g, point)
    return photothermal_contrast(es, ey, plus, minus, vectorial_reference(LAM0, N1, N2, zf))


def fig_photothermal():
    """Photothermal readout of gold."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE)
    a = 10.0
    f = np.geomspace(1e2, 1e9, 300)
    g = thermal_gamma(2 * np.pi * f)
    da = lambda q2, g: lens_polarizability(q2, 1e-6, a, g)
    ref = da(4 * K**2, 0.0)
    back = da(4 * K**2, g) / ref
    fwd = da(0.0, g) / ref
    edge = da(2 * K**2, g) / da(2 * K**2, 0.0)
    ax1.plot(f, np.abs(back), color=BLUE, label=r"reflection, $q=2k$")
    ax1.plot(f, np.abs(edge), color=GREEN, lw=1, ls="--", label=r"$q=\sqrt{2}k$ (pupil edge)")
    ax1.plot(f, np.abs(fwd), color=ORANGE, label=r"transmission, $q=0$")
    ax1.axvline(F_MOD, color=GREY, lw=0.8, ls=":")
    ax1.set(xscale="log", yscale="log", xlabel=r"modulation frequency $\Omega/2\pi$ (Hz)",
            ylabel=r"$|\delta\alpha(\Omega)|\,/\,|\delta\alpha_{q=2k}(0)|$", ylim=(1e-3, 1e5))
    ax1.legend(loc="upper right")
    i_half = np.argmin(np.abs(np.abs(back) - 0.5))
    gm = thermal_gamma(2 * np.pi * F_MOD)
    i_1pc = np.argmax(np.abs(np.abs(back) - 1) > 0.01)
    log(f"[pt] |delta alpha(q=2k)| departs from steady state by 1% at {f[i_1pc]:.2e} Hz; "
        f"arg L_Au = {np.angle(clausius_mossotti(M_AU)):.4f} rad")
    log(f"[pt] thermal lens, a = {a} nm, fixed absorbed power: |delta alpha(q=2k)| halves at {f[i_half]:.2e} Hz "
        f"(D q^2 / 2 pi with q = 2k: {1.43e-7 * 1e18 * 4 * K**2 / (2 * np.pi):.2e} Hz); at {F_MOD:.0e} Hz: "
        f"|delta alpha / steady| = {abs(da(4 * K**2, gm) / ref):.5f}, phase {np.angle(da(4 * K**2, gm) / ref):+.4f} rad; "
        f"forward (q = 0) / back = {abs(da(0.0, gm) / ref):.1f}; steady pupil edge (q = sqrt2 k) / back = "
        f"{(da(2 * K**2, 0.0) / ref).real:.3f}")

    for aa in (5.0, 10.0, 20.0):
        sa = absorption_cross_section(aa, M_AU, K)
        dT = sa * PUMP / (4 * np.pi * 0.6 * aa * 1e-9)
        da = lens_polarizability(4 * K**2, sa * PUMP, aa, 0.0)
        log(f"[pt] gold a = {aa:.0f} nm: sigma_abs = {sa:.1f} nm^2; steady Delta T(a) at 100 kW/cm^2 = {dT:.2f} K; "
            f"delta alpha(2k)/alpha = {da / mie_dipole_polarizability(aa, M_AU, K):+.3e}")
    zgrid = np.linspace(-400, 500, 181)
    dcz = np.array([abs(pt_contrast(5.0, z)) for z in zgrid])
    zf_pt = zgrid[np.argmax(dcz)]
    zf_st = best_focus(5.0, LAM0, N1, N2, NA, alpha_phase=np.angle(clausius_mossotti(M_AU)))
    log(f"[pt] gold a = 5 nm: |dC| largest at zf = {zf_pt:.0f} nm ({dcz.max():.3e}); at zf = 0: "
        f"{abs(pt_contrast(5.0, 0.0)):.3e}; at the static best focus {zf_st:.1f} nm: {abs(pt_contrast(5.0, zf_st)):.3e}")
    for aa in (5.0, 10.0):
        full, point = pt_contrast(aa, zf_pt), pt_contrast(aa, zf_pt, point=True)
        log(f"[pt] a = {aa:.0f} nm, zf = {zf_pt:.0f} nm, {F_MOD:.0e} Hz: dC = {full:.3e} (q-dependent lens), "
            f"{point:.3e} (point dipole at q = 2k): ratio {abs(full / point):.3f}")

    Ns = np.logspace(4, 14, 21)
    sb = speckle_contrast_std(ROUGH_SIGMA, ROUGH_ELL, LAM0, N1, N2, NA, zf_st)
    static = lambda a: peak_contrast_mie(a, M_AU, zf_st)
    a_shot = [detection_limit(static, SIGMAS / np.sqrt(N), 0.05, 200) for N in Ns]
    a_sp = [detection_limit(static, SIGMAS * np.sqrt(1 / N + sb**2), 0.05, 200) for N in Ns]
    floor = detection_limit(static, SIGMAS * sb, 0.05, 200)
    ax2.plot(Ns, a_shot, color=AU_COL, lw=1, ls=":", label="iSCAT, shot noise only")
    ax2.plot(Ns, a_sp, color=AU_COL, label="iSCAT + coverslip speckle")
    ax2.axhline(floor, color=GREY, lw=0.8, ls="--")
    for pump, col, ls, lab in ((PUMP, PURPLE, "-", r"photothermal, 100 kW/cm$^2$"),
                               (10 * PUMP, PURPLE, "--", r"photothermal, 1 MW/cm$^2$")):
        fn = lambda a, pump=pump: abs(pt_contrast(a, zf_pt, pump=pump))
        amin = []
        for N in Ns:
            thr = SIGMAS * np.sqrt(2 / N)
            amin.append(detection_limit(fn, thr, 0.2, 20.0) if fn(20.0) > thr else np.nan)
        amin = np.array(amin)
        ax2.plot(Ns, amin, color=col, ls=ls, label=lab)
        ok = np.isfinite(amin)
        below = np.nonzero(ok & (amin < floor))[0]
        n_cross = Ns[below[0]] if len(below) else np.nan
        log(f"[pt] {lab.split(', ')[1].replace('$^2$', '^2')}, {F_MOD:.0e} Hz, zf = {zf_pt:.0f} nm, lock-in noise "
            f"sqrt(2/N): a_min(1e10) = {amin[np.argmin(np.abs(Ns - 1e10))]:.2f} nm, a_min(1e12) = "
            f"{amin[np.argmin(np.abs(Ns - 1e12))]:.2f} nm; beats the speckle floor ({floor:.2f} nm) from N = {n_cross:.0e}")
    log(f"[pt] iSCAT gold, best focus {zf_st:.1f} nm, speckle rms {sb:.3e}: a_min(1e10) = "
        f"{a_sp[np.argmin(np.abs(Ns - 1e10))]:.2f} nm (shot noise alone {a_shot[np.argmin(np.abs(Ns - 1e10))]:.2f} nm)")
    ax2.set(xscale="log", yscale="log", xlabel=r"photons per pixel $N$", ylabel=r"smallest gold radius $a_{\min}$ (nm)",
            ylim=(0.2, 60))
    ax2.legend(loc="upper right", fontsize=8.5)
    panel(ax1, "a")
    panel(ax2, "b")
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig10_photothermal.png")


def main():
    t0 = time.time()
    FIG.mkdir(exist_ok=True)
    environment_header()
    check_mie()
    fig_mie()
    fig_radius()
    fig_height()
    fig_psf()
    T = fig_image()
    fig_detection(T)
    fig_focus_reference()
    check_convergence()
    fig_tracking()
    fig_roughness()
    fig_photothermal()
    print(f"total {time.time() - t0:.1f} s")
    (FIG / "results.txt").write_text("\n".join(results) + "\n")


if __name__ == "__main__":
    main()
