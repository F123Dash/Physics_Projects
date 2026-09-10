"""
FDTD 3D - validation and result figures.

Reads the CSVs written by 3D.cpp, cavity.cpp and cpml_test.cpp (any missing file is
skipped) and writes PNG + PDF figures to results_outputs/:
  fig1_geometry    TFSF / CPML / flux-surface layout of the Mie and GRIN runs
  fig2_mie         Q_sca of the εr = 4 sphere: FDTD (each resolution) vs Mie theory
  fig3_grin        |Ez/E0|² at 75 GHz around the GRIN sphere (z = centre plane)
  fig4_cavity      PEC-cavity resonances vs discrete-Yee and continuum frequencies
  fig5_cpml        CPML reflection error vs σ_max
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_outputs")
OUT = os.path.join(HERE, "results_outputs")
C0 = 2.99792458e8

# Categorical slots 1-3 (validated: CVD-safe on white; aqua < 3:1 so every series
# also carries its own marker shape and a legend entry)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [(BLUE, "o"), (ORANGE, "s"), (AQUA, "^")]
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"
# Sequential single-hue ramp (light → dark blue) for intensity maps
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
    "lines.linewidth": 2.0, "lines.markersize": 6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        print(f"  [skip] {name} not found", file=sys.stderr)
        return None
    return pd.read_csv(path)


def meta(mode):
    df = load(f"3D_{mode}_meta.csv")
    return None if df is None else dict(zip(df["key"], df["value"].astype(float)))


def savefig(fig, name):
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, name)
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> results_outputs/{name}.png + .pdf")


# Fig 1 — geometry (x-y cut through the sphere centre)
def plot_geometry():
    setups = [(m, meta(m)) for m in ("mie", "grin")]
    setups = [(m, d) for m, d in setups if d is not None]
    if not setups:
        return
    fig, axes = plt.subplots(1, len(setups), figsize=(5.2 * len(setups), 4.6), squeeze=False)
    for ax, (mode, d) in zip(axes[0], setups):
        nx, ny, npml = d["nx"], d["ny"], d["npml"]
        ax.add_patch(mpatches.Rectangle((0, 0), nx, ny, facecolor="#ecebe7", edgecolor=INK2, lw=1))
        ax.add_patch(mpatches.Rectangle((npml, npml), nx - 2 * npml, ny - 2 * npml,
                                        facecolor="white", edgecolor=INK2, lw=0.8))
        ax.add_patch(mpatches.Rectangle((d["flux_x0"], d["flux_y0"]), d["flux_x1"] - d["flux_x0"],
                                        d["flux_y1"] - d["flux_y0"], fill=False, edgecolor=ORANGE,
                                        lw=1.2, ls=":"))
        ax.add_patch(mpatches.Rectangle((d["i0"], d["j0"]), d["i1"] - d["i0"], d["j1"] - d["j0"],
                                        fill=False, edgecolor=BLUE, lw=1.5, ls="--"))
        ax.add_patch(mpatches.Circle((d["cx"], d["cy"]), d["R_cells"], facecolor="#cde2fb",
                                     edgecolor=BLUE, lw=1.5))
        ax.annotate("", xy=(d["i0"] + 14, d["j1"] - 6), xytext=(d["i0"] + 2, d["j1"] - 6),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
        ax.text(d["i0"] + 2, d["j1"] - 3, "k (+x), E ∥ z", fontsize=8, color=INK)
        ax.text(npml / 2, ny / 2, "CPML", rotation=90, ha="center", va="center", fontsize=8, color=INK2)
        ax.text(d["i0"] + 1, d["j0"] + 1.5, "TFSF box", fontsize=8, color=BLUE)
        ax.text(d["flux_x1"] - 1, d["flux_y1"] + 1, "flux surface", fontsize=8, color=ORANGE, ha="right")
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("x (cells)")
        ax.set_ylabel("y (cells)")
        title = ("Mie test: εr = 4 sphere" if mode == "mie" else "GRIN sphere εr = 1 + 3(1 − r/R)")
        ax.set_title(f"{title}\n{int(nx)}×{int(ny)}×{int(d['nz'])} cells, Δ = {d['dx']*1e6:.0f} µm, "
                     f"R = {d['R_cells']:.0f} cells", fontsize=10)
    fig.tight_layout()
    savefig(fig, "fig1_geometry")


# Fig 2 — Mie validation
def plot_mie():
    files = sorted(glob.glob(os.path.join(DATA, "mie_qsca_R*.csv")),
                   key=lambda p: int(re.search(r"R(\d+)", p).group(1)))
    curve = load("mie_curve.csv")
    if not files or curve is None:
        return
    fig, (ax, axe) = plt.subplots(2, 1, figsize=(7, 6.2), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2, 1]))
    xlim = None
    ax.plot(curve["size_param"], curve["Q_mie"], color=INK2, lw=1.5, label="Mie theory")
    for (col, mk), path in zip(SERIES, files):
        R = int(re.search(r"R(\d+)", path).group(1))
        d = pd.read_csv(path)
        lab = f"FDTD, R = {R} cells"
        ax.plot(d["size_param"], d["Q_fdtd"], mk, color=col, ms=6, mec="white", mew=0.8, label=lab)
        mean = 100 * d["rel_err"].abs().mean()
        axe.plot(d["size_param"], 100 * d["rel_err"], "-" + mk, color=col, lw=1.5, ms=5,
                 mec="white", mew=0.8, label=f"R = {R}: mean |err| {mean:.1f} %")
        xlim = (d["size_param"].min() - 0.05, d["size_param"].max() + 0.05)
    ax.set_xlim(*xlim)
    ax.set_ylabel("Q_sca = C_sca / πa²")
    ax.set_title("Scattering efficiency of a dielectric sphere (εr = 4, a = 1.4 mm), FDTD vs Mie")
    ax.legend(loc="upper left")
    axe.axhline(0, color=INK2, lw=0.8)
    axe.set_xlabel("size parameter x = ka")
    axe.set_ylabel("FDTD / Mie − 1  (%)")
    axe.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig2_mie")


# Fig 3 — GRIN focusing map
def plot_grin():
    df, d = load("grin_slice.csv"), meta("grin")
    if df is None or d is None:
        return
    nx, ny, npml = int(df["x_cell"].max()) + 1, int(df["y_cell"].max()) + 1, int(d["npml"])
    I = np.zeros((ny, nx))
    I[df["y_cell"].values, df["x_cell"].values] = df["intensity"].values
    lam_cells = C0 / d["f_map_Hz"] / d["dx"]
    x = (np.arange(nx) - d["cx"]) / lam_cells
    y = (np.arange(ny) - d["cy"]) / lam_cells
    sl = (slice(npml, ny - npml), slice(npml, nx - npml))   # hide the CPML
    Rl = d["R_cells"] / lam_cells

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1.6, 1]))
    im = ax.imshow(I[sl], origin="lower", cmap=SEQ, vmin=0,
                   extent=[x[npml], x[nx - npml - 1], y[npml], y[ny - npml - 1]], aspect="equal")
    ax.add_patch(mpatches.Circle((0, 0), Rl, fill=False, edgecolor="white", lw=1.2, ls="--"))
    iy, ix = np.unravel_index(np.argmax(I[sl]), I[sl].shape)
    fx, fy = x[ix + npml], y[iy + npml]
    ax.plot(fx, fy, "+", color=ORANGE, ms=12, mew=2)
    ax.annotate(f"peak |E/E0|² = {I[sl].max():.1f}\nx = {fx:+.2f} λ", xy=(fx, fy),
                xytext=(fx + 0.45, fy + 0.5), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.grid(False)
    ax.set_xlabel("x / λ   (wave travels +x)")
    ax.set_ylabel("y / λ")
    ax.set_title(f"|Ez / E0|² at {d['f_map_Hz']/1e9:.0f} GHz, plane z = sphere centre")
    fig.colorbar(im, ax=ax, label="|Ez / E0|²", shrink=0.9)

    jc = int(round(d["cy"]))
    ax2.plot(x[npml:nx - npml], I[jc, npml:nx - npml], color=BLUE, label="|Ez/E0|² on axis")
    ax2.axvspan(-Rl, Rl, color="#cde2fb", alpha=0.6, lw=0, label="sphere")
    ax2.axhline(1, color=INK2, lw=0.8, ls=":")
    ax2.text(x[npml] + 0.05, 1.15, "incident level", fontsize=8, color=INK2)
    ax2.set_xlabel("x / λ")
    ax2.set_ylabel("|Ez / E0|²")
    ax2.set_title("Cut along the propagation axis (y = z = centre)")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    savefig(fig, "fig3_grin")


# Fig 4 — cavity dispersion test
def plot_cavity():
    modes, probe = load("cavity_modes.csv"), load("cavity_probe_1x.csv")
    if modes is None or probe is None:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ez = probe["ez"].values
    dt = probe["time_s"].values[1] - probe["time_s"].values[0]
    w = np.hanning(len(ez))
    spec = np.abs(np.fft.rfft(ez * w))
    f = np.fft.rfftfreq(len(ez), dt)
    fmax = modes["f_cont_Hz"].max() * 1.05
    m = (f > modes["f_cont_Hz"].min() * 0.9) & (f < fmax)
    ax.semilogy(f[m] / 1e9, spec[m] / spec[m].max(), color=BLUE, lw=1.2, label="FDTD probe spectrum")
    for i, row in modes.iterrows():
        ax.axvline(row["f_cont_Hz"] / 1e9, color=INK2, lw=0.8, ls=":",
                   label="continuum mode" if i == 0 else None)
        ax.axvline(row["f_yee_coarse_Hz"] / 1e9, color=ORANGE, lw=0.8, ls="--",
                   label="discrete-Yee mode" if i == 0 else None)
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("|Ez(f)| (normalised)")
    ax.set_title("PEC cavity 12×14×17 cells: Hann-windowed spectrum")
    ax.set_ylim(1e-6, 2)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, frameon=True, edgecolor=GRID)

    cpl_c = modes["cells_per_lambda_coarse"].values
    for (col, mk), tag, cpl, lab in [(SERIES[0], "coarse", cpl_c, "Δ = 70 µm"),
                                     (SERIES[1], "fine", 2 * cpl_c, "Δ = 35 µm")]:
        err_c = np.abs(modes[f"f_meas_{tag}_Hz"] / modes["f_cont_Hz"] - 1)
        err_y = np.abs(modes[f"f_meas_{tag}_Hz"] / modes[f"f_yee_{tag}_Hz"] - 1)
        ax2.loglog(cpl, err_c, mk, color=col, mec="white", mew=0.8, ms=7, label=f"vs continuum, {lab}")
        ax2.loglog(cpl, np.maximum(err_y, 1e-12), mk, color=col, mfc="white", mew=1.4, ms=7,
                   label=f"vs discrete Yee, {lab}")
    xs = np.array([cpl_c.min() * 0.9, 2 * cpl_c.max() * 1.1])
    ax2.loglog(xs, 0.25 * (xs / xs[0]) ** -2 * 1e-2, color=INK2, lw=1, ls="--", label="∝ (λ/Δ)⁻²")
    ax2.set_xlabel("cells per wavelength λ/Δ")
    ax2.set_ylabel("|f_meas / f_ref − 1|")
    ax2.set_title("Frequency error: 2nd-order vs continuum, ~0 vs Yee theory")
    ax2.legend(fontsize=7.5, loc="center left")
    fig.tight_layout()
    savefig(fig, "fig4_cavity")


# Fig 5 — CPML σ scan
def plot_cpml():
    df = load("cpml_scan.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    scan = df[df["label"] == "scan"]
    for (col, mk), kap in zip(SERIES, sorted(scan["kappa_max"].unique())):
        s = scan[scan["kappa_max"] == kap]
        ax.plot(s["sigma_factor"], s["err_worst_dB"], "-" + mk, color=col, mec="white", mew=0.8,
                label=f"κ_max = {kap:g}, {int(s['npml'].iloc[0])} cells")
    orig = df[df["label"] == "original"]
    if len(orig):
        o = orig.iloc[0]
        ax.plot(o["sigma_factor"], o["err_worst_dB"], "D", color=INK, ms=7,
                label=f"original (σ = 4e13 /s, κ = 5, {int(o['npml'])} cells)")
    ctrl = df[df["label"].str.startswith("control")]
    if len(ctrl):
        ax.axhline(ctrl["err_worst_dB"].iloc[0], color=INK2, lw=1, ls=":")
        ax.text(df["sigma_factor"].max(), ctrl["err_worst_dB"].iloc[0] - 6,
                "control: σ = 0 (PEC wall reflects)", ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("σ_max / σ_opt   (σ_opt = 0.8(m+1)/(η0 Δ), m = 3)")
    ax.set_ylabel("worst-case error  max|E − E_ref| / max|E_ref|  (dB)")
    ax.set_title("CPML reflection error vs reflection-free reference")
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig5_cpml")


if __name__ == "__main__":
    print("Generating 3D validation figures ...")
    plot_geometry()
    plot_mie()
    plot_grin()
    plot_cavity()
    plot_cpml()
    print("Done.")
