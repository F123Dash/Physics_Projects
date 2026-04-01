import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle

#   Constants (must match 3D_fdtd.cpp)  
C0     = 2.99792458e8
DX     = 7e-5           # m
DT     = 0.99 * DX / (C0 * np.sqrt(3.0))
NX     = 160
SRC_X  = NX // 2        # 80
SRC_Y  = NX // 2        # 80
F_CW   = 75e9           # Hz
W_CW   = 2 * np.pi * F_CW
NPML   = 15
P1_X   = SRC_X + 50     # 130
P2_X   = SRC_X + 58     # 138
NSTEP  = 2500
T_RAMP = 120
SPHERE_R_CELLS = 40
SPHERE_R   = SPHERE_R_CELLS * DX   # 2.8 mm
SPHERE_EPS = 4.0

S_C    = C0 * DT / DX
arg    = np.sin(W_CW * DT / 2) / S_C
k_th   = (2 / DX) * np.arcsin(np.clip(arg, -1, 1))
vph_th = W_CW / k_th    # theoretical phase velocity [m/s]
lam    = C0 / F_CW      # free-space wavelength [m]

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "black",
    "axes.labelcolor":   "black",
    "axes.titlecolor":   "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
    "grid.color":        "0.85",
    "grid.linestyle":    "-",
    "grid.linewidth":    0.5,
    "text.color":        "black",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.frameon":    True,
    "legend.framealpha": 0.9,
    "lines.linewidth":   1.6,
})
plt.rcParams["pdf.fonttype"] = 42   # TrueType -- editable in Illustrator/Inkscape
plt.rcParams["ps.fonttype"]  = 42

# Muted scientific color palette (accessible on white)
C_BLUE   = "#2166ac"
C_RED    = "#d6604d"
C_GREEN  = "#4dac26"
C_ORANGE = "#e08214"
PML_COL  = "#aaaaaa"  # gray for CPML annotation lines


def load(filename):
    path = os.path.join(HERE, filename)
    if not os.path.exists(path):
        print(f"[WARN] {filename} not found — skipping.", file=sys.stderr)
        return None
    return pd.read_csv(path)


def savefig(fig, name):
    """Save PNG (400 dpi) and vector PDF."""
    base = os.path.join(HERE, name)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    fig.savefig(base, dpi=400, bbox_inches="tight")
    pdf_path = base.replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  saved -> {name}  +  {os.path.basename(pdf_path)}")
    plt.close(fig)


def _cpml_lines(ax):
    """Dashed CPML boundary lines on an imshow axis (mm units)."""
    pml_mm = NPML * DX * 1e3
    dom_mm = NX * DX * 1e3 - pml_mm
    for v in [pml_mm, dom_mm]:
        ax.axvline(v, color=PML_COL, lw=1.0, ls="--")
        ax.axhline(v, color=PML_COL, lw=1.0, ls="--")


def _sphere_circle(ax, **kw):
    """Sphere boundary circle on a mm-unit axis."""
    import matplotlib.patches as mpatches
    centre = (SRC_X * DX * 1e3, SRC_Y * DX * 1e3)
    circ = mpatches.Circle(centre, SPHERE_R * 1e3,
                           fill=False, edgecolor=C_RED, lw=1.5, ls="--", **kw)
    ax.add_patch(circ)


#   
# Fig 1 — Simulation geometry
#   
def plot_geometry():
    """Fig 1: CPML domain, GRIN sphere, source, and probe locations."""
    import matplotlib.patches as mpatches
    dom_mm  = NX * DX * 1e3
    pml_mm  = NPML * DX * 1e3
    free_mm = dom_mm - 2 * pml_mm
    src_mm  = SRC_X * DX * 1e3
    sph_mm  = SPHERE_R * 1e3
    p1_mm   = P1_X * DX * 1e3
    p2_mm   = P2_X * DX * 1e3

    fig, ax = plt.subplots(figsize=(6, 5))

    # CPML absorbing layers
    for (x0, y0, w, h) in [
        (0,               0,              dom_mm, pml_mm),
        (0,      dom_mm - pml_mm,         dom_mm, pml_mm),
        (0,               pml_mm,         pml_mm, free_mm),
        (dom_mm - pml_mm, pml_mm,         pml_mm, free_mm),
    ]:
        ax.add_patch(mpatches.Rectangle((x0, y0), w, h,
                                        facecolor="0.88", edgecolor="none"))

    # Computation domain interior
    ax.add_patch(mpatches.Rectangle((pml_mm, pml_mm), free_mm, free_mm,
                                    facecolor="white", edgecolor="black", lw=1.0))
    ax.text(pml_mm / 2, dom_mm / 2, "CPML", ha="center", va="center",
            fontsize=8, color="0.5", rotation=90)

    # GRIN sphere
    ax.add_patch(mpatches.Circle((src_mm, src_mm), sph_mm,
                                  facecolor="#c6dbef", edgecolor=C_BLUE,
                                  lw=1.5))
    ax.add_patch(mpatches.Rectangle((0, 0), dom_mm, dom_mm,
                                    fill=False, edgecolor="black", lw=1.0))

    # Source and probes
    ax.plot(src_mm, src_mm, "k*", ms=10, zorder=5)
    ax.plot(p1_mm, src_mm, "^", color=C_BLUE, ms=8, zorder=5)
    ax.plot(p2_mm, src_mm, "s", color=C_RED, ms=8, zorder=5)

    # Annotations
    ax.annotate("", xy=(src_mm + sph_mm, src_mm), xytext=(src_mm, src_mm),
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.0))
    ax.text(src_mm + sph_mm / 2, src_mm - sph_mm * 0.35,
            f"R = {sph_mm:.1f} mm", ha="center", fontsize=8, color=C_BLUE)
    ax.annotate("", xy=(pml_mm, dom_mm * 0.04), xytext=(0, dom_mm * 0.04),
                arrowprops=dict(arrowstyle="<->", color="0.5", lw=0.8))
    ax.text(pml_mm / 2, dom_mm * 0.07, f"{pml_mm:.1f} mm",
            ha="center", fontsize=7, color="0.4")

    ax.set_xlim(0, dom_mm)
    ax.set_ylim(0, dom_mm)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(
        f"Simulation geometry\n"
        f"{NX}\u00d7{NX}\u00d7{NX} Yee cells, "
        f"\u0394x = {DX*1e6:.0f}\u202f\u03bcm, "
        f"\u03bb/\u0394x \u2248 {lam/DX:.0f}, CPML = {NPML} cells"
    )
    fig.tight_layout()
    savefig(fig, "./results_outputs/fig1_geometry.png")


#   
# Fig 3 — Ez(x,y) wave propagation snapshots  (2×2 panel)
#   
def plot_snapshots(df):
    """Fig 3: Four Ez(x,y) snapshots at physically meaningful times (2x2 grid)."""
    if df is None:
        return
    all_steps = sorted(df["step"].unique())

    step_sphere_hit  = int(T_RAMP + SPHERE_R_CELLS / S_C)
    step_thru_sphere = step_sphere_hit + int(2 * SPHERE_R_CELLS / S_C)
    step_scatter     = step_thru_sphere + int(SPHERE_R_CELLS / S_C)
    step_steady      = max(all_steps)

    target_steps = [
        T_RAMP + 20,
        step_sphere_hit,
        step_thru_sphere,
        step_steady,
    ]
    picks = []
    for t in target_steps:
        closest = min(all_steps, key=lambda s: abs(s - t))
        if closest not in picks:
            picks.append(closest)
    while len(picks) < 4:
        for s in all_steps:
            if s not in picks:
                picks.append(s)
                break
    picks = sorted(picks[:4])

    panel_labels = ["(a) propagation", "(b) sphere interaction",
                    "(c) refraction / focusing", "(d) steady state"]
    global_max = np.percentile(np.abs(df["ez"]), 99.0) or 1.0
    ext = [0, NX * DX * 1e3, 0, NX * DX * 1e3]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        f"Ez(x, y) snapshots  —  f = {F_CW/1e9:.0f} GHz, "
        f"R = {SPHERE_R*1e3:.1f} mm, \u03b5\u1d9c = {SPHERE_EPS}",
        fontsize=12)

    for ax, step, lbl in zip(axes.flat, picks, panel_labels):
        sub  = df[df["step"] == step]
        nx_  = sub["x"].max() + 1
        ny_  = sub["y"].max() + 1
        grid = np.zeros((ny_, nx_))
        grid[sub["y"].values, sub["x"].values] = sub["ez"].values

        im = ax.imshow(grid, origin="lower", aspect="equal",
                       cmap="RdBu_r", vmin=-global_max, vmax=global_max, extent=ext)
        _cpml_lines(ax)
        _sphere_circle(ax)
        ax.plot(SRC_X * DX * 1e3, SRC_Y * DX * 1e3, "k*", ms=6)
        t_ns = step * DT * 1e9
        ax.set_title(f"{lbl}  (t = {t_ns:.2f} ns)", fontsize=9)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax, label="Ez (V/m)", pad=0.02, shrink=0.85, fraction=0.04)

    fig.tight_layout()
    savefig(fig, "./results_outputs/fig3_snapshots.png")


#   
# Fig 4 — Steady-state |Ez| field magnitude
#   
def plot_steady_state(df):
    """Fig 4: |Ez| magnitude at final snapshot (viridis, figsize=(6,5))."""
    if df is None:
        return
    step = df["step"].max()
    sub  = df[df["step"] == step]
    nx_  = sub["x"].max() + 1
    ny_  = sub["y"].max() + 1
    grid = np.zeros((ny_, nx_))
    grid[sub["y"].values, sub["x"].values] = sub["ez"].values
    mag  = np.abs(grid)

    vmax = np.percentile(mag, 99.0) or 1.0
    ext  = [0, NX * DX * 1e3, 0, NX * DX * 1e3]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mag, origin="lower", aspect="equal",
                   cmap="viridis", vmin=0, vmax=vmax, extent=ext)
    _cpml_lines(ax)
    _sphere_circle(ax)
    ax.plot(SRC_X * DX * 1e3, SRC_Y * DX * 1e3, "w*", ms=10)
    ax.plot(P1_X  * DX * 1e3, SRC_Y * DX * 1e3, "^", color=C_ORANGE, ms=7)
    ax.plot(P2_X  * DX * 1e3, SRC_Y * DX * 1e3, "s", color=C_GREEN,  ms=7)

    cb = fig.colorbar(im, ax=ax, label="|Ez| (V/m)", pad=0.02, shrink=0.85, fraction=0.04)
    cb.ax.tick_params()
    t_ns = step * DT * 1e9
    ax.set_title(f"|Ez| steady state  (t = {t_ns:.2f} ns)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.tight_layout()
    savefig(fig, "./results_outputs/fig4_steady_state.png")


#   
# Fig 5 — Probe Ez(t) comparison
#   
def plot_probes(df):
    """Fig 5: Ez(t) at P1 and P2 on shared axes (figsize=(7,4))."""
    if df is None:
        return
    t  = df["time_s"].values * 1e9      # ns
    p1 = df["ez_p1"].values
    p2 = df["ez_p2"].values

    r1 = (P1_X - SRC_X) * DX * 1e3
    r2 = (P2_X - SRC_X) * DX * 1e3

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, p1, color=C_BLUE, lw=1.5)
    ax.plot(t, p2, color=C_RED,  lw=1.5, ls="--")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Ez (V/m)")
    ax.set_title(f"Ez probe comparison  —  f = {F_CW/1e9:.0f} GHz")
    ax.grid(True)
    fig.tight_layout()
    savefig(fig, "./results_outputs/fig5_probes.png")


#   
# Fig 7 — Radial |Ez| decay vs 1/r theory
#   
def plot_envelope_scan(df):
    if df is None:
        return
    x_cell = df["x_cell"].values
    x_mm   = df["x_m"].values * 1e3
    env    = df["max_ez_envelope"].values

    # 1/r reference normalised at P1 — avoids singularity at source
    r_ref  = (P1_X - SRC_X) * DX
    r_all  = np.abs(x_cell - SRC_X) * DX
    inv_r  = np.zeros_like(env)
    mask   = r_all > 0
    E_ref  = np.mean(env[max(0, P1_X-1):P1_X+2])  # average 3 cells to reduce noise
    inv_r[mask] = E_ref * (r_ref / r_all[mask])
    if SRC_X < len(inv_r):
        inv_r[SRC_X] = env[SRC_X]

    fig, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    pml_mm = NPML * DX * 1e3
    dom_mm = NX * DX * 1e3
    for ax in (ax_lin, ax_log):
        # CPML shading
        ax.axvspan(0, pml_mm, color=PML_COL, alpha=0.15, label="CPML")
        ax.axvspan(dom_mm - pml_mm, dom_mm, color=PML_COL, alpha=0.15)
        ax.axvline(SRC_X * DX * 1e3, color=C_ORANGE, lw=1.0, ls=":", label="source")
        ax.axvline(P1_X * DX * 1e3, color=C_BLUE, lw=1.0, ls="--",
                   label=f"P1 (x={P1_X})")
        ax.axvline(P2_X * DX * 1e3, color=C_RED, lw=1.0, ls="--",
                   label=f"P2 (x={P2_X})")

    # linear axes
    ax_lin.plot(x_mm, env, color=C_GREEN, lw=1.8, label="|Ez| envelope")
    ax_lin.plot(x_mm, inv_r, color=C_ORANGE, lw=1.2, ls="--", label="1/r reference")
    ax_lin.set_ylabel("|Ez| max  [V/m]")
    ax_lin.legend(facecolor="white", edgecolor="0.7", fontsize=8, ncol=3)
    ax_lin.grid(True)

    # log axes
    pos_mask = env > 0
    ax_log.semilogy(x_mm[pos_mask], env[pos_mask],
                    color=C_GREEN, lw=1.8, label="|Ez| envelope")
    ax_log.semilogy(x_mm[inv_r > 0], inv_r[inv_r > 0],
                    color=C_ORANGE, lw=1.2, ls="--", label="1/r reference")
    ax_log.set_xlabel("x [mm]")
    ax_log.set_ylabel("|Ez| max  [V/m]  (log)")
    ax_log.legend(facecolor="white", edgecolor="0.7", fontsize=8, ncol=3)
    ax_log.grid(True, which="both")

    # annotate CPML attenuation
    e_in  = env[NX - NPML] if env[NX - NPML] > 0 else np.nan
    e_mid = env[NX - NPML + 8] if env[NX - NPML + 8] > 0 else np.nan
    if np.isfinite(e_in) and np.isfinite(e_mid) and e_mid > 0:
        att = 20 * np.log10(e_in / e_mid)
        ax_log.annotate(f"CPML attenuation\n≈ {att:.1f} dB / 8 cells",
                        xy=((NX - NPML + 4) * DX * 1e3, np.sqrt(e_in * e_mid)),
                        xytext=((NX - NPML - 15) * DX * 1e3, e_in * 0.05),
                        arrowprops=dict(arrowstyle="->", color=PML_COL),
                        fontsize=8, color=PML_COL,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", edgecolor=PML_COL))

    fig.suptitle(f"Fig 7 — Radial |Ez| decay vs 1/r theory\n"
                 f"3D FDTD + CPML + GRIN sphere  (f = {F_CW/1e9:.0f} GHz)",
                 fontsize=12)
    fig.tight_layout()
    savefig(fig, "./results_outputs/fig7_radial_decay.png")


#   
# Fig 2 — Permittivity distribution (cividis)
#   
def plot_permittivity(df):
    """Fig 2: GRIN sphere ε_r map (cividis colormap, figsize=(6,5))."""
    if df is None:
        return
    nx_  = df["x"].max() + 1
    ny_  = df["y"].max() + 1
    grid = np.zeros((ny_, nx_))
    grid[df["y"].values, df["x"].values] = df["eps_r"].values

    ext = [0, NX * DX * 1e3, 0, NX * DX * 1e3]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, origin="lower", aspect="equal",
                   cmap="cividis", vmin=1.0, vmax=SPHERE_EPS, extent=ext)
    _cpml_lines(ax)
    _sphere_circle(ax)
    ax.plot(SRC_X * DX * 1e3, SRC_Y * DX * 1e3, "k*", ms=8)
    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85, fraction=0.04)
    cb.set_label(r"$\varepsilon_r$")
    ax.set_title(
        f"GRIN sphere permittivity  "
        f"(\u03b5\u1d9c = {SPHERE_EPS}, R = {SPHERE_R*1e3:.1f} mm)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.tight_layout()
    savefig(fig, "./results_outputs/fig2_permittivity.png")





#   
# Main
#   
if __name__ == "__main__":
    print("Loading CSV data …")
    df_snap = load("./data_outputs/3D_slice.csv")
    df_disp = load("./data_outputs/dispersion_3D.csv")
    df_eps  = load("./data_outputs/dielectric_slice_3D.csv")

    print("Generating plots …")
    plot_geometry()                    # Fig 1 — simulation domain schematic
    plot_permittivity(df_eps)          # Fig 2 — εᵣ distribution (cividis)
    plot_snapshots(df_snap)            # Fig 3 — 4-panel Ez snapshots
    plot_steady_state(df_snap)         # Fig 4 — |Ez| steady-state (viridis)
    plot_probes(df_disp)               # Fig 5 — Ez(t) probe comparison

    print("Done.")
