"""
2D FDTD - Visualization script
Reads the CSV produced by 2d_fdtd.cpp and produces:
  1. Animated GIF  ->  fdtd_2d_animation.gif
  2. Static multi-panel figure  ->  fdtd_2d_snapshots.png
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm

# Config
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./data_outputs/2D_data.csv")
SAVE_PNG = "./results_outputs/2D_snapshots.png"
SAVE_GIF = "./results_outputs/2D_animation.gif"
DX       = 1e-4   # cell size in x [m]  
DY       = 1e-4   # cell size in y [m]

# Load data
if not os.path.exists(CSV_FILE):
    sys.exit(
        f"ERROR: '{CSV_FILE}' not found.\n"
        "Compile and run 2d_fdtd.cpp first:\n"
        "  g++ -O2 -o fdtd_2d 2d_fdtd.cpp && ./fdtd_2d"
    )

print(f"Loading {CSV_FILE} ...")
df = pd.read_csv(CSV_FILE)

if "ez" not in df.columns:
    sys.exit(
        "ERROR: CSV does not contain an 'ez' column.\n"
        "This plotter expects the 2-D simulation output (step, x, y, ez)."
    )

steps = np.array(sorted(df["step"].unique()))
NX    = int(df["x"].max()) + 1
NY    = int(df["y"].max()) + 1

x_mm = np.arange(NX) * DX * 1e3   # grid centres in mm
y_mm = np.arange(NY) * DY * 1e3

print(f"  {len(steps)} snapshots  |  grid {NX}×{NY}  "
      f"|  domain {x_mm[-1]:.1f}×{y_mm[-1]:.1f} mm")

# Build array EZ[snapshot_index, i, j]
print("  Building field arrays …")
EZ = np.zeros((len(steps), NX, NY))
for k, s in enumerate(steps):
    snap = df[df["step"] == s]
    EZ[k, snap["x"].values, snap["y"].values] = snap["ez"].values
    if (k + 1) % max(1, len(steps) // 10) == 0:
        print(f"    {k+1}/{len(steps)}", end="\r")
print()

vmax = max(float(np.abs(EZ).max()), 1e-12)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
extent = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]

# 1. Animated GIF
fig_a, ax_a = plt.subplots(figsize=(6, 5.5))
fig_a.suptitle("2D FDTD – Ez field (TM mode)", fontsize=13)

im = ax_a.imshow(
    EZ[0].T, origin="lower", extent=extent,
    cmap="seismic", norm=norm, aspect="equal", interpolation="nearest",
)
fig_a.colorbar(im, ax=ax_a, label="Ez  [V/m]", fraction=0.046, pad=0.04)
ax_a.set_xlabel("x  [mm]")
ax_a.set_ylabel("y  [mm]")
step_title = ax_a.set_title(f"step = {steps[0]}")


def _update(frame: int):
    im.set_data(EZ[frame].T)
    step_title.set_text(f"step = {steps[frame]}")
    return im, step_title


ani = animation.FuncAnimation(
    fig_a, _update, frames=len(steps), interval=100, blit=True
)

if SAVE_GIF:
    print(f"Saving animation → {SAVE_GIF}  (may take a moment) …")
    ani.save(SAVE_GIF, writer="pillow", fps=12)
    print(f"  Saved {SAVE_GIF}")

# 2. Static multi-panel snapshots
N_PANELS = min(9, len(steps))
idx_sel  = np.linspace(0, len(steps) - 1, N_PANELS, dtype=int)

NCOLS = 3
NROWS = (N_PANELS + NCOLS - 1) // NCOLS
fig_s, axes = plt.subplots(NROWS, NCOLS, figsize=(5 * NCOLS, 4.5 * NROWS))
axes = np.array(axes).flatten()

for ax2, idx in zip(axes, idx_sel):
    im2 = ax2.imshow(
        EZ[idx].T, origin="lower", extent=extent,
        cmap="seismic", norm=norm, aspect="equal", interpolation="nearest",
    )
    ax2.set_title(f"step = {steps[idx]}", fontsize=10)
    ax2.set_xlabel("x  [mm]", fontsize=8)
    ax2.set_ylabel("y  [mm]", fontsize=8)
    fig_s.colorbar(im2, ax=ax2, label="Ez [V/m]", fraction=0.046, pad=0.04)

for ax2 in axes[N_PANELS:]:
    ax2.set_visible(False)

fig_s.suptitle("2D FDTD snapshots – Ez field (TM mode)", fontsize=14, y=1.01)
fig_s.tight_layout()
fig_s.savefig(SAVE_PNG, dpi=150, bbox_inches="tight")
print(f"Snapshots saved → {SAVE_PNG}")

plt.show()
