"""
FDTD 1D - Visualization script
Reads fdtd_data.csv produced by ftdt.cpp and produces:
  1. Animated GIF  ->  fdtd_animation.gif
  2. Static multi-panel figure  ->  fdtd_snapshots.png
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -- Config --------------------------------------------------------------------
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./data_outputs/1D_data.csv")
SAVE_PNG = "./results_outputs/1D_snapshots.png"
SAVE_GIF = "./results_outputs/1D_animation.gif"   # set to None to skip

print(f"Loading {CSV_FILE} ...")
df = pd.read_csv(CSV_FILE)

steps = np.array(sorted(df["step"].unique()))
z     = df[df["step"] == steps[0]]["z_m"].values * 1e3   # -> mm
NZ    = len(z)

EX = np.stack([df[df["step"] == s]["Ex_V_per_m"].values for s in steps])
HY = np.stack([df[df["step"] == s]["Hy_A_per_m"].values for s in steps])

print(f"  {len(steps)} snapshots  |  NZ={NZ}  z=[{z[0]:.1f} ... {z[-1]:.1f}] mm")

# -- Global y-limits -----------------------------------------------------------
ex_lim = max(np.abs(EX).max(), 1e-12) * 1.15
hy_lim = max(np.abs(HY).max(), 1e-12) * 1.15

# 1.  Animation
fig_a, (ax_ex, ax_hy) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig_a.suptitle("1D FDTD - Maxwell's equations", fontsize=13)

line_ex, = ax_ex.plot(z, EX[0], color="royalblue", lw=1.5)
line_hy, = ax_hy.plot(z, HY[0], color="tomato",    lw=1.5)
ttl = ax_ex.set_title(f"step = {steps[0]}")

for ax, lbl, lim in [(ax_ex, "Ex  [V/m]", ex_lim),
                     (ax_hy, "Hy  [A/m]", hy_lim)]:
    ax.set_ylim(-lim, lim)
    ax.set_ylabel(lbl)
    ax.grid(True, ls="--", alpha=0.4)
    ax.axhline(0, color="k", lw=0.6)
ax_hy.set_xlabel("z  [mm]")

def update(frame):
    line_ex.set_ydata(EX[frame])
    line_hy.set_ydata(HY[frame])
    ttl.set_text(f"step = {steps[frame]}")
    return line_ex, line_hy, ttl

ani = animation.FuncAnimation(
    fig_a, update, frames=len(steps), interval=120, blit=True
)

if SAVE_GIF:
    ani.save(SAVE_GIF, writer="pillow", fps=8)
    print(f"Animation saved -> {SAVE_GIF}")

# 2.  Static multi-panel figure (evenly-spaced snapshots)
n_panels = 6
idx_sel  = np.linspace(0, len(steps) - 1, n_panels, dtype=int)

ncols = 2
nrows = (n_panels + 1) // ncols
fig_s, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows), sharex=True)
axes = axes.flatten()

for k, idx in enumerate(idx_sel):
    ax = axes[k]
    ax.plot(z, EX[idx], color="royalblue", lw=1.5, label="Ex [V/m]")
    ax.plot(z, HY[idx], color="tomato",    lw=1.5, label="Hy [A/m]")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"step = {steps[idx]}", fontsize=10)
    ax.set_ylim(-ex_lim, ex_lim)
    ax.grid(True, ls="--", alpha=0.4)
    if k % ncols == 0:
        ax.set_ylabel("Field amplitude")
    if k >= (nrows - 1) * ncols:
        ax.set_xlabel("z  [mm]")

for k in range(n_panels, len(axes)):
    axes[k].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig_s.legend(handles, labels, loc="upper right", fontsize=10)
fig_s.suptitle("1D FDTD snapshots - Ex & Hy fields", fontsize=13)
fig_s.tight_layout()
fig_s.savefig(SAVE_PNG, dpi=150)
print(f"Static figure saved -> {SAVE_PNG}")

plt.show()
