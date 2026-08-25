"""Shared figure style."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# --small: reduced smoke-test version of a script, written to results/small/
SMALL = "--small" in sys.argv
RES = ROOT / "results" / "small" if SMALL else ROOT / "results"
FIG = RES / "figures" if SMALL else ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

# categorical series, fixed order
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
SERIES = [C1, C2, C3, C4, C5]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd7"

# sequential single-hue map for dB intensity (light = weak, dark = strong)
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cfe0f5", "#7fb0ea", "#2a78d6", "#174a8c", "#0b1f3d"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "legend.frameon": False, "legend.fontsize": 8.5,
    "axes.prop_cycle": matplotlib.cycler(color=SERIES),
})


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"  wrote {(FIG / name).relative_to(ROOT)}")


def map_axes(ax):
    ax.grid(False)
    return ax
