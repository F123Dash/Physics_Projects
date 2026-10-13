import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import drop_partial_sizes, estimate_tc_finite_size
from load_data import load_ising_csv

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['pdf.fonttype'] = 42        # Vector fonts for PDF
plt.rcParams['ps.fonttype'] = 42         # PostScript vector fonts
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']


def ensure_plot_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def savefig(path: str, include_pdf: bool = True) -> None:
    plt.tight_layout()
    png_path = path if path.endswith('.png') else path + '.png'
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    if include_pdf:
        pdf_path = png_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.close()


def plot_m_vs_t(df: pd.DataFrame, tc_inf: float, outdir: str) -> None:
    plt.figure(figsize=(6.5, 4.5))
    for L, g in sorted(df.groupby("L"), key=lambda x: x[0]):
        g = g.sort_values("T")
        plt.plot(g["T"], g["absM"], lw=2.0, 
                label=rf"$L={int(L)}$", alpha=0.85)
    
    plt.axvline(tc_inf, color='red', linestyle='--', alpha=0.5, lw=1.5, label=rf"$T_c={tc_inf:.4f}$")
    plt.xlabel(r"Temperature $T$ (J/$k_B$)", fontsize=11)
    plt.ylabel(r"Magnetization $\langle |M| \rangle$", fontsize=11)
    plt.title("Magnetization vs Temperature")
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10, loc='best')
    plt.xlim(1.8, 3.4)
    savefig(os.path.join(outdir, "fig1_magnetization.png"))


def plot_chi_vs_t(df: pd.DataFrame, tc_inf: float, outdir: str) -> None:
    plt.figure(figsize=(6.5, 4.5))
    for L, g in sorted(df.groupby("L"), key=lambda x: x[0]):
        g = g.sort_values("T")
        t_vals = g["T"].to_numpy()
        chi_vals = g["chi"].to_numpy()
        plt.plot(t_vals, chi_vals, lw=2.0, 
                label=rf"$L={int(L)}$", alpha=0.85)
    plt.axvline(tc_inf, color='red', linestyle='--', alpha=0.5, lw=1.5, label=rf"$T_c={tc_inf:.4f}$")
    plt.xlabel(r"Temperature $T$ (J/$k_B$)", fontsize=11)
    plt.ylabel(r"Susceptibility $\chi = N(⟨M^2⟩ - ⟨|M|⟩^2)/T$", fontsize=11)
    plt.title("Susceptibility vs Temperature")
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10, loc='best')
    plt.xlim(1.8, 3.4)
    savefig(os.path.join(outdir, "fig2_susceptibility.png"))


def plot_binder_vs_t(df: pd.DataFrame, tc_inf: float, outdir: str) -> None:
    if "U" not in df.columns:
        return
    plt.figure(figsize=(6.5, 4.5))
    for L, g in sorted(df.groupby("L"), key=lambda x: x[0]):
        g = g.sort_values("T")
        plt.plot(g["T"], g["U"], lw=2.0, 
                label=rf"$L={int(L)}$", alpha=0.85)
    plt.axvline(tc_inf, color='red', linestyle='--', alpha=0.5, lw=1.5, label=rf"$T_c={tc_inf:.4f}$")
    plt.xlabel(r"Temperature $T$ (J/$k_B$)", fontsize=11)
    plt.ylabel(r"Binder Cumulant $U = 1 - \frac{⟨M^4⟩}{3⟨M^2⟩^2}$", fontsize=11)
    plt.title("Binder Cumulant vs Temperature")
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10, loc='best')
    plt.xlim(1.8, 3.4)
    savefig(os.path.join(outdir, "fig2b_binder.png"))


def plot_c_vs_t(df: pd.DataFrame, tc_inf: float, outdir: str) -> None:
    plt.figure(figsize=(6.5, 4.5))
    for L, g in sorted(df.groupby("L"), key=lambda x: x[0]):
        g = g.sort_values("T")
        plt.plot(g["T"], g["C"], lw=2.0, 
                label=rf"$L={int(L)}$", alpha=0.85)
    plt.axvline(tc_inf, color='red', linestyle='--', alpha=0.5, lw=1.5, label=rf"$T_c={tc_inf:.4f}$")
    plt.xlabel(r"Temperature $T$ (J/$k_B$)", fontsize=11)
    plt.ylabel(r"Specific Heat $C_V = N(⟨e^2⟩ - ⟨e⟩^2)/T^2$", fontsize=11)
    plt.title("Specific Heat vs Temperature")
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10, loc='best')
    plt.xlim(1.8, 3.4)
    savefig(os.path.join(outdir, "fig3_specific_heat.png"))


def plot_tc_extrapolation(tc_by_L: dict, tc_inf: float, tc_inf_stderr: float, outdir: str) -> None:
    Ls = np.array(sorted(tc_by_L.keys()), dtype=float)
    tcL = np.array([tc_by_L[int(L)] for L in Ls])
    x = 1.0 / Ls
    p = np.polyfit(x, tcL, 1)
    xfit = np.linspace(0, x.max() * 1.15, 200)
    yfit = np.polyval(p, xfit)
    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(x, tcL, s=100, alpha=0.7, color='C0', edgecolors='black', linewidth=1.5, zorder=3)
    plt.plot(xfit, yfit, '--', color='C1', lw=2.5, 
            label=rf"$T_c(\infty)={tc_inf:.4f}\pm{tc_inf_stderr:.4f}$", zorder=2)
    plt.scatter([0], [tc_inf], s=200, marker='*', color='red', edgecolors='black', 
               linewidth=1.5, label='Extrapolated Tc', zorder=5)
    plt.xlabel(r"$1/L$", fontsize=11)
    plt.ylabel(r"$T_c(L)$ (from $\chi$ peak)", fontsize=11)
    plt.title("Finite-Size Scaling Extrapolation")
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10)
    savefig(os.path.join(outdir, "fig4_tc_extrapolation.png"))


def plot_data_collapse(df: pd.DataFrame, tc_inf: float, outdir: str) -> None:
    # Consistency check with the exact exponents; measured beta/nu comes from fss_analysis.py.
    beta = 0.125
    nu = 1.0
    
    plt.figure(figsize=(6.5, 5.0))
    for L, g in sorted(df.groupby("L"), key=lambda x: x[0]):
        Lf = float(L)
        g_sorted = g.sort_values("T")
        mask = np.abs(g_sorted["T"].to_numpy() - tc_inf) < 0.08
        if np.count_nonzero(mask) > 0:
            x = (g_sorted.loc[mask, "T"].to_numpy() - tc_inf) * (Lf ** (1.0 / nu))
            y = g_sorted.loc[mask, "absM"].to_numpy() * (Lf ** (beta / nu))
            plt.plot(x, y, lw=2.0, label=rf"$L={int(L)}$", alpha=0.85)
    plt.xlabel(rf"$(T - T_c) L^{{1/\nu}}$", fontsize=11)
    plt.ylabel(rf"$\langle |M| \rangle L^{{\beta/\nu}}$", fontsize=11)
    plt.title(rf"Data Collapse, exact exponents ($\beta=1/8$, $\nu=1$, $|T-T_c| < 0.08$)")
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10)
    savefig(os.path.join(outdir, "fig6_data_collapse.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Ising finite-size scaling diagnostics.")
    parser.add_argument("--data", default="./data_outputs/data.csv")
    parser.add_argument("--outdir", default="./plots")
    args = parser.parse_args()

    ensure_plot_dir(args.outdir)

    df = drop_partial_sizes(load_ising_csv(args.data))

    tc_by_L, tc_inf, tc_inf_stderr = estimate_tc_finite_size(df)

    plot_m_vs_t(df, tc_inf, args.outdir)
    plot_chi_vs_t(df, tc_inf, args.outdir)
    plot_binder_vs_t(df, tc_inf, args.outdir)
    plot_c_vs_t(df, tc_inf, args.outdir)
    plot_tc_extrapolation(tc_by_L, tc_inf, tc_inf_stderr, args.outdir)
    plot_data_collapse(df, tc_inf, args.outdir)

    print(f"\n Saved plots to {args.outdir}/")
if __name__ == "__main__":
    main()
