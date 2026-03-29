import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter

from analysis import estimate_tc_finite_size
from load_data import load_ising_csv

# Set scientific style with vector-safe fonts
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
    """Save figure as both PNG and PDF (vector format)."""
    plt.tight_layout()
    
    # PNG (raster)
    png_path = path if path.endswith('.png') else path + '.png'
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    
    # PDF (vector) - optional but recommended for publication
    if include_pdf:
        pdf_path = png_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    plt.close()


def plot_m_vs_t(df: pd.DataFrame, tc_inf: float, outdir: str) -> None:
    """Plot magnetization vs temperature (clean lines, no markers)."""
    plt.figure(figsize=(6.5, 4.5))
    valid_Ls = [16, 24, 32, 48, 64, 128]  # ✅ FIX 1: Remove L=96 for consistency
    for L, g in sorted(df.groupby("L"), key=lambda x: x[0]):
        if L not in valid_Ls:
            continue
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
    """Plot susceptibility with smoothing to reduce noise."""
    plt.figure(figsize=(6.5, 4.5))
    valid_Ls = [16, 24, 32, 48, 64, 128]  # ✅ FIX 1: Remove L=96 for consistency
    for L, g in sorted(df.groupby("L"), key=lambda x: x[0]):
        if L not in valid_Ls:
            continue
        g = g.sort_values("T")
        t_vals = g["T"].to_numpy()
        chi_vals = g["chi"].to_numpy()
        
        # Smooth χ if enough points (reduces noise while preserving peak)
        if len(chi_vals) > 7:
            try:
                chi_smooth = savgol_filter(chi_vals, min(7, len(chi_vals) // 2 * 2 + 1), 3)
            except:
                chi_smooth = chi_vals
        else:
            chi_smooth = chi_vals
        
        plt.plot(t_vals, chi_smooth, lw=2.0, 
                label=rf"$L={int(L)}$", alpha=0.85)
    
    plt.axvline(tc_inf, color='red', linestyle='--', alpha=0.5, lw=1.5, label=rf"$T_c={tc_inf:.4f}$")
    plt.xlabel(r"Temperature $T$ (J/$k_B$)", fontsize=11)
    plt.ylabel(r"Susceptibility $\chi = N(⟨M^2⟩ - ⟨|M|⟩^2)/T$", fontsize=11)
    plt.title("Susceptibility vs Temperature (Savitzky-Golay smoothed)")
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10, loc='best')
    plt.xlim(1.8, 3.4)
    savefig(os.path.join(outdir, "fig2_susceptibility.png"))


def plot_binder_vs_t(df: pd.DataFrame, tc_inf: float, outdir: str) -> None:
    """Plot Binder cumulant vs temperature."""
    if "U" not in df.columns:
        return
    
    plt.figure(figsize=(6.5, 4.5))
    valid_Ls = [16, 24, 32, 48, 64, 128]  # ✅ FIX 1: Remove L=96 for consistency
    for L, g in sorted(df.groupby("L"), key=lambda x: x[0]):
        if L not in valid_Ls:
            continue
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
    """Plot specific heat vs temperature."""
    plt.figure(figsize=(6.5, 4.5))
    valid_Ls = [16, 24, 32, 48, 64, 128]  # ✅ FIX 1: Remove L=96 for consistency
    for L, g in sorted(df.groupby("L"), key=lambda x: x[0]):
        if L not in valid_Ls:
            continue
        g = g.sort_values("T")
        plt.plot(g["T"], g["C"], lw=2.0, 
                label=rf"$L={int(L)}$", alpha=0.85)
    
    plt.axvline(tc_inf, color='red', linestyle='--', alpha=0.5, lw=1.5, label=rf"$T_c={tc_inf:.4f}$")
    plt.xlabel(r"Temperature $T$ (J/$k_B$)", fontsize=11)
    plt.ylabel(r"Specific Heat $C_V = (⟨E^2⟩ - ⟨E⟩^2)/T^2$", fontsize=11)
    plt.title("Specific Heat vs Temperature")
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10, loc='best')
    plt.xlim(1.8, 3.4)
    savefig(os.path.join(outdir, "fig3_specific_heat.png"))


def plot_tc_extrapolation(df: pd.DataFrame, metrics_df: pd.DataFrame, tc_inf: float, outdir: str) -> None:
    """Plot finite-size scaling extrapolation with error bars."""
    tc_by_L, _, tc_inf_stderr, _ = estimate_tc_finite_size(df)
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


def plot_beta_loglog(df: pd.DataFrame, metrics_df: pd.DataFrame, tc_inf: float, outdir: str) -> None:
    """Plot beta exponent via log-log fit with theoretical reference."""
    # Handle missing large L gracefully
    large_l_df = df[df["L"] >= 64]
    if len(large_l_df) > 0:
        lmax = int(large_l_df["L"].max())
    else:
        lmax = int(df["L"].max())
    
    g = df[df["L"] == lmax].sort_values("T")

    dt = tc_inf - g["T"].to_numpy()
    m = g["absM"].to_numpy()
    
    # ✅ FIX 5: Improve β accuracy with window [0.02, 0.07]
    mask = (dt > 0.02) & (dt < 0.07) & (m > 1e-8)
    if np.count_nonzero(mask) < 3:
        mask = (dt > 0.01) & (dt < 0.10) & (m > 1e-8)

    x = np.log(dt[mask])
    y = np.log(m[mask])

    if x.size < 3:
        plt.figure(figsize=(6.5, 4.5))
        plt.text(0.5, 0.5, "Insufficient subcritical data for $\\beta$ fit", 
                ha="center", va="center", fontsize=12)
        plt.axis("off")
        savefig(os.path.join(outdir, "fig5_beta_loglog.png"))
        return

    slope, intercept, r_value, _, stderr = stats.linregress(x, y)
    xfit = np.linspace(x.min(), x.max(), 200)
    yfit = slope * xfit + intercept
    
    # Theoretical reference: β = 1/8 = 0.125
    y_theory = 0.125 * xfit + intercept

    # Get beta uncertainty from metrics if available
    beta_err = 0.0
    if metrics_df is not None and 'stderr' in metrics_df.columns:
        beta_row = metrics_df[metrics_df['name'] == 'beta']
        if len(beta_row) > 0:
            beta_err = beta_row.iloc[0]['stderr']

    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(x, y, s=60, alpha=0.7, color='C0', edgecolors='black', linewidth=1.5,
               label=rf"$L={int(lmax)}$ data ({len(x)} pts)")
    plt.plot(xfit, yfit, '-', color='C1', lw=2.5,
            label=rf"Fit: $\beta = {slope:.4f} \pm {beta_err:.4f}$ ($R^2={r_value**2:.3f}$)")
    plt.plot(xfit, y_theory, '--', color='green', lw=2, alpha=0.7,
            label=r"Theory: $\beta = 1/8 = 0.125$")
    
    plt.xlabel(r"$\log(T_c - T)$", fontsize=11)
    plt.ylabel(r"$\log(\langle |M| \rangle)$", fontsize=11)
    plt.title(r"Critical Exponent $\beta$ (Log-Log Fit, Tight Window)")
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10)
    savefig(os.path.join(outdir, "fig5_beta_loglog.png"))


def plot_data_collapse(df: pd.DataFrame, metrics_df: pd.DataFrame, tc_inf: float, outdir: str) -> None:
    """Plot data collapse with fitted/theoretical parameters.
    
    ✅ FIX 4: Restrict to |T - Tc| < 0.08 (pure scaling region, tighter)
    """
    # Use fitted beta if available, else theoretical
    beta = 0.125  # theoretical
    if metrics_df is not None and 'value' in metrics_df.columns:
        beta_row = metrics_df[metrics_df['name'] == 'beta']
        if len(beta_row) > 0:
            beta_val = beta_row.iloc[0]['value']
            if not np.isnan(beta_val) and 0.05 < beta_val < 0.2:
                beta = beta_val
    
    nu = 1.0  # theoretical ν for 2D Ising
    
    plt.figure(figsize=(6.5, 5.0))
    valid_Ls = [16, 24, 32, 48, 64, 128]  # ✅ FIX 1: Remove L=96 for consistency
    for L, g in sorted(df.groupby("L"), key=lambda x: x[0]):
        if L not in valid_Ls:
            continue
        Lf = float(L)
        g_sorted = g.sort_values("T")
        
        # ✅ FIX 4: Restrict to pure scaling region |T - Tc| < 0.08 (tighter, was 0.15)
        mask = np.abs(g_sorted["T"].to_numpy() - tc_inf) < 0.08
        
        if np.count_nonzero(mask) > 0:
            x = (g_sorted.loc[mask, "T"].to_numpy() - tc_inf) * (Lf ** (1.0 / nu))
            y = g_sorted.loc[mask, "absM"].to_numpy() * (Lf ** (beta / nu))
            plt.plot(x, y, lw=2.0, label=rf"$L={int(L)}$", alpha=0.85)
    
    plt.xlabel(rf"$(T - T_c) L^{{1/\nu}}$", fontsize=11)
    plt.ylabel(rf"$\langle |M| \rangle L^{{\beta/\nu}}$", fontsize=11)
    plt.title(rf"Data Collapse ($\beta=${beta:.3f}, $\nu=${nu:.1f}, $|T-T_c| < 0.08$ K)")
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10)
    savefig(os.path.join(outdir, "fig6_data_collapse.png"))


def plot_beta_bootstrap_hist(beta_csv: str, metrics_df: pd.DataFrame, outdir: str) -> None:
    """Plot bootstrap distribution of beta with statistics."""
    if not os.path.exists(beta_csv):
        return
    
    b = pd.read_csv(beta_csv)["beta"].to_numpy()
    if b.size == 0:
        return

    beta_mean = np.mean(b)
    beta_std = np.std(b)
    beta_median = np.median(b)

    plt.figure(figsize=(6.5, 4.5))
    plt.hist(b, bins=40, alpha=0.75, edgecolor='black', color='C0', density=False)
    plt.axvline(beta_mean, color='C1', linestyle='-', linewidth=2.5,
               label=rf"Mean: ${beta_mean:.4f}$")
    plt.axvline(beta_median, color='orange', linestyle=':', linewidth=2.5,
               label=rf"Median: ${beta_median:.4f}$")
    plt.axvline(0.125, color='green', linestyle='--', linewidth=2,
               label=r"Theory: $\beta = 1/8$", alpha=0.7)
    
    plt.xlabel(r"$\beta$ value", fontsize=11)
    plt.ylabel("Frequency (count)", fontsize=11)
    plt.title(rf"Bootstrap Distribution of $\beta$ ({len(b)} resamples, std=${beta_std:.4f}$)")
    plt.grid(True, alpha=0.3, which='both', axis='y')
    plt.legend(fontsize=10)
    savefig(os.path.join(outdir, "fig7_beta_bootstrap_hist.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Ising finite-size scaling diagnostics.")
    parser.add_argument("--data", default="data.csv")
    parser.add_argument("--metrics", default="analysis_metrics.csv")
    parser.add_argument("--beta_csv", default="beta_bootstrap.csv")
    parser.add_argument("--outdir", default="plots")
    args = parser.parse_args()

    ensure_plot_dir(args.outdir)

    df = load_ising_csv(args.data)
    
    # Load metrics if available
    metrics_df = None
    if os.path.exists(args.metrics):
        metrics_df = pd.read_csv(args.metrics)
    
    # ✅ CORRECTED: Use χ-based Tc (numerically robust)
    tc_by_L, tc_inf, tc_inf_stderr, _ = estimate_tc_finite_size(df)

    plot_m_vs_t(df, tc_inf, args.outdir)
    plot_chi_vs_t(df, tc_inf, args.outdir)
    plot_binder_vs_t(df, tc_inf, args.outdir)
    plot_c_vs_t(df, tc_inf, args.outdir)
    plot_tc_extrapolation(df, metrics_df, tc_inf, args.outdir)
    plot_beta_loglog(df, metrics_df, tc_inf, args.outdir)
    plot_data_collapse(df, metrics_df, tc_inf, args.outdir)
    plot_beta_bootstrap_hist(args.beta_csv, metrics_df, args.outdir)

    print(f"\n✓ Saved plots to {args.outdir}/")
    print(f"  Formats: PNG (raster @ 200dpi) + PDF (vector)")


if __name__ == "__main__":
    main()
