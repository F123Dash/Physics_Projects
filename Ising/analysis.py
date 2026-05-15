import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d

from load_data import load_ising_csv


@dataclass
class CriticalEstimates:
    tc_by_L: Dict[int, float]
    tc_inf: float
    tc_inf_stderr: float
    beta: float
    beta_stderr: float


def block_average(data: np.ndarray, block_size: int) -> np.ndarray:
    """Compute block averages to reduce autocorrelation bias."""
    n = len(data) // block_size
    if n == 0:
        return data
    blocked = data[:n * block_size].reshape(n, block_size)
    return np.mean(blocked, axis=1)


def _find_binder_crossing(t_grid: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> float:
    """Find crossing point via sign-change detection."""
    diff = u1 - u2  # Don't take absolute value!
    
    # Remove any NaN values that may arise from extrapolation
    valid_mask = ~np.isnan(diff)
    
    if np.count_nonzero(valid_mask) < 10:
        return float('nan')
    
    t_valid = t_grid[valid_mask]
    diff_valid = diff[valid_mask]
    
    sign_changes = np.where(np.diff(np.sign(diff_valid)))[0]
    
    if len(sign_changes) == 0:
        return float('nan')
    
    # Among sign changes, prefer those in physical range [2.1, 2.5]
    # If none in range, take all
    physical_range_mask = np.array([
        2.1 <= (t_valid[i] + t_valid[i+1])/2.0 <= 2.5 
        for i in sign_changes
    ])
    
    if np.any(physical_range_mask):
        candidates = sign_changes[physical_range_mask]
    else:
        candidates = sign_changes
    
    # Pick crossing with smallest MINIMUM absolute difference across bracket
    # (excludes oscillations where one endpoint is far from crossing)
    abs_diffs = np.array([
        min(abs(diff_valid[idx]), abs(diff_valid[idx+1]))
        for idx in candidates
    ])
    best_local_idx = np.argmin(abs_diffs)
    best_idx = candidates[best_local_idx]
    
    # Linear interpolation for exact crossing
    t1, t2 = t_valid[best_idx], t_valid[best_idx + 1]
    d1, d2 = diff_valid[best_idx], diff_valid[best_idx + 1]
    
    if d2 == d1:
        tc_crossing = (t1 + t2) / 2.0
    else:
        tc_crossing = t1 - d1 * (t2 - t1) / (d2 - d1)
    
    return float(tc_crossing)


def estimate_tc_binder(df: pd.DataFrame) -> Tuple[float, float, list]:
    """Estimate Tc via finite-size EXTRAPOLATION of Binder crossings."""
    sizes = sorted(df["L"].unique())
    
    if len(sizes) < 2:
        raise ValueError("Need at least 2 lattice sizes for Binder crossing")
    
    for L in sizes:
        g = df[df["L"] == L]
        if "U" not in g.columns:
            raise ValueError("Binder cumulant U not in dataframe; compile with M4 support")
    
    # Build all adjacent pairs - use ALL for better statistics
    all_pairs = list(zip(sizes[:-1], sizes[1:]))
    pairs = all_pairs
    
    print(f"  Using {len(pairs)} pair(s) for Binder crossing")
    print("  Binder crossings:")
    
    tc_list = []
    
    for L1, L2 in pairs:
        L1, L2 = float(L1), float(L2)
        
        g1 = df[df["L"] == L1].sort_values("T").copy()
        g2 = df[df["L"] == L2].sort_values("T").copy()
        
        t_min = max(g1["T"].min(), g2["T"].min())
        t_max = min(g1["T"].max(), g2["T"].max())
        
        # Restrict to critical region but leave some margin
        t_min = max(t_min, 1.9)
        t_max = min(t_max, 2.8)
        
        if t_max - t_min < 0.05:
            print(f"    L={int(L1)}/{int(L2)}: T range too small ({t_max - t_min:.3f}), skipping")
            continue
        
        t_grid = np.linspace(t_min, t_max, 500)
        
        # Interpolate using linear to avoid extrapolation issues
        try:
            f1 = interp1d(g1["T"], g1["U"], kind='linear', bounds_error=True)
            f2 = interp1d(g2["T"], g2["U"], kind='linear', bounds_error=True)
            u1_interp = f1(t_grid)
            u2_interp = f2(t_grid)
        except ValueError:
            print(f"    L={int(L1)}/{int(L2)}: Interpolation failed, skipping")
            continue
        
        tc_pair = _find_binder_crossing(t_grid, u1_interp, u2_interp)
        
        if np.isnan(tc_pair):
            print(f"    L={int(L1)}/{int(L2)}: No crossing found")
            continue
        
        tc_list.append(tc_pair)
        print(f"    L={int(L1)}/{int(L2)}: Tc = {tc_pair:.6f}")
    
    if len(tc_list) == 0:
        raise RuntimeError("No valid Binder crossings found")
    
    L_eff = np.array([np.sqrt(L1 * L2) for L1, L2 in pairs])
    tc_vals = np.array(tc_list)
    
    x = 1.0 / L_eff  # Inverse scaling variable
    
    slope, intercept, r_sq, _, stderr = stats.linregress(x, tc_vals)
    
    tc_inf = float(intercept)  # Tc(∞)
    tc_stderr = float(stderr)
    
    print(f"  Finite-size extrapolation: Tc(L) = {intercept:.6f} + {slope:.6f}/L_eff")
    print(f"  Tc(∞) = {tc_inf:.6f} ± {tc_stderr:.6f} (R² = {r_sq:.4f})")
    
    return tc_inf, tc_stderr, tc_list


def estimate_tc_finite_size(df: pd.DataFrame) -> Tuple[Dict[int, float], float, float, float]:
    """Estimate Tc(L) from χ peak (primary, numerically robust)."""
    from scipy.signal import savgol_filter
    
    tc_by_L = {}
    for L, g in df.groupby("L"):
        g = g.sort_values("T").copy()
        
        chi_vals = g["chi"].to_numpy()
        n_points = len(chi_vals)
        window_length = min(7, n_points // 2 * 2 + 1) if n_points >= 3 else n_points
        window_length = max(3, window_length)
        chi_smooth = savgol_filter(chi_vals, window_length=window_length, polyorder=2)
        i = int(np.argmax(chi_smooth))
        tc_by_L[int(L)] = float(g.iloc[i]["T"])
    
    print(f"DEBUG Tc(L):", tc_by_L)
    
    Ls_all = sorted(tc_by_L.keys())
    median_tc = np.median(list(tc_by_L.values()))
    
    valid_Ls = []
    for L in Ls_all:
        if L < 64:
            continue
        # Check if Tc is within reasonable range of median (outlier rejection)
        if abs(tc_by_L[L] - median_tc) < 0.2:
            valid_Ls.append(L)
    
    tc_pairs = sorted([(L, tc_by_L[L]) for L in valid_Ls])
    filtered = []
    prev = 10.0  # Start high
    for L, tc in tc_pairs:
        if tc <= prev:  # Enforce non-increasing
            filtered.append((L, tc))
            prev = tc
    
    if len(filtered) == 0:
        Ls_large = np.array(valid_Ls, dtype=float)
        tcL = np.array([tc_by_L[int(L)] for L in Ls_large], dtype=float)
    else:
        Ls_large = np.array([L for L, _ in filtered], dtype=float)
        tcL = np.array([tc for _, tc in filtered], dtype=float)
    
    # If monotonic filtering leaves too few points, use the largest available sizes.
    if len(Ls_large) < 3:
        fallback_sizes = valid_Ls if len(valid_Ls) > 0 else Ls_all
        if len(fallback_sizes) > 0:
            fallback_sizes = sorted(fallback_sizes)
            fallback_sizes = fallback_sizes[-min(3, len(fallback_sizes)) :]
            Ls_large = np.array(fallback_sizes, dtype=float)
            tcL = np.array([tc_by_L[int(L)] for L in Ls_large], dtype=float)
    
    mask = (tcL > 2.1) & (tcL < 2.4)
    if np.count_nonzero(mask) > 0:
        Ls_large = Ls_large[mask]
        tcL = tcL[mask]
    
    x = 1.0 / Ls_large

    weights = Ls_large**2  # larger systems dominate
    
    if len(Ls_large) >= 2:
        coeffs = np.polyfit(x, tcL, 1, w=weights)  # Linear: Tc(L) = Tc + a/L
        slope, intercept = coeffs
        
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum(weights * (tcL - y_pred)**2)
        ss_tot = np.sum(weights * (tcL - np.average(tcL, weights=weights))**2)
        r_sq = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0)
        
        tc_inf = float(intercept)
        if len(tcL) > 2:
            w = weights
            w_sum = np.sum(w)
            w_x = np.sum(w * x)
            w_xx = np.sum(w * x * x)
            det = w_xx * w_sum - w_x * w_x
            ss_res = np.sum(w * (tcL - y_pred) ** 2)
            sigma2 = ss_res / (len(tcL) - 2)
            if det > 0:
                cov_intercept = sigma2 * (w_xx / det)
                stderr = np.sqrt(cov_intercept)
            else:
                stderr = np.sqrt(sigma2)
        else:
            # Cannot estimate statistical error with 2 points - use data scatter
            stderr = np.std(tcL) if len(tcL) > 1 else 0.02
    else:
        tc_inf = float(np.mean(tcL))
        stderr = 0.0
        slope = 0.0
        r_sq = 0.0
    
    tc_inf_stderr = float(stderr)
    
    if not (2.1 < tc_inf < 2.4):
        print(f"⚠ Tc({tc_inf:.4f}) outside physical range [2.1, 2.4]; using mean of largest systems")
        tc_inf = float(np.mean(tcL))
        tc_inf_stderr = float(np.std(tcL)) if len(tcL) > 1 else 0.0
    
    print(f"✓ Using χ-peak extrapolation (largest systems only, weighted): Tc(∞) = {tc_inf:.6f} ± {tc_inf_stderr:.6f}")
    print(f"  Finite-size relation: Tc(L) = {intercept:.6f} + {slope:.6f}/L (R² = {r_sq:.4f})")
    print(f"  Systems used: L = {[int(L) for L in Ls_large]}")
    if len(Ls_large) <= 2:
        print("  ⚠ Using only 2 lattice sizes → Tc uncertainty is underestimated")
    
    # Return tc_inf for BOTH plotting and beta fitting (consistency)
    return tc_by_L, tc_inf, tc_inf_stderr, tc_inf


def estimate_beta_collapse(df: pd.DataFrame, tc_est: float, beta_guess: float = 0.1, nu: float = 1.0) -> Tuple[float, float]:
    """Estimate beta from largest system below Tc using strict fitting window."""
    large_l_df = df[df["L"] >= 64]
    if len(large_l_df) > 0:
        lmax = int(large_l_df["L"].max())
    else:
        lmax = int(df["L"].max())
    
    df_max_l = df[df["L"] == lmax].copy().sort_values("T")
    
    if len(df_max_l) < 10:
        raise RuntimeError(f"Insufficient L={lmax} data")
    
    T_min, T_max = tc_est - 0.08, tc_est - 0.02
    mask = (df_max_l["T"] >= T_min) & (df_max_l["T"] <= T_max)
    df_fit = df_max_l[mask].copy()
    
    if len(df_fit) < 5:
        T_min, T_max = tc_est - 0.10, tc_est
        mask = (df_max_l["T"] >= T_min) & (df_max_l["T"] <= T_max)
        df_fit = df_max_l[mask].copy()
    
    if len(df_fit) < 5:
        raise RuntimeError(f"Insufficient data below Tc (found {len(df_fit)})")
    
    T_vals = df_fit["T"].to_numpy()
    M_vals = df_fit["absM"].to_numpy()
    
    if np.any(M_vals <= 0):
        raise RuntimeError("Non-positive M in fit region")
    
    tau = tc_est - T_vals  # Distance from Tc, positive when below Tc
    
    if np.any(tau <= 0):
        raise RuntimeError("Temperature points not below Tc")
    
    order = np.argsort(tau)
    tau = tau[order]
    M_vals = M_vals[order]
    
    x = np.log(tau)
    y = np.log(M_vals)
    
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise RuntimeError("Non-finite log values")
    
    coeffs = np.polyfit(x, y, 1)
    beta_fit = coeffs[0]
    
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)
    
    window_str = f"[{T_min:.3f}, {T_max:.3f}] K"
    print(f"  Using window: ΔT ∈ [{tau.min():.4f}, {tau.max():.4f}] K")
    print(f"  L={lmax} (largest system): β = {beta_fit:.4f} (R² = {r2:.4f}, T ∈ {window_str}, {len(x)} pts)")
    
    betas = [beta_fit]
    
    for trim in [1, 2]:
        if len(x) > trim + 2:
            x_trim = x[trim:-trim]
            y_trim = y[trim:-trim]
            if len(x_trim) >= 3:
                c = np.polyfit(x_trim, y_trim, 1)
                betas.append(c[0])
    
    stderr = np.std(betas) if len(betas) > 1 else 0.005
    
    return beta_fit, stderr


def _beta_loglog_selection(
    dt: np.ndarray,
    m_vals: np.ndarray,
    min_points: int,
    primary_window: Tuple[float, float],
    fallback_window: Tuple[float, float],
    trim_tail: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    mask = (dt > primary_window[0]) & (dt < primary_window[1]) & (m_vals > 1e-8)
    used_window = primary_window
    if np.count_nonzero(mask) < min_points:
        mask = (dt > fallback_window[0]) & (dt < fallback_window[1]) & (m_vals > 1e-8)
        used_window = fallback_window

    x = np.log(dt[mask])
    y = np.log(m_vals[mask])

    if len(x) < min_points:
        return np.array([], dtype=float), np.array([], dtype=float), used_window

    if trim_tail > 0 and len(x) > min_points + trim_tail:
        x = x[:-trim_tail]
        y = y[:-trim_tail]

    return x, y, used_window


def estimate_beta_loglog(df: pd.DataFrame, tc_est: float) -> Tuple[float, float]:
    """Fit beta from LARGEST-L ONLY below Tc via log M = beta log(Tc-T) + c."""
    min_l_for_beta = 64
    
    large_l_df = df[df["L"] >= min_l_for_beta]
    if len(large_l_df) > 0:
        lmax = int(large_l_df["L"].max())
        print(f"Beta fit: Using L={lmax} (>= {min_l_for_beta})")
    else:
        lmax = int(df["L"].max())
        print(f"⚠ No L >= {min_l_for_beta}; using largest available L={lmax}")
    
    if np.isnan(lmax):
        raise RuntimeError("No valid lattice sizes found")
    
    g = df[df["L"] == lmax].copy().sort_values("T")

    dt = tc_est - g["T"].to_numpy()
    m = g["absM"].to_numpy()

    x, y, window = _beta_loglog_selection(
        dt,
        m,
        min_points=3,
        primary_window=(0.02, 0.07),
        fallback_window=(0.01, 0.10),
        trim_tail=2,
    )
    
    if len(x) < 3:
        raise RuntimeError(f"Insufficient points in fit region (need >=3, got {len(x)})")
    
    n_pts = len(x)
    window_str = f"[Tc-{window[1]:.2f}, Tc-{window[0]:.2f}] K"
    print(f"  Window: {n_pts} points in {window_str} (tail trim applied)")
    
    if n_pts < 3:
        raise RuntimeError(f"Only {n_pts} pts after trim (need ≥3)")

    slope, intercept, r_sq, _, stderr = stats.linregress(x, y)
    _ = intercept
    
    if slope < 0.07 or slope > 0.18:
        print(f"  ⚠ WARNING: Beta={slope:.4f} outside typical range [0.07, 0.18] (likely due to Tc uncertainty)")
    
    print(f"  Fit: log(M) = {slope:.4f} * log(ΔT) (R²={r_sq:.4f})")
    
    return float(slope), float(stderr)


def bootstrap_beta(df: pd.DataFrame, tc_est: float, n_boot: int = 1000, seed: int = 1234) -> np.ndarray:
    """Bootstrap estimate of beta uncertainty (FIX D.12: Uses SAME window as estimate_beta_loglog)."""
    large_l_df = df[df["L"] >= 64]
    if len(large_l_df) > 0:
        lmax = int(large_l_df["L"].max())
    else:
        lmax = int(df["L"].max())
    
    g = df[df["L"] == lmax].copy().sort_values("T")
    
    if len(g) < 6:
        return np.array([], dtype=float)
    
    T_vals = g["T"].to_numpy()
    M_vals = g["absM"].to_numpy()
    
    dt = tc_est - T_vals
    x, y, _ = _beta_loglog_selection(
        dt,
        M_vals,
        min_points=3,
        primary_window=(0.02, 0.07),
        fallback_window=(0.01, 0.10),
        trim_tail=2,
    )

    if len(x) < 3 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return np.array([], dtype=float)
    
    rng = np.random.default_rng(seed)
    betas = []
    
    n = len(x)
    tries = 0
    max_tries = n_boot * 10
    
    while len(betas) < n_boot and tries < max_tries:
        tries += 1
        idx = rng.integers(0, n, n)
        xb = x[idx]
        yb = y[idx]
        
        # Skip if all x values are the same
        if np.allclose(xb, xb[0]):
            continue
        
        # Linear regression
        s, _, _, _, _ = stats.linregress(xb, yb)
        betas.append(s)
    
    return np.array(betas[:n_boot], dtype=float)


def run_analysis(
    data_csv: str,
    metrics_csv: str = "./data_outputs/analysis_metrics.csv",
    bootstrap_csv: str = "./data_outputs/beta_bootstrap.csv",
) -> CriticalEstimates:
    df = load_ising_csv(data_csv)
    available_sizes = sorted(df["L"].unique().tolist())
    print(f"Using dynamic lattice sizes from data: {available_sizes}")

    metrics_dir = os.path.dirname(metrics_csv) or "."
    bootstrap_dir = os.path.dirname(bootstrap_csv) or "."
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(bootstrap_dir, exist_ok=True)

    tc_by_L, tc_inf, tc_inf_stderr, tc_inf_val = estimate_tc_finite_size(df)
    
    print("Beta estimation: Using log-log scaling (largest L)...")
    try:
        beta, beta_stderr = estimate_beta_loglog(df, tc_inf_val)
    except RuntimeError as e:
        print(f"⚠ Log-log fit failed: {e}; trying collapse method")
        try:
            beta, beta_stderr = estimate_beta_collapse(df, tc_inf_val)
        except RuntimeError as e2:
            print(f"⚠ Beta fit completely failed: {e2}")
            beta, beta_stderr = np.nan, np.nan
    
    beta_boot = bootstrap_beta(df, tc_inf_val)

    metrics = pd.DataFrame(
        [
            {"name": "Tc_inf", "value": tc_inf, "stderr": tc_inf_stderr},
            {"name": "beta", "value": beta, "stderr": beta_stderr},
        ]
    )
    metrics.to_csv(metrics_csv, index=False)

    tc_by_l_csv = os.path.join(metrics_dir, "tc_by_L.csv")
    pd.DataFrame(
        [{"L": L, "Tc_L": tc} for L, tc in sorted(tc_by_L.items())]
    ).to_csv(tc_by_l_csv, index=False)

    if beta_boot.size > 0:
        pd.DataFrame({"beta": beta_boot}).to_csv(bootstrap_csv, index=False)

    return CriticalEstimates(
        tc_by_L=tc_by_L,
        tc_inf=tc_inf,
        tc_inf_stderr=tc_inf_stderr,
        beta=beta,
        beta_stderr=beta_stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite-size scaling analysis for 2D Ising model.")
    parser.add_argument("--data", default="./data_outputs/data.csv", help="Input CSV from C++/CUDA simulation")
    parser.add_argument("--metrics", default="./data_outputs/analysis_metrics.csv", help="Output metrics CSV")
    parser.add_argument("--bootstrap", default="./data_outputs/beta_bootstrap.csv", help="Output bootstrap beta CSV")
    args = parser.parse_args()

    est = run_analysis(
        args.data,
        args.metrics,
        args.bootstrap,
    )

    print("\n" + "="*50)
    print("Analysis complete")
    print("="*50)
    print(f"  Tc(infinite) = {est.tc_inf:.6f} +/- {est.tc_inf_stderr:.6f}")
    print(f"  beta         = {est.beta:.6f} +/- {est.beta_stderr:.6f}")
    
    if not np.isnan(est.beta):
        if not (0.05 < est.beta < 0.25):
            print(f"  ⚠ WARNING: Beta is outside expected range [0.05, 0.25]")
            print(f"    This likely indicates an issue with Tc estimation")
            print(f"    Expected: β ≈ 0.125 (2D Ising theory)")
        elif abs(est.beta - 0.125) > 0.06:
            print(f"  ⚠ Beta deviates from theory (0.125) by {abs(est.beta - 0.125):.4f}")
            print(f"    Accuracy could be improved; check Tc and data quality")
    
    print("  Tc(L):")
    for L, tc in sorted(est.tc_by_L.items()):
        print(f"    L={L:3d}: {tc:.6f}")
    print("="*50)


if __name__ == "__main__":
    main()
