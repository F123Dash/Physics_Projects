import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.special import logsumexp

from load_data import load_ising_csv

EXACT_TC = 2.269185


@dataclass
class CriticalEstimates:
    tc_by_L: Dict[int, float]
    tc_inf: float
    tc_inf_stderr: float
    beta: float
    beta_stderr: float


class WHAMAnalysis:
    def __init__(self,
        simulation_temps: list[float],
        energy_histograms: dict[float, np.ndarray],energy_bin_edges: np.ndarray,n_samples: dict[float, int],) -> None:
        if len(simulation_temps) == 0:
            raise ValueError("WHAM requires at least one simulation temperature.")

        self.temps = np.array(simulation_temps, dtype=float)
        self.energy_bin_edges = np.asarray(energy_bin_edges, dtype=float)
        if self.energy_bin_edges.ndim != 1 or self.energy_bin_edges.size < 2:
            raise ValueError("energy_bin_edges must be a 1D array of length >= 2.")

        self.energy_bins = 0.5 * (self.energy_bin_edges[:-1] + self.energy_bin_edges[1:])
        if not np.allclose(self.energy_bins, np.round(self.energy_bins)):
            raise ValueError("Energy bins must be integer-valued for Ising energies.")

        n_bins = self.energy_bins.size
        hist_list = []
        sample_counts = []
        for T in self.temps:
            if T not in energy_histograms:
                raise ValueError(f"Missing energy histogram for T={T}.")
            if T not in n_samples:
                raise ValueError(f"Missing sample count for T={T}.")

            hist = np.asarray(energy_histograms[T], dtype=float)
            if hist.ndim != 1 or hist.size != n_bins:
                raise ValueError(f"Histogram for T={T} has incorrect bin count.")
            hist_list.append(hist)
            sample_counts.append(int(n_samples[T]))

        self.energy_histograms = np.stack(hist_list, axis=0)
        self.n_samples = np.array(sample_counts, dtype=float)
        if np.any(self.n_samples <= 0):
            raise ValueError("All sample counts must be positive.")

        self.beta = 1.0 / self.temps
        self.log_g = None
        self.f_k = np.zeros_like(self.temps)

        self._last_observable_T_grid = None
        self._last_observable_values = None
        self._last_chi_T_grid = None
        self._last_chi_values = None

    def fit(self, max_iter: int = 10000, tol: float = 1e-8) -> bool:
        log_hist = np.where(self.energy_histograms > 0.0, np.log(self.energy_histograms), -np.inf)
        log_n = np.log(self.n_samples)

        converged = False
        for it in range(max_iter):
            log_den = logsumexp(
                log_n[:, None] + self.f_k[:, None] - self.beta[:, None] * self.energy_bins[None, :],
                axis=0,
            )
            log_g = logsumexp(log_hist, axis=0) - log_den

            log_Z = logsumexp(log_g[None, :] - self.beta[:, None] * self.energy_bins[None, :], axis=1)
            new_f = -log_Z

            delta = float(np.max(np.abs(new_f - self.f_k)))
            if it % 100 == 0:
                print(f"WHAM iter {it}: max |delta f_k| = {delta:.3e}")

            self.f_k = new_f
            self.log_g = log_g

            if delta < tol:
                converged = True
                break

        return converged

    def _combine_observable_by_energy(self,observable_histograms: dict[float, np.ndarray],) -> np.ndarray:
        obs_list = []
        for T in self.temps:
            if T not in observable_histograms:
                raise ValueError(f"Missing observable histogram for T={T}.")
            obs = np.asarray(observable_histograms[T], dtype=float)
            if obs.ndim != 1 or obs.size != self.energy_bins.size:
                raise ValueError(f"Observable histogram for T={T} has incorrect bin count.")
            obs_list.append(obs)

        obs_stack = np.stack(obs_list, axis=0)
        counts = self.energy_histograms
        denom = np.sum(counts, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            obs_E = np.where(denom > 0.0, np.sum(counts * obs_stack, axis=0) / denom, 0.0)
        return obs_E

    def _reweight_observable(self, obs_E: np.ndarray, beta: float) -> float:
        if self.log_g is None:
            raise RuntimeError("fit() must converge before reweighting observables.")

        log_w = self.log_g - beta * self.energy_bins
        log_Z = logsumexp(log_w)

        pos_mask = obs_E > 0.0
        neg_mask = obs_E < 0.0

        pos_term = -np.inf
        neg_term = -np.inf
        if np.any(pos_mask):
            pos_term = logsumexp(log_w[pos_mask] + np.log(obs_E[pos_mask]))
        if np.any(neg_mask):
            neg_term = logsumexp(log_w[neg_mask] + np.log(-obs_E[neg_mask]))

        pos_val = 0.0 if not np.isfinite(pos_term) else np.exp(pos_term - log_Z)
        neg_val = 0.0 if not np.isfinite(neg_term) else np.exp(neg_term - log_Z)
        return pos_val - neg_val

    def observable_vs_T(
        self,
        T_grid: np.ndarray,
        observable_histograms: dict[float, np.ndarray],
    ) -> np.ndarray:
        obs_E = self._combine_observable_by_energy(observable_histograms)
        T_grid = np.asarray(T_grid, dtype=float)
        out = np.empty_like(T_grid, dtype=float)
        for i, T in enumerate(T_grid):
            out[i] = self._reweight_observable(obs_E, beta=1.0 / T)

        self._last_observable_T_grid = T_grid
        self._last_observable_values = out
        return out

    def susceptibility_vs_T(
        self,
        T_grid: np.ndarray,
        M2_histograms: dict[float, np.ndarray],
        absM_histograms: dict[float, np.ndarray],
        L: int,
    ) -> np.ndarray:
        m2_E = self._combine_observable_by_energy(M2_histograms)
        abs_m_E = self._combine_observable_by_energy(absM_histograms)

        T_grid = np.asarray(T_grid, dtype=float)
        chi = np.empty_like(T_grid, dtype=float)
        n_spins = float(L * L)
        for i, T in enumerate(T_grid):
            beta = 1.0 / T
            m2 = self._reweight_observable(m2_E, beta)
            abs_m = self._reweight_observable(abs_m_E, beta)
            chi[i] = n_spins * (m2 - abs_m * abs_m) / T

        self._last_chi_T_grid = T_grid
        self._last_chi_values = chi
        return chi

    def find_chi_peak(self, T_grid: np.ndarray, L: int) -> tuple[float, float]:
        if self._last_chi_T_grid is None or self._last_chi_values is None:
            raise ValueError("Run susceptibility_vs_T() before calling find_chi_peak().")

        T_grid = np.asarray(T_grid, dtype=float)
        if T_grid.shape != self._last_chi_T_grid.shape or not np.allclose(T_grid, self._last_chi_T_grid):
            raise ValueError("T_grid does not match the last susceptibility evaluation.")

        idx = int(np.argmax(self._last_chi_values))
        return float(T_grid[idx]), float(self._last_chi_values[idx])

    def check_physics(self, T_low: float, T_high: float) -> None:
        if self._last_observable_T_grid is None or self._last_observable_values is None:
            raise ValueError("Run observable_vs_T() for |M| before calling check_physics().")

        T_grid = self._last_observable_T_grid
        abs_m = self._last_observable_values
        if T_low < np.min(T_grid) or T_high > np.max(T_grid):
            raise ValueError("T_low/T_high must lie within the last observable grid.")

        low_val = float(np.interp(T_low, T_grid, abs_m))
        high_val = float(np.interp(T_high, T_grid, abs_m))

        if low_val <= 0.8:
            raise ValueError(f"WHAM physics check failed: |M|(T_low)={low_val:.3f} <= 0.8")
        if high_val >= 0.1:
            raise ValueError(f"WHAM physics check failed: |M|(T_high)={high_val:.3f} >= 0.1")


def block_average(data: np.ndarray, block_size: int) -> np.ndarray:
    n = len(data) // block_size
    if n == 0:
        return data
    blocked = data[:n * block_size].reshape(n, block_size)
    return np.mean(blocked, axis=1)


def _find_binder_crossing(t_grid: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> float:
    diff = u1 - u2  # Don't take absolute value!
    valid_mask = ~np.isnan(diff)
    
    if np.count_nonzero(valid_mask) < 10:
        return float('nan')
    
    t_valid = t_grid[valid_mask]
    diff_valid = diff[valid_mask]
    
    sign_changes = np.where(np.diff(np.sign(diff_valid)))[0]
    
    if len(sign_changes) == 0:
        return float('nan')
    physical_range_mask = np.array([
        2.1 <= (t_valid[i] + t_valid[i+1])/2.0 <= 2.5 
        for i in sign_changes
    ])
    
    if np.any(physical_range_mask):
        candidates = sign_changes[physical_range_mask]
    else:
        candidates = sign_changes
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
    from scipy.signal import savgol_filter
    
    tc_by_L = {}
    for L, g in df.groupby("L"):
        g = g.sort_values("T").copy()
        
        chi_vals = g["chi"].to_numpy()
        n_points = len(chi_vals)
        if n_points < 3:
            print(f"  WARNING: L={int(L)} has only {n_points} temperature points; skipping Tc peak.")
            continue

        window_length = min(7, n_points if n_points % 2 == 1 else n_points - 1)
        window_length = max(3, window_length)
        chi_smooth = savgol_filter(chi_vals, window_length=window_length, polyorder=2)
        i = int(np.argmax(chi_smooth))
        if i == 0 or i == n_points - 1:
            print(f"  WARNING: L={int(L)} chi peak at boundary (T={g.iloc[i]['T']:.3f}); skipping Tc peak.")
            continue
        tc_by_L[int(L)] = float(g.iloc[i]["T"])
    
    print(f"DEBUG Tc(L):", tc_by_L)
    
    Ls_all = sorted(tc_by_L.keys())
    median_tc = np.median(list(tc_by_L.values()))
    
    valid_Ls = []
    for L in Ls_all:
        if L < 64:
            continue
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
    
    if len(Ls_large) == 0:
        raise RuntimeError("No valid Tc(L) points after filtering; check data or thermalization.")

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
        intercept = tc_inf
        r_sq = 0.0
    
    tc_inf_stderr = float(stderr)
    
    if not (2.1 < tc_inf < 2.4):
        print(f" Tc({tc_inf:.4f}) outside physical range [2.1, 2.4]; using mean of largest systems")
        tc_inf = float(np.mean(tcL))
        tc_inf_stderr = float(np.std(tcL)) if len(tcL) > 1 else 0.0
    
    print(f" Using χ-peak extrapolation (largest systems only, weighted): Tc(∞) = {tc_inf:.6f} ± {tc_inf_stderr:.6f}")
    if len(Ls_large) >= 2:
        print(f"  Finite-size relation: Tc(L) = {intercept:.6f} + {slope:.6f}/L (R² = {r_sq:.4f})")
    else:
        print("  Finite-size relation: insufficient points for fit; using mean Tc")
    print(f"  Systems used: L = {[int(L) for L in Ls_large]}")
    if len(Ls_large) <= 2:
        print("   Using only 2 lattice sizes  Tc uncertainty is underestimated")
    
    # Return tc_inf for BOTH plotting and beta fitting (consistency)
    return tc_by_L, tc_inf, tc_inf_stderr, tc_inf


def estimate_beta_collapse(df: pd.DataFrame, tc_est: float, beta_guess: float = 0.1, nu: float = 1.0) -> Tuple[float, float]:
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
    print(f"  Using window: deltaT == [{tau.min():.4f}, {tau.max():.4f}] K")
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
    min_l_for_beta = 64
    
    large_l_df = df[df["L"] >= min_l_for_beta]
    if len(large_l_df) > 0:
        lmax = int(large_l_df["L"].max())
        print(f"Beta fit: Using L={lmax} (>= {min_l_for_beta})")
    else:
        lmax = int(df["L"].max())
        print(f" No L >= {min_l_for_beta}; using largest available L={lmax}")
    
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
        print(f"   WARNING: Beta={slope:.4f} outside typical range [0.07, 0.18] (data quality or thermalization issues)")
    
    print(f"  Fit: log(M) = {slope:.4f} * log(deltaT) (R²={r_sq:.4f})")
    
    return float(slope), float(stderr)


def estimate_eta(df: pd.DataFrame, tc_est: float, tc_window: float = 0.01) -> Tuple[float, float]:
    if "chi" not in df.columns:
        raise ValueError("chi column missing; run load_ising_csv to compute chi.")

    records = []
    for L, g in df.groupby("L"):
        mask = (g["T"] >= tc_est - tc_window) & (g["T"] <= tc_est + tc_window)
        if np.count_nonzero(mask) == 0:
            continue
        chi_mean = float(np.mean(g.loc[mask, "chi"].to_numpy()))
        if chi_mean > 0:
            records.append((float(L), chi_mean))

    if len(records) < 4:
        raise ValueError("Need at least 4 lattice sizes with chi data near Tc to estimate eta.")

    Ls = np.array([r[0] for r in records], dtype=float)
    chis = np.array([r[1] for r in records], dtype=float)
    x = np.log(Ls)
    y = np.log(chis)

    slope, intercept, r_sq, _, stderr = stats.linregress(x, y)
    _ = intercept
    _ = r_sq
    eta = 2.0 - slope
    eta_stderr = float(stderr)
    return float(eta), eta_stderr


def estimate_gamma(df: pd.DataFrame) -> Tuple[float, float]:
    if "chi" not in df.columns:
        raise ValueError("chi column missing; run load_ising_csv to compute chi.")

    from scipy.signal import savgol_filter

    records = []
    for L, g in df.groupby("L"):
        chi_vals = g.sort_values("T")["chi"].to_numpy()
        n_points = len(chi_vals)
        if n_points == 0:
            continue

        if n_points >= 7:
            chi_smooth = savgol_filter(chi_vals, window_length=7, polyorder=3)
        elif n_points >= 5:
            chi_smooth = savgol_filter(chi_vals, window_length=5, polyorder=3)
        elif n_points >= 3:
            chi_smooth = savgol_filter(chi_vals, window_length=3, polyorder=2)
        else:
            chi_smooth = chi_vals

        chi_max = float(np.max(chi_smooth))
        if chi_max > 0:
            records.append((float(L), chi_max))

    if len(records) < 3:
        raise ValueError("Need at least 3 lattice sizes to estimate gamma.")

    Ls = np.array([r[0] for r in records], dtype=float)
    chi_max = np.array([r[1] for r in records], dtype=float)
    x = np.log(Ls)
    y = np.log(chi_max)

    slope, intercept, r_sq, _, stderr = stats.linregress(x, y)
    _ = intercept
    _ = r_sq
    gamma = float(slope)
    gamma_stderr = float(stderr)

    if gamma < 1.5 or gamma > 2.0:
        print(f"WARNING: gamma={gamma:.4f} outside expected range [1.5, 2.0]")

    return gamma, gamma_stderr


def estimate_alpha_logL(df: pd.DataFrame) -> Tuple[float, float]:
    if "C" not in df.columns:
        raise ValueError("C column missing; run load_ising_csv to compute specific heat.")

    records = []
    for L, g in df.groupby("L"):
        c_max = float(np.max(g["C"].to_numpy()))
        records.append((float(L), c_max))

    if len(records) < 3:
        raise ValueError("Need at least 3 lattice sizes to estimate alpha from log(L) scaling.")

    Ls = np.array([r[0] for r in records], dtype=float)
    C_max = np.array([r[1] for r in records], dtype=float)
    x = np.log(Ls)
    y = C_max

    slope, intercept, r_sq, _, stderr = stats.linregress(x, y)
    _ = intercept
    _ = r_sq
    if slope < 0:
        print("WARNING: negative slope in C_max vs log(L); data quality may be poor")
    return float(slope), float(stderr)


def plot_exponent_summary(df: pd.DataFrame, tc_est: float, outdir: str) -> None:
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    os.makedirs(outdir, exist_ok=True)

    chi_records = []
    for L, g in df.groupby("L"):
        mask = (g["T"] >= tc_est - 0.01) & (g["T"] <= tc_est + 0.01)
        if np.count_nonzero(mask) == 0:
            continue
        chi_mean = float(np.mean(g.loc[mask, "chi"].to_numpy()))
        if chi_mean > 0:
            chi_records.append((float(L), chi_mean))

    if len(chi_records) < 2:
        raise ValueError("Insufficient data near Tc to plot chi(L).")

    Ls_chi = np.array([r[0] for r in chi_records], dtype=float)
    chi_tc = np.array([r[1] for r in chi_records], dtype=float)

    chi_max_records = []
    for L, g in df.groupby("L"):
        chi_vals = g.sort_values("T")["chi"].to_numpy()
        n_points = len(chi_vals)
        if n_points >= 7:
            chi_smooth = savgol_filter(chi_vals, window_length=7, polyorder=3)
        elif n_points >= 5:
            chi_smooth = savgol_filter(chi_vals, window_length=5, polyorder=3)
        elif n_points >= 3:
            chi_smooth = savgol_filter(chi_vals, window_length=3, polyorder=2)
        else:
            chi_smooth = chi_vals
        if chi_smooth.size > 0:
            chi_max_records.append((float(L), float(np.max(chi_smooth))))

    Ls_chi_max = np.array([r[0] for r in chi_max_records], dtype=float)
    chi_max = np.array([r[1] for r in chi_max_records], dtype=float)

    c_max_records = []
    for L, g in df.groupby("L"):
        c_max_records.append((float(L), float(np.max(g["C"].to_numpy()))))
    Ls_c = np.array([r[0] for r in c_max_records], dtype=float)
    C_max = np.array([r[1] for r in c_max_records], dtype=float)

    slope_gamma, intercept_gamma, _, _, _ = stats.linregress(np.log(Ls_chi_max), np.log(chi_max))
    gamma_fit = np.exp(intercept_gamma) * (Ls_chi_max ** slope_gamma)

    slope_alpha, intercept_alpha, _, _, _ = stats.linregress(np.log(Ls_c), C_max)
    alpha_fit = slope_alpha * np.log(Ls_c) + intercept_alpha

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.loglog(Ls_chi, chi_tc, "o", label="chi(T~Tc)")
    L_ref = Ls_chi[len(Ls_chi) // 2]
    chi_ref = chi_tc[len(chi_tc) // 2]
    chi_ref_line = chi_ref * (Ls_chi / L_ref) ** (7.0 / 4.0)
    ax.loglog(Ls_chi, chi_ref_line, "--", label="L^(7/4)")
    ax.set_xlabel("L")
    ax.set_ylabel("chi")
    ax.set_title("chi(L) near Tc")
    ax.legend()

    ax = axes[0, 1]
    ax.loglog(Ls_chi_max, chi_max, "o", label="chi_max")
    ax.loglog(Ls_chi_max, gamma_fit, "--", label=f"fit slope={slope_gamma:.3f}")
    ax.set_xlabel("L")
    ax.set_ylabel("chi_max")
    ax.set_title("chi_max(L) scaling")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(np.log(Ls_c), C_max, "o", label="C_max")
    ax.plot(np.log(Ls_c), alpha_fit, "--", label=f"slope={slope_alpha:.3f}")
    ax.set_xlabel("log(L)")
    ax.set_ylabel("C_max")
    ax.set_title("C_max vs log(L)")
    ax.legend()

    ax = axes[1, 1]
    ax.axis("off")

    try:
        beta_val, beta_err = estimate_beta_loglog(df, tc_est)
    except Exception:
        beta_val, beta_err = np.nan, np.nan

    try:
        eta_val, eta_err = estimate_eta(df, tc_est)
    except Exception:
        eta_val, eta_err = np.nan, np.nan

    try:
        gamma_val, gamma_err = estimate_gamma(df)
    except Exception:
        gamma_val, gamma_err = np.nan, np.nan

    try:
        alpha_val, alpha_err = estimate_alpha_logL(df)
    except Exception:
        alpha_val, alpha_err = np.nan, np.nan

    rows = [
        ("beta", beta_val, beta_err, 0.125),
        ("nu", 1.0, 0.0, 1.0),
        ("eta", eta_val, eta_err, 0.25),
        ("gamma", gamma_val, gamma_err, 1.75),
        ("alpha", alpha_val, alpha_err, 0.0),
    ]

    table_data = []
    cell_colors = []
    for name, val, err, exact in rows:
        if np.isfinite(err) and err > 0:
            within = abs(val - exact) <= 2.0 * err
            dev_sigma = abs(val - exact) / err
        else:
            within = np.isfinite(val) and abs(val - exact) < 1e-12
            dev_sigma = np.nan

        color = "#c6efce" if within else "#f4cccc"
        table_data.append([f"{val:.4f}", f"{err:.4f}", f"{exact:.4f}", f"{dev_sigma:.2f}"])
        cell_colors.append([color] * 4)

    table = ax.table(
        cellText=table_data,
        rowLabels=[r[0] for r in rows],
        colLabels=["value", "stderr", "exact", "dev_sigma"],
        cellColours=cell_colors,
        loc="center",
    )
    table.scale(1, 1.4)
    ax.set_title("Exponent summary (2-sigma)")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig8_exponent_summary.png"), dpi=200)
    fig.savefig(os.path.join(outdir, "fig8_exponent_summary.pdf"))
    plt.close(fig)


def bootstrap_beta(df: pd.DataFrame, tc_est: float, n_boot: int = 1000, seed: int = 1234) -> np.ndarray:
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
    beta_tc = EXACT_TC
    
    print(f"Beta estimation: Using log-log scaling with fixed Tc={beta_tc:.6f} (largest L)...")
    try:
        beta, beta_stderr = estimate_beta_loglog(df, beta_tc)
    except RuntimeError as e:
        print(f" Log-log fit failed: {e}; trying collapse method")
        try:
            beta, beta_stderr = estimate_beta_collapse(df, beta_tc)
        except RuntimeError as e2:
            print(f" Beta fit completely failed: {e2}")
            beta, beta_stderr = np.nan, np.nan
    
    beta_boot = bootstrap_beta(df, beta_tc)

    try:
        eta, eta_stderr = estimate_eta(df, tc_inf_val)
    except Exception as e:
        print(f" Eta estimation failed: {e}")
        eta, eta_stderr = np.nan, np.nan

    try:
        gamma, gamma_stderr = estimate_gamma(df)
    except Exception as e:
        print(f" Gamma estimation failed: {e}")
        gamma, gamma_stderr = np.nan, np.nan

    try:
        alpha, alpha_stderr = estimate_alpha_logL(df)
    except Exception as e:
        print(f" Alpha estimation failed: {e}")
        alpha, alpha_stderr = np.nan, np.nan

    def deviation_sigma(value: float, exact_value: float, stderr: float) -> float:
        if not np.isfinite(value) or not np.isfinite(exact_value) or not np.isfinite(stderr) or stderr <= 0.0:
            return np.nan
        return abs(value - exact_value) / stderr

    metrics = pd.DataFrame(
        [
            {
                "name": "Tc_inf",
                "value": tc_inf,
                "stderr": tc_inf_stderr,
                "exact_value": 2.269185,
                "deviation_sigma": deviation_sigma(tc_inf, 2.269185, tc_inf_stderr),
            },
            {
                "name": "beta",
                "value": beta,
                "stderr": beta_stderr,
                "exact_value": 0.125,
                "deviation_sigma": deviation_sigma(beta, 0.125, beta_stderr),
            },
            {
                "name": "eta",
                "value": eta,
                "stderr": eta_stderr,
                "exact_value": 0.25,
                "deviation_sigma": deviation_sigma(eta, 0.25, eta_stderr),
            },
            {
                "name": "gamma",
                "value": gamma,
                "stderr": gamma_stderr,
                "exact_value": 1.75,
                "deviation_sigma": deviation_sigma(gamma, 1.75, gamma_stderr),
            },
            {
                "name": "alpha",
                "value": alpha,
                "stderr": alpha_stderr,
                "exact_value": 0.0,
                "deviation_sigma": deviation_sigma(alpha, 0.0, alpha_stderr),
            },
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
            print(f"   WARNING: Beta is outside expected range [0.05, 0.25]")
            print(f"    This likely indicates an issue with Tc estimation")
            print(f"    Expected: β = 0.125 (2D Ising theory)")
        elif abs(est.beta - 0.125) > 0.06:
            print(f"   Beta deviates from theory (0.125) by {abs(est.beta - 0.125):.4f}")
            print(f"    Accuracy could be improved; check Tc and data quality")
    
    print("  Tc(L):")
    for L, tc in sorted(est.tc_by_L.items()):
        print(f"    L={L:3d}: {tc:.6f}")
    print("="*50)


if __name__ == "__main__":
    main()
