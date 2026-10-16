import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from load_data import load_ising_csv

EXACT_TC = 2.0 / np.log(1.0 + np.sqrt(2.0))


@dataclass
class CriticalEstimates:
    tc_by_L: Dict[int, float]
    tc_inf: float
    tc_inf_stderr: float
    tc_binder: float
    tc_binder_stderr: float
    gamma: float
    gamma_stderr: float
    eta: float
    eta_stderr: float
    eta_tc_sys: float
    alpha_slope: float
    alpha_slope_stderr: float


def _quadratic_peak(t: np.ndarray, y: np.ndarray, half_width: int = 3) -> Tuple[float, float] | None:
    """Peak of y(t) from a quadratic fit to the points around the raw maximum.

    Returns None when the maximum sits at the edge of the grid (no bracketed peak).
    """
    i = int(np.argmax(y))
    if i == 0 or i == len(y) - 1:
        return None
    lo, hi = max(0, i - half_width), min(len(y), i + half_width + 1)
    if hi - lo < 3:
        return None
    a, b, c = np.polyfit(t[lo:hi], y[lo:hi], 2)
    if a >= 0:
        return float(t[i]), float(y[i])
    t_peak = -b / (2.0 * a)
    if not (t[lo] <= t_peak <= t[hi - 1]):
        return float(t[i]), float(y[i])
    return float(t_peak), float(np.polyval([a, b, c], t_peak))


def drop_partial_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Drop sizes whose scan does not cover the file's full T range (e.g. an interrupted run)."""
    t_lo, t_hi = df["T"].min(), df["T"].max()
    coverage = df.groupby("L")["T"].agg(["min", "max"])
    partial = coverage[(coverage["min"] > t_lo + 1e-6) | (coverage["max"] < t_hi - 1e-6)]
    for L, r in partial.iterrows():
        print(f"WARNING: L={L} only covers T in [{r['min']:.4f}, {r['max']:.4f}] of "
              f"[{t_lo:.4f}, {t_hi:.4f}]; excluded from the analysis")
    return df[~df["L"].isin(partial.index)]


def attach_u_errors(df: pd.DataFrame, series_dir: str, n_blocks: int = 20, n_boot: int = 200,
                    seed: int = 12345) -> pd.DataFrame:
    """Add a U_err column from block-bootstrapping each row's raw time series.

    Series files are the ones written by the drivers' --series-dir option. Rows with
    no series file get U_err = NaN.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    errs = []
    short_blocks = []
    for _, row in df.iterrows():
        path = os.path.join(series_dir, f"series_{row['algo']}_L{int(row['L'])}_T{row['T']:.6f}.csv")
        if not os.path.exists(path):
            errs.append(np.nan)
            continue
        m = pd.read_csv(path, comment="#")["m"].to_numpy()
        block_len = len(m) // n_blocks
        if block_len < 10 * max(row["tau_absM"], 0.5):
            short_blocks.append((int(row["L"]), float(row["T"])))
        m2 = (m[: block_len * n_blocks] ** 2).reshape(n_blocks, block_len)
        m4 = m2 * m2
        b2, b4 = m2.mean(axis=1), m4.mean(axis=1)
        picks = rng.integers(0, n_blocks, (n_boot, n_blocks))
        u = 1.0 - b4[picks].mean(axis=1) / (3.0 * b2[picks].mean(axis=1) ** 2)
        errs.append(float(np.std(u, ddof=1)))
    df["U_err"] = errs
    n_found = int(np.isfinite(df["U_err"]).sum())
    print(f"U errors from series in {series_dir}: {n_found}/{len(df)} rows")
    if short_blocks:
        print(f"  WARNING: {len(short_blocks)} rows have blocks shorter than 10 tau; their U_err is "
              f"underestimated (worst: L={max(short_blocks)[0]})")
    return df


def estimate_tc_binder(df: pd.DataFrame, min_n_eff: float = 100.0) -> Tuple[float, float, list]:
    """Binder-cumulant crossings of adjacent sizes on their common T grid (raw, unsmoothed U).

    A pair's crossing is accepted only if
      * U1 - U2 changes sign exactly once (several sign changes mean noise, not a crossing);
        with a U_err column, only points where |U1 - U2| > 2 sigma count;
      * every point in the bracketing interval has n_eff >= min_n_eff for both sizes.
    Returns (Tc of the largest accepted pair, its error or NaN, list of per-pair records).
    """
    sizes = sorted(df["L"].unique())
    if len(sizes) < 2:
        raise ValueError("Need at least 2 lattice sizes for Binder crossing")
    has_err_col = "U_err" in df.columns

    print(f"  Binder crossings (raw U; need one sign change and n_eff >= {min_n_eff:g}):")
    records = []
    for L1, L2 in zip(sizes[:-1], sizes[1:]):
        cols = ["T", "U", "n_eff"] + (["U_err"] if has_err_col else [])
        g1 = df[df["L"] == L1][cols].assign(key=lambda g: (g["T"] * 1e6).round())
        g2 = df[df["L"] == L2][cols].assign(key=lambda g: (g["T"] * 1e6).round())
        g = g1.merge(g2, on="key", suffixes=("1", "2")).sort_values("T1")
        rec = {"L1": int(L1), "L2": int(L2), "Tc": np.nan, "err": np.nan, "status": ""}
        records.append(rec)
        if len(g) < 3:
            rec["status"] = "fewer than 3 common temperatures"
            print(f"    L={L1}/{L2}: {rec['status']}")
            continue

        t = g["T1"].to_numpy()
        diff = (g["U1"] - g["U2"]).to_numpy()
        use_err = has_err_col and np.isfinite(g["U_err1"]).all() and np.isfinite(g["U_err2"]).all()
        sigma = np.sqrt(g["U_err1"] ** 2 + g["U_err2"] ** 2).to_numpy() if use_err else None
        keep = np.abs(diff) > 2.0 * sigma if use_err else np.ones_like(diff, dtype=bool)
        kept = np.flatnonzero(keep)
        changes = np.flatnonzero(np.diff(np.sign(diff[kept])) != 0)
        if changes.size != 1:
            rec["status"] = ("no crossing" if changes.size == 0 else
                             f"ambiguous: {changes.size} sign changes" + ("" if use_err else " (no U errors; noise)"))
            print(f"    L={L1}/{L2}: {rec['status']}")
            continue

        lo, hi = kept[changes[0]], kept[changes[0] + 1]
        n_eff_min = float(min(g["n_eff1"].iloc[lo:hi + 1].min(), g["n_eff2"].iloc[lo:hi + 1].min()))
        span = slice(lo, hi + 1)
        if use_err and hi - lo >= 1:
            # Weighted straight line through the bracket; root and its error from the covariance.
            w = 1.0 / sigma[span]
            (b, a), cov = np.polyfit(t[span], diff[span], 1, w=w, cov="unscaled")
            tc = -a / b
            grad = np.array([a / b**2, -1.0 / b])  # d(tc)/d(b), d(tc)/d(a)
            rec["err"] = float(np.sqrt(grad @ cov @ grad))
        else:
            t1, t2, d1, d2 = t[lo], t[hi], diff[lo], diff[hi]
            tc = t1 - d1 * (t2 - t1) / (d2 - d1)
        rec["Tc"] = float(tc)
        if n_eff_min < min_n_eff:
            rec["status"] = f"unreliable: n_eff = {n_eff_min:.0f} near crossing"
        else:
            rec["status"] = "ok"
        err_txt = f" +/- {rec['err']:.5f}" if np.isfinite(rec["err"]) else ""
        print(f"    L={L1}/{L2}: Tc = {tc:.5f}{err_txt}  [{rec['status']}]")

    accepted = [r for r in records if r["status"] == "ok"]
    if not accepted:
        raise RuntimeError("No reliable Binder crossing (see per-pair status above)")
    best = accepted[-1]
    return best["Tc"], best["err"], records


def estimate_tc_finite_size(df: pd.DataFrame) -> Tuple[Dict[int, float], float, float]:
    """Tc(L) from the susceptibility peak, extrapolated as Tc(L) = Tc + a / L (nu = 1).

    Every size with a bracketed peak is used, unweighted. The returned stderr is the
    regression error of the intercept; with no per-point errors (one chain per (T, L))
    it does not include the statistical error of each peak position.
    """
    tc_by_L = {}
    for L, g in df.groupby("L"):
        g = g.sort_values("T")
        peak = _quadratic_peak(g["T"].to_numpy(), g["chi"].to_numpy())
        if peak is None:
            print(f"  WARNING: L={int(L)} chi peak at the edge of the T grid; not used")
            continue
        tc_by_L[int(L)] = peak[0]

    if len(tc_by_L) < 3:
        raise RuntimeError(f"Need >= 3 sizes with a chi peak to extrapolate Tc, have {sorted(tc_by_L)}")

    Ls = np.array(sorted(tc_by_L), dtype=float)
    tcL = np.array([tc_by_L[int(L)] for L in Ls])
    fit = stats.linregress(1.0 / Ls, tcL)
    tc_inf = float(fit.intercept)
    tc_inf_stderr = float(fit.intercept_stderr)

    print(f"  chi-peak Tc(L): " + ", ".join(f"L={int(L)}: {t:.4f}" for L, t in zip(Ls, tcL)))
    print(f"  Tc(L) = {fit.intercept:.5f} + {fit.slope:.4f}/L  (R^2 = {fit.rvalue**2:.4f}, {len(Ls)} sizes)")
    print(f"  Tc(inf) = {tc_inf:.5f} +/- {tc_inf_stderr:.5f} (regression error only)")
    return tc_by_L, tc_inf, tc_inf_stderr


def estimate_eta(df: pd.DataFrame, tc_est: float, tc_err: float = 0.0) -> Tuple[float, float, float]:
    """eta from chi(Tc, L) = N <m^2> / T ~ L^(2 - eta), chi interpolated to T = tc_est per L.

    Returns (eta, regression stderr, systematic shift from moving Tc by +/- tc_err).
    """
    def fit_at(T: float) -> float:
        records = []
        for L, g in df.groupby("L"):
            g = g.sort_values("T")
            t = g["T"].to_numpy()
            if not (t[0] <= T <= t[-1]):
                continue
            records.append((float(L), float(np.interp(T, t, g["chi_full"].to_numpy()))))
        if len(records) < 3:
            raise ValueError("Need at least 3 lattice sizes whose T grid brackets Tc to estimate eta.")
        Ls, chis = np.array(records).T
        return stats.linregress(np.log(Ls), np.log(chis))

    fit = fit_at(tc_est)
    eta = 2.0 - fit.slope
    tc_sys = 0.0
    if tc_err > 0:
        shifted = [2.0 - fit_at(tc_est + d * tc_err).slope for d in (-1.0, 1.0)]
        tc_sys = 0.5 * abs(shifted[1] - shifted[0])
    return float(eta), float(fit.stderr), float(tc_sys)


def estimate_gamma(df: pd.DataFrame) -> Tuple[float, float]:
    """gamma/nu from chi_max ~ L^(gamma/nu); equals gamma since nu = 1 in 2D."""
    if "chi" not in df.columns:
        raise ValueError("chi column missing; run load_ising_csv to compute chi.")

    records = []
    for L, g in df.groupby("L"):
        g = g.sort_values("T")
        peak = _quadratic_peak(g["T"].to_numpy(), g["chi"].to_numpy())
        if peak is not None and peak[1] > 0:
            records.append((float(L), peak[1]))

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
        g = g.sort_values("T")
        peak = _quadratic_peak(g["T"].to_numpy(), g["C"].to_numpy())
        if peak is None:
            print(f"  WARNING: L={int(L)} C peak at the edge of the T grid; not used for alpha")
            continue
        records.append((float(L), peak[1]))

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


def run_analysis(
    data_csv: str,
    metrics_csv: str = "./data_outputs/analysis_metrics.csv",
    series_dir: str | None = None,
    min_n_eff: float = 100.0,
) -> CriticalEstimates:
    """Temperature-scan analysis: Tc(L), gamma, eta, alpha.

    beta/nu and the error-controlled Tc come from fss_analysis.py, which works on the
    raw time series of fixed-T runs. Errors here are regression errors only: the
    scan stores one chain per (T, L) and no time series, so they ignore the
    statistical error of each point. See the n_eff column for how many independent
    measurements each point really has. If the scan was run with --series-dir, pass
    that directory to get per-point Binder errors and error-weighted crossings.
    """
    df = drop_partial_sizes(load_ising_csv(data_csv))
    if series_dir:
        df = attach_u_errors(df, series_dir)
    print(f"Lattice sizes: {sorted(df['L'].unique().tolist())}")

    near = df[(df["T"] - EXACT_TC).abs() < 0.05]
    if len(near):
        summary = near.groupby("L").agg(tau_absM=("tau_absM", "max"), n_eff=("n_eff", "min"))
        print("Near Tc (|T - 2.269| < 0.05): worst-case autocorrelation per L (tau in measurements)")
        for L, r in summary.iterrows():
            print(f"    L={L:4d}: tau_absM <= {r['tau_absM']:8.1f}, n_eff >= {r['n_eff']:8.1f}")

    os.makedirs(os.path.dirname(metrics_csv) or ".", exist_ok=True)

    print("Tc from susceptibility peaks:")
    tc_by_L, tc_inf, tc_inf_stderr = estimate_tc_finite_size(df)
    print("Tc from Binder crossings:")
    try:
        tc_binder, tc_binder_err, _ = estimate_tc_binder(df, min_n_eff=min_n_eff)
    except (RuntimeError, ValueError) as e:
        print(f"  Binder crossing: {e}")
        tc_binder, tc_binder_err = np.nan, np.nan

    try:
        eta, eta_stderr, eta_tc_sys = estimate_eta(df, tc_inf, tc_inf_stderr)
    except Exception as e:
        print(f" Eta estimation failed: {e}")
        eta, eta_stderr, eta_tc_sys = np.nan, np.nan, np.nan

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

    rows = [
        ("Tc_chi_peak", tc_inf, tc_inf_stderr, EXACT_TC, "regression error of Tc(L)=Tc+a/L"),
        ("Tc_binder_largest_reliable_pair", tc_binder, tc_binder_err, EXACT_TC,
         "weighted-fit error from per-point U errors" if np.isfinite(tc_binder_err) else "no U errors (scan without --series-dir)"),
        ("gamma_over_nu", gamma, gamma_stderr, 1.75, "regression error"),
        ("eta", eta, eta_stderr, 0.25,
         f"regression error; moving Tc by its stderr shifts eta by {eta_tc_sys:.4f}"),
        ("alpha_logL_slope", alpha, alpha_stderr, np.nan, "C_max = a + b ln L; b > 0 expected (alpha = 0, log)"),
    ]
    metrics = pd.DataFrame(
        [
            {
                "name": name,
                "value": value,
                "stderr": err,
                "exact_value": exact,
                "deviation_sigma": deviation_sigma(value, exact, err),
                "error_kind": kind,
            }
            for name, value, err, exact, kind in rows
        ]
    )
    metrics.to_csv(metrics_csv, index=False)

    tc_by_l_csv = os.path.join(os.path.dirname(metrics_csv) or ".", "tc_by_L.csv")
    pd.DataFrame(
        [{"L": L, "Tc_L": tc} for L, tc in sorted(tc_by_L.items())]
    ).to_csv(tc_by_l_csv, index=False)

    return CriticalEstimates(
        tc_by_L=tc_by_L,
        tc_inf=tc_inf,
        tc_inf_stderr=tc_inf_stderr,
        tc_binder=tc_binder,
        tc_binder_stderr=tc_binder_err,
        gamma=gamma, gamma_stderr=gamma_stderr,
        eta=eta, eta_stderr=eta_stderr, eta_tc_sys=eta_tc_sys,
        alpha_slope=alpha, alpha_slope_stderr=alpha_stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Temperature-scan analysis for the 2D Ising model "
                                                 "(beta/nu: see fss_analysis.py).")
    parser.add_argument("--data", default="./data_outputs/data.csv", help="Input CSV from C++/CUDA simulation")
    parser.add_argument("--metrics", default="./data_outputs/analysis_metrics.csv", help="Output metrics CSV")
    parser.add_argument("--series-dir", default=None,
                        help="Time series written by the scan's --series-dir; enables per-point U errors")
    parser.add_argument("--min-n-eff", type=float, default=100.0,
                        help="Reject Binder crossings where a point in the bracket has fewer independent samples")
    args = parser.parse_args()

    est = run_analysis(args.data, args.metrics, args.series_dir, args.min_n_eff)

    def fmt(value: float, err: float) -> str:
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.5f} +/- {err:.5f}" if np.isfinite(err) else f"{value:.5f} (no error estimate)"

    print("\n" + "=" * 60)
    print("Analysis complete (scan; errors are regression-only unless noted)")
    print("=" * 60)
    print(f"  Tc (chi peaks)            = {fmt(est.tc_inf, est.tc_inf_stderr)}")
    print(f"  Tc (Binder, largest ok)   = {fmt(est.tc_binder, est.tc_binder_stderr)}")
    print(f"  exact Tc                  = {EXACT_TC:.5f}")
    print(f"  gamma/nu (chi_max)        = {fmt(est.gamma, est.gamma_stderr)}   exact 1.75")
    print(f"  eta (chi at Tc_chi)       = {fmt(est.eta, est.eta_stderr)}   exact 0.25"
          + (f"   [Tc +/- stderr shifts eta by {est.eta_tc_sys:.4f}]" if np.isfinite(est.eta_tc_sys) else ""))
    print(f"  C_max vs ln L slope       = {fmt(est.alpha_slope, est.alpha_slope_stderr)}   "
          f"exact (8/pi) Kc^2 = {8.0 / np.pi / EXACT_TC**2:.4f} (alpha = 0, log)")
    print("  Tc(L) from chi peaks:")
    for L, tc in sorted(est.tc_by_L.items()):
        print(f"    L={L:4d}: {tc:.5f}")
    print("  beta/nu, and Tc with bootstrap errors: run fss_analysis.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
