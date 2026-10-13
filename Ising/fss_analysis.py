"""Finite-size scaling at fixed temperature from raw time series.

Workflow (see Makefile target `fss`):
  ./ising2d_wolff --sizes=16,32,64,128,256 --tmin=2.269185314 --tmax=2.269185314 \
      --no-adaptive --therm=10000 --meas=60000 --stride=1 \
      --out=data_outputs/fss_wolff.csv --series-dir=data_outputs/series
  python fss_analysis.py --series-dir data_outputs/series

Every (L) is a single Markov chain at T0. From it this script estimates
  * Tc from Binder-cumulant crossings of adjacent sizes, using single-histogram
    reweighting of each chain to nearby T;
  * beta/nu from <|m|> ~ L^(-beta/nu) and gamma/nu from chi ~ L^(gamma/nu), both at T0
    and at the Binder-crossing Tc measured here (so the exponent does not have to
    lean on the exact Tc);
  * all errors from a block bootstrap: each chain is cut into contiguous blocks much
    longer than its autocorrelation time and blocks are resampled with replacement,
    so correlated samples are not counted as independent.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.special import logsumexp

EXACT_TC = 2.0 / np.log(1.0 + np.sqrt(2.0))
# Binder cumulant at Tc for L -> infinity, periodic square lattice:
# <m^4>/<m^2>^2 = 1.1679229(47), Salas & Sokal, J. Stat. Phys. 98, 551 (2000).
EXACT_U_STAR = 1.0 - 1.1679229 / 3.0


def exact_value(name: str) -> float:
    base = name.split("@")[0]
    if base.startswith("Tc"):
        return EXACT_TC
    if base.startswith("U_L"):
        return EXACT_U_STAR
    return {"beta_over_nu": 0.125, "gamma_over_nu": 1.75, "gamma_over_nu_full": 1.75, "eta": 0.25}[base]


def load_series(series_dir: str, algo: str | None = None) -> list[dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(series_dir, "series_*.csv"))):
        with open(path, "r", encoding="utf-8") as handle:
            meta_line = handle.readline().strip()
        meta = dict(re.findall(r"(\w+)=(\S+)", meta_line))
        if not {"T", "L", "algo", "seed"} <= meta.keys():
            raise ValueError(f"{path}: missing metadata line, got {meta_line!r}")
        if algo is not None and meta["algo"] != algo:
            continue
        data = pd.read_csv(path, comment="#")
        if list(data.columns) != ["m", "e"] or data.isna().any().any():
            raise ValueError(f"{path}: malformed series data")
        runs.append({
            "path": path,
            "L": int(meta["L"]),
            "T0": float(meta["T"]),
            "algo": meta["algo"],
            "seed": int(meta["seed"]),
            "m": data["m"].to_numpy(),
            "e": data["e"].to_numpy(),
        })

    if not runs:
        raise ValueError(f"No series files in {series_dir}" + (f" for algo={algo}" if algo else ""))
    algos = {r["algo"] for r in runs}
    if len(algos) > 1:
        raise ValueError(f"Series from several algorithms {sorted(algos)}; pick one with --algo")
    temps = {round(r["T0"], 6) for r in runs}
    if len(temps) > 1:
        raise ValueError(f"Series at several temperatures {sorted(temps)}; this analysis needs one T0")
    sizes = [r["L"] for r in runs]
    if len(set(sizes)) != len(sizes):
        raise ValueError(f"Duplicate lattice sizes in {series_dir}: {sorted(sizes)}")
    return sorted(runs, key=lambda r: r["L"])


def integrated_tau(x: np.ndarray, c: float = 6.0) -> float:
    """Integrated autocorrelation time (in samples), Sokal automatic window."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / np.arange(n, 0, -1)
    if acf[0] <= 0:
        return 0.5
    rho = acf / acf[0]
    tau = 0.5 + np.cumsum(rho[1:])
    window = np.arange(1, n)
    ok = np.nonzero(window >= c * tau)[0]
    return float(tau[ok[0]] if ok.size else tau[-1])


def reweighted_moments(m: np.ndarray, e: np.ndarray, L: int, T0: float, temps: np.ndarray) -> dict:
    """<|m|>, <m^2>, <m^4> at each T in temps, reweighted from a chain at T0."""
    N = L * L
    temps = np.atleast_1d(np.asarray(temps, dtype=float))
    dbeta = 1.0 / temps - 1.0 / T0
    logw = -dbeta[:, None] * (e * N)[None, :]
    w = np.exp(logw - logsumexp(logw, axis=1, keepdims=True))
    m2 = m * m
    return {"absm": w @ np.abs(m), "m2": w @ m2, "m4": w @ (m2 * m2)}


def derived(mom: dict, L: int, temps: np.ndarray) -> dict:
    N = L * L
    return {
        "absm": mom["absm"],
        "U": 1.0 - mom["m4"] / (3.0 * mom["m2"] ** 2),
        "chi_prime": N * (mom["m2"] - mom["absm"] ** 2) / temps,  # N(<m^2> - <|m|>^2)/T
        "chi_full": N * mom["m2"] / temps,                          # N <m^2> / T
    }


def reweight_halfwidth(e: np.ndarray, L: int, T0: float) -> float:
    """Shift in T that moves <E> by about one standard deviation of E."""
    sigma_E = np.std(e) * L * L
    return T0**2 / sigma_E


def binder_crossing(t_grid: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> float:
    diff = u1 - u2
    idx = np.nonzero(np.diff(np.sign(diff)) != 0)[0]
    if idx.size == 0:
        return np.nan
    # Several sign changes only happen when the curves are noisy; take the one nearest the middle.
    i = idx[np.argmin(np.abs(idx - len(t_grid) / 2))]
    t1, t2, d1, d2 = t_grid[i], t_grid[i + 1], diff[i], diff[i + 1]
    return float(t1 - d1 * (t2 - t1) / (d2 - d1))


def loglog_slope(Ls: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log(Ls), np.log(y), 1)[0])


def analyze(chains: list[tuple[int, np.ndarray, np.ndarray]], T0: float, lmin: int,
            halfwidths: dict, n_grid: int = 81) -> dict:
    """All estimates from one set of chains. chains: [(L, m, e)] sorted by L."""
    out = {}
    Ls = np.array([L for L, _, _ in chains], dtype=float)

    # Binder crossings of adjacent sizes, each pair on its own reweighting window.
    crossings = []
    for (L1, m1, e1), (L2, m2, e2) in zip(chains[:-1], chains[1:]):
        hw = min(halfwidths[L1], halfwidths[L2])
        t_grid = np.linspace(T0 - hw, T0 + hw, n_grid)
        u1 = derived(reweighted_moments(m1, e1, L1, T0, t_grid), L1, t_grid)["U"]
        u2 = derived(reweighted_moments(m2, e2, L2, T0, t_grid), L2, t_grid)["U"]
        tc = binder_crossing(t_grid, u1, u2)
        out[f"Tc_binder_{L1}_{L2}"] = tc
        crossings.append(tc)
    out["Tc_binder"] = crossings[-1] if crossings else np.nan

    fit = Ls >= lmin
    for label, T in (("T0", T0), ("Tc_binder", out["Tc_binder"])):
        if not np.isfinite(T):
            for key in ("beta_over_nu", "gamma_over_nu", "gamma_over_nu_full", "eta"):
                out[f"{key}@{label}"] = np.nan
            continue
        obs = [derived(reweighted_moments(m, e, L, T0, T), L, np.array([T])) for L, m, e in chains]
        absm = np.array([o["absm"][0] for o in obs])
        chi_p = np.array([o["chi_prime"][0] for o in obs])
        chi_f = np.array([o["chi_full"][0] for o in obs])
        out[f"beta_over_nu@{label}"] = -loglog_slope(Ls[fit], absm[fit])
        out[f"gamma_over_nu@{label}"] = loglog_slope(Ls[fit], chi_p[fit])
        out[f"gamma_over_nu_full@{label}"] = loglog_slope(Ls[fit], chi_f[fit])
        out[f"eta@{label}"] = 2.0 - out[f"gamma_over_nu_full@{label}"]
        if label == "T0":
            out["_absm_T0"], out["_chi_full_T0"] = absm, chi_f
            for (L, _, _), o in zip(chains, obs):
                out[f"U_L{L}@T0"] = float(o["U"][0])
    return out


def block_indices(n: int, n_blocks: int, rng: np.random.Generator) -> np.ndarray:
    block_len = n // n_blocks
    starts = rng.integers(0, n_blocks, n_blocks) * block_len
    return (starts[:, None] + np.arange(block_len)[None, :]).ravel()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--series-dir", default="./data_outputs/series")
    parser.add_argument("--algo", default=None, help="Only use series from this algorithm")
    parser.add_argument("--lmin", type=int, default=0, help="Smallest L included in the exponent fits")
    parser.add_argument("--n-blocks", type=int, default=50)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--discard", type=int, default=0, help="Extra leading samples to drop per chain")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--metrics", default="./data_outputs/fss_metrics.csv")
    parser.add_argument("--plot", default="./plots/fss_scaling.png")
    args = parser.parse_args()

    runs = load_series(args.series_dir, args.algo)
    T0 = runs[0]["T0"]
    chains = [(r["L"], r["m"][args.discard:], r["e"][args.discard:]) for r in runs]
    Ls = np.array([L for L, _, _ in chains], dtype=float)
    if len(chains) < 3:
        raise ValueError("Need at least 3 lattice sizes")

    print(f"algo={runs[0]['algo']}  seed={runs[0]['seed']}  T0={T0:.9f}  (exact Tc={EXACT_TC:.9f})")
    print(f"{'L':>5} {'samples':>8} {'tau|m|':>8} {'tauE':>8} {'block/tau':>10} {'n_eff':>8} {'dT_rw':>8}")
    halfwidths = {}
    for L, m, e in chains:
        tau = max(integrated_tau(np.abs(m)), integrated_tau(e))
        block_len = len(m) // args.n_blocks
        halfwidths[L] = reweight_halfwidth(e, L, T0)
        ratio = block_len / tau
        print(f"{L:>5} {len(m):>8} {integrated_tau(np.abs(m)):>8.2f} {integrated_tau(e):>8.2f} "
              f"{ratio:>10.1f} {len(m) / (2 * tau):>8.0f} {halfwidths[L]:>8.4f}"
              + ("   WARNING: blocks < 10 tau, errors underestimated" if ratio < 10 else ""))

    central = analyze(chains, T0, args.lmin, halfwidths)

    rng = np.random.default_rng(args.seed)
    boots = []
    for _ in range(args.n_boot):
        resampled = []
        for L, m, e in chains:
            idx = block_indices(len(m), args.n_blocks, rng)
            resampled.append((L, m[idx], e[idx]))
        boots.append(analyze(resampled, T0, args.lmin, halfwidths))

    keys = [k for k in central if not k.startswith("_")]
    rows = []
    for k in keys:
        samples = np.array([b[k] for b in boots], dtype=float)
        finite = samples[np.isfinite(samples)]
        rows.append({
            "name": k,
            "value": central[k],
            "stderr": float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan,
            "boot_failures": int(samples.size - finite.size),
        })
    metrics = pd.DataFrame(rows)
    metrics["exact_value"] = [exact_value(n) for n in metrics["name"]]
    metrics["deviation_sigma"] = (metrics["value"] - metrics["exact_value"]).abs() / metrics["stderr"]
    metrics["T0"] = T0
    metrics["lmin_fit"] = int(Ls[Ls >= args.lmin].min())
    metrics["L_values"] = ",".join(str(int(L)) for L in Ls)

    os.makedirs(os.path.dirname(args.metrics) or ".", exist_ok=True)
    metrics.to_csv(args.metrics, index=False)

    fit_Ls = [int(L) for L in Ls if L >= args.lmin]
    print(f"\nBlock bootstrap: {args.n_boot} replicates x {args.n_blocks} blocks per chain; fits use L={fit_Ls}")
    print(f"{'quantity':<30} {'value':>10} {'stderr':>9} {'exact':>9} {'dev/sigma':>9}")
    for _, r in metrics.iterrows():
        note = f"  ({r['boot_failures']} replicates without crossing)" if r["boot_failures"] else ""
        print(f"{r['name']:<30} {r['value']:>10.5f} {r['stderr']:>9.5f} {r['exact_value']:>9.5f} "
              f"{r['deviation_sigma']:>9.2f}{note}")
    print("U_L@T0 is compared with the L -> infinity value U* (finite-L corrections are small but nonzero);"
          "\neta = 2 - gamma_over_nu_full.")
    print(f"\nSaved {args.metrics}")

    if args.plot:
        plot_fss(chains, T0, central, halfwidths, args.lmin, args.plot)
        print(f"Saved {args.plot}")


def plot_fss(chains, T0, central, halfwidths, lmin, path) -> None:
    import matplotlib.pyplot as plt

    Ls = np.array([L for L, _, _ in chains], dtype=float)
    fit = Ls >= lmin
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    for ax, y, key, sign, ylabel, theory in (
        (axes[0], central["_absm_T0"], "beta_over_nu@T0", -1, r"$\langle|m|\rangle$", -0.125),
        (axes[1], central["_chi_full_T0"], "gamma_over_nu_full@T0", 1, r"$\chi = N\langle m^2\rangle/T$", 1.75),
    ):
        slope = sign * central[key]
        coeffs = np.polyfit(np.log(Ls[fit]), np.log(y[fit]), 1)
        xs = np.geomspace(Ls.min(), Ls.max(), 50)
        ax.loglog(Ls, y, "o", color="C0", ms=7, label=f"T = {T0:.6f}")
        ax.loglog(xs, np.exp(np.polyval(coeffs, np.log(xs))), "-", color="C1", lw=1.5,
                  label=f"fit slope {slope:+.4f}")
        ax.loglog(xs, np.exp(np.polyval([theory, coeffs[1] + (coeffs[0] - theory) * np.log(Ls[fit]).mean()],
                                        np.log(xs))), "--", color="0.5", lw=1.2,
                  label=f"exact slope {theory:+.3f}")
        ax.set_xticks(Ls, [str(int(L)) for L in Ls])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("L")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    hw = min(halfwidths.values())
    t_grid = np.linspace(T0 - hw, T0 + hw, 81)
    for i, (L, m, e) in enumerate(chains):
        u = derived(reweighted_moments(m, e, L, T0, t_grid), L, t_grid)["U"]
        ax.plot(t_grid, u, color=f"C{i}", lw=1.8, label=f"L={L}")
    ax.axvline(EXACT_TC, color="0.4", ls="--", lw=1, label="exact Tc")
    if np.isfinite(central["Tc_binder"]):
        ax.axvline(central["Tc_binder"], color="k", ls=":", lw=1.2, label="largest-pair crossing")
    ax.set_xlabel("T (reweighted)")
    ax.set_ylabel("Binder cumulant U")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
