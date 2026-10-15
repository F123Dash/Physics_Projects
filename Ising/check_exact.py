"""Compare each simulator with exact enumeration of the 4x4 periodic lattice.

Usage: python check_exact.py [--binaries ./ising2d_cpu ./ising2d_wolff ...]
Exits non-zero if any observable is more than --zmax standard errors off.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

L = 4


def exact_observables(T: float) -> dict:
    n = L * L
    states = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    s = states.reshape(-1, L, L)
    energy = -(s * np.roll(s, 1, axis=1) + s * np.roll(s, 1, axis=2)).sum(axis=(1, 2)) / n
    m = s.sum(axis=(1, 2)) / n
    logw = -energy * n / T
    w = np.exp(logw - logw.max())
    w /= w.sum()
    return {
        "E": float(w @ energy),
        "absM": float(w @ np.abs(m)),
        "M2": float(w @ m**2),
        "E2": float(w @ energy**2),
        "M4": float(w @ m**4),
    }


def run_binary(binary: str, T: float, workdir: str) -> pd.Series:
    out = os.path.join(workdir, os.path.basename(binary) + f"_{T}.csv")
    is_wolff = "wolff" in binary
    meas = 400000 if is_wolff else 200000
    cmd = [binary, f"--sizes={L}", f"--tmin={T}", f"--tmax={T}", "--no-adaptive",
           "--therm=2000", f"--meas={meas}", "--stride=1", f"--out={out}", "--seed=2024"]
    subprocess.run(cmd, check=True, capture_output=True)
    return pd.read_csv(out).iloc[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binaries", nargs="+", default=["./ising2d_cpu", "./ising2d_wolff"])
    parser.add_argument("--temps", nargs="+", type=float, default=[1.5, 2.269185, 3.0])
    parser.add_argument("--zmax", type=float, default=5.0)
    args = parser.parse_args()

    failed = False
    with tempfile.TemporaryDirectory() as workdir:
        for T in args.temps:
            exact = exact_observables(T)
            for binary in args.binaries:
                if not os.path.exists(binary):
                    print(f"skip {binary}: not built")
                    continue
                row = run_binary(binary, T, workdir)
                # Standard error inflated by tau(|m|); a rough but conservative proxy for e too.
                n_eff = row["n_samples"] / max(1.0, 2.0 * row["tau_absM"])
                checks = {
                    "E": (row["E"], np.sqrt(max(row["E2"] - row["E"] ** 2, 1e-12) / n_eff)),
                    "absM": (row["absM"], np.sqrt(max(row["M2"] - row["absM"] ** 2, 1e-12) / n_eff)),
                }
                for name, (value, err) in checks.items():
                    z = abs(value - exact[name]) / err
                    status = "ok" if z < args.zmax else "FAIL"
                    failed |= status == "FAIL"
                    print(f"T={T:<8} {row['algo']:<16} {name:<5} sim={value:+.5f} "
                          f"exact={exact[name]:+.5f} err={err:.5f} z={z:5.2f} {status}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
