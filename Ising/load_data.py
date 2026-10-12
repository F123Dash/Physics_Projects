import csv

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["T", "L", "M", "absM", "E", "M2", "E2", "M4", "n_samples", "tau_absM", "algo", "seed"]


def load_ising_csv(path: str) -> pd.DataFrame:
    """Load a simulation CSV, refusing anything ambiguous.

    Raises on: a missing column (e.g. files written before the algo/seed columns
    existed), any row whose field count differs from the header, non-numeric
    values, and duplicate (T, L) rows.
    """
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Empty CSV: {path}") from None

        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ValueError(
                f"{path}: missing columns {missing}. Files from before the algo/seed "
                "columns were added must be regenerated."
            )

        rows = []
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"{path}:{line_no}: expected {len(header)} fields, got {len(row)}")
            rows.append(row)

    df = pd.DataFrame(rows, columns=header)
    numeric = [c for c in header if c != "algo"]
    for col in numeric:
        converted = pd.to_numeric(df[col], errors="coerce")
        bad = converted.isna()
        if bad.any():
            line_no = int(np.flatnonzero(bad.to_numpy())[0]) + 2
            raise ValueError(f"{path}: non-numeric value in column {col!r} (first at line ~{line_no})")
        df[col] = converted
    df["L"] = df["L"].astype(int)
    df["seed"] = df["seed"].astype("uint64")

    t_key = (df["T"] * 1e6).round().astype(np.int64)
    dup = pd.Series(list(zip(df["L"], t_key))).duplicated(keep=False)
    if dup.any():
        sample = df.loc[dup.to_numpy(), ["T", "L", "algo", "seed"]].head(6)
        raise ValueError(f"{path}: duplicate (T, L) rows:\n{sample.to_string(index=False)}")

    runs = df.groupby(["algo", "seed"])["L"].unique()
    print(f"Loaded {len(df)} rows from {path}:")
    for (algo, seed), Ls in runs.items():
        print(f"  algo={algo} seed={seed} L={sorted(int(x) for x in Ls)}")
    algos_per_L = df.groupby("L")["algo"].nunique()
    if (algos_per_L > 1).any():
        print(f"  WARNING: sizes {list(algos_per_L[algos_per_L > 1].index)} mix algorithms across T")

    df = df.copy()
    N = df["L"].astype(float) ** 2
    # Per-spin observables in the file, so fluctuations carry a factor N.
    # chi uses <|M|> to suppress global-flip artifacts at finite L.
    df["chi"] = N * (df["M2"] - df["absM"] ** 2) / df["T"]
    df["chi_full"] = N * df["M2"] / df["T"]  # no subtraction; cleaner L^(gamma/nu) scaling at Tc
    df["C"] = N * (df["E2"] - df["E"] ** 2) / (df["T"] ** 2)
    df["U"] = 1.0 - df["M4"] / (3.0 * df["M2"].clip(lower=1e-12) ** 2)
    # Effective number of independent measurements of |M|.
    df["n_eff"] = df["n_samples"] / np.maximum(1.0, 2.0 * df["tau_absM"])
    return df
