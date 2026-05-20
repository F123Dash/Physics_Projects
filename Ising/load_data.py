import io

import numpy as np
import pandas as pd


def load_ising_csv(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline()
        if not header:
            raise ValueError(f"Empty CSV: {path}")
        header_cols = header.strip().split(",")
        n_cols = len(header_cols)
        if n_cols == 0:
            raise ValueError(f"Malformed CSV header: {path}")

        cleaned_lines = [header.strip()]
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            cols = stripped.split(",")
            if len(cols) > n_cols:
                cols = cols[:n_cols]
            elif len(cols) < n_cols:
                cols.extend([""] * (n_cols - len(cols)))
            cleaned_lines.append(",".join(cols))

    df = pd.read_csv(io.StringIO("\n".join(cleaned_lines)))

    required = {"T", "L", "M", "absM", "E", "M2", "E2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.copy()
    # Susceptibility uses signed magnetization moments
    # χ = N(⟨M²⟩ − ⟨M⟩²) / T where N = L²
    N = df["L"] ** 2
    df["chi"] = N * (df["M2"] - df["M"] ** 2) / df["T"]
    df["C"] = (df["E2"] - df["E"] ** 2) / (df["T"] ** 2)
    
    if "M4" in df.columns:
        # U = 1 - <M^4> / (3 * <M^2>^2)
        # Clamp M2 to avoid division by zero at high T
        m2_safe = df["M2"].clip(lower=1e-8)
        df["U"] = 1.0 - df["M4"] / (3.0 * m2_safe ** 2)
        
        from scipy.signal import savgol_filter
        for L in df["L"].unique():
            mask = df["L"] == L
            if np.count_nonzero(mask) >= 7:
                df.loc[mask, "U"] = savgol_filter(df.loc[mask, "U"].values, window_length=7, polyorder=2)
    return df
