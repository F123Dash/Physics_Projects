import numpy as np
import pandas as pd


def load_ising_csv(path: str) -> pd.DataFrame:
    """Load simulation CSV and compute susceptibility/specific heat/Binder cumulant."""
    df = pd.read_csv(path)

    required = {"T", "L", "M", "absM", "E", "M2", "E2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.copy()
    # ✅ FIX: Correct susceptibility using absolute magnetization and system size
    # χ = N(⟨|M|²⟩ − ⟨|M|⟩²) / T where N = L²
    N = df["L"] ** 2
    df["chi"] = N * (df["M2"] - df["absM"] ** 2) / df["T"]
    df["C"] = (df["E2"] - df["E"] ** 2) / (df["T"] ** 2)
    
    # Compute Binder cumulant if M4 available
    if "M4" in df.columns:
        # U = 1 - <M^4> / (3 * <M^2>^2)
        # Clamp M2 to avoid division by zero at high T
        m2_safe = df["M2"].clip(lower=1e-8)
        df["U"] = 1.0 - df["M4"] / (3.0 * m2_safe ** 2)
        
        # ✅ FIX: Smooth Binder curves before peak finding (high noise on M4)
        from scipy.signal import savgol_filter
        for L in df["L"].unique():
            mask = df["L"] == L
            if np.count_nonzero(mask) >= 7:
                df.loc[mask, "U"] = savgol_filter(df.loc[mask, "U"].values, window_length=7, polyorder=2)
    
    return df
