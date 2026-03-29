#!/usr/bin/env python3
"""Debug script to understand Binder crossing detection."""

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data.csv')
df = df.sort_values('L')

print("=== Data loaded ===")
print(df.head(10))

# For each pair of consecutive L values, find crossing
pairs = [(df.iloc[i], df.iloc[i+1]) for i in range(len(df)-1)]

print(f"\n=== Analyzing all pairs ===")
for i, (row1, row2) in enumerate(pairs):
    L1, L2 = int(row1['L']), int(row2['L'])
    print(f"\nPair {i}: L={L1} vs L={L2}")
    print(f"  T_L1 sample: {row1['Tc_L']}")
    print(f"  T_L2 sample: {row2['Tc_L']}")
    
    # Extract Binder data for both system sizes
    u1 = row1[[c for c in df.columns if c.startswith('U_')]].values.astype(float)
    u2 = row2[[c for c in df.columns if c.startswith('U_')]].values.astype(float)
    
    # Temperature grid
    t_min = float(row1['T_min'])
    t_max = float(row1['T_max'])
    n_temps = len(u1)
    t_grid = np.linspace(t_min, t_max, n_temps)
    
    # Find crossings manually
    diff = u1 - u2
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    
    print(f"  Diff range: [{diff.min():.4f}, {diff.max():.4f}]")
    print(f"  Sign changes found: {len(sign_changes)}")
    
    for j, idx in enumerate(sign_changes):
        t1, t2 = t_grid[idx], t_grid[idx+1]
        d1, d2 = diff[idx], diff[idx+1]
        tc = t1 - d1 * (t2 - t1) / (d2 - d1)
        print(f"    Sign change {j}: Tc = {tc:.4f} K (between {t1:.4f} and {t2:.4f})")
    
    # Check which would be selected
    if len(sign_changes) > 0:
        # Check physical range
        physical_mask = np.array([
            2.0 <= (t_grid[idx] + t_grid[idx+1])/2.0 <= 2.5 
            for idx in sign_changes
        ])
        print(f"  Physical range [2.0, 2.5]: {np.sum(physical_mask)} crossings")
        
        if np.any(physical_mask):
            candidates = sign_changes[physical_mask]
        else:
            candidates = sign_changes
        
        # Pick one with smallest |diff|
        best_idx = candidates[np.argmin(np.abs(diff[candidates]))]
        t1, t2 = t_grid[best_idx], t_grid[best_idx+1]
        d1, d2 = diff[best_idx], diff[best_idx+1]
        tc_best = t1 - d1 * (t2 - t1) / (d2 - d1)
        print(f"  ✓ Selected crossing: Tc = {tc_best:.4f} K")

