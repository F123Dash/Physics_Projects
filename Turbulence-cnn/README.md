# Turbulence-CNN
### Physics-Informed Super-Resolution of 2-D Turbulence

A U-Net that reconstructs 128×128 velocity, pressure and vorticity fields of two-dimensional turbulence from 4× block-averaged (32×32) inputs, and is tested on Reynolds numbers it never saw. The training data is direct numerical simulation (DNS) of incompressible Navier-Stokes turbulence in a periodic box, in two regimes:

- **Forced turbulence:** Kolmogorov forcing with linear drag, statistically stationary (the set-up of Kochkov et al. 2021).
- **Decaying turbulence:** random initial vortices that merge and decay.

An auxiliary head regresses log Re from the bottleneck features.

The same pipeline also handles a second, laminar flow, the **lid-driven cavity** (Re = 100–3200, 64×64 from 16×16). It is kept as a wall-bounded test case. Each dataset's `meta.json` records its domain (periodic or walls), and every stage adapts to it.

---

## Results

All numbers are relative L2 errors in physical units on the **test set**: 560 snapshots at Re 320, 800 and 2000 (interpolation) and Re 4000 (extrapolation), from runs with their own seeds that were never seen in training. The tables compare the CNN with three baselines, all built from the same 32² block averages the model sees:

- **bilinear:** the model's own input
- **bicubic**
- **nearest train:** the training snapshot whose coarse field is closest; at about 100% error, it confirms the test runs are independent of the training runs

**spec err** is the mean |log₁₀(E_pred/E_true)| over the wavenumbers the coarse input cannot represent (k = 16–63; shells below 10⁻⁷ of a sample's total energy are ignored). **HF energy** is the fraction of the true fine-scale energy (k ≥ 16) that the prediction recovers.

### Final model: `turbulence_03`

Default settings (relative loss, no input noise, divergence-free projection, λ_spec = 0), 300 epochs, best checkpoint at epoch 287.

| Unseen Re, all snapshots | u,v forced | u,v decaying | p forced | p decaying | ω forced | ω decaying |
|---|---|---|---|---|---|---|
| **CNN (run 03)** | **0.9%** | **1.3%** | **1.1%** | **1.6%** | **8.4%** | **10.1%** |
| bicubic | 4.4% | 6.1% | 2.5% | 5.0% | 24.1% | 27.0% |
| bilinear (input) | 8.0% | 10.2% | 6.8% | 10.5% | 28.9% | 32.9% |
| nearest train | 95% | 101% | 99% | 105% | 101% | 104% |

Per Re (velocity / ω), the error grows smoothly with Re, including the extrapolation:

| | Re 320 | Re 800 | Re 2000 | Re 4000 (extrap.) |
|---|---|---|---|---|
| forced | 0.4% / 2.4% | 0.6% / 5.5% | 1.0% / 10.6% | 1.4% / 14.9% |
| decaying | 0.4% / 2.0% | 0.8% / 6.0% | 1.7% / 13.0% | 2.4% / 19.3% |

The predicted velocity is exactly divergence-free (projection). The auxiliary Re estimate is the weakest output: 14% error (decaying) and 20% (forced) on unseen Re, and 31–39% at Re 4000.

### Pointwise accuracy vs. fine-scale statistics

Pointwise losses replace the part of the fine scales that the coarse input cannot determine with its conditional mean, which has too little energy. The optional spectral loss (`--lambda_spec`) restores the energy, but some of the restored structure has the wrong phase, which raises pointwise error. Three settings, 200–300 epochs each, on the test set:

| λ_spec | u,v forced / decaying | ω forced / decaying | p forced / decaying | spec err forced / decaying | HF energy, Re 2000 / 4000 (forced; decaying) |
|---|---|---|---|---|---|
| 0 (run 03) | **0.9% / 1.3%** | **8.4% / 10.1%** | 1.1% / 1.6% | 0.226 / 0.247 | 89.4 / 87.0; 89.9 / 87.0 % |
| 0.002 (run 04b) | 1.0% / 1.5% | 9.4% / 11.0% | **1.0% / 1.6%** | 0.076 / 0.087 | 95.3 / 93.2; 94.6 / 91.9 % |
| 0.01 (run 04a) | 1.1% / 1.6% | 9.7% / 11.3% | 1.2% / 1.8% | **0.066 / 0.061** | **95.5 / 95.3; 95.6 / 94.3 %** |

Before the spectral runs, three conditions were set for replacing run 03: HF energy ≥ 95% at Re 2000 and 4000, lower spectral error, and velocity error up by at most 0.2 points with ω no worse. Neither spectral run met all three. Both raised ω error by about one point, and both fall short of 95% HF energy for decaying flow at Re 4000 (λ = 0.002 also at decaying Re 2000 and forced Re 4000). So **run 03 remains the default**.

**Recommendation:**
- **Pointwise accuracy** (the reconstructed field itself): use `turbulence_03`.
- **Realistic fine-scale statistics** (spectra, small-scale energy): use λ_spec = 0.002 (`turbulence_04_spec002`). It keeps most of the spectral gain of λ = 0.01 (spec err 0.08 vs 0.07, 3× below run 03) at a lower pointwise cost (velocity +0.1–0.2 points instead of +0.2–0.3; ω +0.9–1.0 instead of +1.2–1.3).
- **λ_spec = 0.01** adds little: λ = 0.002 is better on every pointwise metric (velocity, pressure, ω) and only slightly worse spectrally.

Getting both realistic fine scales and the lowest pointwise error would need a generative model that samples the fine scales (for example diffusion-based super-resolution) instead of predicting one field.

### Experiment log

#### Run 01 (`turbulence_01`): first run and what it changed

The first run trained for 150 epochs with MSE loss, input noise 0.05, no projection, and checkpoint selection by total validation loss. Relative L2 error on unseen Re:

| | u,v forced | u,v decaying | p forced | p decaying | ω forced | ω decaying |
|---|---|---|---|---|---|---|
| CNN | **1.7%** | **3.4%** | **2.0%** | 13.7% | **12.4%** | **16.8%** |
| bicubic | 4.4% | 6.1% | 2.5% | **5.0%** | 24.1% | 27.0% |
| bilinear (input) | 8.0% | 10.2% | 6.8% | 10.5% | 28.9% | 32.9% |
| nearest train | 95% | 101% | 99% | 105% | 101% | 104% |

The nearest-train baseline is about 100%, so the test runs are independent of the training runs. Extrapolating to Re 4000 raised the velocity error only slightly (forced 2.2%, decaying 3.8%). Checks on the trained model led to four changes, which are now the turbulence defaults:

| Finding | Change |
|---|---|
| Input noise of 0.05 carried 1.1–1.4× the energy of the true fine scales (k ≥ 16); train HF energy started at 1717%, and on clean inputs the model added fine scales (validation HF energy up to 154%) | No input noise for turbulence (`config.FLOWS`; the cavity keeps 0.05) |
| Decaying pressure error correlated −0.89 with how weak the sample's pressure is: MSE on globally normalised fields underweights weak fields | `--recon relative`: squared relative error per sample and channel, in physical units |
| Projecting the predicted velocity onto divergence-free fields cut mean\|div\| from 0.09 to ~1e-6 and lowered the velocity error (1.70 → 1.62% forced, 3.36 → 3.24% decaying) | Built-in exact projection for periodic models (`--no_project` turns it off) |
| The total validation loss swung with the Re regression error (38–150%), so it picked the checkpoint | Select on validation relative L2, mean over u, v, p, ω |

The predicted ω channel stays: it was more accurate than the curl of the predicted velocity (12.4% vs 13.2% forced). The Re head is the weakest part (27% error); it is auxiliary and no longer affects checkpoint selection.

#### Run 02 (`turbulence_02`): new defaults, 150 epochs

Relative L2 error on unseen Re (150 epochs; best checkpoint at epoch 147):

| | u,v forced | u,v decaying | p forced | p decaying | ω forced | ω decaying | HF energy forced / decaying |
|---|---|---|---|---|---|---|---|
| CNN, run 02 | **0.9%** | **1.5%** | **1.1%** | **1.6%** | **9.3%** | **11.0%** | 94% / 95% |
| CNN, run 01 | 1.7% | 3.4% | 2.0% | 13.7% | 12.4% | 16.8% | 92% / 98% |
| bicubic | 4.4% | 6.1% | 2.5% | 5.0% | 24.1% | 27.0% | 46% / 75% |

Every number improved. Decaying pressure, previously worse than bicubic, is now 3× better than it, and the predicted velocity is exactly divergence-free. Per Re, the error grows smoothly with Re: forced velocity error is 0.4% at Re 320, 1.1% at 2000 and 1.5% at Re 4000 (extrapolated). The remaining weakness is the finest scales at high Re. For forced Re 4000 the predicted spectrum follows the DNS up to k ≈ 25, then falls about 10× below it by k = 63 (HF energy 90%, ω error 16%), which is typical L2 smoothing. Validation error was still falling at epoch 150, so the next run trains longer.

#### Run 03 (`turbulence_03`): 300 epochs

Validation error flattens out around epoch 200 (best checkpoint at epoch 287). Doubling the epochs gave small gains: velocity 0.9% forced and 1.3% decaying; pressure 1.1% and 1.6%; ω 8.4% and 10.1%. Fine-scale energy fell, though: HF energy dropped from 94–95% to 91–92% overall, and to 87–90% at Re ≥ 2000. At forced Re 2000 the predicted spectrum leaves the DNS from k ≈ 30 and is about 10× low at k = 63. This is what pointwise losses do: the unpredictable part of the fine scales is replaced by its conditional mean, which carries less energy, and longer training moves the model further toward that mean. Training longer is therefore exhausted. The next lever is `--lambda_spec`, a phase-blind spectral loss (see Loss Function). On run 03's training predictions the spectral mismatch is about 100× the reconstruction loss, so λ_spec = 0.01 weights the two about equally.

#### Run 04a (`turbulence_04_spec01`): spectral loss, λ_spec = 0.01, 200 epochs

Before the run, three conditions were set for replacing run 03: HF energy ≥ 95% at Re 2000 and 4000; lower spectral error; and velocity error up by at most 0.2 points with ω no worse. On unseen Re:

| | u,v forced / decaying | ω forced / decaying | spec err forced / decaying | HF energy at Re 2000, 4000 (forced; decaying) |
|---|---|---|---|---|
| run 03 (λ_spec = 0) | **0.9% / 1.3%** | **8.4% / 10.1%** | 0.226 / 0.247 | 89%, 87%; 90%, 87% |
| run 04a (λ_spec = 0.01) | 1.1% / 1.6% | 9.7% / 11.3% | **0.066 / 0.061** | **96%, 95%; 96%, 94%** |

The spectral loss works: the predicted spectrum follows the DNS to k = 63, where run 03 was about 10× low, and spectral error is nearly 4× lower. The cost is 15–25% more pointwise error. Fine-scale structure of the right strength is partly at the wrong phase, and an L2 metric counts misplaced structure twice. The vorticity-consistency term also rose about 2.5×. By the conditions set beforehand (decaying velocity +0.3 points, ω worse), **run 03 stays the default**. Use run 03 for pointwise accuracy and a spectral model when fine-scale statistics (spectra, small-scale energy) matter.

The spectral error ignores shells below 10⁻⁷ of a sample's total energy. Without that floor, physically negligible energies (10⁻⁹ at k > 45 for Re 320) dominated the log ratio, and the lowest-Re case scored worst.

#### Run 04b (`turbulence_04_spec002`): spectral loss, λ_spec = 0.002, 200 epochs

Best checkpoint at epoch 195. Spectral error falls 3× (0.076 forced, 0.087 decaying), and HF energy reaches 95% at forced Re 2000 but not at decaying Re 2000 (94.6%) or at Re 4000 (93.2% forced, 91.9% decaying). Velocity error rises 0.1–0.2 points (1.0% / 1.5%), within the limit set beforehand, but ω error rises about one point (9.4% / 11.0%). It therefore does not replace run 03, but it is a better compromise than λ = 0.01, which costs more pointwise accuracy for little extra spectral gain.

---

## Physical Background

### 2-D turbulence

The solver integrates the 2-D incompressible Navier-Stokes equations in vorticity form on the periodic box [0, 2π)²:

$$\frac{\partial \omega}{\partial t} + \mathbf{u}\cdot\nabla\omega = \nu\nabla^2\omega - \alpha\,\omega + f, \qquad \mathbf{u} = \left(\frac{\partial\psi}{\partial y}, -\frac{\partial\psi}{\partial x}\right), \quad \nabla^2\psi = -\omega$$

with $Re = 1/\nu$ (velocities and lengths of order one).

| | Forced | Decaying |
|---|---|---|
| Forcing $f$ | $-4\cos(4y)$, the curl of $\sin(4y)\,\hat{x}$ | none |
| Drag $\alpha$ | 0.1 (removes energy piling up at large scales) | 0 |
| Initial field | weak random vorticity, $k \le 8$ | random, $E(k) \propto k^4 e^{-2(k/8)^2}$, energy 0.5 |
| Snapshots | after $t = 20$ (stationary), every 2 time units | from $t = 2$, every 0.5 time units |

In 2-D, energy cascades to large scales and enstrophy to small scales (Kraichnan 1967). For wavenumbers above the forcing scale, the inviscid theory predicts $E(k) \propto k^{-3}$ (the enstrophy cascade), and the spectrum plots show this as a reference line. At the Re used here the spectrum is much steeper: between k = 16 (the coarse input's cut-off) and k = 63, the measured slope is about −5.5 for forced Re = 1000, identical at 256² and 512². Viscosity and drag steepen the cascade, so the scales the network reconstructs lie in its steep, near-dissipative tail rather than in a clean k⁻³ range. A clean range would need much higher Re and resolution.

### Numerical method (turbulence)

- **Pseudo-spectral:** derivatives are exact in Fourier space, and nonlinear terms use 2/3-rule dealiasing.
- **Time stepping:** integrating-factor RK4. Viscosity and drag are integrated exactly; the time step follows the CFL limit (0.5) and is re-checked every 10 steps.
- **Resolution:** DNS at 256² or 512² depending on flow and Re (see `config.TURB_DNS_GRID`). The criterion is that the enstrophy spectrum at the dealiasing cut-off is below 10⁻³ of its peak, and every run records this in `meta.json`.
- **Saved grid:** snapshots are spectrally truncated to 128² (all modes |k| < 64). u, v, p and ω are computed from the vorticity spectrum, with pressure from $\nabla^2 p = 2(u_x v_y - u_y v_x)$.
- **Independent runs:** every (flow, Re, seed) run starts from its own random field, so test-Re runs share nothing with training runs.

### Lid-driven cavity (second test case)

Chorin projection on a collocated grid: MUSCL/van Leer advection, a DCT pressure solve, and a 127² solver grid saved at 64². It is validated against Ghia et al. (1982) centrelines and primary-vortex data: `python solver/ns_solver.py --validate_ghia`. The flows are laminar and become steady, so the data is the spin-up transient plus the steady state. Above Re ≈ 1000 the 127² grid under-resolves the flow (ψ_min is 9% off Ghia at Re 1000); `--refine 4` fixes this.

### Super-resolution as an inverse problem

Block averaging by 4 keeps 1/16 of the Fourier modes, so many fine fields map to the same coarse one. The CNN learns a prior over the fine fields and predicts a residual on top of the bilinearly upsampled input. For turbulence, the target is every scale between k = 16 and k = 63.

---

## Directory Structure

```
Turbulence-cnn/
├── config.py                 # Shared settings: paths, Re lists, grids, flow presets
├── solver/
│   ├── turbulence.py         # Pseudo-spectral 2-D turbulence (forced/decaying) + data generation
│   ├── ns_solver.py          # Cavity solver, streamfunction/vortex diagnostics, Ghia benchmark
│   ├── pressure_poisson.py   # Cavity DCT pressure solver
│   └── generate_data.py      # Cavity data generation
├── data/
│   ├── dataset.py            # Loading (any layout), split by Re, normalisation, TurbulenceDataset
│   └── augmentation.py       # Mirror flips and periodic shifts (physical units), coarsening
├── models/
│   ├── unet.py               # TurbulenceUNet (7.3M params), residual output, periodic option
│   ├── losses.py             # TurbulenceLoss (recon + div + vort + Re), SSIM, derivatives
│   └── classifier.py         # ReRegressor: log10 Re from the bottleneck
├── train/
│   ├── train.py              # Training loop, checkpoints, val/test tables
│   ├── evaluate.py           # Unseen-Re evaluation, spectra / cavity physics, plots, t-SNE
│   └── metrics.py            # Metrics shared by train.py and evaluate.py
├── tests/                    # python -m pytest tests  (cavity and turbulence paths)
├── notebooks/                # Written for an older cavity pipeline; not yet updated
├── snapshots_turb/           # Turbulence training data: <kind>/Re_*/seed_*/snap_*.npy + meta.json
├── snapshots_turb_ood/       # Turbulence unseen-Re test data, same layout
├── snapshots/, snapshots_ood/  # Cavity data: Re_*/snap_*.npy + meta.json
└── runs/                     # Checkpoints and plots per experiment
```

---

## Installation

```bash
cd Turbulence-cnn
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests      # ~5 s, no data needed
```

`requirements.txt` pins the versions used (Python 3.14). CUDA is used automatically if available; for a CUDA build of PyTorch, follow https://pytorch.org/get-started/locally/. Shared settings live in `config.py`; `--flow turbulence` (the default) or `--flow cavity` picks a dataset preset in `train.py`, `evaluate.py` and `data/dataset.py`.

---

## Pipeline

```
solver/turbulence.py  →  train/train.py  →  train/evaluate.py        (turbulence, default)
solver/generate_data.py  →  train/train.py --flow cavity  →  train/evaluate.py --flow cavity
```

## Stage 0 — Generate snapshots

### Turbulence

```bash
# training set: 6 Re × 3 seeds × {forced, decaying}
OMP_NUM_THREADS=1 python solver/turbulence.py --workers 14
# unseen-Re test set: Re 320, 800, 2000 (interpolation) and 4000 (extrapolation), new seeds
OMP_NUM_THREADS=1 python solver/turbulence.py --re_values 320 800 2000 4000 --seeds 10 11 \
    --save_dir snapshots_turb_ood --workers 14
```

| Argument | Default | Description |
|---|---|---|
| `--kinds` | `forced decaying` | Which regimes to simulate |
| `--re_values` | `config.TURB_RE_VALUES` (250 … 2500) | Reynolds numbers |
| `--seeds` | `0 1 2` | Independent runs per (kind, Re) |
| `--N` | 128 | Saved grid (spectral truncation of the DNS) |
| `--N_dns` | by kind and Re (`config.TURB_DNS_GRID`) | DNS grid override |
| `--n_save`, `--t_spin`, `--save_dt` | per kind (40/20/2 forced, 30/2/0.5 decaying) | Snapshot count, start time, spacing |
| `--save_dir` | `snapshots_turb` | Output root |
| `--overwrite` | off | Replace existing runs; without it the script refuses before starting |
| `--workers` | 1 | Runs in parallel (longest runs are started first) |

**Output:** `{save_dir}/{kind}/Re_{re}/seed_{s}/snap_{n:05d}.npy`, each of shape `(4, 128, 128)` float32 `[u, v, p, ω]` indexed `[x, y]`, plus `meta.json` with times, energy, the per-snapshot resolution diagnostic and the solver settings. The script warns about any run whose enstrophy tail exceeds 10⁻³.

**Cost:** one step takes about 10 ms at 256² and 48 ms at 512² (one core). Forced runs at 512² (Re > 2500, here only the Re 4000 test runs) take about an hour each; everything else takes minutes. At Re 1000, 256² and 512² DNS give the same time-averaged spectrum within ~10% at every saved wavenumber.

### Cavity

```bash
OMP_NUM_THREADS=1 python solver/generate_data.py --workers 12
OMP_NUM_THREADS=1 python solver/generate_data.py --re_values 1800 2200 2800 4000 --save_dir snapshots_ood --workers 4
python solver/ns_solver.py --validate_ghia [--N 129]
```

Options: `--re_values`, `--N` (64), `--refine` (2), `--save_dir`, `--overwrite`, `--workers`, `--t_start_save`/`--save_dt`/`--min_rel_change` (1.0/0.5/0.02), `--no_plots`. A cavity snapshot is saved only once (u, v) has changed by ≥ 2% since the previous one, and the steady state is saved once.

---

## Stage 1 — Data split

`make_dataloaders` splits by **whole Reynolds numbers**, so no run at a test Re is ever seen in training:

| Split | Turbulence (default) | Cavity |
|---|---|---|
| train | `snapshots_turb/`, Re 250, 400, 1000, 1600, 2500 | `snapshots/`, all Re except 600, 2000 |
| val | Re 630 (model selection only) | Re 600, 2000 |
| test | `snapshots_turb_ood/`: Re 320, 800, 2000, 4000 | `snapshots_ood/`: Re 1800, 2200, 2800, 4000 |

`--max_train_re X` trains only on Re ≤ X and moves every higher training-folder Re to the test set, which makes the test a harder extrapolation. Normalisation statistics come from the training split only and are stored in every checkpoint, along with the domain. Training samples are mirrored in x or y (in physical units, with the sign changes the NS equations require), and periodic samples are also shifted randomly. Cavity inputs get Gaussian noise (σ = 0.05, normalised units); turbulence inputs get none (see Results).

---

## Stage 2 — Train

```bash
python train/train.py                       # turbulence -> runs/turbulence_01
python train/train.py --flow cavity         # cavity     -> runs/cavity_01
```

| Argument | Default | Description |
|---|---|---|
| `--flow` | `turbulence` | Dataset preset (`config.FLOWS`) |
| `--data_dir`, `--test_dir` | from the preset | Snapshot directories (several allowed) |
| `--val_re` | from the preset | Re values held out for validation |
| `--max_train_re` | — | Train only on Re ≤ this; higher Re join the test set |
| `--out_dir` | `runs/<flow>_01` | Checkpoints, plots, `history.npy` |
| `--epochs` | 150 | Cosine LR schedule length |
| `--resume` | — | Resume from a checkpoint (model, optimizer, scheduler, best score, history) |
| `--lr`, `--lr_min` | 1e-3, 1e-5 | Cosine schedule bounds (AdamW) |
| `--weight_decay` | 1e-4 | |
| `--lambda_div`, `--lambda_vort`, `--lambda_re` | 0.01 each | Physics and Re loss weights |
| `--recon` | per flow: `relative` (turbulence), `mse` (cavity) | Reconstruction loss |
| `--noise_std` | per flow: 0 (turbulence), 0.05 (cavity) | Input noise on training samples |
| `--lambda_spec` | 0 (off) | Weight of the fine-scale spectral loss (periodic data only) |
| `--no_project` | off | Periodic data: skip the divergence-free projection of the predicted velocity |
| `--base_ch` | 64 | U-Net width |
| `--dropout_p` | 0.1 | Encoder dropout |
| `--batch_size` | 16 | |
| `--grad_clip` | 1.0 | Max gradient norm (0 disables) |
| `--seed`, `--num_workers` | 67, 0 | |

**Outputs:** `best_model.pt` (lowest validation relative L2, mean over u, v, p, ω), `last_model.pt`, `history.npy`, `training_curves.png`, `prediction_ep*.png`, `re_scatter_{val,test}.png`. At the end the script prints validation and test tables against the baselines.

---

## Stage 3 — Evaluate on unseen Re

```bash
python train/evaluate.py                    # turbulence: runs/turbulence_01/best_model.pt -> runs/eval_turbulence
python train/evaluate.py --flow cavity --checkpoint runs/cavity_01/best_model.pt
```

The script prints the CNN's metrics per (flow kind, Re), each marked interpolation or extrapolation, and per-kind tables comparing the CNN with all baselines. The model width, domain and normalisation statistics come from the checkpoint.

- **Turbulence:** `spectrum_<kind>_Re*.png` (mean energy spectra of DNS, CNN and baselines, with the coarse cut-off and a k⁻³ reference) and `vorticity_<kind>_Re*.png`.
- **Cavity:** a steady-state physics table (primary-vortex ψ_min and centre, centreline errors), `centerline_ood_Re*.png` and `vorticity_cavity_Re*.png`.
- **Both:** `tsne_bottleneck.png`, the bottleneck features of training and test snapshots coloured by log Re.

---

## Model Architecture

```
Input (4, H, W)  bilinear upsampling of the 4×4 block averages of [u, v, p, ω]
    │
    ├── enc1: DoubleConv(4→64)    + Dropout  → (64, H, W)
    ├── enc2: Pool + DoubleConv(64→128)  + Dropout → (128, H/2, W/2)
    ├── enc3: Pool + DoubleConv(128→256) + Dropout → (256, H/4, W/4)
    └── bottleneck: Pool + DoubleConv(256→512)     → (512, H/8, W/8) ── ReRegressor → log10 Re
    ┌── dec3: Up + Cat(skip3) + DoubleConv → (256, H/4, W/4)
    ├── dec2: Up + Cat(skip2) + DoubleConv → (128, H/2, W/2)
    ├── dec1: Up + Cat(skip1) + DoubleConv → (64, H, W)
Output = input + Conv1×1(64→4)
```

- **Parameters:** 7,315,973.
- **Blocks:** Conv3×3 → BatchNorm → ReLU, twice. Upsampling is bilinear plus a 1×1 conv.
- **Residual output:** the last 1×1 conv starts at zero, so an untrained model returns its input.
- **Periodic domains:** circular padding in every convolution, and wrap-around upsampling (in the input as well). The network is then exactly equivariant to shifts by multiples of 8 cells (tested). The predicted velocity is projected onto divergence-free fields, û ← û − k(k·û)/|k|², in physical units. This is an orthogonal projection, so it never increases the velocity error.

---

## Loss Function

$$\mathcal{L} = \mathcal{L}_\text{recon} + \lambda_\text{div}\mathcal{L}_\text{div} + \lambda_\text{vort}\mathcal{L}_\text{vort} + \lambda_\text{Re}\mathcal{L}_\text{Re}$$

| Term | Formula | Purpose |
|---|---|---|
| Reconstruction | turbulence: mean of $w_c\,\lVert\hat{f}_c - f_c\rVert^2 / \lVert f_c\rVert^2$ per sample, physical units; cavity: mean of $w_c(\hat{f}_c - f_c)^2$ on normalised fields; $w = [3,3,2,2]$ | Pixel-wise fidelity |
| Divergence | mean of $(\partial_x\hat{u} + \partial_y\hat{v})^2$, physical units | Incompressibility |
| Vorticity | mean of $((\hat{\omega} - (\partial_x\hat{v} - \partial_y\hat{u}))/\sigma_\omega)^2$, physical units | Predicted ω consistent with predicted (u, v) |
| Re | MSE on normalised $\log_{10} Re$ | Re-aware bottleneck |
| Spectral (optional, λ_spec) | mean over $k \ge 16$ of $(\ln E_\text{pred}(k) - \ln E_\text{true}(k))^2$, energies floored at $10^{-7}$ of the total | Keep the fine-scale energy that pointwise losses smooth away; ignores phase |

Derivatives are exact spectral derivatives for periodic data and central differences on interior nodes (Δx = 1/63) for the cavity. On turbulence ground truth both physics terms are about 0 (tested). On cavity ground truth they are about 0.007 and 0.01, because the cavity's ω comes from a finer solver grid and the collocated grid is not exactly divergence-free.

---

## Known Limitations

| Limitation | Cause |
|---|---|
| Turbulence Re limited to a few thousand | Resolved DNS above Re ≈ 2500 (forced) needs 512² or more; CPU cost grows as N³ per unit time |
| 2-D only | 3-D turbulence (forward energy cascade, k⁻⁵ᐟ³) would need a 3-D solver and model |
| Forced snapshots are 2 time units apart | Consecutive snapshots of one run are correlated; splits are by Re, so this does not leak into the test |
| Cavity data under-resolved above Re ≈ 1000 | 127² solver grid; use `--refine 4` |
| Notebooks are out of date | They use an older cavity data format and APIs |

---

## References

- Kochkov, D., et al. (2021). Machine learning–accelerated computational fluid dynamics. *PNAS*, 118(21), e2101784118.
- Kraichnan, R. H. (1967). Inertial ranges in two-dimensional turbulence. *Physics of Fluids*, 10(7), 1417–1423.
- Boffetta, G., & Ecke, R. E. (2012). Two-dimensional turbulence. *Annual Review of Fluid Mechanics*, 44, 427–451.
- Fukami, K., Fukagata, K., & Taira, K. (2019). Super-resolution reconstruction of turbulent flows with machine learning. *Journal of Fluid Mechanics*, 870, 106–120.
- Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387–411.
- Chorin, A. J. (1968). Numerical solution of the Navier-Stokes equations. *Mathematics of Computation*, 22(104), 745–762.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. *MICCAI 2015*.
- Wang, Z., et al. (2004). Image quality assessment: from error visibility to structural similarity. *IEEE TIP*, 13(4), 600–612.
