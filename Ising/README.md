# 2-D Ising Model

*Monte Carlo finite-size scaling of the 2-D Ising model: CPU Metropolis, Wolff clusters and CUDA.*

<p align="center">
  <a href="#quick-start">Quick start</a> •
  <a href="#results">Results</a> •
  <a href="Ising.pdf">Report (PDF)</a> •
  <a href="#usage">Usage</a> •
  <a href="#limitations">Limitations</a>
</p>

C++ and CUDA Monte Carlo simulations of the ferromagnetic Ising model on an $L\times L$ periodic square lattice, with a Python pipeline that extracts the critical temperature and critical exponents by finite-size scaling. Every estimate is compared with Onsager's exact solution, and every error bar comes from a block bootstrap of the raw Markov chains, so correlated samples are not counted as independent.

## Quick start

You need `g++` with C++17, Python with `numpy`, `scipy`, `pandas` and `matplotlib`, and the CUDA toolkit for the GPU binary (see [Requirements](#requirements)).

```bash
make            # all three binaries (or: make cpu / make wolff / make cuda)
make check      # exact 4x4 comparison for every built binary
make fss        # Wolff chains at the exact Tc, then fss_analysis.py (about 3 minutes)
```

## Results

### Fixed-temperature finite-size scaling (Wolff, `make fss`)

One Wolff chain per size at the exact $T_c$, $L = 16, 32, 64, 128, 256$, 60 000 cluster flips each (72 s on one core). Errors: block bootstrap, 200 replicates × 50 blocks per chain; each block is 99–467 integrated autocorrelation times long.

| Quantity | Result | Exact | Deviation |
|---|---|---|---|
| $T_c$, Binder crossing $L=128/256$ | **2.26896 ± 0.00041** | 2.269185 | 0.5σ |
| $\beta/\nu$ at exact $T_c$ | **0.1266 ± 0.0016** | 1/8 | 1.0σ |
| $\gamma/\nu$ at exact $T_c$, $\chi = N\langle m^2\rangle/T$ | **1.7477 ± 0.0025** | 7/4 | 0.9σ |
| $\eta = 2-\gamma/\nu$ at exact $T_c$ | **0.2523 ± 0.0025** | 1/4 | 0.9σ |
| $\beta/\nu$ at the measured $T_c$ | **0.1242 ± 0.0042** | 1/8 | 0.2σ |
| $\gamma/\nu$ at the measured $T_c$ | **1.7517 ± 0.0070** | 7/4 | 0.3σ |
| Binder cumulant $U(T_c)$, $L=16 \ldots 256$ | 0.6116, 0.6114, 0.6136, 0.6103, 0.6092 (± 0.001) | $U^* = 0.61069$ | ≤ 2.6σ |

The rows "at the measured $T_c$" reweight each chain to this run's own Binder crossing, and every bootstrap replicate recomputes that crossing. Those exponents therefore do not rely on the exact $T_c$, and their errors include the $T_c$ uncertainty. With $\nu = 1$ (exact in 2-D), $\beta = 0.1242 \pm 0.0042$.

All 18 quantities agree with the exact values within about 2.5σ. The largest deviation is $U(T_c)$ at $L = 64$ (2.6σ). That same chain explains why the $32/64$ and $64/128$ crossings fall on opposite sides of $T_c$. With 18 quantities, one value at that level is not unusual.

### Temperature scan (CUDA Metropolis)

$L = 16 \ldots 256$, $T \in [2.0, 2.6]$ (76 temperatures, 0.005 steps in $[2.1, 2.4)$), 200 000 sweeps per point. This gives a qualitative overview; errors are regression errors only (see [Limitations](#limitations)).

| Quantity | Result | Exact |
|---|---|---|
| $T_c$, $\chi'$-peak extrapolation $T_c(L) = T_c + a/L$ | 2.2701 ± 0.0026 | 2.269185 |
| $T_c$, Binder crossing $L=32/64$ | 2.2681 (no error estimate) | 2.269185 |
| $\gamma/\nu$ from $\chi'_\text{max} \sim L^{\gamma/\nu}$ | 1.746 ± 0.016 | 7/4 |
| $\eta$ from $\chi(T_c) \sim L^{2-\eta}$ | 0.262 ± 0.010 | 1/4 |
| $C_\text{max}$ vs $\ln L$ slope | 0.487 ± 0.011 | $8K_c^2/\pi = 0.4945$ |

The scan's $\eta$ depends strongly on $T_c$: moving $T_c$ by its standard error shifts $\eta$ by 0.039. Quote the fixed-$T$ value (0.2523 ± 0.0025) instead. The scan rejects the $64/128$ and $128/256$ Binder crossings as ambiguous: with only about 16 independent samples per point at $L = 256$, the cumulant curves cross several times by noise.

### Figures

Figures are written to `plots/`, which is git-ignored, so regenerate them locally:

| File | Content | Made by |
|---|---|---|
| `fss_scaling.png` | $\langle\|m\|\rangle$ and $\chi$ vs $L$ at $T_c$ (log-log, fit vs exact slope); reweighted Binder curves | `fss_analysis.py` |
| `fig1_magnetization` | $\langle\|m\|\rangle(T)$ per $L$ | `plotting.py` |
| `fig2_susceptibility` | $\chi'(T)$ per $L$ | `plotting.py` |
| `fig2b_binder` | $U(T)$ per $L$ | `plotting.py` |
| `fig3_specific_heat` | $C(T)$ per $L$ | `plotting.py` |
| `fig4_tc_extrapolation` | $T_c(L)$ vs $1/L$ | `plotting.py` |
| `fig6_data_collapse` | $\langle\|m\|\rangle L^{\beta/\nu}$ vs $(T-T_c)L^{1/\nu}$ with exact exponents | `plotting.py` |

## Physical background

### Model and exact results

$$H = -J\sum_{\langle ij\rangle} s_i s_j, \qquad s_i = \pm 1, \qquad J = k_B = 1$$

Onsager's solution gives $T_c = 2/\ln(1+\sqrt 2) = 2.269185\ldots$. The 2-D Ising exponents are $\nu = 1$, $\beta = 1/8$, $\gamma = 7/4$, $\eta = 1/4$ and $\alpha = 0$, where the specific heat diverges logarithmically.

### Observables

All are per spin, with $N = L^2$, $m = \frac1N\sum_i s_i$ and $e = H/N$:

| Observable | Definition |
|---|---|
| Susceptibility | $\chi' = N(\langle m^2\rangle - \langle\|m\|\rangle^2)/T$, and $\chi = N\langle m^2\rangle/T$ at $T_c$ |
| Specific heat | $C = N(\langle e^2\rangle - \langle e\rangle^2)/T^2$ |
| Binder cumulant | $U = 1 - \langle m^4\rangle / (3\langle m^2\rangle^2)$ |

### Finite-size scaling

At $T_c$, $\langle|m|\rangle \sim L^{-\beta/\nu}$ and $\chi \sim L^{\gamma/\nu}$. The exponents are therefore slopes on a log-log plot against $L$, and need no fit window in $T_c - T$. Near $T_c$ the curves $U_L(T)$ for different $L$ cross at a common point that approaches $T_c$, with the universal value $U^* \approx 0.6107$ (Salas & Sokal, *J. Stat. Phys.* **98**, 551, 2000). Away from $T_c$, the susceptibility peak shifts as $T_c(L) = T_c + aL^{-1/\nu}$.

### Single-histogram reweighting

A chain at $T_0$ predicts averages at a nearby $T$:

$$\langle A\rangle_T = \frac{\sum_t A_t\, e^{-(1/T - 1/T_0)E_t}}{\sum_t e^{-(1/T - 1/T_0)E_t}}$$

The usable range is about $|T - T_0| \lesssim T_0^2/\sigma_E$, which is ±0.005 at $L = 256$. `fss_analysis.py` restricts each pair's crossing search to that window.

### Errors from correlated chains

Successive Monte Carlo samples are correlated over an integrated autocorrelation time $\tau_\text{int}$ (Sokal windowing, $W \ge 6\tau$), so a chain of $n$ samples holds only about $n/2\tau_\text{int}$ independent ones. Errors come from a **block bootstrap**: each chain is cut into 50 contiguous blocks much longer than $\tau_\text{int}$, and the blocks are resampled with replacement. For Metropolis at $T_c$, $\tau_\text{int} \approx 1000$ sweeps at $L = 128$ and about 6000 at $L = 256$, so 200 000 sweeps hold only about 100 and 16 independent samples. Cluster updates reduce this to a few cluster flips.

## Algorithms

| Binary | Update | One "sweep" means | Notes |
|---|---|---|---|
| `ising2d_cpu` | Metropolis, checkerboard order | $N$ attempted flips | Reference implementation |
| `ising2d_wolff` | Wolff single-cluster | **one cluster flip** | Bonds activated with $p = 1-e^{-2/T}$; best near $T_c$, slow far below $T_c$ (clusters cover the lattice) |
| `ising2d_cuda` | Metropolis, checkerboard, one thread per site of a sublattice | $N$ attempted flips | `curand` XORWOW per thread; **needs even $L$** (with odd $L$, periodic neighbours share a colour) |

All drivers start from the ordered state, because a hot start below $T_c$ can freeze Metropolis into a striped two-domain state on large lattices. Metropolis thermalization scales as $(L/32)^{2.17}$ times `--therm`.

### Validation

`make check` runs every built binary on the 4×4 lattice and compares $\langle e\rangle$ and $\langle|m|\rangle$ with exact enumeration of all $2^{16}$ states at $T = 1.5$, $T_c$ and $3.0$. It fails above 5 standard errors. All 18 checks currently pass. The CUDA build is also clean under `compute-sanitizer` memcheck and racecheck.

## Usage

### Requirements

- `g++` with C++17
- CUDA toolkit for `ising2d_cuda`. The Makefile targets `sm_86`; change `CUDA_ARCH` for other GPUs.
- Python ≥ 3.10 with `numpy`, `scipy`, `pandas`, `matplotlib`

### Build and test

```bash
make            # all three binaries (or: make cpu / make wolff / make cuda)
make check      # exact 4x4 comparison for every built binary
```

### Reproduce the results

```bash
# Fixed-T finite-size scaling (about 3 minutes in total)
make fss        # runs ising2d_wolff at exact Tc, then fss_analysis.py

# Temperature scan on the GPU (about 30 minutes on an RTX 3050 Laptop GPU)
./ising2d_cuda --sizes=16,32,64 --tmin=2.0 --tmax=2.6 --therm=5000 --meas=200000 --stride=50 \
    --seed=1 --out=data_outputs/data.csv
./ising2d_cuda --append=128,256 --tmin=2.0 --tmax=2.6 --therm=5000 --meas=200000 --stride=50 \
    --seed=1 --out=data_outputs/data.csv
python3 analysis.py      # -> data_outputs/analysis_metrics.csv, tc_by_L.csv
python3 plotting.py      # -> plots/
```

`make fss` refuses to overwrite existing series files. Delete `data_outputs/fss/series` to rerun it.

### Simulation options (all three binaries)

| Option | Default | Meaning |
|---|---|---|
| `--sizes=16,32,...` | 32 … 1024 | Lattice sizes |
| `--tmin`, `--tmax` | 1.8, 3.4 | Temperature range |
| `--dt` | 0.02 | Step for `--no-adaptive` |
| `--no-adaptive` | off | Uniform grid; `--tmin=--tmax` gives a single temperature |
| `--fine-dt` | 0.005 | Adaptive-grid step inside $[2.1, 2.4)$; 0.02 gives 15 points instead of 60 |
| `--therm`, `--meas`, `--stride` | 30000, 200000, 50 | Thermalization, measurement and sampling interval (cluster flips for Wolff) |
| `--seed` | 123456789 | RNG seed, recorded in every row |
| `--out` | `data_outputs/data.csv` | Output CSV; overwritten unless `--append` |
| `--append`, `--append=128,256` | — | Add sizes not already in the file |
| `--series-dir=DIR` | — | Also write the raw $(m, e)$ time series of every $(T, L)$ |

### Analysis options

```bash
python3 analysis.py [--data CSV] [--metrics CSV] [--series-dir DIR] [--min-n-eff 100]
python3 plotting.py [--data CSV] [--outdir DIR]
python3 fss_analysis.py [--series-dir DIR] [--algo NAME] [--lmin L] [--n-blocks 50] [--n-boot 200]
```

`analysis.py --series-dir` adds block-bootstrap Binder-cumulant errors to a scan run with `--series-dir`. Crossings then count only significant differences and get an error from a weighted fit. Without series files, a crossing is accepted only if $U_1 - U_2$ changes sign exactly once and every point near it has $n_\text{eff} \ge$ `--min-n-eff`.

### Data format

`data.csv`, one row per $(T, L)$:

```csv
T,L,M,absM,E,M2,E2,M4,n_samples,tau_absM,algo,seed
```

Moments are per spin. `tau_absM` is $\tau_\text{int}$ of $|m|$ in units of measurements, so $n_\text{eff} = n_\text{samples}/2\tau$. Series files (`series_<algo>_L<L>_T<T>.csv`) start with a `# T=... L=... algo=... seed=... stride=...` line, followed by `m,e` columns.

**Integrity checks:** these stop runs from mixing unnoticed in one file.
- **Drivers:**
  - `--append` refuses a file with a different header (for example an old 8-column file) or an existing $(T, L)$ row.
  - Series files are never overwritten.
- **`load_ising_csv`:** rejects missing columns, rows with the wrong field count, non-numeric values and duplicate $(T, L)$ rows, and prints which `algo`/`seed` produced which sizes.
- **`analysis.py`:** excludes sizes whose scan does not cover the file's full temperature range, such as an interrupted run.

## Limitations

- **Scan errors are regression-only.** One chain per $(T, L)$, without stored series, gives no per-point error. Use `--series-dir` for Binder errors, or the fixed-$T$ pipeline for quoted numbers.
- **Metropolis is too slow at large $L$ near $T_c$.** $\tau_\text{int} \propto L^{z}$ with $z \approx 2.17$, so at $L = 256$ the scan has about 16 independent samples per point. More sweeps over a coarser critical band (`--fine-dt=0.02`), or Wolff near $T_c$, are the practical fixes.
- **$\nu = 1$ is assumed**, not fitted, in the $T_c(L)$ extrapolation and the data collapse. The fixed-$T$ exponents are ratios ($\beta/\nu$, $\gamma/\nu$).
- **Corrections to scaling** are not fitted. Dropping $L = 16$ (`--lmin 32`) moves $\beta/\nu$ from 0.1266 to 0.1286 ± 0.0024, which is within errors.

## Directory structure

<details>
<summary>Files and what they contain</summary>

```text
Ising/
├── ising.hpp / ising.cpp     # SimulationConfig, CLI parser (shared by all drivers)
├── algorithms.cpp            # Ising2D: Metropolis and Wolff updates
├── observables.cpp           # SampleAccumulator, finalize
├── run_io.hpp                # CSV schema, duplicate checks, series output, tau_int, T grid
├── main.cpp                  # ising2d_cpu   (Metropolis)
├── wolff.cpp                 # ising2d_wolff (Wolff)
├── ising_cuda.cuh / .cu      # Ising2DCUDA: checkerboard kernel, reductions
├── main_cuda.cu              # ising2d_cuda  (Metropolis on GPU)
├── load_data.py              # strict CSV loader + derived observables (chi, C, U, n_eff)
├── analysis.py               # temperature-scan analysis: Tc, gamma/nu, eta, C_max
├── plotting.py               # scan figures
├── fss_analysis.py           # fixed-T FSS: reweighting, Binder crossings, exponents, block bootstrap
├── check_exact.py            # exact-enumeration regression test (4x4)
├── Makefile
├── data_outputs/             # generated data (git-ignored)
│   ├── data.csv              #   temperature scan
│   ├── analysis_metrics.csv  #   scan results
│   └── fss/                  #   fixed-T runs: fss_wolff.csv, series/, fss_metrics.csv
└── plots/                    # generated figures (git-ignored)
```

</details>

## Corrections from version 3.0.0

Earlier results from this folder ($\beta = 0.096$, $T_c = 2.260$) are superseded:

- **Wolff bug:** a neighbour that failed one bond test was marked visited and could never join through another bond. At $T = 1.5$ on 4×4 this gave $e = -0.83$ instead of $-1.95$, so all earlier `ising2d_wolff` data is invalid. The CPU and CUDA Metropolis paths were unaffected.
- **$\beta = 0.096$ was an artefact of the $T_c$ estimate.** A log-log fit of $\langle|m|\rangle$ vs $T_c - T$ amplifies a 0.4 % $T_c$ error into a 24 % $\beta$ error. It was replaced by $L$-scaling at fixed $T$.
- **The $T_c$ pipeline was tuned to the known answer:**
  - It dropped sizes below 64, enforced a decreasing $T_c(L)$, and fell back whenever the result left 2.1–2.4.
  - It weighted the fit by $L^2$.
  - It smoothed $U$ with Savitzky–Golay. All of this is removed.
- **Specific heat was missing a factor of $N$**, so $C$ shrank with $L$.
- **Errors ignored autocorrelation.** Naive errors were about 6× too small at $L = 128$.
- **The CUDA kernel read past the RNG array for odd $L$.** Odd $L$ is now rejected.
