# Finite-Difference Time-Domain (FDTD) Electromagnetic Simulation

*Yee solvers in 1D, 2D and 3D, with a 3-D engine checked against cavity eigenfrequencies, a reflection-free reference run and Mie theory.*

<p align="center">
  <a href="#build-and-run">Build and run</a> •
  <a href="#validation-results">Validation results</a> •
  <a href="FDTD.pdf">Report (PDF)</a> •
  <a href="#theory">Theory</a> •
  <a href="#references">References</a>
</p>

C++ implementations of the Yee FDTD method in 1D, 2D and 3D. Python scripts handle
visualisation. The 3-D code includes a CPML absorbing boundary and a total-field /
scattered-field (TFSF) plane-wave source, and it is validated against three independent
references: discrete-Yee cavity eigenfrequencies, a reflection-free reference run, and
Mie theory.

## Overview

| Program | What it does |
|---|---|
| `1D.cpp` | Gaussian pulse in free space (Ex, Hy). Exact absorbing boundary at S = 0.5. Prints the wave impedance Ex/Hy and the residual field after the pulse leaves. |
| `2D.cpp` | TMz (Ez, Hx, Hy) CW line source at 75 GHz with a first-order Mur boundary. Qualitative demo. |
| `3D.cpp` | 3-D Yee + CPML + TFSF plane wave. `mie`: εr = 4 sphere, Q_sca vs Mie theory. `grin`: graded-index sphere focusing map. `leak`: TFSF leakage check. |
| `cavity.cpp` | Dispersion validation: PEC-cavity resonances vs discrete-Yee and continuum eigenfrequencies, at two resolutions. |
| `cpml_test.cpp` | CPML reflection error vs a reflection-free reference run, scanned over σ_max and κ_max. |
| `fdtd3d.hpp` | Shared 3-D engine: Yee updates, CPML with ψ only in the PML slabs, per-component sub-pixel-averaged εr, OpenMP. |
| `1D.py`, `2D.py`, `3D.py` | Plotting only (they read the CSVs; they do not simulate). |

## Build and run

Requirements: a C++17 compiler with OpenMP (g++ ≥ 9), and Python 3 with `numpy`,
`pandas` and `matplotlib` (`pip install numpy pandas matplotlib`). Run everything from the
`FDTD/` directory.

```bash
make all          # check tools, create output dirs, build every *.cpp
./1D && python3 1D.py
./2D && python3 2D.py
make validate     # cavity, cpml_test, 3D leak / mie (R = 10, 30, 20) / grin, then 3D.py
```

Individual 3-D runs:

```bash
./cavity            # ~30 s
./cpml_test         # ~1 min (the reference domain is 218³ cells, ~0.6 GB)
./3D leak           # TFSF leakage, no scatterer
./3D mie [R]        # sphere radius 1.4 mm resolved by R cells (default 20)
./3D grin           # GRIN sphere, 198 × 136 × 136 cells
python3 3D.py
```

The kernels are memory-bound. `OMP_NUM_THREADS=8` is faster than 16 on an 8-core / 16-thread
CPU, and the Makefile sets it by default.

## Validation results

All numbers below are printed by the programs themselves and written to `data_outputs/`.

### 1. Numerical dispersion: PEC cavity (`cavity.cpp`, Fig. 4)

A broadband Ez pulse rings a 12 × 14 × 17-cell PEC cavity (Δ = 70 µm). The resonance
frequencies are taken from a Hann-windowed spectrum (32 768 steps) and compared with the
discrete-Yee and continuum eigenfrequencies of the 8 lowest isolated TMz modes. The run is
repeated at Δ/2 (24 × 28 × 34 cells, 65 536 steps).

| | Δ = 70 µm | Δ = 35 µm |
|---|---|---|
| Max \|f_meas − f_Yee\| / f_Yee | 0.06 ppm | 0.05 ppm |
| \|f_meas − f_cont\| / f_cont | 0.020 – 0.56 % | 0.005 – 0.14 % |
| Convergence order vs continuum | 2.004 – 2.011 (all 8 modes) | |

The grid reproduces the discrete dispersion relation to well below 1 ppm. The deviation
from continuum physics converges at second order, as expected.

### 2. Absorbing boundary: CPML (`cpml_test.cpp`, Fig. 5)

A differentiated-Gaussian dipole sits in a 30-cell vacuum interior surrounded by CPML. Ez
is recorded 2 cells inside the PML interface (face, edge and corner directions) and compared
with a 218³-cell reference run whose own boundary cannot be reached within the 300-step
window. Error = max_t |E − E_ref| / max_t |E_ref|, worst probe:

| CPML | worst-case error |
|---|---|
| 12 cells, σ_max = 0.75 σ_opt, κ_max = 1 (used in `3D.cpp`) | −95.6 dB |
| 12 cells, σ_max = σ_opt, κ_max = 1 | −93.1 dB |
| 15 cells, σ_max = 4×10¹³ s⁻¹ (2.9 σ_opt), κ_max = 5 (original parameters) | −89.3 dB |
| control: σ = 0 (the outer PEC wall reflects) | +2.8 dB |

This is a field-error ratio at near-boundary probes for a broadband pulse, which is the
standard CPML benchmark. It is **not** a plane-wave reflection coefficient.

### 3. Plane-wave scattering: Mie theory (`3D.cpp mie`, Fig. 2)

A homogeneous sphere (εr = 4, a = 1.4 mm) is illuminated by a TFSF plane-wave pulse. The
scattered power is the Poynting flux of the DFT'd scattered field through a closed surface
outside the TFSF box, at 41 frequencies from 30 to 110 GHz (size parameter x = ka = 0.88–3.2).
The result is normalised to the incident intensity and compared with Mie theory (Bohren &
Huffman). The Mie code was cross-checked against an independent scipy implementation
(agreement to 3×10⁻⁶).

| Resolution | Δ | mean \|Q_FDTD/Q_Mie − 1\| | max |
|---|---|---|---|
| R = 10 cells | 140 µm | 3.93 % | 10.0 % |
| R = 20 cells | 70 µm | 1.91 % | 5.0 % |
| R = 30 cells | 46.7 µm | 1.30 % | 3.4 % |

TFSF leakage with no scatterer: 1.2×10⁻¹⁵ (−299 dB) relative to the incident field (round-off).

### 4. GRIN sphere focusing (`3D.cpp grin`, Fig. 3)

The graded-index sphere εr(r) = 1 + 3(1 − r/R), with R = 40 cells = 2.8 mm (0.7 λ at
75 GHz), is illuminated by the same TFSF plane wave. On the z = centre plane, |Ez/E₀|² at
75 GHz peaks at **|Ez/E₀|² = 9.9** on the propagation axis, +0.70 R (+0.49 λ) from the sphere
centre. The focus lies *inside* the sphere near its rear surface. This is a demonstration
with no analytic reference.

## Theory

### Maxwell's curl equations

$$\nabla \times \mathbf{E} = -\mu \frac{\partial \mathbf{H}}{\partial t}, \qquad
  \nabla \times \mathbf{H} = \varepsilon \frac{\partial \mathbf{E}}{\partial t}$$

### Yee scheme

E and H are staggered by half a cell in space and half a step in time (leapfrog). Both
derivatives are **central differences**, so the scheme is second-order accurate in space
and time:

$$\frac{\partial f}{\partial z}\bigg|_{i} \approx \frac{f_{i+1/2} - f_{i-1/2}}{\Delta z}, \qquad
  \frac{\partial f}{\partial t}\bigg|^{n} \approx \frac{f^{n+1/2} - f^{n-1/2}}{\Delta t}$$

For the 1-D TEM wave (Ex, Hy) travelling along z:

$$H_y^{n+1/2}(i+\tfrac12) = H_y^{n-1/2}(i+\tfrac12) - \frac{\Delta t}{\mu \Delta z}\left[E_x^{n}(i+1) - E_x^{n}(i)\right]$$

$$E_x^{n+1}(i) = E_x^{n}(i) - \frac{\Delta t}{\varepsilon \Delta z}\left[H_y^{n+1/2}(i+\tfrac12) - H_y^{n+1/2}(i-\tfrac12)\right]$$

With these signs a +z wave has Ex/Hy = +η₀, so E×H points along +z. `1D.cpp` checks this
and prints 378 Ω. The 0.4 % offset from 376.7 Ω comes from averaging the staggered Hy.

### Courant (CFL) stability limit

| Dimension | Limit | Used here |
|---|---|---|
| 1D | Δt ≤ Δz / c | Δt = Δz / (2c), S = 0.5 |
| 2D | Δt ≤ 1 / (c √(1/Δx² + 1/Δy²)) | 0.99 × limit |
| 3D | Δt ≤ 1 / (c √(1/Δx² + 1/Δy² + 1/Δz²)) | 0.99 × limit, S = cΔt/Δ = 0.5716 |

### Numerical dispersion

On the Yee grid a plane wave satisfies

$$\frac{1}{(c\Delta t)^2}\sin^2\frac{\omega\Delta t}{2} = \sum_{u=x,y,z}\frac{1}{\Delta u^2}\sin^2\frac{k_u\Delta u}{2}$$

so waves travel slightly slower than c, with an error that scales as (Δ/λ)². At
57 cells/λ along an axis (the 3-D setup at 75 GHz) the phase-velocity error is 0.034 %.

### CPML

Stretched-coordinate PML in the convolutional form (Roden & Gedney 2000), with polynomial
grading of order m = 3:

σ(d) = σ_max d³, κ(d) = 1 + (κ_max − 1) d³, α(d) = α_max (1 − d),
b = exp(−(σ/κ + α)Δt), a = σ/(κ(σ + κα)) · (b − 1).

The profiles are sampled at integer nodes for E-field derivatives and at half-integer nodes
for H-field derivatives, symmetrically on both sides. The textbook optimum is
σ_opt = 0.8(m+1)/(η₀Δ).

### TFSF plane wave

The incident wave (+x, Ez-polarised) comes from a 1-D auxiliary Yee grid with the same Δ
and Δt. Its numerical dispersion therefore matches the 3-D grid exactly for axis-aligned
incidence, and the measured leakage is at round-off level.

## Outputs

| File | Contents |
|---|---|
| `1D_data.csv` | step, z_m, Ex_V_per_m, Hy_A_per_m (Hy is at z + Δz/2) |
| `2D_data.csv` | step, x, y, ez (snapshots after the source ramp) |
| `cavity_modes.csv`, `cavity_probe_{1x,2x}.csv` | mode table, probe time series |
| `cpml_scan.csv` | σ/κ scan, error per probe in dB |
| `mie_qsca_R<R>.csv`, `mie_curve.csv` | Q_sca FDTD vs Mie, and the Mie curve |
| `grin_slice.csv` | \|Ez/E₀\|² and εr on the z = centre plane |
| `3D_<mode>_meta.csv` | grid / box geometry used by `3D.py` |

Figures in `results_outputs/`: `1D_snapshots.png`, `1D_animation.gif`, `2D_snapshots.png`,
`2D_animation.gif`, and `fig1_geometry` … `fig5_cpml` (`.png` + `.pdf`).

## Key parameters

| Parameter | 1D | 2D | 3D |
|---|---|---|---|
| Cell size Δ | 100 µm | 100 µm | 70 µm (Mie: 1.4 mm / R) |
| Courant number S = cΔt/Δ | 0.5 | 0.70 | 0.5716 |
| Grid | 200 | 200 × 200 | 88³ (Mie, R = 20), 198 × 136 × 136 (GRIN) |
| Source | Gaussian pulse | 75 GHz CW, cosine ramp | TFSF Gaussian-modulated pulse, 30–110 GHz |
| Boundary | exact 2-step ABC | 1st-order Mur | CPML, 12 cells |

Memory: each field component takes 8 bytes per node. The 3-D engine stores 6 field
components plus 3 εr coefficient arrays (72 B/node), and ψ arrays only inside the PML slabs.

## Project structure

<details>
<summary>Files and what they contain</summary>

```text
FDTD/
├── fdtd3d.hpp            # shared 3-D Yee/CPML engine
├── 1D.cpp, 1D.py         # 1-D pulse + plots
├── 2D.cpp, 2D.py         # 2-D CW demo + plots
├── 3D.cpp                # TFSF scattering: Mie validation, GRIN focusing, leakage check
├── cavity.cpp            # dispersion validation (PEC cavity)
├── cpml_test.cpp         # absorbing-boundary validation
├── 3D.py                 # all 3-D / validation figures
├── Makefile
├── data_outputs/         # CSV output (generated)
└── results_outputs/      # PNG / PDF / GIF figures (generated)
```

</details>

## Limitations

- The sphere surfaces are staircased, which sub-pixel averaging of εr only partly offsets.
  This is the main source of the Mie error. The error shrinks roughly as Δ (first order), as
  expected for staircased curved interfaces (see the table above).
- The 2-D first-order Mur boundary reflects a few percent at oblique incidence, so the 2-D
  code is a visual demo only.
- The TFSF source supports only axis-aligned incidence.

## References

1. A. Taflove & S. C. Hagness, *Computational Electrodynamics: The FDTD Method*, 3rd ed., Artech House, 2005.
2. K. S. Yee, "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media," *IEEE Trans. Antennas Propag.* 14(3), 302–307, 1966.
3. J. A. Roden & S. D. Gedney, "Convolution PML (CPML): an efficient FDTD implementation of the CFS-PML for arbitrary media," *Microw. Opt. Technol. Lett.* 27(5), 334–339, 2000.
4. J. B. Schneider, *Understanding the FDTD Method*, 2010 (TFSF: ch. 9).
5. C. F. Bohren & D. R. Huffman, *Absorption and Scattering of Light by Small Particles*, Wiley, 1983.
6. D. M. Sullivan, *Electromagnetic Simulation Using the FDTD Method*, IEEE Press, 2000.
