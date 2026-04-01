# FDTD - Finite-Difference Time-Domain Electromagnetic Simulation

A comprehensive electromagnetic wave simulation framework implementing the **Finite-Difference Time-Domain (FDTD)** method in C++ and Python for 1D, 2D, and 3D geometries.

## Table of Contents

- [FDTD - Finite-Difference Time-Domain Electromagnetic Simulation](#fdtd---finite-difference-time-domain-electromagnetic-simulation)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Theory](#theory)
    - [Maxwell's Equations](#maxwells-equations)
    - [FDTD Method](#fdtd-method)
    - [Yee Grid Staggering](#yee-grid-staggering)
    - [Courant Condition](#courant-condition)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
      - [Option 1: Use the Makefile (Recommended)](#option-1-use-the-makefile-recommended)
      - [Option 2: Manual Setup](#option-2-manual-setup)
  - [Usage](#usage)
    - [C++ Implementation](#c-implementation)
    - [Python Implementation](#python-implementation)
  - [Output](#output)
    - [Data Files](#data-files)
    - [Visualization](#visualization)
  - [References](#references)
  - [Key Parameters](#key-parameters)
    - [Time and Space Discretization](#time-and-space-discretization)
    - [Physical Constants](#physical-constants)
  - [Troubleshooting](#troubleshooting)
    - [Compilation Issues](#compilation-issues)
    - [Simulation Issues](#simulation-issues)
  - [Performance Notes](#performance-notes)

---

## Overview

The FDTD method is a numerical technique for solving Maxwell's equations directly in both space and time. It discretizes the electromagnetic fields on a spatial grid and updates them at discrete time steps, making it ideal for simulating electromagnetic wave propagation, scattering, and interaction with materials.

**Key Features:**
-  1D, 2D, and 3D implementations
-  Both C++ (high-performance) and Python (ease-of-use) versions
-  Gaussian pulse sources
-  CSV output and visualization support
-  Free space and material simulations

---

## Theory

### Maxwell's Equations

The FDTD method solves Maxwell's curl equations in their differential form:

$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$$

$$\nabla \times \mathbf{H} = \frac{\partial \mathbf{D}}{\partial t} + \mathbf{J}$$

In free space or linear materials, these become:

$$\nabla \times \mathbf{E} = -\mu_0 \mu_r \frac{\partial \mathbf{H}}{\partial t}$$

$$\nabla \times \mathbf{H} = \epsilon_0 \epsilon_r \frac{\partial \mathbf{E}}{\partial t}$$

where:
- **E**: Electric field (V/m)
- **H**: Magnetic field (A/m)
- **μ₀ = 1.25663706 × 10⁻⁶** H/m (permeability of free space)
- **ε₀ = 8.85418782 × 10⁻¹²** F/m (permittivity of free space)
- **μᵣ, εᵣ**: Relative permeability and permittivity of materials

### FDTD Method

The FDTD method replaces spatial derivatives with **finite differences** and time derivatives with **forward differences**:

**Spatial derivative:**

$$\frac{\partial f}{\partial x}\bigg|_{i} \approx \frac{f_{i+1/2} - f_{i-1/2}}{\Delta x}$$

**Temporal derivative:**

$$\frac{\partial f}{\partial t}\bigg|^{n} \approx \frac{f^{n+1} - f^n}{\Delta t}$$

For a 1D TM mode (Ex and Hy fields), the update equations are:

$$\mathbf{E}_x^{n+1}(i) = \mathbf{E}_x^n(i) + \frac{\Delta t}{\epsilon_0 \epsilon_r(i) \Delta z}[\mathbf{H}_y^n(i) - \mathbf{H}_y^n(i-1)]$$

$$\mathbf{H}_y^{n+1}(i) = \mathbf{H}_y^n(i) + \frac{\Delta t}{\mu_0 \mu_r(i) \Delta z}[\mathbf{E}_x^n(i+1) - \mathbf{E}_x^n(i)]$$

### Yee Grid Staggering

The Yee grid staggeres the field components in space and time to achieve **second-order accuracy** and minimize numerical dispersion:

- **Electric field components** are located at integer grid points
- **Magnetic field components** are located at half-integer grid points (offset by Δx/2)
- **Time staggering**: E-fields and H-fields are updated at (n) and (n+1/2) respectively

This staggering ensures that field differences are computed at points closest to where differences matter physically.

```
1D Yee Cell:
    ↓ Ex(i)
    |
----|----●----  Hy(i-1/2)
    |
    |----●----
    |
    ↓ Ex(i+1)
```

### Courant Condition

Stability of the FDTD scheme requires the **Courant-Friedrichs-Lewy (CFL) condition**:

**For 1D:**

$$\Delta t \leq \frac{\Delta z}{2c}$$

**For 2D:**

$$\Delta t \leq \frac{1}{c\sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2}}}$$

**For 3D:**

$$\Delta t \leq \frac{1}{c\sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2} + \frac{1}{\Delta z^2}}}$$

where $c = 3 \times 10^8$ m/s is the speed of light. Violating this condition leads to exponential growth of numerical errors.

---

## Project Structure

```
FDTD/
├── 1D.cpp, 1D.py          # 1D FDTD simulations
├── 2D.cpp, 2D.py          # 2D FDTD simulations
├── 3D.cpp, 3D.py          # 3D FDTD simulations
├── Makefile                # Build system
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data_outputs/          # Simulation output data (CSV)
└── results_outputs/       # Visualizations (PNG, GIF)
```

---

## Installation

### Prerequisites

- **C++17** compiler (g++, clang++, or MSVC)
- **Python 3.7+**
- **Git** (optional, for cloning)

### Setup

Clone or download the project:

```bash
cd FDTD
```

#### Option 1: Use the Makefile (Recommended)

```bash
make all
```

This will:
1. Check for required tools (g++, python3, pip)
2. Install Python dependencies
3. Create output directories
4. Compile C++ executables

#### Option 2: Manual Setup

**Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**Compile C++ files:**
```bash
g++ -O3 -march=native -ffast-math -std=c++17 -Wall -Wextra 1D.cpp -o 1D
g++ -O3 -march=native -ffast-math -std=c++17 -Wall -Wextra 2D.cpp -o 2D
g++ -O3 -march=native -ffast-math -std=c++17 -Wall -Wextra 3D.cpp -o 3D
```

**Create output directories:**
```bash
mkdir -p data_outputs results_outputs
```

---

## Usage

### C++ Implementation

High-performance simulations compiled to native executables.

**Run 1D FDTD:**
```bash
./1D
```

**Run 2D FDTD:**
```bash
./2D
```

**Run 3D FDTD:**
```bash
./3D
```

**Output:** CSV files in `data_outputs/` containing:
- `1D_data.csv`: step, z_m (position), Ex_V_per_m, Hy_A_per_m
- `2D_data.csv`: step, x_m, y_m, Ex_V_per_m, Ey_V_per_m, Hx_A_per_m, Hy_A_per_m, Hz_A_per_m
- `3D_data.csv`: step, x_m, y_m, z_m, Ex_V_per_m, Ey_V_per_m, Ez_V_per_m, Hx_A_per_m, Hy_A_per_m, Hz_A_per_m

### Python Implementation

Python scripts with built-in visualization and analysis.

**Run 1D simulation with visualization:**
```bash
python3 1D.py
```

**Run 2D simulation with visualization:**
```bash
python3 2D.py
```

**Run 3D simulation:**
```bash
python3 3D.py
```

**Output:**
- Animated GIF: `results_outputs/*.gif`
- Snapshot figures: `results_outputs/*_snapshots.png`
- Data CSV: `data_outputs/*_data.csv`

---

## Output

### Data Files

Simulations generate CSV files with electromagnetic field data at each time step:

**1D Example (`1D_data.csv`):**
```
step,z_m,Ex_V_per_m,Hy_A_per_m
0,0.0000e-04,0.0,0.0
0,1.0000e-04,0.0,0.0
...
50,0.0000e-04,1.2345,0.0003456
```

### Visualization

Python scripts automatically generate:
- **Animated GIF**: Wave propagation over time
- **Snapshots**: Multi-panel figure showing field distributions at selected time steps
- **Static plots**: Field magnitude, phase, energy distribution

---

## References

1. **Taflove, A., & Hagness, S. C.** (2005). *Computational Electrodynamics: The Finite-Difference Time-Domain Method* (3rd ed.). Artech House Publishers.
   - Comprehensive reference for FDTD theory and applications

2. **Yee, K.** (1966). "Numerical Solution of Initial Boundary Value Problems Involving Maxwell's Equations in Isotropic Media." *IEEE Transactions on Antennas and Propagation*, 14(3), 302–307.
   - Original Yee grid paper

3. **Sullivan, D. M.** (2000). *Electromagnetic Simulation Using the FDTD Method*. IEEE Press.
   - Practical guide to FDTD implementation

4. **Kunz, K. S., & Luebbers, R. J.** (1993). *The Finite Difference Time Domain Method for Electromagnetics*. CRC Press.
   - Foundational FDTD textbook

5. **Allen Taflove's FDTD Website**: https://www.ece.northwestern.edu/~taflove/
   - Extensive FDTD resources and publications

6. **Bourke, P.** "Electromagnetic Wave Propagation": http://paulbourke.net/physics/cw/
   - Educational guide on wave propagation and FDTD basics

7. **Numerical Recipes**: https://numerical.recipes/
   - Reference for finite difference methods and numerical stability

---

## Key Parameters

### Time and Space Discretization

| Parameter | Symbol | Typical Value | Unit | Role |
|-----------|--------|---------------|------|------|
| Cell size | Δx/Δy/Δz | 0.1-1 | mm | Spatial resolution |
| Time step | Δt | ~Δz/(2c) | ps | Must satisfy Courant |
| Simulation time | T | 100-1000 | ps | Number of steps × Δt |
| Number of cells | N | 100-500 | — | 1D/2D, can scale to 1M+ for 3D |

### Physical Constants

| Constant | Symbol | Value |
|----------|--------|-------|
| Speed of light | c | 2.99792458 × 10⁸ m/s |
| Permeability | μ₀ | 1.25663706 × 10⁻⁶ H/m |
| Permittivity | ε₀ | 8.85418782 × 10⁻¹² F/m |

---

## Troubleshooting

### Compilation Issues

**Error: "g++: command not found"**
- Install gcc: `sudo apt install build-essential` (Ubuntu/Debian)
- Or use `clang++`: `clang++ -O3 -std=c++17 -ffast-math 1D.cpp -o 1D`

**Error: Python modules not found**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Simulation Issues

**Numerical instability (fields growing exponentially):**
- Check Courant condition: ensure `Δt ≤ Δz / (2c)`
- Reduce time step or cell size
- Use lower source amplitude

**No output files:**
- Ensure `data_outputs/` and `results_outputs/` directories exist
- Check write permissions
- Run from the FDTD directory

---

## Performance Notes

- **C++ version**: 10-100× faster than Python, ideal for large 3D simulations
- **Python version**: Easier to modify and visualize, good for learning and prototyping
- **Memory usage**: Roughly 8 × Nx × Ny × Nz bytes for double precision fields (both E and H)

---

**For questions or contributions, refer to the references above or FDTD literature.**
