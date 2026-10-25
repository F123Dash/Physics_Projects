# Physics Projects

*Computational physics from first principles: ten projects, each with its own solver, tests and write-up.*

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue?style=flat)](LICENSE)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue?style=flat)](pyproject.toml)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue?style=flat)](FDTD/Makefile)
[![CUDA](https://img.shields.io/badge/CUDA-GPU-blue?style=flat)](Ising/README.md)

<p align="center">
  <a href="Fourier-optics/README.md"><img src="Fourier-optics/figures/fig3_lg_modes.png" height="140" alt="Laguerre–Gauss vortex modes: intensity rings and spiral phase"></a>
  <a href="Supercontinuum/README.md"><img src="Supercontinuum/figures/fig3_benchmark_evolution.png" height="140" alt="Supercontinuum: spectral and temporal evolution of a femtosecond pulse over 15 cm of fibre"></a>
  <a href="SSH-model/README.md"><img src="SSH-model/figures/fig2_open_spectrum.png" height="140" alt="SSH chain: open-chain spectrum with zero-energy edge states"></a>
  <a href="nh-SSH/README.md"><img src="nh-SSH/figures/fig4_phase_diagram.png" height="140" alt="Non-Hermitian SSH phase diagram"></a>
  <a href="non-Bloch/README.md"><img src="non-Bloch/figures/fig4_range2.png" height="140" alt="Non-circular generalised Brillouin zone and its spectrum"></a>
  <a href="iSCAT/README.md"><img src="iSCAT/figures/fig9_roughness.png" height="140" alt="iSCAT: speckle from coverslip roughness and its removal"></a>
  <a href="Berry-phase/README.md"><img src="Berry-phase/figures/fig4_loop_order.png" height="140" alt="Non-Abelian holonomy: two loop orders leave different final states"></a>
</p>

<p align="center">
  <a href="#projects">Projects</a> •
  <a href="#quick-start">Quick start</a> •
  <a href="#tools-and-methods">Tools</a> •
  <a href="#roadmap">Roadmap</a>
</p>

This repository is a growing archive of physics simulations, numerical solvers and data-driven models, built from first principles. Together the projects cover statistical mechanics, computational electromagnetics, fluid dynamics and scientific machine learning. They also cover topological and non-Hermitian band theory, geometric phases and gauge fields, Fourier and interferometric optics, and nonlinear fibre optics.

Each project checks its numbers against exact results or published benchmarks, and most record every quantitative claim as a pytest test. The aims are physical correctness, mathematical rigour, reproducible simulations, high-performance implementations and open scientific computing.

## Projects

| # | Project | Area | What it does | Checks | Report |
|---|---|---|---|---|---|
| 1 | [Ising Model](Ising/README.md) | Statistical mechanics | 2-D Ising model on CPU (Metropolis, Wolff) and GPU (CUDA); $T_c$ and critical exponents by finite-size scaling | exact 4×4 enumeration, `make check` | [PDF](Ising/Ising.pdf) |
| 2 | [FDTD](FDTD/README.md) | Computational electromagnetics | 1D, 2D and 3D Yee solvers; the 3-D engine has CPML and a TFSF plane-wave source | cavity modes, CPML and Mie validation, `make validate` | [PDF](FDTD/FDTD.pdf) |
| 3 | [Turbulence-CNN](Turbulence-cnn/README.md) | Fluid dynamics, scientific ML | U-Net super-resolution of 2-D turbulence, tested on Reynolds numbers it never saw | 33 pytest tests | [PDF](Turbulence-cnn/Turbulence_CNN.pdf) |
| 4 | [SSH Model](SSH-model/README.md) | Topological band theory | Topology of the Su–Schrieffer–Heeger chain from scratch | 55 pytest tests | [PDF](SSH-model/SSH-model.pdf) |
| 5 | [Berry Phase](Berry-phase/README.md) | Geometric phases | Geometric phases from a gauge-invariant discretisation, up to a non-Abelian holonomy | 87 pytest tests | [PDF](Berry-phase/Berry-phase.pdf) |
| 6 | [Non-Hermitian SSH](nh-SSH/README.md) | Non-Hermitian physics | Biorthogonal bulk-boundary correspondence of Kunst *et al.*, PRL 2018 | 48 pytest tests | [PDF](nh-SSH/nh-SSH.pdf) |
| 7 | [Non-Bloch Band Theory](non-Bloch/README.md) | Non-Hermitian physics | Generalised Brillouin zone and non-Bloch winding number | 56 pytest tests | [PDF](non-Bloch/non-Bloch.pdf) |
| 8 | [Fourier Optics](Fourier-optics/README.md) | Structured light | Band-limited angular-spectrum propagation of Gaussian, Laguerre–Gauss and grating fields | 136 pytest tests | [PDF](Fourier-optics/Fourier-optics.pdf) |
| 9 | [iSCAT Microscopy](iSCAT/README.md) | Light scattering, microscopy | Signal, image and detection limit of iSCAT from first principles | 133 pytest tests | [PDF](iSCAT/iSCAT.pdf) |
| 10 | [Supercontinuum](Supercontinuum/README.md) | Nonlinear fibre optics | GNLSE with RK4IP and split-step integrators, and the Dudley–Genty–Coen benchmark | 16 pytest tests | [PDF](Supercontinuum/Supercontinuum.pdf) |

Each project folder has its own README with results, method, figures and run commands.

### Project details

<details>
<summary><b>1. Ising Model</b>: phase transitions and critical phenomena</summary>

Implementation of the 2D Ising Model for studying phase transitions and critical phenomena in statistical mechanics.

*Features:* Metropolis Monte Carlo algorithm · Wolff cluster algorithm · CUDA acceleration · data analysis utilities · visualization tools

*Topics:* critical temperature · magnetization · susceptibility · phase transitions · finite-size effects

</details>

<details>
<summary><b>2. FDTD</b>: electromagnetic wave propagation</summary>

Electromagnetic wave propagation simulations using the Finite Difference Time Domain method, with 1D, 2D and 3D solvers.

*Topics:* Maxwell's equations · wave propagation · electromagnetic fields · numerical electromagnetics · computational electrodynamics

</details>

<details>
<summary><b>3. Turbulence-CNN</b>: turbulence modelling with machine learning</summary>

A hybrid computational fluid dynamics and machine learning project focused on turbulence modeling.

*Components:* Navier-Stokes solver · dataset generation pipeline · U-Net based neural networks · model training and evaluation framework

*Topics:* computational fluid dynamics (CFD) · turbulence modeling · deep learning · scientific machine learning (SciML) · physics-informed data generation

</details>

<details>
<summary><b>4. SSH Model</b>: topology of the Su–Schrieffer–Heeger chain</summary>

- Bloch bands, winding number and Zak phase (gauge-invariant Wilson loop)
- Open-chain zero modes, edge-state localisation length and finite-size splitting
- Real-space winding number under hopping disorder, including the topological Anderson insulator

*Topics:* topological insulators · chiral symmetry · bulk-boundary correspondence · Anderson localisation

</details>

<details>
<summary><b>5. Berry Phase and Non-Abelian Holonomy</b>: geometric phases</summary>

- Spin-1/2 Berry phase, its closed form at finite discretisation, and the Chern number of the sphere
- Fourth-order Magnus integrator checked against the exact rotating-frame solution; non-adiabatic correction and Aharonov–Anandan phase
- Tripod dark states: U(2) Wilson loop, analytic connection and field strength, full four-level evolution
- Loop-order dependence of the final state (non-Abelian holonomy) with an Abelian control

*Topics:* adiabatic theorem · Berry connection and curvature · Wilczek–Zee holonomy · artificial gauge fields for cold atoms

</details>

<details>
<summary><b>6. Non-Hermitian SSH Model</b>: biorthogonal bulk-boundary correspondence</summary>

- Periodic vs open spectrum and the non-Hermitian skin effect in left and right eigenvectors
- Exact similarity map of the open chain onto a Hermitian SSH chain, used as the reference for every numerical result
- $(t_1, \gamma)$ phase diagram and the paper's biorthogonal polarisation, with its $1/N$ finite-size term
- Where `numpy.linalg.eig` breaks down, checked against 60-digit mpmath

*Topics:* non-Hermitian topology · biorthogonal quantum mechanics · exceptional points · non-normal matrices and eigensolver accuracy

</details>

<details>
<summary><b>7. Non-Bloch Band Theory</b>: the generalised Brillouin zone</summary>

- One θ-scan GBZ solver for Hatano–Nelson, the non-reciprocal SSH chain (as one band in $E^2$) and a range-2 chain with a non-circular GBZ
- Non-Bloch winding checked against open-chain edge states; the Bloch winding is 1/2 where they disagree
- Leading finite-size term: open eigenvalues miss $E(\mathrm{GBZ})$ by $\ln|C(E)|/(N+1)$, derived and checked with no fitted parameter
- Exact condition for a circular GBZ, and an imaginary-gauge transform that keeps `eig` accurate to $N \approx 150$

*Topics:* non-Hermitian skin effect · non-Bloch band theory · banded Toeplitz matrices · topological invariants beyond Bloch's theorem

</details>

<details>
<summary><b>8. Fourier Optics and Structured Light</b>: angular-spectrum propagation</summary>

- Propagator checked against plane waves, Parseval and an exact Hankel integral; the band limit and the distance where aliasing starts
- Gaussian width, curvature and Gouy phase, with the nonparaxial corrections derived and matched to $10^{-11}$ (width) and $10^{-8}$ rad (Gouy)
- LG vortex modes: phase winding, OAM per photon, mode-order Gouy phase, and a two-mode beam that rotates by the Gouy phase
- Talbot carpet: exact vs paraxial self-imaging distance, and a derived $0.514\sqrt{\lambda/d}$ defect for binary gratings

*Topics:* scalar diffraction theory · Gaussian beam optics and the Gouy phase · orbital angular momentum of light · Talbot effect · sampling and aliasing in FFT methods

</details>

<details>
<summary><b>9. Interferometric Scattering (iSCAT) Microscopy</b>: detecting single nanoparticles</summary>

- Mie series checked against Bohren–Huffman and `miepython`; derived leading departures from Rayleigh for $\sigma_{\rm sca}$ and for the backscattered (iSCAT) amplitude
- Scalar image in closed form and a vectorial dipole-near-interface image (Fresnel s/p, aplanatic objective, supercritical light)
- Fringe period vs height: $2\lambda_m/(3+\cos\theta_m)$ at the PSF peak, $\lambda_m/2$ only for the spot-summed signal; Gouy phase and the best focus
- Shot-noise images, PSF-fit contrast, detection limits for polystyrene, gold and proteins; why attenuating the reference helps
- Extensions: 3D tracking at the Cramér–Rao bound with MSD analysis, speckle from coverslip roughness (and why focusing on the interface hides it), photothermal detection of gold

*Topics:* light scattering by small particles · interferometric detection · high-NA vectorial imaging · shot-noise-limited sensing

</details>

<details>
<summary><b>10. Supercontinuum Generation in a Photonic Crystal Fibre</b>: the GNLSE</summary>

- RK4IP and symmetric split-step Fourier, fixed or adaptive step; measured orders $h^{4.02}$ and $h^{2.00}$, RK4IP 6–20× cheaper at equal error
- Dispersion to $\beta_{10}$, Kerr, self-steepening and the delayed Raman response; NLS limits reproduced to round-off
- Benchmark: fission at $0.95\,L_D/N$, $-40$ dB span 491–1307 nm; photon number conserved to $3.5\times10^{-7}$ while energy drops 8.9%
- Gordon-formula excess $\propto T_0^{-1.95}$, recoil-corrected dispersive-wave phase matching, coherence collapse between $N \approx 9$ and 17
- Batched PyTorch/GPU solver identical to the serial one to $10^{-15}$

*Topics:* nonlinear fibre optics · optical solitons and soliton fission · Raman self-frequency shift and dispersive waves · supercontinuum coherence

</details>

## Quick start

Clone the repository and install every Python dependency with Poetry:

```bash
git clone https://github.com/F123Dash/Physics_Projects.git
cd Physics_Projects
poetry install
```

Run any Python project's tests from the repository root, for example:

```bash
poetry run pytest SSH-model/tests
```

The C++ projects build with `make` in their own folders:

```bash
make -C Ising check   # build the CPU and Wolff binaries, then compare with exact 4x4 results
make -C FDTD all      # build every FDTD program
```

> [!IMPORTANT]
> The CUDA binaries need an NVIDIA GPU, the CUDA Toolkit and a compatible C++ compiler. Each project README lists its own requirements and run times.

## Tools and methods

| Area | Tools |
|---|---|
| Languages | C++, CUDA, Python |
| Scientific computing | NumPy, SciPy and Matplotlib; mpmath (high-precision eigenvalue references); pytest (every quantitative claim is a test); LaTeX (project reports) |
| Numerical methods | finite difference methods, Monte Carlo methods, numerical PDE solvers, FFT / spectral methods (angular spectrum, split-step, RK4IP), gauge-invariant discretisations (Wilson loops, lattice Chern numbers) |
| Machine learning and GPU computing | PyTorch (U-Net training and batched GPU ODE integration), U-Net architectures, scientific machine learning |

## Roadmap

Planned projects:

- [ ] Lattice Boltzmann methods
- [ ] N-body simulations
- [ ] Quantum mechanics solvers
- [ ] Plasma physics simulations
- [ ] General relativity numerics
- [ ] Molecular dynamics
- [ ] Finite element methods
- [ ] Physics-informed neural networks (PINNs)

## Author

Kevin ([@F123Dash](https://github.com/F123Dash)).

## License

Released under the [GNU General Public License v3.0](LICENSE).
