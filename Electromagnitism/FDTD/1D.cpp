// =============================================================================
// 1D Finite-Difference Time-Domain (FDTD) Simulation
// Solves Maxwell's equations in free space (1D, TM mode)
// Fields: Ex (electric) and Hy (magnetic)
// =============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

// ── Physical constants ────────────────────────────────────────────────────────
constexpr double C0    = 2.99792458e8;   // Speed of light [m/s]
constexpr double MU0   = 1.25663706e-6;  // Permeability of free space [H/m]
constexpr double EPS0  = 8.85418782e-12; // Permittivity of free space [F/m]

// ── Simulation parameters ──────────────────────────────────────────────────────
constexpr int    NZ    = 200;            // Number of spatial cells
constexpr double DZ    = 1e-4;          // Cell size [m]
constexpr double DT    = DZ / (2.0 * C0); // Time step (Courant condition: dt ≤ dz/c)
constexpr int    NSTEP = 5000;            // Number of time steps

// Source parameters (Gaussian pulse)
constexpr double T0    = 40.0 * DT;     // Pulse center [s]
constexpr double SIGMA = 12.0 * DT;     // Pulse width  [s]
constexpr int    SRC_Z = NZ / 4;        // Source position (cell index)

// ── FDTD update coefficients ──────────────────────────────────────────────────
// Derived from curl equations:
//   dEx/dt = (1/eps) * dHy/dz    →   Ex[n+1] = Ex[n] + (dt/eps/dz) * (Hy[i] - Hy[i-1])
//   dHy/dt = (1/mu)  * dEx/dz    →   Hy[n+1] = Hy[n] + (dt/mu/dz)  * (Ex[i+1] - Ex[i])
constexpr double CE = DT / (EPS0 * DZ); // Electric field update coeff
constexpr double CH = DT / (MU0  * DZ); // Magnetic field update coeff
constexpr int         SNAP_EVERY  = 50;              // record a snapshot every N steps
constexpr const char* OUTPUT_CSV  = "./data_outputs/1D_data.csv";  // single output file

// ── Helper: append one snapshot (all NZ rows) to the open CSV file ───────────
void write_snapshot(std::ofstream& f,
                    const std::vector<double>& ex,
                    const std::vector<double>& hy, int step)
{
    for (int i = 0; i < NZ; ++i)
        f << step << "," << i * DZ << "," << ex[i] << "," << hy[i] << "\n";
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main()
{
    // Field arrays (Yee grid staggering)
    //   Ex lives at integer cells:  i = 0 … NZ-1
    //   Hy lives at half cells:     i+0.5, stored at index i
    std::vector<double> ex(NZ, 0.0);
    std::vector<double> hy(NZ, 0.0);

    // Material arrays (relative permittivity / permeability per cell)
    std::vector<double> eps_r(NZ, 1.0);
    std::vector<double> mu_r (NZ, 1.0);

    // Precompute per-cell coefficients
    std::vector<double> ce(NZ), ch(NZ);
    for (int i = 0; i < NZ; ++i) {
        ce[i] = DT / (EPS0 * eps_r[i] * DZ);
        ch[i] = DT / (MU0  * mu_r[i]  * DZ);
    }

    // ── Open output CSV ───────────────────────────────────────────────────
    std::ofstream csv(OUTPUT_CSV);
    if (!csv) { std::cerr << "Cannot open " << OUTPUT_CSV << "\n"; return 1; }
    csv << "step,z_m,Ex_V_per_m,Hy_A_per_m\n";

    std::cout << "FDTD 1D  |  NZ=" << NZ
              << "  DZ=" << DZ*1e3 << " mm"
              << "  DT=" << DT*1e12 << " ps"
              << "  NSTEP=" << NSTEP << "\n";

    // ── Time-stepping loop ────────────────────────────────────────────────
    for (int n = 0; n < NSTEP; ++n)
    {
        double t = n * DT;

        // 1. Update Hy (magnetic field) — uses Ex at step n
        //    Hy[i] lives between Ex[i] and Ex[i+1]; skip last cell (boundary)
        for (int i = 0; i < NZ - 1; ++i)
            hy[i] += ch[i] * (ex[i + 1] - ex[i]);

        // 2. Update Ex (electric field) — uses Hy just updated (leapfrog)
        //    Ex[i] lives between Hy[i-1] and Hy[i]; skip first cell (boundary)
        for (int i = 1; i < NZ; ++i)
            ex[i] += ce[i] * (hy[i] - hy[i - 1]);

        // 3. Inject soft source (additive Gaussian pulse at SRC_Z)
        double pulse = std::exp(-0.5 * std::pow((t - T0) / SIGMA, 2));
        ex[SRC_Z] += pulse;

        // 4. Write snapshot to CSV
        if (n % SNAP_EVERY == 0) {
            write_snapshot(csv, ex, hy, n);
            std::cout << "  step " << n << " / " << NSTEP
                      << "  peak Ex = " << *std::max_element(ex.begin(), ex.end())
                      << "\n";
        }
    }

    // Write final state
    write_snapshot(csv, ex, hy, NSTEP);
    std::cout << "Done. Data written to " << OUTPUT_CSV << "\n";

    return 0;
}
