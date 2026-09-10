// 2D Finite-Difference Time-Domain Simulation (TMz: Ez, Hx, Hy)
//
// A soft CW line source at the grid centre radiates a cylindrical wave; the outer boundary
// is a first-order Mur absorbing boundary. This is a qualitative demo of wave propagation —
// first-order Mur reflects a few percent at oblique incidence, so the field is not suitable
// for precision phase measurements. Quantitative validation of the Yee scheme is done in 3-D
// (cavity.cpp: dispersion, cpml_test.cpp: absorbing boundary, 3D.cpp: Mie scattering).
//
// Output: ./data_outputs/2D_data.csv  (step, x, y, ez) — snapshots after the source ramp
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <filesystem>

constexpr double C0   = 2.99792458e8;    // Speed of light        [m/s]
constexpr double MU0  = 1.25663706e-6;   // Permeability          [H/m]
constexpr double EPS0 = 8.85418782e-12;  // Permittivity          [F/m]

constexpr int    NX    = 200;
constexpr int    NY    = 200;
constexpr double DX    = 1e-4;           // Cell size             [m]
constexpr double DY    = 1e-4;
const     double DT    = 0.99 * DX / (C0 * std::sqrt(2.0)); // 2-D CFL limit ×0.99 ≈ 2.34e-13 s
constexpr int    NSTEP = 1500;
constexpr int    SNAP  = 20;             // Snapshot every N steps (post-ramp only)

const double UpMagX = DT / (MU0  * DX);
const double UpMagY = DT / (MU0  * DY);
const double UPEleX = DT / (EPS0 * DX);
const double UPEleY = DT / (EPS0 * DY);

// λ = c/f ≈ 4 mm = 40 cells  →  5 wavelengths across the 20-mm grid, ≈57 steps/period
constexpr double F_CW   = 75e9;                   // 75 GHz
const     double W_CW   = 2.0 * M_PI * F_CW;     // angular frequency  [rad/s]
constexpr double T_RAMP = 200.0;                  // cosine ramp length [steps]

constexpr int SRC_X = NX / 2;   // 100
constexpr int SRC_Y = NY / 2;   // 100

constexpr const char* OUTPUT_CSV = "./data_outputs/2D_data.csv";

int main()
{
    std::vector<std::vector<double>> ez(NX, std::vector<double>(NY, 0.0));
    std::vector<std::vector<double>> hx(NX, std::vector<double>(NY, 0.0));
    std::vector<std::vector<double>> hy(NX, std::vector<double>(NY, 0.0));
    std::vector<double> ez_left(NY,  0.0);
    std::vector<double> ez_right(NY, 0.0);
    std::vector<double> ez_bot(NX,   0.0);
    std::vector<double> ez_top(NX,   0.0);
    const double mur_cx = (C0*DT - DX) / (C0*DT + DX);
    const double mur_cy = (C0*DT - DY) / (C0*DT + DY);

    // Create output directory if it doesn't exist
    std::filesystem::create_directories("./data_outputs");

    std::ofstream out_snap(OUTPUT_CSV);
    if(!out_snap){ std::cerr << "Cannot open " << OUTPUT_CSV << "\n"; return 1; }
    out_snap << "step,x,y,ez\n";
    std::cout << "Mode: CW monochromatic  f = " << F_CW/1e9 << " GHz"
              << "  T_ramp = " << T_RAMP << " steps\n";
    std::cout << "DT = " << DT << " s  |  Courant (per axis) = " << C0*DT/DX << "\n\n";

    for(int n = 0; n < NSTEP; ++n){
        const double t = n * DT;
        // Mur needs the neighbour values from the previous step
        for(int j = 0; j < NY; ++j){
            ez_left[j]  = ez[1][j];
            ez_right[j] = ez[NX-2][j];
        }
        for(int i = 0; i < NX; ++i){
            ez_bot[i] = ez[i][1];
            ez_top[i] = ez[i][NY-2];
        }

        // Update Hx
        for(int i = 0; i < NX; ++i)
            for(int j = 0; j < NY-1; ++j)
                hx[i][j] -= UpMagY * (ez[i][j+1] - ez[i][j]);

        // Update Hy
        for(int i = 0; i < NX-1; ++i)
            for(int j = 0; j < NY; ++j)
                hy[i][j] += UpMagX * (ez[i+1][j] - ez[i][j]);

        // Update Ez
        for(int i = 1; i < NX-1; ++i)
            for(int j = 1; j < NY-1; ++j)
                ez[i][j] += UPEleX * (hy[i][j] - hy[i-1][j])
                           - UPEleY * (hx[i][j] - hx[i][j-1]);

        {
            double ramp = (n < T_RAMP)
                          ? 0.5 * (1.0 - std::cos(M_PI * n / T_RAMP))
                          : 1.0;
            ez[SRC_X][SRC_Y] += ramp * std::sin(W_CW * t);
        }

        // First-order Mur on the edges (corners excluded), then corners from their two neighbours
        for(int j = 1; j < NY-1; ++j){
            ez[0][j]    = ez_left[j]  + mur_cx * (ez[1][j]    - ez[0][j]);
            ez[NX-1][j] = ez_right[j] + mur_cx * (ez[NX-2][j] - ez[NX-1][j]);
        }
        for(int i = 1; i < NX-1; ++i){
            ez[i][0]    = ez_bot[i] + mur_cy * (ez[i][1]    - ez[i][0]);
            ez[i][NY-1] = ez_top[i] + mur_cy * (ez[i][NY-2] - ez[i][NY-1]);
        }
        ez[0][0]       = 0.5 * (ez[1][0]       + ez[0][1]);
        ez[NX-1][0]    = 0.5 * (ez[NX-2][0]    + ez[NX-1][1]);
        ez[0][NY-1]    = 0.5 * (ez[1][NY-1]    + ez[0][NY-2]);
        ez[NX-1][NY-1] = 0.5 * (ez[NX-2][NY-1] + ez[NX-1][NY-2]);

        if(n > T_RAMP && n % SNAP == 0){
            for(int i = 0; i < NX; ++i)
                for(int j = 0; j < NY; ++j)
                    out_snap << n << ',' << i << ',' << j << ',' << ez[i][j] << '\n';
        }

        if(n % 100 == 0)
            std::cout << "step " << n << "/" << NSTEP << "\n";
    }

    out_snap.close();
    std::cout << "\nSimulation complete.\n";
    std::cout << "Fields written to : " << OUTPUT_CSV << "\n";
    return 0;
}
