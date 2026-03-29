// 3D_fdtd.cpp — Full 3-D FDTD with CPML absorbing boundary
// Maxwell curl equations (SI, lossless free space):
//   ∂H/∂t = -(1/μ₀) ∇×E
//   ∂E/∂t =  (1/ε₀) ∇×H
//
// Yee staggered-grid, 6-component update.
// Boundary: CPML (Convolutional Perfectly Matched Layer), NPML cells thick.
// Source  : soft CW Ez injection with cosine-ramp A(t) = ½(1-cos(πn/T_RAMP)).
//
// Diagnostics:
//    ./data_outputs/phase_vel_3D.csv   — leading-edge sphere radius vs time (ramp window)
//    ./data_outputs/dispersion_3D.csv  — Ez probe time series
//    ./data_outputs/swr_scan_3D.csv    — peak |Ez| envelope along x
//    ./data_outputs/3D_slice.csv  — Ez(x,y) snapshots at z=SRC_Z

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

//   Physical constants   
static constexpr double C0   = 2.99792458e8;
static constexpr double MU0  = 1.25663706212e-6;
static constexpr double EPS0 = 8.8541878128e-12;

//   Grid — 160³  (11.2 mm × 11.2 mm × 11.2 mm)  —  λ/Δx ≈ 57 @ 75 GHz  
static constexpr int    NX = 160, NY = 160, NZ = 160;
static constexpr double DX = 7e-5, DY = 7e-5, DZ = 7e-5;  // [m]

// CFL: Δt = 0.99·Δ/(c·√3)  →  S ≈ 0.5716
static const double DT  = 0.99 * DX / (C0 * std::sqrt(3.0));
static const double S_C = C0 * DT / DX;

//   Source   
static constexpr int    SRC_X  = NX / 2;   // 50
static constexpr int    SRC_Y  = NY / 2;   // 50
static constexpr int    SRC_Z  = NZ / 2;   // 50
static constexpr double F_CW   = 75e9;
static const     double W_CW   = 2.0 * M_PI * F_CW;
// Keep wavefront inside grid during ramp: n_wall ≈ (NX/2)/S ≈ 140 → use 120
static constexpr int    T_RAMP = 120;

//   Simulation   
static constexpr int NSTEP      = 2500;
static constexpr int SNAP       = 50;    // snapshot every 50 steps
static constexpr int SWR_WINDOW = 500;

// Dispersion probes along +x — both in free space (PML starts at NX-NPML = 145).
// r1 = 3.5 mm (50 cells), r2 = 4.06 mm (58 cells)  — far from sphere edge (R=2.8mm)
static constexpr int P1_X = SRC_X + 50;   // 130
static constexpr int P2_X = SRC_X + 58;   // 138

//   CPML parameters   
static constexpr int    NPML      = 15;
// sigma and alpha in 1/s  (normalised: b = exp(-(sigma/kappa + alpha)·dt), a is dimensionless)
static constexpr double SIGMA_MAX = 4.0e13;  // 1/s
static constexpr double KAPPA_MAX = 5.0;
static constexpr double ALPHA_MAX = 1.0e11;  // 1/s  (stabilises evanescent waves)

//   Field arrays   
static double Ex[NX][NY][NZ], Ey[NX][NY][NZ], Ez[NX][NY][NZ];
static double Hx[NX][NY][NZ], Hy[NX][NY][NZ], Hz[NX][NY][NZ];

//   Relative permittivity grid (1.0 everywhere by default)  
static double eps_r[NX][NY][NZ];

//   Gradient-index dielectric sphere   
// Profile: eps_r(r) = 1 + (EPS_C - 1)·(1 - r/R)   for r < SPHERE_R
//          eps_r(r) = 1                              for r ≥ SPHERE_R
// Centred on the source; eps_r(0)=EPS_C, eps_r(R)=1 → smooth impedance match.
// GRIN sphere: R = 40 cells = 2.8 mm ≈ 0.7λ;  ε_c = 4  →  n_max = 2
static constexpr double SPHERE_R   = 40.0 * DX;   // radius = 2.8 mm
static constexpr double SPHERE_EPS = 4.0;          // centre relative permittivity

//   CPML coefficient 1-D arrays   
static double bx[NX], ax[NX], kx[NX];
static double by[NY], ay[NY], ky_[NY];  // ky_ avoids collision with math macro
static double bz[NZ], az[NZ], kz[NZ];

//   CPML auxiliary (convolution-history) fields  
// H-update auxiliaries
static double psi_Hx_y[NX][NY][NZ], psi_Hx_z[NX][NY][NZ];
static double psi_Hy_x[NX][NY][NZ], psi_Hy_z[NX][NY][NZ];
static double psi_Hz_x[NX][NY][NZ], psi_Hz_y[NX][NY][NZ];
// E-update auxiliaries
static double psi_Ex_y[NX][NY][NZ], psi_Ex_z[NX][NY][NZ];
static double psi_Ey_x[NX][NY][NZ], psi_Ey_z[NX][NY][NZ];
static double psi_Ez_x[NX][NY][NZ], psi_Ez_y[NX][NY][NZ];

//   Probe & diagnostic storage  
static std::vector<double> probe1_ts, probe2_ts;
static double swr_max[NX] = {};
struct PVRec { int n; double t, r, vph; };
static std::vector<PVRec> pv_recs;

// Permittivity initialisation — fill eps_r with the GRIN sphere profile
static void init_sphere()
{
    const double cx = SRC_X * DX;
    const double cy = SRC_Y * DY;
    const double cz = SRC_Z * DZ;

    for (int i = 0; i < NX; ++i)
     for (int j = 0; j < NY; ++j)
      for (int k = 0; k < NZ; ++k) {
          eps_r[i][j][k] = 1.0;   // default: free space
          const double dx_ = i*DX - cx;
          const double dy_ = j*DY - cy;
          const double dz_ = k*DZ - cz;
          const double r   = std::sqrt(dx_*dx_ + dy_*dy_ + dz_*dz_);
          if (r < SPHERE_R)
              eps_r[i][j][k] = 1.0 + (SPHERE_EPS - 1.0) * (1.0 - r / SPHERE_R);
      }
}

//   
// CPML initialisation
// Profile: sigma(x) = sigma_max·(d/δ)³,  kappa(x) = 1+(kappa_max-1)·(d/δ)³
//          alpha(x) = alpha_max·(1-d/δ)  (d = depth into PML, δ = NPML·Δ)
// All in 1/s.  b = exp(-(sigma/kappa + alpha)·dt)
//              a = sigma/(sigma·kappa + kappa²·alpha) · (b - 1)   [dimensionless]
// In steady state: 1/kappa + a/(1-b) → 0  → perfect absorption.
//   
static void init_cpml()
{
    auto fill = [](int N, double* b, double* a, double* k) {
        for (int i = 0; i < N; ++i) {
            double sigma = 0, kappa = 1, alpha = 0;
            if (i < NPML) {
                double x = double(NPML - i) / double(NPML);
                sigma = SIGMA_MAX * std::pow(x, 3);
                kappa = 1.0 + (KAPPA_MAX - 1.0) * std::pow(x, 3);
                alpha = ALPHA_MAX * (1.0 - x);
            } else if (i >= N - NPML) {
                double x = double(i - (N - NPML)) / double(NPML);
                sigma = SIGMA_MAX * std::pow(x, 3);
                kappa = 1.0 + (KAPPA_MAX - 1.0) * std::pow(x, 3);
                alpha = ALPHA_MAX * (1.0 - x);
            }
            // All quantities in 1/s  —  no EPS0 in exponent
            b[i] = std::exp(-(sigma / kappa + alpha) * DT);
            double denom = sigma * kappa + kappa * kappa * alpha;
            a[i] = (denom > 0.0) ? sigma / denom * (b[i] - 1.0) : 0.0;
            k[i] = kappa;
        }
    };
    fill(NX, bx, ax, kx);
    fill(NY, by, ay, ky_);
    fill(NZ, bz, az, kz);
}

//   
// H-field update — CPML version
// Full update equations for all three magnetic components.
//   
static void update_H()
{
    const double coeff = DT / MU0;

    for (int i = 0; i < NX-1; ++i)
     for (int j = 0; j < NY-1; ++j)
      for (int k = 0; k < NZ-1; ++k) {

        // Determine if this cell touches the PML region on any axis
        const bool in_pml = (i < NPML || i >= NX-NPML ||
                              j < NPML || j >= NY-NPML ||
                              k < NPML || k >= NZ-NPML);

        const double dEzdy = (Ez[i][j+1][k] - Ez[i][j][k]) / DY;
        const double dEydz = (Ey[i][j][k+1] - Ey[i][j][k]) / DZ;
        const double dExdz = (Ex[i][j][k+1] - Ex[i][j][k]) / DZ;
        const double dEzdx = (Ez[i+1][j][k] - Ez[i][j][k]) / DX;
        const double dEydx = (Ey[i+1][j][k] - Ey[i][j][k]) / DX;
        const double dExdy = (Ex[i][j+1][k] - Ex[i][j][k]) / DY;

        if (in_pml) {
            //   CPML update: stretched-coordinate curl  
            psi_Hx_y[i][j][k] = by[j]*psi_Hx_y[i][j][k] + ay[j]*dEzdy;
            psi_Hx_z[i][j][k] = bz[k]*psi_Hx_z[i][j][k] + az[k]*dEydz;
            Hx[i][j][k] -= coeff * (
                dEzdy/ky_[j] + psi_Hx_y[i][j][k]
              - dEydz/kz[k]  - psi_Hx_z[i][j][k]);

            psi_Hy_z[i][j][k] = bz[k]*psi_Hy_z[i][j][k] + az[k]*dExdz;
            psi_Hy_x[i][j][k] = bx[i]*psi_Hy_x[i][j][k] + ax[i]*dEzdx;
            Hy[i][j][k] -= coeff * (
                dExdz/kz[k]  + psi_Hy_z[i][j][k]
              - dEzdx/kx[i]  - psi_Hy_x[i][j][k]);

            psi_Hz_x[i][j][k] = bx[i]*psi_Hz_x[i][j][k] + ax[i]*dEydx;
            psi_Hz_y[i][j][k] = by[j]*psi_Hz_y[i][j][k] + ay[j]*dExdy;
            Hz[i][j][k] -= coeff * (
                dEydx/kx[i]  + psi_Hz_x[i][j][k]
              - dExdy/ky_[j] - psi_Hz_y[i][j][k]);
        } else {
            //   Standard Yee update (interior — no CPML overhead)  
            Hx[i][j][k] -= coeff * (dEzdy - dEydz);
            Hy[i][j][k] -= coeff * (dExdz - dEzdx);
            Hz[i][j][k] -= coeff * (dEydx - dExdy);
        }
    }
}

//   
// E-field update — CPML version
//   
static void update_E()
{
    const double coeff_base = DT / EPS0;   // scaled per cell by eps_r[i][j][k]

    for (int i = 1; i < NX-1; ++i)
     for (int j = 1; j < NY-1; ++j)
      for (int k = 1; k < NZ-1; ++k) {

        const bool   in_pml = (i < NPML || i >= NX-NPML ||
                               j < NPML || j >= NY-NPML ||
                               k < NPML || k >= NZ-NPML);
        const double coeff  = coeff_base / eps_r[i][j][k];  // local permittivity

        const double dHzdy = (Hz[i][j][k] - Hz[i][j-1][k]) / DY;
        const double dHydz = (Hy[i][j][k] - Hy[i][j][k-1]) / DZ;
        const double dHxdz = (Hx[i][j][k] - Hx[i][j][k-1]) / DZ;
        const double dHzdx = (Hz[i][j][k] - Hz[i-1][j][k]) / DX;
        const double dHydx = (Hy[i][j][k] - Hy[i-1][j][k]) / DX;
        const double dHxdy = (Hx[i][j][k] - Hx[i][j-1][k]) / DY;

        if (in_pml) {
            //   CPML update   
            psi_Ex_y[i][j][k] = by[j]*psi_Ex_y[i][j][k] + ay[j]*dHzdy;
            psi_Ex_z[i][j][k] = bz[k]*psi_Ex_z[i][j][k] + az[k]*dHydz;
            Ex[i][j][k] += coeff * (
                dHzdy/ky_[j] + psi_Ex_y[i][j][k]
              - dHydz/kz[k]  - psi_Ex_z[i][j][k]);

            psi_Ey_z[i][j][k] = bz[k]*psi_Ey_z[i][j][k] + az[k]*dHxdz;
            psi_Ey_x[i][j][k] = bx[i]*psi_Ey_x[i][j][k] + ax[i]*dHzdx;
            Ey[i][j][k] += coeff * (
                dHxdz/kz[k]  + psi_Ey_z[i][j][k]
              - dHzdx/kx[i]  - psi_Ey_x[i][j][k]);

            psi_Ez_x[i][j][k] = bx[i]*psi_Ez_x[i][j][k] + ax[i]*dHydx;
            psi_Ez_y[i][j][k] = by[j]*psi_Ez_y[i][j][k] + ay[j]*dHxdy;
            Ez[i][j][k] += coeff * (
                dHydx/kx[i]  + psi_Ez_x[i][j][k]
              - dHxdy/ky_[j] - psi_Ez_y[i][j][k]);
        } else {
            //   Standard Yee update (interior)  
            Ex[i][j][k] += coeff * (dHzdy - dHydz);
            Ey[i][j][k] += coeff * (dHxdz - dHzdx);
            Ez[i][j][k] += coeff * (dHydx - dHxdy);
        }
    }
}

//   
// Leading-edge sphere radius: outermost cell where |Ez| > 1 % of peak
//   
static double wavefront_radius_3d()
{
    double peak = 0.0;
    for (int i = 0; i < NX; ++i)
     for (int j = 0; j < NY; ++j)
      for (int k = 0; k < NZ; ++k)
        peak = std::max(peak, std::abs(Ez[i][j][k]));
    if (peak < 1e-20) return 0.0;

    const double thresh = 0.01 * peak;
    double rmax = 0.0;
    for (int i = 0; i < NX; ++i)
     for (int j = 0; j < NY; ++j)
      for (int k = 0; k < NZ; ++k) {
        if (std::abs(Ez[i][j][k]) > thresh) {
            double dx_ = (i - SRC_X) * DX;
            double dy_ = (j - SRC_Y) * DY;
            double dz_ = (k - SRC_Z) * DZ;
            rmax = std::max(rmax, std::sqrt(dx_*dx_ + dy_*dy_ + dz_*dz_));
        }
    }
    return rmax;
}

//   
// Summary: dispersion (DFT phase), SWR, phase-velocity fit
//   
static void print_summary()
{
    //   DFT phase-difference in pre-reflection window  
    const int first_refl = T_RAMP
        + (int)((NX-1-SRC_X + NX-1-P1_X) / S_C) - 5;
    const int win_start = T_RAMP + 5;
    const int win_end   = std::min(first_refl, (int)probe1_ts.size());

    double re1 = 0, im1 = 0, re2 = 0, im2 = 0;
    for (int i = win_start; i < win_end; ++i) {
        double phi = W_CW * i * DT;
        re1 += probe1_ts[i] * std::cos(phi);
        im1 += probe1_ts[i] * std::sin(phi);
        re2 += probe2_ts[i] * std::cos(phi);
        im2 += probe2_ts[i] * std::sin(phi);
    }
    double dphi = std::atan2(im2, re2) - std::atan2(im1, re1);
    while (dphi >  M_PI) dphi -= 2.0 * M_PI;
    while (dphi < -M_PI) dphi += 2.0 * M_PI;

    const double r1 = (P1_X - SRC_X) * DX;
    const double r2 = (P2_X - SRC_X) * DX;
    const double dr = r2 - r1;
    const double arg_d = std::sin(W_CW * DT / 2.0) / S_C;
    const double k_th  = (std::abs(arg_d) <= 1.0)
                       ? (2.0/DX) * std::asin(arg_d) : 0.0;
    const double vph_th  = (k_th > 0) ? W_CW / k_th : C0;
    const double k_meas  = dphi / dr;
    const double vph_meas= (k_meas != 0) ? W_CW / k_meas : C0;
    const double err_pct = std::abs(vph_meas - vph_th) / C0 * 100.0;

    std::cout << "\n  Dispersion (CW, 75 GHz)   \n";
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "  Courant S           = " << S_C << "\n";
    std::cout << "  DFT window          = steps " << win_start
              << "-" << win_end << "  (pre-reflection)\n";
    std::cout << "  Δphi (DFT)            = " << dphi << " rad  over Δr="
              << dr*1e3 << " mm  (r1=" << r1*1e3 << " mm, r2=" << r2*1e3 << " mm)\n";
    std::cout << "  k_th                = " << k_th
              << " rad/m   vph_th  = " << vph_th/C0 << " c\n";
    std::cout << "  k_meas              = " << k_meas
              << " rad/m   vph_meas = " << vph_meas/C0 << " c\n";
    std::cout << "  Δvph/c              = " << err_pct << " %\n";

    //   PML absorption: compare field at PML entry vs 8 cells inside  
    // The SWR metric is invalid for 3D point-source (1/r amplitude decay
    // naturally gives large Emax/Emin).  Instead, report the PML attenuation:
    //   Att = |Ez| at (NX-NPML)  / |Ez| at (NX-NPML+8)
    const int i_pml_in  = NX - NPML;       // first PML cell
    const int i_pml_mid = NX - NPML + 8;   // 8 cells into PML
    const double E_entry   = (i_pml_in  < NX) ? swr_max[i_pml_in]  : 1.0;
    const double E_inside  = (i_pml_mid < NX) ? swr_max[i_pml_mid] : 1.0;
    const double att_dB    = (E_inside > 0 && E_entry > 0)
                           ? 20.0 * std::log10(E_entry / E_inside) : 0.0;

    std::cout << "\n  CPML performance   \n";
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "  E at PML entry (i=" << i_pml_in  << ") = " << E_entry  << " V/m\n";
    std::cout << "  E inside PML   (i=" << i_pml_mid << ") = " << E_inside << " V/m\n";
    std::cout << "  CPML attenuation                = " << att_dB << " dB over 8 cells\n";
    std::cout << "  (SWR metric invalid here: 1/r spreading makes Emax/Emin large\n"
              << "   even with zero reflection)\n";

    //   Phase-velocity from DFT k-measurement (same data as dispersion)  
    std::cout << "\n  Phase velocity (DFT probe method)   \n";
    std::cout << "  v_ph = W_CW / k_meas = " << vph_meas/C0 << " c\n";
    std::cout << "  v_ph_theoretical    = " << vph_th/C0  << " c\n";
    std::cout << "  (Note: CW wavefront tracker unreliable for v_ph;\n"
              << "   DFT phase-difference is the authoritative measurement)\n";
}

//   
int main()
{
    // Initialise CPML coefficients
    init_cpml();

    // Build gradient-index dielectric sphere
    init_sphere();

    probe1_ts.reserve(NSTEP);
    probe2_ts.reserve(NSTEP);

    // Create output directory if it doesn't exist
    std::filesystem::create_directories("./data_outputs");

    // Open output files
    FILE* f_snap = std::fopen("./data_outputs/3D_slice.csv",  "w");
    FILE* f_pv   = std::fopen("./data_outputs/phase_vel_3D.csv",   "w");
    FILE* f_disp = std::fopen("./data_outputs/dispersion_3D.csv",  "w");
    FILE* f_swr  = std::fopen("./data_outputs/swr_scan_3D.csv",    "w");
    if (!f_snap || !f_pv || !f_disp || !f_swr) {
        std::cerr << "Cannot open output files.\n"; return 1;
    }
    std::fprintf(f_snap, "step,x,y,ez\n");
    std::fprintf(f_pv,   "step,time_s,leading_edge_m,vph_m_per_s\n");
    std::fprintf(f_disp, "step,time_s,ez_p1,ez_p2\n");

    // Export permittivity slice at z=SRC_Z for visualisation
    {
        FILE* f_eps = std::fopen("./data_outputs/dielectric_slice_3D.csv", "w");
        if (f_eps) {
            std::fprintf(f_eps, "x,y,eps_r\n");
            for (int i = 0; i < NX; ++i)
             for (int j = 0; j < NY; ++j)
                std::fprintf(f_eps, "%d,%d,%.6f\n", i, j,
                             eps_r[i][j][SRC_Z]);
            std::fclose(f_eps);
        }
    }

    std::cout << "3D FDTD + CPML  " << NX << "×" << NY << "×" << NZ
              << "  f=" << F_CW/1e9 << " GHz"
              << "  S=" << S_C
              << "  DT=" << DT << " s"
              << "  NSTEP=" << NSTEP << "\n";
    std::cout << "CPML: NPML=" << NPML
              << "  sigma_max=" << SIGMA_MAX << " /s"
              << "  kappa_max=" << KAPPA_MAX
              << "  alpha_max=" << ALPHA_MAX << " /s\n\n";

    double prev_r = 0.0, prev_t = 0.0;

    //   Main loop   
    for (int n = 0; n < NSTEP; ++n)
    {
        const double t = n * DT;

        // 1. H update (CPML)
        update_H();

        // 2. E update (CPML)
        update_E();

        // 3. Soft CW source injection
        {
            const double ramp = (n < T_RAMP)
                ? 0.5 * (1.0 - std::cos(M_PI * n / T_RAMP))
                : 1.0;
            Ez[SRC_X][SRC_Y][SRC_Z] += ramp * std::sin(W_CW * t);
        }

        //   Diagnostics   

        // ① Phase velocity (ramp window only — CW steady-state has no propagating front)
        if (n <= T_RAMP && n % 2 == 0) {
            const double r = wavefront_radius_3d();
            double vph = 0.0;
            if (n > 0 && prev_r > 0.0 && r > prev_r)
                vph = (r - prev_r) / (t - prev_t);
            if (r > 0.0) {
                std::fprintf(f_pv, "%d,%.6e,%.6e,%.6e\n", n, t, r, vph);
                pv_recs.push_back({n, t, r, vph});
            }
            prev_r = r; prev_t = t;
        }

        // ② Probe records
        probe1_ts.push_back(Ez[P1_X][SRC_Y][SRC_Z]);
        probe2_ts.push_back(Ez[P2_X][SRC_Y][SRC_Z]);
        std::fprintf(f_disp, "%d,%.6e,%.6e,%.6e\n", n, t,
                     Ez[P1_X][SRC_Y][SRC_Z], Ez[P2_X][SRC_Y][SRC_Z]);

        // ③ SWR envelope (last SWR_WINDOW steps)
        if (n >= NSTEP - SWR_WINDOW) {
            for (int i = 0; i < NX; ++i)
                swr_max[i] = std::max(swr_max[i],
                                      std::abs(Ez[i][SRC_Y][SRC_Z]));
        }

        // ④ Ez slice snapshot at z=SRC_Z (post-ramp)
        if (n > T_RAMP && n % SNAP == 0) {
            for (int i = 0; i < NX; ++i)
             for (int j = 0; j < NY; ++j)
                std::fprintf(f_snap, "%d,%d,%d,%.6e\n",
                             n, i, j, Ez[i][j][SRC_Z]);
        }

        if (n % 100 == 0)
            std::cout << "  step " << n << "/" << NSTEP
                      << "  Ez_src = " << Ez[SRC_X][SRC_Y][SRC_Z] << "\n";
    }

    // Write SWR scan
    std::fprintf(f_swr, "x_cell,x_m,max_ez_envelope\n");
    for (int i = 0; i < NX; ++i)
        std::fprintf(f_swr, "%d,%.6e,%.6e\n", i, i*DX, swr_max[i]);

    std::fclose(f_snap); std::fclose(f_pv);
    std::fclose(f_disp); std::fclose(f_swr);

    std::cout << "\nSimulation complete.\n";
    std::cout << "   ./data_outputs/3D_slice.csv            — Ez(x,y) snapshots at z=SRC_Z\n";
    std::cout << "   ./data_outputs/phase_vel_3D.csv        — leading-edge radius\n";
    std::cout << "   ./data_outputs/dispersion_3D.csv       — Ez probe time series\n";
    std::cout << "   ./data_outputs/swr_scan_3D.csv         — |Ez| envelope along x\n";
    std::cout << "   ./data_outputs/dielectric_slice_3D.csv — eps_r(x,y) at z=SRC_Z\n";

    print_summary();
    return 0;
}
