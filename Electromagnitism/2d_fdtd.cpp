// 2D Finite-Difference Time-Domain Simulation
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>

constexpr double C0   = 2.99792458e8;    // Speed of light        [m/s]
constexpr double MU0  = 1.25663706e-6;   // Permeability          [H/m]
constexpr double EPS0 = 8.85418782e-12;  // Permittivity          [F/m]

constexpr int    NX    = 200;
constexpr int    NY    = 200;
constexpr double DX    = 1e-4;           // Cell size             [m]
constexpr double DY    = 1e-4;
const     double DT    = 0.99 * DX / (C0 * std::sqrt(2.0)); // CFL ≈ 2.36e-13 s
constexpr int    NSTEP = 1500;
constexpr int    SNAP  = 20;             // Snapshot every N steps (post-ramp only)

const double UpMagX = DT / (MU0  * DX);
const double UpMagY = DT / (MU0  * DY);
const double UPEleX = DT / (EPS0 * DX);
const double UPEleY = DT / (EPS0 * DY);

// λ = c/f ≈ 4 mm  →  ~5 wavelengths across 20-mm grid, ~57 steps/period
constexpr double F_CW   = 75e9;                   // 75 GHz
const     double W_CW   = 2.0 * M_PI * F_CW;     // angular frequency  [rad/s]
constexpr double T_RAMP = 200.0;                  // cosine ramp length [steps]
constexpr int    PV_TRACK_END = static_cast<int>(T_RAMP); // stop at step 200

constexpr int SRC_X = NX / 2;   // 100
constexpr int SRC_Y = NY / 2;   // 100

constexpr int P1_X = SRC_X + 20;   // 120
constexpr int P1_Y = SRC_Y;
constexpr int P2_X = SRC_X + 40;   // 140
constexpr int P2_Y = SRC_Y;

constexpr int REFL_PX = SRC_X + 60;   // 160  (40 cells from right wall)
constexpr int REFL_PY = SRC_Y;

constexpr const char* OUTPUT_CSV     = "fdtd_2d_data.csv";
constexpr const char* PHASE_VEL_CSV  = "phase_vel.csv";
constexpr const char* DISPERSION_CSV = "dispersion.csv";
constexpr const char* SWR_CSV        = "swr_scan.csv";   // steady-state |Ez| envelope along x

double wavefront_radius(const std::vector<std::vector<double>>& ez,
                        int sx, int sy)
{
    double ez_max = 0.0;
    for(int i = 0; i < NX; ++i)
        for(int j = 0; j < NY; ++j)
            ez_max = std::max(ez_max, std::abs(ez[i][j]));
    if(ez_max < 1e-15) return 0.0;   // field hasn't left the source yet
    const double threshold = 0.01 * ez_max;  // 1 % of peak
    int max_r = 0;
    for(int i = 0; i < NX; ++i){
        for(int j = 0; j < NY; ++j){
            if(std::abs(ez[i][j]) > threshold){
                int r = static_cast<int>(
                    std::sqrt(double((i-sx)*(i-sx) + (j-sy)*(j-sy))));
                if(r > max_r) max_r = r;
            }
        }
    }
    return max_r * DX;   // metres
}
void print_dispersion_summary(const std::vector<double>& ts1,
                               const std::vector<double>& ts2)
{
    const double S   = C0 * DT / DX;          // Courant number per axis
    const double arg = (1.0/S) * std::sin(W_CW * DT / 2.0);
    double k_theoretical = 0.0;
    if(std::abs(arg) <= 1.0)
        k_theoretical = (2.0 / DX) * std::asin(arg);
    const double vph_theoretical = (k_theoretical > 0) ? W_CW / k_theoretical : C0;
    int N = static_cast<int>(ts1.size());
    int half = N / 2;
    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for(int i = half; i < N; ++i){
        dot   += ts1[i] * ts2[i];
        norm1 += ts1[i] * ts1[i];
        norm2 += ts2[i] * ts2[i];
    }
    const double dx_probes = (P2_X - P1_X) * DX;
    double k_measured = 0.0;
    double vph_measured = C0; // fallback
    if(norm1 > 0 && norm2 > 0){
        double cos_dphi = dot / std::sqrt(norm1 * norm2);
        cos_dphi = std::max(-1.0, std::min(1.0, cos_dphi));
        double dphi = std::acos(cos_dphi);
        k_measured  = dphi / dx_probes;
        vph_measured = (k_measured > 0) ? W_CW / k_measured : C0;
    }

    double disp_err_pct = (k_theoretical > 0)
        ? 100.0 * std::abs(vph_measured - vph_theoretical) / C0 : 0.0;
    std::cout << "\n=== Dispersion Analysis (CW, f = " << F_CW/1e9 << " GHz) ===\n";
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  FDTD Δt             = " << DT         << " s\n";
    std::cout << "  Courant (per axis)  = " << S          << "\n";
    std::cout << "  ω                   = " << W_CW       << " rad/s\n";
    std::cout << "  k_theoretical (num) = " << k_theoretical << " rad/m\n";
    std::cout << "  v_ph_theoretical    = " << vph_theoretical << " m/s\n";
    std::cout << "  k_measured (probes) = " << k_measured << " rad/m\n";
    std::cout << "  v_ph_measured       = " << vph_measured   << " m/s\n";
    std::cout << "  Δv_ph / c           = " << disp_err_pct << " %\n";
    std::cout << "\n  FDTD dispersion relation (along x, ky=0):\n";
    std::cout << "    sin²(ωΔt/2) = (cΔt/Δx)² · sin²(k·Δx/2)\n";
    std::cout << "    LHS = " << std::pow(std::sin(W_CW*DT/2),2)
              << "  RHS(k_th) = " << S*S*std::pow(std::sin(k_theoretical*DX/2),2)
              << "\n";
}
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
    std::vector<double> probe1_ts;
    std::vector<double> probe2_ts;
    probe1_ts.reserve(NSTEP);
    probe2_ts.reserve(NSTEP);
    std::vector<double> swr_profile(NX, 0.0);
    constexpr int SWR_WINDOW = 200;            // accumulate over last N steps
    std::ofstream out_snap(OUTPUT_CSV);
    if(!out_snap){ std::cerr << "Cannot open " << OUTPUT_CSV << "\n"; return 1; }
    out_snap << "step,x,y,ez\n";
    std::ofstream out_disp(DISPERSION_CSV);
    if(!out_disp){ std::cerr << "Cannot open " << DISPERSION_CSV << "\n"; return 1; }
    out_disp << "step,time_s,ez_p1,ez_p2\n";
    std::ofstream out_pv(PHASE_VEL_CSV);
    if(!out_pv){ std::cerr << "Cannot open " << PHASE_VEL_CSV << "\n"; return 1; }
    out_pv << "step,time_s,leading_edge_m,vph_m_per_s\n";
    std::cout << "Mode: CW monochromatic  f = " << F_CW/1e9 << " GHz"
              << "  T_ramp = " << T_RAMP << " steps\n";
    std::cout << "DT = " << DT << " s  |  Courant (per axis) = " << C0*DT/DX << "\n\n";
    double prev_radius = 0.0;
    double prev_time   = 0.0;
    for(int n = 0; n < NSTEP; ++n){
        const double t = n * DT;
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

        for(int j = 0; j < NY; ++j){
            ez[0][j]    = ez_left[j]  + mur_cx * (ez[1][j]    - ez[0][j]);
            ez[NX-1][j] = ez_right[j] + mur_cx * (ez[NX-2][j] - ez[NX-1][j]);
        }
        for(int i = 0; i < NX; ++i){
            ez[i][0]    = ez_bot[i] + mur_cy * (ez[i][1]    - ez[i][0]);
            ez[i][NY-1] = ez_top[i] + mur_cy * (ez[i][NY-2] - ez[i][NY-1]);
        }

        probe1_ts.push_back(ez[P1_X][P1_Y]);
        probe2_ts.push_back(ez[P2_X][P2_Y]);

        out_disp << n << ',' << t << ',' << probe1_ts.back() << ',' << probe2_ts.back() << '\n';
        if(n >= NSTEP - SWR_WINDOW){
            for(int i = 0; i < NX; ++i)
                swr_profile[i] = std::max(swr_profile[i], std::abs(ez[i][SRC_Y]));
        }
        if(n > T_RAMP && n % SNAP == 0){
            for(int i = 0; i < NX; ++i)
                for(int j = 0; j < NY; ++j)
                    out_snap << n << ',' << i << ',' << j << ',' << ez[i][j] << '\n';
            out_snap << n << ',' << SRC_X << ',' << SRC_Y << ',' << ez[SRC_X][SRC_Y] << '\n';
        }
        if(n <= PV_TRACK_END && n % SNAP == 0){
            double r_now = wavefront_radius(ez, SRC_X, SRC_Y);
            double vph   = 0.0;
            if(n > 0 && (t - prev_time) > 0.0)
                vph = (r_now - prev_radius) / (t - prev_time);
            if(r_now > 0.0)
                out_pv << n << ',' << t << ',' << r_now << ',' << vph << '\n';
            prev_radius = r_now;
            prev_time   = t;
        }

        if(n % 100 == 0)
            std::cout << "step " << n << "/" << NSTEP << "\n";
    }

    out_snap.close();
    out_disp.close();
    out_pv.close();
    {
        std::ofstream out_swr(SWR_CSV);
        if(!out_swr){ std::cerr << "Cannot open " << SWR_CSV << "\n"; return 1; }
        out_swr << "x_cell,x_m,max_ez_envelope\n";
        for(int i = 0; i < NX; ++i)
            out_swr << i << ',' << (i * DX) << ',' << swr_profile[i] << '\n';
        out_swr.close();
    }

    std::cout << "\nSimulation complete.\n";
    std::cout << "Fields written to      : " << OUTPUT_CSV     << "\n";
    std::cout << "Probe time series to   : " << DISPERSION_CSV << "\n";
    std::cout << "Phase velocity data to : " << PHASE_VEL_CSV  << "\n";
    std::cout << "SWR scan written to    : " << SWR_CSV        << "\n";
    print_dispersion_summary(probe1_ts, probe2_ts);

    return 0;
}