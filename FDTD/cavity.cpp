// cavity.cpp — numerical-dispersion validation in a closed PEC cavity

#include "fdtd3d.hpp"

#include <complex>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace fdtd;

struct Mode { int m, n, p; double f_cont, f_yee; };

// |Hann-windowed DTFT| of x at frequency f
static double dtft_mag(const std::vector<double>& x, double dt, double f)
{
    const size_t N = x.size();
    const std::complex<double> rot = std::polar(1.0, -2.0 * M_PI * f * dt);
    std::complex<double> ph = 1.0, s = 0.0;
    for (size_t i = 0; i < N; ++i) {
        const double w = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / (N - 1));
        s += w * x[i] * ph;
        ph *= rot;
    }
    return std::abs(s);
}

// Peak of the windowed spectrum inside [f_lo, f_hi]: dense scan, then golden-section refine
static double find_peak(const std::vector<double>& x, double dt, double f_lo, double f_hi)
{
    const double bin = 1.0 / (x.size() * dt);
    const double step = bin / 8.0;
    double best_f = f_lo, best = -1;
    for (double f = f_lo; f <= f_hi; f += step) {
        const double v = dtft_mag(x, dt, f);
        if (v > best) { best = v; best_f = f; }
    }
    double a = best_f - step, b = best_f + step;
    const double g = 0.5 * (std::sqrt(5.0) - 1.0);
    double c = b - g * (b - a), d = a + g * (b - a);
    double fc = dtft_mag(x, dt, c), fd = dtft_mag(x, dt, d);
    for (int it = 0; it < 40; ++it) {
        if (fc > fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = dtft_mag(x, dt, c); }
        else         { a = c; c = d; fc = fd; d = a + g * (b - a); fd = dtft_mag(x, dt, d); }
    }
    return 0.5 * (a + b);
}

struct Result { std::vector<double> f_meas; double dt; int nsteps; };

static Result run(int scale, const std::vector<Mode>& modes_unit, double dx0, double S, int base_cells[3],
                  int nsteps)
{
    const int nx = base_cells[0] * scale, ny = base_cells[1] * scale, nz = base_cells[2] * scale;
    const double dx = dx0 / scale;
    Grid g(nx, ny, nz, dx, S, CPMLParams{});   // npml = 0 → PEC box

    // Asymmetric source / probe positions (avoid nodal planes of low-order modes)
    const int sx = 3 * scale, sy = 4 * scale, sz = 5 * scale;
    const int px = 8 * scale, py = 9 * scale, pz = 11 * scale;
    const double tau = 4.0 * scale * g.dt, t0 = 5.0 * tau;

    std::vector<double> probe(nsteps);
    for (int n = 0; n < nsteps; ++n) {
        g.update_H();
        g.update_E();
        g.at(EZ, sx, sy, sz) += dgauss((n + 1) * g.dt, t0, tau);
        probe[n] = g.at(EZ, px, py, pz);
    }

    const std::string name = "./data_outputs/cavity_probe_" + std::to_string(scale) + "x.csv";
    if (FILE* fp = std::fopen(name.c_str(), "w")) {
        std::fprintf(fp, "step,time_s,ez\n");
        for (int n = 0; n < nsteps; ++n) std::fprintf(fp, "%d,%.10e,%.10e\n", n, (n + 1) * g.dt, probe[n]);
        std::fclose(fp);
    }

    // Predicted frequencies at this resolution
    Result res{{}, g.dt, nsteps};
    const double bin = 1.0 / (nsteps * g.dt);
    for (const auto& mu : modes_unit) {
        const int idx[3] = {mu.m, mu.n, mu.p};
        const int N[3] = {nx, ny, nz};
        double s2 = 0;
        for (int a = 0; a < 3; ++a) s2 += std::pow(std::sin(idx[a] * M_PI / (2.0 * N[a])), 2);
        const double f_yee = std::asin(S / std::sqrt(3.0) * std::sqrt(s2)) / (M_PI * g.dt);
        const double lo = std::min(f_yee, mu.f_cont) - 4 * bin, hi = std::max(f_yee, mu.f_cont) + 4 * bin;
        res.f_meas.push_back(find_peak(probe, g.dt, lo, hi));
    }
    return res;
}

int main()
{
    std::filesystem::create_directories("./data_outputs");

    const double dx0 = 7e-5;               // coarse cell size [m] (same as 3D.cpp)
    const double S   = 0.99;               // deltat = 0.99 delta/(c√3)
    int cells[3] = {12, 14, 17};           // coarse cavity, a×b×d in cells
    const double a = cells[0] * dx0, b = cells[1] * dx0, d = cells[2] * dx0;

    // Continuum TM_z modes (m, n ≥ 1), lowest first; keep isolated ones only
    std::vector<Mode> all;
    for (int m = 1; m <= 4; ++m)
        for (int n = 1; n <= 4; ++n)
            for (int p = 0; p <= 4; ++p) {
                const double f = 0.5 * C0 * std::sqrt(std::pow(m / a, 2) + std::pow(n / b, 2) + std::pow(p / d, 2));
                all.push_back({m, n, p, f, 0});
            }
    std::sort(all.begin(), all.end(), [](const Mode& x, const Mode& y) { return x.f_cont < y.f_cont; });
    std::vector<Mode> modes;
    for (size_t i = 0; i < all.size() && modes.size() < 8; ++i) {
        bool iso = true;   // skip modes within 1 % of another TM_z mode
        for (size_t j = 0; j < all.size(); ++j)
            if (j != i && std::abs(all[j].f_cont / all[i].f_cont - 1.0) < 0.01) iso = false;
        if (iso) modes.push_back(all[i]);
    }

    std::cout << "PEC cavity " << a * 1e3 << " x " << b * 1e3 << " x " << d * 1e3 << " mm, S = "
              << S / std::sqrt(3.0) << " (deltat = 0.99 delta/(c√3))\n";
    std::cout << "Running coarse grid (delta = " << dx0 * 1e6 << " µm) ...\n";
    const Result rc = run(1, modes, dx0, S, cells, 1 << 15);
    std::cout << "Running fine grid   (delta = " << dx0 * 0.5e6 << " µm) ...\n";
    const Result rf = run(2, modes, dx0, S, cells, 1 << 16);

    FILE* fp = std::fopen("./data_outputs/cavity_modes.csv", "w");
    std::fprintf(fp, "m,n,p,f_cont_Hz,cells_per_lambda_coarse,"
                     "f_yee_coarse_Hz,f_meas_coarse_Hz,f_yee_fine_Hz,f_meas_fine_Hz\n");

    std::printf("\n mode       f_cont     labmda/delta |  coarse: meas-Yee   meas-cont |  fine: meas-Yee   meas-cont | order\n");
    std::printf("            [GHz]          |          [ppm]       [%%]     |        [ppm]       [%%]     |\n");
    double worst_yee = 0, order_min = 1e9, order_max = 0;
    for (size_t i = 0; i < modes.size(); ++i) {
        const auto& mu = modes[i];
        auto yee = [&](int scale, double dt) {
            const int N[3] = {cells[0] * scale, cells[1] * scale, cells[2] * scale};
            const int idx[3] = {mu.m, mu.n, mu.p};
            double s2 = 0;
            for (int ax = 0; ax < 3; ++ax) s2 += std::pow(std::sin(idx[ax] * M_PI / (2.0 * N[ax])), 2);
            return std::asin(S / std::sqrt(3.0) * std::sqrt(s2)) / (M_PI * dt);
        };
        const double fyc = yee(1, rc.dt), fyf = yee(2, rf.dt);
        const double fmc = rc.f_meas[i], fmf = rf.f_meas[i];
        const double ec_y = (fmc / fyc - 1) * 1e6, ef_y = (fmf / fyf - 1) * 1e6;
        const double ec_c = (fmc / mu.f_cont - 1) * 100, ef_c = (fmf / mu.f_cont - 1) * 100;
        const double order = std::log2(ec_c / ef_c);
        const double cpl = C0 / mu.f_cont / dx0;
        worst_yee = std::max({worst_yee, std::abs(ec_y), std::abs(ef_y)});
        order_min = std::min(order_min, order);
        order_max = std::max(order_max, order);
        std::printf(" (%d,%d,%d)  %8.4f  %5.1f |  %+10.3f  %+9.4f   | %+10.3f  %+9.4f   | %.3f\n",
                    mu.m, mu.n, mu.p, mu.f_cont / 1e9, cpl, ec_y, ec_c, ef_y, ef_c, order);
        std::fprintf(fp, "%d,%d,%d,%.10e,%.4f,%.10e,%.10e,%.10e,%.10e\n", mu.m, mu.n, mu.p, mu.f_cont, cpl,
                     fyc, fmc, fyf, fmf);
    }
    std::fclose(fp);

    std::printf("\nWorst |f_meas - f_Yee| / f_Yee        = %.3g ppm\n", worst_yee);
    std::printf("Convergence order vs continuum (delta→delta/2) = %.3f … %.3f  (expected 2)\n", order_min, order_max);
    std::printf("Written: ./data_outputs/cavity_modes.csv, cavity_probe_{1x,2x}.csv\n");
    return 0;
}
