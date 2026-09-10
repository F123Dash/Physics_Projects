// cpml_test.cpp — CPML reflection error against a reflection-free reference run

#include "fdtd3d.hpp"

#include <array>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <vector>

using namespace fdtd;

constexpr double DX       = 7e-5;
constexpr double COURANT  = 0.99;    // Δt = 0.99 delta/(c√3)
constexpr int    INTERIOR = 30;      // cells between the CPML layers of the small domain
constexpr int    NSTEP    = 300;
constexpr double TAU_STEPS = 10.0;   // pulse width [steps]
constexpr double T0_STEPS  = 50.0;

// probe offsets from the source (2 cells inside the CPML interface)
constexpr int OFF = INTERIOR / 2 - 2;
const std::array<std::array<int, 3>, 3> PROBES = {{{OFF, 0, 0}, {OFF, OFF, 0}, {OFF, OFF, OFF}}};
const char* PROBE_NAME[3] = {"face", "edge", "corner"};

using Trace = std::vector<std::array<double, 3>>;

static Trace run(int ncell, const CPMLParams& p)
{
    Grid g(ncell, ncell, ncell, DX, COURANT, p);
    const int c = ncell / 2;
    const double tau = TAU_STEPS * g.dt, t0 = T0_STEPS * g.dt;
    Trace tr(NSTEP);
    for (int n = 0; n < NSTEP; ++n) {
        g.update_H();
        g.update_E();
        g.at(EZ, c, c, c) += dgauss((n + 1) * g.dt, t0, tau);
        for (int q = 0; q < 3; ++q)
            tr[n][q] = g.at(EZ, c + PROBES[q][0], c + PROBES[q][1], c + PROBES[q][2]);
    }
    return tr;
}

static std::array<double, 3> error_db(const Trace& s, const Trace& r)
{
    std::array<double, 3> out{};
    for (int q = 0; q < 3; ++q) {
        double emax = 0, rmax = 0;
        for (int n = 0; n < NSTEP; ++n) {
            emax = std::max(emax, std::abs(s[n][q] - r[n][q]));
            rmax = std::max(rmax, std::abs(r[n][q]));
        }
        out[q] = 20.0 * std::log10(emax / rmax);
    }
    return out;
}

int main()
{
    std::filesystem::create_directories("./data_outputs");
    const int NPML = 12;
    const double S = COURANT / std::sqrt(3.0);
    const double s_opt = sigma_opt(3, DX);

    // Reference: CPML interface D cells from the source. The earliest reflection reaches a probe
    // after travelling D + (D - OFF) cells at ≤ c, i.e. after (2D - OFF)/S steps.
    const int D = int(std::ceil((NSTEP * S + OFF) / 2.0)) + 4;
    const int n_ref = 2 * (D + NPML);
    const double t_clean = (2.0 * D - OFF) / S;
    std::printf("Reference domain %d^3 cells; first possible reflection at step %.0f > NSTEP = %d\n",
                n_ref, t_clean, NSTEP);
    CPMLParams pref{NPML, s_opt, 1.0, 1e11, 3};
    const Trace ref = run(n_ref, pref);

    FILE* fp = std::fopen("./data_outputs/cpml_scan.csv", "w");
    std::fprintf(fp, "label,npml,sigma_factor,sigma_max,kappa_max,alpha_max,err_face_dB,err_edge_dB,err_corner_dB,err_worst_dB\n");
    std::printf("\nsigma_opt = %.3e 1/s  (0.8(m+1)/(η0Δ)/ε0, m = 3)\n", s_opt);
    std::printf("Probes 2 cells inside the CPML interface; error = max|E-E_ref|/max|E_ref|\n\n");
    std::printf(" npml  sigma/sigma_opt  κ_max    sigma_max   |  face     edge    corner  |  worst [dB]\n");

    auto report = [&](const char* label, int npml, double fac, double kap, double alp) {
        CPMLParams p{npml, fac * s_opt, kap, alp, 3};
        const Trace s = run(INTERIOR + 2 * npml, p);
        const auto e = error_db(s, ref);
        const double worst = std::max({e[0], e[1], e[2]});
        std::printf(" %3d   %6.2f   %4.1f   %8.1e  | %6.1f  %6.1f  %6.1f   |  %6.1f  %s\n", npml, fac, kap, alp,
                    e[0], e[1], e[2], worst, label);
        std::fprintf(fp, "%s,%d,%.4f,%.6e,%.2f,%.3e,%.3f,%.3f,%.3f,%.3f\n", label, npml, fac, fac * s_opt, kap,
                     alp, e[0], e[1], e[2], worst);
        return worst;
    };

    const double facs[] = {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0};
    double best = 0, best_fac = 0, best_kap = 0;
    for (double kap : {1.0, 5.0})
        for (double fac : facs) {
            const double w = report("scan", NPML, fac, kap, 1e11);
            if (w < best) { best = w; best_fac = fac; best_kap = kap; }
        }
    report("original", 15, 4.0e13 / s_opt, 5.0, 1e11);
    report("control: sigma=0 (PEC wall behind)", NPML, 0.0, 1.0, 0.0);
    std::fclose(fp);

    std::printf("\nBest (npml = %d): sigma_max = %.2f sigma_opt, κ_max = %.0f → %.1f dB worst-case\n", NPML, best_fac,
                best_kap, best);
    std::printf("Written: ./data_outputs/cpml_scan.csv\n");
    std::printf("Note: the CPML interior here is a vacuum box; the dB values are the field error at the\n"
                "probes relative to a reflection-free run, not a plane-wave reflection coefficient.\n");
    return 0;
}
