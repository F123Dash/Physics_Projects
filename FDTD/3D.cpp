// 3D.cpp — 3-D FDTD plane-wave scattering: TFSF source + CPML (engine in fdtd3d.hpp)
//
// An x-propagating, z-polarised plane wave (Gaussian-modulated pulse) is injected with a
// total-field / scattered-field (TFSF) box. The incident field comes from a 1-D auxiliary
// Yee grid with the same Δ and Δt, so for axis-aligned incidence its numerical dispersion
// matches the 3-D grid exactly.
//
// Modes
//   ./3D mie [R]   homogeneous sphere εr = 4, radius a = 1.4 mm resolved by R cells
//                  (default 20 → Δ = 70 µm); changing R is a grid-convergence study.
//                  Scattered power through a closed surface in the scattered-field region is
//                  DFT'd at 41 frequencies (30–110 GHz) → Q_sca, compared with Mie theory.
//   ./3D leak      same geometry with no sphere: TFSF leakage (should be ~round-off).
//   ./3D grin      GRIN sphere εr(r) = 1 + 3(1 − r/R), R = 40 cells (2.8 mm ≈ 0.7 λ at 75 GHz).
//                  |Ez/E0|² at 75 GHz on the z = centre plane, showing the focal spot.
//
// Outputs (./data_outputs/): 3D_<mode>_meta.csv, mie_qsca_R<R>.csv, mie_curve.csv, grin_slice.csv

#include "fdtd3d.hpp"

#include <complex>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace fdtd;
using cplx = std::complex<double>;

constexpr double DX_GRIN = 7e-5;    // [m]  λ/Δ ≈ 57 at 75 GHz
constexpr double A_MIE   = 1.4e-3;  // [m]  Mie sphere radius; Δ = A_MIE / R_cells
constexpr double COURANT = 0.99;    // Δt = 0.99 Δ/(c√3)  →  S = 0.5716
constexpr int    NPML    = 12;
// CPML parameters from cpml_test (best worst-case error at 12 cells)
constexpr double SIGMA_FACTOR = 0.75, KAPPA_MAX = 1.0, ALPHA_MAX = 1e11;

// ── Mie theory (Bohren & Huffman §4.4), real relative index m, μ = 1 ────────────────────────
static double mie_qsca(double m, double x)
{
    const int nmax = int(x + 4.0 * std::cbrt(x) + 2.0);
    auto psi  = [](int n, double z) { return z * std::sph_bessel(n, z); };
    auto dpsi = [&](int n, double z) { return psi(n - 1, z) - n * psi(n, z) / z; };
    auto xi   = [](int n, double z) { return z * cplx(std::sph_bessel(n, z), std::sph_neumann(n, z)); };
    auto dxi  = [&](int n, double z) { return xi(n - 1, z) - double(n) * xi(n, z) / z; };
    double q = 0;
    for (int n = 1; n <= nmax; ++n) {
        const double mx = m * x;
        const cplx a = (m * psi(n, mx) * dpsi(n, x) - psi(n, x) * dpsi(n, mx)) /
                       (m * psi(n, mx) * dxi(n, x) - xi(n, x) * dpsi(n, mx));
        const cplx b = (psi(n, mx) * dpsi(n, x) - m * psi(n, x) * dpsi(n, mx)) /
                       (psi(n, mx) * dxi(n, x) - m * xi(n, x) * dpsi(n, mx));
        q += (2 * n + 1) * (std::norm(a) + std::norm(b));
    }
    return 2.0 * q / (x * x);
}

// ── TFSF plane wave: +x propagation, Ez polarisation, Hy = −Ez/η ───────────────────────────
// Total-field region: E nodes with i∈[i0,i1], j∈[j0,j1], k∈[k0,k1]. The 1-D grid shares the
// 3-D x index: ez1[m] at x = m, hy1[m] at x = m + ½. It is long enough that its far-end
// reflection never returns to the box during the run.
struct TFSF {
    int i0, i1, j0, j1, k0, k1;
    std::vector<double> ez1, hy1;
    double ce0, ch;

    TFSF(const Grid& g, const int b[6], int nstep) : i0(b[0]), i1(b[1]), j0(b[2]), j1(b[3]), k0(b[4]), k1(b[5])
    {
        const size_t na = g.n[0] + 1 + size_t(0.5 * nstep * g.courant()) + 20;
        ez1.assign(na, 0.0);
        hy1.assign(na, 0.0);
        ce0 = g.ce0;
        ch  = g.ch;
    }
    double ce(const Grid& g, int c, size_t p) const { return g.ce[c].empty() ? ce0 : g.ce[c][p]; }

    void correct_H(Grid& g) const   // uses Ez_inc at time n
    {
        for (int j = j0; j <= j1; ++j)
            for (int k = k0; k < k1; ++k) {
                g.at(HY, i0 - 1, j, k) -= ch * ez1[i0];
                g.at(HY, i1, j, k)     += ch * ez1[i1];
            }
        for (int i = i0; i <= i1; ++i)
            for (int k = k0; k < k1; ++k) {
                g.at(HX, i, j0 - 1, k) += ch * ez1[i];
                g.at(HX, i, j1, k)     -= ch * ez1[i];
            }
    }
    void correct_E(Grid& g) const   // uses Hy_inc at time n + ½
    {
        for (int j = j0; j <= j1; ++j)
            for (int k = k0; k < k1; ++k) {
                size_t p = g.id(i0, j, k);
                g.f[EZ][p] -= ce(g, EZ, p) * hy1[i0 - 1];
                p = g.id(i1, j, k);
                g.f[EZ][p] += ce(g, EZ, p) * hy1[i1];
            }
        for (int i = i0; i < i1; ++i)
            for (int j = j0; j <= j1; ++j) {
                size_t p = g.id(i, j, k0);
                g.f[EX][p] += ce(g, EX, p) * hy1[i];
                p = g.id(i, j, k1);
                g.f[EX][p] -= ce(g, EX, p) * hy1[i];
            }
    }
    void step_H()
    {
        for (size_t m = 0; m + 1 < hy1.size(); ++m) hy1[m] += ch * (ez1[m + 1] - ez1[m]);
    }
    void step_E(double src)
    {
        for (size_t m = 1; m + 1 < ez1.size(); ++m) ez1[m] += ce0 * (hy1[m] - hy1[m - 1]);
        ez1[0] = src;
    }
};

// ── DFT of the tangential fields on a closed box (scattered-field region) ──────────────────
// Sample points are the face-cell centres; each staggered component is averaged onto them.
struct FluxBox {
    struct Pt { int face; size_t idx[4][4]; double w[4][4]; };   // comps: E_t1, E_t2, H_t1, H_t2
    std::vector<Pt> pts;
    std::vector<double> freqs;
    std::vector<cplx> acc;            // [pt][freq][comp]
    double max_e = 0;                 // max |E_t| seen on the surface (for leakage / decay checks)

    FluxBox(const Grid& g, const int lo[3], const int hi[3], std::vector<double> f) : freqs(std::move(f))
    {
        for (int ax = 0; ax < 3; ++ax)
            for (int side = 0; side < 2; ++side) {
                const int t1 = (ax + 1) % 3, t2 = (ax + 2) % 3;
                for (int u = lo[t1]; u < hi[t1]; ++u)
                    for (int v = lo[t2]; v < hi[t2]; ++v) {
                        double pos[3];
                        pos[ax] = side ? hi[ax] : lo[ax];
                        pos[t1] = u + 0.5;
                        pos[t2] = v + 0.5;
                        Pt p{};
                        p.face = 2 * ax + side;
                        const int comps[4] = {t1, t2, 3 + t1, 3 + t2};
                        for (int c = 0; c < 4; ++c) stencil(g, comps[c], pos, p.idx[c], p.w[c]);
                        pts.push_back(p);
                    }
            }
        acc.assign(pts.size() * freqs.size() * 4, 0.0);
    }

    // Average component c (0..5) onto pos (cell units)
    static void stencil(const Grid& g, int c, const double pos[3], size_t idx[4], double w[4])
    {
        int lo[3], cnt[3];
        for (int a = 0; a < 3; ++a) {
            const bool half_comp = (c < 3) ? (a == c) : (a != c - 3);
            const bool half_pos  = std::abs(pos[a] - std::floor(pos[a]) - 0.5) < 1e-9;
            if (half_comp == half_pos) { lo[a] = int(std::floor(pos[a])); cnt[a] = 1; }
            else if (half_comp)        { lo[a] = int(pos[a]) - 1;        cnt[a] = 2; }  // integer pos
            else                       { lo[a] = int(std::floor(pos[a])); cnt[a] = 2; } // half pos
        }
        int m = 0;
        const double wt = 1.0 / (cnt[0] * cnt[1] * cnt[2]);
        for (int a = 0; a < cnt[0]; ++a)
            for (int b = 0; b < cnt[1]; ++b)
                for (int d = 0; d < cnt[2]; ++d) {
                    idx[m] = g.id(lo[0] + a, lo[1] + b, lo[2] + d);
                    w[m++] = wt;
                }
        for (; m < 4; ++m) { idx[m] = 0; w[m] = 0; }
    }

    double accumulate(const Grid& g, int n)   // returns max |E_t| on the surface this step
    {
        const size_t nf = freqs.size();
        std::vector<cplx> phE(nf), phH(nf);
        for (size_t q = 0; q < nf; ++q) {
            const double w = 2 * M_PI * freqs[q];
            phE[q] = std::polar(1.0, -w * (n + 1.0) * g.dt);
            phH[q] = std::polar(1.0, -w * (n + 0.5) * g.dt);
        }
        double me = 0;
        const int comps_of_face[3][4] = {{EY, EZ, HY, HZ}, {EZ, EX, HZ, HX}, {EX, EY, HX, HY}};
        #pragma omp parallel for schedule(static) reduction(max : me)
        for (size_t i = 0; i < pts.size(); ++i) {
            const Pt& p = pts[i];
            double val[4];
            for (int c = 0; c < 4; ++c) {
                const auto& F = g.f[comps_of_face[p.face / 2][c]];
                val[c] = 0;
                for (int s = 0; s < 4; ++s) val[c] += p.w[c][s] * F[p.idx[c][s]];
            }
            me = std::max({me, std::abs(val[0]), std::abs(val[1])});
            cplx* a = &acc[i * nf * 4];
            for (size_t q = 0; q < nf; ++q) {
                a[4 * q + 0] += val[0] * phE[q];
                a[4 * q + 1] += val[1] * phE[q];
                a[4 * q + 2] += val[2] * phH[q];
                a[4 * q + 3] += val[3] * phH[q];
            }
        }
        max_e = std::max(max_e, me);
        return me;
    }

    // Outward time-averaged power at frequency q:  ½ Re ∮ (E × H*)·n dA
    double power(size_t q, double dx) const
    {
        const size_t nf = freqs.size();
        double P = 0;
        for (size_t i = 0; i < pts.size(); ++i) {
            const cplx* a = &acc[(i * nf + q) * 4];
            const double sgn = (pts[i].face % 2) ? 1.0 : -1.0;
            P += sgn * 0.5 * std::real(a[0] * std::conj(a[3]) - a[1] * std::conj(a[2]));
        }
        return P * dx * dx;
    }
};

static void write_meta(const std::string& mode, const Grid& g, double cx, double cy, double cz, double R,
                       const int box[6], const int flux_lo[3], const int flux_hi[3], double f_map)
{
    FILE* fp = std::fopen(("./data_outputs/3D_" + mode + "_meta.csv").c_str(), "w");
    std::fprintf(fp, "key,value\n");
    std::fprintf(fp, "nx,%d\nny,%d\nnz,%d\nnpml,%d\ndx,%.10e\ndt,%.10e\n", g.n[0], g.n[1], g.n[2], NPML, g.dx, g.dt);
    std::fprintf(fp, "cx,%.3f\ncy,%.3f\ncz,%.3f\nR_cells,%.3f\n", cx, cy, cz, R);
    std::fprintf(fp, "i0,%d\ni1,%d\nj0,%d\nj1,%d\nk0,%d\nk1,%d\n", box[0], box[1], box[2], box[3], box[4], box[5]);
    std::fprintf(fp, "flux_x0,%d\nflux_x1,%d\nflux_y0,%d\nflux_y1,%d\n", flux_lo[0], flux_hi[0], flux_lo[1], flux_hi[1]);
    std::fprintf(fp, "f_map_Hz,%.6e\n", f_map);
    std::fclose(fp);
}

int main(int argc, char** argv)
{
    const std::string mode = argc > 1 ? argv[1] : "mie";
    if (mode != "mie" && mode != "leak" && mode != "grin") {
        std::cerr << "usage: ./3D [mie [R_cells] | leak | grin]\n";
        return 1;
    }
    std::filesystem::create_directories("./data_outputs");
    const bool grin = mode == "grin";
    const double R  = grin ? 40.0 : (argc > 2 ? std::atof(argv[2]) : 20.0);

    // ── Geometry (cells): CPML | gap | flux surface | 3 | TFSF box | margin | sphere ───────
    const int margin = grin ? 8 : 4, flux_gap = 3, pml_gap = 5;
    const int Ri = int(std::ceil(R));
    const int behind = grin ? 70 : margin;            // extra total-field room behind the sphere
    const int lead = NPML + pml_gap + flux_gap;
    const int c  = lead + Ri + margin;                // sphere centre (y, z, and x)
    const int nx = c + Ri + behind + lead;
    const int ny = 2 * c, nz = 2 * c;
    const int box[6] = {c - Ri - margin, c + Ri + behind, c - Ri - margin, c + Ri + margin,
                        c - Ri - margin, c + Ri + margin};
    const int flux_lo[3] = {box[0] - flux_gap, box[2] - flux_gap, box[4] - flux_gap};
    const int flux_hi[3] = {box[1] + flux_gap, box[3] + flux_gap, box[5] + flux_gap};

    const double DX = grin ? DX_GRIN : A_MIE / R;
    CPMLParams pml{NPML, SIGMA_FACTOR * sigma_opt(3, DX), KAPPA_MAX, ALPHA_MAX, 3};
    Grid g(nx, ny, nz, DX, COURANT, pml);

    // ── Material ─────────────────────────────────────────────────────────────────────
    const double cx = c, cy = c, cz = c;
    if (mode != "leak") {
        auto eps = [&](double x, double y, double z) {
            const double r = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
            if (r >= R) return 1.0;
            return grin ? 1.0 + 3.0 * (1.0 - r / R) : 4.0;
        };
        const double lo[3] = {cx - R, cy - R, cz - R}, hi[3] = {cx + R, cy + R, cz + R};
        g.set_eps(eps, 4, lo, hi);
    }

    // ── Source and run length ────────────────────────────────────────────────────────
    const double f0 = 70e9, tau = 1.125e-11, t0 = 4.0 * tau;  // spectrum ≥ e^-2 of peak over 30–110 GHz
    // Run until the pulse has crossed the grid and the surface field has rung down to
    // DECAY × its peak (checked over windows of 200 steps), so the DFTs are not truncated.
    const double DECAY = 1e-6;
    const int nmin = int((t0 + 4 * tau) / g.dt + nx / g.courant());
    const int nmax = 40000, window = 200;
    TFSF tf(g, box, nmax);

    std::vector<double> freqs;
    const double f_map = 75e9;
    if (grin) freqs = {f_map};
    else for (int q = 0; q <= 40; ++q) freqs.push_back(30e9 + q * 2e9);
    FluxBox fb(g, flux_lo, flux_hi, freqs);

    // Incident reference (at the sphere centre) and, for grin, the Ez slice DFT
    std::vector<cplx> Einc(freqs.size(), 0.0), Hinc(freqs.size(), 0.0);
    std::vector<cplx> slice, inc_line;
    if (grin) { slice.assign(size_t(nx + 1) * (ny + 1), 0.0); inc_line.assign(nx + 1, 0.0); }
    double max_inc = 0;

    std::printf("3D FDTD %s  |  grid %d x %d x %d, Δ = %.0f µm, S = %.4f, CPML %d cells\n", mode.c_str(), nx,
                ny, nz, DX * 1e6, g.courant(), NPML);
    std::printf("Sphere R = %.1f cells = %.3f mm, TFSF box x[%d,%d] y[%d,%d] z[%d,%d]\n", R,
                R * DX * 1e3, box[0], box[1], box[2], box[3], box[4], box[5]);

    int nstep = nmax;
    double win_max = 0, last_win = 0;
    for (int n = 0; n < nstep; ++n) {
        g.update_H();
        tf.correct_H(g);
        tf.step_H();
        g.update_E();
        tf.correct_E(g);
        tf.step_E(gauss_sine((n + 1) * g.dt, t0, tau, f0));

        win_max = std::max(win_max, fb.accumulate(g, n));
        if ((n + 1) % window == 0) {
            last_win = win_max;
            // leak mode: the surface field is round-off noise, so stop once the pulse has passed
            if (n >= nmin && (mode == "leak" || win_max < DECAY * fb.max_e)) nstep = n + 1;
            win_max = 0;
        }
        max_inc = std::max(max_inc, std::abs(tf.ez1[c]));
        for (size_t q = 0; q < freqs.size(); ++q) {
            const double w = 2 * M_PI * freqs[q];
            Einc[q] += tf.ez1[c] * std::polar(1.0, -w * (n + 1.0) * g.dt);
            Hinc[q] += 0.5 * (tf.hy1[c - 1] + tf.hy1[c]) * std::polar(1.0, -w * (n + 0.5) * g.dt);
        }
        if (grin) {
            const cplx ph = std::polar(1.0, -2 * M_PI * f_map * (n + 1.0) * g.dt);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i <= nx; ++i) {
                inc_line[i] += tf.ez1[i] * ph;
                for (int j = 0; j <= ny; ++j)  // Ez at z = c−½ and c+½ → average onto z = c
                    slice[size_t(i) * (ny + 1) + j] += 0.5 * (g.at(EZ, i, j, c - 1) + g.at(EZ, i, j, c)) * ph;
            }
        }
        if (n % 1000 == 0) std::printf("  step %d\n", n);
    }

    std::printf("\nStopped after %d steps; surface |E| in the last %d steps / peak = %.2e%s\n", nstep, window,
                last_win / fb.max_e, (mode == "leak" || last_win < DECAY * fb.max_e) ? "" : "  (WARNING: not decayed)");

    if (mode == "leak") {
        std::printf("TFSF leakage: max |E_scat| on flux surface / max |E_inc| = %.2e  (%.1f dB)\n",
                    fb.max_e / max_inc, 20 * std::log10(fb.max_e / max_inc));
        write_meta(mode, g, cx, cy, cz, 0, box, flux_lo, flux_hi, 0);
        return 0;
    }

    if (!grin) {
        const double a = R * DX;
        const std::string qfile = "./data_outputs/mie_qsca_R" + std::to_string(int(R)) + ".csv";
        FILE* fp = std::fopen(qfile.c_str(), "w");
        std::fprintf(fp, "f_Hz,size_param,Q_fdtd,Q_mie,rel_err\n");
        std::printf("\n  f [GHz]   x=ka    Q_sca FDTD   Q_sca Mie   rel. err\n");
        double worst = 0, sum = 0;
        for (size_t q = 0; q < freqs.size(); ++q) {
            const double I = 0.5 * std::real(-Einc[q] * std::conj(Hinc[q]));   // |S_inc|
            const double Q = fb.power(q, DX) / (I * M_PI * a * a);
            const double x = 2 * M_PI * freqs[q] * a / C0;
            const double Qm = mie_qsca(2.0, x);
            const double e = Q / Qm - 1.0;
            worst = std::max(worst, std::abs(e));
            sum += std::abs(e);
            std::fprintf(fp, "%.6e,%.6f,%.8e,%.8e,%.6e\n", freqs[q], x, Q, Qm, e);
            if (q % 4 == 0) std::printf("  %6.1f   %6.3f   %10.5f   %10.5f   %+7.2f %%\n", freqs[q] / 1e9, x, Q, Qm, 100 * e);
        }
        std::fclose(fp);
        std::printf("\nQ_sca vs Mie (εr = 4, R = %.0f cells, %zu freqs): mean |err| = %.2f %%, max |err| = %.2f %%\n",
                    R, freqs.size(), 100 * sum / freqs.size(), 100 * worst);

        fp = std::fopen("./data_outputs/mie_curve.csv", "w");
        std::fprintf(fp, "size_param,Q_mie\n");
        for (int q = 0; q <= 600; ++q) {
            const double x = 0.3 + q * (4.0 - 0.3) / 600;
            std::fprintf(fp, "%.6f,%.8e\n", x, mie_qsca(2.0, x));
        }
        std::fclose(fp);
        write_meta(mode, g, cx, cy, cz, R, box, flux_lo, flux_hi, 0);
        std::printf("Written: %s, ./data_outputs/mie_curve.csv, 3D_mie_meta.csv\n", qfile.c_str());
        return 0;
    }

    // ── GRIN: total-field intensity map |Ez/E0|² at 75 GHz on z = c ────────────────────────
    const cplx E0 = Einc[0];
    FILE* fp = std::fopen("./data_outputs/grin_slice.csv", "w");
    std::fprintf(fp, "x_cell,y_cell,intensity,eps_r\n");
    double peak = 0;
    int pi_ = 0, pj = 0;
    for (int i = 0; i <= nx; ++i)
        for (int j = 0; j <= ny; ++j) {
            cplx e = slice[size_t(i) * (ny + 1) + j];
            const bool total = i >= box[0] && i <= box[1] && j >= box[2] && j <= box[3];
            if (!total) e += inc_line[i];            // scattered region: add the incident field back
            const double I = std::norm(e / E0);
            const double r = std::hypot(i - cx, j - cy);
            const double er = r < R ? 1.0 + 3.0 * (1.0 - r / R) : 1.0;
            std::fprintf(fp, "%d,%d,%.6e,%.4f\n", i, j, I, er);
            const bool interior = i > NPML && i < nx - NPML && j > NPML && j < ny - NPML;
            if (interior && I > peak) { peak = I; pi_ = i; pj = j; }
        }
    std::fclose(fp);
    const double lam = C0 / f_map;
    std::printf("\nGRIN sphere at %.0f GHz: peak |Ez/E0|² = %.2f at x − x_c = %+.1f cells (%+.2f λ, %+.2f R), "
                "y − y_c = %+d cells\n", f_map / 1e9, peak, pi_ - cx, (pi_ - cx) * DX / lam, (pi_ - cx) / R,
                int(pj - cy));
    write_meta(mode, g, cx, cy, cz, R, box, flux_lo, flux_hi, f_map);
    std::printf("Written: ./data_outputs/grin_slice.csv, 3D_grin_meta.csv\n");
    return 0;
}
