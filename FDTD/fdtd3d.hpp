// fdtd3d.hpp — shared 3-D Yee engine with CPML, used by 3D.cpp, cavity.cpp, cpml_test.cpp
//
// Grid layout (nx × ny × nz cells, nodes 0..n along each axis, cubic cells of size dx):
//   Ex(i+½, j,   k  )   Hx(i,   j+½, k+½)
//   Ey(i,   j+½, k  )   Hy(i+½, j,   k+½)
//   Ez(i,   j,   k+½)   Hz(i+½, j+½, k  )
// Every array holds (nx+1)(ny+1)(nz+1) values; array index i stores the component at
// its staggered position above. Tangential E on the outer faces is never updated, so the
// outermost boundary is a PEC wall (that is the whole boundary when npml = 0).
//
// CPML (Roden & Gedney 2000; Taflove & Hagness ch. 7):
//   σ(d) = σ_max d^m,  κ(d) = 1 + (κ_max-1) d^m,  α(d) = α_max (1-d),   d = depth / (npml Δ)
//   b = exp(-(σ/κ + α) Δt),  a = σ / (κ (σ + κ α)) · (b - 1)
// σ and α are stored as σ/ε0 and α/ε0 (units 1/s). Profiles are sampled at integer nodes for
// E-field derivatives and at half-integer nodes for H-field derivatives, symmetric on both
// sides. ψ arrays exist only inside the PML slab of their own axis.

#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

namespace fdtd {

constexpr double C0   = 2.99792458e8;
constexpr double MU0  = 1.25663706212e-6;
constexpr double EPS0 = 8.8541878128e-12;
constexpr double ETA0 = 376.730313668;

enum Comp { EX, EY, EZ, HX, HY, HZ };

struct CPMLParams {
    int    npml      = 0;     // layer thickness [cells]; 0 → plain PEC box
    double sigma_max = 0.0;   // σ_max/ε0 [1/s]
    double kappa_max = 1.0;
    double alpha_max = 0.0;   // α_max/ε0 [1/s]
    int    m         = 3;     // polynomial grading order
};

// Taflove & Hagness eq. (7.67): σ_opt ≈ 0.8 (m+1) / (η0 Δ)  [S/m], returned as σ/ε0 [1/s]
inline double sigma_opt(int m, double dx) { return 0.8 * (m + 1) / (ETA0 * dx) / EPS0; }

struct Grid {
    int    n[3];
    double dx, dt;
    size_t stride[3];
    std::vector<double> f[6];        // ex, ey, ez, hx, hy, hz
    std::vector<double> ce[3];       // Δt/(ε0 εr Δ) at Ex, Ey, Ez nodes; empty → vacuum
    double ce0, ch;                  // vacuum E coefficient, H coefficient Δt/(μ0 Δ)
    CPMLParams pml;

    // per-axis profile, index m = node (E) or node+½ (H)
    std::vector<double> kinv_e[3], b_e[3], a_e[3], kinv_h[3], b_h[3], a_h[3];

    struct Term { int F, G, ax; double sgn; std::vector<double> psi; };
    std::vector<Term> hterms, eterms;

    Grid(int nx, int ny, int nz, double dx_, double courant, const CPMLParams& p)
        : n{nx, ny, nz}, dx(dx_), pml(p)
    {
        dt  = courant * dx / (C0 * std::sqrt(3.0));
        ce0 = dt / (EPS0 * dx);
        ch  = dt / (MU0 * dx);
        stride[2] = 1;
        stride[1] = size_t(nz + 1);
        stride[0] = size_t(ny + 1) * (nz + 1);
        const size_t N = size_t(nx + 1) * stride[0];
        for (auto& v : f) v.assign(N, 0.0);
        for (int a = 0; a < 3; ++a) build_profile(a);
        if (pml.npml > 0) {
            // ∂H/∂t = -(1/μ0) ∇×E ;  ∂E/∂t = (1/ε) ∇×H
            hterms = {{HX, EZ, 1, -1, {}}, {HX, EY, 2, +1, {}}, {HY, EX, 2, -1, {}},
                      {HY, EZ, 0, +1, {}}, {HZ, EY, 0, -1, {}}, {HZ, EX, 1, +1, {}}};
            eterms = {{EX, HZ, 1, +1, {}}, {EX, HY, 2, -1, {}}, {EY, HX, 2, +1, {}},
                      {EY, HZ, 0, -1, {}}, {EZ, HY, 0, +1, {}}, {EZ, HX, 1, -1, {}}};
            for (auto* T : {&hterms, &eterms})
                for (auto& t : *T) t.psi.assign(slab_size(t.ax), 0.0);
        }
    }

    double courant() const { return C0 * dt / dx; }
    size_t id(int i, int j, int k) const { return i * stride[0] + j * stride[1] + k; }
    double& at(Comp c, int i, int j, int k) { return f[c][id(i, j, k)]; }

    //  Materials 
    // eps(x, y, z) in cell units (node i is at x = i). Each E component gets εr averaged
    // over an nsub³ sub-grid of the unit cell around its own staggered position; only nodes
    // inside [lo, hi] (cell units) are sampled, everything else is vacuum.
    void set_eps(const std::function<double(double, double, double)>& eps, int nsub,
                 const double lo[3], const double hi[3])
    {
        const double off[3][3] = {{0.5, 0, 0}, {0, 0.5, 0}, {0, 0, 0.5}};
        for (int c = 0; c < 3; ++c) {
            ce[c].assign(f[0].size(), ce0);
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i <= n[0]; ++i)
                for (int j = 0; j <= n[1]; ++j)
                    for (int k = 0; k <= n[2]; ++k) {
                        const double p[3] = {i + off[c][0], j + off[c][1], k + off[c][2]};
                        bool inside = true;
                        for (int a = 0; a < 3; ++a)
                            inside &= (p[a] >= lo[a] - 1 && p[a] <= hi[a] + 1);
                        if (!inside) continue;
                        double s = 0;
                        for (int u = 0; u < nsub; ++u)
                            for (int v = 0; v < nsub; ++v)
                                for (int w = 0; w < nsub; ++w)
                                    s += eps(p[0] - 0.5 + (u + 0.5) / nsub,
                                             p[1] - 0.5 + (v + 0.5) / nsub,
                                             p[2] - 0.5 + (w + 0.5) / nsub);
                        ce[c][id(i, j, k)] = ce0 / (s / (nsub * nsub * nsub));
                    }
        }
    }

    //  Time stepping 
    void update_H()
    {
        auto& ex = f[EX]; auto& ey = f[EY]; auto& ez = f[EZ];
        auto& hx = f[HX]; auto& hy = f[HY]; auto& hz = f[HZ];
        const auto &kx = kinv_h[0], &ky = kinv_h[1], &kz = kinv_h[2];
        const size_t sx = stride[0], sy = stride[1];
        const int nx = n[0], ny = n[1], nz = n[2];

        #pragma omp parallel for schedule(static)
        for (int i = 0; i <= nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k) {
                    const size_t p = id(i, j, k);
                    hx[p] -= ch * ((ez[p + sy] - ez[p]) * ky[j] - (ey[p + 1] - ey[p]) * kz[k]);
                }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j <= ny; ++j)
                for (int k = 0; k < nz; ++k) {
                    const size_t p = id(i, j, k);
                    hy[p] -= ch * ((ex[p + 1] - ex[p]) * kz[k] - (ez[p + sx] - ez[p]) * kx[i]);
                }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k <= nz; ++k) {
                    const size_t p = id(i, j, k);
                    hz[p] -= ch * ((ey[p + sx] - ey[p]) * kx[i] - (ex[p + sy] - ex[p]) * ky[j]);
                }
        for (auto& t : hterms) cpml_term(t, true);
    }

    void update_E()
    {
        auto& ex = f[EX]; auto& ey = f[EY]; auto& ez = f[EZ];
        auto& hx = f[HX]; auto& hy = f[HY]; auto& hz = f[HZ];
        const auto &kx = kinv_e[0], &ky = kinv_e[1], &kz = kinv_e[2];
        const size_t sx = stride[0], sy = stride[1];
        const int nx = n[0], ny = n[1], nz = n[2];
        const bool vac = ce[0].empty();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nx; ++i)
            for (int j = 1; j < ny; ++j)
                for (int k = 1; k < nz; ++k) {
                    const size_t p = id(i, j, k);
                    const double c = vac ? ce0 : ce[0][p];
                    ex[p] += c * ((hz[p] - hz[p - sy]) * ky[j] - (hy[p] - hy[p - 1]) * kz[k]);
                }
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 1; k < nz; ++k) {
                    const size_t p = id(i, j, k);
                    const double c = vac ? ce0 : ce[1][p];
                    ey[p] += c * ((hx[p] - hx[p - 1]) * kz[k] - (hz[p] - hz[p - sx]) * kx[i]);
                }
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < nx; ++i)
            for (int j = 1; j < ny; ++j)
                for (int k = 0; k < nz; ++k) {
                    const size_t p = id(i, j, k);
                    const double c = vac ? ce0 : ce[2][p];
                    ez[p] += c * ((hy[p] - hy[p - sx]) * kx[i] - (hx[p] - hx[p - sy]) * ky[j]);
                }
        for (auto& t : eterms) cpml_term(t, false);
    }

    // Update range [lo, hi) of each component along each axis (see layout above)
    static void range(int c, const int n[3], int lo[3], int hi[3])
    {
        for (int a = 0; a < 3; ++a) { lo[a] = 0; hi[a] = n[a]; }
        if (c < 3) {                       // E: tangential nodes on the outer faces stay 0
            for (int a = 0; a < 3; ++a) if (a != c) lo[a] = 1;
        } else {                           // H: the normal axis includes node n
            hi[c - 3] = n[c - 3] + 1;
        }
    }

private:
    void build_profile(int a)
    {
        const int N = n[a], np = pml.npml;
        for (auto* v : {&kinv_e[a], &b_e[a], &a_e[a], &kinv_h[a], &b_h[a], &a_h[a]})
            v->assign(N + 1, 0.0);
        auto coef = [&](double x, double& kinv, double& b, double& av) {
            double d = 0;
            if (np > 0) {
                if (x < np)            d = (np - x) / np;
                else if (x > N - np)   d = (x - (N - np)) / np;
            }
            d = std::clamp(d, 0.0, 1.0);
            const double sigma = pml.sigma_max * std::pow(d, pml.m);
            const double kappa = 1.0 + (pml.kappa_max - 1.0) * std::pow(d, pml.m);
            const double alpha = d > 0 ? pml.alpha_max * (1.0 - d) : 0.0;
            kinv = 1.0 / kappa;
            b    = std::exp(-(sigma / kappa + alpha) * dt);
            av   = sigma > 0 ? sigma / (kappa * (sigma + kappa * alpha)) * (b - 1.0) : 0.0;
        };
        for (int m = 0; m <= N; ++m) {
            coef(m,       kinv_e[a][m], b_e[a][m], a_e[a][m]);
            coef(m + 0.5, kinv_h[a][m], b_h[a][m], a_h[a][m]);
        }
    }

    int planes() const { return 2 * pml.npml + 1; }
    int plane(int ax, int m) const
    {
        const int N = n[ax], np = pml.npml;
        if (m < np) return m;
        if (m >= N - np) return np + (m - (N - np));
        return -1;
    }
    size_t slab_size(int ax) const
    {
        size_t s = planes();
        for (int a = 0; a < 3; ++a) if (a != ax) s *= size_t(n[a] + 1);
        return s;
    }

    // F += sgn · coef · ψ,  ψ ← b ψ + a ΔG   (ΔG = forward difference for H, backward for E)
    void cpml_term(Term& t, bool hfield)
    {
        int lo[3], hi[3];
        range(t.F, n, lo, hi);
        const int ax = t.ax;
        const auto& B = hfield ? b_h[ax] : b_e[ax];
        const auto& A = hfield ? a_h[ax] : a_e[ax];
        auto& F = f[t.F];
        const auto& G = f[t.G];
        const size_t s = stride[ax];
        const bool vac = hfield || ce[0].empty();
        const std::vector<double>* cv = hfield ? nullptr : &ce[t.F];
        const double c0 = hfield ? ch : ce0;
        size_t pd[3];                       // ψ strides
        int pdim[3] = {n[0] + 1, n[1] + 1, n[2] + 1};
        pdim[ax] = planes();
        pd[2] = 1; pd[1] = size_t(pdim[2]); pd[0] = size_t(pdim[1]) * pdim[2];

        const int np = pml.npml, N = n[ax];
        const int r[2][2] = {{lo[ax], std::min(hi[ax], np)},
                             {std::max(lo[ax], N - np), hi[ax]}};
        for (const auto& rr : r) {
            int L[3] = {lo[0], lo[1], lo[2]}, H[3] = {hi[0], hi[1], hi[2]};
            L[ax] = rr[0];
            H[ax] = rr[1];
            // natural (i, j, k) order keeps the k loop contiguous in memory for every slab
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = L[0]; i < H[0]; ++i)
                for (int j = L[1]; j < H[1]; ++j) {
                    int c[3] = {i, j, 0};
                    for (int k = L[2]; k < H[2]; ++k) {
                        c[2] = k;
                        const int m = c[ax];
                        int q3[3] = {i, j, k};
                        q3[ax] = plane(ax, m);
                        const size_t p = id(i, j, k);
                        const size_t q = q3[0] * pd[0] + q3[1] * pd[1] + q3[2];
                        const double dG = hfield ? G[p + s] - G[p] : G[p] - G[p - s];
                        t.psi[q] = B[m] * t.psi[q] + A[m] * dG;
                        F[p] += t.sgn * (vac ? c0 : (*cv)[p]) * t.psi[q];
                    }
                }
        }
    }
};

// Differentiated Gaussian (zero mean → leaves no static charge behind)
inline double dgauss(double t, double t0, double tau)
{
    const double x = (t - t0) / tau;
    return -x * std::exp(-x * x);
}

// Gaussian-modulated sinusoid centred on f0
inline double gauss_sine(double t, double t0, double tau, double f0)
{
    const double x = (t - t0) / tau;
    return std::exp(-x * x) * std::sin(2.0 * M_PI * f0 * (t - t0));
}

}  // namespace fdtd
