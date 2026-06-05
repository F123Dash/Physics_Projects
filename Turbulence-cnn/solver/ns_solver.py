import numpy as np
import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .pressure_poisson import solve_pressure_poisson, apply_pressure_bc, gradient_p
    from .generate_data    import run_simulation, plot_results
except:
    from pressure_poisson import solve_pressure_poisson, apply_pressure_bc, gradient_p
    from generate_data    import run_simulation, plot_results

PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SNAPSHOT_ROOT = os.path.join(PROJECT_ROOT, "snapshots")
IMAGE_ROOT    = os.path.join(PROJECT_ROOT, "img_out")


def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seed",type=int, default=None)
    p.add_argument("--validate_ghia", action="store_true",help="Run N=128 Ghia validation")
    return p.parse_args()

def init_seed(seed):
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**32 - 1))
    np.random.seed(seed)
    print(f"Seed: {seed}")
    return seed
class NSconfig:
    def __init__(self, Re: int = 1000, N: int = 64):
        self.N  = N
        self.L  = 1.0
        self.dx = self.L / N
        self.dy = self.L / N
        self.Re  = Re
        self.rho = 1.0
        self.U   = 1.0
        self.nu  = self.U * self.L / Re
        self.dt          = 1e-3
        self.t_end       = max(80.0, Re / 8.0)
        self.t_start_save = self.t_end * 0.50
        if Re > 1500:self.t_start_save = self.t_end * 0.40
        if Re <= 400:T_save = 2.0
        elif Re <= 1000:T_save = 1.0
        else:T_save = 0.5
        self.save_every = max(10, int(T_save / 1e-3))
        self._T_save    = T_save   # stored for diagnostic printing
        self.n_poisson   = 400
        self.sor_omega   = 1.7
        self.poisson_tol = 1e-6
        print(f"Config: Re={Re}, N={N}x{N}, nu={self.nu:.5f}, "f"t_end={self.t_end:.1f}, t_start_save={self.t_start_save:.1f}, "
            f"save_every={self.save_every} (~{T_save:.1f}s intervals)")

def laplacian(f, dx, dy):
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = (
        (f[2:,   1:-1] - 2.0*f[1:-1, 1:-1] + f[:-2,  1:-1]) / dx**2
      + (f[1:-1, 2:  ] - 2.0*f[1:-1, 1:-1] + f[1:-1, :-2 ]) / dy**2
    )
    return lap

def divergence(u, v, dx, dy):
    div = np.zeros_like(u)
    div[1:-1, 1:-1] = (
        (u[2:,   1:-1] - u[:-2,  1:-1]) / (2.0*dx)
      + (v[1:-1, 2:  ] - v[1:-1, :-2 ]) / (2.0*dy)
    )
    return div

def advect_upwind(u, v, phi, dx, dy):
    adv = np.zeros_like(phi)
    i  = slice(1,-1); ip = slice(2,None); im = slice(None,-2)
    j  = slice(1,-1); jp = slice(2,None); jm = slice(None,-2)
    u_c = u[i,j]; v_c = v[i,j]
    dphi_dx = np.where(u_c>0, (phi[i,j]-phi[im,j])/dx,  (phi[ip,j]-phi[i,j])/dx)
    dphi_dy = np.where(v_c>0, (phi[i,j]-phi[i,jm])/dy,  (phi[i,jp]-phi[i,j])/dy)
    adv[i,j] = u_c*dphi_dx + v_c*dphi_dy
    return adv


def apply_bc(u, v, U_lid):
    u[:, -1] = U_lid;  v[:, -1] = 0.0
    u[:,  0] = 0.0;    v[:,  0] = 0.0
    u[0,  :] = 0.0;    v[0,  :] = 0.0
    u[-1, :] = 0.0;    v[-1, :] = 0.0
    return u, v

def step(u, v, p, cfg):
    dt=cfg.dt; dx=cfg.dx; dy=cfg.dy; nu=cfg.nu; rho=cfg.rho
    u_star = u + dt*(-advect_upwind(u,v,u,dx,dy) + nu*laplacian(u,dx,dy))
    v_star = v + dt*(-advect_upwind(u,v,v,dx,dy) + nu*laplacian(v,dx,dy))
    u_star, v_star = apply_bc(u_star, v_star, cfg.U)
    b = (rho/dt)*divergence(u_star, v_star, dx, dy)
    b -= b.mean()
    p = solve_pressure_poisson(p, b, dx, dy, cfg.n_poisson,cfg.sor_omega, tol=cfg.poisson_tol)
    p -= p.mean()
    dpdx, dpdy = gradient_p(p, dx, dy)
    u_new = u_star - (dt/rho)*dpdx
    v_new = v_star - (dt/rho)*dpdy
    u_new, v_new = apply_bc(u_new, v_new, cfg.U)
    u_new = np.clip(u_new, -10.0, 10.0)
    v_new = np.clip(v_new, -10.0, 10.0)
    return u_new, v_new, p

def stable_dt(u, v, dx, dy, nu, safety=0.2):
    max_vel = max(np.max(np.abs(u)), np.max(np.abs(v)), 1.0)
    h       = min(dx, dy)
    return min(safety*h/max_vel, safety*h**2/(4.0*nu), 1e-3)

def vorticity(u, v, dx, dy):
    omega = np.zeros_like(u)
    omega[1:-1, 1:-1] = (
        (v[2:,   1:-1] - v[:-2,  1:-1]) / (2.0*dx)
      - (u[1:-1, 2:  ] - u[1:-1, :-2 ]) / (2.0*dy))
    return omega

def energy_spectrum(u, v):
    N       = u.shape[0]
    u_hat   = np.fft.fft2(u)
    v_hat   = np.fft.fft2(v)
    k_cut   = N // 3
    kx      = np.fft.fftfreq(N) * N
    ky      = np.fft.fftfreq(N) * N
    KX, KY  = np.meshgrid(kx, ky, indexing="ij")
    alias   = (np.abs(KX) > k_cut) | (np.abs(KY) > k_cut)
    u_hat[alias] = 0.0; v_hat[alias] = 0.0
    energy  = 0.5*(np.abs(u_hat)**2 + np.abs(v_hat)**2) / N**4
    K       = np.sqrt(KX**2 + KY**2)
    k_bins  = np.arange(1, k_cut+1)
    E_k     = np.array([energy[(K>=k-0.5)&(K<k+0.5)].sum() for k in k_bins])
    return k_bins, E_k

def is_converged(u, u_prev, v, v_prev, tol=1e-6):
    return max(np.max(np.abs(u-u_prev)), np.max(np.abs(v-v_prev))) < tol

def diagnostics(u, v, p, t, step_n, cfg):
    div_max = np.max(np.abs(divergence(u, v, cfg.dx, cfg.dy)))
    ke      = 0.5*np.mean(u**2 + v**2)
    print(f"  t={t:.3f}  step={step_n:5d}  KE={ke:.4f}  "
          f"|del·u|_max={div_max:.2e}  p=[{p.min():.3f},{p.max():.3f}]")

_GHIA_U = np.array([
    [1.0000, 1.00000, 1.00000, 1.00000],
    [0.9766, 0.84123, 0.75837, 0.65928],
    [0.9688, 0.78871, 0.68439, 0.57492],
    [0.9609, 0.73722, 0.61756, 0.51117],
    [0.9531, 0.68717, 0.55892, 0.46604],
    [0.8516, 0.23151, 0.29093, 0.33304],
    [0.7344, 0.00332, 0.16256, 0.18719],
    [0.6172,-0.13641, 0.02135, 0.05702],
    [0.5000,-0.20581,-0.11477,-0.06080],
    [0.4531,-0.21090,-0.17119,-0.10648],
    [0.2813,-0.15662,-0.32726,-0.27805],
    [0.1719,-0.10372,-0.24299,-0.38289],
    [0.1016,-0.06434,-0.14612,-0.29730],
    [0.0703,-0.04775,-0.10338,-0.22220],
    [0.0625,-0.04192,-0.09266,-0.20196],
    [0.0547,-0.03717,-0.08186,-0.18109],
    [0.0000, 0.00000, 0.00000, 0.00000],
])
_GHIA_V = np.array([
    [1.0000, 0.00000, 0.00000, 0.00000],
    [0.9688,-0.05906,-0.12146,-0.21388],
    [0.9609,-0.07391,-0.15663,-0.27669],
    [0.9531,-0.08864,-0.19254,-0.33714],
    [0.9453,-0.10313,-0.22847,-0.39188],
    [0.9063,-0.16914,-0.23827,-0.51550],
    [0.8594,-0.22445,-0.44993,-0.42665],
    [0.8047,-0.24533,-0.38598,-0.31966],
    [0.5000, 0.05454, 0.05186, 0.02526],
    [0.2344, 0.17527, 0.30174, 0.32235],
    [0.2266, 0.17507, 0.30203, 0.33075],
    [0.1563, 0.16077, 0.28124, 0.37095],
    [0.0938, 0.12317, 0.22965, 0.32627],
    [0.0781, 0.10890, 0.20920, 0.30353],
    [0.0703, 0.10091, 0.19713, 0.29012],
    [0.0625, 0.09233, 0.18360, 0.27485],
    [0.0000, 0.00000, 0.00000, 0.00000],
])
_GHIA_COL = {100:1, 400:2, 1000:3}

def ghia_u(Re): col=_GHIA_COL.get(Re); return (None,None) if col is None else (_GHIA_U[:,0],_GHIA_U[:,col])
def ghia_v(Re): col=_GHIA_COL.get(Re); return (None,None) if col is None else (_GHIA_V[:,0],_GHIA_V[:,col])

def plot_energy_spectrum(u, v, cfg):
    k, Ek = energy_spectrum(u, v)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.loglog(k, Ek, "b-", lw=2, label=f"Simulation Re={cfg.Re}")
    k_anchor = 5
    idx = np.argmin(np.abs(k - k_anchor))
    k_ref = np.array([float(k_anchor), float(k[-1])])
    E_ref = Ek[idx] * (k_ref / k[idx])**(-5.0/3.0)
    ax.loglog(k_ref, E_ref, "r--", lw=1.5, label=r"$k^{-5/3}$ Kolmogorov")
    ax.set_xlabel("Wavenumber k", fontsize=12)
    ax.set_ylabel("E(k)",         fontsize=12)
    ax.set_title(f"Turbulent energy spectrum  Re={cfg.Re}", fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, which="both", alpha=0.3)
    os.makedirs(IMAGE_ROOT, exist_ok=True)
    fname = os.path.join(IMAGE_ROOT, f"spectrum_Re{cfg.Re}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    pdf_path = os.path.splitext(fname)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

def plot_centerline(u, v, cfg):
    N   = cfg.N; mid = N//2
    y_vals = np.linspace(0.0, 1.0, N)
    x_vals = np.linspace(0.0, 1.0, N)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Centreline velocity profiles  Re={cfg.Re}  (Ghia et al. 1982)",fontsize=12)
    ax1.plot(u[mid,:], y_vals, "b-", lw=2, label="Simulation")
    gy, gu = ghia_u(cfg.Re)
    if gy is not None: ax1.plot(gu, gy, "ko", ms=5, label="Ghia et al. 1982")
    ax1.axvline(0, color="k", lw=0.5, ls="--")
    ax1.set_xlabel("u-velocity"); ax1.set_ylabel("y")
    ax1.set_title("u  along  x = 0.5"); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    ax2.plot(x_vals, v[:,mid], "r-", lw=2, label="Simulation")
    gx, gv = ghia_v(cfg.Re)
    if gx is not None: ax2.plot(gx, gv, "ko", ms=5, label="Ghia et al. 1982")
    ax2.axhline(0, color="k", lw=0.5, ls="--")
    ax2.set_xlabel("x"); ax2.set_ylabel("v-velocity")
    ax2.set_title("v  along  y = 0.5"); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(IMAGE_ROOT, exist_ok=True)
    fname = os.path.join(IMAGE_ROOT, f"centerline_Re{cfg.Re}_N{cfg.N}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    pdf_path = os.path.splitext(fname)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

def run_ghia_validation():
    print("GHIA VALIDATION MODE — N=128")
    for Re in [100, 1000]:
        cfg = NSconfig(Re=Re, N=128)
        u   = np.zeros((128, 128))
        v   = np.zeros((128, 128))
        p   = np.zeros((128, 128))
        u, v = apply_bc(u, v, cfg.U)
        t = 0.0; step_n = 0
        print(f"\nRunning Re={Re} N=128...")
        while t < cfg.t_end:
            u_prev = u.copy(); v_prev = v.copy()
            cfg.dt = stable_dt(u, v, cfg.dx, cfg.dy, cfg.nu)
            u, v, p = step(u, v, p, cfg)
            t += cfg.dt; step_n += 1
            if step_n % 10000 == 0:
                diagnostics(u, v, p, t, step_n, cfg)
            if step_n > 2000 and is_converged(u, u_prev, v, v_prev, tol=1e-7):
                print(f"  Converged at t={t:.3f}")
                break
        plot_centerline(u, v, cfg)
        N = cfg.N; mid = N//2
        gy, gu_ghia = ghia_u(Re)
        if gy is not None:
            y_sim = np.linspace(0, 1, N)
            u_sim_at_ghia = np.interp(gy, y_sim, u[mid, :])
            errors = np.abs(u_sim_at_ghia - gu_ghia)
            print(f"  Re={Re} u-profile Ghia errors:")
            print(f"    max  = {errors.max():.4f}")
            print(f"    mean = {errors.mean():.4f}")
            print(f"    worst point: y={gy[errors.argmax()]:.4f}  "
                  f"sim={u_sim_at_ghia[errors.argmax()]:.4f}  "f"ghia={gu_ghia[errors.argmax()]:.4f}")

if __name__ == "__main__":
    args = get_args()
    init_seed(args.seed)
    if args.validate_ghia:
        run_ghia_validation()
    else:
        RE_VALUES = [100, 200, 400, 600, 800, 1000, 1200, 1500,1700, 2000, 2500, 3200]
        results = {}
        for Re in RE_VALUES:
            u, v, p, snaps = run_simulation(Re=Re, N=64,save_dir=SNAPSHOT_ROOT,NSconfig_class=NSconfig,
                                            apply_bc_fn=apply_bc,step_fn=step,stable_dt_fn=stable_dt,
                                            diagnostics_fn=diagnostics,is_converged_fn=is_converged,)
            results[Re] = (u, v, p)
            cfg = NSconfig(Re=Re, N=64)
            plot_results(u, v, p, cfg, vorticity_fn=vorticity, title="(t=final)")
            plot_centerline(u, v, cfg)
        Re_turb = max(RE_VALUES)
        u_t, v_t, _ = results[Re_turb]
        cfg_turb = NSconfig(Re=Re_turb, N=64)
        plot_energy_spectrum(u_t, v_t, cfg_turb)
        print(f"\nSnapshots: {SNAPSHOT_ROOT}/Re_*/")
        print(f"Images:    {IMAGE_ROOT}/")