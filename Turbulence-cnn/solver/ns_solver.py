import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pressure_poisson import solve_pressure_poisson, apply_pressure_bc, gradient_p
from generate_data import run_simulation, plot_results

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SNAPSHOT_ROOT = os.path.join(PROJECT_ROOT, "snapshots")
IMAGE_ROOT = os.path.join(PROJECT_ROOT, "img_out")


def get_args():
    p = argparse.ArgumentParser(description="Run NS solver and save snapshots")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def init_seed(seed):
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**32 - 1))
    np.random.seed(seed)
    print(f"Seed: {seed}")
    return seed

class NSconfig:
    def __init__(self,Re=1000,N=64):
        #Grid
        self.N = N                                      # grid size
        self.L = 1.0                                    # domain length
        self.dx = self.L/N                              # spacing in x
        self.dy = self.L/N                              # spacing in y
        
        #physics
        self.Re = Re                                    # Reynold's Number
        self.rho = 1.0                                  # Density
        self.U = 1.0                                    # Vleocity
        self.nu  = self.U * self.L / Re                 # kinematic viscosity = UL/Re
 
        # Time
        self.dt       = 1e-3#0.5 * self.dx / self.U          # initial CFL-safe timestep
        self.t_end    = max(80.0, Re / 8.0)             # longer run for developed flow
        self.t_start_save = self.t_end * 0.30           # skip transient
        self.save_every = 10                            # save snapshot every N steps
        self.n_poisson = 400                           # pressure iterations each step
        self.sor_omega = 1.7                           # SOR relaxation for RBGS
        self.poisson_tol = 1e-6
 
        print(f"Config: Re={Re}, N={N}x{N}, nu={self.nu:.5f}, "
              f"t_end={self.t_end:.1f}, save after t={self.t_start_save:.1f}")

def laplacian(f,dx,dy):
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = (
        (f[2:,   1:-1] - 2*f[1:-1, 1:-1] + f[:-2,  1:-1]) / dx**2 +
        (f[1:-1, 2:  ] - 2*f[1:-1, 1:-1] + f[1:-1, :-2 ]) / dy**2
    )
    return lap

def divergence(u, v, dx, dy):
    div = np.zeros_like(u)
    div[1:-1, 1:-1] = (
        (u[2:,   1:-1] - u[:-2,  1:-1]) / (2*dx) +
        (v[1:-1, 2:  ] - v[1:-1, :-2 ]) / (2*dy)
    )
    return div

def advect_upwind(u, v, phi, dx, dy):
    adv = np.zeros_like(phi)
    i = slice(1, -1)
    ip = slice(2, None)
    im = slice(None, -2)
    j = slice(1, -1)
    jp = slice(2, None)
    jm = slice(None, -2)

    u_c = u[i, j]
    v_c = v[i, j]

    # x-direction
    dphi_dx = np.where(
        u_c > 0,
        (phi[i, j] - phi[im, j]) / dx,
        (phi[ip, j] - phi[i, j]) / dx,
    )

    # y-direction
    dphi_dy = np.where(
        v_c > 0,
        (phi[i, j] - phi[i, jm]) / dy,
        (phi[i, jp] - phi[i, j]) / dy,
    )

    #trace_val = u_c * dphi_dx + v_c * dphi_dy
    #print(
    #    "TRACE:",
    #    "max(u)=", np.max(np.abs(u_c)),
    #    "max(v)=", np.max(np.abs(v_c)),
    #    "max(dphi_dx)=", np.max(np.abs(dphi_dx)),
    #    "max(dphi_dy)=", np.max(np.abs(dphi_dy)),
    #    "max(adv)=", np.max(np.abs(trace_val)),
    #)
    adv[i, j] = u_c * dphi_dx + v_c * dphi_dy#trace_val
    return adv




def apply_bc(u, v, U_lid):
    # Top lid (moving)
    u[:, -1] = U_lid
    v[:, -1] = 0.0
 
    # Bottom wall
    u[:, 0] = 0.0
    v[:, 0] = 0.0
 
    # Left wall
    u[0, :] = 0.0
    v[0, :] = 0.0
 
    # Right wall
    u[-1, :] = 0.0
    v[-1, :] = 0.0
 
    return u, v

def step(u, v, p, cfg):
    dt = cfg.dt
    dx = cfg.dx
    dy = cfg.dy
    nu = cfg.nu
    rho = cfg.rho
 
    # u* = u^n + dt * ( -advection + viscous diffusion )
    adv_u = advect_upwind(u, v, u, dx, dy)
    adv_v = advect_upwind(u, v, v, dx, dy)
    lap_u = laplacian(u, dx, dy)
    lap_v = laplacian(v, dx, dy)
 
    u_star = u + dt * (-adv_u + nu * lap_u)
    v_star = v + dt * (-adv_v + nu * lap_v)
 
    # Re-apply BCs to u_star (walls must stay no-slip)
    u_star, v_star = apply_bc(u_star, v_star, cfg.U)
 
    # RHS: b = (rho/dt) * ∇·u*
    div_u_star = divergence(u_star, v_star, dx, dy)
    b = (rho / dt) * div_u_star
    b -= b.mean()

    p = solve_pressure_poisson(
        p, b, dx, dy,
        cfg.n_poisson, cfg.sor_omega,
        tol=getattr(cfg, "poisson_tol", 1e-6)
    )

    # Prevent pressure drift
    p -= p.mean()
 
    # u^{n+1} = u* - (dt/rho) * ∇p
    dpdx, dpdy = gradient_p(p, dx, dy)
    u_new = u_star - (dt / rho) * dpdx
    v_new = v_star - (dt / rho) * dpdy
 
    # Final BCs
    u_new, v_new = apply_bc(u_new, v_new, cfg.U)
 
    u_new = np.clip(u_new, -10.0, 10.0)
    v_new = np.clip(v_new, -10.0, 10.0)
    return u_new, v_new, p
 
def stable_dt(u, v, dx, dy, nu, safety=0.2):

    max_vel = max(
        np.max(np.abs(u)),
        np.max(np.abs(v)),
        1.0
    )

    dt_conv = safety * min(dx, dy) / max_vel

    dt_visc = safety * min(dx, dy)**2 / (4.0 * nu)

    dt = min(dt_conv, dt_visc)

    # absolute hard cap
    dt = min(dt, 1e-3)

    return dt
 
def diagnostics(u, v, p, t, step_n, cfg):
    div_max = np.max(np.abs(divergence(u, v, cfg.dx, cfg.dy)))
    ke = 0.5 * np.mean(u**2 + v**2)
    print(f"  t={t:.3f}  step={step_n:5d}  "
          f"KE={ke:.4f}  |∇·u|_max={div_max:.2e}  "
          f"p_range=[{p.min():.3f},{p.max():.3f}]")
 
 
def vorticity(u, v, dx, dy):
    omega = np.zeros_like(u)
    omega[1:-1, 1:-1] = (
        (v[2:,   1:-1] - v[:-2,  1:-1]) / (2*dx) -
        (u[1:-1, 2:  ] - u[1:-1, :-2 ]) / (2*dy)
    )
    return omega
 
def energy_spectrum(u, v):
    N = u.shape[0]
    # 2D FFT of velocity components
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
 
    # Energy in Fourier space
    energy = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / N**2
 
    # Wavenumber array
    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
 
    # Bin energy into annular rings in k-space
    k_max = N // 2
    k_bins = np.arange(1, k_max)
    E_k = np.zeros(len(k_bins))
    for i, k in enumerate(k_bins):
        mask = (K >= k - 0.5) & (K < k + 0.5)
        E_k[i] = np.sum(energy[mask])
 
    return k_bins, E_k


def is_converged(u, u_prev, v, v_prev, tol=1e-6):
    du = np.max(np.abs(u - u_prev))
    dv = np.max(np.abs(v - v_prev))
    return max(du, dv) < tol
    print(f"Saved figure: {fname}")
    plt.close(fig)
 
 
def plot_energy_spectrum(u, v, cfg):
    k, Ek = energy_spectrum(u, v)
 
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(k, Ek, 'b-', linewidth=2, label=f'Simulation Re={cfg.Re}')
 
    # Kolmogorov -5/3 reference line
    k_ref = np.array([2, cfg.N//4])
    # Scale reference to pass through the simulation data
    idx_ref = 5
    E_ref = Ek[idx_ref] * (k_ref / k[idx_ref])**(-5/3)
    ax.loglog(k_ref, E_ref, 'r--', linewidth=1.5,
              label=r'$k^{-5/3}$ Kolmogorov')
 
    ax.set_xlabel("Wavenumber k", fontsize=12)
    ax.set_ylabel("E(k)", fontsize=12)
    ax.set_title(f"Turbulent Energy Spectrum  Re={cfg.Re}", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
 
    os.makedirs(IMAGE_ROOT, exist_ok=True)
    fname = os.path.join(IMAGE_ROOT, f"spectrum_Re{cfg.Re}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved energy spectrum: {fname}")
    plt.close(fig)
 
 
def plot_centerline(u, v, cfg):
    N = cfg.N
    mid = N // 2
    y_vals = np.linspace(0, 1, N)
    x_vals = np.linspace(0, 1, N)
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Centerline velocity profiles  Re={cfg.Re}", fontsize=12)
 
    ax1.plot(u[mid, :], y_vals, 'b-', linewidth=2, label='Simulation')
    ax1.axvline(0, color='k', linewidth=0.5, linestyle='--')
    ax1.set_xlabel("u-velocity")
    ax1.set_ylabel("y")
    ax1.set_title("u along x = 0.5 (vertical centreline)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
 
    ax2.plot(x_vals, v[:, mid], 'r-', linewidth=2, label='Simulation')
    ax2.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax2.set_xlabel("x")
    ax2.set_ylabel("v-velocity")
    ax2.set_title("v along y = 0.5 (horizontal centreline)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
 
    plt.tight_layout()
    os.makedirs(IMAGE_ROOT, exist_ok=True)
    fname = os.path.join(IMAGE_ROOT, f"centerline_Re{cfg.Re}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Saved centerline profiles: {fname}")
    plt.close(fig)
 
if __name__ == "__main__":
    args = get_args()
    init_seed(args.seed)

    RE_VALUES = [100, 400, 1000]
 
    results = {}
    for Re in RE_VALUES:
        u, v, p, snaps = run_simulation(
            Re=Re,
            N=64,
            save_dir=SNAPSHOT_ROOT,
            NSconfig_class=NSconfig,
            apply_bc_fn=apply_bc,
            step_fn=step,
            stable_dt_fn=stable_dt,
            diagnostics_fn=diagnostics,
            is_converged_fn=is_converged,
        )
        results[Re] = (u, v, p)
 
        cfg = NSconfig(Re=Re, N=64)
        plot_results(u, v, p, cfg, vorticity_fn=vorticity, title=f"(t=final)")
        plot_centerline(u, v, cfg)
    Re_turb = max(RE_VALUES)
    u_t, v_t, _ = results[Re_turb]
    cfg_turb = NSconfig(Re=Re_turb, N=64)
    plot_energy_spectrum(u_t, v_t, cfg_turb)   
    print(f"Snapshot data saved in: {SNAPSHOT_ROOT}/Re_*/")
    print(f"Images saved in: {IMAGE_ROOT}/")
 

