import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt

try:
    from .pressure_poisson import solve_pressure_poisson, gradient_p
except ImportError:
    from pressure_poisson import solve_pressure_poisson, gradient_p
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import config

IMAGE_ROOT = config.IMAGE_ROOT


def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--validate_ghia", action="store_true", help="Run the Ghia et al. (1982) benchmark")
    p.add_argument("--N", type=int, default=129, help="Grid nodes per side for --validate_ghia (odd puts a node on x=0.5)")
    return p.parse_known_args()

class NSconfig:
    def __init__(self, Re: int = 1000, N: int = 64, verbose: bool = True):
        self.N  = N                       # nodes per side, including both walls
        self.L  = 1.0
        self.dx = self.L / (N - 1)
        self.dy = self.L / (N - 1)
        self.Re  = Re
        self.rho = 1.0
        self.U   = 1.0
        self.nu  = self.U * self.L / Re
        self.dt          = 1e-3           # upper bound; stable_dt() sets the actual step
        self.t_end       = max(80.0, Re / 8.0)
        self.conv_tol    = 1e-4           # steady when max|du/dt| < conv_tol
        self.t_min_conv  = 1.0
        # Snapshot selection, used by generate_data.run_simulation
        self.t_start_save   = 1.0
        self.save_dt        = 0.5         # minimum time between saved snapshots
        self.min_rel_change = 0.02        # minimum relative L2 change of (u, v) since the last saved snapshot
        if verbose:
            print(f"Config: Re={Re}, N={N}x{N}, dx={self.dx:.5f}, nu={self.nu:.5f}, t_end={self.t_end:.1f}")

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

def _upwind_derivative(vel, phi, h):
    """vel * dphi/dx along axis 0, MUSCL upwind with the van Leer limiter.

    Second order where phi is smooth, first-order upwind at extrema and next to
    the walls, so it is far less diffusive than plain first-order upwind.
    """
    d = phi[1:] - phi[:-1]
    a, b = d[:-1], d[1:]
    ab = a * b
    slope = np.zeros_like(phi)
    slope[1:-1] = np.where(ab > 0, 2.0*ab / np.where(ab > 0, a + b, 1.0), 0.0)
    phi_L = phi[:-1] + 0.5*slope[:-1]     # face i+1/2 reconstructed from node i
    phi_R = phi[1:]  - 0.5*slope[1:]      # face i+1/2 reconstructed from node i+1
    out = np.zeros_like(phi)
    v_c = vel[1:-1]
    out[1:-1] = v_c * np.where(v_c > 0, phi_L[1:] - phi_L[:-1], phi_R[1:] - phi_R[:-1]) / h
    return out

def advect(u, v, phi, dx, dy):
    adv = _upwind_derivative(u, phi, dx) + _upwind_derivative(v.T, phi.T, dy).T
    adv[0, :] = adv[-1, :] = adv[:, 0] = adv[:, -1] = 0.0
    return adv


def apply_bc(u, v, U_lid):
    u[:, -1] = U_lid;  v[:, -1] = 0.0
    u[:,  0] = 0.0;    v[:,  0] = 0.0
    u[0,  :] = 0.0;    v[0,  :] = 0.0
    u[-1, :] = 0.0;    v[-1, :] = 0.0
    return u, v

def step(u, v, p, cfg):
    dt=cfg.dt; dx=cfg.dx; dy=cfg.dy; nu=cfg.nu; rho=cfg.rho
    u_star = u + dt*(-advect(u,v,u,dx,dy) + nu*laplacian(u,dx,dy))
    v_star = v + dt*(-advect(u,v,v,dx,dy) + nu*laplacian(v,dx,dy))
    u_star, v_star = apply_bc(u_star, v_star, cfg.U)
    b = (rho/dt)*divergence(u_star, v_star, dx, dy)
    p = solve_pressure_poisson(b, dx, dy)
    dpdx, dpdy = gradient_p(p, dx, dy)
    u_new = u_star - (dt/rho)*dpdx
    v_new = v_star - (dt/rho)*dpdy
    u_new, v_new = apply_bc(u_new, v_new, cfg.U)
    vmax = max(np.max(np.abs(u_new)), np.max(np.abs(v_new)))
    if not np.isfinite(vmax) or vmax > 10.0*cfg.U:
        raise FloatingPointError(f"Solver blew up at Re={cfg.Re}: max|u|={vmax}")
    return u_new, v_new, p

def stable_dt(u, v, dx, dy, nu, safety=0.2, dt_max=1e-3):
    max_vel = max(np.max(np.abs(u)), np.max(np.abs(v)), 1.0)
    h       = min(dx, dy)
    return min(safety*h/max_vel, safety*h**2/(4.0*nu), dt_max)

def vorticity(u, v, dx, dy):
    # Central differences inside, second-order one-sided on the walls, where
    # the vorticity is largest (the lid).
    return np.gradient(v, dx, axis=0, edge_order=2) - np.gradient(u, dy, axis=1, edge_order=2)

def energy_spectrum(u, v):
    N       = u.shape[0]
    u_hat   = np.fft.fft2(u)
    v_hat   = np.fft.fft2(v)
    kx      = np.fft.fftfreq(N) * N
    KX, KY  = np.meshgrid(kx, kx, indexing="ij")
    energy  = 0.5*(np.abs(u_hat)**2 + np.abs(v_hat)**2) / N**4
    K       = np.sqrt(KX**2 + KY**2)
    k_bins  = np.arange(1, N//2 + 1)
    E_k     = np.array([energy[(K>=k-0.5)&(K<k+0.5)].sum() for k in k_bins])
    return k_bins, E_k

def is_converged(u, u_prev, v, v_prev, tol=1e-6, dt=None):
    change = max(np.max(np.abs(u-u_prev)), np.max(np.abs(v-v_prev)))
    if dt is not None:
        change /= dt
    return change < tol

def diagnostics(u, v, p, t, step_n, cfg):
    div_max = np.max(np.abs(divergence(u, v, cfg.dx, cfg.dy)))
    ke      = 0.5*np.mean(u**2 + v**2)
    print(f"  Re={cfg.Re}  t={t:.3f}  step={step_n:6d}  KE={ke:.4f}  "
          f"|del·u|_max={div_max:.2e}  p=[{p.min():.3f},{p.max():.3f}]", flush=True)

def run(cfg, on_step=None, report_every=50000):
    """March from rest until steady state (or t_end). on_step(t, u, v, p) is called after every step."""
    N = cfg.N
    u = np.zeros((N, N)); v = np.zeros((N, N)); p = np.zeros((N, N))
    u, v = apply_bc(u, v, cfg.U)
    t = 0.0; step_n = 0; converged = False
    while t < cfg.t_end:
        u_prev, v_prev = u, v
        cfg.dt = stable_dt(u, v, cfg.dx, cfg.dy, cfg.nu)
        u, v, p = step(u, v, p, cfg)
        t += cfg.dt; step_n += 1
        if on_step is not None:
            on_step(t, u, v, p)
        if report_every and step_n % report_every == 0:
            diagnostics(u, v, p, t, step_n, cfg)
        if t > cfg.t_min_conv and is_converged(u, u_prev, v, v_prev, tol=cfg.conv_tol, dt=cfg.dt):
            converged = True
            break
    return u, v, p, t, converged

def centerline_u(u):
    """u along the vertical centreline x = 0.5, as a function of y."""
    m = u.shape[0] // 2
    return u[m, :] if u.shape[0] % 2 else 0.5*(u[m-1, :] + u[m, :])

def centerline_v(v):
    """v along the horizontal centreline y = 0.5, as a function of x."""
    m = v.shape[1] // 2
    return v[:, m] if v.shape[1] % 2 else 0.5*(v[:, m-1] + v[:, m])

def streamfunction(u, dy):
    """phi with u = dphi/dy and phi = 0 on the bottom wall (trapezoidal rule up each column)."""
    psi = np.zeros_like(u, dtype=np.float64)
    psi[:, 1:] = np.cumsum(0.5*(u[:, 1:] + u[:, :-1]), axis=1) * dy
    return psi

def _parabolic_offset(fm, f0, fp):
    """Sub-grid offset (in cells) and value of the extremum of a parabola through 3 points."""
    den = fm - 2.0*f0 + fp
    if den == 0.0:
        return 0.0, f0
    off = 0.5*(fm - fp)/den
    return off, f0 - 0.25*(fm - fp)*off

def primary_vortex(u, dx, dy):
    """(phi_min, x_c, y_c) of the primary vortex: the minimum of phi, refined to sub-grid accuracy."""
    psi = streamfunction(u, dy)
    i, j = np.unravel_index(np.argmin(psi[1:-1, 1:-1]), (psi.shape[0]-2, psi.shape[1]-2))
    i += 1; j += 1
    ox, vx = _parabolic_offset(psi[i-1, j], psi[i, j], psi[i+1, j])
    oy, vy = _parabolic_offset(psi[i, j-1], psi[i, j], psi[i, j+1])
    psi_min = vx + vy - psi[i, j]
    return psi_min, (i + ox)*dx, (j + oy)*dy

# Ghia et al. (1982), Table V: primary vortex phi_min and centre (x, y)
_GHIA_VORTEX = {100: (-0.103423, 0.6172, 0.7344), 400: (-0.113909, 0.5547, 0.6055), 1000: (-0.117929, 0.5313, 0.5625)}

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

def plot_centerline(u, v, cfg):
    N   = cfg.N
    y_vals = np.linspace(0.0, 1.0, N)
    x_vals = np.linspace(0.0, 1.0, N)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Centreline velocity profiles  Re={cfg.Re}  (Ghia et al. 1982)",fontsize=12)
    ax1.plot(centerline_u(u), y_vals, "b-", lw=2, label="Simulation")
    gy, gu = ghia_u(cfg.Re)
    if gy is not None: ax1.plot(gu, gy, "ko", ms=5, label="Ghia et al. 1982")
    ax1.axvline(0, color="k", lw=0.5, ls="--")
    ax1.set_xlabel("u-velocity"); ax1.set_ylabel("y")
    ax1.set_title("u  along  x = 0.5"); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    ax2.plot(x_vals, centerline_v(v), "r-", lw=2, label="Simulation")
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

def ghia_errors(u, v, Re):
    """Max and mean |sim - Ghia| for the u and v centreline profiles."""
    s = np.linspace(0, 1, u.shape[0])
    gy, gu = ghia_u(Re); gx, gv = ghia_v(Re)
    eu = np.abs(np.interp(gy, s, centerline_u(u)) - gu)
    ev = np.abs(np.interp(gx, s, centerline_v(v)) - gv)
    return dict(u_max=eu.max(), u_mean=eu.mean(), v_max=ev.max(), v_mean=ev.mean())

def run_ghia_validation(N=129):
    print(f"GHIA VALIDATION MODE — N={N}")
    for Re in sorted(_GHIA_COL):
        cfg = NSconfig(Re=Re, N=N)
        u, v, p, t, converged = run(cfg)
        print(f"  Re={Re}: {'converged' if converged else 'NOT converged'} at t={t:.2f}")
        plot_centerline(u, v, cfg)
        e = ghia_errors(u, v, Re)
        print(f"  Re={Re} Ghia errors:  u max={e['u_max']:.4f} mean={e['u_mean']:.4f}   "
              f"v max={e['v_max']:.4f} mean={e['v_mean']:.4f}")
        psi_min, xc, yc = primary_vortex(u, cfg.dx, cfg.dy)
        g_psi, g_x, g_y = _GHIA_VORTEX[Re]
        print(f"  Re={Re} primary vortex:  phi_min={psi_min:.5f} (Ghia {g_psi:.5f}, {abs(psi_min/g_psi-1):.1%} off)   "
              f"centre=({xc:.4f}, {yc:.4f}) (Ghia ({g_x:.4f}, {g_y:.4f}), {np.hypot(xc-g_x, yc-g_y):.4f} away)")

if __name__ == "__main__":
    args, _ = get_args()
    if args.validate_ghia:
        run_ghia_validation(args.N)
    else:
        # Data generation lives in generate_data.py; forward all arguments to it.
        try:
            from . import generate_data
        except ImportError:
            import generate_data
        generate_data.main(sys.argv[1:])
