import numpy as np
import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _solver():
    import importlib, sys
    _dir = os.path.dirname(os.path.abspath(__file__))
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
    return importlib.import_module("ns_solver")

def run_simulation(Re:int= 1000,N:int= 64,save_dir: str= None,divergence_fn:object = None,
                   apply_bc_fn:object = None,step_fn:object = None,stable_dt_fn:object = None,
                   diagnostics_fn:object = None,is_converged_fn:object = None,NSconfig_class:object = None,
                   vorticity_fn:object = None):
    s = _solver()
    NSconfig_cls  = NSconfig_class   or s.NSconfig
    _apply_bc     = apply_bc_fn      or s.apply_bc
    _step         = step_fn          or s.step
    _stable_dt    = stable_dt_fn     or s.stable_dt
    _diagnostics  = diagnostics_fn   or s.diagnostics
    _is_converged = is_converged_fn  or s.is_converged
    _vorticity    = vorticity_fn     or s.vorticity
    PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    SNAPSHOT_ROOT = os.path.join(PROJECT_ROOT, "snapshots")
    if save_dir is None:save_dir = SNAPSHOT_ROOT
    re_dir = os.path.join(save_dir, f"Re_{Re}")
    os.makedirs(re_dir, exist_ok=True)
    cfg = NSconfig_cls(Re=Re, N=N)
    u = np.zeros((N, N))
    v = np.zeros((N, N))
    p = np.zeros((N, N))
    u, v = _apply_bc(u, v, cfg.U)
    t= 0.0
    step_n = 0
    snap_n = 0
    snapshots = []
    start_save_t = getattr(cfg, "t_start_save", 0.0)
    print(f"\nStarting simulation: Re={Re}, grid={N}X{N}")
    print(f"  Running to t={cfg.t_end:.1f}, "f"saving every {cfg.save_every} steps "f"(after t={start_save_t:.2f})\n")
    while t < cfg.t_end:
        u_prev = u.copy()
        v_prev = v.copy()
        cfg.dt = _stable_dt(u, v, cfg.dx, cfg.dy, cfg.nu)
        u, v, p = _step(u, v, p, cfg)
        t+= cfg.dt
        step_n += 1
        if t >= start_save_t and step_n % cfg.save_every == 0:
            omega = _vorticity(u, v, cfg.dx, cfg.dy)
            snap  = np.stack([u, v, p, omega], axis=0).astype(np.float32)
            fname = os.path.join(re_dir, f"snap_{snap_n:04d}.npy")
            np.save(fname, snap)
            snapshots.append(snap)
            snap_n += 1
            _diagnostics(u, v, p, t, step_n, cfg)
        if step_n > 1000 and _is_converged(u, u_prev, v, v_prev, tol=1e-7):
            print(f"  Converged at t={t:.3f}, step={step_n}")
            for extra in range(150):
                cfg.dt = _stable_dt(u, v, cfg.dx, cfg.dy, cfg.nu)
                u, v, p = _step(u, v, p, cfg)
                p -= p.mean()
                omega = _vorticity(u, v, cfg.dx, cfg.dy)
                snap  = np.stack([u, v, p, omega], axis=0).astype(np.float32)
                np.save(os.path.join(re_dir, f"snap_{snap_n:05d}.npy"), snap)
                snapshots.append(snap)
                snap_n += 1
            break
    print(f"\nDone — saved {snap_n} snapshots to {re_dir}/")
    return u, v, p, snapshots

def plot_results(u:np.ndarray,v:np.ndarray,p:np.ndarray,
                 cfg:object,vorticity_fn:object = None,title:str = "",):
    _vorticity = vorticity_fn or _solver().vorticity
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    IMAGE_ROOT   = os.path.join(PROJECT_ROOT, "img_out")
    omega = _vorticity(u, v, cfg.dx, cfg.dy)
    x = np.linspace(0, cfg.L, cfg.N)
    y = np.linspace(0, cfg.L, cfg.N)
    X, Y = np.meshgrid(x, y)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle(
        f"Navier-Stokes: lid-driven cavity   Re={cfg.Re}  {title}",
        fontsize=13, fontweight="bold",
    )
    panels = [
        (axes[0, 0], u.T,     "u-velocity (m/s)",  "RdBu_r"),
        (axes[0, 1], v.T,     "v-velocity (m/s)",  "RdBu_r"),
        (axes[1, 0], p.T,     "Pressure (Pa)",     "viridis"),
        (axes[1, 1], omega.T, "Vorticity (1/s)",   "seismic"),
    ]
    for ax, field, label, cmap in panels:
        vmax = np.max(np.abs(field)) + 1e-8
        im   = ax.contourf(
            X, Y, field, levels=32, cmap=cmap,
            vmin=-vmax if cmap in ("RdBu_r", "seismic") else None,
            vmax=vmax,
        )
        if "Vorticity" in label:
            ax.streamplot(x, y, u.T, v.T, color="k",
                          linewidth=0.5, density=1.2, arrowsize=0.8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
    plt.tight_layout()
    os.makedirs(IMAGE_ROOT, exist_ok=True)
    fname = os.path.join(IMAGE_ROOT, f"cavity_Re{cfg.Re}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    pdf_path = os.path.splitext(fname)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

def _get_args():
    p = argparse.ArgumentParser(
        description="Generate NS snapshots for CNN training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--re_min",type=int,default=100,help="Lowest Reynolds number")
    p.add_argument("--re_max",type=int,default=3200,help="Highest Reynolds number")
    p.add_argument("--n_re",type=int,default=12,help="Number of Re values (log-spaced between re_min and re_max)")
    p.add_argument("--N",type=int,default=64,help="Grid size (NxN)")
    p.add_argument("--save_dirt",type=str,default=None)
    p.add_argument("--seed",type=int,default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = _get_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Seed: {args.seed}")
    re_values = np.unique(
        np.round(
            np.logspace(
                np.log10(args.re_min),
                np.log10(args.re_max),
                args.n_re,
            )
        ).astype(int)
    ).tolist()
    print(f"\nRe sweep ({len(re_values)} values): {re_values}")
    s = _solver()
    PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    IMAGE_ROOT    = os.path.join(PROJECT_ROOT, "img_out")
    for Re in re_values:
        u, v, p, snaps = run_simulation(Re=Re, N=args.N, save_dir=args.save_dir)
        cfg = s.NSconfig(Re=Re, N=args.N)
        plot_results(u, v, p, cfg, vorticity_fn=s.vorticity, title="(t=final)")
        s.plot_centerline(u, v, cfg)
    print(f"\nAll Re done.  Snapshots in {args.save_dir or '(default)'}/Re_*/")