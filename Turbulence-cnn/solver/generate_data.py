import numpy as np
import os
import matplotlib.pyplot as plt


def run_simulation(Re=1000, N=64, save_dir=None, divergence_fn=None, apply_bc_fn=None, 
                   step_fn=None, stable_dt_fn=None, diagnostics_fn=None, is_converged_fn=None,
                   NSconfig_class=None):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    SNAPSHOT_ROOT = os.path.join(PROJECT_ROOT, "snapshots")
    if save_dir is None:
        save_dir = SNAPSHOT_ROOT
    cfg = NSconfig_class(Re=Re, N=N)
    os.makedirs(os.path.join(save_dir, f"Re_{Re}"), exist_ok=True)
    u = np.zeros((N, N))
    v = np.zeros((N, N))
    p = np.zeros((N, N))
    u, v = apply_bc_fn(u, v, cfg.U)
    t = 0.0
    step_n = 0
    snap_n = 0
    snapshots = []
    start_save_t = getattr(cfg, "t_start_save", 0.0)
    print(f"\nStarting simulation: Re={Re}, grid={N}x{N}")
    print(f"Running to t={cfg.t_end}, saving every {cfg.save_every} steps")
    if start_save_t > 0:
        print(f"Saving snapshots only after t={start_save_t:.2f}\n")
    else:
        print()
    while t < cfg.t_end:
        u_prev = u.copy()
        v_prev = v.copy()
        cfg.dt = stable_dt_fn(u, v, cfg.dx, cfg.dy, cfg.nu)
        u, v, p = step_fn(u, v, p, cfg)
        t += cfg.dt
        step_n += 1
        if t >= start_save_t and step_n % cfg.save_every == 0:
            snap = np.stack([u, v, p], axis=0)  # shape: (3, N, N)
            fname = os.path.join(save_dir, f"Re_{Re}", f"snap_{snap_n:04d}.npy")
            np.save(fname, snap)
            snapshots.append(snap)
            snap_n += 1
            diagnostics_fn(u, v, p, t, step_n, cfg)
        if step_n > 1000 and is_converged_fn(u, u_prev, v, v_prev, tol=1e-7):
            print(f"   Converged at t={t:.3f}, step={step_n}")
            for extra in range(150):
                cfg.dt = stable_dt_fn(
                    u,
                    v,
                    cfg.dx,
                    cfg.dy,
                    cfg.nu
                )
                u, v, p = step_fn(u, v, p, cfg)
                p -= p.mean()
                snap = np.stack([u, v, p], axis=0)
                np.save(
                    os.path.join(
                        save_dir,
                        f"Re_{Re}",
                        f"snap_{snap_n:05d}.npy"
                    ),
                    snap.astype(np.float32)
                )
                snap_n += 1
            break

    print(f"\nDone. Saved {snap_n} snapshots to {os.path.join(save_dir, f'Re_{Re}')}/")
    return u, v, p, snapshots


def plot_results(u, v, p, cfg, vorticity_fn=None, title=""):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    IMAGE_ROOT = os.path.join(PROJECT_ROOT, "img_out")
    omega = vorticity_fn(u, v, cfg.dx, cfg.dy)
    x = np.linspace(0, cfg.L, cfg.N)
    y = np.linspace(0, cfg.L, cfg.N)
    X, Y = np.meshgrid(x, y)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle(f"Navier-Stokes: Lid-Driven Cavity  Re={cfg.Re}  {title}",
                 fontsize=13, fontweight='bold')
    panels = [
        (axes[0, 0], u.T,     "u-velocity (m/s)",   "RdBu_r"),
        (axes[0, 1], v.T,     "v-velocity (m/s)",   "RdBu_r"),
        (axes[1, 0], p.T,     "Pressure (Pa)",      "viridis"),
        (axes[1, 1], omega.T, "Vorticity (1/s)",    "seismic"),
    ]
    for ax, field, label, cmap in panels:
        vmax = np.max(np.abs(field)) + 1e-8
        im = ax.contourf(X, Y, field, levels=32, cmap=cmap,
                         vmin=-vmax if cmap in ["RdBu_r","seismic"] else None,
                         vmax=vmax)
        if "Vorticity" in label:
            speed = np.sqrt(u**2 + v**2)
            ax.streamplot(x, y, u.T, v.T, color='k', linewidth=0.5,
                          density=1.2, arrowsize=0.8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
    plt.tight_layout()
    os.makedirs(IMAGE_ROOT, exist_ok=True)
    fname = os.path.join(IMAGE_ROOT, f"cavity_Re{cfg.Re}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')