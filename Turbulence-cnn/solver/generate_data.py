import numpy as np
import os
import sys
import glob
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

try:
    from . import ns_solver as ns
except ImportError:
    import ns_solver as ns

config = ns.config
SNAPSHOT_ROOT = config.SNAPSHOT_ROOT
IMAGE_ROOT    = config.IMAGE_ROOT


def _prepare_dir(re_dir: str, overwrite: bool) -> None:
    os.makedirs(re_dir, exist_ok=True)
    old = glob.glob(os.path.join(re_dir, "snap_*.npy")) + glob.glob(os.path.join(re_dir, "meta.json"))
    if not old:
        return
    if not overwrite:
        raise FileExistsError(f"{re_dir} already holds {len(old)} file(s) from an earlier run. "
                              f"Pass --overwrite to replace them, or choose a new --save_dir.")
    for f in old:
        os.remove(f)

def _rel_change(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))

def run_simulation(Re: int = 1000, N: int = 64, refine: int = 2, save_dir: str = None,
                   overwrite: bool = False, t_start_save: float = None, save_dt: float = None,
                   min_rel_change: float = None):
    N_sim = (N - 1) * refine + 1
    cfg = ns.NSconfig(Re=Re, N=N_sim)
    if t_start_save   is not None: cfg.t_start_save   = t_start_save
    if save_dt        is not None: cfg.save_dt        = save_dt
    if min_rel_change is not None: cfg.min_rel_change = min_rel_change
    if save_dir is None: save_dir = SNAPSHOT_ROOT
    re_dir = os.path.join(save_dir, f"Re_{Re}")
    _prepare_dir(re_dir, overwrite)
    state = dict(times=[], last_uv=None, next_t=cfg.t_start_save)

    def save(t, u, v, p):
        omega = ns.vorticity(u, v, cfg.dx, cfg.dy)
        snap  = np.stack([u, v, p, omega], axis=0)[:, ::refine, ::refine].astype(np.float32)
        np.save(os.path.join(re_dir, f"snap_{len(state['times']):05d}.npy"), snap)
        state["times"].append(float(t))
        state["last_uv"] = snap[:2]

    def on_step(t, u, v, p):
        if t < state["next_t"]:
            return
        state["next_t"] = t + cfg.save_dt
        uv = np.stack([u, v])[:, ::refine, ::refine]
        if state["last_uv"] is None or _rel_change(uv, state["last_uv"]) >= cfg.min_rel_change:
            save(t, u, v, p)

    print(f"\nStarting simulation: Re={Re}, solver grid={N_sim}x{N_sim}, saved grid={N}x{N}", flush=True)
    u, v, p, t, converged = ns.run(cfg, on_step)
    final_uv = np.stack([u, v])[:, ::refine, ::refine]
    if state["last_uv"] is None or _rel_change(final_uv, state["last_uv"]) > 1e-6:
        save(t, u, v, p)
    meta = dict(Re=Re, N=N, N_sim=N_sim, refine=refine, dx=1.0/(N-1), nu=cfg.nu,
                advection="MUSCL van Leer", poisson="DCT exact",
                t_start_save=cfg.t_start_save, save_dt=cfg.save_dt,
                min_rel_change=cfg.min_rel_change, conv_tol=cfg.conv_tol,
                converged=bool(converged), t_final=float(t), times=state["times"])
    with open(os.path.join(re_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    status = f"converged at t={t:.2f}" if converged else f"NOT converged by t_end={cfg.t_end:.1f}"
    print(f"Done — Re={Re}: {status}, saved {len(state['times'])} snapshots to {re_dir}/", flush=True)
    return u[::refine, ::refine], v[::refine, ::refine], p[::refine, ::refine], len(state["times"])

def plot_results(u:np.ndarray,v:np.ndarray,p:np.ndarray,
                 cfg:object,title:str = "",):
    omega = ns.vorticity(u, v, cfg.dx, cfg.dy)
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

def _get_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate lid-driven cavity snapshots for CNN training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--re_values",type=int,nargs="+",default=config.RE_VALUES,help="Reynolds numbers to simulate")
    p.add_argument("--N",type=int,default=64,help="Saved grid size (NxN nodes, walls included)")
    p.add_argument("--refine",type=int,default=2,help="Solver grid is (N-1)*refine+1 nodes per side")
    p.add_argument("--save_dir",type=str,default=SNAPSHOT_ROOT)
    p.add_argument("--overwrite",action="store_true",help="Delete existing snapshots in each Re_* folder first")
    p.add_argument("--workers",type=int,default=1,help="Re values simulated in parallel")
    p.add_argument("--t_start_save",type=float,default=None,help="Override NSconfig.t_start_save")
    p.add_argument("--save_dt",type=float,default=None,help="Override NSconfig.save_dt")
    p.add_argument("--min_rel_change",type=float,default=None,help="Override NSconfig.min_rel_change")
    p.add_argument("--no_plots",action="store_true")
    return p.parse_args(argv)

def main(argv=None):
    args = _get_args(argv)
    re_values = sorted(set(args.re_values), reverse=True)   # slowest (highest Re) first
    print(f"\nRe sweep ({len(re_values)} values): {sorted(re_values)}  ->  {args.save_dir}")
    kw = dict(N=args.N, refine=args.refine, save_dir=args.save_dir, overwrite=args.overwrite,
              t_start_save=args.t_start_save, save_dt=args.save_dt, min_rel_change=args.min_rel_change)
    for Re in re_values:   # fail before any simulation starts, not hours in
        _prepare_dir(os.path.join(args.save_dir, f"Re_{Re}"), args.overwrite)
    kw["overwrite"] = True   # folders were just checked or cleared
    results = {}
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_simulation, Re=Re, **kw): Re for Re in re_values}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
    else:
        for Re in re_values:
            results[Re] = run_simulation(Re=Re, **kw)
    print(f"\nSnapshots per Re:")
    for Re in sorted(results):
        print(f"  Re={Re:5d}  n={results[Re][3]}")
    if not args.no_plots:
        for Re in sorted(results):
            u, v, p, _ = results[Re]
            cfg = ns.NSconfig(Re=Re, N=args.N, verbose=False)
            plot_results(u, v, p, cfg, title="(t=final)")
            ns.plot_centerline(u, v, cfg)
    print(f"\nAll Re done.  Snapshots in {args.save_dir}/Re_*/")

if __name__ == "__main__":
    main()
