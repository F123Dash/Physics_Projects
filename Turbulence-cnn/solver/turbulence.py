"""2-D incompressible turbulence in a doubly periodic box [0, 2π)², pseudo-spectral."""
import os
import sys
import glob
import json
import time
import argparse
import numpy as np
import scipy.fft as sfft
from concurrent.futures import ProcessPoolExecutor, as_completed

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import config

L_BOX = 2.0 * np.pi


class Spectral:
    """Wavenumbers and transforms for an NxN periodic grid (rfft along y)."""
    def __init__(self, N: int):
        self.N  = N
        self.kx = sfft.fftfreq(N, 1.0 / N)[:, None]
        self.ky = sfft.rfftfreq(N, 1.0 / N)[None, :]
        self.k2 = self.kx**2 + self.ky**2
        self.k2_inv = np.where(self.k2 > 0, 1.0 / np.where(self.k2 > 0, self.k2, 1.0), 0.0)
        k_cut = N // 3
        self.dealias = (np.abs(self.kx) <= k_cut) & (np.abs(self.ky) <= k_cut)
    def fft(self, f):  return sfft.rfft2(f, workers=1)
    def ifft(self, f): return sfft.irfft2(f, s=(self.N, self.N), workers=1)
    def velocity_hat(self, w_hat):
        psi = w_hat * self.k2_inv
        return 1j * self.ky * psi, -1j * self.kx * psi

def nonlinear(sp: Spectral, w_hat, force_hat):
    """FFT of −u·∇ω + f, dealiased."""
    u_hat, v_hat = sp.velocity_hat(w_hat)
    u, v = sp.ifft(u_hat), sp.ifft(v_hat)
    wx, wy = sp.ifft(1j * sp.kx * w_hat), sp.ifft(1j * sp.ky * w_hat)
    return sp.dealias * (force_hat - sp.fft(u * wx + v * wy))

def pressure_hat(sp: Spectral, w_hat):
    """Pressure from ∇²p = -del_idel_j(u_i u_j) = 2(u_x v_y - u_y v_x), zero mean (rho = 1)."""
    u_hat, v_hat = sp.velocity_hat(w_hat)
    ux, uy = sp.ifft(1j*sp.kx*u_hat), sp.ifft(1j*sp.ky*u_hat)
    vx, vy = sp.ifft(1j*sp.kx*v_hat), sp.ifft(1j*sp.ky*v_hat)
    rhs_hat = sp.dealias * sp.fft(2.0 * (ux*vy - uy*vx))
    return -rhs_hat * sp.k2_inv

def energy_spectrum(w_hat, sp: Spectral):
    """Shell-summed kinetic energy E(k), k = 1 .. N/2, with SIGMA E(k) = mean(|u|²)/2."""
    N = sp.N
    weight = np.full(w_hat.shape, 2.0); weight[:, 0] = 1.0
    if N % 2 == 0: weight[:, -1] = 1.0
    e = 0.5 * weight * np.abs(w_hat)**2 * sp.k2_inv / N**4
    k = np.sqrt(sp.k2)
    kb = np.arange(1, N // 2 + 1)
    idx = np.rint(k).astype(int)
    E = np.bincount(idx.ravel(), weights=e.ravel(), minlength=N)[1:N // 2 + 1]
    return kb, E

class TurbConfig:
    def __init__(self, Re: float, kind: str, N: int, seed: int = 0):
        assert kind in ("forced", "decaying")
        self.Re, self.kind, self.N, self.seed = Re, kind, N, seed
        self.nu      = 1.0 / Re
        self.cfl     = 0.5
        self.dt_max  = 0.05
        if kind == "forced":
            self.k_f, self.F, self.drag = 4, 1.0, 0.1
            self.t_spin, self.save_dt, self.n_save = 20.0, 2.0, 40     # stationary by t ≈ 10-15
        else:
            self.k_f, self.F, self.drag = 0, 0.0, 0.0
            self.k0      = 8            # peak wavenumber of the initial energy spectrum
            self.t_spin, self.save_dt, self.n_save = 2.0, 0.5, 30      # skip the under-resolved first instants

def initial_vorticity(sp: Spectral, cfg: TurbConfig):
    # Seeded by (seed, Re, kind): runs at different Re must not share an initial field,
    # or test-Re runs would be correlated with training-Re runs.
    rng = np.random.default_rng([cfg.seed, int(round(cfg.Re)), 0 if cfg.kind == "forced" else 1])
    k = np.sqrt(sp.k2)
    if cfg.kind == "forced":
        amp = np.where((k >= 1) & (k <= 8), 1.0, 0.0); energy = 0.01
    else:
        amp = k**4 * np.exp(-2.0 * (k / cfg.k0)**2); energy = 0.5
    # Random phases, amplitude chosen so E(k) follows amp(k)/k
    w_hat = np.sqrt(amp * k) * np.exp(2j * np.pi * rng.random(sp.k2.shape)) * sp.dealias
    w_hat[0, 0] = 0.0
    w_hat = sp.fft(sp.ifft(w_hat))                       # enforce Hermitian symmetry
    _, E = energy_spectrum(w_hat, sp)
    return w_hat * np.sqrt(energy / E.sum())

def make_stepper(sp: Spectral, cfg: TurbConfig):
    lin = -cfg.nu * sp.k2 - cfg.drag
    force_hat = np.zeros_like(sp.k2, dtype=complex)
    if cfg.kind == "forced":
        y = np.arange(sp.N) * L_BOX / sp.N
        force_hat = sp.fft(np.broadcast_to(-cfg.F * cfg.k_f * np.cos(cfg.k_f * y)[None, :], (sp.N, sp.N)))
    cache = {}
    def step(w_hat, dt):
        if dt not in cache:
            cache.clear()
            cache[dt] = (np.exp(lin * dt / 2), np.exp(lin * dt))
        E2, E = cache[dt]
        k1 = nonlinear(sp, w_hat, force_hat)
        k2 = nonlinear(sp, E2 * (w_hat + 0.5*dt*k1), force_hat)
        k3 = nonlinear(sp, E2 * w_hat + 0.5*dt*k2, force_hat)
        k4 = nonlinear(sp, E * w_hat + dt*E2*k3, force_hat)
        return E * w_hat + dt/6.0 * (E*k1 + 2.0*E2*(k2 + k3) + k4)
    return step

def max_velocity(sp, w_hat):
    u_hat, v_hat = sp.velocity_hat(w_hat)
    return max(np.abs(sp.ifft(u_hat)).max(), np.abs(sp.ifft(v_hat)).max(), 1e-6)

def advance(sp, step, w_hat, cfg, duration, recheck=10):
    """Advance by exactly `duration`. The CFL limit is re-evaluated every `recheck`
    steps, and the remaining time is split into equal steps below it."""
    dx = L_BOX / sp.N
    t_left, n = duration, 0
    while t_left > 1e-12:
        if n % recheck == 0:
            dt_lim = min(cfg.cfl * dx / max_velocity(sp, w_hat), cfg.dt_max)
            dt = t_left / np.ceil(t_left / dt_lim - 1e-9)
        w_hat = step(w_hat, dt)
        t_left -= dt; n += 1
    if not np.isfinite(w_hat).all():
        raise FloatingPointError(f"turbulence solver blew up: Re={cfg.Re} {cfg.kind} seed={cfg.seed}")
    return w_hat, n

def truncate(f_hat, N: int, n: int):
    """Keep wavenumbers |k| < n/2 of an N-grid rfft2 array; return the n-grid physical field."""
    h = n // 2
    out = np.zeros((n, h + 1), dtype=complex)
    out[:h, :h] = f_hat[:h, :h]
    out[-h + 1:, :h] = f_hat[-h + 1:, :h]
    return sfft.irfft2(out, s=(n, n), workers=1) * (n / N)**2

def snapshot(sp: Spectral, w_hat, n: int):
    u_hat, v_hat = sp.velocity_hat(w_hat)
    p_hat = pressure_hat(sp, w_hat)
    return np.stack([truncate(f, sp.N, n) for f in (u_hat, v_hat, p_hat, w_hat)]).astype(np.float32)

def resolution_check(sp: Spectral, w_hat):
    """Enstrophy spectrum at the dealiasing cut-off relative to its peak (want < 1e-3)."""
    k, E = energy_spectrum(w_hat, sp)
    Z = k**2 * E
    k_cut = sp.N // 3
    return float(Z[k_cut - 1] / Z.max())

def run_turbulence(Re, kind, seed, N_dns, n_save_grid, save_dir, overwrite=False, n_save=None,
                   t_spin=None, save_dt=None):
    cfg = TurbConfig(Re, kind, N_dns, seed)
    if n_save  is not None: cfg.n_save  = n_save
    if t_spin  is not None: cfg.t_spin  = t_spin
    if save_dt is not None: cfg.save_dt = save_dt
    sp  = Spectral(N_dns)
    out = os.path.join(save_dir, kind, f"Re_{Re}", f"seed_{seed}")
    os.makedirs(out, exist_ok=True)
    old = glob.glob(os.path.join(out, "snap_*.npy")) + glob.glob(os.path.join(out, "meta.json"))
    if old and not overwrite:
        raise FileExistsError(f"{out} already holds files; pass --overwrite or choose a new --save_dir")
    for f in old: os.remove(f)
    step = make_stepper(sp, cfg)
    t0 = time.time()
    w_hat = initial_vorticity(sp, cfg)
    w_hat, steps = advance(sp, step, w_hat, cfg, cfg.t_spin)
    t = cfg.t_spin
    times, energies, res = [], [], []
    for i in range(cfg.n_save):
        if i > 0:
            w_hat, n = advance(sp, step, w_hat, cfg, cfg.save_dt); steps += n; t += cfg.save_dt
        np.save(os.path.join(out, f"snap_{i:05d}.npy"), snapshot(sp, w_hat, n_save_grid))
        _, E = energy_spectrum(w_hat, sp)
        times.append(t); energies.append(float(E.sum())); res.append(resolution_check(sp, w_hat))
    meta = dict(flow=kind, domain="periodic", L=L_BOX, Re=Re, nu=cfg.nu, N=n_save_grid, N_dns=N_dns,
                seed=seed, drag=cfg.drag, forcing=(f"-{cfg.F}*{cfg.k_f}*cos({cfg.k_f}y)" if kind == "forced" else None),
                t_spin=cfg.t_spin, save_dt=cfg.save_dt, times=times, energy=energies,
                enstrophy_tail=res, steps=steps, wall_s=time.time() - t0)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"Done — {kind:8s} Re={Re:5d} seed={seed}: {cfg.n_save} snapshots, "
          f"E={np.mean(energies):.3f}, worst enstrophy tail={max(res):.1e}, "
          f"{steps} steps, {time.time() - t0:.0f}s", flush=True)
    return kind, Re, seed, max(res)

def dns_grid(Re, kind):
    """DNS resolution per flow kind and Re (see config.TURB_DNS_GRID)."""
    for re_max, N in config.TURB_DNS_GRID[kind]:
        if Re <= re_max:
            return N
    raise ValueError(f"No DNS grid configured for Re={Re}")

def main(argv=None):
    p = argparse.ArgumentParser(description="Generate 2-D periodic turbulence snapshots",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--kinds", nargs="+", default=["forced", "decaying"], choices=["forced", "decaying"])
    p.add_argument("--re_values", type=int, nargs="+", default=config.TURB_RE_VALUES)
    p.add_argument("--seeds", type=int, nargs="+", default=config.TURB_SEEDS)
    p.add_argument("--N", type=int, default=config.TURB_FINE_SIZE, help="Saved grid size")
    p.add_argument("--N_dns", type=int, default=None, help="DNS grid (default: config.TURB_DNS_GRID by Re)")
    p.add_argument("--n_save", type=int, default=None, help="Snapshots per run (default: per kind in TurbConfig)")
    p.add_argument("--t_spin", type=float, default=None, help="Time before the first snapshot (default: per kind)")
    p.add_argument("--save_dt", type=float, default=None, help="Time between snapshots (default: per kind)")
    p.add_argument("--save_dir", default=config.TURB_ROOT)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args(argv)
    jobs = [(Re, kind, seed) for Re in args.re_values for kind in args.kinds for seed in args.seeds]
    cost = lambda j: (args.N_dns or dns_grid(j[0], j[1]))**3 * (98 if j[1] == "forced" else 16)
    jobs.sort(key=cost, reverse=True)   # longest runs first, so the pool stays busy
    if not args.overwrite:   # fail before any simulation starts, not hours in
        clash = [d for Re, kind, seed in jobs
                 if glob.glob(os.path.join(d := os.path.join(args.save_dir, kind, f"Re_{Re}", f"seed_{seed}"), "snap_*.npy"))]
        if clash:
            sys.exit(f"{len(clash)} output folders already hold snapshots (e.g. {clash[0]}). "
                     f"Pass --overwrite or choose a new --save_dir.")
    print(f"{len(jobs)} runs: kinds={args.kinds} Re={sorted(args.re_values)} seeds={args.seeds} -> {args.save_dir}")
    kw = dict(n_save_grid=args.N, save_dir=args.save_dir, overwrite=args.overwrite, n_save=args.n_save,
              t_spin=args.t_spin, save_dt=args.save_dt)
    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_turbulence, Re, kind, seed, args.N_dns or dns_grid(Re, kind), **kw) for Re, kind, seed in jobs]
            for f in as_completed(futs):
                results.append(f.result())
    else:
        for Re, kind, seed in jobs:
            results.append(run_turbulence(Re, kind, seed, args.N_dns or dns_grid(Re, kind), **kw))
    bad = [r for r in results if r[3] > 1e-3]
    if bad:
        print(f"\nWARNING: {len(bad)} runs look under-resolved (enstrophy tail > 1e-3): {sorted(set((k, R) for k, R, _, _ in bad))}")
    print(f"\nAll runs done.  Snapshots in {args.save_dir}/<kind>/Re_*/seed_*/")

if __name__ == "__main__":
    main()
