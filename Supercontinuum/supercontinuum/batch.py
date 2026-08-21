"""Batched GNLSE: many independent propagations advanced together on the GPU.

Each row of the batch is its own run (grid, fibre, input pulse, length, tolerance,
save points); rows only have to share the number of grid points n. Every row keeps
its own step size, so an easy run is never held back by a hard one, and a row that
has reached its fibre end is dropped from the batch (the batch shrinks as it goes).

The integrators are the same as in ``solver.GNLSE``:
  tol given      embedded RK4(3)IP with first-same-as-last, per-row step control
  nsteps given   fixed-step RK4IP or symmetric split step (method="ssfm")

Backend: PyTorch on CUDA when available, otherwise PyTorch on the CPU (still
batched); without PyTorch the runs fall back to ``solver.GNLSE`` one at a time.
Environment overrides: SC_DEVICE (e.g. "cpu", "cuda:1"), SC_MAX_ROWS (rows per
chunk), SC_THREADS (CPU threads for the torch CPU backend).

Precision: complex128 by default. ``precision="single"`` (complex64) is ~5x faster
on consumer GPUs but its rounding (~1e-7) limits the usable tolerance to >~1e-5.
"""

import os
import time
from dataclasses import dataclass
from itertools import groupby

import numpy as np

from .fibre import raman_response
from .solver import GNLSE, Result

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class Run:
    """One propagation. Arguments mean the same as in ``GNLSE.propagate``;
    ``callback(z, Aw)`` gets each saved spectrum as a NumPy array."""
    grid: object
    fibre: object
    A0: np.ndarray
    length: float = None
    nsave: int = 2
    method: str = "rk4ip"
    tol: float = None
    nsteps: int = None
    h0: float = None
    h_max: float = None
    keep: str = "all"
    store: object = np.complex128
    callback: object = None

    def __post_init__(self):
        if self.length is None:
            self.length = self.fibre.length
        if self.tol is None and self.nsteps is None:
            raise ValueError("give nsteps (fixed step) or tol (adaptive)")
        if self.tol is not None and self.method != "rk4ip":
            raise ValueError("adaptive stepping is batched for rk4ip only")

    @property
    def kerr_only(self):
        return not self.fibre.raman and not self.fibre.self_steepening

    @property
    def key(self):
        """Runs with the same key are integrated in one batch."""
        kind = "erk" if self.tol is not None else self.method
        exact = kind == "ssfm" and self.kerr_only
        return (self.grid.n, kind, exact)


# ------------------------------------------------------------------ device
def available_mb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 4096.0


def default_device():
    env = os.environ.get("SC_DEVICE")
    if env:
        return env
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def max_rows(n, device, precision="double"):
    """Rows per chunk: ~28 working arrays of n complex numbers per row."""
    env = os.environ.get("SC_MAX_ROWS")
    if env:
        return max(1, int(env))
    per_row = 28 * n * (16 if precision == "double" else 8)
    if str(device).startswith("cuda"):
        free = torch.cuda.mem_get_info(torch.device(device))[0]
    else:
        free = available_mb() * 2**20
    return max(1, int(0.6 * free // per_row))


# ------------------------------------------------------------------ engine
class _Batch:
    """The operators of all rows of one group, stacked as (rows, n) tensors."""

    def __init__(self, runs, device, precision):
        self.dev = torch.device(device)
        self.c = torch.complex128 if precision == "double" else torch.complex64
        self.r = torch.float64 if precision == "double" else torch.float32
        self.n = runs[0].grid.n
        T = lambda a, dt: torch.as_tensor(np.asarray(a), device=self.dev).to(dt)

        L, pref, R = [], [], []
        for run in runs:
            g, f = run.grid, run.fibre
            L.append(f.linear_operator(g.omega))
            ss = (1 + g.omega / g.omega0) if f.self_steepening else np.ones(g.n)
            pref.append(1j * f.gamma * ss)
            if f.raman:
                h = raman_response(np.arange(g.n) * g.dt, f.tau1, f.tau2)
                h /= h.sum() * g.dt
                R.append((1 - f.fR) + f.fR * np.fft.rfft(h) * g.dt)
            else:
                R.append(np.ones(g.n // 2 + 1))
        self.L = T(np.array(L), self.c)
        self.pref = T(np.array(pref), self.c)
        # rows without Raman get a unit filter; skip the filter if no row has Raman
        self.R = T(np.array(R), self.c) if any(r.fibre.raman for r in runs) else None
        self.evals = np.zeros(len(runs), dtype=np.int64)

    def select(self, idx):
        """Operators of the active rows ``idx`` (a CPU index array)."""
        it = torch.as_tensor(idx, device=self.dev)
        return (self.L[it], self.pref[it], None if self.R is None else self.R[it])

    def N(self, Aw, p, active):
        self.evals[active] += 1
        _, pref, R = p
        A = torch.fft.fft(Aw)
        I = A.real**2 + A.imag**2
        if R is not None:
            I = torch.fft.irfft(R * torch.fft.rfft(I), n=self.n)
        return pref * torch.fft.ifft(A * I)

    def kerr_exact(self, Aw, h, p, active):
        """Kerr-only nonlinear sub-step, exact: A -> A exp(i gamma |A|^2 h)."""
        self.evals[active] += 1
        A = torch.fft.fft(Aw)
        gamma = p[1][:, :1].imag      # pref = i gamma when there is no shock term
        return torch.fft.ifft(A * torch.exp(1j * gamma * (A.real**2 + A.imag**2) * h))

    # ---- steps; h is a (rows, 1) real tensor
    def erk43(self, Aw, k5, h, p, a):
        E = torch.exp(p[0] * (h / 2))
        AI = E * Aw
        k1 = E * k5
        k2 = self.N(AI + (h / 2) * k1, p, a)
        k3 = self.N(AI + (h / 2) * k2, p, a)
        k4 = self.N(E * (AI + h * k3), p, a)
        u4 = E * (AI + (h / 6) * (k1 + 2 * k2 + 2 * k3)) + (h / 6) * k4
        k5n = self.N(u4, p, a)
        err = (h[:, 0] / 10) * torch.linalg.vector_norm(k4 - k5n, dim=1) \
            / torch.linalg.vector_norm(u4, dim=1)
        return u4, k5n, err

    def rk4ip(self, Aw, h, p, a):
        E = torch.exp(p[0] * (h / 2))
        AI = E * Aw
        k1 = E * self.N(Aw, p, a) * h
        k2 = self.N(AI + k1 / 2, p, a) * h
        k3 = self.N(AI + k2 / 2, p, a) * h
        k4 = self.N(E * (AI + k3), p, a) * h
        return E * (AI + k1 / 6 + k2 / 3 + k3 / 3) + k4 / 6

    def ssfm(self, Aw, h, p, a, exact):
        E = torch.exp(p[0] * (h / 2))
        Aw = E * Aw
        if exact:
            Aw = self.kerr_exact(Aw, h, p, a)
        else:
            k1 = self.N(Aw, p, a)
            k2 = self.N(Aw + 0.5 * h * k1, p, a)
            k3 = self.N(Aw + 0.5 * h * k2, p, a)
            k4 = self.N(Aw + h * k3, p, a)
            Aw = Aw + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return E * Aw


def _propagate_group(runs, device, precision):
    B = len(runs)
    eng = _Batch(runs, device, precision)
    _, kind, exact = runs[0].key
    t_start = time.perf_counter()

    z_save = [np.linspace(0, r.length, r.nsave) for r in runs]
    nsave = np.array([r.nsave for r in runs])
    length = np.array([r.length for r in runs])
    outs, results = [], []
    for r, zs in zip(runs, z_save):
        rows = 2 if r.keep == "ends" else r.nsave
        outs.append(np.empty((rows, eng.n), dtype=r.store))
        results.append(Result(r.grid, zs if r.keep == "all" else zs[[0, -1]], outs[-1]))

    def save(i, k, Aw_np):
        r = runs[i]
        if r.keep == "all":
            outs[i][k] = Aw_np
        elif k in (0, r.nsave - 1):
            outs[i][0 if k == 0 else 1] = Aw_np
        if r.callback is not None:
            r.callback(z_save[i][k], Aw_np)

    A0w = np.fft.ifft(np.array([np.asarray(r.A0, dtype=complex) for r in runs]), axis=-1)
    for i in range(B):
        save(i, 0, A0w[i])
    Aw = torch.as_tensor(A0w, device=eng.dev).to(eng.c)

    act = np.arange(B)                        # rows still propagating
    z = np.zeros(B)
    si = np.ones(B, dtype=int)                # index of the next save point
    tgt = np.array([zs[1] for zs in z_save])
    steps = np.zeros(B, dtype=np.int64)
    rejected = np.zeros(B, dtype=np.int64)
    log = []
    p = eng.select(act)

    if kind == "erk":
        tol = np.array([r.tol for r in runs])
        h = np.array([r.h0 if r.h0 is not None else r.length / 1000 for r in runs])
        h_max = np.array([r.h_max if r.h_max is not None else r.length / 10 for r in runs])
        k5 = eng.N(Aw, p, act)
    else:
        per = np.array([int(np.ceil(r.nsteps / (r.nsave - 1))) for r in runs])
        h_fix = np.array([zs[1] / q for zs, q in zip(z_save, per)])
        cnt = np.zeros(B, dtype=int)

    while act.size:
        if kind == "erk":
            rem = tgt[act] - z[act]
            clamped = h[act] >= rem
            hh = np.where(clamped, rem, h[act])
            ht = torch.as_tensor(hh, device=eng.dev).to(eng.r)[:, None]
            u4, k5n, err_t = eng.erk43(Aw, k5, ht, p, act)
            err = err_t.cpu().numpy()
            fac = np.where(err == 0, 2.0,
                           np.clip(0.9 * (tol[act] / np.where(err == 0, 1, err)) ** 0.25, 0.2, 2.0))
            acc = err <= tol[act]
            acc_t = torch.as_tensor(acc, device=eng.dev)[:, None]
            Aw = torch.where(acc_t, u4, Aw)
            k5 = torch.where(acc_t, k5n, k5)
            # same controller as the serial solver; a step shortened to land on a
            # save point does not shrink h
            grown = np.minimum(np.where(hh < h[act], np.maximum(h[act], hh * fac), hh * fac),
                               h_max[act])
            h[act] = np.where(acc, grown, hh * fac)
            rejected[act[~acc]] += 1
            z_new = z[act] + hh
            reached = acc & (clamped | (z_new >= tgt[act] - 1e-15 * length[act]))
            z[act] = np.where(acc, np.where(reached, tgt[act], z_new), z[act])
        else:
            hh = h_fix[act]
            ht = torch.as_tensor(hh, device=eng.dev).to(eng.r)[:, None]
            if kind == "rk4ip":
                Aw = eng.rk4ip(Aw, ht, p, act)
            else:
                Aw = eng.ssfm(Aw, ht, p, act, exact)
            acc = np.ones(act.size, dtype=bool)
            cnt[act] += 1
            reached = cnt[act] == per[act]
            cnt[act[reached]] = 0
            z[act] = np.where(reached, tgt[act], z[act] + hh)

        steps[act[acc]] += 1
        log.append((act[acc], z[act[acc]], hh[acc]))

        if reached.any():
            j = np.nonzero(reached)[0]
            got = Aw[torch.as_tensor(j, device=eng.dev)].cpu().numpy()
            for jj, A_np in zip(j, got):
                i = act[jj]
                save(i, si[i], A_np)
                si[i] += 1
                if si[i] < nsave[i]:
                    tgt[i] = z_save[i][si[i]]
            done = si[act] >= nsave[act]
            if done.any():
                keep = torch.as_tensor(np.nonzero(~done)[0], device=eng.dev)
                act = act[~done]
                Aw = Aw[keep]
                if kind == "erk":
                    k5 = k5[keep]
                p = tuple(None if x is None else x[keep] for x in p)

    wall = time.perf_counter() - t_start
    rows_all = np.concatenate([l[0] for l in log]) if log else np.empty(0, int)
    z_all = np.concatenate([l[1] for l in log]) if log else np.empty(0)
    h_all = np.concatenate([l[2] for l in log]) if log else np.empty(0)
    for i, res in enumerate(results):
        res.steps, res.rejected = int(steps[i]), int(rejected[i])
        res.evaluations, res.wall_time = int(eng.evals[i]), wall
        m = rows_all == i
        res.step_sizes = np.column_stack([z_all[m], h_all[m]])
    return results


def run_batch(runs, device=None, precision="double", verbose=True):
    """Propagate all ``runs``; returns their ``Result``s in the same order.

    Runs are grouped by (n, integrator); each group is integrated as one batch,
    split into chunks that fit the device memory.
    """
    runs = list(runs)
    out = [None] * len(runs)
    if torch is None:
        for i, r in enumerate(runs):
            out[i] = GNLSE(r.grid, r.fibre).propagate(
                r.A0, length=r.length, nsave=r.nsave, method=r.method, nsteps=r.nsteps,
                tol=r.tol, h0=r.h0, h_max=r.h_max, store=r.store, keep=r.keep,
                callback=r.callback)
        return out
    device = default_device() if device is None else device
    if not str(device).startswith("cuda") and os.environ.get("SC_THREADS"):
        torch.set_num_threads(int(os.environ["SC_THREADS"]))
    order = sorted(range(len(runs)), key=lambda i: runs[i].key)
    for key, grp in groupby(order, key=lambda i: runs[i].key):
        idx = list(grp)
        chunk = max_rows(key[0], device, precision)
        for s in range(0, len(idx), chunk):
            part = idx[s:s + chunk]
            t = time.perf_counter()
            res = _propagate_group([runs[i] for i in part], device, precision)
            for i, r in zip(part, res):
                out[i] = r
            if verbose:
                print(f"  batch {len(part):3d} runs  n={key[0]:<6d} {key[1]:5s} on {device}: "
                      f"{time.perf_counter() - t:6.1f} s, "
                      f"{max(r.steps for r in res)} steps max", flush=True)
    return out


def combine(*plans):
    """Several plans as one plan whose output is the list of their outputs."""
    runs = [r for p in plans for r in p[0]]

    def finish(results):
        out, k = [], 0
        for p_runs, fin in plans:
            out.append(fin(results[k:k + len(p_runs)]))
            k += len(p_runs)
        return out
    return runs, finish


def execute(*plans, **kw):
    """Run the propagations of several plans in one batched call and finish each.

    A plan is (runs, finish): ``finish(results)`` turns the results of its own
    runs into the plan's output. Returns the outputs in order.
    """
    runs, finish = combine(*plans)
    return finish(run_batch(runs, **kw))
