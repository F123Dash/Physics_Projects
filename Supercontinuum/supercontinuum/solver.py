"""GNLSE integrators: symmetric split-step Fourier (SSFM) and RK4IP.

The state is the spectrum Aw (FFT order). The equation in the frequency domain is

    dAw/dz = L(Omega) Aw + N(Aw),
    L = i beta(Omega) - alpha/2,
    N = i gamma (1 + Omega/omega0) F{ A * [(1-fR)|A|^2 + fR h_R (*) |A|^2] },

where (1 + Omega/omega0) is the self-steepening factor (dT -> -i Omega).

Adaptive stepping
  rk4ip : embedded RK4(3)IP with first-same-as-last (Balac & Mahe 2013):
          4 nonlinear evaluations per step, error from the embedded 3rd-order solution.
  ssfm  : step doubling (local-error method, Sinkin et al. 2003) with Richardson
          extrapolation; 3 split steps per accepted step.
"""

from dataclasses import dataclass, field
import time

import numpy as np
import scipy.fft as sfft

from .fibre import raman_response

# scipy.fft is ~20% faster than numpy.fft here and has real transforms. The
# transforms below are numerically identical in convention to numpy's.
fft = sfft.fft
ifft = sfft.ifft
rfft = sfft.rfft
irfft = sfft.irfft


@dataclass
class Result:
    grid: object
    z: np.ndarray            # saved positions (m)
    Aw: np.ndarray           # spectra, shape (len(z), n), FFT order
    steps: int = 0           # accepted steps
    rejected: int = 0        # rejected steps (adaptive only)
    evaluations: int = 0     # nonlinear-operator evaluations (cost measure)
    wall_time: float = 0.0
    step_sizes: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    @property
    def At(self):
        return np.fft.fft(self.Aw, axis=-1)

    @property
    def final(self):
        return np.fft.fft(self.Aw[-1])


class GNLSE:
    def __init__(self, grid, fibre):
        if fibre.self_steepening and grid.omega0 is None:
            raise ValueError("self-steepening needs a carrier wavelength on the grid")
        self.grid = grid
        self.fibre = fibre
        self.L = fibre.linear_operator(grid.omega)
        self.gamma = fibre.gamma
        ss = (1 + grid.omega / grid.omega0) if fibre.self_steepening else 1.0
        # prefactor i gamma (1 + Omega/omega0), folded into one array (or scalar)
        self.pref = 1j * fibre.gamma * ss
        self.fR = fibre.fR if fibre.raman else 0.0
        if fibre.raman:
            # causal response sampled from t = 0, normalised so sum(h) dt = 1;
            # circular convolution (h * I)(T) = irfft(rfft(h) rfft(I)) dt.
            # (1-fR) I + fR (h * I) is folded into one real-FFT filter.
            h = raman_response(np.arange(grid.n) * grid.dt, fibre.tau1, fibre.tau2)
            h /= h.sum() * grid.dt
            self.R = (1 - self.fR) + self.fR * rfft(h) * grid.dt
        else:
            self.R = None
        self.kerr_only = not fibre.raman and not fibre.self_steepening
        self._exp_cache = {}
        self.evaluations = 0

    # ---------------------------------------------------------------- operators
    def nonlinear(self, Aw):
        self.evaluations += 1
        A = fft(Aw)
        I = A.real**2 + A.imag**2
        if self.R is not None:
            I = irfft(self.R * rfft(I), n=I.size)
        A *= I
        return self.pref * ifft(A, overwrite_x=True)

    def _exp(self, h):
        key = float(h)
        e = self._exp_cache.get(key)
        if e is None:
            if len(self._exp_cache) >= 8:
                self._exp_cache.clear()
            e = np.exp(self.L * h)
            self._exp_cache[key] = e
        return e

    # -------------------------------------------------------------------- steps
    def step_rk4ip(self, Aw, h):
        """Fourth-order Runge-Kutta in the interaction picture (Hult 2007)."""
        E = self._exp(h / 2)
        AI = E * Aw
        k1 = E * self.nonlinear(Aw) * h
        k2 = self.nonlinear(AI + k1 / 2) * h
        k3 = self.nonlinear(AI + k2 / 2) * h
        k4 = self.nonlinear(E * (AI + k3)) * h
        return E * (AI + k1 / 6 + k2 / 3 + k3 / 3) + k4 / 6

    def step_erk43(self, Aw, k5, h):
        """Embedded RK4(3)IP step. ``k5`` is N(Aw) from the previous step (FSAL).

        Returns (4th-order solution, N at it, relative error estimate).
        """
        N = self.nonlinear
        E = self._exp(h / 2)
        AI = E * Aw
        k1 = E * k5
        k2 = N(AI + (h / 2) * k1)
        k3 = N(AI + (h / 2) * k2)
        k4 = N(E * (AI + h * k3))
        beta = E * (AI + (h / 6) * (k1 + 2 * k2 + 2 * k3))
        u4 = beta + (h / 6) * k4
        k5_new = N(u4)
        # embedded 3rd order: u3 = beta + h/30 (2 k4 + 3 k5), so u4 - u3 = h/10 (k4 - k5)
        err = (h / 10) * np.linalg.norm(k4 - k5_new) / np.linalg.norm(u4)
        return u4, k5_new, err

    def step_ssfm(self, Aw, h):
        """Symmetric (Strang) split step: D/2, N, D/2.

        Kerr only: the nonlinear sub-step is exact, A -> A exp(i gamma |A|^2 h).
        Otherwise the nonlinear sub-step is integrated with classical RK4,
        which keeps the splitting second order overall.
        """
        E = self._exp(h / 2)
        Aw = E * Aw
        if self.kerr_only:
            self.evaluations += 1
            A = fft(Aw)
            A = A * np.exp(1j * self.gamma * (A.real**2 + A.imag**2) * h)
            Aw = ifft(A)
        else:
            k1 = self.nonlinear(Aw)
            k2 = self.nonlinear(Aw + 0.5 * h * k1)
            k3 = self.nonlinear(Aw + 0.5 * h * k2)
            k4 = self.nonlinear(Aw + h * k3)
            Aw = Aw + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return E * Aw

    # ---------------------------------------------------------------- propagate
    def propagate(self, A0, length=None, nsave=2, method="rk4ip", nsteps=None,
                  tol=None, h0=None, h_max=None, progress=False, store=np.complex128,
                  keep="all", callback=None):
        """Propagate the time-domain field A0 over ``length``.

        Fixed step: give ``nsteps`` (total, spread evenly over the save intervals).
        Adaptive: give ``tol`` (relative local error per step).
        ``store``: dtype of the saved spectra (complex64 halves memory for maps).
        ``keep``: "all" keeps every save point; "ends" keeps only input and output
        (Result.z is then [0, length]); use with ``callback(z, Aw)``, which is
        called at every save point, to reduce the history on the fly.
        """
        length = self.fibre.length if length is None else length
        z_save = np.linspace(0, length, nsave)
        Aw = ifft(np.asarray(A0, dtype=complex))
        if keep == "all":
            out = np.empty((nsave, self.grid.n), dtype=store)
            rows = np.arange(nsave)
        else:
            out = np.empty((2, self.grid.n), dtype=store)
            rows = np.full(nsave, -1)
            rows[0] = 0
            rows[-1] = 1

        def save(i, Aw):
            if rows[i] >= 0:
                out[rows[i]] = Aw
            if callback is not None:
                callback(z_save[i], Aw)

        save(0, Aw)
        self.evaluations = 0
        t_start = time.perf_counter()
        res = Result(self.grid, z_save if keep == "all" else z_save[[0, -1]], out)
        steps_log = []

        if tol is None:
            if nsteps is None:
                raise ValueError("give nsteps (fixed step) or tol (adaptive)")
            step = {"rk4ip": self.step_rk4ip, "ssfm": self.step_ssfm}[method]
            per = int(np.ceil(nsteps / (nsave - 1)))
            for i in range(1, nsave):
                h = (z_save[i] - z_save[i - 1]) / per
                for _ in range(per):
                    Aw = step(Aw, h)
                res.steps += per
                save(i, Aw)
        elif method == "rk4ip":
            h = h0 if h0 is not None else length / 1000
            h_max = h_max if h_max is not None else length / 10
            z = 0.0
            k5 = self.nonlinear(Aw)
            for i in range(1, nsave):
                z_target = z_save[i]
                while z < z_target - 1e-15 * length:
                    hh = min(h, z_target - z)
                    u4, k5_new, err = self.step_erk43(Aw, k5, hh)
                    # step-size controller for a 3rd-order error estimate
                    fac = 2.0 if err == 0 else min(2.0, max(0.2, 0.9 * (tol / err) ** 0.25))
                    if err > tol:
                        res.rejected += 1
                        h = hh * fac
                        continue
                    Aw, k5 = u4, k5_new
                    z += hh
                    res.steps += 1
                    steps_log.append((z, hh))
                    # a step shortened to land on a save point does not shrink h
                    h = min(max(h, hh * fac) if hh < h else hh * fac, h_max)
                save(i, Aw)
                if progress:
                    print(f"\r  z = {z*100:7.3f} cm  steps {res.steps}", end="", flush=True)
        else:  # adaptive split step: step doubling
            h = h0 if h0 is not None else length / 1000
            h_max = h_max if h_max is not None else length / 10
            z = 0.0
            for i in range(1, nsave):
                z_target = z_save[i]
                while z < z_target - 1e-15 * length:
                    hh = min(h, z_target - z)
                    coarse = self.step_ssfm(Aw, hh)
                    fine = self.step_ssfm(self.step_ssfm(Aw, hh / 2), hh / 2)
                    err = np.linalg.norm(fine - coarse) / np.linalg.norm(fine)
                    if err > 2 * tol:
                        h = hh / 2
                        res.rejected += 1
                        continue
                    Aw = fine + (fine - coarse) / 3
                    z += hh
                    res.steps += 1
                    steps_log.append((z, hh))
                    if err > tol:
                        h = hh / 2 ** (1 / 3)
                    elif err < tol / 2:
                        h = min(hh * 2 ** (1 / 3), h_max)
                    else:
                        h = hh
                save(i, Aw)
        if progress:
            print()
        res.evaluations = self.evaluations
        res.wall_time = time.perf_counter() - t_start
        if steps_log:
            res.step_sizes = np.array(steps_log)
        return res
