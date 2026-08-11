"""iSCAT image formation."""

import numpy as np
from scipy.special import j0, jv, roots_legendre


def wrap(phi):
    """Phase mapped to (-pi, pi]."""
    return np.pi - np.mod(np.pi - np.asarray(phi), 2 * np.pi)


def fresnel(n1, n2):
    """Normal-incidence (r, t, t') for glass/water."""
    return (n1 - n2) / (n1 + n2), 2 * n1 / (n1 + n2), 2 * n2 / (n1 + n2)


def fresnel_angular(kz_i, kz_t, n_i, n_t):
    """(r_s, r_p, t_s, t_p) from medium i into medium t."""
    ni2, nt2 = n_i ** 2, n_t ** 2
    r_s = (kz_i - kz_t) / (kz_i + kz_t)
    t_s = 2 * kz_i / (kz_i + kz_t)
    r_p = (nt2 * kz_i - ni2 * kz_t) / (nt2 * kz_i + ni2 * kz_t)
    t_p = 2 * n_i * n_t * kz_i / (nt2 * kz_i + ni2 * kz_t)
    return r_s, r_p, t_s, t_p


def contrast(es, er, ey=0.0):
    """iSCAT contrast (I - |E_r|^2)/|E_r|^2 with I = |E_r + E_s|^2 + |E_y|^2."""
    s = np.asarray(es) / er
    return 2 * s.real + np.abs(s) ** 2 + np.abs(np.asarray(ey) / er) ** 2


def scalar_peak(h, f, lam0, n_m, NA, n1, zf=0.0):
    _, t, tp = fresnel(n1, n_m)
    k = 2 * np.pi * n_m / lam0
    kzm = k * np.sqrt(max(0.0, 1 - (min(NA, n_m) / n_m) ** 2))
    h = np.asarray(h, dtype=float)
    d = h - zf
    small = np.abs(k * d) < 1e-6
    ds = np.where(small, 1.0, d)
    g = np.where(small, 1j * (k - kzm) * (1 + 0.5j * (k + kzm) * d),
                 (np.exp(1j * k * d) - np.exp(1j * kzm * d)) / ds)
    return t * tp * f * np.exp(1j * k * h) * g


def scalar_psf(rho, h, f, lam0, n_m, NA, n1, zf=0.0, order=400):
    """Scalar image E_s/E0 at radius rho."""
    _, t, tp = fresnel(n1, n_m)
    k = 2 * np.pi * n_m / lam0
    kzm = k * np.sqrt(max(0.0, 1 - (min(NA, n_m) / n_m) ** 2))
    x, w = roots_legendre(order)
    u = kzm + (k - kzm) * (x + 1) / 2
    w = w * (k - kzm) / 2
    rho = np.asarray(rho, dtype=float)
    kappa = np.sqrt(np.maximum(k ** 2 - u ** 2, 0))
    phase = np.exp(1j * u * (h - zf))
    integral = (j0(np.multiply.outer(rho, kappa)) * phase) @ w
    return t * tp * f * np.exp(1j * k * h) * 1j * integral


def _pupil_nodes(k1, k2, kmax, order):
    """Quadrature nodes in kappa, split at kappa = k2 so that kz2 is smooth on each piece."""
    x, w = roots_legendre(order)
    th_max = np.arcsin(min(kmax / k2, 1.0))
    th = th_max * (x + 1) / 2
    kap = [k2 * np.sin(th)]
    kz2 = [k2 * np.cos(th)]
    wt = [w * th_max / 2 * k2 ** 2 * np.sin(th) * np.cos(th)]
    if kmax > k2:
        u_max = np.arccosh(kmax / k2)
        u = u_max * (x + 1) / 2
        kap.append(k2 * np.cosh(u))
        kz2.append(1j * k2 * np.sinh(u))
        wt.append(w * u_max / 2 * k2 ** 2 * np.cosh(u) * np.sinh(u))
    kap, kz2, wt = map(np.concatenate, (kap, kz2, wt))
    kz1 = np.sqrt(k1 ** 2 - kap ** 2 + 0j)
    return kap, kz1, kz2, wt


def _pupil_ab(kap, kz1, kz2, h, zf, k1, k2, n1, n2):
    """Pupil amplitudes a, b (per unit kappa dkappa) of the J0 and J2 terms of the x-dipole image."""
    T_s = 2 / (kz2 + kz1)
    T_p = 2 * n1 * n2 / (n1 ** 2 * kz2 + n2 ** 2 * kz1)
    cos2 = kz2 / k2
    g = np.sqrt(kz1.real / k1) * np.exp(1j * kz2 * h - 1j * kz1 * zf)
    return g * (T_s + cos2 * T_p), g * (T_s - cos2 * T_p)


def momentum_transfer_sq(kz2, k2):
    """Momentum transfer q^2 = 2 k2 (k2 + kz2) for a collected plane wave."""
    return 2 * k2 * (k2 + kz2)


def vectorial_psf(x, y, h, alpha, lam0, n1, n2, NA, zf=0.0, order=300, form_factor=None):
    if NA >= n1:
        raise ValueError("NA must be below n1")
    k0 = 2 * np.pi / lam0
    k1, k2 = n1 * k0, n2 * k0
    _, t, _ = fresnel(n1, n2)
    kap, kz1, kz2, wt = _pupil_nodes(k1, k2, NA * k0, order)
    a, b = _pupil_ab(kap, kz1, kz2, h, zf, k1, k2, n1, n2)
    A, B = wt * a, wt * b
    if form_factor is not None:
        ff = form_factor(momentum_transfer_sq(kz2, k2))
        A, B = A * ff, B * ff
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)
    arg = np.multiply.outer(rho, kap)
    I0 = j0(arg) @ A
    I2 = jv(2, arg) @ B
    pre = 1j * k2 ** 2 * alpha * t * np.exp(1j * k2 * h) / (8 * np.pi)
    ex = pre * (I0 + I2 * np.cos(2 * phi))
    ey = pre * I2 * np.sin(2 * phi)
    return ex, ey


def vectorial_transfer(kx, ky, h, lam0, n1, n2, NA, zf=0.0):
    k0 = 2 * np.pi / lam0
    k1, k2 = n1 * k0, n2 * k0
    _, t, _ = fresnel(n1, n2)
    kap = np.hypot(kx, ky)
    kz1 = np.sqrt(k1 ** 2 - kap ** 2 + 0j)
    kz2 = np.sqrt(k2 ** 2 - kap ** 2 + 0j)
    a, b = _pupil_ab(kap, kz1, kz2, h, zf, k1, k2, n1, n2)
    phik = np.arctan2(ky, kx)
    pre = 2 * np.pi * 1j * k2 ** 2 * t * np.exp(1j * k2 * h) / (8 * np.pi)
    inside = kap <= NA * k0
    return (np.where(inside, pre * (a - b * np.cos(2 * phik)), 0),
            np.where(inside, -pre * b * np.sin(2 * phik), 0))


class DipoleImager:
    def __init__(self, lam0, n1, n2, NA, zf=0.0, rho_max=2000.0, d_rho=0.5, order=200):
        if NA >= n1:
            raise ValueError("NA must be below n1")
        k0 = 2 * np.pi / lam0
        self.k1, self.k2 = n1 * k0, n2 * k0
        self.t = fresnel(n1, n2)[1]
        kap, kz1, self.kz2, wt = _pupil_nodes(self.k1, self.k2, NA * k0, order)
        a, b = _pupil_ab(kap, kz1, self.kz2, 0.0, zf, self.k1, self.k2, n1, n2)
        self.A0, self.B0 = wt * a, wt * b
        self.rho = np.arange(0.0, rho_max + d_rho, d_rho)
        arg = np.multiply.outer(self.rho, kap)
        self.J0, self.J2 = j0(arg), jv(2, arg)

    def field(self, x, y, h, alpha):
        """(E_x, E_y)/E0 at camera points (x, y) relative to the particle."""
        ph = np.exp(1j * self.kz2 * h)
        I0t = self.J0 @ (self.A0 * ph)
        I2t = self.J2 @ (self.B0 * ph)
        rho = np.hypot(x, y)
        if np.max(rho) > self.rho[-1]:
            raise ValueError("camera point beyond the tabulated radius rho_max")
        phi = np.arctan2(y, x)
        interp = lambda tab: np.interp(rho, self.rho, tab.real) + 1j * np.interp(rho, self.rho, tab.imag)
        I0, I2 = interp(I0t), interp(I2t)
        pre = 1j * self.k2 ** 2 * alpha * self.t * np.exp(1j * self.k2 * h) / (8 * np.pi)
        return pre * (I0 + I2 * np.cos(2 * phi)), pre * I2 * np.sin(2 * phi)


def scalar_reference(lam0, n_m, n1, n2, zf=0.0):
    """Reference r exp(-i k_m zf) in the homogeneous scalar model."""
    return fresnel(n1, n2)[0] * np.exp(-2j * np.pi * n_m / lam0 * zf)


def vectorial_reference(lam0, n1, n2, zf=0.0):
    """Reference r exp(-i k1 zf), consistent with vectorial_psf (kappa = 0 wave)."""
    return fresnel(n1, n2)[0] * np.exp(-2j * np.pi * n1 / lam0 * zf)


def vectorial_window_signal(h, alpha, lam0, n1, n2, NA, W, zf=0.0, order=400):
    """Average of E_x/E0 over the camera with Gaussian weight exp(-rho^2/W^2)."""
    k0 = 2 * np.pi / lam0
    k1, k2 = n1 * k0, n2 * k0
    _, t, _ = fresnel(n1, n2)
    kap, kz1, kz2, wt = _pupil_nodes(k1, k2, min(NA * k0, 8 / W), order)
    a, _ = _pupil_ab(kap, kz1, kz2, np.asarray(h)[..., None], zf, k1, k2, n1, n2)
    A = wt * a
    pre = 1j * k2 ** 2 * alpha * t * np.exp(1j * k2 * np.asarray(h)) / (8 * np.pi)
    return pre * np.sum(A * np.exp(-kap ** 2 * W ** 2 / 4), axis=-1)


def best_focus(h, lam0, n1, n2, NA, alpha_phase=0.0, bounds=(-400.0, 500.0)):
    """Focal plane that makes the centre interference term most negative."""
    from scipy.optimize import minimize_scalar

    def interference(zf):
        ex, _ = vectorial_psf(0.0, 0.0, h, np.exp(1j * alpha_phase), lam0, n1, n2, NA, zf=zf)
        return (ex / vectorial_reference(lam0, n1, n2, zf)).real

    return minimize_scalar(interference, bounds=bounds, method="bounded", options={"xatol": 1e-3}).x
