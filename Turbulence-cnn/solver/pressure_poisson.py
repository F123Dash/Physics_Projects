import numpy as np
from scipy.fft import dctn, idctn

def gradient_p(p: np.ndarray, dx: float, dy: float):
    dpdx = np.zeros_like(p)
    dpdy = np.zeros_like(p)
    dpdx[1:-1, 1:-1] = (p[2:,   1:-1] - p[:-2,  1:-1]) / (2.0 * dx)
    dpdy[1:-1, 1:-1] = (p[1:-1, 2:  ] - p[1:-1, :-2 ]) / (2.0 * dy)
    return dpdx, dpdy

def apply_pressure_bc(p: np.ndarray) -> np.ndarray:
    p[:, 0]  = p[:, 1]    # bottom  (j = 0)
    p[:, -1] = p[:, -2]   # top     (j = N-1)
    p[0, :]  = p[1, :]    # left    (i = 0)
    p[-1, :] = p[-2, :]   # right   (i = N-1)
    return p

_EIG_CACHE = {}

def _laplacian_eigenvalues(nx: int, ny: int, dx: float, dy: float) -> np.ndarray:
    # The 5-point Laplacian on the interior nodes, with the Neumann condition
    # p[0] = p[1] eliminated, is diagonalised exactly by the DCT-II.
    key = (nx, ny, dx, dy)
    if key not in _EIG_CACHE:
        lx = -4.0 / dx**2 * np.sin(np.pi * np.arange(nx) / (2 * nx))**2
        ly = -4.0 / dy**2 * np.sin(np.pi * np.arange(ny) / (2 * ny))**2
        lam = lx[:, None] + ly[None, :]
        lam[0, 0] = 1.0   # constant mode is the Neumann null space; zeroed below
        _EIG_CACHE[key] = lam
    return _EIG_CACHE[key]

def solve_pressure_poisson(b: np.ndarray, dx: float, dy: float) -> np.ndarray:
    rhs = b[1:-1, 1:-1]
    lam = _laplacian_eigenvalues(*rhs.shape, dx, dy)
    p_hat = dctn(rhs, type=2, norm="ortho") / lam
    p_hat[0, 0] = 0.0
    p = np.zeros_like(b)
    p[1:-1, 1:-1] = idctn(p_hat, type=2, norm="ortho")
    p = apply_pressure_bc(p)
    return p - p.mean()
