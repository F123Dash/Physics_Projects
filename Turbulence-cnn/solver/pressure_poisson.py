import numpy as np

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

def solve_pressure_poisson(
    p:      np.ndarray,
    b:      np.ndarray,
    dx:     float,
    dy:     float,
    n_iter: int,
    omega:  float = 1.7,
    tol:    float = 1e-6,
) -> np.ndarray:
    if not (0.0 < omega < 2.0):
        omega = 1.7
    dx2   = dx * dx
    dy2   = dy * dy
    denom = 2.0 * (dx2 + dy2)
    for it in range(n_iter):
        p = apply_pressure_bc(p)
        p[1:-1:2, 1:-1:2] = (
            (1.0 - omega) * p[1:-1:2, 1:-1:2]
            + omega * (
                (p[2::2,   1:-1:2] + p[:-2:2,  1:-1:2]) * dy2
              + (p[1:-1:2, 2::2  ] + p[1:-1:2, :-2:2 ]) * dx2
              - b[1:-1:2, 1:-1:2] * dx2 * dy2
            ) / denom
        )
        p[2:-1:2, 2:-1:2] = (
            (1.0 - omega) * p[2:-1:2, 2:-1:2]
            + omega * (
                (p[3::2,   2:-1:2] + p[1:-2:2, 2:-1:2]) * dy2
              + (p[2:-1:2, 3::2  ] + p[2:-1:2, 1:-2:2]) * dx2
              - b[2:-1:2, 2:-1:2] * dx2 * dy2
            ) / denom
        )
        p = apply_pressure_bc(p)
        p[1:-1:2, 2:-1:2] = (
            (1.0 - omega) * p[1:-1:2, 2:-1:2]
            + omega * (
                (p[2::2,   2:-1:2] + p[:-2:2,  2:-1:2]) * dy2
              + (p[1:-1:2, 3::2  ] + p[1:-1:2, 1:-2:2]) * dx2
              - b[1:-1:2, 2:-1:2] * dx2 * dy2
            ) / denom
        )
        p[2:-1:2, 1:-1:2] = (
            (1.0 - omega) * p[2:-1:2, 1:-1:2]
            + omega * (
                (p[3::2,   1:-1:2] + p[1:-2:2, 1:-1:2]) * dy2
              + (p[2:-1:2, 2::2  ] + p[2:-1:2, :-2:2 ]) * dx2
              - b[2:-1:2, 1:-1:2] * dx2 * dy2
            ) / denom
        )
        p = apply_pressure_bc(p)
        if tol is not None and (it + 1) % 25 == 0:
            lap_p = (
                (p[2:,   1:-1] - 2.0 * p[1:-1, 1:-1] + p[:-2,  1:-1]) / dx2
              + (p[1:-1, 2:  ] - 2.0 * p[1:-1, 1:-1] + p[1:-1, :-2 ]) / dy2
            )
            if np.max(np.abs(lap_p - b[1:-1, 1:-1])) < tol:
                break
    return p