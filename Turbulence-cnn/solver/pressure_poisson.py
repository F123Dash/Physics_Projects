import numpy as np


def gradient_p(p, dx, dy):
    dpdx = np.zeros_like(p)
    dpdy = np.zeros_like(p)
    dpdx[1:-1, 1:-1] = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2*dx)
    dpdy[1:-1, 1:-1] = (p[1:-1, 2:] - p[1:-1, :-2]) / (2*dy)
    return dpdx, dpdy


def apply_pressure_bc(p):
    p[:, 0]  = p[:, 1]    # bottom
    p[:, -1] = p[:, -2]   # top
    p[0, :]  = p[1, :]    # left
    p[-1, :] = p[-2, :]   # right
    return p


def solve_pressure_poisson(p, b, dx, dy, n_iter, omega=1.7, tol=1e-6):
    if omega <= 0.0 or omega >= 2.0:
        omega = 1.7
    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2.0 * (dx2 + dy2)

    for it in range(n_iter):
        p[1:-1:2, 1:-1:2] = (
            (1.0 - omega) * p[1:-1:2, 1:-1:2] +
            omega * (
                ((p[2::2, 1:-1:2] + p[:-2:2, 1:-1:2]) * dy2 +
                 (p[1:-1:2, 2::2] + p[1:-1:2, :-2:2]) * dx2 -
                 b[1:-1:2, 1:-1:2] * dx2 * dy2) / denom
            )
        )
        p[2:-1:2, 2:-1:2] = (
            (1.0 - omega) * p[2:-1:2, 2:-1:2] +
            omega * (
                ((p[3::2, 2:-1:2] + p[1:-2:2, 2:-1:2]) * dy2 +
                 (p[2:-1:2, 3::2] + p[2:-1:2, 1:-2:2]) * dx2 -
                 b[2:-1:2, 2:-1:2] * dx2 * dy2) / denom
            )
        )

        p[1:-1:2, 2:-1:2] = (
            (1.0 - omega) * p[1:-1:2, 2:-1:2] +
            omega * (
                ((p[2::2, 2:-1:2] + p[:-2:2, 2:-1:2]) * dy2 +
                 (p[1:-1:2, 3::2] + p[1:-1:2, 1:-2:2]) * dx2 -
                 b[1:-1:2, 2:-1:2] * dx2 * dy2) / denom
            )
        )
        p[2:-1:2, 1:-1:2] = (
            (1.0 - omega) * p[2:-1:2, 1:-1:2] +
            omega * (
                ((p[3::2, 1:-1:2] + p[1:-2:2, 1:-1:2]) * dy2 +
                 (p[2:-1:2, 2::2] + p[2:-1:2, :-2:2]) * dx2 -
                 b[2:-1:2, 1:-1:2] * dx2 * dy2) / denom
            )
        )

        p = apply_pressure_bc(p)

        if tol is not None and (it + 1) % 25 == 0:
            r = (
                (p[2:, 1:-1] - 2.0 * p[1:-1, 1:-1] + p[:-2, 1:-1]) / dx2 +
                (p[1:-1, 2:] - 2.0 * p[1:-1, 1:-1] + p[1:-1, :-2]) / dy2 -
                b[1:-1, 1:-1]
            )
            if np.max(np.abs(r)) < tol:
                break

    return p