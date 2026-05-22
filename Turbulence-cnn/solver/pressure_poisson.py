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


def solve_pressure_poisson(p, b, dx, dy, n_iter, omega=1.5):
    dx2 = dx * dx
    for _ in range(n_iter):
        p[1:-1:2, 1:-1:2] = (
            (1 - omega) * p[1:-1:2, 1:-1:2] +
            omega * (
                (p[2::2, 1:-1:2] + p[0:-2:2, 1:-1:2] +
                 p[1:-1:2, 2::2] + p[1:-1:2, 0:-2:2]) / 4
                - dx2 / 4 * b[1:-1:2, 1:-1:2]
            )
        )

        # Red update: (even, even)
        p[2:-1:2, 2:-1:2] = (
            (1 - omega) * p[2:-1:2, 2:-1:2] +
            omega * (
                (p[3::2, 2:-1:2] + p[1:-2:2, 2:-1:2] +
                 p[2:-1:2, 3::2] + p[2:-1:2, 1:-2:2]) / 4
                - dx2 / 4 * b[2:-1:2, 2:-1:2]
            )
        )

        # Black update: (odd, even)
        p[1:-1:2, 2:-1:2] = (
            (1 - omega) * p[1:-1:2, 2:-1:2] +
            omega * (
                (p[2::2, 2:-1:2] + p[0:-2:2, 2:-1:2] +
                 p[1:-1:2, 3::2] + p[1:-1:2, 1:-2:2]) / 4
                - dx2 / 4 * b[1:-1:2, 2:-1:2]
            )
        )

        # Black update: (even, odd)
        p[2:-1:2, 1:-1:2] = (
            (1 - omega) * p[2:-1:2, 1:-1:2] +
            omega * (
                (p[3::2, 1:-1:2] + p[1:-2:2, 1:-1:2] +
                 p[2:-1:2, 2::2] + p[2:-1:2, 0:-2:2]) / 4
                - dx2 / 4 * b[2:-1:2, 1:-1:2]
            )
        )

        p = apply_pressure_bc(p)
    return p
