"""Tripod scheme: ground states |1>, |2>, |3> coupled to |0> by resonant lasers.

Basis order (|0>, |1>, |2>, |3>). Parameters x = (theta, phi, S):
    Omega_1 = Omega0 sin(theta) cos(phi),  Omega_2 = Omega0 sin(theta) sin(phi),
    Omega_3 = Omega0 cos(theta) e^{iS},    H = sum_j Omega_j |0><j| + h.c.
Only the phase of laser 3 is varied: with S fixed all couplings are real and the
connection is Abelian.
"""

import numpy as np
from scipy.linalg import logm, null_space

from .dynamics import magnus4
from .geometry import holonomy

SY = np.array([[0, -1j], [1j, 0]])


def couplings(theta, phi, S, Omega0=1.0):
    theta, phi, S = np.broadcast_arrays(*(np.asarray(a, float) for a in (theta, phi, S)))
    return Omega0 * np.stack([np.sin(theta) * np.cos(phi) + 0j,
                              np.sin(theta) * np.sin(phi) + 0j,
                              np.cos(theta) * np.exp(1j * S)], axis=-1)


def H_tripod(theta, phi, S, Omega0=1.0):
    """Shape (..., 4, 4). Spectrum: 0 (twice, dark) and +-Omega0."""
    om = couplings(theta, phi, S, Omega0)
    H = np.zeros(om.shape[:-1] + (4, 4), dtype=complex)
    H[..., 0, 1:] = om
    H[..., 1:, 0] = np.conj(om)
    return H


def dark_states(theta, phi, S):
    """Smooth, single-valued dark basis (Ruseckas et al. 2005), shape (..., 4, 2):
    D1 = e^{iS}(sin phi |1> - cos phi |2>),
    D2 = e^{iS} cos theta (cos phi |1> + sin phi |2>) - sin theta |3>."""
    theta, phi, S = np.broadcast_arrays(*(np.asarray(a, float) for a in (theta, phi, S)))
    e, z = np.exp(1j * S), np.zeros(theta.shape)
    d1 = np.stack([z + 0j, e * np.sin(phi), -e * np.cos(phi), z + 0j], axis=-1)
    d2 = np.stack([z + 0j, e * np.cos(theta) * np.cos(phi), e * np.cos(theta) * np.sin(phi),
                   -np.sin(theta) + 0j], axis=-1)
    return np.stack([d1, d2], axis=-1)


def dark_basis_numeric(theta, phi, S):
    """Null space of the row (Omega_1, Omega_2, Omega_3), in whatever gauge SciPy returns."""
    ns = null_space(couplings(theta, phi, S)[None, :])
    return np.vstack([np.zeros((1, 2)), ns])


def connection(x, h=1e-5):
    """A_mu^{ab} = i <D_a|d_mu D_b> in the dark_states gauge, central differences.
    Returns shape (3, 2, 2) for mu = theta, phi, S."""
    x = np.asarray(x, float)
    D = dark_states(*x)
    out = []
    for mu in range(3):
        dx = np.zeros(3)
        dx[mu] = h
        dD = (dark_states(*(x + dx)) - dark_states(*(x - dx))) / (2 * h)
        out.append(1j * np.conj(D.T) @ dD)
    return np.array(out)


def connection_exact(theta):
    c = np.cos(theta)
    return np.array([np.zeros((2, 2)), c * SY, -np.diag([1.0, c * c])], dtype=complex)


def field_strength(x, mu, nu, h=1e-4):
    """F_{mu nu} = d_mu A_nu - d_nu A_mu - i [A_mu, A_nu], so that a small
    counter-clockwise loop of area a in the (mu, nu) plane has U = 1 + i F a."""
    x = np.asarray(x, float)

    def d(k, comp):
        dx = np.zeros(3)
        dx[k] = h
        return (connection(x + dx)[comp] - connection(x - dx)[comp]) / (2 * h)

    A = connection(x)
    return d(mu, nu) - d(nu, mu) - 1j * (A[mu] @ A[nu] - A[nu] @ A[mu])


def field_strength_exact(theta):
    """Analytic F_{theta phi}, F_{theta S}, F_{phi S}."""
    s, c = np.sin(theta), np.cos(theta)
    SX = np.array([[0, 1], [1, 0]], dtype=complex)
    return {(0, 1): -s * SY,
            (0, 2): np.diag([0.0, 2 * s * c]).astype(complex),
            (1, 2): -c * s * s * SX}


def circle(x0, plane, radius):
    """Counter-clockwise circle in the (mu, nu) plane through x0 at s = 0 and s = 1."""
    x0 = np.asarray(x0, float)
    mu, nu = plane

    def path(s):
        s = np.asarray(s, float)
        x = np.broadcast_to(x0, s.shape + (3,)).copy()
        x[..., mu] += radius * (np.cos(2 * np.pi * s) - 1)
        x[..., nu] += radius * np.sin(2 * np.pi * s)
        return x
    return path


BASE = (1.3, 0.0, 0.0)


def loop_pair(scale=1.0, abelian=False):
    """The two loops of the loop-order demonstration, both through BASE.

    Loop a circles in (theta, phi). Loop b circles in (phi, S), where A_phi and A_S
    do not commute; abelian=True swaps it for a second (theta, phi) loop as a control.
    Every loop keeps 0 < theta < pi/2, so all laser amplitudes stay positive.
    """
    a = circle(BASE, (0, 1), 0.6 * scale)
    b = circle(BASE, (1, 0), 0.25 * scale) if abelian else circle(BASE, (1, 2), 1.5 * scale)
    return a, b


def ramp(u, smooth=True):
    """Loop parameter s(t/T). The smooth ramp starts and stops with zero velocity."""
    u = np.asarray(u, float)
    return u - np.sin(2 * np.pi * u) / (2 * np.pi) if smooth else u


def loop_hamiltonian(path, T, Omega0=1.0, smooth=True):
    return lambda t: H_tripod(*np.moveaxis(path(ramp(np.asarray(t) / T, smooth)), -1, 0), Omega0)


def evolve_loops(loops, T, psi0, Omega0=1.0, smooth=True, steps_per_time=40):
    """Full four-level evolution around each closed loop in turn, each taking time T."""
    psi = psi0
    for path in loops:
        psi = magnus4(loop_hamiltonian(path, T, Omega0, smooth), psi, T, int(steps_per_time * T * Omega0))
    return psi


def loop_frames(path, M):
    """Analytic dark frames at s_j = j/M, j = 0..M-1, shape (M, 4, 2)."""
    return dark_states(*np.moveaxis(path(np.arange(M) / M), -1, 0))


def loop_holonomy(path, M=2000):
    return holonomy(loop_frames(path, M))


def small_loop_field(x0, plane, radius, M=400):
    """-i log(U) / area for a circle of the given radius: tends to F_{mu nu} at its centre.
    The area is that of the inscribed M-gon the discrete loop actually encloses."""
    U = loop_holonomy(circle(x0, plane, radius), M)
    return -1j * logm(U) / (0.5 * M * radius ** 2 * np.sin(2 * np.pi / M))
