"""Gauge-invariant discrete geometry: Berry phase, Wilson loop, lattice Berry flux.

Convention throughout: A = i<u|grad u>, gamma = loop integral of A, and the
adiabatic phase factor is exp(+i gamma). Then gamma = -Im ln prod <u_j|u_{j+1}>.
"""

import numpy as np

OVERLAP_TOL = 1e-8


def wrap(x):
    """Map an angle to (-pi, pi]."""
    return -np.angle(np.exp(-1j * np.asarray(x)))


def berry_phase(states):
    """Abelian Berry phase of a closed loop of states with shape (M, n), u_M = u_0."""
    ov = np.sum(np.conj(states) * np.roll(states, -1, axis=0), axis=1)
    if np.min(np.abs(ov)) < OVERLAP_TOL:
        raise ValueError("loop too coarse: neighbouring states are orthogonal")
    return -np.angle(np.prod(ov))


def link_matrices(frames):
    """M_j = F_j^dag F_{j+1} for frames of shape (M, n, d), closing link included."""
    return np.conj(np.swapaxes(frames, 1, 2)) @ np.roll(frames, -1, axis=0)


def unitary_part(m):
    """Unitary factor of the polar decomposition (closest unitary in Frobenius norm)."""
    u, s, vh = np.linalg.svd(m)
    if np.min(s) < OVERLAP_TOL:
        raise ValueError("loop too coarse: a link matrix is singular")
    return u @ vh


def wilson_loop(frames):
    """W = M_0 M_1 ... M_{M-1}. Gauge covariant: W -> G_0^dag W G_0."""
    links = link_matrices(frames)
    w = np.eye(frames.shape[2], dtype=complex)
    for m in links:
        w = w @ m
    unitary_part(w)
    return w


def holonomy(frames):
    """Adiabatic holonomy U acting on components in the basis frames[0].

    A state F_0 c returns as F_0 U c. U = (prod_j unitary_part(M_j))^dag, which is
    exactly the parallel-transport result and exactly unitary at any M.
    """
    u = np.eye(frames.shape[2], dtype=complex)
    for m in link_matrices(frames):
        u = u @ unitary_part(m)
    return np.conj(u.T)


def parallel_transport(frames):
    """Rotate frames[1:] so each link F_j^dag F_{j+1} is Hermitian positive.

    Returns M + 1 frames: the last one is frames[0] transported once around the loop,
    so that frames[0]^dag out[-1] = holonomy(frames).
    """
    closed = np.concatenate([frames, frames[:1]])
    out = closed.copy()
    for j in range(len(frames)):
        out[j + 1] = closed[j + 1] @ unitary_part(np.conj(closed[j + 1].T) @ out[j])
    return out


def random_gauge(frames, rng):
    """Multiply every frame by an independent Haar-random U(d)."""
    M, _, d = frames.shape
    z = rng.normal(size=(M, d, d)) + 1j * rng.normal(size=(M, d, d))
    q, r = np.linalg.qr(z)
    q = q * (np.diagonal(r, axis1=1, axis2=2) / np.abs(np.diagonal(r, axis1=1, axis2=2)))[:, None, :]
    return frames @ q


def plaquette_flux(grid):
    """Berry flux through each plaquette of a (N1, N2, n) grid of states.

    The second axis is periodic and the first is open, which covers a sphere
    sampled pole to pole. Each flux lies in (-pi, pi], so the sum over a closed
    surface is exactly 2 pi times an integer (Fukui, Hatsugai, Suzuki 2005).
    """
    def link(a, b):
        return np.sum(np.conj(a) * b, axis=-1)

    u00 = grid[:-1]
    u10 = grid[1:]
    u11 = np.roll(grid[1:], -1, axis=1)
    u01 = np.roll(grid[:-1], -1, axis=1)
    loop = link(u00, u10) * link(u10, u11) * link(u11, u01) * link(u01, u00)
    return -np.angle(loop)
