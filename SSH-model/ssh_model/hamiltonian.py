import numpy as np


def h_k(k, v, w):
    return v + w * np.exp(1j * np.asarray(k))


def H_k(k, v, w):
    """Bloch Hamiltonian in the (A, B) basis, H(k + 2pi) = H(k)."""
    h = h_k(k, v, w)
    return np.array([[0.0, np.conj(h)], [h, 0.0]], dtype=complex)


def bands(k, v, w):
    e = np.abs(h_k(k, v, w))
    return -e, e


def H_open(N, v, w):
    """Open chain of N cells, sites ordered A0, B0, A1, B1, ...

    v is a scalar or N intracell hoppings, w a scalar or N - 1 intercell hoppings.
    """
    v = np.broadcast_to(np.asarray(v, dtype=float), (N,))
    w = np.broadcast_to(np.asarray(w, dtype=float), (N - 1,))
    off = np.empty(2 * N - 1)
    off[0::2] = v
    off[1::2] = w
    return np.diag(off, 1) + np.diag(off, -1)


def chiral_operator(N):
    return np.tile([1.0, -1.0], N)


def cell_position(N):
    return np.repeat(np.arange(N, dtype=float), 2)
