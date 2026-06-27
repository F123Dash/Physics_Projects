import numpy as np

from .hamiltonian import H_open, chiral_operator


def open_spectrum(N, v, w):
    return np.linalg.eigh(H_open(N, v, w))


def edge_states(N, v, w):
    """The two states closest to E = 0, rotated onto sublattice A (left) and B (right)."""
    E, U = open_spectrum(N, v, w)
    sub = U[:, np.argsort(np.abs(E))[:2]]
    gamma = sub.T @ (chiral_operator(N)[:, None] * sub)
    _, c = np.linalg.eigh(gamma)
    psi_B, psi_A = (sub @ c).T
    return psi_A, psi_B


def fit_xi(N, v, w, floor=1e-12):
    """Fit ln|psi(A, n)| = c - n / xi on the left edge state."""
    psi_A, _ = edge_states(N, v, w)
    amp = np.abs(psi_A[0::2])
    amp = amp / amp[0]
    n = np.arange(N)
    mask = (amp > floor) & (n < N // 2)
    slope, _ = np.polyfit(n[mask], np.log(amp[mask]), 1)
    return -1.0 / slope, n, amp


def xi_exact(v, w):
    return 1.0 / np.log(abs(w / v))


def edge_splitting(N, v, w):
    return np.min(np.abs(np.linalg.eigvalsh(H_open(N, v, w))))


def splitting_prediction(N, v, w):
    r = v / w
    return w * (1.0 - r * r) * r**N
