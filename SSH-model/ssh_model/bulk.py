import numpy as np

from .hamiltonian import H_k, h_k


def k_grid(Nk):
    return 2.0 * np.pi * np.arange(Nk) / Nk


def winding_number(v, w, Nk=1001, gap_tol=1e-9):
    """Winding of h(k) around the origin; nan when the gap is closed."""
    if abs(abs(v) - abs(w)) < gap_tol:
        return np.nan
    h = h_k(k_grid(Nk), v, w)
    return np.sum(np.angle(np.roll(h, -1) / h)) / (2.0 * np.pi)


def lower_band_states(v, w, Nk=400):
    return np.array([np.linalg.eigh(H_k(k, v, w))[1][:, 0] for k in k_grid(Nk)])


def zak_phase(v, w, Nk=400, random_gauge=None):
    """Zak phase of the lower band from the discretised Wilson loop, in (-pi, pi].

    random_gauge: optional Generator, multiplies each u(k_j) by a random phase.
    """
    u = lower_band_states(v, w, Nk)
    if random_gauge is not None:
        u = u * np.exp(2j * np.pi * random_gauge.random(Nk))[:, None]
    overlaps = np.sum(np.conj(u) * np.roll(u, -1, axis=0), axis=1)
    return -np.angle(np.prod(overlaps))
