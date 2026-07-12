import numpy as np
import scipy.linalg as sla

from .model import H_open, H_partner, cell_index, chiral_operator, hoppings, skin_factor


def biorthogonal_eig(H):
    """Eigenvalues, right vectors R and left vectors L with L^dagger R = 1.

    Columns of L satisfy H^dagger L_n = E_n^* L_n. Only the product of the two
    normalisations is fixed; R is scaled to unit norm and L takes the rest.
    For unit vectors |L^dagger R| = 1/kappa, which can be far below machine
    epsilon without being wrong, so there is no cutoff; an exact zero (an
    exceptional point) gives inf/nan in L.
    """
    E, Lv, Rv = sla.eig(H, left=True, right=True)
    Rv = Rv / np.linalg.norm(Rv, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Lv = Lv / np.conj(np.sum(np.conj(Lv) * Rv, axis=0))
    return E, Rv, Lv


def exact_open_spectrum(N, t1, t2, g, extra_site=False):
    """Open-chain eigenvalues from the similar partner chain SSH(v', t2).

    For real v' the partner is Hermitian and eigvalsh is backward stable. For
    imaginary v' the partner is complex symmetric with well-conditioned
    eigenvectors (checked against mpmath in the tests).
    """
    Hp = H_partner(N, t1, t2, g, extra_site)
    if np.isrealobj(Hp):
        return np.linalg.eigvalsh(Hp).astype(complex)
    return np.linalg.eigvals(Hp)


def spectrum_error(N, t1, t2, g):
    """Largest distance from each eig eigenvalue to the nearest exact one."""
    E = np.linalg.eigvals(H_open(N, t1, t2, g))
    Ex = exact_open_spectrum(N, t1, t2, g)
    return np.max(np.min(np.abs(E[:, None] - Ex[None, :]), axis=1))


def min_abs_energy(N, t1, t2, g, exact=False):
    if exact:
        return np.min(np.abs(exact_open_spectrum(N, t1, t2, g)))
    return np.min(np.abs(np.linalg.eigvals(H_open(N, t1, t2, g))))


def eigenvector_condition(N, t1, t2, g):
    """2-norm condition number of the matrix of right eigenvectors (unit columns)."""
    _, R, _ = biorthogonal_eig(H_open(N, t1, t2, g))
    return np.linalg.cond(R)


def cell_weight(psi):
    w = np.abs(psi) ** 2
    return np.bincount(cell_index(len(psi)), weights=w)


def skin_decay_factor(N, t1, t2, g, which="right", edge_cut=1e-3):
    """Per-cell amplitude ratio of the bulk eigenvectors, from a fit of the summed cell weight.

    Each bulk eigenvector is normalised to unit length before summing, so the
    oscillations of the individual standing waves average out.
    """
    E, R, L = biorthogonal_eig(H_open(N, t1, t2, g))
    V = R if which == "right" else L
    V = V / np.linalg.norm(V, axis=0)
    bulk = np.abs(E) > edge_cut
    w = sum(cell_weight(V[:, j]) for j in np.where(bulk)[0])
    n = np.arange(N)
    inner = (n >= 2) & (n < N - 2)
    slope, _ = np.polyfit(n[inner], np.log(w[inner]), 1)
    return np.exp(slope / 2)


def chiral_edge_states(N, t1, t2, g):
    """The two modes closest to E = 0, rotated onto sublattice A and B.

    Returns (R_A, L_A), (R_B, L_B), each pair biorthonormal. The rotation
    diagonalises Gamma inside the pair, so the states are picked by sublattice,
    not by eigenvalue order.
    """
    E, R, L = biorthogonal_eig(H_open(N, t1, t2, g))
    idx = np.argsort(np.abs(E))[:2]
    R2, L2 = R[:, idx], L[:, idx]
    G = L2.conj().T @ (chiral_operator(2 * N)[:, None] * R2)
    gam, c = np.linalg.eig(G)
    R2 = R2 @ c
    L2 = L2 @ np.linalg.inv(c).conj().T
    out = []
    for j in np.argsort(-gam.real):
        r, l = R2[:, j], L2[:, j]
        r = r / np.linalg.norm(r)
        l = l / np.conj(np.vdot(l, r))
        out.append((r, l))
    return out[0], out[1]


def hermitian_edge_states(N, t1, t2, g):
    """Gamma-rotated zero modes of the Hermitian partner SSH(v', t2), v' real."""
    E, U = np.linalg.eigh(H_partner(N, t1, t2, g))
    sub = U[:, np.argsort(np.abs(E))[:2]]
    G = sub.T @ (chiral_operator(2 * N)[:, None] * sub)
    gam, c = np.linalg.eigh(G)
    psi = sub @ c
    return psi[:, 1], psi[:, 0]
