import numpy as np


def hoppings(t1, g):
    """(A <- B, B <- A) intracell amplitudes, t1 + g/2 and t1 - g/2."""
    return t1 + g / 2, t1 - g / 2


def H_open(N, t1, t2, g, extra_site=False):
    """Open chain of N cells, sites ordered A0, B0, A1, B1, ...

    t1 may be a scalar or N values. extra_site appends a lone A site, giving the
    odd chain of Kunst et al. that hosts an exact zero mode for any parameters.
    """
    t1 = np.broadcast_to(np.asarray(t1, dtype=float), (N,))
    up, down = hoppings(t1, g)
    n_bonds = 2 * N if extra_site else 2 * N - 1
    above = np.full(n_bonds, float(t2))
    below = np.full(n_bonds, float(t2))
    above[0::2] = up
    below[0::2] = down
    return np.diag(above, 1) + np.diag(below, -1)


def H_bloch(k, t1, t2, g):
    up, down = hoppings(t1, g)
    return np.array([[0.0, up + t2 * np.exp(-1j * k)],
                     [down + t2 * np.exp(1j * k), 0.0]])


def bloch_energies(k, t1, t2, g):
    """Both bands, E = +-sqrt(H_AB H_BA), on a k grid."""
    up, down = hoppings(t1, g)
    k = np.asarray(k)
    e = np.sqrt((up + t2 * np.exp(-1j * k)) * (down + t2 * np.exp(1j * k)) + 0j)
    return np.concatenate([e, -e])


def bloch_gap_closings(t2, g):
    """Values of t1 >= 0 with |t1 + g/2| = t2 or |t1 - g/2| = t2."""
    cand = np.array([t2 - g / 2, g / 2 - t2, g / 2 + t2])
    return np.unique(cand[cand >= 0])


def chiral_operator(n_sites):
    return np.where(np.arange(n_sites) % 2 == 0, 1.0, -1.0)


def cell_index(n_sites):
    return np.arange(n_sites) // 2


def v_prime(t1, g):
    """Intracell hopping of the Hermitian partner chain; imaginary for |t1| < |g|/2."""
    return np.sqrt(complex(t1 * t1 - g * g / 4))


def skin_factor(t1, g):
    """s = sqrt((t1 + g/2) / (t1 - g/2)); |s| > 1 piles right eigenvectors on the left."""
    up, down = hoppings(t1, g)
    return np.sqrt(complex(up / down))


def similarity_diagonal(n_sites, t1, g):
    """Diagonal of D with H_open = D H_partner D^-1.

    D_{A,n} = s^-n, D_{B,n} = s^-(n+1): the A and B sites of one cell differ by s,
    and B_n, A_{n+1} share a value so the reciprocal t2 bond is untouched.
    """
    s = skin_factor(t1, g)
    i = np.arange(n_sites)
    return s ** (-((i + 1) // 2)).astype(float)


def H_partner(N, t1, t2, g, extra_site=False):
    """SSH chain with symmetric hoppings (v', t2). Hermitian when v' is real."""
    vp = v_prime(t1, g)
    n_bonds = 2 * N if extra_site else 2 * N - 1
    off = np.full(n_bonds, complex(t2))
    off[0::2] = vp
    H = np.diag(off, 1) + np.diag(off, -1)
    return H.real if vp.imag == 0 else H
