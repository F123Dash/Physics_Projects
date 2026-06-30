import numpy as np

from .hamiltonian import H_open, cell_position, chiral_operator


def random_hoppings(N, v, w, Wv, Ww, rng):
    vn = v + Wv * (rng.random(N) - 0.5)
    wn = w + Ww * (rng.random(N - 1) - 0.5)
    return vn, wn


def flat_band(H):
    E, U = np.linalg.eigh(H)
    return (U * np.sign(E)) @ U.T.conj()


def real_space_winding(H, N, L_central=None):
    """nu = (1 / 2L') Tr'[Gamma Q [Q, X]] over the central L' cells (Mondragon-Shem et al. 2014)."""
    if L_central is None:
        L_central = N // 2
    Q = flat_band(H)
    x = cell_position(N)
    comm = Q * x[None, :] - x[:, None] * Q
    start = N // 2 - L_central // 2
    sl = slice(2 * start, 2 * (start + L_central))
    diag = chiral_operator(N)[sl] * np.sum(Q[sl, :] * comm[:, sl].T, axis=1)
    return np.real(np.sum(diag)) / (2.0 * L_central)


def disorder_average(N, v, w, Wv, Ww, realisations, rng, L_central=None):
    vals = np.empty(realisations)
    for i in range(realisations):
        vn, wn = random_hoppings(N, v, w, Wv, Ww, rng)
        vals[i] = real_space_winding(H_open(N, vn, wn), N, L_central)
    return vals.mean(), vals.std(ddof=1) / np.sqrt(realisations)


def mean_log_abs(a, W):
    """<ln|a + W r|> for r uniform in [-1/2, 1/2]."""
    if W == 0:
        return np.log(abs(a))

    def F(x):
        return x * np.log(abs(x)) - x if x != 0 else 0.0

    return (F(a + W / 2) - F(a - W / 2)) / W


def critical_disorder(v, w, ratio, W_max=10.0, n=20001):
    """W where <ln|v_n|> = <ln|w_n|>, with Wv = W and Ww = ratio * W."""
    Ws = np.linspace(0.0, W_max, n)
    d = np.array([mean_log_abs(v, W) - mean_log_abs(w, ratio * W) for W in Ws])
    i = np.where(np.diff(np.sign(d)) != 0)[0]
    return Ws[i] - d[i] * (Ws[i + 1] - Ws[i]) / (d[i + 1] - d[i])
