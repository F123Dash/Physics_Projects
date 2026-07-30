"""Generalised Brillouin zone (Yokomizo and Murakami 2019) for a single band.

With p the order of the pole at beta = 0, beta^p (E(beta) - E) has p + q roots.
Sorted by modulus, the open-chain continuum is the set of E with
|beta_p| = |beta_{p+1}| (1-based), and the GBZ is the set of those two roots.
"""

import numpy as np


def char_roots(chain, E):
    """Roots of beta^p (E(beta) - E) = 0, sorted by modulus."""
    poly = chain.coeffs[::-1].copy()
    poly[chain.q] -= E
    r = np.roots(poly)
    return r[np.argsort(np.abs(r))]


def theta_grid(n):
    """theta in (0, 2 pi), clustered at both ends where the GBZ pair merges (branch points)."""
    u = np.linspace(0.0, np.pi, n + 2)[1:-1]
    return np.pi * (1.0 - np.cos(u))


def gbz_points(chain, n_theta=4000, tol=1e-9):
    """GBZ by the theta-scan, sorted by arg(beta).

    For each theta solve E(beta) = E(beta e^{i theta}). For one band E is an explicit
    function of beta, so this is the resultant of the two characteristic polynomials
    in E, up to a factor. Each solution beta and its partner beta e^{i theta} are
    two roots of equal modulus at the same E. The pair is on the GBZ if exactly p - 1
    of the other roots are smaller and q - 1 are larger. Counting, rather than
    sorting and picking indices p and p+1, is safe at near-degeneracies.
    """
    c, k = chain.coeffs, chain.powers
    pts = []
    for th in theta_grid(n_theta):
        diff = (c * (1.0 - np.exp(1j * k * th)))[::-1]
        for b in np.roots(diff):
            if b == 0:
                continue
            m = np.abs(char_roots(chain, chain(b)))
            rb = abs(b)
            if np.sum(m < rb * (1 - tol)) == chain.p - 1 and np.sum(m > rb * (1 + tol)) == chain.q - 1:
                pts.append(b)
    pts = np.array(pts)
    return pts[np.argsort(np.angle(pts))]


def radial_range(points):
    r = np.abs(points)
    return r.min(), r.max()


def root_gap(chain, E):
    """ln|beta_{p+1}(E)| - ln|beta_p(E)| >= 0; zero exactly on the open-chain continuum."""
    m = np.abs(char_roots(chain, E))
    return np.log(m[chain.p] / m[chain.p - 1])


def boundary_log_ratio(chain, E):
    """ln|C(E)|, the predicted (N + 1) * root_gap of an open-chain eigenvalue.

    psi_n = sum_j c_j beta_j^n must vanish on p sites left of the chain and q sites
    right of it. Expanding that determinant, the two largest terms for large N give
    (beta_{p+1} / beta_p)^{N+1} = C(E): ratio of 2-term Laplace minors (README §2.3).
    Neglected terms are O(|beta_{p-1}/beta_p|^N, |beta_{p+1}/beta_{p+2}|^N).
    """
    b = char_roots(chain, E)
    p, q = chain.p, chain.q
    n = p + q

    def top(S):
        return np.linalg.det(np.array([[b[j] ** (-m) for j in S] for m in range(p)]))

    def bottom(S):
        return np.linalg.det(np.array([[b[j] ** m for j in S] for m in range(q)]))

    def term(S):
        # Laplace sign dropped: only |C| is needed
        return top(S) * bottom([j for j in range(n) if j not in S])

    A = list(range(p))
    B = list(range(p - 1)) + [p]
    return np.log(abs(term(B) / term(A)))


def third_root_separation(chain, E):
    """min(ln|beta_p/beta_{p-1}|, ln|beta_{p+2}/beta_{p+1}|): how fast the neglected terms die."""
    m = np.log(np.abs(char_roots(chain, E)))
    p = chain.p
    seps = []
    if p >= 2:
        seps.append(m[p - 1] - m[p - 2])
    if p + 2 <= len(m):
        seps.append(m[p + 1] - m[p])
    return min(seps) if seps else np.inf


def growth_factor(H, skip=2):
    """Per-site amplitude ratio of the eigenvectors of H, from a fit to their summed weight.

    Each eigenvector is normalised first, so the standing-wave oscillations average out.
    """
    _, V = np.linalg.eig(H)
    w = np.sum(np.abs(V / np.linalg.norm(V, axis=0)) ** 2, axis=1)
    n = np.arange(len(w))[skip:len(w) - skip]
    slope, _ = np.polyfit(n, np.log(w[skip:len(w) - skip]), 1)
    return np.exp(slope / 2)


def distance_to_curve(E, curve):
    """Distance from each E to the polyline through curve (consecutive points joined)."""
    a, b = curve[:-1], curve[1:]
    d = b - a
    L2 = np.maximum(np.abs(d) ** 2, 1e-300)
    E = np.atleast_1d(E)[:, None]
    s = np.clip(((E - a) * np.conj(d)).real / L2, 0.0, 1.0)
    return np.min(np.abs(E - (a + s * d)), axis=1)
