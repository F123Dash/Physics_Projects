import numpy as np
import pytest

from nh_ssh import exact_open_spectrum, skin_factor
from non_bloch import (
    H_obc,
    boundary_log_ratio,
    char_roots,
    distance_to_curve,
    gbz_points,
    gbz_radius_ssh,
    growth_factor,
    hatano_nelson,
    hatano_nelson_open_spectrum,
    radial_range,
    range2,
    root_gap,
    ssh_lambda,
    third_root_separation,
)

R2 = range2(0.3, 1.0, 0.5, 0.4)
RHO = 1.171  # sqrt(min |beta| * max |beta|) on the R2 GBZ, the gauge used for eig


@pytest.mark.parametrize("tR, tL", [(1.5, 0.5), (0.4, 1.0), (1.0, 1.0), (2.0, 0.1)])
def test_hatano_nelson_gbz_radius(tR, tL):
    """Theta-scan radius = sqrt(t_R / t_L) to 1e-10."""
    lo, hi = radial_range(gbz_points(hatano_nelson(tR, tL), 400))
    r = np.sqrt(tR / tL)
    assert abs(lo / r - 1) < 1e-10 and abs(hi / r - 1) < 1e-10


def test_hatano_nelson_eigenvectors_are_r_n_times_sines():
    """psi_n / r^n are the orthonormal sine vectors, so |beta| = r with t_R > t_L means growth to the right."""
    N, tR, tL = 20, 1.5, 0.5
    r = np.sqrt(tR / tL)
    E, V = np.linalg.eig(H_obc(N, hatano_nelson(tR, tL)))
    phi = V / (r ** np.arange(N))[:, None]
    phi = phi / np.linalg.norm(phi, axis=0)
    assert np.allclose(np.abs(phi.conj().T @ phi), np.eye(N), atol=1e-8)
    assert np.all(np.argmax(np.abs(V), axis=0) > N // 2)
    assert abs(growth_factor(H_obc(N, hatano_nelson(tR, tL))) / r - 1) < 0.02


def test_hatano_nelson_open_spectrum_on_gbz_not_on_unit_circle():
    """Open eigenvalues have root gap 0 (on E(GBZ)); E(unit circle) is an ellipse far from them."""
    ch = hatano_nelson(1.5, 0.5)
    E = hatano_nelson_open_spectrum(30, 1.5, 0.5)
    assert max(root_gap(ch, e) for e in E) < 1e-10
    ellipse = ch(np.exp(2j * np.pi * np.arange(400) / 400))
    assert np.min(np.abs(ellipse.imag[np.abs(ellipse.real) < 1.5])) > 0.3


@pytest.mark.parametrize("t1", [0.3, 0.7, 1.0, 1.4, 2.0])
def test_nh_ssh_gbz_radius(t1):
    """Root-finding radius = sqrt|(t1 - g/2)/(t1 + g/2)| = 1/|s| of nh-SSH."""
    g = 4 / 3
    lo, hi = radial_range(gbz_points(ssh_lambda(t1, 1.0, g), 400))
    r = gbz_radius_ssh(t1, 1.0, g)
    assert abs(lo / r - 1) < 1e-10 and abs(hi / r - 1) < 1e-10
    assert np.isclose(r, 1 / abs(skin_factor(t1, g)))


@pytest.mark.parametrize("t1", [0.7, 1.0, 1.4])
def test_nh_ssh_scan_of_open_eigenvalues(t1):
    """For every bulk open-chain E, both roots of the quadratic in beta have modulus r (E-scan)."""
    g, N = 4 / 3, 40
    E = exact_open_spectrum(N, t1, 1.0, g)
    bulk = E[np.abs(E) > 0.1]  # edge modes at t1 = 1.0 are only 3e-6 from zero at N = 40
    r = gbz_radius_ssh(t1, 1.0, g)
    mods = np.array([np.abs(char_roots(ssh_lambda(t1, 1.0, g), e * e)) for e in bulk])
    assert np.allclose(mods, r, rtol=1e-8)


def test_nh_ssh_edge_modes_are_off_the_continuum():
    """The zero modes are not on E(GBZ): their root gap is finite."""
    t1, g = 1.0, 4 / 3
    assert root_gap(ssh_lambda(t1, 1.0, g), 0.0) > 0.5


def test_range2_gbz_is_not_a_circle():
    """Max/min |beta| along the GBZ, reported in the README as 1.193."""
    lo, hi = radial_range(gbz_points(R2, 2000))
    assert hi / lo > 1.15
    assert np.isclose(hi / lo, 1.1929, atol=1e-3)


def test_range2_gbz_is_gauge_covariant():
    """Invariance: GBZ of E(rho beta) is the GBZ of E divided by rho."""
    rho = 1.3
    lo, hi = radial_range(gbz_points(R2, 1000))
    lo2, hi2 = radial_range(gbz_points(R2.scaled(rho), 1000))
    assert np.isclose(lo2 * rho, lo, rtol=1e-9) and np.isclose(hi2 * rho, hi, rtol=1e-9)


def test_circular_negative_control():
    """t_2 = t_{-2} (t_1/t_{-1})^2: the GBZ is the circle |beta| = sqrt(t_{-1}/t_1) exactly."""
    lo, hi = radial_range(gbz_points(range2(0.3, 1.0, 0.5, 0.075), 500))
    assert abs(hi / lo - 1) < 1e-10
    assert np.isclose(lo, np.sqrt(2), rtol=1e-10)


def test_reciprocal_chain_has_unit_circle_gbz():
    """Hermitian limit: t_k = t_{-k} gives |beta| = 1."""
    lo, hi = radial_range(gbz_points(range2(0.4, 1.0, 1.0, 0.4), 500))
    assert abs(lo - 1) < 1e-10 and abs(hi - 1) < 1e-10


def test_range2_open_spectrum_converges_as_1_over_N():
    """Open eigenvalues approach E(GBZ); root gap and E-distance both fall as 1/N."""
    curve = R2(gbz_points(R2, 4000))
    Ns = np.array([20, 40, 80])
    gaps, dists = [], []
    for N in Ns:
        E = np.linalg.eigvals(H_obc(N, R2, RHO))
        gaps.append(max(root_gap(R2, e) for e in E))
        dists.append(np.max(distance_to_curve(E, curve)))
    slope = np.polyfit(np.log(Ns), np.log(gaps), 1)[0]
    assert abs(slope + 1) < 0.1
    assert dists[-1] < dists[0] / 3
    assert dists[-1] < 0.01


@pytest.mark.parametrize("N", [20, 40, 80])
def test_finite_size_gap_matches_boundary_prediction(N):
    """(N + 1) * root gap = ln|C(E)| with no free parameter.

    The neglected terms are ~ e^{-N d}, d = third_root_separation, so that is the
    tolerance per eigenvalue (factor 2 margin, plus a round-off floor).
    """
    E = np.linalg.eigvals(H_obc(N, R2, RHO))
    sep = np.array([third_root_separation(R2, e) * N for e in E])
    ok = sep > 3
    assert ok.sum() > 0.8 * N
    err = np.array([abs((N + 1) * root_gap(R2, e) - boundary_log_ratio(R2, e)) for e in E[ok]])
    assert np.all(err < 2 * np.exp(-sep[ok]) + 1e-9)
    assert np.median(err) < 1e-6


def test_boundary_prediction_is_zero_for_hatano_nelson():
    """With one root on each side C = 1: the open eigenvalues sit exactly on E(GBZ) at every N."""
    ch = hatano_nelson(1.5, 0.5)
    for e in hatano_nelson_open_spectrum(15, 1.5, 0.5)[1:-1]:
        assert abs(boundary_log_ratio(ch, e)) < 1e-10


def test_gauged_eig_matches_mpmath():
    """The gauged spectrum agrees with 30-digit mpmath at N = 50."""
    import mpmath as mp

    N = 50
    mp.mp.dps = 30
    H = mp.zeros(N)
    for k, c in zip(R2.powers, R2.coeffs.real):
        for n in range(max(0, -k), min(N, N - k)):
            H[n, n + k] = mp.mpf(c)
    Em = np.array([complex(e) for e in mp.eig(H, left=False, right=False)])
    E = np.linalg.eigvals(H_obc(N, R2, RHO))
    assert np.max(np.min(np.abs(E[:, None] - Em[None, :]), axis=1)) < 1e-12


def test_plain_eig_drifts_at_N_100():
    """Without the gauge, eig at N = 100 differs from the gauged result by more than 1e-9 (README §3.7)."""
    N = 100
    E1 = np.linalg.eigvals(H_obc(N, R2))
    E2 = np.linalg.eigvals(H_obc(N, R2, RHO))
    assert np.max(np.min(np.abs(E1[:, None] - E2[None, :]), axis=1)) > 1e-9
