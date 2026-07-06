import numpy as np
import pytest

from berry_phase import (
    BASE,
    H_tripod,
    circle,
    connection,
    connection_exact,
    dark_basis_numeric,
    dark_states,
    evolve_loops,
    field_strength,
    field_strength_exact,
    holonomy,
    loop_frames,
    loop_holonomy,
    loop_pair,
    parallel_transport,
    random_gauge,
    small_loop_field,
    unitary_part,
    wilson_loop,
)

X = np.array([0.7, 0.4, 0.3])
C0 = np.array([1.0, 0.0], dtype=complex)


def test_hamiltonian_structure_and_spectrum():
    H = H_tripod(*X, Omega0=1.7)
    assert H.shape == (4, 4)
    assert np.allclose(H, H.conj().T)
    assert np.allclose(np.diag(H), 0) and np.allclose(H[1:, 1:], 0)
    assert np.isclose(H[0, 3], 1.7 * np.cos(0.7) * np.exp(0.3j))
    assert np.allclose(np.linalg.eigvalsh(H), [-1.7, 0, 0, 1.7], atol=1e-12)


def test_dark_states_are_orthonormal_and_annihilated_by_H():
    D = dark_states(*X)
    assert np.allclose(D.conj().T @ D, np.eye(2))
    assert np.allclose(H_tripod(*X) @ D, 0)


def test_analytic_and_null_space_bases_span_the_same_subspace():
    D, N = dark_states(*X), dark_basis_numeric(*X)
    assert np.allclose(D @ D.conj().T, N @ N.conj().T)


def test_connection_matches_closed_form():
    A = connection(X)
    assert np.allclose(A, np.conj(np.swapaxes(A, 1, 2)))
    assert np.allclose(A, connection_exact(X[0]), atol=1e-9)


def test_connection_does_not_commute_on_the_demonstration_loop():
    _, b = loop_pair()
    expected = np.sqrt(2) * np.cos(BASE[0]) * np.sin(BASE[0]) ** 2
    for x in b(np.linspace(0, 1, 9)):
        A = connection(x)
        assert np.isclose(np.linalg.norm(A[1] @ A[2] - A[2] @ A[1]), expected, rtol=1e-6)
    assert expected > 0.3


def test_real_couplings_give_abelian_holonomy():
    a, b = loop_pair(abelian=True)
    Ua, Ub = loop_holonomy(a, 400), loop_holonomy(b, 400)
    assert np.linalg.norm(Ua @ Ub - Ub @ Ua) < 1e-12


@pytest.mark.parametrize("plane", [(0, 1), (0, 2), (1, 2)])
def test_field_strength_matches_closed_form(plane):
    assert np.allclose(field_strength(X, *plane), field_strength_exact(X[0])[plane], atol=1e-6)


@pytest.mark.parametrize("plane", [(0, 1), (0, 2), (1, 2)])
def test_small_loop_holonomy_is_one_plus_i_F_area(plane):
    r = 0.01
    x0 = X.copy()
    x0[plane[0]] += r
    F = field_strength_exact(X[0])[plane]
    assert np.linalg.norm(small_loop_field(x0, plane, r) - F) < r


def test_holonomy_is_unitary_and_equals_parallel_transport():
    frames = loop_frames(circle(X, (1, 2), 0.8), 500)
    U = holonomy(frames)
    assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-13)
    assert np.allclose(frames[0].conj().T @ parallel_transport(frames)[-1], U, atol=1e-13)


def test_wilson_loop_unitary_part_approaches_holonomy_as_one_over_m_squared():
    path = circle(X, (1, 2), 0.8)
    err, defect = [], []
    for M in (500, 1000):
        frames = loop_frames(path, M)
        W = wilson_loop(frames)
        err.append(np.linalg.norm(unitary_part(W).conj().T - holonomy(frames)))
        defect.append(1 - np.min(np.linalg.svd(W)[1]))
    assert err[1] < 1e-6 and 3.5 < err[0] / err[1] < 4.5
    # the raw product is not unitary: |link| = 1 - O(1/M^2) on each of M links
    assert 1.8 < defect[0] / defect[1] < 2.2


def test_holonomy_is_gauge_covariant():
    rng = np.random.default_rng(11)
    frames = loop_frames(circle(X, (1, 2), 0.8), 300)
    other = random_gauge(frames, rng)
    G0 = frames[0].conj().T @ other[0]
    assert np.allclose(holonomy(other), G0.conj().T @ holonomy(frames) @ G0, atol=1e-12)


def test_null_space_gauge_gives_same_wilson_loop_spectrum():
    path = circle(X, (1, 2), 0.8)
    xs = path(np.arange(300) / 300)
    numeric = np.array([dark_basis_numeric(*x) for x in xs])
    ev = np.sort(np.angle(np.linalg.eigvals(holonomy(numeric))))
    ev_ref = np.sort(np.angle(np.linalg.eigvals(loop_holonomy(path, 300))))
    assert np.allclose(ev, ev_ref, atol=1e-12)


def test_slow_four_level_evolution_follows_the_holonomy():
    _, b = loop_pair()
    D0 = dark_states(*BASE)
    c = D0.conj().T @ evolve_loops([b], 200.0, D0 @ C0)
    assert 1 - np.linalg.norm(c) ** 2 < 1e-6
    assert 1 - abs(np.vdot(loop_holonomy(b, 2000) @ C0, c)) ** 2 < 1e-6


def test_loop_order_matters_and_time_evolution_agrees():
    a, b = loop_pair()
    Ua, Ub = loop_holonomy(a, 2000), loop_holonomy(b, 2000)
    wilson = abs(np.vdot(Ub @ Ua @ C0, Ua @ Ub @ C0))
    assert wilson < 0.3

    D0 = dark_states(*BASE)
    final = [D0.conj().T @ evolve_loops(order, 200.0, D0 @ C0) for order in ((a, b), (b, a))]
    infid = [1 - abs(np.vdot(Ub @ Ua @ C0, final[0])) ** 2, 1 - abs(np.vdot(Ua @ Ub @ C0, final[1])) ** 2]
    assert max(infid) < 1e-5
    # an infidelity 1 - F moves the overlap by at most ~ 2 sqrt(1 - F) per state
    assert abs(abs(np.vdot(*final)) - wilson) < 4 * np.sqrt(max(infid))


def test_abelian_control_loops_commute():
    a, b = loop_pair(abelian=True)
    Ua, Ub = loop_holonomy(a, 400), loop_holonomy(b, 400)
    assert np.isclose(abs(np.vdot(Ub @ Ua @ C0, Ua @ Ub @ C0)), 1.0, atol=1e-12)
