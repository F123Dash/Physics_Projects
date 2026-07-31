import numpy as np
import pytest

from ssh_model import winding_number
from non_bloch import (
    Laurent,
    bloch_winding,
    edge_pair_count,
    nonbloch_winding,
    nonbloch_winding_closed_form,
    ssh_h1,
    ssh_h2,
    winding,
    winding_by_counting,
    wrap,
)

G = 4 / 3


def test_wrap():
    phi = np.array([0.0, np.pi, -np.pi, 3 * np.pi / 2, -3 * np.pi / 2, 7.0])
    w = wrap(phi)
    assert np.all((w > -np.pi) & (w <= np.pi))
    assert np.allclose(np.exp(1j * w), np.exp(1j * phi))


@pytest.mark.parametrize("m", [-2, -1, 1, 3])
def test_winding_of_monomial(m):
    f = Laurent([1.0], -m)  # beta^m
    assert np.isclose(winding(f, 0.37), m)
    assert np.isclose(winding(f, 2.5), m)


@pytest.mark.parametrize("t1", [0.2, 0.6, 0.95, 1.05, 1.6])
def test_hermitian_limit_reduces_to_project_1(t1):
    """At g = 0 the GBZ is the unit circle and nu = the SSH-model winding."""
    assert np.isclose(nonbloch_winding(t1, 1.0, 0.0), winding_number(t1, 1.0))
    assert np.isclose(bloch_winding(t1, 1.0, 0.0), winding_number(t1, 1.0))


def test_phase_sum_matches_zero_counting():
    """Two independent winding methods: summed phase steps and the argument principle."""
    for g in (0.0, G, 3.0):
        for t1 in np.linspace(0.05, 2.4, 25):
            for r in (0.3, 1.0, 1.7):
                for f in (ssh_h1(t1, 1.0, g), ssh_h2(t1, 1.0, g)):
                    w = winding(f, r)
                    if not np.isnan(w):
                        assert np.isclose(w, winding_by_counting(f, r), atol=1e-8)


def test_nonbloch_winding_matches_closed_form():
    for g in (0.5, G, 3.0):
        for t1 in np.linspace(0.02, 2.5, 60):
            nb, cf = nonbloch_winding(t1, 1.0, g), nonbloch_winding_closed_form(t1, 1.0, g)
            if not (np.isnan(nb) or np.isnan(cf)):
                assert abs(nb - cf) < 1e-8


POINTS = [  # (t1, g, nu_nonBloch, nu_Bloch)
    (0.7, G, 1, 0.5),    # between the Bloch closings: Bloch gives 1/2
    (1.0, G, 1, 0.5),
    (1.3, G, 0, 0.5),    # OBC trivial, Bloch still 1/2
    (0.2, G, 1, 1),      # below both closings: Bloch and non-Bloch agree
    (2.0, G, 0, 0),
    (0.9, 3.0, 0, 0.5),  # below the second OBC transition sqrt(g^2/4 - t2^2) = 1.118
    (1.4, 3.0, 1, 0.5),
    (2.2, 3.0, 0, 0.5),
]


@pytest.mark.parametrize("t1, g, nu, nu_bloch", POINTS)
def test_nonbloch_winding_counts_edge_states(t1, g, nu, nu_bloch):
    """nu_nonBloch = number of zero-mode pairs of the open chain (N = 200); Bloch disagrees."""
    assert np.isclose(nonbloch_winding(t1, 1.0, g), nu, atol=1e-8)
    assert edge_pair_count(200, t1, 1.0, g) == nu
    assert np.isclose(bloch_winding(t1, 1.0, g), nu_bloch, atol=1e-8)


def test_brief_formula_misses_second_transition():
    """The brief's nu = 1 for t1^2 - g^2/4 < t2^2 fails at g = 3, t1 = 0.9: no edge states there."""
    t1, g = 0.9, 3.0
    assert t1 ** 2 - g ** 2 / 4 < 1
    assert nonbloch_winding(t1, 1.0, g) == pytest.approx(0, abs=1e-8)
    assert edge_pair_count(200, t1, 1.0, g) == 0


def test_winding_undefined_at_transitions():
    """Edge cases: nan on the OBC transition, the Bloch closing (for Bloch) and the exceptional point."""
    assert np.isnan(nonbloch_winding(np.sqrt(1 + G * G / 4), 1.0, G))
    assert np.isnan(bloch_winding(1 / 3, 1.0, G))
    assert np.isnan(nonbloch_winding(G / 2, 1.0, G))
    assert np.isnan(nonbloch_winding_closed_form(np.sqrt(1 + G * G / 4), 1.0, G))
