import numpy as np
import pytest

from ssh_model import (
    H_open,
    critical_disorder,
    disorder_average,
    flat_band,
    mean_log_abs,
    real_space_winding,
    winding_number,
)


def test_flat_band_squares_to_one():
    Q = flat_band(H_open(30, 1.3, 1.0))
    assert np.allclose(Q @ Q, np.eye(60))


@pytest.mark.parametrize("v", [0.2, 0.5, 0.8, 1.2, 1.5, 2.5])
def test_clean_real_space_equals_bloch(v):
    nu_r = real_space_winding(H_open(200, v, 1.0), 200)
    assert abs(nu_r - winding_number(v, 1.0)) < 1e-6


def test_mean_log_abs_closed_form():
    rng = np.random.default_rng(0)
    r = rng.random(2_000_000) - 0.5
    for a, W in [(1.0, 0.5), (0.5, 3.0), (1.0, 2.5)]:
        assert np.isclose(mean_log_abs(a, W), np.mean(np.log(np.abs(a + W * r))), atol=3e-3)


def test_equal_disorder_never_reaches_critical_line():
    assert critical_disorder(0.5, 1.0, ratio=1.0).size == 0


def test_weak_disorder_stays_quantised():
    rng = np.random.default_rng(2)
    for W in [0.3, 0.6, 1.0]:
        mean, _ = disorder_average(200, 0.5, 1.0, W, W / 2, 40, rng)
        assert abs(mean - 1.0) < 0.01


def test_strong_disorder_destroys_winding():
    Wc = critical_disorder(0.5, 1.0, ratio=0.5)
    assert Wc.size == 1 and 3.8 < Wc[0] < 4.1
    rng = np.random.default_rng(3)
    mean, _ = disorder_average(200, 0.5, 1.0, 7.0, 3.5, 40, rng)
    assert abs(mean) < 0.15
