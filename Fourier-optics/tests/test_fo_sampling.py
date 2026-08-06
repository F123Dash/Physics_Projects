from functools import lru_cache

import numpy as np
import pytest

from fourier_optics import (
    beam_width,
    critical_distance,
    gaussian,
    grid,
    lg,
    power,
    propagate,
    rayleigh_range,
    second_moment_width,
    transfer_function,
)

LAM = 0.5
N, DX, A = 1024, 0.5, 20.0  # 1D window 512 um, dx = lambda, slit width 20 um
N_REF = 2**16                # 32.8 mm window: no aliasing up to z = 28 mm


def slit(n):
    return (np.abs(grid(n, DX)) < A / 2).astype(complex)


@lru_cache
def slit_error(z, bandlimit, pad):
    """Relative L2 error on |x| < L/4 against the same slit propagated in a window 64x wider."""
    ref = propagate(slit(N_REF), DX, LAM, z, bandlimit=False)[N_REF // 2 - N // 2:N_REF // 2 + N // 2]
    E = propagate(slit(N), DX, LAM, z, bandlimit=bandlimit, pad=pad)
    c = np.abs(grid(N, DX)) < N * DX / 4
    return np.linalg.norm((E - ref)[c]) / np.linalg.norm(ref[c])


def test_critical_distance():
    """z_c = (L/2) sqrt((2 dx/lambda)^2 - 1); below it the band limit removes nothing."""
    L = N * DX
    zc = critical_distance(L, DX, LAM)
    assert zc == pytest.approx(L / 2 * np.sqrt(3))
    assert np.array_equal(transfer_function((N,), DX, LAM, 0.99 * zc), transfer_function((N,), DX, LAM, 0.99 * zc, bandlimit=False))
    assert not np.array_equal(transfer_function((N,), DX, LAM, 1.01 * zc), transfer_function((N,), DX, LAM, 1.01 * zc, bandlimit=False))
    assert critical_distance(L, 0.2, LAM) == 0.0


def test_reference_window_is_alias_free():
    assert critical_distance(N_REF * DX, DX, LAM) > 3 * 8000.0


def test_band_limit_needed_beyond_critical_distance():
    """Without band limiting the padded propagator is wrong by > 5% at 4.5 z_c; with it, < 0.2%."""
    z = 4000.0
    assert z > 4 * critical_distance(2 * N * DX, DX, LAM)
    assert slit_error(z, False, True) > 0.05
    assert slit_error(z, True, True) < 2e-3


def test_band_limit_irrelevant_below_critical_distance():
    z = 300.0
    assert z < critical_distance(N * DX, DX, LAM)
    assert slit_error(z, False, True) == slit_error(z, True, True) < 1e-3


def test_padding_needed_for_large_angle_wraparound():
    """Without padding the band limit is set by L, not 2L: at 2 z_c(L) it removes more and the error rises."""
    z = 1000.0
    assert slit_error(z, True, True) < slit_error(z, True, False)


@pytest.mark.parametrize("N_win, ok", [(512, True), (128, False)])
def test_window_must_contain_beam(N_win, ok):
    """A 64 um window cannot hold a beam that grows to w = 31.6 um at 3 zR: the width is wrong by > 1%."""
    w0, dx = 10.0, 0.5
    x = grid(N_win, dx)
    X, Y = np.meshgrid(x, x, indexing="ij")
    z = 3 * rayleigh_range(w0, LAM)
    E = propagate(gaussian(X, Y, w0, LAM), dx, LAM, z)
    err = abs(second_moment_width(E, x) / beam_width(z, w0, LAM) - 1)
    assert (err < 1e-3) if ok else (err > 1e-2)


@pytest.mark.parametrize("dx, ok", [(2.0, True), (8.0, False)])
def test_input_sampling_must_resolve_beam(dx, ok):
    """dx = w0/5 is enough for LG_0^3 at zR; dx = 0.8 w0 aliases its spectrum and the propagated field is wrong."""
    w0 = 10.0
    L = 1024.0
    x = grid(int(L / dx), dx)
    X, Y = np.meshgrid(x, x, indexing="ij")
    z = rayleigh_range(w0, LAM)
    E = propagate(lg(X, Y, 0, 3, w0, LAM), dx, LAM, z, paraxial=True)
    ref = lg(X, Y, 0, 3, w0, LAM, z)
    err = np.sqrt(power(E - ref, dx) / power(ref, dx))
    assert (err < 1e-6) if ok else (err > 0.1)
