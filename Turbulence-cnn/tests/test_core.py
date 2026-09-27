"""Regression tests for bugs found in review. Run with:  python -m pytest tests"""
import os, sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver import ns_solver as ns
from solver.pressure_poisson import solve_pressure_poisson
from data.augmentation import mirror, make_coarse_field
from data.dataset import split_by_re, compute_stats, normalise, make_dataloaders
from models.losses import TurbulenceLoss, ssim
from models.unet import TurbulenceUNet
from train.metrics import sample_metrics
from train.train import LOSS_KEYS

N = 64
DX = 1.0 / (N - 1)


def smooth_field(seed=0):
    """A smooth (4, N, N) physical field [u, v, p, omega] with omega from the solver's own stencil."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    a, b, c = rng.uniform(0.5, 2.0, 3)
    u = np.sin(np.pi*a*X) * np.cos(np.pi*b*Y) + 0.3
    v = -np.cos(np.pi*c*X) * np.sin(np.pi*a*Y) - 0.1
    p = 0.05*np.cos(2*np.pi*X*Y)
    w = ns.vorticity(u, v, DX, DX)
    return np.stack([u, v, p, w]).astype(np.float32)

@pytest.fixture
def stats():
    # Deliberately non-zero means and unequal stds, so unit mistakes show up.
    return {"mean": [0.3, -0.1, 0.0, -1.2], "std": [0.2, 0.13, 0.036, 7.6],
            "log_re_mean": 3.0, "log_re_std": 0.4}

def to_batch(fields, stats):
    return torch.from_numpy(np.stack([normalise(f, stats) for f in fields])).float()

#  physics losses

def test_vorticity_loss_is_zero_on_solver_fields(stats):
    x = to_batch([smooth_field(0), smooth_field(1)], stats)
    assert TurbulenceLoss(stats).vorticity_loss(x).item() < 1e-6

def test_vorticity_loss_detects_wrong_vorticity(stats):
    f = smooth_field(0); f[3] *= 1.5
    assert TurbulenceLoss(stats).vorticity_loss(to_batch([f], stats)).item() > 1e-3

@pytest.mark.parametrize("axis", ["x", "y"])
def test_mirrored_fields_stay_consistent(stats, axis):
    # Wrong flip signs would break the omega = curl(u, v) relation.
    x = to_batch([mirror(smooth_field(2), axis)], stats)
    assert TurbulenceLoss(stats).vorticity_loss(x).item() < 1e-6

@pytest.mark.parametrize("axis", ["x", "y"])
def test_mirror_twice_is_identity(axis):
    f = smooth_field(3)
    np.testing.assert_allclose(mirror(mirror(f, axis), axis), f, atol=0)

#  SSIM-

def test_ssim_of_identical_fields_is_one():
    x = torch.randn(3, 4, N, N)
    assert ssim(x, x).item() == pytest.approx(1.0, abs=1e-5)

def test_ssim_is_unchanged_by_shift_and_scale():
    t = torch.randn(2, 4, N, N); p = t + 0.3*torch.randn_like(t)
    s0 = ssim(p, t).item()
    assert ssim(p + 5.0, t + 5.0).item() == pytest.approx(s0, abs=1e-4)
    assert ssim(3.0*p - 1.0, 3.0*t - 1.0).item() == pytest.approx(s0, abs=1e-4)

def test_ssim_drops_with_noise():
    t = torch.from_numpy(np.stack([smooth_field(0)]))
    assert ssim(t + 0.5*torch.randn_like(t), t).item() < ssim(t + 0.05*torch.randn_like(t), t).item() < 1.0

#  coarse/fine alignment 

@pytest.mark.parametrize("axis", [-2, -1])
def test_coarse_input_is_aligned_with_fine_grid(axis):
    # Block averaging + bilinear upsampling must reproduce a linear field away from the walls.
    lin = torch.linspace(0, 1, N)
    field = (lin[:, None] if axis == -2 else lin[None, :]).expand(N, N)[None, None].clone()
    err = (make_coarse_field(field) - field)[..., 2:-2, 2:-2].abs().max().item()
    assert err < 1e-5

#  solver

def test_poisson_solve_is_exact():
    rng = np.random.default_rng(0)
    b = np.zeros((N, N)); b[1:-1, 1:-1] = rng.standard_normal((N-2, N-2))
    b[1:-1, 1:-1] -= b[1:-1, 1:-1].mean()
    p = solve_pressure_poisson(b, DX, DX)
    lap = ns.laplacian(p, DX, DX)
    np.testing.assert_allclose(lap[1:-1, 1:-1], b[1:-1, 1:-1], atol=1e-8)

def test_primary_vortex_of_analytic_streamfunction():
    # ψ = -sin²(πx) sin²(πy)/8 has its minimum -1/8 at the centre; u = dψ/dy.
    x = np.linspace(0, 1, N); X, Y = np.meshgrid(x, x, indexing="ij")
    u = -np.sin(np.pi*X)**2 * 2*np.pi*np.sin(np.pi*Y)*np.cos(np.pi*Y) / 8
    psi_min, xc, yc = ns.primary_vortex(u, DX, DX)
    assert psi_min == pytest.approx(-0.125, rel=2e-3)
    assert (xc, yc) == pytest.approx((0.5, 0.5), abs=2e-3)

#  data split-

def test_split_holds_out_whole_re_values():
    re = np.repeat([100, 400, 600, 1000, 2000, 3200], 5)
    train, val, held = split_by_re(re, val_re=[600, 2000], max_train_re=1500)
    assert set(re[train]) == {100, 400, 1000}
    assert set(re[val]) == {600}
    assert set(re[held]) == {2000, 3200}
    assert not (train & val).any() and not (train & held).any() and not (val & held).any()

def test_stats_come_from_training_split_only(tmp_path):
    for Re, level in ((100, 0.0), (200, 0.0), (600, 50.0)):
        d = tmp_path / f"Re_{Re}"; d.mkdir()
        for k in range(3):
            np.save(d / f"snap_{k:05d}.npy", (smooth_field(k) + level).astype(np.float32))
    _, val_loader, _, st = make_dataloaders(str(tmp_path), val_re=[600], batch_size=2)
    assert st["train_re"] == [100, 200] and st["val_re"] == [600]
    assert abs(st["mean"][0]) < 5.0          # the +50 offset of the validation Re is not in the stats
    assert len(val_loader.dataset) == 3

#  model and metrics 

def test_untrained_model_returns_its_input():
    model = TurbulenceUNet(base_ch=8).eval()
    x = torch.randn(2, 4, N, N)
    with torch.no_grad():
        pred, _ = model(x)
    torch.testing.assert_close(pred, x)

def test_metric_and_loss_names_do_not_collide(stats):
    # train.run_epoch merges both dicts; a shared key once replaced mean|div| with the loss.
    x = to_batch([smooth_field(0)], stats)
    assert not set(sample_metrics(x, x, stats)) & set(LOSS_KEYS)

def test_perfect_prediction_has_zero_error(stats):
    x = to_batch([smooth_field(0), smooth_field(1)], stats)
    m = sample_metrics(x, x, stats)
    assert m["rel_l2_vel"].abs().max().item() < 1e-6
    assert m["ssim"].min().item() == pytest.approx(1.0, abs=1e-5)
