"""Tests for the periodic (turbulence) path. Run with:  python -m pytest tests"""
import os, sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver import turbulence as T
from data.augmentation import make_coarse_field, mirror
from data.dataset import normalise
from models.losses import TurbulenceLoss
from models.unet import TurbulenceUNet
from train.metrics import energy_spectrum, sample_metrics

N_DNS, N = 64, 32

@pytest.fixture(scope="module")
def turb_snapshot():
    """A decaying-turbulence snapshot from the real solver, truncated to NxN."""
    sp = T.Spectral(N_DNS); cfg = T.TurbConfig(500, "decaying", N_DNS, seed=3); cfg.k0 = 4
    w = T.initial_vorticity(sp, cfg)
    w, _ = T.advance(sp, T.make_stepper(sp, cfg), w, cfg, 0.5)
    return T.snapshot(sp, w, N)

@pytest.fixture
def stats():
    return {"mean": [0.01, -0.02, 0.0, 0.05], "std": [0.7, 0.6, 0.3, 3.0],
            "log_re_mean": 3.0, "log_re_std": 0.3, "domain": "periodic", "L": T.L_BOX}

def batch(fields, stats):
    return torch.from_numpy(np.stack([normalise(f, stats) for f in fields])).float()

def test_physics_losses_vanish_on_solver_fields(turb_snapshot, stats):
    x = batch([turb_snapshot], stats)
    crit = TurbulenceLoss(stats)
    assert crit.vorticity_loss(x).item() < 1e-8
    assert crit.divergence_loss(x).item() < 1e-8

@pytest.mark.parametrize("axis", ["x", "y"])
def test_mirrored_turbulence_stays_consistent(turb_snapshot, stats, axis):
    x = batch([mirror(turb_snapshot, axis)], stats)
    assert TurbulenceLoss(stats).vorticity_loss(x).item() < 1e-8

def test_spectra_satisfy_parseval(turb_snapshot):
    u, v = turb_snapshot[0], turb_snapshot[1]
    E_torch = energy_spectrum(torch.from_numpy(turb_snapshot[None])).sum().item()
    assert E_torch == pytest.approx(0.5 * np.mean(u**2 + v**2), rel=1e-5)
    sp = T.Spectral(N_DNS); cfg = T.TurbConfig(500, "decaying", N_DNS, seed=1)
    w = T.initial_vorticity(sp, cfg)
    uh, vh = sp.velocity_hat(w)
    _, E = T.energy_spectrum(w, sp)
    assert E.sum() == pytest.approx(0.5 * np.mean(sp.ifft(uh)**2 + sp.ifft(vh)**2), rel=1e-6)

def test_periodic_coarsening_commutes_with_shifts():
    x = torch.randn(1, 4, N, N)
    s = 4   # a multiple of the coarsening factor
    a = make_coarse_field(torch.roll(x, (s, 2*s), (2, 3)), periodic=True)
    b = torch.roll(make_coarse_field(x, periodic=True), (s, 2*s), (2, 3))
    torch.testing.assert_close(a, b, atol=1e-5, rtol=0)

def test_periodic_unet_is_shift_equivariant():
    model = TurbulenceUNet(base_ch=8, periodic=True).eval()
    for p in model.output_conv.parameters():
        torch.nn.init.normal_(p, std=0.1)          # make the residual branch non-trivial
    x = torch.randn(1, 4, N, N)
    with torch.no_grad():
        a = model(torch.roll(x, (8, 16), (2, 3)))[0]
        b = torch.roll(model(x)[0], (8, 16), (2, 3))
    torch.testing.assert_close(a, b, atol=1e-5, rtol=0)

def test_solver_conserves_energy_without_viscosity():
    sp = T.Spectral(N_DNS); cfg = T.TurbConfig(10**9, "decaying", N_DNS, seed=0); cfg.k0 = 4
    w0 = T.initial_vorticity(sp, cfg)
    w1, _ = T.advance(sp, T.make_stepper(sp, cfg), w0, cfg, 0.5)
    E0 = T.energy_spectrum(w0, sp)[1].sum(); E1 = T.energy_spectrum(w1, sp)[1].sum()
    assert abs(E1 / E0 - 1) < 1e-4

def test_spectral_metrics_are_perfect_for_perfect_prediction(turb_snapshot, stats):
    x = batch([turb_snapshot], stats)
    m = sample_metrics(x, x, stats)
    assert m["spec_err"].item() < 1e-6 and m["hf_energy"].item() == pytest.approx(1.0, abs=1e-6)

def test_runs_at_different_re_do_not_share_initial_conditions():
    sp = T.Spectral(N_DNS)
    w = [T.initial_vorticity(sp, T.TurbConfig(Re, "decaying", N_DNS, seed=0)) for Re in (500, 1000)]
    corr = np.abs(np.vdot(w[0], w[1])) / (np.linalg.norm(w[0]) * np.linalg.norm(w[1]))
    assert corr < 0.2

def _proj_model(stats):
    vs = (stats["mean"][0], stats["std"][0], stats["mean"][1], stats["std"][1])
    return TurbulenceUNet(base_ch=8, periodic=True, project=True, vel_stats=vs).eval()

def test_projection_makes_velocity_divergence_free_and_never_hurts(turb_snapshot, stats):
    model = _proj_model(stats)
    true = batch([turb_snapshot], stats)
    noisy = true + 0.3 * torch.randn_like(true)
    proj = model.project_divergence_free(noisy)
    crit = TurbulenceLoss(stats)
    assert crit.divergence_loss(proj).item() < 1e-8
    err = lambda x: (x[:, :2] - true[:, :2]).norm().item()
    assert err(proj) < err(noisy)
    torch.testing.assert_close(model.project_divergence_free(proj), proj, atol=1e-5, rtol=0)   # idempotent
    torch.testing.assert_close(proj[:, 2:], noisy[:, 2:])                                      # p, ω untouched

def test_projected_unet_is_still_shift_equivariant(stats):
    model = _proj_model(stats)
    for p in model.output_conv.parameters():
        torch.nn.init.normal_(p, std=0.1)
    x = torch.randn(1, 4, N, N)
    with torch.no_grad():
        a = model(torch.roll(x, (8, 16), (2, 3)))[0]
        b = torch.roll(model(x)[0], (8, 16), (2, 3))
    torch.testing.assert_close(a, b, atol=1e-5, rtol=0)

def test_relative_loss_weights_weak_and_strong_fields_equally(turb_snapshot, stats):
    crit = TurbulenceLoss(stats, recon="relative")
    strong = turb_snapshot; weak = turb_snapshot * 0.1
    for f in (strong, weak):
        t = batch([f], stats)
        assert crit.reconstruction_loss(t, t).item() < 1e-12
    # the same 10% error on each field gives the same loss, unlike MSE
    loss = lambda f: crit.reconstruction_loss(batch([1.1 * f], stats), batch([f], stats)).item()
    assert loss(weak) == pytest.approx(loss(strong), rel=1e-4)
    mse = TurbulenceLoss(stats, recon="mse")
    mse_loss = lambda f: mse.reconstruction_loss(batch([1.1 * f], stats), batch([f], stats)).item()
    assert mse_loss(weak) < 0.05 * mse_loss(strong)

def test_old_checkpoint_args_build_a_model_without_projection(stats):
    from train.train import build_model
    old = {"base_ch": 8, "dropout_p": 0.1}                # no "no_project" key
    new = {"base_ch": 8, "dropout_p": 0.1, "no_project": False}
    assert not build_model(old, stats).project and build_model(new, stats).project

def test_spectral_loss_detects_missing_fine_scales(turb_snapshot, stats):
    crit = TurbulenceLoss(stats, lambda_spec=1.0, coarse_factor=4)
    true = batch([turb_snapshot], stats)
    assert crit.spectral_loss(true, true).item() < 1e-10
    smooth = make_coarse_field(true, periodic=True).requires_grad_(True)    # fine scales smoothed away
    loss = crit.spectral_loss(smooth, true)
    assert loss.item() > 0.1
    loss.backward()
    assert smooth.grad is not None and torch.isfinite(smooth.grad).all() and smooth.grad.abs().sum() > 0

def test_spectral_loss_is_off_by_default_and_refused_for_the_cavity(turb_snapshot, stats):
    true = batch([turb_snapshot], stats)
    assert TurbulenceLoss(stats).spectral_loss(true + 1.0, true).item() == 0.0
    with pytest.raises(ValueError):
        TurbulenceLoss({**stats, "domain": "cavity"}, lambda_spec=0.1)
