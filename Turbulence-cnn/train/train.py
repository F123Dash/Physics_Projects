import os, sys, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import config
from data.dataset  import make_dataloaders, denormalise_re, is_periodic
from models.unet   import TurbulenceUNet, count_parameters
from models.losses import TurbulenceLoss
from train.metrics import (sample_metrics, re_metrics, baseline_prediction,
                           MetricAccumulator, format_table)

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--flow",choices=sorted(config.FLOWS),default=config.DEFAULT_FLOW,
                   help="Dataset preset from config.FLOWS (sets --data_dir, --test_dir, --val_re defaults)")
    p.add_argument("--data_dir",nargs="+",default=None,help="Training snapshot directories")
    p.add_argument("--test_dir",nargs="+",default=None,help="Unseen-Re snapshot directories for the final test (skipped if missing)")
    p.add_argument("--val_re",type=int,nargs="+",default=None,help="Re values held out for validation")
    p.add_argument("--max_train_re",type=int,default=None,help="Train only on Re <= this; higher Re in --data_dir join the test set")
    p.add_argument("--out_dir",default=None,help="Default: runs/<flow>_01")
    p.add_argument("--resume",default=None)
    p.add_argument("--epochs",type=int,   default=150)
    p.add_argument("--batch_size",type=int,   default=16)
    p.add_argument("--lr",type=float, default=1e-3)
    p.add_argument("--lr_min",type=float, default=1e-5)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--lambda_div",type=float, default=0.01)
    p.add_argument("--lambda_vort",type=float, default=0.01)
    p.add_argument("--lambda_re",type=float, default=0.01)
    p.add_argument("--noise_std",type=float, default=None,help="Input noise on training samples (default: per flow, config.FLOWS)")
    p.add_argument("--recon",choices=["mse","relative"],default=None,help="Reconstruction loss (default: per flow, config.FLOWS)")
    p.add_argument("--lambda_spec",type=float,default=0.0,help="Weight of the fine-scale spectral loss (periodic data only; 0 = off)")
    p.add_argument("--no_project",action="store_true",help="Periodic data: do not make the predicted velocity divergence-free")
    p.add_argument("--grad_clip",type=float, default=1.0)
    p.add_argument("--base_ch",type=int,   default=64)
    p.add_argument("--dropout_p",type=float, default=0.1)
    p.add_argument("--seed",type=int,   default=67)
    p.add_argument("--num_workers",type=int,   default=0)
    args = p.parse_args()
    flow = config.FLOWS[args.flow]
    args.data_dir = args.data_dir or flow["train_dirs"]
    args.test_dir = args.test_dir or flow["test_dirs"]
    args.val_re   = args.val_re   or flow["val_re"]
    args.noise_std = flow["noise_std"] if args.noise_std is None else args.noise_std
    args.recon     = args.recon or flow["recon"]
    args.out_dir  = args.out_dir  or os.path.join(config.RUNS_ROOT, f"{args.flow}_01")
    return args


def init_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}"); return seed

LOSS_KEYS = ["loss", "recon", "div", "vort", "re", "spec"]

def run_epoch(model, loader, criterion, device, stats, optimizer=None, grad_clip=0.0, epoch=None):
    """One pass over loader. Trains if optimizer is given, otherwise evaluates."""
    train = optimizer is not None
    model.train(train)
    loss_sums = dict.fromkeys(LOSS_KEYS, 0.0)
    acc = MetricAccumulator()
    n = len(loader)
    with torch.set_grad_enabled(train):
        for bi, (coarse, fine, re_true) in enumerate(loader):
            coarse = coarse.to(device); fine = fine.to(device); re_true = re_true.to(device)
            pred_field, re_pred = model(coarse)
            losses = criterion(pred_field, fine, re_pred, re_true)
            if train:
                optimizer.zero_grad(); losses[0].backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            B = fine.shape[0]
            for k, v in zip(LOSS_KEYS, losses):
                loss_sums[k] += v.item() * B
            acc.add({**sample_metrics(pred_field.detach(), fine, stats),
                     **re_metrics(re_pred.detach(), re_true, stats)})
            if train and ((bi+1) % 20 == 0 or bi == n-1):
                print(f"  Ep {epoch:3d}  batch {bi+1:4d}/{n}  loss={losses[0].item():.4f}"
                      f"  recon={losses[1].item():.4f}  vort={losses[3].item():.4f}")
    out = acc.result()
    out.update({k: v / max(acc.n, 1) for k, v in loss_sums.items()})
    return out

@torch.no_grad()
def evaluate_baselines(loader, device, stats):
    """Bilinear (the model's own input, without noise) and bicubic interpolation."""
    res = {}
    for mode in ("bilinear", "bicubic"):
        acc = MetricAccumulator()
        for _, fine, _ in loader:
            fine = fine.to(device)
            acc.add(sample_metrics(baseline_prediction(fine, mode, stats), fine, stats))
        res[mode] = acc.result()
    return res

def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"  Checkpoint  {path}")

def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"Loaded {path}  (epoch {ckpt['epoch']}, best val score={ckpt.get('best_val_score', ckpt.get('best_val_loss')):.4f})")
    return ckpt

def plot_loss_curves(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Training dashboard — super-resolution U-Net",
                 fontsize=13, fontweight="bold")
    panels = [
        (axes[0,0], "loss",       "Total loss",                    "Loss"),
        (axes[0,1], "recon",      "Reconstruction loss",           "loss"),
        (axes[0,2], "div",        "Divergence loss",               "(1/s)²"),
        (axes[1,0], "vort",       "Vorticity consistency",         "sigma_ω²"),
        (axes[1,1], "rel_l2_vel", "Relative L2 error, u and v",    "rel L2"),
        (axes[1,2], "ssim",       "SSIM",                          "SSIM"),
    ]
    if "val_spec_err" in history:   # periodic data: the fine-scale spectrum says more than SSIM
        panels[-1] = (axes[1,2], "spec_err", "Fine-scale spectral error", "mean |log10 E_pred/E_true|")
    for ax, key, title, ylabel in panels:
        ax.plot(epochs, history[f"train_{key}"], "b-",  lw=1.5, label="Train")
        ax.plot(epochs, history[f"val_{key}"],   "r--", lw=1.5, label="Val (held-out Re)")
        ax.set_title(title, fontsize=11); ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fpath = os.path.join(out_dir, "training_curves.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")


def plot_prediction_panel(model, loader, device, out_dir, epoch, stats):
    model.eval()
    coarse, fine, re_true = next(iter(loader))
    coarse = coarse.to(device); fine = fine.to(device)
    with torch.no_grad():
        pred, re_pred = model(coarse)
    re_t = denormalise_re(re_true[0].item(), stats); re_p = denormalise_re(re_pred[0].item(), stats)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(f"Epoch {epoch}  |  True Re={re_t:.0f}  Predicted Re={re_p:.0f}",
                 fontsize=12, fontweight="bold")
    for row, (ch, ch_name) in enumerate([(0,"u-velocity"),(3,"vorticity ω")]):
        c_np = coarse[0,ch].cpu().numpy(); f_np = fine[0,ch].cpu().numpy()
        p_np = pred[0,ch].cpu().numpy();   e_np = np.abs(f_np - p_np)
        vmax = max(np.abs(f_np).max(), 1e-8); emax = e_np.max() + 1e-8
        for col, (field, title) in enumerate([
            (c_np, f"Coarse — {ch_name}"),
            (f_np, f"Target — {ch_name}"),
            (p_np, f"Pred   — {ch_name}"),
            (e_np, f"|error| — {ch_name}"),
        ]):
            ax = axes[row, col]
            cmap = "RdBu_r" if col < 3 else "hot"
            # Fields are indexed [x, y]; transpose so x is horizontal and the lid is on top.
            im = ax.imshow(field.T, cmap=cmap, origin="lower",vmin=-vmax if col<3 else 0,vmax=vmax  if col<3 else emax)
            ax.set_title(title, fontsize=9); ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fpath = os.path.join(out_dir, f"prediction_ep{epoch:03d}.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")

@torch.no_grad()
def plot_re_scatter(model, loader, device, out_dir, stats, name):
    model.eval()
    t_all, p_all = [], []
    for coarse, _, re_true in loader:
        _, re_pred = model(coarse.to(device))
        t_all.append(denormalise_re(re_true.double().numpy(), stats))
        p_all.append(denormalise_re(re_pred.double().cpu().numpy(), stats))
    t_all = np.concatenate(t_all); p_all = np.concatenate(p_all)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t_all, p_all, s=10, alpha=0.6)
    lim = [min(t_all.min(), p_all.min()) * 0.8, max(t_all.max(), p_all.max()) * 1.2]
    ax.plot(lim, lim, "k--", lw=1); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True Re"); ax.set_ylabel("Predicted Re")
    ax.set_title(f"Re regression — {name}", fontsize=12, fontweight="bold"); ax.grid(True, which="both", alpha=0.3)
    fpath = os.path.join(out_dir, f"re_scatter_{name}.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")

def report(name, model_m, baselines):
    print(format_table({"CNN": model_m, "bilinear input": baselines["bilinear"],
                        "bicubic": baselines["bicubic"]}, title=f"\n{name}"))
    print(f"  (ground-truth mean|div| = {model_m['abs_div_true']:.4f})")

def build_model(args: dict, stats: dict) -> TurbulenceUNet:
    """The U-Net for a dataset; args is a dict (vars(args) or a checkpoint's "args")."""
    periodic = is_periodic(stats)
    project  = periodic and not args.get("no_project", True)   # older checkpoints: no projection
    return TurbulenceUNet(in_ch=4, out_ch=4, base_ch=args["base_ch"], dropout_p=args["dropout_p"],
                          periodic=periodic, project=project,
                          vel_stats=(stats["mean"][0], stats["std"][0], stats["mean"][1], stats["std"][1]))

def main():
    args   = get_args()
    seed   = init_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}  |  Out: {args.out_dir}")
    train_loader, val_loader, test_loader, stats = make_dataloaders(
        data_dirs=args.data_dir, val_re=args.val_re, test_dirs=args.test_dir, max_train_re=args.max_train_re,
        batch_size=args.batch_size, num_workers=args.num_workers, seed=seed, noise_std=args.noise_std)
    model = build_model(vars(args), stats).to(device)
    print(f"Model parameters: {count_parameters(model):,}  periodic={model.periodic}  divergence-free projection={model.project}")
    print(f"Reconstruction loss: {args.recon}   input noise: {args.noise_std}   spectral loss weight: {args.lambda_spec}")
    criterion = TurbulenceLoss(stats, lambda_div=args.lambda_div, lambda_vort=args.lambda_vort,
                               lambda_re=args.lambda_re, recon=args.recon, lambda_spec=args.lambda_spec,
                               coarse_factor=config.COARSE_FACTOR).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    start_epoch    = 1
    best_val_score = float("inf")
    history = {}
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        if ckpt["stats"] != stats:
            raise ValueError("Checkpoint normalisation stats differ from this dataset; resume with the same --data_dir/--val_re")
        start_epoch   = ckpt["epoch"] + 1
        best_val_score = ckpt.get("best_val_score", float("inf"))
        history       = ckpt.get("history", {})
    best_ckpt_path = os.path.join(args.out_dir, "best_model.pt")
    last_ckpt_path = os.path.join(args.out_dir, "last_model.pt")
    print(f"\nTraining {args.epochs} epochs (start={start_epoch})")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, criterion, device, stats, optimizer, args.grad_clip, epoch)
        val_m   = run_epoch(model, val_loader, criterion, device, stats)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        for k in LOSS_KEYS + [k for k in ("rel_l2_vel", "rel_l2_all", "ssim", "re_rel_err", "spec_err", "hf_energy") if k in val_m]:
            history.setdefault(f"train_{k}", []).append(train_m[k])
            history.setdefault(f"val_{k}", []).append(val_m[k])
        history.setdefault("lr", []).append(lr_now)
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch:3d}/{args.epochs}  [{elapsed:.1f}s]")
        for name, m in (("Train", train_m), ("Val", val_m)):
            print(f"  {name:<5}  loss={m['loss']:.4f}  recon={m['recon']:.4f}  vort={m['vort']:.4f}"
                  f"  relL2(u,v)={m['rel_l2_vel']:.2%}  relL2(all)={m['rel_l2_all']:.2%}  ssim={m['ssim']:.4f}  Re err={m['re_rel_err']:.1%}"
                  + (f"  spec err={m['spec_err']:.3f}  HF energy={m['hf_energy']:.0%}" if "spec_err" in m else ""))
        print(f"  LR={lr_now:.2e}")
        # Select on field accuracy only: the total loss includes the Re regression term,
        # whose noise decided the checkpoint in the first turbulence run.
        is_best = val_m["rel_l2_all"] < best_val_score
        if is_best:
            best_val_score = val_m["rel_l2_all"]
        ckpt = dict(epoch=epoch, model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    best_val_score=best_val_score, best_val_loss=val_m["loss"], history=history,
                    args=vars(args), stats=stats)
        if is_best:
            save_checkpoint(ckpt, best_ckpt_path)
        save_checkpoint(ckpt, last_ckpt_path)
        if epoch % 10 == 0 or epoch == args.epochs:
            plot_loss_curves(history, args.out_dir)
            plot_prediction_panel(model, val_loader, device, args.out_dir, epoch, stats)
    print("\nLoading best checkpoint (lowest validation relative L2, mean over u, v, p, ω)...")
    load_checkpoint(best_ckpt_path, model, device=device)
    val_m = run_epoch(model, val_loader, criterion, device, stats)
    report(f"Validation — held-out Re {stats['val_re']}", val_m, evaluate_baselines(val_loader, device, stats))
    plot_re_scatter(model, val_loader, device, args.out_dir, stats, "val")
    if test_loader is not None:
        test_m = run_epoch(model, test_loader, criterion, device, stats)
        report(f"Test — unseen Re {sorted(set(test_loader.dataset.re.tolist()))}", test_m, evaluate_baselines(test_loader, device, stats))
        plot_re_scatter(model, test_loader, device, args.out_dir, stats, "test")
        print("  Per-Re test results: python train/evaluate.py --checkpoint", best_ckpt_path)
    plot_loss_curves(history, args.out_dir)
    np.save(os.path.join(args.out_dir, "history.npy"), history)
    print(f"\nTraining complete.  Best val relL2 (all channels): {best_val_score:.2%}  Outputs: {args.out_dir}/")

if __name__ == "__main__":
    main()
