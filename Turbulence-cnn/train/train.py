import os, sys, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.ndimage import zoom as scipy_zoom
from data.dataset  import make_dataloaders, N_CLASSES, RE_VALUES
from models.unet   import TurbulenceUNet, count_parameters
from models.losses import TurbulenceLoss, ssim as ssim_metric

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir",default="snapshots")
    p.add_argument("--out_dir",default="runs/exp_01")
    p.add_argument("--resume",default=None)
    p.add_argument("--epochs",type=int,   default=150)  # FIX-4
    p.add_argument("--batch_size",type=int,   default=16)
    p.add_argument("--lr",type=float, default=1e-3)
    p.add_argument("--lr_min",type=float, default=1e-5)
    p.add_argument("--lambda_div",type=float, default=0.01)
    p.add_argument("--lambda_vort",type=float, default=0.001)  # FIX-2
    p.add_argument("--lambda_cls",type=float, default=0.01)
    p.add_argument("--grad_clip",type=float, default=1.0)
    p.add_argument("--base_ch",type=int,   default=64)
    p.add_argument("--dropout_p",type=float, default=0.1)
    p.add_argument("--seed",type=int,   default=None)
    p.add_argument("--num_workers",type=int,   default=0)
    return p.parse_args()


def init_seed(seed):
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**32 - 1))
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}"); return seed

def relative_l2(pred, true):
    diff  = (pred - true).reshape(pred.shape[0], -1)
    denom = true.reshape(true.shape[0], -1).norm(dim=1).clamp(min=1e-8)
    return (diff.norm(dim=1) / denom).mean().item()


def mean_divergence_physical(pred, stats, dx=1.0/64):
    sigma_u = float(stats["std"][0])
    sigma_v = float(stats["std"][1])
    u_norm = pred[:, 0]; v_norm = pred[:, 1]
    du_dx = (u_norm[:, 2:, 1:-1] - u_norm[:, :-2, 1:-1]) / (2*dx) * sigma_u
    dv_dy = (v_norm[:, 1:-1, 2:] - v_norm[:, 1:-1, :-2]) / (2*dx) * sigma_v
    return (du_dx + dv_dy).abs().mean().item()


def classification_accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


def bicubic_upsample(coarse_batch: torch.Tensor) -> torch.Tensor:
    c_np = coarse_batch.cpu().numpy()
    B, C, H, W = c_np.shape
    fine_size = H
    coarse_size = 16
    factor = fine_size // coarse_size
    out = np.zeros_like(c_np)
    for b in range(B):
        for ch in range(C):
            small = c_np[b, ch].reshape(
                coarse_size, factor, coarse_size, factor).mean(axis=(1,3))
            out[b, ch] = scipy_zoom(small, factor, order=3)
    return torch.from_numpy(out.astype(np.float32))

def train_one_epoch(model, loader, criterion, optimizer,
                    grad_clip, device, epoch, stats):
    model.train()
    totals = dict(loss=0, recon=0, div=0, vort=0, cls=0,
                  rel_l2=0, mean_div=0, ssim=0, acc=0)
    n = len(loader)
    for bi, (coarse, fine, labels) in enumerate(loader):
        coarse = coarse.to(device); fine = fine.to(device)
        labels = labels.to(device)
        pred_field, logits = model(coarse)
        total, recon, div_l, vort_l, cls_l = criterion(
            pred_field, fine, logits, labels)
        optimizer.zero_grad(); total.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        with torch.no_grad():
            totals["loss"]     += total.item()
            totals["recon"]    += recon.item()
            totals["div"]      += div_l.item()
            totals["vort"]     += vort_l.item()
            totals["cls"]      += cls_l.item()
            totals["rel_l2"]   += relative_l2(pred_field, fine)
            totals["mean_div"] += mean_divergence_physical(pred_field, stats)
            totals["ssim"]     += ssim_metric(pred_field.detach(), fine.detach()).item()
            totals["acc"]      += classification_accuracy(logits, labels)
        if (bi+1) % 20 == 0 or bi == n-1:
            print(f"  Ep {epoch:3d}  batch {bi+1:4d}/{n}"
                  f"  loss={total.item():.4f}"
                  f"  recon={recon.item():.4f}"
                  f"  vort={vort_l.item():.4f}"
                  f"  acc={classification_accuracy(logits,labels)*100:.1f}%")
    return {k: v/n for k,v in totals.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, stats):
    model.eval()
    totals = dict(loss=0, recon=0, div=0, vort=0, cls=0,
                  rel_l2=0, mean_div=0, ssim=0, acc=0)
    n = len(loader)
    for coarse, fine, labels in loader:
        coarse = coarse.to(device); fine = fine.to(device)
        labels = labels.to(device)
        pred_field, logits = model(coarse)
        total, recon, div_l, vort_l, cls_l = criterion(
            pred_field, fine, logits, labels)
        totals["loss"]     += total.item()
        totals["recon"]    += recon.item()
        totals["div"]      += div_l.item()
        totals["vort"]     += vort_l.item()
        totals["cls"]      += cls_l.item()
        totals["rel_l2"]   += relative_l2(pred_field, fine)
        totals["mean_div"] += mean_divergence_physical(pred_field, stats)
        totals["ssim"]     += ssim_metric(pred_field, fine).item()
        totals["acc"]      += classification_accuracy(logits, labels)
    return {k: v/n for k,v in totals.items()}

def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"  Checkpoint  {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"Resumed from {path}  "
          f"(epoch {ckpt['epoch']}, val_loss={ckpt['best_val_loss']:.4f})")
    return ckpt["epoch"], ckpt["best_val_loss"]

def plot_loss_curves(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Training dashboard — Turbulence U-Net",
                 fontsize=13, fontweight="bold")
    panels = [
        (axes[0,0], "train_loss",  "val_loss",  "Total loss",           "Loss"),
        (axes[0,1], "train_recon", "val_recon", "Reconstruction (MSE)", "MSE"),
        (axes[0,2], "train_div",   "val_div",   "Divergence (physical)", "m/s per m"),
        (axes[1,0], "train_vort",  "val_vort",  "Vorticity loss",       "Penalty"),
        (axes[1,1], "train_acc",   "val_acc",   "Classifier accuracy",  "Accuracy"),
        (axes[1,2], "train_ssim",  "val_ssim",  "SSIM",                 "SSIM"),
    ]
    for ax, tk, vk, title, ylabel in panels:
        if history.get(tk):
            ax.plot(epochs, history[tk], "b-",  lw=1.5, label="Train")
            ax.plot(epochs, history[vk], "r--", lw=1.5, label="Val")
        ax.set_title(title, fontsize=11); ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fpath = os.path.join(out_dir, "training_curves.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    pdf_path = os.path.splitext(fpath)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")


def plot_prediction_panel(model, loader, device, out_dir, epoch):
    re_names = {i: f"Re={re}" for i, re in enumerate(RE_VALUES)}
    model.eval()
    coarse, fine, label = next(iter(loader))
    coarse = coarse.to(device); fine = fine.to(device)
    with torch.no_grad():
        pred, logits = model(coarse)
    true_re = re_names.get(label[0].item(), "?")
    pred_re = re_names.get(logits[0].argmax().item(), "?")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(f"Epoch {epoch}  |  True: {true_re}  Pred: {pred_re}",
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
            im = ax.imshow(field, cmap=cmap, origin="lower",vmin=-vmax if col<3 else 0,vmax=vmax  if col<3 else emax)
            ax.set_title(title, fontsize=9); ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fpath = os.path.join(out_dir, f"prediction_ep{epoch:03d}.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    pdf_path = os.path.splitext(fpath)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")


def plot_confusion_matrix(model, loader, device, out_dir):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for coarse, fine, labels in loader:
            _, logits = model(coarse.to(device))
            all_true.extend(labels.numpy())
            all_pred.extend(logits.argmax(1).cpu().numpy())
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(all_true, all_pred):
        if 0 <= t < N_CLASSES and 0 <= p < N_CLASSES: cm[t,p] += 1
    class_names = [f"Re={re}" for re in RE_VALUES]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted Re"); ax.set_ylabel("True Re")
    ax.set_title("Re-regime confusion matrix", fontsize=12, fontweight="bold")
    cm_max = cm.max() if cm.max() > 0 else 1
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i,j] > cm_max/2 else "black")
    plt.colorbar(im, ax=ax); plt.tight_layout()
    fpath = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    pdf_path = os.path.splitext(fpath)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")

@torch.no_grad()
def evaluate_bicubic_baseline(loader, device):
    totals = dict(mse=0.0, rel_l2=0.0, ssim=0.0)
    n = len(loader)
    for coarse, fine, _ in loader:
        bic = bicubic_upsample(coarse).to(device)
        fine = fine.to(device)
        totals["mse"]    += nn.functional.mse_loss(bic, fine).item()
        diff  = (bic - fine).reshape(bic.shape[0], -1)
        denom = fine.reshape(fine.shape[0], -1).norm(dim=1).clamp(min=1e-8)
        totals["rel_l2"] += (diff.norm(dim=1)/denom).mean().item()
        totals["ssim"]   += ssim_metric(bic, fine).item()
    return {k: v/n for k,v in totals.items()}

def print_comparison_table(model_m, bicubic_m):
    print(f"\n{'='*58}")
    print(f"{'Metric':<20} {'Bicubic':>14} {'CNN model':>14} {'Δ':>8}")
    for key, fmt, better in [("mse",    ".4f", "lower"),("rel_l2", ".1%", "lower"),
                             ("ssim",   ".4f", "higher"),]:
        bv = bicubic_m[key]; mv = model_m.get(key, model_m.get("recon", 0))
        if key == "mse":   mv = model_m.get("recon", 0)
        if key == "rel_l2": mv = model_m.get("rel_l2", 0)
        if key == "ssim":   mv = model_m.get("ssim", 0)
        imp = mv - bv if better == "lower" else bv - mv
        sign = "high" if (better=="lower" and mv < bv) else ("low" if (better=="higher" and mv > bv) else "—")
        print(f"  {key:<18} {bv:>14{fmt}} {mv:>14{fmt}} {sign} {abs(imp):{fmt}}")

def main():
    args   = get_args()
    seed   = init_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}  |  Out: {args.out_dir}")
    print(f"Key hyperparameters:")
    print(f"  epochs={args.epochs}  λ_vort={args.lambda_vort} "f"λ_div={args.lambda_div}  lr={args.lr} {args.lr_min}")
    result = make_dataloaders(data_dir= args.data_dir,batch_size= args.batch_size,num_workers = args.num_workers,
                              seed= seed,)
    train_loader, val_loader, test_loader, stats, class_weights, stats_path = result
    print(f"Stats path: {stats_path}")
    model = TurbulenceUNet(in_ch=4, out_ch=4, base_ch=args.base_ch,n_classes=N_CLASSES, dropout_p=args.dropout_p,).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    criterion = TurbulenceLoss(
        lambda_div   = args.lambda_div,
        lambda_vort  = args.lambda_vort,
        lambda_cls   = args.lambda_cls,
        n_channels   = 4,
        class_weights= class_weights.to(device),
        sigma_u      = float(stats["std"][0]),   # FIX-3
        sigma_v      = float(stats["std"][1]),   # FIX-3
    )
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    start_epoch   = 1
    best_val_loss = float("inf")
    best_val_ssim = -float("inf")   # FIX-1: track best SSIM too
    if args.resume and os.path.exists(args.resume):
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler)
        start_epoch += 1
    history = {k: [] for k in [
        "train_loss", "train_recon", "train_div", "train_vort",
        "train_cls",  "train_acc",   "train_ssim",
        "val_loss",   "val_recon",   "val_div",   "val_vort",
        "val_cls",    "val_acc",     "val_ssim",  "lr",
    ]}
    best_ckpt_path      = os.path.join(args.out_dir, "best_model.pt")
    best_ssim_ckpt_path = os.path.join(args.out_dir, "best_ssim_model.pt")  # FIX-1
    last_ckpt_path      = os.path.join(args.out_dir, "last_model.pt")
    print(f"\nTraining {args.epochs} epochs (start={start_epoch})")
    print(f"  Saves: best_model.pt (val_loss) + best_ssim_model.pt (val_ssim)")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion,
                                  optimizer, args.grad_clip, device, epoch, stats)
        val_m   = validate(model, val_loader, criterion, device, stats)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        for k in ["loss","recon","div","vort","cls","acc","ssim"]:
            history[f"train_{k}"].append(train_m[k])
            history[f"val_{k}"].append(val_m[k])
        history["lr"].append(lr_now)
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch:3d}/{args.epochs}  [{elapsed:.1f}s]")
        print(f"  Train  loss={train_m['loss']:.4f}"
              f"  recon={train_m['recon']:.4f}"
              f"  vort={train_m['vort']:.4f}"
              f"  ssim={train_m['ssim']:.4f}"
              f"  acc={train_m['acc']*100:.1f}%")
        print(f"  Val    loss={val_m['loss']:.4f}"
              f"  recon={val_m['recon']:.4f}"
              f"  vort={val_m['vort']:.4f}"
              f"  ssim={val_m['ssim']:.4f}"
              f"  acc={val_m['acc']*100:.1f}%")
        print(f"  LR={lr_now:.2e}")
        ckpt = dict(epoch=epoch, model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    best_val_loss=best_val_loss,
                    best_val_ssim=best_val_ssim,
                    args=vars(args), stats=stats)
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            ckpt["best_val_loss"] = best_val_loss
            save_checkpoint(ckpt, best_ckpt_path)
        if val_m["ssim"] > best_val_ssim:
            best_val_ssim = val_m["ssim"]
            ckpt["best_val_ssim"] = best_val_ssim
            save_checkpoint(ckpt, best_ssim_ckpt_path)
            print(f"  New best SSIM: {best_val_ssim:.4f}  {best_ssim_ckpt_path}")
        save_checkpoint(ckpt, last_ckpt_path)
        if epoch % 10 == 0 or epoch == args.epochs:
            plot_loss_curves(history, args.out_dir)
            plot_prediction_panel(model, val_loader, device, args.out_dir, epoch)
    print("\nLoading best SSIM checkpoint for test evaluation...")  # FIX-1
    ssim_path = os.path.join(args.out_dir, "best_ssim_model.pt")
    ckpt_to_load = ssim_path if os.path.exists(ssim_path) else best_ckpt_path
    load_checkpoint(ckpt_to_load, model)
    test_m = validate(model, test_loader, criterion, device, stats)
    print(f"\nTest results (best_ssim_model.pt):")
    print(f"  loss    = {test_m['loss']:.4f}")
    print(f"  recon   = {test_m['recon']:.4f}")
    print(f"  rel_l2  = {test_m['rel_l2']:.4f}  ({test_m['rel_l2']*100:.2f}%)")
    print(f"  mean_div (physical) = {test_m['mean_div']:.5f}")
    print(f"  ssim    = {test_m['ssim']:.4f}")
    print(f"  cls_acc = {test_m['acc']*100:.1f}%")
    print(f"\nComputing bicubic baseline on test set...")
    bic_m = evaluate_bicubic_baseline(test_loader, device)
    print_comparison_table(test_m, bic_m)
    plot_confusion_matrix(model, test_loader, device, args.out_dir)
    plot_loss_curves(history, args.out_dir)
    np.save(os.path.join(args.out_dir, "history.npy"), history)
    print(f"\nTraining complete.")
    print(f"  Best val loss : {best_val_loss:.4f}  (ep saved in best_model.pt)")
    print(f"  Best val SSIM : {best_val_ssim:.4f}  (ep saved in best_ssim_model.pt)")
    print(f"  Outputs       : {args.out_dir}/")

if __name__ == "__main__":
    main()