import os
import sys
import time
import argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset import make_dataloaders
from models.unet    import TurbulenceUNet, TurbulenceLoss, count_parameters

def get_args():
    p = argparse.ArgumentParser(description="Train turbulence U-Net")
    p.add_argument("--data_dir",    default="snapshots")
    p.add_argument("--out_dir",     default="runs/exp_01")
    p.add_argument("--epochs",      type=int,   default=80)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--lr_min",      type=float, default=1e-5)
    p.add_argument("--lambda_div",  type=float, default=0.1)
    p.add_argument("--lambda_cls",  type=float, default=0.01)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--base_ch",     type=int,   default=64)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num_workers", type=int,   default=0)
    return p.parse_args()

def relative_l2(pred, true):
    diff  = (pred - true).reshape(pred.shape[0], -1)
    denom = true.reshape(true.shape[0], -1).norm(dim=1).clamp(min=1e-8)
    return (diff.norm(dim=1) / denom).mean().item()


def mean_divergence(pred, dx=1.0/64):
    u_x = pred[:, 0, :, :]
    u_y = pred[:, 1, :, :]
    du_dx = (u_x[:, 2:, 1:-1] - u_x[:, :-2, 1:-1]) / (2 * dx)
    du_dy = (u_y[:, 1:-1, 2:] - u_y[:, 1:-1, :-2]) / (2 * dx)
    return (du_dx + du_dy).abs().mean().item()


def classification_accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()

def train_one_epoch(model, loader, criterion, optimizer,
                    grad_clip, device, epoch):
    model.train()
    totals = dict(loss=0, recon=0, div=0, cls=0,
                  rel_l2=0, mean_div=0, acc=0)
    n_batches = len(loader)
    for batch_idx, (coarse, fine, labels) in enumerate(loader):
        coarse = coarse.to(device)
        fine   = fine.to(device)
        labels = labels.to(device)
        pred_field, logits = model(coarse)
        total, recon, div_l, cls_l = criterion(
            pred_field, fine, logits, labels
        )
        optimizer.zero_grad()
        total.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        with torch.no_grad():
            totals['loss']     += total.item()
            totals['recon']    += recon.item()
            totals['div']      += div_l.item()
            totals['cls']      += cls_l.item()
            totals['rel_l2']   += relative_l2(pred_field, fine)
            totals['mean_div'] += mean_divergence(pred_field)
            totals['acc']      += classification_accuracy(logits, labels)
        if (batch_idx + 1) % 20 == 0 or batch_idx == n_batches - 1:
            print(f"  Epoch {epoch:3d}  batch {batch_idx+1:4d}/{n_batches}"
                  f"  loss={total.item():.4f}"
                  f"  recon={recon.item():.4f}"
                  f"  div={div_l.item():.4f}"
                  f"  acc={classification_accuracy(logits,labels)*100:.1f}%")

    return {k: v / n_batches for k, v in totals.items()}

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    totals = dict(loss=0, recon=0, div=0, cls=0,
                  rel_l2=0, mean_div=0, acc=0)
    n_batches = len(loader)
    for coarse, fine, labels in loader:
        coarse = coarse.to(device)
        fine   = fine.to(device)
        labels = labels.to(device)
        pred_field, logits = model(coarse)
        total, recon, div_l, cls_l = criterion(
            pred_field, fine, logits, labels
        )
        totals['loss']     += total.item()
        totals['recon']    += recon.item()
        totals['div']      += div_l.item()
        totals['cls']      += cls_l.item()
        totals['rel_l2']   += relative_l2(pred_field, fine)
        totals['mean_div'] += mean_divergence(pred_field)
        totals['acc']      += classification_accuracy(logits, labels)
    return {k: v / n_batches for k, v in totals.items()}

def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"  Checkpoint saved  {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    if optimizer  and 'optimizer'  in ckpt: optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler  and 'scheduler'  in ckpt: scheduler.load_state_dict(ckpt['scheduler'])
    print(f"Loaded checkpoint from {path}  (epoch {ckpt['epoch']})")
    return ckpt['epoch'], ckpt['best_val_loss']

def plot_loss_curves(history, out_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training dashboard — Turbulence U-Net",
                 fontsize=13, fontweight='bold')
    panels = [
        (axes[0,0], 'train_loss',  'val_loss',  "Total loss",           "Loss"),
        (axes[0,1], 'train_recon', 'val_recon', "Reconstruction loss",  "MSE"),
        (axes[1,0], 'train_div',   'val_div',   "Divergence loss",      "Div penalty"),
        (axes[1,1], 'train_acc',   'val_acc',   "Classifier accuracy",  "Accuracy"),
    ]

    for ax, tk, vk, title, ylabel in panels:
        ax.plot(epochs, history[tk], 'b-',  linewidth=1.5, label='Train')
        ax.plot(epochs, history[vk], 'r--', linewidth=1.5, label='Val')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fpath = os.path.join(out_dir, "training_curves.png")
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fpath}")


def plot_prediction_panel(model, loader, stats, device, out_dir, epoch):
    model.eval()
    coarse, fine, label = next(iter(loader))
    coarse = coarse.to(device)
    fine   = fine.to(device)
    with torch.no_grad():
        pred, logits = model(coarse)
    c  = coarse[0, 0].cpu().numpy()
    f  = fine[0, 0].cpu().numpy()
    p  = pred[0, 0].cpu().numpy()
    err = np.abs(f - p)
    re_names = {0:"Re=100", 1:"Re=400", 2:"Re=1000", 3:"Re=5000", 4:"Re=10000"}
    true_re  = re_names.get(label[0].item(), "?")
    pred_re  = re_names.get(logits[0].argmax().item(), "?")
    vmax = max(np.abs(f).max(), 1e-8)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"Epoch {epoch} — u-velocity  |  True: {true_re}  Pred: {pred_re}",
        fontsize=12, fontweight='bold'
    )

    titles = ["Coarse input", "DNS target", "CNN prediction", "|error|"]
    fields = [c, f, p, err]
    cmaps  = ["RdBu_r", "RdBu_r", "RdBu_r", "hot"]

    for ax, field, title, cmap in zip(axes, fields, titles, cmaps):
        v = vmax if cmap != "hot" else err.max() + 1e-8
        im = ax.imshow(field, cmap=cmap, origin='lower',
                       vmin=-v if cmap=="RdBu_r" else 0, vmax=v)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fpath = os.path.join(out_dir, f"prediction_ep{epoch:03d}.png")
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fpath}")


def plot_confusion_matrix(model, loader, device, out_dir):
    model.eval()
    all_true = []
    all_pred = []
    with torch.no_grad():
        for coarse, fine, labels in loader:
            coarse = coarse.to(device)
            _, logits = model(coarse)
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(logits.argmax(1).cpu().numpy())
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    n_classes = 5
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_true, all_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    class_names = ["Re=100", "Re=400", "Re=1000", "Re=5000", "Re=10000"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(class_names, rotation=30)
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Re regime")
    ax.set_ylabel("True Re regime")
    ax.set_title("Re regime confusion matrix", fontsize=12, fontweight='bold')
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2 else 'black',
                    fontsize=11)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fpath = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fpath}")

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Output dir: {args.out_dir}")
    train_loader, val_loader, test_loader, stats = make_dataloaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )
    model = TurbulenceUNet(
        in_ch=3, out_ch=3,
        base_ch=args.base_ch,
        n_classes=5
    ).to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")
    criterion = TurbulenceLoss(
        lambda_div=args.lambda_div,
        lambda_cls=args.lambda_cls,
    )

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_steps = args.epochs * len(train_loader)
    scheduler   = CosineAnnealingLR(
        optimizer,
        T_max   = total_steps,
        eta_min = args.lr_min,
    )
    history = {k: [] for k in [
        'train_loss','train_recon','train_div','train_cls','train_acc',
        'val_loss',  'val_recon',  'val_div',  'val_cls',  'val_acc',
        'lr'
    ]}
    best_val_loss  = float('inf')
    best_ckpt_path = os.path.join(args.out_dir, "best_model.pt")
    last_ckpt_path = os.path.join(args.out_dir, "last_model.pt")
    print(f"Training for {args.epochs} epochs")
    print(f"  lr: {args.lr} → {args.lr_min} (cosine)")
    print(f"  lambda_div={args.lambda_div}  lambda_cls={args.lambda_cls}")
    print(f"  grad_clip={args.grad_clip}")

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion,
            optimizer, args.grad_clip, device, epoch
        )
        val_metrics = validate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_metrics['loss'])
        history['train_recon'].append(train_metrics['recon'])
        history['train_div'].append(train_metrics['mean_div'])
        history['train_cls'].append(train_metrics['cls'])
        history['train_acc'].append(train_metrics['acc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_recon'].append(val_metrics['recon'])
        history['val_div'].append(val_metrics['mean_div'])
        history['val_cls'].append(val_metrics['cls'])
        history['val_acc'].append(val_metrics['acc'])
        history['lr'].append(current_lr)
        elapsed = time.time() - t_start
        print(f"\nEpoch {epoch:3d}/{args.epochs}  [{elapsed:.1f}s]")
        print(f"  Train  loss={train_metrics['loss']:.4f}"
              f"  recon={train_metrics['recon']:.4f}"
              f"  div={train_metrics['mean_div']:.4f}"
              f"  acc={train_metrics['acc']*100:.1f}%")
        print(f"  Val    loss={val_metrics['loss']:.4f}"
              f"  recon={val_metrics['recon']:.4f}"
              f"  div={val_metrics['mean_div']:.4f}"
              f"  acc={val_metrics['acc']*100:.1f}%")
        print(f"  LR={current_lr:.2e}")
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint({
                'epoch'         : epoch,
                'model'         : model.state_dict(),
                'optimizer'     : optimizer.state_dict(),
                'scheduler'     : scheduler.state_dict(),
                'best_val_loss' : best_val_loss,
                'args'          : vars(args),
                'stats'         : stats,
            }, best_ckpt_path)
        save_checkpoint({
            'epoch'         : epoch,
            'model'         : model.state_dict(),
            'optimizer'     : optimizer.state_dict(),
            'scheduler'     : scheduler.state_dict(),
            'best_val_loss' : best_val_loss,
            'args'          : vars(args),
            'stats'         : stats,
        }, last_ckpt_path)

        scheduler.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            plot_loss_curves(history, args.out_dir)
            plot_prediction_panel(
                model, val_loader, stats, device, args.out_dir, epoch
            )
    print("Loading best checkpoint for final test evaluation...")
    load_checkpoint(best_ckpt_path, model)
    test_metrics = validate(model, test_loader, criterion, device)
    print(f"\nTest set results:")
    print(f"  loss={test_metrics['loss']:.4f}")
    print(f"  recon={test_metrics['recon']:.4f}")
    print(f"  rel_l2={test_metrics['rel_l2']:.4f}  ({test_metrics['rel_l2']*100:.2f}%)")
    print(f"  mean_div={test_metrics['mean_div']:.5f}")
    print(f"  classifier acc={test_metrics['acc']*100:.1f}%")
    plot_confusion_matrix(model, test_loader, device, args.out_dir)
    plot_loss_curves(history, args.out_dir)
    np.save(os.path.join(args.out_dir, "history.npy"), history)
    print("Training complete.")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Outputs saved : {args.out_dir}/")
    print(f"  best_model.pt, training_curves.png, confusion_matrix.png")
    print(f"  prediction_ep*.png  (every 10 epochs)")

if __name__ == "__main__":
    main()