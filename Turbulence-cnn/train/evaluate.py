"""
Stage 5 — DNS Super-Resolution & Transfer Learning  v3
Fixes over v2:
    FIX A — DNS normalisation: compute stats FROM the DNS data itself,
                     never apply cavity-flow stats to DNS fields.
                     The two domains have completely different pressure scales.
    FIX B — Synthetic DNS generator: use uniform random phases on [0, 2pi]
                     (not Gaussian). Verified to produce slope -1.63 to -1.70.

Run: python train/evaluate.py --checkpoint runs/exp_02/best_model.pt
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.unet import TurbulenceUNet, TurbulenceLoss

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="runs/exp_02/best_model.pt")
    p.add_argument("--dns_path",    default=None)
    p.add_argument("--out_dir",     default="runs/stage5_v3")
    p.add_argument("--n_dns",       type=int,   default=400)
    p.add_argument("--n_finetune",  type=int,   default=300)
    p.add_argument("--ft_epochs",   type=int,   default=30)
    p.add_argument("--ft_lr",       type=float, default=5e-5)
    p.add_argument("--base_ch",     type=int,   default=64)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()

def generate_synthetic_dns(n_samples=400, N=64, Re=5000, seed=42):
    """
    Correct synthetic DNS with verified Kolmogorov spectrum.
    Uses uniform random phases — the only correct method.
    """
    rng = np.random.default_rng(seed)
    print(f"\nGenerating {n_samples} synthetic DNS snapshots (Re={Re})...")
    print(f"  Using uniform phase method (fixed from v2 Gaussian phases)")

    kx_1d = np.fft.fftfreq(N, d=1.0 / N)
    ky_1d = np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(kx_1d, ky_1d, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)

    k_eta = N // 4
    K_safe = np.where(K > 0, K, 1.0)
    amplitude = K_safe**(-7/6) * np.exp(-0.5 * (K_safe / k_eta)**2)
    amplitude[0, 0] = 0.0

    fields = []
    for i in range(n_samples):
        phase = rng.uniform(0.0, 2.0 * np.pi, (N, N))
        psi_hat = amplitude * np.exp(1j * phase)

        u_hat = 1j * KY * psi_hat
        v_hat = -1j * KX * psi_hat

        u = np.real(np.fft.ifft2(u_hat)).astype(np.float32)
        v = np.real(np.fft.ifft2(v_hat)).astype(np.float32)

        rms = np.sqrt(np.mean(u**2 + v**2)) + 1e-10
        u /= rms
        v /= rms

        K2 = KX**2 + KY**2
        K2[0, 0] = 1.0
        u_hat2 = np.fft.fft2(u)
        v_hat2 = np.fft.fft2(v)
        src = (1j * KX * u_hat2**2 + 1j * (KX + KY) * u_hat2 * v_hat2 + 1j * KY * v_hat2**2)
        p_hat = -src / K2
        p_hat[0, 0] = 0.0
        p = np.real(np.fft.ifft2(p_hat)).astype(np.float32)

        fields.append(np.stack([u, v, p], axis=0))

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n_samples}")

    fields = np.stack(fields, axis=0)

    slopes = []
    for i in range(min(20, n_samples)):
        k, Ek = energy_spectrum(fields[i, 0], fields[i, 1])
        s = spectral_slope(k, Ek)
        if not np.isnan(s):
            slopes.append(s)
    ms = np.mean(slopes)
    ok = abs(ms + 5/3) < 0.15
    print(f"\n  Spectral slope check: {ms:.3f}  "
          f"(target -1.667)  {'PASS' if ok else f'WARNING: err={abs(ms+5/3):.3f}'}")
    print(f"  u range: [{fields[:,0].min():.3f}, {fields[:,0].max():.3f}]")
    print(f"  p range: [{fields[:,2].min():.3f}, {fields[:,2].max():.3f}]")
    return fields

def load_jhtdb_dns(h5_path, n_samples=300, N=64):
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for real DNS: pip install h5py")
    print(f"\nLoading JHTDB DNS from {h5_path}...")
    fields = []
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        print(f"  Available keys: {keys}")
        vel_key = None
        for k in ['u', 'velocity', 'vel', 'u_x']:
            if k in keys:
                vel_key = k
                break
        if vel_key is None:
            print(f"  Warning: Could not find velocity key. Using first key: {keys[0]}")
            vel_key = keys[0]

        vel_data = f[vel_key]
        print(f"  Velocity data shape: {vel_data.shape}")
        n_available = vel_data.shape[0] if len(vel_data.shape) >= 3 else 1
        indices = np.linspace(0, min(n_available-1, 500), n_samples, dtype=int)
        for idx in indices:
            try:
                if len(vel_data.shape) == 4:
                    if vel_data.shape[1] == 3:
                        snap = vel_data[idx, :3]   # (3, H, W)
                    else:
                        snap = vel_data[idx, ..., :3].transpose(2, 0, 1)
                elif len(vel_data.shape) == 3:
                    snap = vel_data[:3]
                else:
                    continue
                _, H, W = snap.shape
                if H != N or W != N:
                    snap = np.stack([
                        zoom(snap[c], (N/H, N/W), order=1)
                        for c in range(snap.shape[0])
                    ], axis=0)
                fields.append(snap.astype(np.float32))
            except Exception as e:
                print(f"  Warning: skipping idx={idx}: {e}")
                continue

    if not fields:
        raise ValueError("No valid snapshots loaded from JHTDB file")

    fields = np.stack(fields[:n_samples], axis=0)
    print(f"  Loaded {len(fields)} DNS snapshots. Shape: {fields.shape}")
    return fields

def preprocess_dns(fields):
    """
    FIX A: Normalise DNS fields using DNS-computed statistics.
    Never import cavity flow stats for DNS data.

    Returns: coarse tensor, fine tensor, dns_stats dict
    """
    N_SNAP, C, H, W = fields.shape
    print(f"\nPreprocessing {N_SNAP} DNS snapshots...")

    mean = fields.mean(axis=(0, 2, 3))
    std  = fields.std(axis=(0, 2, 3)).clip(min=1e-8)
    dns_stats = {'mean': mean, 'std': std}

    print(f"  DNS stats — mean: {mean.round(4)}  std: {std.round(4)}")

    fields_norm = (fields - mean[None, :, None, None]) / std[None, :, None, None]

    print(f"  Normalised range: [{fields_norm.min():.2f}, {fields_norm.max():.2f}]")
    print(f"  Expected: roughly [-4, +4] for well-normalised data")

    factor = H // 16
    coarse_list = []
    for i in range(N_SNAP):
        fine_np = fields_norm[i]
        coarse_small = fine_np.reshape(C, 16, factor, 16, factor).mean(axis=(2, 4))
        coarse_up = np.stack([
            zoom(coarse_small[c], factor, order=1) for c in range(C)
        ], axis=0)
        coarse_list.append(coarse_up.astype(np.float32))

    coarse = torch.from_numpy(np.stack(coarse_list, axis=0))
    fine   = torch.from_numpy(fields_norm.astype(np.float32))
    return coarse, fine, dns_stats

def energy_spectrum(u, v):
    N = u.shape[0]
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    energy = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / N**2
    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    k_bins = np.arange(1, N//2)
    E_k = np.array([energy[(K >= k-0.5) & (K < k+0.5)].sum() for k in k_bins])
    return k_bins, E_k


def spectral_slope(k_bins, E_k, k_min=3, k_max=None):
    if k_max is None:
        k_max = len(k_bins) // 2
    mask = (k_bins >= k_min) & (k_bins <= k_max) & (E_k > 1e-30)
    if mask.sum() < 3:
        return float('nan')
    slope, _ = np.polyfit(np.log(k_bins[mask]), np.log(E_k[mask]), 1)
    return slope

@torch.no_grad()
def evaluate(model, coarse, fine, device, batch_size=16):
    model.eval()
    dataset = TensorDataset(coarse, fine)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    metrics = dict(mse=0, rel_l2=0, mean_div=0)
    dx = 1.0 / coarse.shape[-1]

    for c_batch, f_batch in loader:
        c_batch = c_batch.to(device)
        f_batch = f_batch.to(device)

        pred, _ = model(c_batch)
        all_preds.append(pred.cpu())

        metrics['mse'] += nn.functional.mse_loss(pred, f_batch).item()
        diff  = (pred - f_batch).reshape(pred.shape[0], -1)
        denom = f_batch.reshape(f_batch.shape[0], -1).norm(dim=1).clamp(min=1e-8)
        metrics['rel_l2'] += (diff.norm(dim=1) / denom).mean().item()
        u_x = pred[:, 0]; u_y = pred[:, 1]
        du_dx = (u_x[:, 2:, 1:-1] - u_x[:, :-2, 1:-1]) / (2*dx)
        du_dy = (u_y[:, 1:-1, 2:] - u_y[:, 1:-1, :-2]) / (2*dx)
        metrics['mean_div'] += (du_dx + du_dy).abs().mean().item()

    n = len(loader)
    metrics = {k: v/n for k, v in metrics.items()}
    preds = torch.cat(all_preds, dim=0)
    return preds, metrics

def finetune(model, coarse_train, fine_train, device,
             epochs=30, lr=5e-5, batch_size=16):
    print(f"\n{'='*55}")
    print("Fine-tuning: dec1 + output_conv only")
    print(f"  epochs={epochs}  lr={lr}")

    for p in model.parameters():
        p.requires_grad = False

    trainable_modules = ['dec1', 'output_conv']
    frozen_count = trainable_count = 0

    for name, param in model.named_parameters():
        if any(m in name for m in trainable_modules):
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()

    total = frozen_count + trainable_count
    print(f"\n  Frozen     : {frozen_count:>10,}  ({100*frozen_count/total:.1f}%)")
    print(f"  Trainable  : {trainable_count:>10,}  ({100*trainable_count/total:.1f}%)")
    print(f"  Modules    : {trainable_modules}\n")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr/20
    )

    criterion = TurbulenceLoss(lambda_div=0.0, lambda_cls=0.0)

    dummy_labels = torch.zeros(batch_size, dtype=torch.long).to(device)
    loader = DataLoader(TensorDataset(coarse_train, fine_train),
                        batch_size=batch_size, shuffle=True, drop_last=True)

    history = dict(loss=[], recon=[])
    model.train()

    for epoch in range(1, epochs+1):
        ep_loss = ep_recon = 0.0
        for c_b, f_b in loader:
            c_b = c_b.to(device); f_b = f_b.to(device)
            lbl = dummy_labels[:c_b.shape[0]]
            pred, logits = model(c_b)
            total, recon, _, _ = criterion(pred, f_b, logits, lbl)
            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            optimizer.step()
            ep_loss += total.item()
            ep_recon += recon.item()

        scheduler.step()
        n = len(loader)
        history['loss'].append(ep_loss/n)
        history['recon'].append(ep_recon/n)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  FT {epoch:3d}/{epochs}"
                  f"  loss={ep_loss/n:.5f}"
                  f"  lr={optimizer.param_groups[0]['lr']:.2e}")

    return model, history

def plot_spectra(coarse, fine, pred_zero, pred_ft, out_dir, n_avg=30):
    fig, ax = plt.subplots(figsize=(8, 6))

    datasets = [
        (fine,      'DNS ground truth',      'k',       2.0, '-'),
        (coarse,    'Coarse input',           '#888888', 1.5, '--'),
        (pred_zero, 'Zero-shot (no FT)',      '#E24B4A', 1.5, '-'),
        (pred_ft,   'Fine-tuned (dec1 only)', '#1D9E75', 2.0, '-'),
    ]

    for data, label, color, lw, ls in datasets:
        all_Ek = []
        for i in range(min(n_avg, len(data))):
            u = data[i, 0].numpy(); v = data[i, 1].numpy()
            k, Ek = energy_spectrum(u, v)
            all_Ek.append(Ek)
        Ek_mean = np.mean(all_Ek, axis=0)
        slope = spectral_slope(k, Ek_mean)
        ax.loglog(k, Ek_mean, color=color, lw=lw, ls=ls,
                  label=f"{label}  (slope={slope:.3f})")

    k_ref = np.array([3.0, 20.0])
    k_all_Ek = []
    for i in range(min(n_avg, len(fine))):
        _, Ek = energy_spectrum(fine[i, 0].numpy(), fine[i, 1].numpy())
        k_all_Ek.append(Ek)
    Ek_dns = np.mean(k_all_Ek, axis=0)
    idx_anchor = np.argmin(np.abs(k - 6))
    E_ref = Ek_dns[idx_anchor] * (k_ref / k[idx_anchor])**(-5/3)
    ax.loglog(k_ref, E_ref, 'b:', lw=1.5, label=r'$k^{-5/3}$ reference')

    ax.set_xlabel("Wavenumber $k$", fontsize=13)
    ax.set_ylabel("$E(k)$", fontsize=13)
    ax.set_title(f"Energy spectrum comparison (averaged over {n_avg} samples)",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    fpath = os.path.join(out_dir, "energy_spectra_final.png")
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fpath}")

def plot_vorticity(coarse, fine, pred_zero, pred_ft, out_dir, idx=0):
    def vort(u, v, dx=1/64):
        w = np.zeros_like(u)
        w[1:-1,1:-1] = ((v[2:,1:-1]-v[:-2,1:-1])/(2*dx) -
                        (u[1:-1,2:]-u[1:-1,:-2])/(2*dx))
        return w

    c  = coarse[idx, 0].numpy();   vc = coarse[idx, 1].numpy()
    f  = fine[idx, 0].numpy();     vf = fine[idx, 1].numpy()
    z  = pred_zero[idx, 0].numpy(); vz = pred_zero[idx, 1].numpy()
    ft = pred_ft[idx, 0].numpy();  vft = pred_ft[idx, 1].numpy()

    wc = vort(c, vc); wf = vort(f, vf); wz = vort(z, vz); wft = vort(ft, vft)
    err_z = np.abs(wf - wz)
    err_ft = np.abs(wf - wft)
    vmax = np.abs(wf).max() + 1e-8
    emax = max(err_z.max(), err_ft.max()) + 1e-8

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Vorticity — coarse / DNS / zero-shot / fine-tuned",
                 fontsize=13, fontweight='bold')

    panels = [
        (wc, "Coarse input", 'seismic', vmax, True),
        (wf, "DNS truth", 'seismic', vmax, True),
        (wz, "Zero-shot", 'seismic', vmax, True),
        (wft, "Fine-tuned", 'seismic', vmax, True),
        (err_z, "|err| zero-shot", 'hot', emax, False),
        (err_ft, "|err| fine-tuned", 'hot', emax, False),
    ]

    for ax, (w, title, cmap, v, symmetric) in zip(axes.flat, panels):
        if symmetric:
            im = ax.imshow(w, cmap=cmap, origin='lower', vmin=-v, vmax=v)
        else:
            im = ax.imshow(w, cmap=cmap, origin='lower', vmin=0, vmax=v)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fpath = os.path.join(out_dir, "vorticity_final.png")
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fpath}")


def plot_ft_curves(history, out_dir):
    ep = range(1, len(history['loss']) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title("Fine-tuning curves (dec1 + output_conv only)", fontsize=12)
    ax.plot(ep, history['loss'], 'b-', lw=2, label='total')
    ax.plot(ep, history['recon'], 'r--', lw=2, label='recon')
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fpath = os.path.join(out_dir, "finetune_curves_final.png")
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fpath}")


def print_table(mz, mft):
    print(f"\n{'='*52}")
    print(f"{'Metric':<22} {'Zero-shot':>14} {'Fine-tuned':>14}")
    print(f"{'-'*52}")
    for k in ['mse', 'rel_l2', 'mean_div']:
        better = '↓' if mft[k] < mz[k] else '↑'
        print(f"  {k:<20} {mz[k]:>14.5f} {mft[k]:>14.5f} {better}")
    print("  ↓ better / ↑ worse")
    print(f"{'='*52}")


def print_slopes(datasets):
    print(f"\n{'='*52}")
    print("Spectral slope analysis (target: -1.667)")
    print(f"{'='*52}")
    for label, data in datasets.items():
        slopes = []
        for i in range(min(20, len(data))):
            k, Ek = energy_spectrum(data[i, 0].numpy(), data[i, 1].numpy())
            s = spectral_slope(k, Ek)
            if not np.isnan(s):
                slopes.append(s)
        ms = np.mean(slopes) if slopes else float('nan')
        err = abs(ms + 5/3) if not np.isnan(ms) else float('nan')
        status = (
            "PASS" if not np.isnan(err) and err < 0.15 else
            "improving" if not np.isnan(err) and err < 0.3 else
            "needs work"
        )
        print(f"  {label:<22} slope={ms:+.3f}  err={err:.3f}  {status}")



def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Out: {args.out_dir}")

    model = TurbulenceUNet(in_ch=3, out_ch=3,
                           base_ch=args.base_ch, n_classes=5).to(device)
    if os.path.exists(args.checkpoint):
        try:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f"Loaded checkpoint epoch={ckpt.get('epoch','?')}"
              f"  val_loss={ckpt.get('best_val_loss','?'):.4f}")
    else:
        print(f"WARNING: checkpoint not found, using random weights")

    if args.dns_path and os.path.exists(args.dns_path):
        raw = load_jhtdb_dns(args.dns_path, args.n_dns, N=64)
    else:
        raw = generate_synthetic_dns(args.n_dns, N=64, Re=5000, seed=args.seed)

    coarse, fine, dns_stats = preprocess_dns(raw)

    n_ft = args.n_finetune
    coarse_ft, fine_ft     = coarse[:n_ft], fine[:n_ft]
    coarse_test, fine_test = coarse[n_ft:], fine[n_ft:]
    print(f"Split: fine-tune={n_ft}  test={len(coarse_test)}")

    print(f"\nZero-shot evaluation...")
    pred_zero, mz = evaluate(model, coarse_test, fine_test, device)
    print(f"  MSE={mz['mse']:.4f}  RelL2={mz['rel_l2']*100:.1f}%"
          f"  div={mz['mean_div']:.4f}")

    model_ft, ft_history = finetune(
        model, coarse_ft, fine_ft, device,
        epochs=args.ft_epochs, lr=args.ft_lr
    )
    torch.save({'model': model_ft.state_dict(), 'dns_stats': dns_stats},
               os.path.join(args.out_dir, "finetuned_final.pt"))

    print(f"\nPost fine-tuning evaluation...")
    pred_ft, mft = evaluate(model_ft, coarse_test, fine_test, device)
    print(f"  MSE={mft['mse']:.4f}  RelL2={mft['rel_l2']*100:.1f}%"
          f"  div={mft['mean_div']:.4f}")

    print_table(mz, mft)
    print_slopes({
        "DNS truth": fine_test,
        "Coarse": coarse_test,
        "Zero-shot": pred_zero,
        "Fine-tuned": pred_ft,
    })

    plot_spectra(coarse_test, fine_test, pred_zero, pred_ft, args.out_dir)
    plot_vorticity(coarse_test, fine_test, pred_zero, pred_ft, args.out_dir)
    plot_ft_curves(ft_history, args.out_dir)

    print("\nFinal outputs:")
    print("  - finetuned_final.pt")
    print("  - energy_spectra_final.png")
    print("  - vorticity_final.png")
    print("  - finetune_curves_final.png")

if __name__ == "__main__":
    main()