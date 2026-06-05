import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.unet   import TurbulenceUNet
from models.losses import TurbulenceLoss, ssim as ssim_fn
from data.dataset  import N_CLASSES, RE_VALUES, CHANNEL_NAMES
from data.augmentation import COARSE_SIZE
from sklearn.manifold import TSNE

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint",default="runs/exp_01/best_ssim_model.pt")
    p.add_argument("--stats_path",default="train-data/stats.npy")
    p.add_argument("--ood_snap_dir",default=None,help="If None, auto-generates using the solver.")
    p.add_argument("--out_dir",default="runs/eval_ood")
    p.add_argument("--base_ch",type=int,default=64)
    p.add_argument("--n_samples",type=int,default=100,help="OOD snapshots per Re value")
    p.add_argument("--finetune",action="store_true",help="Fine-tune (not recommended for same-domain OOD)")
    p.add_argument("--ft_epochs",type=int,   default=30)
    p.add_argument("--ft_lr",type=float, default=5e-5)
    p.add_argument("--seed",type=int,   default=None)
    return p.parse_args()


def init_seed(seed):
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**32 - 1))
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}"); return seed

def energy_spectrum(u, v):
    N = u.shape[0]
    u_hat = np.fft.fft2(u); v_hat = np.fft.fft2(v)
    k_cut = N // 3
    kx = np.fft.fftfreq(N)*N; ky = np.fft.fftfreq(N)*N
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    alias = (np.abs(KX) > k_cut) | (np.abs(KY) > k_cut)
    u_hat[alias] = 0.0; v_hat[alias] = 0.0
    energy = 0.5*(np.abs(u_hat)**2 + np.abs(v_hat)**2) / N**4
    K      = np.sqrt(KX**2 + KY**2)
    k_bins = np.arange(1, k_cut+1)
    E_k    = np.array([energy[(K>=k-0.5)&(K<k+0.5)].sum() for k in k_bins])
    return k_bins, E_k


def spectral_slope(k, Ek, k_min=4, k_max=None):
    if k_max is None: k_max = len(k)//2
    mask = (k>=k_min)&(k<=k_max)&(Ek>1e-30)
    if mask.sum() < 3: return float("nan")
    slope, _ = np.polyfit(np.log(k[mask]), np.log(Ek[mask]), 1)
    return slope

def generate_ood_cavity_snapshots(ood_re_values, n_per_re, out_dir, solver_dir):
    print(f"Generating OOD cavity snapshots")
    print(f"  Re values: {ood_re_values}")
    print(f"  n per Re : {n_per_re}")
    sys.path.insert(0, solver_dir)
    try:
        import solver.ns_solver as ns
        import solver.generate_data   as gd
        import solver.pressure_poisson as pp
    except ImportError:
        import solver.ns_solver as ns
        import solver.generate_data as gd
        import solver.pressure_poisson as pp
    os.makedirs(out_dir, exist_ok=True)
    for Re in ood_re_values:
        re_dir = os.path.join(out_dir, f"Re_{Re}")
        if os.path.exists(re_dir) and len(os.listdir(re_dir)) >= n_per_re:
            print(f"  Re={Re}: already have snapshots, skipping")
        print(f"\n  Running Re={Re}...")
        u, v, p, snaps = gd.run_simulation(
            Re=Re, N=64,
            save_dir=out_dir,
            NSconfig_class=ns.NSconfig,
            apply_bc_fn=ns.apply_bc,
            step_fn=ns.step,
            stable_dt_fn=ns.stable_dt,
            diagnostics_fn=ns.diagnostics,
            is_converged_fn=ns.is_converged,
            vorticity_fn=ns.vorticity,
        )
        snap_files = sorted(os.listdir(re_dir))
        if len(snap_files) > n_per_re:
            to_delete = snap_files[:-n_per_re]
            for f in to_delete:
                os.remove(os.path.join(re_dir, f))
        print(f"  Re={Re}: {min(len(snaps), n_per_re)} snapshots saved to {re_dir}")
    return out_dir

def load_ood_snapshots(ood_dir, training_stats, ood_re_values):
    VEL_CLIP_CHANNELS  = [0, 1]   # u, v only
    VEL_CLIP_THRESHOLD = 10.0     # ≈ 1.9 m/s physical — impossible at steady state
    all_coarse, all_fine, all_re_labels = [], [], []
    print(f"\nLoading OOD snapshots from {ood_dir}...")
    for Re in ood_re_values:
        re_dir = os.path.join(ood_dir, f"Re_{Re}")
        if not os.path.exists(re_dir):
            print(f"  Re={Re}: directory not found, skipping")
            continue
        snap_files = sorted(f for f in os.listdir(re_dir)if f.endswith(".npy"))
        if not snap_files:
            print(f"  Re={Re}: no .npy files found")
            continue
        loaded = 0
        for fname in snap_files:
            snap = np.load(os.path.join(re_dir, fname)).astype(np.float32)
            if snap.shape == (3, 64, 64):
                u, v = snap[0], snap[1]
                dx   = 1.0 / 64
                omega = np.zeros((64, 64), dtype=np.float32)
                omega[1:-1, 1:-1] = (
                    (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dx)
                  - (u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx)
                )
                snap = np.concatenate([snap, omega[None]], axis=0)  # → (4,64,64)
            elif snap.shape != (4, 64, 64):
                continue   # unexpected shape — skip silently
            norm = ((snap - training_stats["mean"][:,None,None])
                    / training_stats["std"][:,None,None])
            is_outlier = False
            for ch in VEL_CLIP_CHANNELS:
                if np.max(np.abs(norm[ch])) > VEL_CLIP_THRESHOLD:
                    is_outlier = True
                    break
            if is_outlier:
                continue
            C, H, W = norm.shape
            factor = H // COARSE_SIZE
            coarse_small = norm.reshape(C, COARSE_SIZE, factor,COARSE_SIZE, factor).mean(axis=(2,4))
            coarse_up = np.stack([zoom(coarse_small[c], factor, order=1)for c in range(C)], axis=0)
            all_coarse.append(coarse_up.astype(np.float32))
            all_fine.append(norm.astype(np.float32))
            all_re_labels.append(Re)
            loaded += 1
        print(f"  Re={Re}: {loaded} snapshots loaded")
    if len(all_coarse) == 0:
        raise RuntimeError(
            f"No OOD snapshots loaded from {ood_dir}.\n"
            f"Expected subdirs: {['Re_'+str(r) for r in ood_re_values]}\n"
            f"Check that the solver ran and saved .npy files there.")
    coarse = torch.from_numpy(np.stack(all_coarse, axis=0))
    fine   = torch.from_numpy(np.stack(all_fine,   axis=0))
    print(f"\nTotal OOD snapshots: {len(all_re_labels)}")
    print(f"  coarse range: [{coarse.min():.3f}, {coarse.max():.3f}]")
    print(f"  fine   range: [{fine.min():.3f}, {fine.max():.3f}]")
    return coarse, fine, all_re_labels

@torch.no_grad()
def evaluate_model(model, coarse, fine, device, training_stats=None, batch_size=16):
    model.eval()
    loader   = DataLoader(TensorDataset(coarse, fine),batch_size=batch_size, shuffle=False)
    all_pred = []
    metrics  = dict(mse=0.0, rel_l2=0.0, mean_div=0.0, ssim=0.0)
    dx = 1.0 / coarse.shape[-1]
    for c_b, f_b in loader:
        c_b = c_b.to(device); f_b = f_b.to(device)
        pred, _ = model(c_b)
        if training_stats is not None:
            pred = model.project_divergence_free(pred, training_stats)
        all_pred.append(pred.cpu())
        metrics["mse"]     += nn.functional.mse_loss(pred, f_b).item()
        diff = (pred-f_b).reshape(pred.shape[0],-1)
        denom = f_b.reshape(f_b.shape[0],-1).norm(dim=1).clamp(min=1e-8)
        metrics["rel_l2"]  += (diff.norm(dim=1)/denom).mean().item()
        metrics["ssim"]    += ssim_fn(pred, f_b).item()
        u_f = pred[:,0]; v_f = pred[:,1]
        du_dx = (u_f[:,2:,1:-1]-u_f[:,:-2,1:-1])/(2*dx)
        dv_dy = (v_f[:,1:-1,2:]-v_f[:,1:-1,:-2])/(2*dx)
        metrics["mean_div"] += (du_dx+dv_dy).abs().mean().item()
    n = len(loader)
    preds = torch.cat(all_pred, dim=0)
    return preds, {k: v/n for k,v in metrics.items()}

@torch.no_grad()
def evaluate_per_re(model, coarse, fine, re_labels, device):
    model.eval()
    re_labels_np = np.array(re_labels)
    results = {}
    for Re in sorted(set(re_labels)):
        mask = re_labels_np == Re
        c_re = coarse[mask]; f_re = fine[mask]
        if len(c_re) == 0: continue
        loader = DataLoader(TensorDataset(c_re, f_re), batch_size=16, shuffle=False)
        ssim_vals, rl2_vals = [], []
        for cb, fb in loader:
            cb = cb.to(device); fb = fb.to(device)
            pred, _ = model(cb)
            ssim_vals.append(ssim_fn(pred, fb).item())
            diff  = (pred-fb).reshape(pred.shape[0],-1)
            denom = fb.reshape(fb.shape[0],-1).norm(dim=1).clamp(min=1e-8)
            rl2_vals.append((diff.norm(dim=1)/denom).mean().item())
        results[Re] = dict(ssim=np.mean(ssim_vals), rel_l2=np.mean(rl2_vals),n=len(c_re))
    return results

@torch.no_grad()
def evaluate_per_channel(model, coarse, fine, device):
    model.eval()
    loader = DataLoader(TensorDataset(coarse, fine), batch_size=16, shuffle=False)
    C = fine.shape[1]
    channel_ssim = [[] for _ in range(C)]
    for cb, fb in loader:
        cb = cb.to(device); fb = fb.to(device)
        pred, _ = model(cb)
        for ch in range(C):
            s = ssim_fn(pred[:,ch:ch+1], fb[:,ch:ch+1]).item()
            channel_ssim[ch].append(s)
    results = {}
    for ch in range(C):
        name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"ch{ch}"
        results[name] = float(np.mean(channel_ssim[ch]))
    return results

@torch.no_grad()
def bicubic_baseline(coarse, fine, device):
    B, C, H, W = coarse.shape
    factor = H // COARSE_SIZE
    bic_list = []
    c_np = coarse.numpy()
    for b in range(B):
        small = c_np[b].reshape(C, COARSE_SIZE, factor,
                                  COARSE_SIZE, factor).mean(axis=(2,4))
        bic_list.append(np.stack([zoom(small[c], factor, order=3)
                                   for c in range(C)], axis=0))
    bic = torch.from_numpy(np.stack(bic_list).astype(np.float32)).to(device)
    fine = fine.to(device)
    mse    = nn.functional.mse_loss(bic, fine).item()
    diff   = (bic-fine).reshape(B,-1)
    denom  = fine.reshape(B,-1).norm(dim=1).clamp(min=1e-8)
    rel_l2 = (diff.norm(dim=1)/denom).mean().item()
    s_val  = ssim_fn(bic, fine).item()
    return dict(mse=mse, rel_l2=rel_l2, ssim=s_val)

@torch.no_grad()
def plot_tsne(model, coarse_train, labels_train, device, out_dir):
    model.eval()
    print(f"\nComputing t-SNE on {len(coarse_train)} samples...")
    loader = DataLoader(TensorDataset(coarse_train),batch_size=32, shuffle=False)
    feats = []
    for (cb,) in loader:
        cb = cb.to(device)
        f = model.extract_features(cb)
        feats.append(f.cpu().numpy())
    feats_np = np.concatenate(feats, axis=0)
    tsne = TSNE(n_components=2, perplexity=30, random_state=67,max_iter=1000, verbose=0)
    emb = tsne.fit_transform(feats_np)
    re_labels = np.array(labels_train)
    unique_re = sorted(set(re_labels.tolist()))
    cmap = matplotlib.colormaps.get_cmap("plasma")
    colors = cmap(np.linspace(0.0, 1.0, max(1, len(unique_re))))
    re_to_idx = {re: i for i, re in enumerate(unique_re)}
    fig, ax = plt.subplots(figsize=(9, 7))
    for re in unique_re:
        mask = re_labels == re
        ax.scatter(emb[mask, 0], emb[mask, 1],
               c=[colors[re_to_idx[re]]]*mask.sum(),
                   label=f"Re={re}", s=12, alpha=0.7)
    ax.set_title("t-SNE of U-Net bottleneck features (256-dim)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.legend(fontsize=8, ncol=3, loc="best")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fpath = os.path.join(out_dir, "tsne_bottleneck.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    pdf_path = os.path.splitext(fpath)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fpath}")

def plot_ood_spectra(coarse, fine, preds, re_labels, out_dir, n_avg=20):
    re_labels_np = np.array(re_labels)
    unique_re    = sorted(set(re_labels_np.tolist()))
    for Re in unique_re:
        mask = re_labels_np == Re
        if mask.sum() == 0: continue
        c_re = coarse[mask]; f_re = fine[mask]; p_re = preds[mask]
        fig, ax = plt.subplots(figsize=(8, 5))
        datasets = [
            (f_re,  "Fine DNS",         "k",       2.0, "-"),
            (c_re,  "Coarse input",     "#888888", 1.5, "--"),
            (p_re,  "CNN prediction",   "#1D9E75", 2.0, "-"),
        ]
        for data, label, color, lw, ls in datasets:
            all_Ek = []
            for i in range(min(n_avg, len(data))):
                k, Ek = energy_spectrum(data[i,0].numpy(), data[i,1].numpy())
                all_Ek.append(Ek)
            Ek_mean = np.mean(all_Ek, axis=0)
            slope   = spectral_slope(k, Ek_mean)
            ax.loglog(k, Ek_mean, color=color, lw=lw, ls=ls,
                      label=f"{label}  (slope={slope:.3f})")
        all_Ek_fine = []
        for i in range(min(n_avg, len(f_re))):
            _, Ek_i = energy_spectrum(f_re[i,0].numpy(), f_re[i,1].numpy())
            all_Ek_fine.append(Ek_i)
        Ek_dns  = np.mean(all_Ek_fine, axis=0)
        idx     = np.argmin(np.abs(k - 5))
        k_ref   = np.array([5.0, float(k[-1])])
        E_ref   = Ek_dns[idx] * (k_ref / k[idx])**(-5/3)
        ax.loglog(k_ref, E_ref, "b:", lw=1.5, label=r"$k^{-5/3}$ reference")
        ax.set_xlabel("Wavenumber $k$", fontsize=12)
        ax.set_ylabel("$E(k)$", fontsize=12)
        ax.set_title(f"Energy spectrum — OOD Re={Re} (avg {n_avg} samples)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        fpath = os.path.join(out_dir, f"spectrum_ood_Re{Re}.png")
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        pdf_path = os.path.splitext(fpath)[0] + ".pdf"
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fpath}")

def plot_vorticity_ood(coarse, fine, preds, re_labels, out_dir):
    re_labels_np = np.array(re_labels)
    for Re in sorted(set(re_labels_np.tolist())):
        mask = re_labels_np == Re
        if mask.sum() == 0: continue
        idxs = np.where(mask)[0]
        vars_ = [np.var(fine[i, 3].numpy()) for i in idxs]
        idx   = idxs[np.argmax(vars_)]
        def vort(u, v, dx=1/64):
            w = np.zeros_like(u)
            w[1:-1,1:-1] = ((v[2:,1:-1]-v[:-2,1:-1])/(2*dx) -
                            (u[1:-1,2:]-u[1:-1,:-2])/(2*dx))
            return w
        wc  = vort(coarse[idx,0].numpy(), coarse[idx,1].numpy())
        wf  = vort(fine[idx,0].numpy(),   fine[idx,1].numpy())
        wp  = vort(preds[idx,0].numpy(),  preds[idx,1].numpy())
        err = np.abs(wf - wp)
        vmax = np.abs(wf).max() + 1e-8; emax = err.max() + 1e-8
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f"Vorticity — OOD Re={Re}",
                     fontsize=12, fontweight="bold")
        for ax, (w, title, cmap, v, sym) in zip(axes, [
            (wc,  "Coarse",        "seismic", vmax, True),
            (wf,  "Fine DNS",      "seismic", vmax, True),
            (wp,  "CNN prediction","seismic", vmax, True),
            (err, "|error|",       "hot",     emax, False),
        ]):
            im = ax.imshow(w, cmap=cmap, origin="lower",
                           vmin=-v if sym else 0, vmax=v)
            ax.set_title(title, fontsize=11); ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fpath = os.path.join(out_dir, f"vorticity_ood_Re{Re}.png")
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        pdf_path = os.path.splitext(fpath)[0] + ".pdf"
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fpath}")

def print_per_re_table(per_re, bicubic_overall=None):
    print(f"Per-Re OOD evaluation  (training Re not included)")
    print(f"  {'Re':>6}  {'n':>5}  {'SSIM':>8}  {'rel_l2':>8}")
    for Re in sorted(per_re.keys()):
        r = per_re[Re]
        print(f"  {Re:>6}  {r['n']:>5}  {r['ssim']:>8.4f}  "
              f"{r['rel_l2']*100:>7.1f}%")
    print(f"  {'-'*40}")
    ssim_mean  = np.mean([v['ssim']   for v in per_re.values()])
    rel_l2_mean= np.mean([v['rel_l2'] for v in per_re.values()])
    print(f"  {'Mean':>6}         {ssim_mean:>8.4f}  {rel_l2_mean*100:>7.1f}%")
    if bicubic_overall:
        print(f"  {'Bicubic':>6}         {bicubic_overall['ssim']:>8.4f}  "
              f"{bicubic_overall['rel_l2']*100:>7.1f}%")


def print_per_channel_table(ch_ssim):
    print(f"Per-channel SSIM on OOD test set")
    for name, val in ch_ssim.items():
        bar = "*" * int(val * 20)
        print(f"  {name:<15}  {val:.4f}  {bar}")



def main():
    args = get_args()
    seed = init_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Out: {args.out_dir}")
    model = TurbulenceUNet(in_ch=4, out_ch=4, base_ch=args.base_ch,
                           n_classes=N_CLASSES).to(device)
    ckpt_path = args.checkpoint
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded: {ckpt_path}  "
              f"epoch={ckpt.get('epoch','?')}  "
              f"ssim={ckpt.get('best_val_ssim','?')}")
    else:
        print(f"WARNING: checkpoint not found — using random weights")
    training_stats = np.load(args.stats_path, allow_pickle=True).item()
    print(f"Training stats loaded from {args.stats_path}")
    ch = CHANNEL_NAMES
    for i in range(4):
        print(f"  [{i}] {ch[i]:<15} mean={training_stats['mean'][i]:+.4f} "
              f"std={training_stats['std'][i]:.4f}")
    OOD_RE = [1800, 2200, 2800]
    solver_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "solver")
    if args.ood_snap_dir and os.path.exists(args.ood_snap_dir):
        ood_dir = args.ood_snap_dir
        print(f"\nUsing existing OOD snapshots from {ood_dir}")
    else:
        ood_dir = os.path.join(args.out_dir, "ood_snapshots")
        ood_dir = generate_ood_cavity_snapshots(
            OOD_RE, args.n_samples, ood_dir, solver_dir)
    coarse, fine, re_labels = load_ood_snapshots(ood_dir, training_stats, OOD_RE)
    print(f"\nEvaluating model on OOD cavity snapshots...")
    preds, metrics = evaluate_model(model, coarse, fine, device, training_stats)
    print(f"  MSE={metrics['mse']:.4f}  "
          f"RelL2={metrics['rel_l2']*100:.1f}%  "
          f"SSIM={metrics['ssim']:.4f}  "
          f"div={metrics['mean_div']:.4f}")
    print(f"\nComputing bicubic baseline...")
    bic_m = bicubic_baseline(coarse, fine, device)
    print(f"  Bicubic: MSE={bic_m['mse']:.4f}  "
          f"RelL2={bic_m['rel_l2']*100:.1f}%  SSIM={bic_m['ssim']:.4f}")
    print(f"  Improvement: SSIM +{metrics['ssim']-bic_m['ssim']:.4f}  "
          f"RelL2 {(bic_m['rel_l2']-metrics['rel_l2'])*100:.1f}% better")
    per_re = evaluate_per_re(model, coarse, fine, re_labels, device)
    print_per_re_table(per_re, bic_m)
    ch_ssim = evaluate_per_channel(model, coarse, fine, device)
    print_per_channel_table(ch_ssim)
    re_label_array = np.array(re_labels)
    try:
        plot_tsne(model, coarse, re_labels, device, args.out_dir)
    except Exception as e:
        print(f"  t-SNE skipped: {e}  (install scikit-learn if missing)")
    plot_ood_spectra(coarse, fine, preds, re_labels, args.out_dir)
    plot_vorticity_ood(coarse, fine, preds, re_labels, args.out_dir)
    if args.finetune:
        print(f"\nFine-tuning mode enabled (--finetune)")
        print(f"  Note: fine-tuning is NOT recommended for same-domain OOD")
        print(f"  The model already knows cavity physics — fine-tuning")
        print(f"  dec1 on a few OOD samples may hurt generalisation.")
    print(f"OOD Evaluation Summary")
    print(f"  OOD Re values tested: {OOD_RE}")
    print(f"  Model SSIM (aggregate): {metrics['ssim']:.4f}")
    print(f"  Bicubic SSIM:           {bic_m['ssim']:.4f}")
    print(f"  Improvement:           +{metrics['ssim']-bic_m['ssim']:.4f}")
    print(f"\n  Per-Re SSIM:")
    for Re, r in sorted(per_re.items()):
        print(f"    Re={Re}: {r['ssim']:.4f}")
    print(f"\n  Per-channel SSIM:")
    for name, val in ch_ssim.items():
        print(f"    {name:<15}: {val:.4f}")
    print(f"\n  Outputs in: {args.out_dir}/")
    print(f"    spectrum_ood_Re*.png")
    print(f"    vorticity_ood_Re*.png")
    print(f"    tsne_bottleneck.png")

if __name__ == "__main__":
    main()