import os
import glob
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .augmentation import make_coarse_field, random_flip, COARSE_SIZE, NOISE_STD
except ImportError:
    from augmentation import make_coarse_field, random_flip, COARSE_SIZE, NOISE_STD

FINE_SIZE   = 64
CHANNELS    = 4          # [u, v, p, ω]
RE_VALUES   = [100, 200, 400, 600, 800, 1000,1200, 1500, 1700, 2000, 2500, 3200]
RE_TO_LABEL = {re: i for i, re in enumerate(RE_VALUES)}
N_CLASSES   = len(RE_VALUES)   # 12
CHANNEL_NAMES = ["u-velocity", "v-velocity", "pressure", "vorticity ω"]
VEL_CLIP_CHANNELS   = [0, 1]   # u and v only
VEL_CLIP_THRESHOLD  = 10.0     # normalised units; physical ≈ 1.9 m/s for u

def label_histogram(labels):
    counts = {}
    for l in labels:
        counts[int(l)] = counts.get(int(l), 0) + 1
    return counts

def stratified_split_indices(labels, val_fraction, test_fraction, seed):
    labels_np = np.array(labels, dtype=int)
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []
    for lab in sorted(set(labels_np.tolist())):
        idx = np.where(labels_np == lab)[0]
        rng.shuffle(idx)
        n = len(idx)
        if n < 3:n_val = n_test = 0
        else:
            n_val  = max(1, int(round(n * val_fraction)))
            n_test = max(1, int(round(n * test_fraction)))
        n_train = n - n_val - n_test
        if n_train < 1:
            ov = 1 - n_train
            if n_test > 0:r = min(n_test, ov); n_test -= r; ov -= r
            if ov > 0 and n_val > 0:n_val -= min(n_val, ov)
            n_train = n - n_val - n_test
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train+n_val])
        test_idx.extend(idx[n_train+n_val:])
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx

def load_all_snapshots(data_dir: str, stats: dict = None):
    fields, labels = [], []
    skipped = 0
    re_dirs = sorted(glob.glob(os.path.join(data_dir, "Re_*")))
    for re_dir in re_dirs:
        re_val = int(re_dir.split("Re_")[-1])
        label  = RE_TO_LABEL.get(re_val, -1)
        if label < 0:
            print(f"Re={re_val} not in RE_TO_LABEL ( skipping ).")
            continue
        snap_files = sorted(glob.glob(os.path.join(re_dir, "*.npy")))
        if not snap_files:
            print(f"no snapshots in {re_dir}  skipping.")
            continue
        re_skipped = 0
        for fpath in snap_files:
            snap = np.load(fpath).astype(np.float32)
            if stats is not None:
                is_outlier = False
                for ch in VEL_CLIP_CHANNELS:
                    norm_ch = ((snap[ch] - stats["mean"][ch])/ stats["std"][ch])
                    if np.max(np.abs(norm_ch)) > VEL_CLIP_THRESHOLD:
                        is_outlier = True
                        break
                if is_outlier:
                    skipped += 1; re_skipped += 1
                    continue
                fields.append(snap)
            else:fields.append(snap)
            labels.append(label)
        kept = len(snap_files) - re_skipped
        print(f"  Re {re_val:5d}  label {label:2d}  "
              f"snapshots {kept:4d}"
              + (f"  (skipped {re_skipped} outliers)" if re_skipped else ""))
    print(f"\nTotal snapshots loaded: {len(fields)}"
          + (f"  (skipped {skipped} outliers total)" if skipped else ""))
    return fields, labels

def compute_stats(fields: list, output_dir: str = None) -> tuple:
    all_f = np.stack(fields, axis=0)   # (N, C, H, W)
    mean  = all_f.mean(axis=(0,2,3))
    std   = all_f.std(axis=(0,2,3)).clip(min=1e-8)
    stats = {"mean": mean, "std": std}
    omega_mean = mean[3]
    assert omega_mean < 0, (f"ω_mean = {omega_mean:.4f} is non-negative.")
    print(f"  ω_mean = {omega_mean:.4f} (negative = clockwise = correct )")
    save_path = (os.path.join(output_dir, "stats.npy")if output_dir else "stats.npy")
    if output_dir:os.makedirs(output_dir, exist_ok=True)
    np.save(save_path, stats)
    ch = CHANNEL_NAMES
    print(f"\nNormalisation statistics:")
    for i in range(len(mean)):
        print(f" [{i}] {ch[i]:<15}  mean={mean[i]:+.4f}  std={std[i]:.4f}")
    print(f" Saved  {save_path}")
    return stats, save_path

def normalise(field, stats):
    return (field - stats["mean"][:,None,None]) / stats["std"][:,None,None]

def denormalise(field, stats):
    return field * stats["std"][:,None,None] + stats["mean"][:,None,None]

class TurbulenceDataset(Dataset):
    def __init__(self, fields, labels, stats,
                 augment=True, noise_std=NOISE_STD):
        self.stats     = stats
        self.augment   = augment
        self.noise_std = noise_std
        self.items     = []
        for field, label in zip(fields, labels):
            self.items.append((normalise(field, stats).astype(np.float32),int(label)))
        print(f"  Dataset: {len(self.items)} samples  "
              f"augment={augment}  noise_std={noise_std}")
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        fine_np, label = self.items[idx]
        if self.augment:
            fine_np = random_flip(fine_np)
        coarse_np = make_coarse_field(fine_np, COARSE_SIZE, self.noise_std)
        return (torch.from_numpy(coarse_np),torch.from_numpy(fine_np.copy()),torch.tensor(label, dtype=torch.long))

def make_dataloaders(data_dir,batch_size=16,val_fraction=0.15,
                     test_fraction=0.05,num_workers=0,seed=67,
                     output_dir =None,max_per_class=None,):
    print(f"\nBuilding dataset from: {data_dir}")
    if seed is None:seed = int(np.random.default_rng().integers(0, 2**32 - 1))
    print(f"RNG seed: {seed}")
    fields_all, labels_all = load_all_snapshots(data_dir, stats=None)
    stats, stats_path = compute_stats(fields_all, output_dir=output_dir)
    print("\nSecond pass — applying outlier filter...")
    fields, labels = load_all_snapshots(data_dir, stats=stats)
    if max_per_class is not None:
        counts = Counter(labels)
        if max_per_class == "auto":
            median_count = int(np.median(list(counts.values())))
            cap = median_count * 4
            print(f"\nAuto cap: 4 X median({median_count}) = {cap} per class")
        else:
            cap = int(max_per_class)
            print(f"\nHard cap: {cap} per class")
        capped_f, capped_l = [], []
        seen = Counter()
        for f, l in zip(fields, labels):
            if seen[l] < cap:capped_f.append(f); capped_l.append(l); seen[l] += 1
        old_n = len(fields); fields = capped_f; labels = capped_l
        print(f"After cap: {len(fields)} samples (removed {old_n-len(fields)})")
        new_counts = Counter(labels)
        print(f"New imbalance ratio: "f"{max(new_counts.values())/min(new_counts.values()):.1f}X")
    label_counts = label_histogram(labels)
    print(f"\nLabel distribution:")
    for k in sorted(label_counts):
        re = RE_VALUES[k] if k < len(RE_VALUES) else "?"
        print(f"  class {k:2d}  Re={re:5d}  n={label_counts[k]:4d}")
    torch.manual_seed(seed)
    tr_idx, va_idx, te_idx = stratified_split_indices(labels, val_fraction, test_fraction, seed)
    def _sel(idx): return [fields[i] for i in idx], [labels[i] for i in idx]
    tr_f, tr_l = _sel(tr_idx)
    va_f, va_l = _sel(va_idx)
    te_f, te_l = _sel(te_idx)
    print(f"\nSplit: train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")
    train_ds = TurbulenceDataset(tr_f, tr_l, stats, augment=True,  noise_std=NOISE_STD)
    val_ds   = TurbulenceDataset(va_f, va_l, stats, augment=False, noise_std=0.0)
    test_ds  = TurbulenceDataset(te_f, te_l, stats, augment=False, noise_std=0.0)
    n_total = len(labels)
    counts  = Counter(labels)
    class_weights = torch.tensor([n_total/(N_CLASSES*counts.get(i, 1)) for i in range(N_CLASSES)],dtype=torch.float32)
    print(f"\nClass weights: {class_weights.numpy().round(3)}")
    gen = torch.Generator(); gen.manual_seed(seed)
    def _wi(wid): np.random.seed(seed+wid+1); torch.manual_seed(seed+wid+1)
    def _dl(ds, shuffle, drop):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True,
                          drop_last=drop, generator=gen, worker_init_fn=_wi)
    train_loader = _dl(train_ds, True,  True)
    val_loader   = _dl(val_ds,   False, False)
    test_loader  = _dl(test_ds,  False, False)
    print(f"\nDataloaders:  train={len(train_loader)} batches  "
          f"val={len(val_loader)}  test={len(test_loader)}")
    return train_loader, val_loader, test_loader, stats, class_weights, stats_path

def visualise_sample(dataset, idx=0, stats=None, output_dir=None):
    coarse, fine, label = dataset[idx]
    c_np = coarse.numpy(); f_np = fine.numpy()
    C    = c_np.shape[0]
    ch   = CHANNEL_NAMES[:C]
    re_str = (f"Re={RE_VALUES[label.item()]}"
              if label.item() < len(RE_VALUES) else f"label={label.item()}")
    fig, axes = plt.subplots(2, C, figsize=(4*C, 8))
    fig.suptitle(f"Coarse - Fine pair  |  sample {idx}  |  {re_str}",fontsize=13, fontweight="bold")
    for ch_i in range(C):
        vmax = max(np.abs(f_np[ch_i]).max(), 1e-8)
        for row, arr, prefix in [(0,c_np,"Coarse"),(1,f_np,"Fine (DNS)")]:
            ax = axes[row, ch_i]
            im = ax.imshow(arr[ch_i], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           origin="lower")
            ax.set_title(f"{prefix} — {ch[ch_i]}", fontsize=10)
            ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    sp = (os.path.join(output_dir, "dataset_preview.png")if output_dir else "dataset_preview.png")
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    plt.savefig(sp, dpi=150, bbox_inches="tight")
    pdf_path = os.path.splitext(sp)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {sp}")


def visualise_downsampling_quality(dataset, n_samples=4, output_dir=None):
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3.5*n_samples))
    fig.suptitle("Coarse  |  Fine DNS  |  |difference|  (u-velocity)",
                 fontsize=13, fontweight="bold")
    for row in range(n_samples):
        c, f, _ = dataset[row]
        cn = c[0].numpy(); fn = f[0].numpy()
        diff = np.abs(fn - cn); vmax = max(np.abs(fn).max(), 1e-8)
        axes[row,0].imshow(cn,   cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        axes[row,0].set_title("Coarse (u)", fontsize=10); axes[row,0].axis("off")
        axes[row,1].imshow(fn,   cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        axes[row,1].set_title("Fine DNS (u)", fontsize=10); axes[row,1].axis("off")
        im = axes[row,2].imshow(diff, cmap="hot", origin="lower")
        axes[row,2].set_title("|coarse - fine|", fontsize=10); axes[row,2].axis("off")
        plt.colorbar(im, ax=axes[row,2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    sp = (os.path.join(output_dir, "coarse_vs_fine.png")if output_dir else "coarse_vs_fine.png")
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    plt.savefig(sp, dpi=150, bbox_inches="tight")
    pdf_path = os.path.splitext(sp)[0] + ".pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {sp}")

def print_dataset_stats(train_loader, val_loader, test_loader, stats):
    c, f, lbl = next(iter(train_loader))
    C = f.shape[1]; ch = CHANNEL_NAMES[:C]
    print(f"\nDataset summary")
    print(f"  Train: {len(train_loader)} batches  "
          f"Val: {len(val_loader)}  Test: {len(test_loader)}")
    print(f"\n  Batch shapes:")
    print(f"    coarse : {tuple(c.shape)}  (U-Net input)")
    print(f"    fine   : {tuple(f.shape)}    (U-Net target)")
    print(f"    label  : {tuple(lbl.shape)}   (classifier target)")
    print(f"\n  Normalisation:")
    for i, name in enumerate(ch):
        print(f"    [{i}] {name:<15}  "
              f"mean={stats['mean'][i]:+.4f}  std={stats['std'][i]:.4f}")
    print(f"\n  coarse range: [{c.min():.3f}, {c.max():.3f}]")
    print(f"  fine   range: [{f.min():.3f}, {f.max():.3f}]")
    if c.max() > VEL_CLIP_THRESHOLD or c.min() < -VEL_CLIP_THRESHOLD:
        print(f"\n  WARNING: coarse range exceeds ±6sigma — outlier snapshots "
              f"may still be present. Increase t_start_save in NSconfig.")
    else:
        print(f"  coarse range within ±6sigma ")

if __name__ == "__main__":
    root        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    snap_dir    = os.path.join(root, "snapshots")
    out_dir     = os.path.join(root, "train-data")
    result = make_dataloaders(data_dir= snap_dir,batch_size= 16,
                              val_fraction = 0.15,test_fraction= 0.05,
                              output_dir= out_dir,max_per_class= "auto",seed=None)
    train_loader, val_loader, test_loader, stats, class_weights, stats_path = result
    print(f"\nstats.npy saved to: {stats_path}")
    print_dataset_stats(train_loader, val_loader, test_loader, stats)
    train_ds = train_loader.dataset
    visualise_sample(train_ds, idx=0, stats=stats, output_dir=out_dir)
    visualise_downsampling_quality(train_ds, n_samples=4, output_dir=out_dir)