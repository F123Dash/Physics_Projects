import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
try:
    from .augmentation import make_coarse_field as aug_make_coarse_field, random_flip
except ImportError:
    from augmentation import make_coarse_field as aug_make_coarse_field, random_flip


FINE_SIZE   = 64
COARSE_SIZE = 16
CHANNELS    = 3
NOISE_STD   = 0.05

RE_TO_LABEL = {100: 0, 400: 1, 1000: 2, 5000: 3, 10000: 4}


def label_histogram(labels):
    counts = {}
    for lab in labels:
        lab_i = int(lab)
        counts[lab_i] = counts.get(lab_i, 0) + 1
    return counts


def stratified_split_indices(labels, val_fraction, test_fraction, seed):
    labels_np = np.array(labels, dtype=int)
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []
    for lab in sorted(set(labels_np.tolist())):
        idx = np.where(labels_np == lab)[0]
        rng.shuffle(idx)
        n_total = len(idx)
        if n_total < 3:
            n_val = 0
            n_test = 0
        else:
            n_val = max(1, int(round(n_total * val_fraction)))
            n_test = max(1, int(round(n_total * test_fraction)))

        n_train = n_total - n_val - n_test
        if n_train < 1:
            overflow = 1 - n_train
            if n_test > 0:
                reduce = min(n_test, overflow)
                n_test -= reduce
                overflow -= reduce
            if overflow > 0 and n_val > 0:
                reduce = min(n_val, overflow)
                n_val -= reduce
                overflow -= reduce
            n_train = n_total - n_val - n_test

        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def load_all_snapshots(data_dir: str):
    fields = []
    labels = []

    re_dirs = sorted(glob.glob(os.path.join(data_dir, "Re_*")))
    if not re_dirs:
        raise FileNotFoundError(
            f"No Re_* directories found in '{data_dir}'.\n"
            f"Run ns_solver_v2.py first to generate snapshots."
        )

    for re_dir in re_dirs:
        re_val = int(re_dir.split("Re_")[-1])
        label  = RE_TO_LABEL.get(re_val, -1)
        if label < 0:
            print(f"  Warning: Re={re_val} not in RE_TO_LABEL, skipping.")
            continue

        snap_files = sorted(glob.glob(os.path.join(re_dir, "*.npy")))
        if not snap_files:
            print(f"  Warning: no snapshots found in {re_dir}, skipping.")
            continue

        for fpath in snap_files:
            snap = np.load(fpath).astype(np.float32)
            assert snap.shape == (CHANNELS, FINE_SIZE, FINE_SIZE), \
                f"Unexpected snapshot shape {snap.shape} in {fpath}"
            fields.append(snap)
            labels.append(label)

        print(f"  Loaded {len(snap_files):4d} snapshots from Re={re_val}")

    print(f"\nTotal snapshots loaded: {len(fields)}")
    return fields, labels


def compute_stats(fields: list, output_dir: str = None) -> dict:
    all_fields = np.stack(fields, axis=0)

    mean = all_fields.mean(axis=(0, 2, 3))
    std  = all_fields.std(axis=(0, 2, 3))

    std = np.maximum(std, 1e-8)

    stats = {'mean': mean, 'std': std}
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "stats.npy")
    else:
        save_path = "stats.npy"
    
    np.save(save_path, stats)

    print(f"\nNormalisation statistics (per channel [u, v, p]):")
    print(f"  mean = {mean}")
    print(f"  std  = {std}")
    print(f"  saved to {save_path}")
    return stats


def normalise(field: np.ndarray, stats: dict) -> np.ndarray:
    mean = stats['mean'][:, None, None]
    std  = stats['std'][:, None, None]
    return (field - mean) / std


def denormalise(field: np.ndarray, stats: dict) -> np.ndarray:
    mean = stats['mean'][:, None, None]
    std  = stats['std'][:, None, None]
    return field * std + mean

class TurbulenceDataset(Dataset):
    def __init__(self, fields, labels, stats,
                 augment=True, noise_std=NOISE_STD):
        self.stats     = stats
        self.augment   = augment
        self.noise_std = noise_std
        self.items     = []

        for field, label in zip(fields, labels):
            norm = normalise(field, stats)
            self.items.append((norm, label))

        print(f"Dataset ready: {len(self.items)} samples, "
              f"augment={augment}, noise_std={noise_std}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fine_np, label = self.items[idx]

        if self.augment:
            fine_np = random_flip(fine_np)

        coarse_np = aug_make_coarse_field(fine_np, coarse_size=COARSE_SIZE)
        if self.noise_std == 0:
            C, H, W = fine_np.shape
            factor = H // COARSE_SIZE
            coarse_small = fine_np.reshape(
                C, COARSE_SIZE, factor, COARSE_SIZE, factor
            ).mean(axis=(2, 4))
            from scipy.ndimage import zoom as _zoom
            zoom_factor = H / COARSE_SIZE
            coarse_np = np.stack([
                _zoom(coarse_small[c], zoom_factor, order=1)
                for c in range(C)
            ], axis=0).astype(np.float32)

        coarse = torch.from_numpy(coarse_np)
        fine   = torch.from_numpy(fine_np.copy())
        label  = torch.tensor(label, dtype=torch.long)

        return coarse, fine, label


def make_dataloaders(data_dir: str,
                     batch_size: int = 16,
                     val_fraction: float = 0.15,
                     test_fraction: float = 0.05,
                     num_workers: int = 0,
                     seed: int = 42,
                     output_dir: str = None):
    print(f"Building turbulence dataset from: {data_dir}")

    fields, labels = load_all_snapshots(data_dir)

    if len(fields) == 0:
        raise ValueError("No snapshots loaded. Run the NS solver first.")

    stats = compute_stats(fields, output_dir=output_dir)

    label_counts = label_histogram(labels)
    label_info = ", ".join([f"{k}:{v}" for k, v in sorted(label_counts.items())])
    print(f"Label distribution: {label_info}")
    if len(label_counts) < 2:
        print("Warning: only one class present; accuracy will be trivial.")

    torch.manual_seed(seed)
    train_idx, val_idx, test_idx = stratified_split_indices(
        labels, val_fraction, test_fraction, seed
    )

    n_train = len(train_idx)
    n_val   = len(val_idx)
    n_test  = len(test_idx)

    train_fields = [fields[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_fields   = [fields[i] for i in val_idx]
    val_labels   = [labels[i] for i in val_idx]
    test_fields  = [fields[i] for i in test_idx]
    test_labels  = [labels[i] for i in test_idx]

    print(f"\nSplit: train={n_train}  val={n_val}  test={n_test}")
    print(f"  Train labels: {label_histogram(train_labels)}")
    print(f"  Val labels  : {label_histogram(val_labels)}")
    print(f"  Test labels : {label_histogram(test_labels)}")

    train_ds = TurbulenceDataset(train_fields, train_labels, stats,
                                  augment=True,  noise_std=NOISE_STD)
    val_ds   = TurbulenceDataset(val_fields,   val_labels,   stats,
                                  augment=False, noise_std=0.0)
    test_ds  = TurbulenceDataset(test_fields,  test_labels,  stats,
                                  augment=False, noise_std=0.0)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True,  num_workers=num_workers,
                               pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                               shuffle=False, num_workers=num_workers,
                               pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                               shuffle=False, num_workers=num_workers,
                               pin_memory=True)

    print(f"\nDataloaders ready:")
    print(f"  train: {len(train_loader)} batches × batch_size={batch_size}")
    print(f"  val:   {len(val_loader)} batches")
    print(f"  test:  {len(test_loader)} batches")

    return train_loader, val_loader, test_loader, stats


def visualise_sample(dataset: TurbulenceDataset,
                     idx: int = 0,
                     stats: dict = None,
                     output_dir: str = None):
    coarse, fine, label = dataset[idx]
    coarse_np = coarse.numpy()
    fine_np   = fine.numpy()

    channel_names = ["u-velocity", "v-velocity", "pressure"]
    re_names = {0: "Re=100", 1: "Re=400", 2: "Re=1000",
                3: "Re=5000", 4: "Re=10000"}
    re_str = re_names.get(label.item(), f"Re_label={label.item()}")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(f"Coarse-Fine Pair  |  Sample {idx}  |  {re_str}",
                 fontsize=13, fontweight='bold')

    for ch in range(3):
        vmax = max(np.abs(fine_np[ch]).max(), 1e-8)
        vmin = -vmax

        ax = axes[0, ch]
        im = ax.imshow(coarse_np[ch], cmap='RdBu_r',
                        vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f"Coarse — {channel_names[ch]}", fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, ch]
        im = ax.imshow(fine_np[ch], cmap='RdBu_r',
                        vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f"Fine (DNS) — {channel_names[ch]}", fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        savepath = os.path.join(output_dir, "dataset_preview.png")
    else:
        savepath = "dataset_preview.png"
    
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    print(f"Saved: {savepath}")
    plt.close(fig)


def visualise_downsampling_quality(dataset: TurbulenceDataset, n_samples=4, output_dir: str = None):
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3.5 * n_samples))
    fig.suptitle("Coarse input  |  Fine DNS target  |  Difference",
                 fontsize=13, fontweight='bold')

    for row in range(n_samples):
        coarse, fine, label = dataset[row]
        c = coarse[0].numpy()
        f = fine[0].numpy()
        diff = np.abs(f - c)

        vmax = max(np.abs(f).max(), 1e-8)

        axes[row, 0].imshow(c, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             origin='lower')
        axes[row, 0].set_title("Coarse input (u)", fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(f, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             origin='lower')
        axes[row, 1].set_title("Fine DNS (u)", fontsize=10)
        axes[row, 1].axis('off')

        im = axes[row, 2].imshow(diff, cmap='hot', origin='lower')
        axes[row, 2].set_title("Missing detail |coarse−fine|", fontsize=10)
        axes[row, 2].axis('off')
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        savepath = os.path.join(output_dir, "coarse_vs_fine.png")
    else:
        savepath = "coarse_vs_fine.png"
    
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    print(f"Saved: {savepath}")
    plt.close(fig)


def print_dataset_stats(train_loader, val_loader, test_loader, stats):
    coarse, fine, label = next(iter(train_loader))
    print(f"Dataset summary")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")
    print(f"\n  Batch shapes:")
    print(f"    coarse : {tuple(coarse.shape)}  (U-Net input)")
    print(f"    fine   : {tuple(fine.shape)}   (U-Net target)")
    print(f"    label  : {tuple(label.shape)}  (classifier target)")
    print(f"\n  Normalisation:")
    print(f"    mean [u,v,p] = {stats['mean'].round(4)}")
    print(f"    std  [u,v,p] = {stats['std'].round(4)}")
    print(f"\n  coarse value range: [{coarse.min():.3f}, {coarse.max():.3f}]")
    print(f"  fine   value range: [{fine.min():.3f},   {fine.max():.3f}]")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    snapshot_dir = os.path.join(project_root, "snapshots")
    train_data_dir = os.path.join(project_root, "train-data")
    
    train_loader, val_loader, test_loader, stats = make_dataloaders(
        data_dir   = snapshot_dir,
        batch_size = 16,
        val_fraction  = 0.15,
        test_fraction = 0.05,
        output_dir = train_data_dir,
    )

    print_dataset_stats(train_loader, val_loader, test_loader, stats)

    train_ds = train_loader.dataset
    visualise_sample(train_ds, idx=0, stats=stats, output_dir=train_data_dir)
    visualise_downsampling_quality(train_ds, n_samples=4, output_dir=train_data_dir)