import os
import sys
import json
import argparse
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import config

try:
    from .augmentation import make_coarse_field, random_flip, random_shift, COARSE_FACTOR, NOISE_STD
except ImportError:
    from augmentation import make_coarse_field, random_flip, random_shift, COARSE_FACTOR, NOISE_STD

CHANNEL_NAMES = config.CHANNEL_NAMES

@dataclass
class Snapshots:
    """Snapshots in physical units plus per-snapshot labels."""
    fields: np.ndarray     # (N, 4, H, W) float32 [u, v, p, ω], indexed [x, y]
    re:     np.ndarray     # (N,) int
    t:      np.ndarray     # (N,) snapshot time (NaN if unknown)
    kind:   np.ndarray     # (N,) "cavity", "forced" or "decaying"
    seed:   np.ndarray     # (N,) run seed (0 for the cavity)
    domain: str            # "cavity" (walls, dx = L/(N-1)) or "periodic" (dx = L/N)
    L:      float          # domain side length

    def __len__(self): return len(self.re)
    def subset(self, mask):
        return Snapshots(self.fields[mask], self.re[mask], self.t[mask], self.kind[mask],
                         self.seed[mask], self.domain, self.L)
    @staticmethod
    def concat(parts):
        parts = [p for p in parts if p is not None and len(p)]
        if not parts:
            return None
        if len({(p.domain, p.L, p.fields.shape[-1]) for p in parts}) > 1:
            raise ValueError("Cannot mix datasets with different domains or grid sizes")
        cat = lambda a: np.concatenate([getattr(p, a) for p in parts])
        return Snapshots(cat("fields"), cat("re"), cat("t"), cat("kind"), cat("seed"), parts[0].domain, parts[0].L)

def _run_dirs(root):
    """Every directory under root that holds snap_*.npy files."""
    out = []
    for d, _, files in os.walk(root):
        if any(f.startswith("snap_") and f.endswith(".npy") for f in files):
            out.append(d)
    return sorted(out)

def load_snapshots(data_dirs, re_values=None) -> Snapshots:
    """Load every snap_*.npy below one or more directories.

    Works for the cavity layout (Re_*/snap_*.npy) and the turbulence layout
    (<kind>/Re_*/seed_*/snap_*.npy); Re, flow kind, domain and times come from
    the meta.json next to the snapshots.
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    parts = []
    for root in data_dirs:
        for d in _run_dirs(root):
            meta_path = os.path.join(d, "meta.json")
            meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
            re_val = int(meta.get("Re", d.split("Re_")[-1].split(os.sep)[0]))
            if re_values is not None and re_val not in re_values:
                continue
            files = sorted(f for f in os.listdir(d) if f.startswith("snap_") and f.endswith(".npy"))
            t = meta.get("times", [])
            if len(t) != len(files):
                t = [float("nan")] * len(files)
            n = len(files)
            fields = np.stack([np.load(os.path.join(d, f)).astype(np.float32) for f in files])
            parts.append(Snapshots(fields, np.full(n, re_val, dtype=np.int64), np.asarray(t, dtype=np.float64),
                                   np.full(n, meta.get("flow", "cavity")), np.full(n, meta.get("seed", 0)),
                                   meta.get("domain", "cavity"), float(meta.get("L", 1.0))))
    snaps = Snapshots.concat(parts)
    if snaps is None:
        raise RuntimeError(f"No snapshots found under {data_dirs}")
    for k in sorted(set(snaps.kind.tolist())):
        m = snaps.kind == k
        print(f"  {k:<9} Re={sorted(set(snaps.re[m].tolist()))}  runs={len(set(zip(snaps.re[m], snaps.seed[m])))}  snapshots={int(m.sum())}")
    print(f"Total snapshots loaded from {data_dirs}: {len(snaps)}  ({snaps.domain}, grid {snaps.fields.shape[-1]}²)")
    return snaps

def compute_stats(snaps: Snapshots, val_re=()) -> dict:
    """Per-channel mean/std of the fields, mean/std of log10(Re), and the domain.
    Stored as plain types so the dict can live in a weights_only checkpoint."""
    fields = snaps.fields
    mean = fields.mean(axis=(0, 2, 3))
    std  = fields.std(axis=(0, 2, 3)).clip(min=1e-8)
    log_re = np.log10(np.unique(snaps.re))
    stats = {"mean": mean.tolist(), "std": std.tolist(),
             "log_re_mean": float(log_re.mean()), "log_re_std": float(max(log_re.std(), 1e-8)),
             "train_re": sorted(int(r) for r in np.unique(snaps.re)),
             "val_re": sorted(int(r) for r in val_re),
             "domain": snaps.domain, "L": snaps.L, "N": int(fields.shape[-1])}
    if snaps.domain == "cavity" and mean[3] >= 0:
        print(f"  WARNING: ω_mean = {mean[3]:.4f} >= 0, expected < 0 for a clockwise primary vortex")
    print(f"\nNormalisation statistics (training split):")
    for i in range(len(mean)):
        print(f" [{i}] {CHANNEL_NAMES[i]:<15}  mean={mean[i]:+.4f}  std={std[i]:.4f}")
    return stats

def is_periodic(stats) -> bool:
    return stats.get("domain", "cavity") == "periodic"

def _arr(stats, key):
    return np.asarray(stats[key], dtype=np.float32)[:, None, None]

def normalise(field, stats):
    return (field - _arr(stats, "mean")) / _arr(stats, "std")

def denormalise(field, stats):
    return field * _arr(stats, "std") + _arr(stats, "mean")

def normalise_re(re, stats):
    return (np.log10(re) - stats["log_re_mean"]) / stats["log_re_std"]

def denormalise_re(z, stats):
    return 10.0 ** (z * stats["log_re_std"] + stats["log_re_mean"])

class TurbulenceDataset(Dataset):
    """Yields (coarse input, fine target, normalised log10 Re), fields normalised."""
    def __init__(self, snaps: Snapshots, stats, augment=True, noise_std=NOISE_STD):
        self.snaps     = snaps             # physical units; flips happen before normalising
        self.re        = snaps.re
        self.stats     = stats
        self.periodic  = is_periodic(stats)
        self.augment   = augment
        self.noise_std = noise_std
        self.re_target = normalise_re(snaps.re.astype(np.float64), stats).astype(np.float32)
        print(f"  Dataset: {len(snaps)} samples  Re={sorted(set(snaps.re.tolist()))}  "
              f"augment={augment}  noise_std={noise_std}")
    def __len__(self): return len(self.snaps)
    def __getitem__(self, idx):
        fine_np = self.snaps.fields[idx]
        if self.augment:
            fine_np = random_flip(fine_np)
            if self.periodic:
                fine_np = random_shift(fine_np)
        fine   = torch.from_numpy(np.ascontiguousarray(normalise(fine_np, self.stats), dtype=np.float32))
        coarse = make_coarse_field(fine, COARSE_FACTOR, self.noise_std, periodic=self.periodic)
        return coarse, fine, torch.tensor(self.re_target[idx])

def split_by_re(re, val_re, max_train_re=None):
    """Boolean masks (train, val, held_out) over snapshots, by whole Re value.

    val_re are held out for validation. With max_train_re, every Re above it is
    removed from training and validation and goes to the test set, so the test
    is a genuine extrapolation in Re.
    """
    held_out = re > max_train_re if max_train_re is not None else np.zeros(len(re), bool)
    val      = np.isin(re, val_re) & ~held_out
    train    = ~val & ~held_out
    if not train.any() or not val.any():
        raise ValueError(f"val_re={val_re}, max_train_re={max_train_re} leave no training or no validation data "
                         f"among Re {sorted(set(re.tolist()))}")
    return train, val, held_out

def make_dataloaders(data_dirs, val_re, test_dirs=None, max_train_re=None, batch_size=16,
                     num_workers=0, seed=67, noise_std=NOISE_STD):
    """Split by whole Re values: val_re are held out of training entirely, and
    the test set comes from separate directories of unseen Re plus, with
    max_train_re, every training-directory Re above that cut-off."""
    print(f"\nBuilding dataset from: {data_dirs}")
    snaps = load_snapshots(data_dirs)
    train, val, held_out = split_by_re(snaps.re, val_re, max_train_re)
    stats = compute_stats(snaps.subset(train), val_re=sorted(set(snaps.re[val].tolist())))
    print(f"\nSplit by Re:  train={int(train.sum())}  val={int(val.sum())}  moved to test={int(held_out.sum())}")
    train_ds = TurbulenceDataset(snaps.subset(train), stats, augment=True,  noise_std=noise_std)
    val_ds   = TurbulenceDataset(snaps.subset(val),   stats, augment=False, noise_std=0.0)
    test_parts = [snaps.subset(held_out)]
    for d in ([test_dirs] if isinstance(test_dirs, str) else (test_dirs or [])):
        if os.path.isdir(d):
            test_parts.append(load_snapshots(d))
    test = Snapshots.concat(test_parts)
    test_ds = TurbulenceDataset(test, stats, augment=False, noise_std=0.0) if test is not None else None
    gen = torch.Generator(); gen.manual_seed(seed)
    def _wi(wid): np.random.seed(seed+wid+1); torch.manual_seed(seed+wid+1)
    def _dl(ds, shuffle, drop):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                          drop_last=drop, generator=gen, worker_init_fn=_wi)
    train_loader = _dl(train_ds, True,  True)
    val_loader   = _dl(val_ds,   False, False)
    test_loader  = _dl(test_ds,  False, False) if test_ds is not None else None
    print(f"\nDataloaders:  train={len(train_loader)} batches  val={len(val_loader)}"
          + (f"  test={len(test_loader)}" if test_loader else "  test=none"))
    return train_loader, val_loader, test_loader, stats

def visualise_sample(dataset, idx=0, output_dir=None):
    coarse, fine, _ = dataset[idx]
    c_np = coarse.numpy(); f_np = fine.numpy()
    C    = c_np.shape[0]
    fig, axes = plt.subplots(2, C, figsize=(4*C, 8))
    fig.suptitle(f"Coarse - Fine pair  |  sample {idx}  |  {dataset.snaps.kind[idx]}  Re={dataset.re[idx]}",fontsize=13, fontweight="bold")
    for ch_i in range(C):
        vmax = max(np.abs(f_np[ch_i]).max(), 1e-8)
        for row, arr, prefix in [(0,c_np,"Coarse"),(1,f_np,"Fine (DNS)")]:
            ax = axes[row, ch_i]
            im = ax.imshow(arr[ch_i].T, cmap="RdBu_r", vmin=-vmax, vmax=vmax,   # [x, y] -> rows = y
                           origin="lower")
            ax.set_title(f"{prefix} — {CHANNEL_NAMES[ch_i]}", fontsize=10)
            ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    sp = (os.path.join(output_dir, "dataset_preview.png")if output_dir else "dataset_preview.png")
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    plt.savefig(sp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {sp}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--flow", choices=sorted(config.FLOWS), default=config.DEFAULT_FLOW)
    p.add_argument("--data_dir", nargs="+", default=None)
    p.add_argument("--test_dir", nargs="+", default=None)
    p.add_argument("--val_re",   type=int, nargs="+", default=None)
    p.add_argument("--out_dir",  default=os.path.join(config.PROJECT_ROOT, "train-data"))
    args = p.parse_args()
    flow = config.FLOWS[args.flow]
    train_loader, val_loader, test_loader, stats = make_dataloaders(
        args.data_dir or flow["train_dirs"], val_re=args.val_re or flow["val_re"],
        test_dirs=args.test_dir or flow["test_dirs"])
    c, f, r = next(iter(train_loader))
    print(f"\nBatch shapes: coarse {tuple(c.shape)}  fine {tuple(f.shape)}  re {tuple(r.shape)}")
    visualise_sample(train_loader.dataset, idx=0, output_dir=args.out_dir)
