import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from train.train       import build_model
from data.dataset      import Snapshots, load_snapshots, normalise, normalise_re, is_periodic
from data.augmentation import make_coarse_field, block_average
from models.losses     import denormalise_t
from solver.ns_solver  import centerline_u, centerline_v, primary_vortex
from train.metrics     import (sample_metrics, re_metrics, baseline_prediction, energy_spectrum,
                               spectral_band, MetricAccumulator, format_table, SHORT)

def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--flow",choices=sorted(config.FLOWS),default=config.DEFAULT_FLOW,
                   help="Dataset preset from config.FLOWS (sets the default directories)")
    p.add_argument("--checkpoint",default=None,help="Default: runs/<flow>_01/best_model.pt")
    p.add_argument("--ood_dir",nargs="+",default=None,help="Snapshots at Re never used in training")
    p.add_argument("--train_dir",nargs="+",default=None,help="Training snapshots, for the nearest-neighbour baseline and t-SNE")
    p.add_argument("--out_dir",default=None,help="Default: runs/eval_<flow>")
    p.add_argument("--batch_size",type=int,default=32)
    args = p.parse_args()
    flow = config.FLOWS[args.flow]
    args.checkpoint = args.checkpoint or os.path.join(config.RUNS_ROOT, f"{args.flow}_01", "best_model.pt")
    args.ood_dir    = args.ood_dir    or flow["test_dirs"]
    args.train_dir  = args.train_dir  or flow["train_dirs"]
    args.out_dir    = args.out_dir    or os.path.join(config.RUNS_ROOT, f"eval_{args.flow}")
    return args

def load_model(path, device):
    ckpt  = torch.load(path, map_location=device, weights_only=True)
    stats = ckpt["stats"]
    model = build_model(ckpt["args"], stats).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    score = ckpt.get("best_val_score")
    print(f"Loaded: {path}  epoch={ckpt['epoch']}  domain={stats.get('domain', 'cavity')}  "
          f"divergence-free projection={model.project}"
          + (f"  best val relL2(all)={score:.2%}" if score is not None else f"  best val loss={ckpt['best_val_loss']:.4f}"))
    return model, stats

def to_normalised(fields, stats):
    return torch.from_numpy(np.stack([normalise(f, stats) for f in fields])).float()

@torch.no_grad()
def predict(model, snaps: Snapshots, stats, device, batch_size):
    """-> normalised fine, CNN prediction, normalised Re target and prediction."""
    fine = to_normalised(snaps.fields, stats)
    preds, re_preds = [], []
    for i in range(0, len(fine), batch_size):
        p, r = model(make_coarse_field(fine[i:i+batch_size].to(device), periodic=is_periodic(stats)))
        preds.append(p.cpu()); re_preds.append(r.cpu())
    re_true = torch.from_numpy(normalise_re(snaps.re.astype(np.float64), stats)).float()
    return fine, torch.cat(preds), re_true, torch.cat(re_preds)

def metrics_for(pred, fine, stats, re_pred=None, re_true=None):
    acc = MetricAccumulator()
    for i in range(0, len(fine), 256):
        m = sample_metrics(pred[i:i+256], fine[i:i+256], stats)
        if re_pred is not None:
            m.update(re_metrics(re_pred[i:i+256], re_true[i:i+256], stats))
        acc.add(m)
    return acc.result()

def nearest_train_prediction(fine, train_fields, stats):
    train_fine = to_normalised(train_fields, stats)
    c_tr = block_average(train_fine).flatten(1); c_te = block_average(fine).flatten(1)
    idx = torch.cat([torch.cdist(c_te[i:i+256], c_tr).argmin(dim=1) for i in range(0, len(c_te), 256)])
    return train_fine[idx]

def groups(snaps: Snapshots):
    """(label, mask) per flow kind and Re, in a stable order."""
    out = []
    for k in sorted(set(snaps.kind.tolist())):
        for Re in sorted(set(snaps.re[snaps.kind == k].tolist())):
            out.append((k, Re, (snaps.kind == k) & (snaps.re == Re)))
    return out

def evaluate(snaps, fine, preds, re_true, re_pred, stats):
    """Per-(kind, Re) table for the CNN, then per-kind tables for every method."""
    lo, hi = min(stats["train_re"]), max(stats["train_re"])
    rows = {}
    for kind, Re, m in groups(snaps):
        tag = "interp" if lo <= Re <= hi else "EXTRAP"
        mt = torch.from_numpy(m)
        rows[f"{kind} Re={Re} {tag}"] = metrics_for(preds["CNN"][mt], fine[mt], stats, re_pred[mt], re_true[mt])
    print(format_table(rows, title=f"\nCNN per flow and Re  (training Re: {stats['train_re']})"))
    for kind in sorted(set(snaps.kind.tolist())):
        mt = torch.from_numpy(snaps.kind == kind)
        table = {name: metrics_for(P[mt], fine[mt], stats, *( (re_pred[mt], re_true[mt]) if name == "CNN" else ()))
                 for name, P in preds.items()}
        print(format_table(table, title=f"\nAll unseen-Re {kind} snapshots"))
        print(f"  (ground-truth mean|div| = {table['CNN']['abs_div_true']:.4f})")
        print("  Per-channel SSIM:")
        for label, mm in table.items():
            print(f"    {label:<16}" + "".join(f"  {n}={mm[f'ssim_{s}']:.4f}" for n, s in zip(["u", "v", "p", "ω"], SHORT)))


def _final_index(snaps, mask):
    idx = np.where(mask)[0]
    t = snaps.t[idx]
    return idx[np.nanargmax(t)] if np.isfinite(t).any() else idx[-1]

def physics_table(snaps, fine, preds, stats):
    """Cavity steady state at each Re: primary vortex ψ_min and centre, and the
    largest centreline-velocity error, for every method relative to the DNS."""
    phys = {k: denormalise_t(v, stats).numpy() for k, v in preds.items()}
    dns = denormalise_t(fine, stats).numpy()
    dx = 1.0 / (dns.shape[-1] - 1)
    print("\nSteady-state physics vs DNS (final snapshot at each Re)")
    print(f"  {'Re':>5} {'method':<15} {'ψ_min':>9} {'ψ_min err':>10} {'centre (x, y)':>16} {'centre err':>11} {'max|Δu_c|':>10} {'max|Δv_c|':>10}")
    for _, Re, m in groups(snaps):
        i = _final_index(snaps, m)
        psi0, x0, y0 = primary_vortex(dns[i, 0], dx, dx)
        print(f"  {Re:>5} {'DNS':<15} {psi0:9.5f} {'':>10} {f'({x0:.3f}, {y0:.3f})':>16}")
        for name, P in phys.items():
            psi, xc, yc = primary_vortex(P[i, 0], dx, dx)
            du = np.abs(centerline_u(P[i, 0]) - centerline_u(dns[i, 0])).max()
            dv = np.abs(centerline_v(P[i, 1]) - centerline_v(dns[i, 1])).max()
            print(f"  {'':>5} {name:<15} {psi:9.5f} {abs(psi/psi0 - 1):10.1%} {f'({xc:.3f}, {yc:.3f})':>16} "
                  f"{np.hypot(xc - x0, yc - y0):11.4f} {du:10.4f} {dv:10.4f}")

def plot_centerlines(snaps, fine, preds, stats, out_dir):
    """Cavity centreline profiles of the final (steady) snapshot at each Re."""
    f_p = denormalise_t(fine, stats).numpy()
    phys = {k: denormalise_t(v, stats).numpy() for k, v in preds.items() if k in ("CNN", "bilinear input")}
    for _, Re, m in groups(snaps):
        i = _final_index(snaps, m)
        s = np.linspace(0, 1, f_p.shape[-1])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(f"Centreline profiles, steady state  Re={Re}", fontsize=12, fontweight="bold")
        for data, label, style in [(f_p, "DNS", "k-")] + [(phys[k], k, st) for k, st in (("CNN", "g--"), ("bilinear input", "r:"))]:
            ax1.plot(centerline_u(data[i, 0]), s, style, lw=1.8, label=label)
            ax2.plot(s, centerline_v(data[i, 1]), style, lw=1.8, label=label)
        ax1.set_xlabel("u"); ax1.set_ylabel("y"); ax1.set_title("u along x = 0.5")
        ax2.set_xlabel("x"); ax2.set_ylabel("v"); ax2.set_title("v along y = 0.5")
        for ax in (ax1, ax2): ax.grid(alpha=0.3); ax.legend(fontsize=9)
        plt.tight_layout()
        fpath = os.path.join(out_dir, f"centerline_ood_Re{Re}.png")
        plt.savefig(fpath, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"Saved: {fpath}")


STYLES = {"DNS": ("k", "-", 2.2), "CNN": ("tab:green", "--", 1.8), "bilinear input": ("tab:red", ":", 1.6),
          "bicubic": ("tab:orange", "-.", 1.4), "nearest train": ("tab:blue", (0, (1, 1)), 1.4)}

def plot_spectra(snaps, fine, preds, stats, out_dir):
    """Mean energy spectrum per (kind, Re): DNS vs every method, with the coarse-input
    cut-off and a k^-3 reference (the 2-D enstrophy cascade)."""
    dns = denormalise_t(fine, stats)
    phys = {k: denormalise_t(v, stats) for k, v in preds.items()}
    N = dns.shape[-1]; k_c, k_hi = spectral_band(N)
    k = np.arange(N // 2 + 1)
    for kind, Re, m in groups(snaps):
        mt = torch.from_numpy(m)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        E_dns = energy_spectrum(dns[mt]).mean(0).numpy()
        for name, P in [("DNS", dns)] + list(phys.items()):
            E = E_dns if name == "DNS" else energy_spectrum(P[mt]).mean(0).numpy()
            c, ls, lw = STYLES.get(name, ("gray", "-", 1.2))
            ax.loglog(k[1:k_hi + 1], E[1:k_hi + 1], color=c, ls=ls, lw=lw, label=name)
        k_ref = np.array([8.0, k_hi])
        ax.loglog(k_ref, 2.0 * E_dns[8] * (k_ref / 8.0)**-3, color="gray", lw=1, label=r"$k^{-3}$ (inviscid enstrophy cascade)")
        ax.axvline(k_c, color="gray", ls="--", lw=1); ax.text(k_c * 1.03, ax.get_ylim()[1] * 0.3, "coarse input\ncut-off", fontsize=8)
        ax.set_xlabel("wavenumber k"); ax.set_ylabel("E(k)"); ax.grid(True, which="both", alpha=0.3)
        ax.set_title(f"Energy spectrum — {kind}, unseen Re={Re} (mean of {int(m.sum())} snapshots)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        fpath = os.path.join(out_dir, f"spectrum_{kind}_Re{Re}.png")
        plt.savefig(fpath, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"Saved: {fpath}")


def plot_vorticity(snaps, fine, preds, stats, out_dir):
    periodic = is_periodic(stats)
    coarse = denormalise_t(preds["bilinear input"], stats).numpy()
    f_p = denormalise_t(fine, stats).numpy(); p_p = denormalise_t(preds["CNN"], stats).numpy()
    for kind, Re, m in groups(snaps):
        if periodic:   # middle snapshot of the first run
            idx = np.where(m & (snaps.seed == snaps.seed[m].min()))[0]; i = idx[len(idx) // 2]
        else:          # steady state
            i = _final_index(snaps, m)
        wc, wf, wp = coarse[i, 3], f_p[i, 3], p_p[i, 3]
        err = np.abs(wf - wp)
        vmax = np.percentile(np.abs(wf), 99) + 1e-8; emax = err.max() + 1e-8
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        when = f"t={snaps.t[i]:.0f}" if periodic else "steady state"
        fig.suptitle(f"Vorticity ω — {kind}, unseen Re={Re}, {when}", fontsize=12, fontweight="bold")
        for ax, (w, title, cmap, v, sym) in zip(axes, [
            (wc,  "Coarse input",  "seismic", vmax, True),
            (wf,  "DNS",           "seismic", vmax, True),
            (wp,  "CNN",           "seismic", vmax, True),
            (err, "|DNS - CNN|",   "hot",     emax, False),
        ]):
            # Fields are indexed [x, y]; transpose so x is horizontal (and the cavity lid is on top).
            im = ax.imshow(w.T, cmap=cmap, origin="lower", vmin=-v if sym else 0, vmax=v)
            ax.set_title(title, fontsize=11); ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fpath = os.path.join(out_dir, f"vorticity_{kind}_Re{Re}.png")
        plt.savefig(fpath, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"Saved: {fpath}")

@torch.no_grad()
def plot_tsne(model, snaps: Snapshots, stats, device, out_dir, batch_size, max_points=3000):
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(67)
    sel = np.sort(rng.choice(len(snaps), size=min(max_points, len(snaps)), replace=False))
    fine = to_normalised(snaps.fields[sel], stats)
    feats = []
    for i in range(0, len(fine), batch_size):
        feats.append(model.extract_features(make_coarse_field(fine[i:i+batch_size].to(device),
                                                              periodic=is_periodic(stats))).cpu().numpy())
    emb = TSNE(n_components=2, perplexity=min(30, len(fine) - 1), random_state=67).fit_transform(np.concatenate(feats))
    fig, ax = plt.subplots(figsize=(9, 7))
    markers = {"cavity": "o", "forced": "o", "decaying": "^"}
    for kind in sorted(set(snaps.kind[sel].tolist())):
        mk = snaps.kind[sel] == kind
        sc = ax.scatter(emb[mk, 0], emb[mk, 1], c=np.log10(snaps.re[sel][mk]), cmap="plasma",
                        vmin=np.log10(snaps.re.min()), vmax=np.log10(snaps.re.max()),
                        s=10, alpha=0.8, marker=markers.get(kind, "o"), label=kind)
    cb = plt.colorbar(sc, ax=ax); cb.set_label("log10 Re")
    if len(set(snaps.kind.tolist())) > 1: ax.legend()
    ax.set_title("t-SNE of U-Net bottleneck features (256-dim)", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2"); ax.grid(True, alpha=0.2)
    fpath = os.path.join(out_dir, "tsne_bottleneck.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {fpath}")

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Out: {args.out_dir}")
    model, stats = load_model(args.checkpoint, device)
    ood_dirs = [d for d in args.ood_dir if os.path.isdir(d)]
    test = load_snapshots(ood_dirs) if ood_dirs else None
    if test is not None:
        seen = sorted(set(test.re.tolist()) & set(stats["train_re"]))
        if seen:
            print(f"WARNING: Re {seen} in {ood_dirs} were used in training; they are not out-of-distribution")
    train = None
    train_dirs = [d for d in args.train_dir if os.path.isdir(d)]
    if train_dirs:
        all_train = load_snapshots(train_dirs)
        # Re in the training folders that the model never trained on (e.g. above --max_train_re)
        # are also unseen; the validation Re are left out because they chose the checkpoint.
        val_re = stats.get("val_re", config.FLOWS[args.flow]["val_re"])   # older checkpoints lack val_re
        unseen = ~np.isin(all_train.re, stats["train_re"]) & ~np.isin(all_train.re, val_re)
        if unseen.any():
            print(f"Adding unseen Re {sorted(set(all_train.re[unseen].tolist()))} from {train_dirs} to the test set")
            test = Snapshots.concat([test, all_train.subset(unseen)])
        train = all_train.subset(np.isin(all_train.re, stats["train_re"]))
    if test is None:
        sys.exit(f"No unseen-Re snapshots found in {args.ood_dir}. Generate them first (see README, Stage 0).")
    if test.domain != stats.get("domain", "cavity"):
        sys.exit(f"Test data domain '{test.domain}' does not match the checkpoint's '{stats.get('domain', 'cavity')}'")
    fine, pred, re_true, re_pred = predict(model, test, stats, device, args.batch_size)
    preds = {"CNN": pred,
             "bilinear input": baseline_prediction(fine, "bilinear", stats),
             "bicubic": baseline_prediction(fine, "bicubic", stats)}
    if train is not None:
        preds["nearest train"] = nearest_train_prediction(fine, train.fields, stats)
    evaluate(test, fine, preds, re_true, re_pred, stats)
    if is_periodic(stats):
        plot_spectra(test, fine, preds, stats, args.out_dir)
    else:
        physics_table(test, fine, {k: v for k, v in preds.items() if k != "bicubic"}, stats)
        plot_centerlines(test, fine, preds, stats, args.out_dir)
    plot_vorticity(test, fine, preds, stats, args.out_dir)
    if train is not None:
        plot_tsne(model, Snapshots.concat([train, test]), stats, device, args.out_dir, args.batch_size)
    print(f"\nOutputs in: {args.out_dir}/")

if __name__ == "__main__":
    main()
