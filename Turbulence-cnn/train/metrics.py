import numpy as np
import torch
import config
from data.augmentation import make_coarse_field
from data.dataset      import denormalise_re, is_periodic
from models.losses     import denormalise_t, divergence_phys, ssim, energy_spectrum, SPEC_FLOOR

SHORT = ["u", "v", "p", "w"]

def baseline_prediction(fine: torch.Tensor, mode: str, stats: dict) -> torch.Tensor:
    """Interpolation baseline from the same block averages the model sees."""
    return make_coarse_field(fine, noise_std=0.0, mode=mode, periodic=is_periodic(stats))

def spectral_band(N: int):
    """Wavenumbers the coarse input cannot represent: from its Nyquist k = N/(2·factor) to N/2 − 1."""
    return N // (2 * config.COARSE_FACTOR), N // 2 - 1

@torch.no_grad()
def sample_metrics(pred: torch.Tensor, fine: torch.Tensor, stats: dict) -> dict:
    """Per-sample metrics for normalised (B, 4, H, W) prediction and target."""
    p = denormalise_t(pred, stats); t = denormalise_t(fine, stats)
    err  = (p - t).flatten(2).norm(dim=2)                          # (B, C)
    ref  = t.flatten(2).norm(dim=2).clamp(min=1e-12)
    rel  = err / ref
    rel_vel = (p[:, :2] - t[:, :2]).flatten(1).norm(dim=1) / t[:, :2].flatten(1).norm(dim=1).clamp(min=1e-12)
    s = ssim(pred, fine, reduction="none")                          # (B, C)
    out = {"rel_l2_vel": rel_vel, "ssim": s.mean(dim=1),
           "abs_div": divergence_phys(p, stats).abs().flatten(1).mean(dim=1),
           "abs_div_true": divergence_phys(t, stats).abs().flatten(1).mean(dim=1)}
    for c, name in enumerate(SHORT):
        out[f"rel_l2_{name}"] = rel[:, c]
        out[f"ssim_{name}"]   = s[:, c]
    out["rel_l2_all"] = rel.mean(dim=1)      # mean over u, v, p, ω: the checkpoint-selection score
    if is_periodic(stats):
        lo, hi = spectral_band(p.shape[-1])
        E_tot = energy_spectrum(t)
        Ep, Et = energy_spectrum(p)[:, lo:hi + 1], E_tot[:, lo:hi + 1]
        # Floor at SPEC_FLOOR × the sample's total energy: shells far below it (deep in the
        # dissipation range at low Re) are physically negligible and would dominate a log ratio.
        eps = SPEC_FLOOR * E_tot.sum(dim=1, keepdim=True)
        out["spec_err"]  = torch.log10((Ep + eps) / (Et + eps)).abs().mean(dim=1)
        out["hf_energy"] = Ep.sum(dim=1) / Et.sum(dim=1).clamp(min=1e-30)
    return out

@torch.no_grad()
def re_metrics(re_pred: torch.Tensor, re_true: torch.Tensor, stats: dict) -> dict:
    """Relative error of the predicted Reynolds number, from normalised log10 Re."""
    rp = denormalise_re(re_pred.double().cpu().numpy(), stats)
    rt = denormalise_re(re_true.double().cpu().numpy(), stats)
    return {"re_rel_err": torch.from_numpy(np.abs(rp - rt) / rt)}

class MetricAccumulator:
    def __init__(self):
        self.sums, self.n = {}, 0
    def add(self, per_sample: dict):
        for k, v in per_sample.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v.sum())
        self.n += len(next(iter(per_sample.values())))
    def result(self) -> dict:
        return {k: v / max(self.n, 1) for k, v in self.sums.items()}

def format_table(rows: dict, title: str = "") -> str:
    """rows: {label: metrics dict}. Physical-unit relative L2 per channel, SSIM, divergence,
    and for periodic data the fine-scale spectral error and recovered energy fraction."""
    cols = [("rel_l2_vel", "relL2 u,v", ".1%"), ("rel_l2_u", "relL2 u", ".1%"), ("rel_l2_v", "relL2 v", ".1%"),
            ("rel_l2_p", "relL2 p", ".1%"), ("rel_l2_w", "relL2 ω", ".1%"), ("ssim", "SSIM", ".4f"),
            ("abs_div", "mean|div|", ".4f"), ("spec_err", "spec err", ".3f"), ("hf_energy", "HF energy", ".1%"),
            ("re_rel_err", "Re err", ".1%")]
    cols = [c for c in cols if any(c[0] in m for m in rows.values())]
    w = max([16] + [len(l) + 1 for l in rows])
    lines = [title] if title else []
    lines.append(f"  {'':<{w}}" + "".join(f"{h:>11}" for _, h, _ in cols))
    for label, m in rows.items():
        lines.append(f"  {label:<{w}}" + "".join(
            f"{format(m[k], fmt):>11}" if k in m else f"{'—':>11}" for k, _, fmt in cols))
    return "\n".join(lines)
