import torch
import torch.nn as nn
import torch.nn.functional as F


def _stat(stats, key, device):
    return torch.as_tensor(stats[key], dtype=torch.float32, device=device)

def denormalise_t(field: torch.Tensor, stats: dict) -> torch.Tensor:
    """(B, C, H, W) normalised -> physical units."""
    std  = _stat(stats, "std",  field.device)[: field.shape[1]].view(1, -1, 1, 1)
    mean = _stat(stats, "mean", field.device)[: field.shape[1]].view(1, -1, 1, 1)
    return field * std + mean

def _periodic(stats) -> bool:
    return stats.get("domain", "cavity") == "periodic"

def interior(field: torch.Tensor, stats: dict) -> torch.Tensor:
    """Nodes where derivatives are defined: all of them for a periodic domain,
    the interior for the cavity (central differences need both neighbours)."""
    return field if _periodic(stats) else field[..., 1:-1, 1:-1]

def _spectral_grad(f: torch.Tensor, L: float):
    """Exact (d/dx, d/dy) of a periodic field (..., N, N) indexed [x, y]; the Nyquist mode is dropped."""
    N = f.shape[-1]
    k = torch.fft.fftfreq(N, d=L / (2 * torch.pi * N), device=f.device)
    k[N // 2] = 0.0
    f_hat = torch.fft.fft2(f)
    dx = torch.fft.ifft2(1j * k[:, None] * f_hat).real
    dy = torch.fft.ifft2(1j * k[None, :] * f_hat).real
    return dx, dy

def _central_grad(f: torch.Tensor, L: float):
    """Central differences on interior nodes of a wall-bounded grid (dx = L/(N-1)). (..., N-2, N-2)"""
    h = L / (f.shape[-1] - 1)
    return ((f[..., 2:, 1:-1] - f[..., :-2, 1:-1]) / (2*h), (f[..., 1:-1, 2:] - f[..., 1:-1, :-2]) / (2*h))

def _grad(f, stats):
    L = float(stats.get("L", 1.0))
    return _spectral_grad(f, L) if _periodic(stats) else _central_grad(f, L)

def divergence_phys(phys: torch.Tensor, stats: dict) -> torch.Tensor:
    """du/dx + dv/dy on the nodes given by interior()."""
    ux, _ = _grad(phys[:, 0], stats); _, vy = _grad(phys[:, 1], stats)
    return ux + vy

def curl_phys(phys: torch.Tensor, stats: dict) -> torch.Tensor:
    """dv/dx - du/dy on the nodes given by interior()."""
    _, uy = _grad(phys[:, 0], stats); vx, _ = _grad(phys[:, 1], stats)
    return vx - uy


SPEC_FLOOR = 1e-7      # spectral comparisons ignore shells below this fraction of a sample's total energy

def energy_spectrum(phys: torch.Tensor) -> torch.Tensor:
    N = phys.shape[-1]
    uh = torch.fft.fft2(phys[:, 0]) / N**2
    vh = torch.fft.fft2(phys[:, 1]) / N**2
    e = 0.5 * (uh.abs()**2 + vh.abs()**2)                           # (B, N, N)
    k = torch.fft.fftfreq(N, d=1.0 / N, device=phys.device)
    shell = torch.sqrt(k[:, None]**2 + k[None, :]**2).round().long().clamp(max=N // 2)
    E = torch.zeros(phys.shape[0], N // 2 + 1, device=phys.device, dtype=e.dtype)
    return E.scatter_add_(1, shell.flatten().expand(phys.shape[0], -1), e.flatten(1))

def log_spectrum_mismatch(pred_phys: torch.Tensor, true_phys: torch.Tensor, k_lo: int) -> torch.Tensor:
    hi = pred_phys.shape[-1] // 2 - 1
    E_t = energy_spectrum(true_phys)
    eps = SPEC_FLOOR * E_t.sum(dim=1, keepdim=True)
    E_p = energy_spectrum(pred_phys)
    d = torch.log(E_p[:, k_lo:hi + 1] + eps) - torch.log(E_t[:, k_lo:hi + 1] + eps)
    return (d ** 2).mean(dim=1)

class TurbulenceLoss(nn.Module):
    def __init__(self,stats:dict,lambda_div:float= 0.01,lambda_vort:float= 0.01,lambda_re:float= 0.01,
                 ch_weights:list= None,recon:str= "mse",lambda_spec:float= 0.0,coarse_factor:int= 4,):
        super().__init__()
        assert recon in ("mse", "relative"), recon
        if lambda_spec > 0 and not _periodic(stats):
            raise ValueError("the spectral loss needs a periodic domain")
        self.recon       = recon
        self.lambda_spec = lambda_spec
        self.coarse_factor = coarse_factor
        self.stats       = stats
        self.lambda_div  = lambda_div
        self.lambda_vort = lambda_vort
        self.lambda_re   = lambda_re
        if ch_weights is None:
            ch_weights = [3.0, 3.0, 2.0, 2.0]   # [u, v, p, ω]
        self.register_buffer("ch_weights",torch.tensor(ch_weights, dtype=torch.float32).view(1, -1, 1, 1),)

    def divergence_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Mean squared divergence of the predicted velocity, in (1/s)^2."""
        return torch.mean(divergence_phys(denormalise_t(pred, self.stats), self.stats) ** 2)

    def vorticity_loss(self, pred: torch.Tensor) -> torch.Tensor:
        phys = denormalise_t(pred, self.stats)
        sigma_w = float(self.stats["std"][3])
        return torch.mean(((interior(phys[:, 3], self.stats) - curl_phys(phys, self.stats)) / sigma_w) ** 2)

    def spectral_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        if self.lambda_spec == 0:
            return pred.new_zeros(())
        k_lo = pred.shape[-1] // (2 * self.coarse_factor)
        return log_spectrum_mismatch(denormalise_t(pred, self.stats), denormalise_t(true, self.stats), k_lo).mean()

    def reconstruction_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        if self.recon == "mse":
            return ((pred - true) ** 2 * self.ch_weights).mean()
        std = _stat(self.stats, "std", pred.device)[: pred.shape[1]].view(1, -1, 1, 1)
        err2 = (((pred - true) * std) ** 2).flatten(2).sum(dim=2)                  # (B, C)
        ref2 = (denormalise_t(true, self.stats) ** 2).flatten(2).sum(dim=2).clamp(min=1e-12)
        return (err2 / ref2 * self.ch_weights.view(1, -1)).mean()

    def forward(self,pred_field:torch.Tensor,true_field:torch.Tensor,
                pred_re: torch.Tensor,true_re: torch.Tensor,):
        recon_loss = self.reconstruction_loss(pred_field, true_field)
        div_loss  = self.divergence_loss(pred_field)
        vort_loss = self.vorticity_loss(pred_field)
        re_loss   = F.mse_loss(pred_re, true_re)
        spec_loss = self.spectral_loss(pred_field, true_field)
        total = (recon_loss+self.lambda_div*div_loss+self.lambda_vort*vort_loss+self.lambda_re*re_loss
                 +self.lambda_spec*spec_loss)
        return total, recon_loss, div_loss, vort_loss, re_loss, spec_loss

def _gaussian_window(size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-x**2 / (2 * sigma**2)); g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]

def ssim(pred:torch.Tensor,target:torch.Tensor,reduction:str = "mean",) -> torch.Tensor:
    B, C, H, W = target.shape
    t_min = target.amin(dim=(-2, -1), keepdim=True)
    t_rng = (target.amax(dim=(-2, -1), keepdim=True) - t_min).clamp(min=1e-8)
    x = ((pred - t_min) / t_rng).reshape(B * C, 1, H, W)
    y = ((target - t_min) / t_rng).reshape(B * C, 1, H, W)
    win = _gaussian_window(device=target.device)
    filt = lambda z: F.conv2d(z, win)          # 'valid' windows only, as in the reference implementation
    mu_x, mu_y = filt(x), filt(y)
    sxx = filt(x * x) - mu_x**2
    syy = filt(y * y) - mu_y**2
    sxy = filt(x * y) - mu_x * mu_y
    C1, C2 = 0.01**2, 0.03**2
    s_map = ((2*mu_x*mu_y + C1) * (2*sxy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sxx + syy + C2))
    s = s_map.mean(dim=(-2, -1)).reshape(B, C)
    return s.mean() if reduction == "mean" else s
