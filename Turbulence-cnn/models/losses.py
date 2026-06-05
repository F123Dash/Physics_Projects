import torch
import torch.nn as nn


class TurbulenceLoss(nn.Module):
    def __init__(self,lambda_div:float= 0.01,lambda_vort:float= 0.01,lambda_cls:float= 0.01,
                 dx:float= 1.0 / 64,ch_weights:list= None,class_weights:torch.Tensor= None,
                 n_channels:int= 4,sigma_u:float= None,sigma_v:float= None,
    ):
        super().__init__()
        self.lambda_div  = lambda_div
        self.lambda_vort = lambda_vort
        self.lambda_cls  = lambda_cls
        self.dx          = dx
        self.n_channels  = n_channels
        self.sigma_u = float(sigma_u) if sigma_u is not None else None
        self.sigma_v = float(sigma_v) if sigma_v is not None else None
        if ch_weights is None:
            if n_channels == 4:ch_weights = [3.0, 3.0, 2.0, 2.0]   # [u, v, p, ω]
            else:ch_weights = [3.0, 3.0, 1.0]          # [u, v, p]
        assert len(ch_weights) == n_channels, (f"ch_weights length {len(ch_weights)} != n_channels {n_channels}")
        self.register_buffer("ch_weights",torch.tensor(ch_weights, dtype=torch.float32).view(1, n_channels, 1, 1),)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
            self.ce = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.class_weights = None
            self.ce = nn.CrossEntropyLoss()
    def divergence_loss(self, pred: torch.Tensor) -> torch.Tensor:
        dx = self.dx
        u_field = pred[:, 0, :, :]   # (B, H, W)
        v_field = pred[:, 1, :, :]   # (B, H, W)
        du_dx = (u_field[:, 2:, 1:-1] - u_field[:, :-2, 1:-1]) / (2.0 * dx)
        dv_dy = (v_field[:, 1:-1, 2:] - v_field[:, 1:-1, :-2]) / (2.0 * dx)
        if self.sigma_u is not None and self.sigma_v is not None:
            du_dx = du_dx * self.sigma_u
            dv_dy = dv_dy * self.sigma_v
        divergence = du_dx + dv_dy
        return torch.mean(divergence ** 2)

    def vorticity_loss(self,pred:torch.Tensor,true_field: torch.Tensor,) -> torch.Tensor:
        dx = self.dx
        u_pred = pred[:, 0, :, :]
        v_pred = pred[:, 1, :, :]
        dv_dx_pred = (v_pred[:, 2:, 1:-1] - v_pred[:, :-2, 1:-1]) / (2.0 * dx)
        du_dy_pred = (u_pred[:, 1:-1, 2:] - u_pred[:, 1:-1, :-2]) / (2.0 * dx)
        omega_pred = dv_dx_pred - du_dy_pred   # (B, H-2, W-2)
        if self.n_channels == 4:
            omega_true = true_field[:, 3, 1:-1, 1:-1]  # pre-stored in snapshot
        else:
            u_true = true_field[:, 0, :, :]
            v_true = true_field[:, 1, :, :]
            dv_dx_true = (v_true[:, 2:, 1:-1] - v_true[:, :-2, 1:-1]) / (2.0 * dx)
            du_dy_true = (u_true[:, 1:-1, 2:] - u_true[:, 1:-1, :-2]) / (2.0 * dx)
            omega_true = dv_dx_true - du_dy_true
        return torch.mean((omega_pred - omega_true) ** 2)

    def forward(self,pred_field:torch.Tensor,true_field:torch.Tensor,
                pred_logits: torch.Tensor,true_labels: torch.Tensor,):
        weights = self.ch_weights.to(pred_field.device)
        recon_loss = ((pred_field - true_field) ** 2 * weights).mean()
        div_loss  = self.divergence_loss(pred_field)
        vort_loss = self.vorticity_loss(pred_field, true_field)
        cls_loss = self.ce(pred_logits, true_labels)
        total = (recon_loss+self.lambda_div*div_loss+self.lambda_vort*vort_loss+self.lambda_cls*cls_loss)
        return total, recon_loss, div_loss, vort_loss, cls_loss

def ssim(pred:torch.Tensor,target:torch.Tensor,C1:float = 0.01**2,C2:float = 0.03**2,) -> torch.Tensor:
    mu_p = pred.mean(dim=(-2, -1), keepdim=True)      # (B, C, 1, 1)
    mu_t = target.mean(dim=(-2, -1), keepdim=True)
    diff_p = pred   - mu_p
    diff_t = target - mu_t
    sig_p  = (diff_p ** 2).mean(dim=(-2, -1))          # (B, C)
    sig_t  = (diff_t ** 2).mean(dim=(-2, -1))
    sig_pt = (diff_p * diff_t).mean(dim=(-2, -1))
    mu_p = mu_p.squeeze((-2, -1))                      # (B, C)
    mu_t = mu_t.squeeze((-2, -1))
    numerator   = (2.0 * mu_p * mu_t + C1) * (2.0 * sig_pt + C2)
    denominator = (mu_p ** 2 + mu_t ** 2 + C1) * (sig_p + sig_t + C2)
    return (numerator / denominator).mean()