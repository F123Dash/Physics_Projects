import torch
import torch.nn as nn


class TurbulenceLoss(nn.Module):
    """
    Physics-informed loss.

    FIX vs v1:
      - lambda_div: 0.1 → 0.001  (was dominating at init: 56344 vs 23 recon)
      - Divergence computed ONLY on channels 0,1 (u,v) — never channel 2 (p)
      - Added per-channel reconstruction weight so pressure (large values)
        doesn't dominate MSE either. Velocity channels weighted 3:3:1 (u:v:p).

    L_total = L_recon + lambda_div * L_div + lambda_cls * L_cls

    Starting values (can tune):
      lambda_div = 0.001   (increase to 0.01 after epoch 10 if div still high)
      lambda_cls = 0.01
    """

    def __init__(self, lambda_div: float = 0.001,
                 lambda_cls: float = 0.01,
                 dx: float = 1.0 / 64):
        super().__init__()
        self.lambda_div = lambda_div
        self.lambda_cls = lambda_cls
        self.dx = dx
        self.ce = nn.CrossEntropyLoss()

        self.register_buffer(
            "ch_weights",
            torch.tensor([3.0, 3.0, 1.0]).view(1, 3, 1, 1)
        )

    def divergence_loss(self, pred: torch.Tensor) -> torch.Tensor:
        u_x = pred[:, 0, :, :]
        u_y = pred[:, 1, :, :]

        du_dx = (u_x[:, 2:, 1:-1] - u_x[:, :-2, 1:-1]) / (2.0 * self.dx)
        du_dy = (u_y[:, 1:-1, 2:] - u_y[:, 1:-1, :-2]) / (2.0 * self.dx)

        div = du_dx + du_dy
        return torch.mean(div ** 2)

    def forward(self, pred_field, true_field, pred_logits, true_labels):
        weights = self.ch_weights.to(pred_field.device)
        diff = (pred_field - true_field) ** 2 * weights
        recon_loss = diff.mean()

        div_loss = self.divergence_loss(pred_field)
        cls_loss = self.ce(pred_logits, true_labels)

        total = recon_loss + self.lambda_div * div_loss + self.lambda_cls * cls_loss
        return total, recon_loss, div_loss, cls_loss
