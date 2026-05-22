import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)     # (B, out_ch, H, W)  — save for skip
        pooled   = self.pool(features)  # (B, out_ch, H/2, W/2) — pass down
        return features, pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.upsample(x)    # (B, in_ch, 2H, 2W)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)   # concat along channel axis
        return self.conv(x)

try:
    from .classifier import RegimeClassifier
except Exception:
    from classifier import RegimeClassifier

class TurbulenceUNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3,
                 base_ch: int = 64, n_classes: int = 5):
        super().__init__()
        c1 = base_ch        # 64
        c2 = base_ch * 2    # 128
        c3 = base_ch * 4    # 256
        c4 = base_ch * 8    # 512  (bottleneck)
        self.enc1 = EncoderBlock(in_ch, c1)   # 3   → 64,  64×64 → 32×32
        self.enc2 = EncoderBlock(c1, c2)       # 64  → 128, 32×32 → 16×16
        self.enc3 = EncoderBlock(c2, c3)       # 128 → 256, 16×16 →  8×8
        self.bottleneck = DoubleConv(c3, c4)   # 256 → 512, 8×8
        self.classifier = RegimeClassifier(c4, n_classes)
        self.dec3 = DecoderBlock(c4, c3, c3)   # 512 + 256 → 256, 8→16
        self.dec2 = DecoderBlock(c3, c2, c2)   # 256 + 128 → 128, 16→32
        self.dec1 = DecoderBlock(c2, c1, c1)   # 128 + 64  → 64,  32→64
        self.output_conv = nn.Conv2d(c1, out_ch, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        s1, x = self.enc1(x)   # s1: (B, 64,  64,64), x: (B, 64,  32,32)
        s2, x = self.enc2(x)   # s2: (B, 128, 32,32), x: (B, 128, 16,16)
        s3, x = self.enc3(x)   # s3: (B, 256, 16,16), x: (B, 256,  8, 8)
        x = self.bottleneck(x)  # (B, 512, 8, 8)
        class_logits = self.classifier(x)  # (B, 5)
        x = self.dec3(x, s3)    # (B, 256, 16, 16)
        x = self.dec2(x, s2)    # (B, 128, 32, 32)
        x = self.dec1(x, s1)    # (B,  64, 64, 64)
        fine_pred = self.output_conv(x)  # (B, 3, 64, 64)
        return fine_pred, class_logits
try:
    from .losses import TurbulenceLoss
except Exception:
    from losses import TurbulenceLoss

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_shape=(1, 3, 64, 64)):
    print(f"TurbulenceUNet — model summary")
    print(f"  Total trainable parameters: "
          f"{count_parameters(model):,}")

    device = next(model.parameters()).device
    x = torch.zeros(input_shape).to(device)

    print(f"\n  Forward pass shape trace (batch_size=1):")
    print(f"  {'Layer':<28} {'Output shape'}")

    with torch.no_grad():
        print(f"  {'Input':<28} {tuple(x.shape)}")
        s1, x2 = model.enc1(x)
        print(f"  {'Encoder 1 (skip)':<28} {tuple(s1.shape)}")
        print(f"  {'Encoder 1 (pooled)':<28} {tuple(x2.shape)}")
        s2, x3 = model.enc2(x2)
        print(f"  {'Encoder 2 (skip)':<28} {tuple(s2.shape)}")
        print(f"  {'Encoder 2 (pooled)':<28} {tuple(x3.shape)}")
        s3, x4 = model.enc3(x3)
        print(f"  {'Encoder 3 (skip)':<28} {tuple(s3.shape)}")
        print(f"  {'Encoder 3 (pooled)':<28} {tuple(x4.shape)}")
        bn = model.bottleneck(x4)
        print(f"  {'Bottleneck':<28} {tuple(bn.shape)}")
        logits = model.classifier(bn)
        print(f"  {'Classifier logits':<28} {tuple(logits.shape)}")
        d3 = model.dec3(bn, s3)
        print(f"  {'Decoder 3':<28} {tuple(d3.shape)}")
        d2 = model.dec2(d3, s2)
        print(f"  {'Decoder 2':<28} {tuple(d2.shape)}")
        d1 = model.dec1(d2, s1)
        print(f"  {'Decoder 1':<28} {tuple(d1.shape)}")
        out = model.output_conv(d1)
        print(f"  {'Output (fine field)':<28} {tuple(out.shape)}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = TurbulenceUNet(
        in_ch     = 3,
        out_ch    = 3,
        base_ch   = 64,
        n_classes = 5,
    ).to(device)
    model_summary(model)
    criterion = TurbulenceLoss(lambda_div=0.1, lambda_cls=0.01)
    B = 4
    coarse = torch.randn(B, 3, 64, 64).to(device)
    fine   = torch.randn(B, 3, 64, 64).to(device)
    labels = torch.randint(0, 5, (B,)).to(device)
    model.train()
    pred_field, pred_logits = model(coarse)
    print(f"Pred field shape  : {tuple(pred_field.shape)}")
    print(f"Pred logits shape : {tuple(pred_logits.shape)}")
    total, recon, div, cls = criterion(
        pred_field, fine, pred_logits, labels
    )
    print(f"\nLoss components:")
    print(f"  L_total : {total.item():.4f}")
    print(f"  L_recon : {recon.item():.4f}")
    print(f"  L_div   : {div.item():.4f}")
    print(f"  L_cls   : {cls.item():.4f}")
    total.backward()
    print(f"\nBackward pass: OK")
    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms.append((name, p.grad.norm().item()))
    print(f"Parameters with gradients: {len(grad_norms)}")
    print(f"Max gradient norm: {max(v for _,v in grad_norms):.4f}")
    print(f"Min gradient norm: {min(v for _,v in grad_norms):.6f}")