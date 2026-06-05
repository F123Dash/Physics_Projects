import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .classifier import RegimeClassifier
    from .losses     import TurbulenceLoss, ssim
except ImportError:  # allow running as a script
    from classifier import RegimeClassifier
    from losses     import TurbulenceLoss, ssim

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:layers.append(nn.Dropout2d(p=dropout_p))
        self.block = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, dropout_p=dropout_p)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x: torch.Tensor):
        features = self.conv(x)       # (B, out_ch, H,   W)   — skip connection
        pooled   = self.pool(features) # (B, out_ch, H/2, W/2) — passed down
        return features, pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                      nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False),)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)   # (B, in_ch//2, 2H, 2W)
        if x.shape != skip.shape:x = F.interpolate(x, size=skip.shape[2:],mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)  # (B, in_ch//2 + skip_ch, 2H, 2W)
        return self.conv(x)
class TurbulenceUNet(nn.Module):
    def __init__(self,in_ch:int= 4,out_ch:int= 4,
                 base_ch:int= 64,n_classes:int= 5,dropout_p:float= 0.1,):
        super().__init__()
        c1 = base_ch        #  64
        c2 = base_ch * 2    # 128
        c3 = base_ch * 4    # 256
        c4 = base_ch * 8    # 512  (bottleneck)
        self.enc1 = EncoderBlock(in_ch, c1, dropout_p=dropout_p)  # 64×64→32×32
        self.enc2 = EncoderBlock(c1,    c2, dropout_p=dropout_p)  # 32×32→16×16
        self.enc3 = EncoderBlock(c2,    c3, dropout_p=dropout_p)  # 16×16→ 8× 8
        self.bottleneck = DoubleConv(c3, c4, dropout_p=0.0)       #  8× 8
        self.classifier = RegimeClassifier(in_ch= c4,n_classes= n_classes,dropout_p= 0.3,)
        self.dec3 = DecoderBlock(c4, c3, c3)   # 512 + 256 → 256,  8→16
        self.dec2 = DecoderBlock(c3, c2, c2)   # 256 + 128 → 128, 16→32
        self.dec1 = DecoderBlock(c2, c1, c1)   # 128 +  64 →  64, 32→64
        self.output_conv = nn.Conv2d(c1, out_ch, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",nonlinearity="relu")
                if m.bias is not None:nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x: torch.Tensor):
        s1, x = self.enc1(x)   # s1: (B, c1, H,   W),   x: (B, c1, H/2, W/2)
        s2, x = self.enc2(x)   # s2: (B, c2, H/2, W/2), x: (B, c2, H/4, W/4)
        s3, x = self.enc3(x)   # s3: (B, c3, H/4, W/4), x: (B, c3, H/8, W/8)
        x = self.bottleneck(x)              # (B, c4, H/8, W/8)
        class_logits = self.classifier(x)   # (B, n_classes)
        x = self.dec3(x, s3)   # (B, c3, H/4, W/4)
        x = self.dec2(x, s2)   # (B, c2, H/2, W/2)
        x = self.dec1(x, s1)   # (B, c1, H,   W)
        fine_pred = self.output_conv(x)     # (B, out_ch, H, W)
        return fine_pred, class_logits
    def project_divergence_free(self, u_pred, stats):
        device = u_pred.device
        B, C, H, W = u_pred.shape
        su = float(stats["std"][0]);  mu = float(stats["mean"][0])
        sv = float(stats["std"][1]);  mv = float(stats["mean"][1])
        u = u_pred[:,0]*su + mu
        v = u_pred[:,1]*sv + mv
        kx = torch.fft.fftfreq(W, d=1./W).to(device)
        ky = torch.fft.fftfreq(H, d=1./H).to(device)
        KX,KY = torch.meshgrid(kx, ky, indexing="ij")
        K2 = KX**2 + KY**2;  K2[0,0] = 1.0
        uh = torch.fft.fft2(u);  vh = torch.fft.fft2(v)
        ph = (1j*KX*uh + 1j*KY*vh) / K2
        uc = torch.fft.ifft2(uh - 1j*KX*ph).real
        vc = torch.fft.ifft2(vh - 1j*KY*ph).real
        out = u_pred.clone()
        out[:,0] = (uc - mu) / su
        out[:,1] = (vc - mv) / sv
        return out
    def extract_features(self, x: torch.Tensor):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        x = self.bottleneck(x)
        _, features = self.classifier(x, return_features=True)
        return features   # (B, 256)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_by_module(model: nn.Module) -> dict:
    counts = {}
    for name, module in model.named_children():
        counts[name] = sum(p.numel() for p in module.parameters()
                           if p.requires_grad)
    return counts

def model_summary(model: nn.Module, input_shape=(1, 4, 64, 64)) -> None:
    print("TurbulenceUNet — model summary")
    total = count_parameters(model)
    print(f"  Total trainable parameters: {total:,}\n")
    by_module = count_parameters_by_module(model)
    print(f"  {'Sub-module':<20} {'Params':>12}  {'Share':>7}")
    for name, n in by_module.items():
        print(f"  {name:<20} {n:>12,}  {100*n/total:>6.1f}%")
    print(f"\n  Forward pass shape trace (batch_size=1):")
    device = next(model.parameters()).device
    x = torch.zeros(input_shape).to(device)
    print(f"  {'Layer':<30} {'Output shape'}")
    was_training = model.training
    model.eval()
    with torch.no_grad():
        print(f"  {'Input':<30} {tuple(x.shape)}")
        s1, x2 = model.enc1.conv(x), model.enc1.pool(model.enc1.conv(x))
        s1, x2 = model.enc1(x)
        print(f"  {'enc1 skip':<30} {tuple(s1.shape)}")
        print(f"  {'enc1 pooled':<30} {tuple(x2.shape)}")
        s2, x3 = model.enc2(x2)
        print(f"  {'enc2 skip':<30} {tuple(s2.shape)}")
        print(f"  {'enc2 pooled':<30} {tuple(x3.shape)}")
        s3, x4 = model.enc3(x3)
        print(f"  {'enc3 skip':<30} {tuple(s3.shape)}")
        print(f"  {'enc3 pooled':<30} {tuple(x4.shape)}")
        bn = model.bottleneck(x4)
        print(f"  {'bottleneck':<30} {tuple(bn.shape)}")
        logits = model.classifier(bn)
        print(f"  {'classifier logits':<30} {tuple(logits.shape)}")
        d3 = model.dec3(bn, s3)
        print(f"  {'dec3':<30} {tuple(d3.shape)}")
        d2 = model.dec2(d3, s2)
        print(f"  {'dec2':<30} {tuple(d2.shape)}")
        d1 = model.dec1(d2, s1)
        print(f"  {'dec1':<30} {tuple(d1.shape)}")
        out = model.output_conv(d1)
        print(f"  {'output (fine field)':<30} {tuple(out.shape)}")
    if was_training:model.train()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    model = TurbulenceUNet(in_ch= 4,out_ch= 4,base_ch= 64,n_classes= 5,dropout_p= 0.1,).to(device)
    model_summary(model, input_shape=(1, 4, 64, 64))
    criterion = TurbulenceLoss(lambda_div=0.1,lambda_vort=0.05,lambda_cls=0.01,n_channels=4,)
    B = 4
    coarse = torch.randn(B, 4, 64, 64).to(device)
    fine   = torch.randn(B, 4, 64, 64).to(device)
    labels = torch.randint(0, 5, (B,)).to(device)
    model.train()
    pred_field, pred_logits = model(coarse)
    total, recon, div_l, vort_l, cls_l = criterion(pred_field, fine, pred_logits, labels)
    print(f"\nLoss components (random inputs):")
    print(f"  L_total : {total.item():.4f}")
    print(f"  L_recon : {recon.item():.4f}")
    print(f"  L_div   : {div_l.item():.4f}")
    print(f"  L_vort  : {vort_l.item():.4f}")
    print(f"  L_cls   : {cls_l.item():.4f}")
    total.backward()
    grad_norms = [(n, p.grad.norm().item())for n, p in model.named_parameters()if p.grad is not None]
    print(f"\nBackward pass: OK")
    print(f"  Parameters with gradients: {len(grad_norms)}")
    print(f"  Max grad norm: {max(v for _,v in grad_norms):.4f}")
    print(f"  Min grad norm: {min(v for _,v in grad_norms):.6f}")
    ssim_self = ssim(pred_field.detach(), pred_field.detach())
    ssim_diff = ssim(pred_field.detach(), fine)
    print(f"\nSSIM (self):  {ssim_self:.4f}  (should be 1.0)")
    print(f"SSIM (noise): {ssim_diff:.4f}  (should be < 1.0)")
    model.eval()
    with torch.no_grad():
        features = model.extract_features(coarse)
    print(f"\nBottleneck features shape: {tuple(features.shape)}  "f"(expected ({B}, 256))")