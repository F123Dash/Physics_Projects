import torch
import torch.nn as nn

class ReRegressor(nn.Module):
    """Predicts normalised log10(Re) from the bottleneck features.
    """
    def __init__(self,in_ch:int = 512,hidden:int = 256,dropout_p:float = 0.3,):
        super().__init__()
        self.in_ch     = in_ch
        self.pool = nn.AdaptiveAvgPool2d(1)   # (B, in_ch, 1, 1)
        self.flat = nn.Flatten()              # (B, in_ch)
        self.fc1= nn.Linear(in_ch, hidden)
        self.act= nn.ReLU(inplace=True)
        self.drop= nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden, 1)
    def forward(self, x: torch.Tensor,return_features: bool = False):
        h = self.pool(x)          # (B, in_ch, 1, 1)
        h = self.flat(h)          # (B, in_ch)
        h = self.act(self.fc1(h)) # (B, hidden)
        h = self.drop(h)
        re_pred = self.fc2(h).squeeze(-1)   # (B,)
        if return_features: return re_pred, h
        return re_pred
