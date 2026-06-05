import torch
import torch.nn as nn

class RegimeClassifier(nn.Module):
    def __init__(self,in_ch:int = 512,n_classes:int=5,hidden:int = 256,dropout_p:float = 0.3,):
        super().__init__()
        self.in_ch     = in_ch
        self.n_classes = n_classes
        self.pool = nn.AdaptiveAvgPool2d(1)   # (B, in_ch, 1, 1)
        self.flat = nn.Flatten()              # (B, in_ch)
        self.fc1= nn.Linear(in_ch, hidden)
        self.act= nn.ReLU(inplace=True)
        self.drop= nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden, n_classes)
    def forward(self, x: torch.Tensor,return_features: bool = False):
        h = self.pool(x)          # (B, in_ch, 1, 1)
        h = self.flat(h)          # (B, in_ch)
        h = self.act(self.fc1(h)) # (B, hidden)
        h = self.drop(h)
        logits = self.fc2(h)      # (B, n_classes)
        if return_features: return logits, h
        return logits