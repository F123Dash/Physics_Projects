import torch
import torch.nn as nn

class RegimeClassifier(nn.Module):
    def __init__(self, in_ch: int = 512, n_classes: int = 5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # (B, 512, 1, 1)
            nn.Flatten(),               # (B, 512)
            nn.Linear(in_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(x)
