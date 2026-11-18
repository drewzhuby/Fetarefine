"""M1: Pure 2D High-Resolution Slice Segmentation Baseline (DeepLabV3+).

Inputs single slice replicated across 3 RGB channels to exploit ImageNet features.
"""

import torch
import torch.nn as nn
from .deeplabv3plus import DeepLabV3Plus


class M1DeepLab2D(nn.Module):
    """M1 2D DeepLabV3+ Baseline."""

    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        self.net = DeepLabV3Plus(in_channels=3, num_classes=num_classes, pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Expects B x 3 x H x W (3 identical central slice channels)."""
        return self.net(x)
