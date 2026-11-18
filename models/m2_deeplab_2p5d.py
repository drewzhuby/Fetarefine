"""M2: 2.5D Slice Aggregation Baseline (DeepLabV3+).

Inputs 3 adjacent axial slices (z-1, z, z+1) to capture local through-plane continuity.
"""

import torch
import torch.nn as nn
from .deeplabv3plus import DeepLabV3Plus


class M2DeepLab2p5D(nn.Module):
    """M2 2.5D DeepLabV3+ Baseline."""

    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        self.net = DeepLabV3Plus(in_channels=3, num_classes=num_classes, pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Expects B x 3 x H x W (slices z-1, z, z+1)."""
        return self.net(x)
