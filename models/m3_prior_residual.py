"""M3: Swin Prior-Guided Residual Refinement Model.

Inputs 11 channels (3 adjacent MRI slices + 8 channels of M0 Swin softmax prior probabilities).
Outputs an 8-channel additive unconstrained residual logit update: L_out = L_prior + Delta_L.
"""

import torch
import torch.nn as nn
from .deeplabv3plus import DeepLabV3Plus


class M3PriorResidual(nn.Module):
    """M3 Prior-guided residual refinement network."""

    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        # 11 input channels: 3 slices MRI + 8 prior probability channels
        self.net = DeepLabV3Plus(in_channels=11, num_classes=num_classes, pretrained=pretrained)

        # Zero initialize final classification head so initial residual starts at exactly 0
        nn.init.zeros_(self.net.head.weight)
        if self.net.head.bias is not None:
            nn.init.zeros_(self.net.head.bias)

    def forward(self, x_11ch: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x_11ch: B x 11 x H x W (3 MRI slices + 8 prior softmax probabilities)
            prior_logits: B x 8 x H x W (M0 Swin UNETR raw logits)

        Returns:
            Refined logits: prior_logits + delta_residual
        """
        delta_residual = self.net(x_11ch)
        return prior_logits + delta_residual
