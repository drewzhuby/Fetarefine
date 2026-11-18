"""Ablation Study 2: Unbounded Residual Network.

Replaces the bounded tanh compression Delta = 2 * tanh(R_theta) with a linear unconstrained Delta = 2 * R_theta.
Preserves identical first-order derivative at origin to isolate the effect of bounded logit constraints.
"""

import torch
import torch.nn as nn


class UnboundedResidualNet(nn.Module):
    """Unbounded linear residual network."""

    def __init__(self, in_channels: int = 35, width: int = 16, num_classes: int = 8, scale: float = 2.0):
        super().__init__()
        self.scale = float(scale)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(width, num_classes, 1)
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(
        self,
        image_3ch: torch.Tensor,
        deep_logits: torch.Tensor,
        swin_logits: torch.Tensor,
        l4_logits: torch.Tensor = None,
        return_residual: bool = False,
    ) -> torch.Tensor:
        """Linear unbounded forward pass.

        Note: UB is built directly on top of M4 anchor (l4_logits), exactly like M5,
        but without tanh saturation bounds. If l4_logits is None, it defaults to M4 (deep + swin) * 0.5.
        """
        if l4_logits is None:
            l4_logits = (deep_logits + swin_logits) * 0.5

        diff = (deep_logits - swin_logits).abs()
        evidence = torch.cat([image_3ch, deep_logits, swin_logits, l4_logits, diff], dim=1)
        # Linear unconstrained residual
        residual = self.scale * self.head(self.body(evidence))
        out = l4_logits + residual
        return (out, residual) if return_residual else out
