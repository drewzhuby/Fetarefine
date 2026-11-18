"""M4: Zero-Parameter Equal-Weight Logit Fusion Layer.

Fuses 3D global expert (M0 Swin UNETR) and 2.5D local slice expert (M2 DeepLabV3+):
L_4 = 0.5 * (L_swin + L_deeplab)
"""

import torch
import torch.nn as nn


class M4LogitFusion(nn.Module):
    """M4 Equal-Weight Raw-Logit Fusion Layer."""

    def __init__(self):
        super().__init__()

    def forward(self, swin_logits: torch.Tensor, deep_logits: torch.Tensor) -> torch.Tensor:
        """Compute equal-weight arithmetic average of FP32 raw logits.

        Args:
            swin_logits: B x 8 x (D x) H x W
            deep_logits: B x 8 x (D x) H x W

        Returns:
            Fused logit tensor of identical shape.
        """
        assert swin_logits.shape == deep_logits.shape, (
            f"Shape mismatch: swin {swin_logits.shape} vs deep {deep_logits.shape}"
        )
        return 0.5 * (swin_logits.float() + deep_logits.float())
