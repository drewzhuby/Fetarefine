"""M0: Global 3D Coarse Segmentation Baseline using Swin UNETR.

Captures global 3D spatial anatomical priors from 96x96x96 volume patches.
"""

import torch
import inspect
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class M0SwinUNETR(nn.Module):
    """Swin UNETR 3D segmentation baseline."""

    def __init__(
        self,
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=8,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
    ):
        super().__init__()
        self.net = SwinUNETR(
            **({"img_size": img_size} if "img_size" in inspect.signature(SwinUNETR).parameters else {}),
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Expects B x 1 x D x H x W, returns B x 8 x D x H x W."""
        return self.net(x)
