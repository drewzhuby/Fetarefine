"""M5: Two-Stage Bounded Residual and 3D Selective Gating Refinement Network.

Architecture:
- Stage 1 (Bounded Residual, BR):
  Inputs 35 channels: [3-slice MRI, DeepLab logits (8), Swin logits (8), M4 logits (8), |DeepLab - Swin| (8)].
  Outputs Delta = 2 * tanh(R_theta), L_R = L_4 + Delta.
  Parameters: 7,512.

- Stage 2 (3D Selective Retention Gating):
  Inputs 17 channels: [1-channel MRI, Softmax(L_4) (8), Softmax(L_R) (8)].
  Outputs per-voxel per-class gating factor alpha in [0, 1]^8.
  Fuses L_final = L_4 + alpha * (L_R - L_4).
  Trained with Selective Retention Loss: 0.5 * CE + 0.5 * Dice + 0.05 * K (penalizing drops on already-correct voxels).
  Parameters: 14,424.
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundedResidualNet(nn.Module):
    """Stage 1: 2D Bounded Residual Network (BR).

    Restricts logit modifications via tanh to [-delta_max, +delta_max] (default 2.0).
    """

    def __init__(self, in_channels: int = 35, width: int = 16, num_classes: int = 8, delta_max: float = 2.0):
        super().__init__()
        self.delta_max = float(delta_max)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(width, num_classes, 1)
        # Zero-initialize head so initial residual starts at 0
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
        """Forward pass. Expects B x 3 x H x W image, B x 8 x H x W expert logits.

        Note: M5 is strictly built on top of M4 anchor (l4_logits). If l4_logits is not
        explicitly provided, it defaults to the exact M4 equal average (deep + swin) * 0.5.
        """
        if l4_logits is None:
            l4_logits = (deep_logits + swin_logits) * 0.5

        diff = (deep_logits - swin_logits).abs()
        evidence = torch.cat([image_3ch, deep_logits, swin_logits, l4_logits, diff], dim=1)  # 35 channels
        residual = self.delta_max * torch.tanh(self.head(self.body(evidence)))
        out = l4_logits + residual
        return (out, residual) if return_residual else out


class SelectiveGating3DNet(nn.Module):
    """Stage 2: 3D Selective Retention Gating Network.

    Dynamically selects between robust anchor L_4 and refined residual L_R per voxel per class.
    """

    def __init__(self, in_channels: int = 17, width: int = 16, num_classes: int = 8, initial_acceptance: float = 0.99):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(in_channels, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv3d(width, num_classes, 1)
        # Bias initialized to log(p / (1-p)) so initial gate alpha starts at ~initial_acceptance
        nn.init.zeros_(self.head.weight)
        bias_init = math.log(initial_acceptance / (1.0 - initial_acceptance))
        nn.init.constant_(self.head.bias, bias_init)

    def forward(
        self,
        image_1ch: torch.Tensor,
        l4_logits: torch.Tensor,
        lr_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            image_1ch: B x 1 x D x H x W
            l4_logits: B x 8 x D x H x W (anchor)
            lr_logits: B x 8 x D x H x W (bounded residual output)

        Returns:
            fused_logits: B x 8 x D x H x W
            gate_alpha: B x 8 x D x H x W in [0, 1]
        """
        l4_prob = l4_logits.softmax(dim=1)
        lr_prob = lr_logits.softmax(dim=1)
        evidence = torch.cat([image_1ch, l4_prob, lr_prob], dim=1)  # 17 channels

        gate_alpha = torch.sigmoid(self.head(self.body(evidence)))

        # Selective linear interpolation: L_F = (1 - alpha) * L4 + alpha * LR
        fused = l4_logits + gate_alpha * (lr_logits - l4_logits)
        return fused, gate_alpha


class SelectiveRetentionLoss(nn.Module):
    """M5 Compound Loss with Asymmetric Retention Penalty.

    Loss = 0.5 * CE + 0.5 * SoftDice + 0.05 * K
    Where K penalizes probability drops on voxels where L_R was already correct.
    """

    def __init__(self, keep_weight: float = 0.05):
        super().__init__()
        self.keep_weight = keep_weight

    def forward(
        self,
        fused_logits: torch.Tensor,
        lr_logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute compound loss."""
        m = valid_mask.float()
        p_fused = fused_logits.softmax(dim=1)

        # 1. Masked Cross Entropy
        ce = F.cross_entropy(fused_logits, target, reduction="none")
        loss_ce = (ce * m).sum() / m.sum().clamp_min(1)

        # 2. Reference-present Soft Dice over foreground classes 1..7
        target_onehot = F.one_hot(target, 8).movedim(-1, 1).float()
        mask = m[:, None]
        axes = (0, *range(2, fused_logits.ndim))
        gt_sum = (target_onehot * mask).sum(axes)
        pred_sum = (p_fused * mask).sum(axes)
        intersection = (p_fused * target_onehot * mask).sum(axes)
        present = gt_sum[1:] > 0
        terms = (1 - (2 * intersection[1:] + 1e-5) / (gt_sum[1:] + pred_sum[1:] + 1e-5))[present]
        loss_dice = terms.mean() if terms.numel() else fused_logits.sum() * 0

        # 3. Retention Penalty K
        # Target probability under L_R and L_fused
        p_lr_target = lr_logits.softmax(dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        p_fused_target = p_fused.gather(1, target.unsqueeze(1)).squeeze(1)

        # Region Q: where L_R prediction matches ground truth
        lr_correct_mask = m * (lr_logits.argmax(dim=1) == target).float()
        retention_drop = F.relu(p_lr_target - p_fused_target)
        loss_retention = (retention_drop * lr_correct_mask).sum() / (lr_correct_mask.sum().clamp_min(1))

        total_loss = 0.5 * loss_ce + 0.5 * loss_dice + self.keep_weight * loss_retention

        metrics = {
            "loss": float(total_loss.item()),
            "ce": float(loss_ce.item()),
            "dice": float(loss_dice.item()),
            "retention": float(loss_retention.item()),
        }
        return total_loss, metrics
