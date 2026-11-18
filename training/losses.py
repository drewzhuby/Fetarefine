"""Standard Training Losses for FeTA Models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.m5_selective import SelectiveRetentionLoss


class CompoundSegmentationLoss(nn.Module):
    """Standard 0.5 * Masked CE + 0.5 * Soft Dice Loss."""

    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        m = valid_mask.float()
        # Masked Cross-Entropy
        ce = F.cross_entropy(logits, target, reduction="none")
        loss_ce = (ce * m).sum() / (m.sum() + self.smooth)

        # Foreground Soft Dice (classes 1..7)
        probs = logits.softmax(dim=1)
        dice_terms = []
        for b in range(target.shape[0]):
            for c in range(1, logits.shape[1]):
                gt_c = (target[b] == c).float() * m[b]
                if gt_c.any():
                    pred_c = probs[b, c] * m[b]
                    intersect = 2.0 * (pred_c * gt_c).sum() + self.smooth
                    denom = pred_c.sum() + gt_c.sum() + self.smooth
                    dice_terms.append(1.0 - intersect / denom)
        loss_dice = torch.stack(dice_terms).mean() if dice_terms else torch.tensor(0.0, device=logits.device)

        return self.ce_weight * loss_ce + self.dice_weight * loss_dice


class DiffUNetLoss(nn.Module):
    """DiffUNet BTCV compound loss: Dice(sigmoid) + BCE + MSE on 7 foreground classes."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target_onehot7: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        z = logits.float()
        t = target_onehot7.float()
        m = valid_mask[:, None].float()

        p = z.sigmoid()
        axes = tuple(range(2, z.ndim))
        dice = 1.0 - (2.0 * (p * t * m).sum(axes) + self.smooth) / (
            (p * m).sum(axes) + (t * m).sum(axes) + self.smooth
        )
        den = m.sum() * z.shape[1] + 1e-6
        bce = (F.binary_cross_entropy_with_logits(z, t, reduction="none") * m).sum() / den
        mse = ((p - t).square() * m).sum() / den
        return dice.mean() + bce + mse
