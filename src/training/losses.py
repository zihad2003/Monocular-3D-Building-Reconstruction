"""Multi-task segmentation + masked height losses."""
from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class MultiTaskLoss(nn.Module):
    """total = w_seg * (BCE + Dice) + w_height * MaskedSmoothL1."""

    def __init__(self, w_seg: float = 1.0, w_height: float = 1.0, huber_beta: float = 0.05):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.huber = nn.SmoothL1Loss(beta=huber_beta, reduction="none")
        self.w_seg = w_seg
        self.w_height = w_height

    def forward(self, mask_logits, height_pred, mask_gt, height_gt):
        seg_loss = self.bce(mask_logits, mask_gt) + self.dice(mask_logits, mask_gt)
        per_pixel = self.huber(height_pred, height_gt)
        denom = mask_gt.sum().clamp_min(1.0)
        height_loss = (per_pixel * mask_gt).sum() / denom
        total = self.w_seg * seg_loss + self.w_height * height_loss
        return total, seg_loss.detach(), height_loss.detach()
