"""Evaluation metrics for footprint and height heads."""
from __future__ import annotations

import torch

from src.data.dataset import MAX_HEIGHT_M


@torch.no_grad()
def compute_metrics(
    mask_logits,
    height_pred,
    mask_gt,
    height_gt,
    max_height_m=MAX_HEIGHT_M,
    thresh=0.5,
):
    pred_mask = (torch.sigmoid(mask_logits) > thresh).float()

    inter = (pred_mask * mask_gt).sum(dim=(1, 2, 3))
    union = ((pred_mask + mask_gt) > 0).float().sum(dim=(1, 2, 3))
    iou = (inter / union.clamp_min(1e-6)).mean().item()

    tp = inter
    fp = (pred_mask * (1 - mask_gt)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_mask) * mask_gt).sum(dim=(1, 2, 3))
    precision = tp / (tp + fp).clamp_min(1e-6)
    recall = tp / (tp + fn).clamp_min(1e-6)
    f1 = (2 * precision * recall / (precision + recall).clamp_min(1e-6)).mean().item()

    building_px = mask_gt.sum().clamp_min(1.0)
    height_pred_m = height_pred * max_height_m
    height_gt_m = height_gt * max_height_m
    abs_err = (height_pred_m - height_gt_m).abs() * mask_gt
    sq_err = ((height_pred_m - height_gt_m) ** 2) * mask_gt
    mae = (abs_err.sum() / building_px).item()
    rmse = torch.sqrt(sq_err.sum() / building_px).item()

    return {"iou": iou, "f1": f1, "height_mae_m": mae, "height_rmse_m": rmse}
