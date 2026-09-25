"""Inference helpers for mask + height prediction."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from src.data.dataset import IMG_SIZE, MAX_HEIGHT_M


@torch.no_grad()
def predict_mask_and_height(
    model,
    image_rgb,
    device,
    img_size=IMG_SIZE,
    max_height_m=MAX_HEIGHT_M,
    mask_thresh=0.5,
):
    orig_h, orig_w = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    img_t = TF.to_tensor(resized)
    img_t = TF.normalize(img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_t = img_t.unsqueeze(0).to(device)

    model.eval()
    mask_logits, height_pred = model(img_t)
    mask_prob = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()
    height_norm = height_pred[0, 0].cpu().numpy()

    mask_prob_full = cv2.resize(mask_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    height_norm_full = cv2.resize(height_norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    mask_bin_full = (mask_prob_full > mask_thresh).astype(np.uint8)
    height_m_full = height_norm_full * max_height_m * mask_bin_full
    return mask_prob_full, mask_bin_full, height_m_full
