"""Inference helpers for mask + height prediction using native-resolution sliding-window tiles."""
from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.functional as TF

from src.data.dataset import IMG_SIZE, MAX_HEIGHT_M


def get_2d_hann_window(size: int = 256) -> np.ndarray:
    """Construct a 2D Hann window with strictly positive values for smooth overlap blending."""
    w1d = np.sin(np.pi * (np.arange(size) + 0.5) / size).astype(np.float32)
    return np.outer(w1d, w1d)


def _predict_single_tile(
    model: torch.nn.Module,
    tile_rgb: np.ndarray,
    device: torch.device,
    use_tta: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on one 256x256 tile, optionally with Test-Time Augmentation (flip + rot90)."""
    img_t = TF.to_tensor(tile_rgb)
    img_t = TF.normalize(img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0).to(device)

    if not use_tta:
        with torch.no_grad():
            mask_logits, height_pred = model(img_t)
        return mask_logits[0, 0].cpu().numpy(), height_pred[0, 0].cpu().numpy()

    # Test-time augmentation (identity, hflip, vflip, rot90)
    aug_inputs = [
        img_t,
        torch.flip(img_t, dims=[3]),
        torch.flip(img_t, dims=[2]),
        torch.rot90(img_t, k=1, dims=[2, 3]),
    ]

    with torch.no_grad():
        logits_list = []
        heights_list = []
        for i, aug_in in enumerate(aug_inputs):
            m_l, h_p = model(aug_in)
            if i == 1:
                m_l = torch.flip(m_l, dims=[3])
                h_p = torch.flip(h_p, dims=[3])
            elif i == 2:
                m_l = torch.flip(m_l, dims=[2])
                h_p = torch.flip(h_p, dims=[2])
            elif i == 3:
                m_l = torch.rot90(m_l, k=-1, dims=[2, 3])
                h_p = torch.rot90(h_p, k=-1, dims=[2, 3])
            logits_list.append(m_l[0, 0].cpu().numpy())
            heights_list.append(h_p[0, 0].cpu().numpy())

    return np.mean(logits_list, axis=0), np.mean(heights_list, axis=0)


@torch.no_grad()
def predict_mask_and_height(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: torch.device,
    tile_size: int = IMG_SIZE,
    stride: int = 128,
    max_height_m: float = MAX_HEIGHT_M,
    mask_thresh: float = 0.5,
    use_tta: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window inference at NATIVE resolution.
    - 256x256 tiles at native scale (no stretching/distorting aspect ratio)
    - 50% overlap (stride 128)
    - Reflect-padded borders (images < 256 are padded, not stretched)
    - 2D Hann window blending
    - Averages logits and heights, then thresholds mask and scales height
    """
    orig_h, orig_w = image_rgb.shape[:2]
    model.eval()

    # Pad if smaller than tile_size
    pad_h_min = max(0, tile_size - orig_h)
    pad_w_min = max(0, tile_size - orig_w)

    pad_top = pad_h_min // 2
    pad_bottom = pad_h_min - pad_top
    pad_left = pad_w_min // 2
    pad_right = pad_w_min - pad_left

    if pad_h_min > 0 or pad_w_min > 0:
        padded = np.pad(
            image_rgb,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="reflect" if min(orig_h, orig_w) > 1 else "edge",
        )
    else:
        padded = image_rgb.copy()

    curr_h, curr_w = padded.shape[:2]

    # Generate sliding window coordinates that cover the entire padded image
    def get_coords(length: int, window: int, step: int) -> list[int]:
        if length <= window:
            return [0]
        coords = list(range(0, length - window + 1, step))
        if coords[-1] != length - window:
            coords.append(length - window)
        return coords

    y_coords = get_coords(curr_h, tile_size, stride)
    x_coords = get_coords(curr_w, tile_size, stride)

    window = get_2d_hann_window(tile_size)

    accum_logits = np.zeros((curr_h, curr_w), dtype=np.float32)
    accum_height = np.zeros((curr_h, curr_w), dtype=np.float32)
    accum_weight = np.zeros((curr_h, curr_w), dtype=np.float32)

    for y in y_coords:
        for x in x_coords:
            tile = padded[y : y + tile_size, x : x + tile_size]
            t_logits, t_height = _predict_single_tile(model, tile, device, use_tta=use_tta)

            accum_logits[y : y + tile_size, x : x + tile_size] += t_logits * window
            accum_height[y : y + tile_size, x : x + tile_size] += t_height * window
            accum_weight[y : y + tile_size, x : x + tile_size] += window

    accum_weight = np.maximum(accum_weight, 1e-6)
    blended_logits = accum_logits / accum_weight
    blended_height = accum_height / accum_weight

    # Crop back to original dimensions (unpad)
    unpadded_logits = blended_logits[pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]
    unpadded_height = blended_height[pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]

    mask_prob = 1.0 / (1.0 + np.exp(-np.clip(unpadded_logits, -15.0, 15.0)))
    mask_bin = (mask_prob > mask_thresh).astype(np.uint8)

    height_norm = np.maximum(unpadded_height, 0.0)
    height_m = height_norm * max_height_m * mask_bin

    return mask_prob, mask_bin, height_m
