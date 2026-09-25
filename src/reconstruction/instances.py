"""Post-hoc watershed split of touching buildings in a semantic mask."""
from __future__ import annotations

import cv2
import numpy as np


def split_touching_instances(mask_bin, min_area_px=40, min_distance=8):
    """
    Distance-transform + watershed. This is NOT learned instance segmentation.
    """
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    mask_u8 = (mask_bin > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return np.zeros_like(mask_u8, dtype=np.int32)

    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    coords = peak_local_max(dist, min_distance=min_distance, labels=mask_u8)
    peak_mask = np.zeros_like(dist, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)
    labels = watershed(-dist, markers, mask=mask_u8)

    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        if (labels == lbl).sum() < min_area_px:
            labels[labels == lbl] = 0
    return labels
