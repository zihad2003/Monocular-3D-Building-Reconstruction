"""
BONAI COCO-style annotations -> images/masks tiles.

IMPORTANT (project draft): BONAI public annotations emphasize roof, footprint,
and roof-to-footprint offset. They are NOT a complete metric-height dataset.
This preprocessor therefore writes footprint masks only. Height maps are left
as zeros unless you later fuse a separate height source via preprocess_height.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image

from src.data.dataset import IMG_SIZE


def preprocess_bonai(
    coco_json_path,
    images_dir,
    out_root,
    img_size=IMG_SIZE,
    write_zero_heights=True,
):
    """Rasterize BONAI footprints into fixed-size tiles (no metric height)."""
    from pycocotools.coco import COCO
    from pycocotools import mask as mask_utils

    os.makedirs(os.path.join(out_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "masks"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "heights"), exist_ok=True)

    coco = COCO(coco_json_path)
    ids = []

    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        h, w = img_info["height"], img_info["width"]

        src_path = os.path.join(images_dir, file_name)
        if not os.path.exists(src_path):
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        if len(anns) == 0:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            if "segmentation" not in ann:
                continue
            if isinstance(ann["segmentation"], list):
                rle = mask_utils.frPyObjects(ann["segmentation"], h, w)
                inst_mask = mask_utils.decode(mask_utils.merge(rle))
            else:
                inst_mask = mask_utils.decode(ann["segmentation"])
            inst_mask = inst_mask > 0
            mask[inst_mask] = 255

        if mask.sum() == 0:
            continue

        img = np.array(Image.open(src_path).convert("RGB"))
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        sample_id_base = os.path.splitext(file_name)[0]
        for ty in range(0, h - img_size + 1, img_size):
            for tx in range(0, w - img_size + 1, img_size):
                m_tile = mask[ty : ty + img_size, tx : tx + img_size]
                if m_tile.sum() == 0:
                    continue
                img_tile = img[ty : ty + img_size, tx : tx + img_size]
                sid = f"{sample_id_base}_{ty}_{tx}"
                Image.fromarray(img_tile).save(os.path.join(out_root, "images", f"{sid}.png"))
                Image.fromarray(m_tile).save(os.path.join(out_root, "masks", f"{sid}.png"))
                if write_zero_heights:
                    np.save(
                        os.path.join(out_root, "heights", f"{sid}.npy"),
                        np.zeros((img_size, img_size), dtype=np.float32),
                    )
                ids.append(sid)

    print(f"BONAI preprocessing done: {len(ids)} tiles from {coco_json_path}")
    return ids
