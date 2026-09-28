"""
BONAI COCO-style annotations -> images/masks tiles.

DOCUMENTATION OF FIELDS IN BONAI (ROOF VS FOOTPRINT):
In off-nadir (oblique) satellite imagery like BONAI, building structures exhibit substantial geometric perspective distortion:
- 'roof' (or 'roof_segmentation'): The elevated top boundary of the building visible in the image.
- 'footprint': The ground-level boundary where the building walls intersect the earth.
- 'offset': The 2D displacement vector [dx, dy] separating roof from footprint due to building height and sensor look angle.
- 'segmentation': The standard COCO polygon key (in BONAI instances, standard format provides footprint or roof under this key).

WHY FOOTPRINT IS USED FOR 3D RECONSTRUCTION:
In LoD1 3D architectural reconstruction, building volumes are extruded vertically UPWARD starting from ground elevation (z=0).
Using the roof polygon causes buildings to be anchored at displaced spatial positions, producing incorrect footprints and misaligned
terrain intersections. The ground-level 'footprint' is the true physical anchor for vertical extrusion.
Default is field='footprint' (with automatic fallback to 'segmentation').
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from PIL import Image

from src.data.dataset import IMG_SIZE


def preprocess_bonai(
    coco_json_path,
    images_dir,
    out_root,
    field="footprint",
    img_size=IMG_SIZE,
    write_zero_heights=True,
):
    """Rasterize BONAI footprints (or roofs) into fixed-size tiles (no metric height)."""
    from pycocotools.coco import COCO
    from pycocotools import mask as mask_utils

    os.makedirs(os.path.join(out_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "masks"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "heights"), exist_ok=True)

    print(f"Loading BONAI COCO annotations from: {coco_json_path}")
    print(f"Target polygon field: '{field}' (for 3D building reconstruction, ground footprint is required)")
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
            target_field = field
            if target_field not in ann:
                if field == "footprint" and "segmentation" in ann:
                    target_field = "segmentation"
                elif field == "segmentation" and "footprint" in ann:
                    target_field = "footprint"
                elif "segmentation" in ann:
                    target_field = "segmentation"
                else:
                    continue

            seg_data = ann.get(target_field)
            if not seg_data:
                continue

            if isinstance(seg_data, list):
                rle = mask_utils.frPyObjects(seg_data, h, w)
                inst_mask = mask_utils.decode(mask_utils.merge(rle))
            else:
                inst_mask = mask_utils.decode(seg_data)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess BONAI dataset with selectable polygon field")
    parser.add_argument("--json", type=str, required=True, help="Path to BONAI COCO JSON annotation file")
    parser.add_argument("--images", type=str, required=True, help="Path to BONAI images directory")
    parser.add_argument("--out", type=str, default="./data/bonai", help="Output directory for tiles")
    parser.add_argument("--field", type=str, default="footprint", choices=["footprint", "segmentation", "roof", "roof_segmentation"], help="Target polygon annotation field (default: footprint)")
    args = parser.parse_args()

    preprocess_bonai(args.json, args.images, args.out, field=args.field)
