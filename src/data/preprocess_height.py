"""
Pipeline for processing datasets with real height labels (e.g., US3D, DFC2019, SpaceNet).
This extracts synchronized tiles of RGB images, footprint masks (derived from height > 0),
and metric height maps (.npy).
"""
from __future__ import annotations

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from src.data.dataset import IMG_SIZE

def preprocess_real_height_data(
    image_paths: list[str],
    dsm_paths: list[str],
    out_root: str,
    mask_paths: list[str] | None = None,
    img_size: int = IMG_SIZE,
    height_threshold_m: float = 2.0,
    building_class_id: int = 8,
    isolate_building_height: bool = True,
):
    """
    Process paired RGB, DSM (height), and optional semantic mask images into training tiles.
    
    Args:
        image_paths: List of paths to RGB images (.png, .jpg, .tif)
        dsm_paths: List of paths to corresponding DSM / nDSM (height) images
        out_root: Output directory for the dataset tiles (images/, masks/, heights/)
        mask_paths: Optional list of paths to semantic segmentation masks
        img_size: Tile size to crop (default 256)
        height_threshold_m: Minimum height in meters if masks are not provided
        building_class_id: Semantic class ID for buildings (default 8 for SynRS3D)
        isolate_building_height: If True, set height to 0 outside building footprints
    """
    os.makedirs(os.path.join(out_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "masks"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "heights"), exist_ok=True)

    ids = []
    scene_to_tiles: dict[str, list[str]] = {}
    has_masks = mask_paths is not None and len(mask_paths) == len(image_paths)

    for idx, (img_p, dsm_p) in enumerate(zip(image_paths, dsm_paths)):
        if not os.path.exists(img_p) or not os.path.exists(dsm_p):
            continue
            
        # Load RGB image
        try:
            img = np.array(Image.open(img_p).convert("RGB"))
        except Exception as e:
            print(f"Error loading image {img_p}: {e}")
            continue

        # Load DSM (float32 height in meters)
        dsm = cv2.imread(dsm_p, cv2.IMREAD_UNCHANGED)
        if dsm is None:
            continue
            
        if len(dsm.shape) > 2:
            dsm = dsm[:, :, 0]
            
        h, w = img.shape[:2]
        if dsm.shape[:2] != (h, w):
            dsm = cv2.resize(dsm, (w, h), interpolation=cv2.INTER_NEAREST)

        # Load semantic mask if available
        mask = None
        if has_masks:
            m_p = mask_paths[idx]
            if os.path.exists(m_p):
                raw_mask = cv2.imread(m_p, cv2.IMREAD_UNCHANGED)
                if raw_mask is not None:
                    if len(raw_mask.shape) > 2:
                        raw_mask = raw_mask[:, :, 0]
                    if raw_mask.shape[:2] != (h, w):
                        raw_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    vals, counts = np.unique(raw_mask, return_counts=True)
                    if idx < 20:
                        hist_str = ", ".join(f"class {int(v)}: {int(c)}px" for v, c in zip(vals, counts))
                        print(f"[Mask Hist {idx + 1}/20] {Path(m_p).name}: {hist_str}")

                    # SynRS3D uses class 8 for buildings (verified from SynRS3D official docs).
                    # Explicit validation: do NOT silently fallback to raw_mask > 0.
                    if building_class_id in vals:
                        mask = (raw_mask == building_class_id).astype(np.uint8) * 255
                    else:
                        print(
                            f"[Warning] building_class_id {building_class_id} absent in {Path(m_p).name} "
                            f"(unique classes present: {vals.tolist()}); skipping scene."
                        )
                        continue

        file_name = Path(img_p).stem
        scene_to_tiles[file_name] = []

        # Tiling process (sliding window / grid)
        for ty in range(0, h - img_size + 1, img_size):
            for tx in range(0, w - img_size + 1, img_size):
                dsm_tile = dsm[ty : ty + img_size, tx : tx + img_size].astype(np.float32)
                # Clean invalid height values
                dsm_tile = np.nan_to_num(dsm_tile, nan=0.0, posinf=0.0, neginf=0.0)
                dsm_tile = np.clip(dsm_tile, 0.0, None)

                if mask is not None:
                    m_tile = mask[ty : ty + img_size, tx : tx + img_size]
                else:
                    m_tile = (dsm_tile > height_threshold_m).astype(np.uint8) * 255

                # Skip tiles with no buildings to avoid empty sample imbalance
                if m_tile.sum() == 0:
                    continue

                if isolate_building_height:
                    # Height strictly inside building footprint (zero out trees/ground)
                    dsm_tile = dsm_tile * (m_tile > 0)

                img_tile = img[ty : ty + img_size, tx : tx + img_size]
                sid = f"{file_name}_{ty}_{tx}"

                Image.fromarray(img_tile).save(os.path.join(out_root, "images", f"{sid}.png"))
                Image.fromarray(m_tile).save(os.path.join(out_root, "masks", f"{sid}.png"))
                np.save(os.path.join(out_root, "heights", f"{sid}.npy"), dsm_tile)

                ids.append(sid)
                scene_to_tiles[file_name].append(sid)

        # Remove scene if no valid tiles were produced
        if not scene_to_tiles[file_name]:
            del scene_to_tiles[file_name]

    print(f"Height data preprocessing done: {len(ids)} valid tiles from {len(scene_to_tiles)} scenes -> {out_root}")
    return ids, scene_to_tiles


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess height dataset with explicit building class ID validation")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing RGB images")
    parser.add_argument("--dsm_dir", type=str, required=True, help="Directory containing DSM height images")
    parser.add_argument("--mask_dir", type=str, default=None, help="Directory containing semantic masks")
    parser.add_argument("--out_dir", type=str, default="data/synrs3d", help="Output root directory")
    parser.add_argument("--building-class-id", type=int, default=8, help="Semantic class ID for buildings (default: 8 for SynRS3D)")
    args = parser.parse_args()

    img_files = sorted(os.listdir(args.image_dir))
    img_paths = [os.path.join(args.image_dir, f) for f in img_files]
    dsm_paths = [os.path.join(args.dsm_dir, f) for f in img_files if os.path.exists(os.path.join(args.dsm_dir, f))]
    mask_paths = [os.path.join(args.mask_dir, f) for f in img_files if os.path.exists(os.path.join(args.mask_dir, f))] if args.mask_dir else None

    preprocess_real_height_data(
        image_paths=img_paths,
        dsm_paths=dsm_paths,
        out_root=args.out_dir,
        mask_paths=mask_paths,
        building_class_id=args.building_class_id,
    )
