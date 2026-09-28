import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess_height import preprocess_real_height_data

def main():
    parser = argparse.ArgumentParser(description="Prepare real/synthetic height dataset (e.g. SynRS3D, US3D) for training")
    parser.add_argument("--image_dir", type=str, default="raw_data/height_data/images", help="Directory containing RGB images")
    parser.add_argument("--dsm_dir", type=str, default="raw_data/height_data/dsm", help="Directory containing corresponding DSM/nDSM height images")
    parser.add_argument("--mask_dir", type=str, default="raw_data/height_data/masks", help="Directory containing semantic masks (optional)")
    parser.add_argument("--out_dir", type=str, default="data/synrs3d", help="Output root directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of image pairs to process (default: all)")
    parser.add_argument("--building-class-id", type=int, default=8, help="Semantic class ID for buildings (default: 8 for SynRS3D)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of scenes to assign to validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()

    # Fast O(N) index-based matching
    img_files = sorted(os.listdir(args.image_dir))
    dsm_files_set = {f: os.path.join(args.dsm_dir, f) for f in os.listdir(args.dsm_dir)}
    mask_files_set = {f: os.path.join(args.mask_dir, f) for f in os.listdir(args.mask_dir)} if os.path.exists(args.mask_dir) else {}

    image_paths = []
    dsm_paths = []
    mask_paths = []

    for fname in img_files:
        if fname in dsm_files_set:
            image_paths.append(os.path.join(args.image_dir, fname))
            dsm_paths.append(dsm_files_set[fname])
            if fname in mask_files_set:
                mask_paths.append(mask_files_set[fname])

    if not image_paths:
        print("No matching image/DSM pairs found. Please check your directories and filenames.")
        return

    if args.max_samples is not None and args.max_samples > 0:
        image_paths = image_paths[:args.max_samples]
        dsm_paths = dsm_paths[:args.max_samples]
        if mask_paths:
            mask_paths = mask_paths[:args.max_samples]

    print(f"Found {len(image_paths)} matching pairs (masks available: {len(mask_paths) == len(image_paths)}). Starting preprocessing...")
    ids, scene_to_tiles = preprocess_real_height_data(
        image_paths=image_paths,
        dsm_paths=dsm_paths,
        out_root=args.out_dir,
        mask_paths=mask_paths if len(mask_paths) == len(image_paths) else None,
        building_class_id=args.building_class_id,
    )

    # Generate train / val splits.json split by SCENE (source image filename)
    # Tiles from one source image MUST all go to the same split.
    if scene_to_tiles:
        scenes = sorted(list(scene_to_tiles.keys()))
        rng = random.Random(args.seed)
        shuffled_scenes = scenes.copy()
        rng.shuffle(shuffled_scenes)

        n_val_scenes = max(1, int(len(shuffled_scenes) * args.val_ratio))
        val_scenes = sorted(shuffled_scenes[:n_val_scenes])
        train_scenes = sorted(shuffled_scenes[n_val_scenes:])

        # Assert scene sets are strictly disjoint
        assert set(train_scenes).isdisjoint(set(val_scenes)), "Train and Val scene sets must be disjoint!"

        train_ids = [tid for sid in train_scenes for tid in scene_to_tiles[sid]]
        val_ids = [tid for sid in val_scenes for tid in scene_to_tiles[sid]]

        # Assert tile sets are also strictly disjoint
        assert set(train_ids).isdisjoint(set(val_ids)), "Train and Val tile sets must be disjoint!"

        split_path = Path(args.out_dir) / "splits.json"
        splits_payload = {
            "train": train_ids,
            "val": val_ids,
            "train_scenes": train_scenes,
            "val_scenes": val_scenes,
            "stats": {
                "total_scenes": len(scenes),
                "train_scenes_count": len(train_scenes),
                "val_scenes_count": len(val_scenes),
                "total_tiles": len(ids),
                "train_tiles_count": len(train_ids),
                "val_tiles_count": len(val_ids),
            }
        }
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(splits_payload, f, indent=2)
        print(
            f"Saved scene-disjoint splits to {split_path}:\n"
            f"  Train: {len(train_scenes)} scenes ({len(train_ids)} tiles)\n"
            f"  Val:   {len(val_scenes)} scenes ({len(val_ids)} tiles)"
        )

if __name__ == "__main__":
    main()
