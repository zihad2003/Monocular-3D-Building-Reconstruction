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
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fraction of tiles to assign to validation split")
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
    ids = preprocess_real_height_data(
        image_paths=image_paths,
        dsm_paths=dsm_paths,
        out_root=args.out_dir,
        mask_paths=mask_paths if len(mask_paths) == len(image_paths) else None,
    )

    # Generate train / val splits.json
    if ids:
        rng = random.Random(args.seed)
        shuffled = ids.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * args.val_ratio))
        val_ids = shuffled[:n_val]
        train_ids = shuffled[n_val:]
        
        split_path = Path(args.out_dir) / "splits.json"
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump({"train": train_ids, "val": val_ids}, f, indent=2)
        print(f"Saved splits to {split_path}: {len(train_ids)} train, {len(val_ids)} val tiles.")

if __name__ == "__main__":
    main()
