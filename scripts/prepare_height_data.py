import argparse
import os
import glob
from src.data.preprocess_height import preprocess_real_height_data

def main():
    parser = argparse.ArgumentParser(description="Prepare real height dataset (e.g. US3D) for training")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing high-res RGB images")
    parser.add_argument("--dsm_dir", type=str, required=True, help="Directory containing corresponding DSM/Height images")
    parser.add_argument("--out_dir", type=str, default="data/real_height", help="Output root directory")
    args = parser.parse_args()

    # Find matching files (assuming they have the same base name)
    img_files = sorted(glob.glob(os.path.join(args.image_dir, "*.*")))
    
    image_paths = []
    dsm_paths = []
    
    for img_p in img_files:
        base_name = os.path.splitext(os.path.basename(img_p))[0]
        # Look for matching DSM file (could be .tif, .png)
        possible_dsm = glob.glob(os.path.join(args.dsm_dir, f"{base_name}*"))
        if possible_dsm:
            image_paths.append(img_p)
            dsm_paths.append(possible_dsm[0])
            
    if not image_paths:
        print("No matching image/DSM pairs found. Please check your directories and filenames.")
        return

    print(f"Found {len(image_paths)} matching pairs. Starting preprocessing...")
    preprocess_real_height_data(image_paths, dsm_paths, args.out_dir)

if __name__ == "__main__":
    main()
