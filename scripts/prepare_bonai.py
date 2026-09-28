import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess_bonai import preprocess_bonai

def main():
    parser = argparse.ArgumentParser(description="Preprocess BONAI data")
    parser.add_argument("--json", type=str, required=True, help="Path to BONAI COCO JSON annotation")
    parser.add_argument("--images", type=str, required=True, help="Path to BONAI images directory")
    parser.add_argument("--out", type=str, default="./data/bonai", help="Output directory for tiles")
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    print(f"Starting BONAI preprocessing...")
    print(f"Annotations: {args.json}")
    print(f"Images: {args.images}")
    print(f"Output: {args.out}")
    
    preprocess_bonai(args.json, args.images, args.out)
    
if __name__ == "__main__":
    main()
