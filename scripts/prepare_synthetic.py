"""Prepare synthetic train/val tiles under data/."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.synthetic import build_synthetic_dataset
from src.paths import data_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-val", type=int, default=40)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.out) if args.out else data_root()
    train_ids = build_synthetic_dataset(args.n_train, out, seed_offset=0)
    val_ids = build_synthetic_dataset(args.n_val, out, seed_offset=args.n_train)

    split_path = out / "splits.json"
    split_path.write_text(
        json.dumps({"train": train_ids, "val": val_ids}, indent=2),
        encoding="utf-8",
    )
    print(f"Train: {len(train_ids)} | Val: {len(val_ids)}")
    print(f"Wrote splits -> {split_path}")


if __name__ == "__main__":
    main()
