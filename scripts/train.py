"""Train MultiTaskBuildingNet from configs/default.yaml (or CLI overrides)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import SatelliteBuildingDataset
from src.data.synthetic import build_synthetic_dataset
from src.models import MultiTaskBuildingNet
from src.paths import checkpoints_dir, data_root
from src.training import get_device, train_model


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_splits(cfg: dict):
    root = data_root() if cfg["data"].get("root") in (None, "data") else Path(cfg["data"]["root"])
    root = Path(root)
    for sub in ("images", "masks", "heights"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    split_path = root / "splits.json"
    if split_path.exists():
        splits = json.loads(split_path.read_text(encoding="utf-8"))
        print(f"Loaded existing splits from {split_path}")
        return root, splits["train"], splits["val"]

    if not cfg["data"].get("use_synthetic", True):
        raise FileNotFoundError(
            f"No {split_path}. Run prepare_synthetic.py or provide BONAI/height splits."
        )

    n_train = int(cfg["data"]["n_train"])
    n_val = int(cfg["data"]["n_val"])
    print(f"Building synthetic dataset: train={n_train}, val={n_val} -> {root}")
    train_ids = build_synthetic_dataset(n_train, root, seed_offset=0)
    val_ids = build_synthetic_dataset(n_val, root, seed_offset=n_train)
    split_path.write_text(
        json.dumps({"train": train_ids, "val": val_ids}, indent=2),
        encoding="utf-8",
    )
    return root, train_ids, val_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--encoder", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.encoder is not None:
        cfg["model"]["encoder_name"] = args.encoder

    set_seed(int(cfg.get("seed", 42)))
    device = get_device()
    print(f"Device: {device}")

    data_dir, train_ids, val_ids = ensure_splits(cfg)
    img_size = int(cfg.get("img_size", 256))
    max_h = float(cfg.get("max_height_m", 60.0))

    train_ds = SatelliteBuildingDataset(
        data_dir, train_ids, img_size=img_size, augment=True, max_height_m=max_h
    )
    val_ds = SatelliteBuildingDataset(
        data_dir, val_ids, img_size=img_size, augment=False, max_height_m=max_h
    )
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    encoder_name = cfg["model"]["encoder_name"]
    model = MultiTaskBuildingNet(
        encoder_name=encoder_name,
        encoder_weights=cfg["model"].get("encoder_weights", "imagenet"),
    )

    ckpt = checkpoints_dir() / cfg["paths"].get("checkpoint_name", "best_model.pt")
    history = train_model(
        model,
        train_ds,
        val_ds,
        epochs=int(cfg["train"]["epochs"]),
        batch_size=int(cfg["train"]["batch_size"]),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
        num_workers=int(cfg["train"].get("num_workers", 0)),
        w_seg=float(cfg["train"].get("w_seg", 1.0)),
        w_height=float(cfg["train"].get("w_height", 1.0)),
        checkpoint_path=ckpt,
        encoder_name=encoder_name,
        max_height_m=max_h,
        img_size=img_size,
        device=device,
    )

    hist_path = checkpoints_dir() / "history.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"History saved -> {hist_path}")


if __name__ == "__main__":
    main()
