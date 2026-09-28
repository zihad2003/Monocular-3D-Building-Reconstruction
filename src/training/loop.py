"""Training / validation epoch loop and train_model entrypoint."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.training.losses import MultiTaskLoss
from src.training.metrics import compute_metrics


def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def get_package_versions() -> dict[str, str]:
    versions = {"torch": torch.__version__}
    try:
        import segmentation_models_pytorch as smp
        versions["smp"] = smp.__version__
    except Exception:
        versions["smp"] = "unknown"
    return versions


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    agg = {
        "loss": 0.0,
        "seg_loss": 0.0,
        "height_loss": 0.0,
        "iou": 0.0,
        "f1": 0.0,
        "height_mae_m": 0.0,
        "height_mae_gt_m": 0.0,
        "height_mae_pred_m": 0.0,
        "height_rmse_m": 0.0,
    }
    n_batches = 0

    from tqdm import tqdm
    
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc="Train" if is_train else "Val")
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            heights = batch["height"].to(device, non_blocking=True)

            mask_logits, height_pred = model(images)
            loss, seg_l, height_l = criterion(mask_logits, height_pred, masks, heights)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            metrics = compute_metrics(mask_logits, height_pred, masks, heights)
            agg["loss"] += loss.item()
            agg["seg_loss"] += seg_l.item()
            agg["height_loss"] += height_l.item()
            for k, v in metrics.items():
                agg[k] += v
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in agg.items()}


def train_model(
    model,
    train_dataset,
    val_dataset,
    epochs=15,
    batch_size=16,
    lr=3e-4,
    weight_decay=1e-4,
    num_workers=0,
    w_seg=1.0,
    w_height=10.0,
    checkpoint_path: str | Path = "checkpoints/best_model.pt",
    encoder_name: str = "resnet34",
    max_height_m: float = 60.0,
    img_size: int = 256,
    device=None,
):
    device = device or get_device()
    model = model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    criterion = MultiTaskLoss(w_seg=w_seg, w_height=w_height)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train": [], "val": []}
    best_val_score = float("-inf")
    best_val_iou = -1.0
    best_val_mae = float("inf")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_path.parent / f"history_{checkpoint_path.stem}.json"

    git_hash = get_git_commit_hash()
    pkg_versions = get_package_versions()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_stats = run_epoch(model, train_loader, criterion, device, optimizer)
        val_stats = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()

        history["train"].append(train_stats)
        history["val"].append(val_stats)

        # Select best checkpoint with combined score: val_iou - 0.02 * val_height_mae_m
        val_score = val_stats["iou"] - 0.02 * val_stats["height_mae_m"]
        if val_score > best_val_score:
            best_val_score = val_score
            best_val_iou = val_stats["iou"]
            best_val_mae = val_stats["height_mae_m"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "encoder_name": encoder_name,
                    "max_height_m": max_height_m,
                    "img_size": img_size,
                    "best_val_score": best_val_score,
                    "best_val_iou": best_val_iou,
                    "best_val_mae": best_val_mae,
                    "epoch": epoch,
                    "git_commit": git_hash,
                    "package_versions": pkg_versions,
                },
                checkpoint_path,
            )

        # Save history JSON every epoch
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{epochs} | {dt:5.1f}s | "
            f"train loss {train_stats['loss']:.4f} (seg {train_stats['seg_loss']:.4f}, hgt {train_stats['height_loss']:.4f}) IoU {train_stats['iou']:.3f} | "
            f"val loss {val_stats['loss']:.4f} IoU {val_stats['iou']:.3f} "
            f"F1 {val_stats['f1']:.3f} MAE_GT {val_stats['height_mae_gt_m']:.2f}m "
            f"MAE_Pred {val_stats['height_mae_pred_m']:.2f}m Score {val_score:.4f}",
            flush=True,
        )

    print(
        f"\nBest val score: {best_val_score:.4f} (IoU: {best_val_iou:.3f}, MAE: {best_val_mae:.2f}m) — checkpoint saved to {checkpoint_path}",
        flush=True,
    )
    return history

