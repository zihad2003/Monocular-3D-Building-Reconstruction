"""Training / validation epoch loop and train_model entrypoint."""
from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.training.losses import MultiTaskLoss
from src.training.metrics import compute_metrics


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    agg = {"loss": 0.0, "iou": 0.0, "f1": 0.0, "height_mae_m": 0.0, "height_rmse_m": 0.0}
    n_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            heights = batch["height"].to(device, non_blocking=True)

            mask_logits, height_pred = model(images)
            loss, _, _ = criterion(mask_logits, height_pred, masks, heights)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            metrics = compute_metrics(mask_logits, height_pred, masks, heights)
            agg["loss"] += loss.item()
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
    w_height=1.0,
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
    best_val_iou = -1.0
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_stats = run_epoch(model, train_loader, criterion, device, optimizer)
        val_stats = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()

        history["train"].append(train_stats)
        history["val"].append(val_stats)

        if val_stats["iou"] > best_val_iou:
            best_val_iou = val_stats["iou"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "encoder_name": encoder_name,
                    "max_height_m": max_height_m,
                    "img_size": img_size,
                    "best_val_iou": best_val_iou,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{epochs} | {dt:5.1f}s | "
            f"train loss {train_stats['loss']:.4f} IoU {train_stats['iou']:.3f} | "
            f"val loss {val_stats['loss']:.4f} IoU {val_stats['iou']:.3f} "
            f"F1 {val_stats['f1']:.3f} MAE {val_stats['height_mae_m']:.2f}m "
            f"RMSE {val_stats['height_rmse_m']:.2f}m",
            flush=True,
        )

    print(
        f"\nBest val IoU: {best_val_iou:.3f} — checkpoint saved to {checkpoint_path}",
        flush=True,
    )
    return history
