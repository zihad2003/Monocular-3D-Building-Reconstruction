"""Evaluate MultiTaskBuildingNet checkpoint on scene-disjoint validation split.
Computes IoU, F1, Height MAE (GT & Predicted mask), Height RMSE, and saves qualitative 5-panel visualizations.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import SatelliteBuildingDataset, MAX_HEIGHT_M
from src.models.multitask_net import MultiTaskBuildingNet
from src.training.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 3D Building Reconstruction checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--data-root", type=str, default="data/synrs3d", help="Path to data directory containing images, masks, heights")
    parser.add_argument("--splits", type=str, default="data/synrs3d/splits.json", help="Path to splits.json")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Directory to save evaluation results and panels")
    parser.add_argument("--num-panels", type=int, default=10, help="Number of qualitative panels to save")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    return parser.parse_args()


def colorize_height(height_m: np.ndarray, max_val: float = 40.0) -> np.ndarray:
    """Colorize 2D height map using Turbo colormap."""
    norm = np.clip(height_m / max(max_val, 1e-3), 0.0, 1.0)
    uint8_img = (norm * 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(uint8_img, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"==> Evaluating on device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    encoder_name = ckpt.get("encoder_name", "resnet18") if isinstance(ckpt, dict) else "resnet18"
    state_dict = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    max_height_m = ckpt.get("max_height_m", MAX_HEIGHT_M) if isinstance(ckpt, dict) else MAX_HEIGHT_M

    print(f"Loading MultiTaskBuildingNet with encoder: {encoder_name}, max_height_m: {max_height_m}")
    model = MultiTaskBuildingNet(encoder_name=encoder_name, encoder_weights=None)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Load splits
    splits_path = Path(args.splits)
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_path}")
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    val_ids = splits.get("val", [])
    if not val_ids:
        raise ValueError(f"No validation samples found in {splits_path}")

    # Validate scene disjointness if scene keys exist
    if "train_scenes" in splits and "val_scenes" in splits:
        train_s = set(splits["train_scenes"])
        val_s = set(splits["val_scenes"])
        overlap = train_s.intersection(val_s)
        assert len(overlap) == 0, f"Scene leakage detected! Overlapping scenes: {overlap}"
        print(f"Verified scene-disjoint validation set: {len(val_s)} val scenes ({len(val_ids)} tiles).")

    val_dataset = SatelliteBuildingDataset(
        root=args.data_root,
        ids=val_ids,
        augment=False,
        max_height_m=max_height_m,
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"Running inference over {len(val_dataset)} validation tiles...")
    agg = {
        "iou": 0.0,
        "f1": 0.0,
        "height_mae_m": 0.0,
        "height_mae_gt_m": 0.0,
        "height_mae_pred_m": 0.0,
        "height_rmse_m": 0.0,
    }
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            heights = batch["height"].to(device)

            mask_logits, height_pred = model(images)
            metrics = compute_metrics(
                mask_logits,
                height_pred,
                masks,
                heights,
                max_height_m=max_height_m,
            )
            for k, v in metrics.items():
                if k in agg:
                    agg[k] += v
            n_batches += 1

    final_metrics = {k: round(v / max(n_batches, 1), 4) for k, v in agg.items()}

    print("\n" + "=" * 50)
    print("       SCENE-DISJOINT EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Checkpoint:           {args.checkpoint}")
    print(f"  Validation Tiles:     {len(val_dataset)}")
    print(f"  Footprint IoU:        {final_metrics['iou']:.4f} ({final_metrics['iou']*100:.2f}%)")
    print(f"  Footprint F1 Score:   {final_metrics['f1']:.4f}")
    print(f"  Height MAE (GT Foot): {final_metrics['height_mae_gt_m']:.2f} m")
    print(f"  Height MAE (Pred Ft): {final_metrics['height_mae_pred_m']:.2f} m")
    print(f"  Height RMSE:          {final_metrics['height_rmse_m']:.2f} m")
    print("=" * 50 + "\n")

    # Save metrics to JSON
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "eval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_path}")

    # Generate and save qualitative panels
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    num_to_save = min(args.num_panels, len(val_dataset))
    print(f"Saving {num_to_save} qualitative comparison panels to: {panels_dir} ...")

    # Sample evenly across validation set
    indices = np.linspace(0, len(val_dataset) - 1, num_to_save, dtype=int)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample_id = val_ids[idx]
            batch_item = val_dataset[idx]
            img_t = batch_item["image"].unsqueeze(0).to(device)
            mask_gt = batch_item["mask"].squeeze().cpu().numpy()
            height_gt = batch_item["height"].squeeze().cpu().numpy() * max_height_m

            mask_logits, height_pred = model(img_t)
            pred_prob = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
            pred_mask = (pred_prob > 0.5).astype(np.float32)
            pred_height = (height_pred.squeeze().cpu().numpy() * max_height_m) * pred_mask

            # Denormalize image for display
            img_np = batch_item["image"].permute(1, 2, 0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            rgb = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)

            # Colorize heights
            max_disp_h = max(30.0, float(height_gt.max()), float(pred_height.max()))
            gt_h_color = colorize_height(height_gt, max_val=max_disp_h)
            pred_h_color = colorize_height(pred_height, max_val=max_disp_h)

            fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
            axes[0].imshow(rgb)
            axes[0].set_title(f"RGB ({sample_id})", fontsize=10)
            axes[0].axis("off")

            axes[1].imshow(mask_gt, cmap="gray")
            axes[1].set_title("GT Mask", fontsize=10)
            axes[1].axis("off")

            axes[2].imshow(pred_mask, cmap="gray")
            axes[2].set_title("Pred Mask", fontsize=10)
            axes[2].axis("off")

            axes[3].imshow(gt_h_color)
            axes[3].set_title(f"GT Height (max {height_gt.max():.1f}m)", fontsize=10)
            axes[3].axis("off")

            axes[4].imshow(pred_h_color)
            axes[4].set_title(f"Pred Height (max {pred_height.max():.1f}m)", fontsize=10)
            axes[4].axis("off")

            plt.tight_layout()
            panel_path = panels_dir / f"panel_{i + 1:02d}_{sample_id}.png"
            plt.savefig(panel_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"All {num_to_save} qualitative panels generated successfully in: {panels_dir}")


if __name__ == "__main__":
    main()
