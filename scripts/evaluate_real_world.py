"""Run qualitative inference on real-world satellite crops (without height GT).
Produces qualitative footprint and heightmap visualizations labeled clearly as qualitative.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.multitask_net import MultiTaskBuildingNet
from src.reconstruction.predict import predict_mask_and_height
from src.reconstruction.extrude import mask_and_height_to_3d_mesh


def parse_args():
    parser = argparse.ArgumentParser(description="Real-world qualitative evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--samples-dir", type=str, default="test_samples", help="Directory with real satellite images")
    parser.add_argument("--output-dir", type=str, default="eval_results/real_world_qualitative", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu' or 'cuda')")
    return parser.parse_args()


def colorize_height(height_m: np.ndarray, max_val: float = 40.0) -> np.ndarray:
    norm = np.clip(height_m / max(max_val, 1e-3), 0.0, 1.0)
    uint8_img = (norm * 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(uint8_img, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    encoder_name = ckpt.get("encoder_name", "resnet18") if isinstance(ckpt, dict) else "resnet18"
    state_dict = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt

    model = MultiTaskBuildingNet(encoder_name=encoder_name, encoder_weights=None)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    samples_dir = Path(args.samples_dir)
    image_paths = sorted(list(samples_dir.glob("*.png")) + list(samples_dir.glob("*.jpg")))
    print(f"Processing {len(image_paths)} real-world satellite crops from {samples_dir}...")

    summary_list = []
    for img_p in image_paths:
        name = img_p.stem
        img_pil = Image.open(img_p).convert("RGB")
        img_np = np.array(img_pil)

        prob, mask, height_m = predict_mask_and_height(model, img_np, device=device)
        n_pixels = int(mask.sum())
        mean_h = float(height_m[mask > 0].mean()) if n_pixels > 0 else 0.0
        max_h = float(height_m[mask > 0].max()) if n_pixels > 0 else 0.0

        # Try extruding OBJ
        obj_path = out_dir / f"{name}_mesh.obj"
        num_buildings = 0
        try:
            mesh = mask_and_height_to_3d_mesh(img_np, mask, height_m, output_path=obj_path)
            num_buildings = len(mesh.split()) if hasattr(mesh, "split") else 1
        except ValueError:
            pass

        # Visualization panel
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np)
        axes[0].set_title(f"Real Satellite Input: {name}", fontsize=9)
        axes[0].axis("off")

        axes[1].imshow(prob, cmap="magma")
        axes[1].set_title(f"Predicted Footprint Prob (Buildings: {num_buildings})", fontsize=9)
        axes[1].axis("off")

        height_vis = colorize_height(height_m * mask, max_val=max(30.0, max_h))
        axes[2].imshow(height_vis)
        axes[2].set_title(f"Predicted Heightmap (Max: {max_h:.1f}m, Mean: {mean_h:.1f}m)", fontsize=9)
        axes[2].axis("off")

        plt.suptitle("Qualitative Real-World Evaluation (No Ground Truth Height Available)", fontsize=11, y=0.98)
        plt.tight_layout()
        panel_path = out_dir / f"{name}_qualitative_panel.png"
        plt.savefig(panel_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        summary_list.append({
            "sample": name,
            "buildings_detected": num_buildings,
            "max_height_m": round(max_h, 2),
            "mean_height_m": round(mean_h, 2),
            "panel": str(panel_path.name),
        })
        print(f"  Processed {name}: {num_buildings} buildings, max height {max_h:.1f}m -> {panel_path.name}")

    import json
    with open(out_dir / "real_world_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_list, f, indent=2)
    print(f"\nReal-world qualitative evaluation finished. Saved to: {out_dir}")


if __name__ == "__main__":
    main()
