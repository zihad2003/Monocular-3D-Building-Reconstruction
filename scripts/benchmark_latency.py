"""Benchmark CPU and GPU inference & end-to-end reconstruction latency.
Produces empirical, measured numbers without guessing or inventing metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.multitask_net import MultiTaskBuildingNet
from src.reconstruction.predict import predict_mask_and_height
from src.reconstruction.extrude import mask_and_height_to_3d_mesh


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark inference and end-to-end latency")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--output-json", type=str, default="eval_results/latency_benchmark.json", help="Path to output JSON")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--warmup-runs", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--device", type=str, default="cpu", help="Device to benchmark on ('cpu' or 'cuda')")
    return parser.parse_args()


def benchmark_size(model, device, height, width, num_runs=10, warmup_runs=3, sample_img=None):
    if sample_img is not None:
        img = sample_img
        if img.shape[:2] != (height, width):
            import cv2
            img = cv2.resize(img, (width, height))
    else:
        img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # Warmup
    for _ in range(warmup_runs):
        _ = predict_mask_and_height(model, img, device=device)

    # Timed runs
    inference_times = []
    end_to_end_times = []
    tmp_obj = Path("outputs") / "_bench_temp.obj"
    tmp_obj.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(num_runs):
        t0 = time.perf_counter()
        prob, mask, h_map = predict_mask_and_height(model, img, device=device)
        t_infer = (time.perf_counter() - t0) * 1000.0
        inference_times.append(t_infer)

        t1 = time.perf_counter()
        try:
            mask_and_height_to_3d_mesh(img, mask, h_map, output_path=tmp_obj)
        except ValueError:
            pass  # if no buildings in random noise
        t_total = (time.perf_counter() - t0) * 1000.0
        end_to_end_times.append(t_total)

    if tmp_obj.exists():
        tmp_obj.unlink()

    return {
        "image_size": f"{width}x{height}",
        "num_runs": num_runs,
        "inference_ms": {
            "mean": round(float(np.mean(inference_times)), 2),
            "median": round(float(np.median(inference_times)), 2),
            "std": round(float(np.std(inference_times)), 2),
            "min": round(float(np.min(inference_times)), 2),
            "max": round(float(np.max(inference_times)), 2),
        },
        "end_to_end_ms": {
            "mean": round(float(np.mean(end_to_end_times)), 2),
            "median": round(float(np.median(end_to_end_times)), 2),
            "min": round(float(np.min(end_to_end_times)), 2),
            "max": round(float(np.max(end_to_end_times)), 2),
        },
    }


def main():
    args = parse_args()
    torch.set_num_threads(1)  # Match Render production configuration
    device = torch.device(args.device)
    print(f"==> Benchmarking latency on {device} (torch threads: {torch.get_num_threads()})")

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    encoder_name = ckpt.get("encoder_name", "resnet18") if isinstance(ckpt, dict) else "resnet18"
    state_dict = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt

    model = MultiTaskBuildingNet(encoder_name=encoder_name, encoder_weights=None)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    sample_img = None
    sample_p = ROOT / "test_samples" / "01_tall_skyscraper_towers.png"
    if sample_p.exists():
        from PIL import Image
        sample_img = np.array(Image.open(sample_p).convert("RGB"))

    results = {
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "encoder": encoder_name,
        "benchmarks": [],
    }

    test_sizes = [(256, 256), (512, 512), (1024, 1024)]
    for h, w in test_sizes:
        print(f"Benchmarking {w}x{h} px...")
        bench = benchmark_size(
            model,
            device,
            h,
            w,
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs,
            sample_img=sample_img,
        )
        print(f"  {w}x{h}: Inference Mean = {bench['inference_ms']['mean']} ms | End-to-End Mean = {bench['end_to_end_ms']['mean']} ms")
        results["benchmarks"].append(bench)

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nLatency benchmark results saved to: {out_p}")


if __name__ == "__main__":
    main()
