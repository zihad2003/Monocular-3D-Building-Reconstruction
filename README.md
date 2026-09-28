# Geo3D — Monocular 3D Building Reconstruction

Transform single optical satellite imagery into accurate 3D building models using deep learning (FastAPI + PyTorch + Three.js).

## Live Demo
Deployed on Render: `https://monocular-3d-building-reconstruction.onrender.com`

## Empirical Evaluation Metrics (Scene-Disjoint Evaluation)
Evaluated on **120 validation tiles** across **30 strictly scene-disjoint scenes** via `scripts/evaluate.py`:
- **91.53% (0.9153)** Validation IoU (Building Footprint Segmentation)
- **0.9550** Footprint F1 Score
- **2.69m** Height MAE on Ground-Truth Footprints
- **2.70m** Height MAE on Predicted Footprints
- **5.22m** Height RMSE

> **Dataset & Measurement Reality:**
> - **Height Supervision:** Metric heights are supervised on the **SynRS3D** synthetic dataset, which supplies true normalized Digital Surface Models (nDSM).
> - **BONAI Footprints:** The **BONAI** dataset supplies off-nadir roof/footprint annotations used for visual footprint pretraining; it contains **no metric height labels**.
> - **Real-World Satellite Generalization:** Evaluated **qualitatively** on real satellite tiles (stored in `test_samples/` and `eval_results/real_world_qualitative/`) because real-world satellite imagery lacks ground-truth LiDAR/nDSM height maps.

## Measured Latency Benchmarks (CPU, 1 Thread)
Empirically measured via `scripts/benchmark_latency.py` with `torch.set_num_threads(1)` (matching the Render free-tier environment):
- **256×256 px tile**: **302.6 ms** neural inference / **365.4 ms** end-to-end extrusion
- **512×512 px tile (sliding window)**: **2,971.6 ms** (~2.97s) inference / **3,195.8 ms** (~3.20s) end-to-end
- **1024×1024 px tile (sliding window)**: **18,274.3 ms** (~18.27s) inference / **20,251.0 ms** (~20.25s) end-to-end

## Architecture & Features
- **Encoder**: ResNet-18 pretrained backbone with native-resolution sliding window tiling (50% overlap, 2D Hann window blending).
- **Decoders**: Dual-head U-Net via `segmentation-models-pytorch==0.5.0` (Footprint Logits + Metric Height Map).
- **3D Pipeline**: Watershed instance separation → Douglas-Peucker polygon regularization → LoD1 3D extrusion (Earcut triangulation).
- **Deployment**: Render free-tier compliant, automatic cold-start retry with 5s backoff, per-request UUID workspace cleanup (30-min TTL), upload memory & dimension caps (10 MB, 2048 px).

## Quick Start (Local)
```bash
# 1. Create clean Python 3.11 virtual environment
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run test suite
pytest

# 4. Launch web server
python src/demo/server.py
# Open http://localhost:8000
```

## Running Evaluation & Benchmarks
```bash
# Evaluate checkpoint on scene-disjoint validation set
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt

# Run qualitative evaluation on real-world satellite crops
python scripts/evaluate_real_world.py

# Benchmark CPU latency
python scripts/benchmark_latency.py --num-runs 5
```

## Project Structure
```
├── src/
│   ├── models/          # MultiTaskBuildingNet (shared encoder + dual U-Net decoders)
│   ├── reconstruction/  # predict.py (sliding window) + extrude.py (3D mesh)
│   ├── data/            # dataset.py, preprocess_height.py, preprocess_bonai.py
│   ├── training/        # loop.py, losses.py, metrics.py
│   └── demo/            # server.py (FastAPI application with lifespan)
├── web_ui/              # Dashboard + Three.js viewer (HTML/CSS/JS)
├── test_samples/        # Real-world satellite sanity crops
├── checkpoints/         # Model weights (checkpoints/best_model.pt)
├── eval_results/        # Evaluation metrics JSON, qualitative panels, latency reports
├── tests/               # Pytest suite (model load, inference shapes, API endpoints)
├── render.yaml          # Render.com Blueprint deployment config
└── build.sh             # CPU-optimized build script
```

