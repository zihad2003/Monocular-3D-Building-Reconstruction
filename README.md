# Geo3D — Monocular 3D Building Reconstruction

Transform single RGB satellite imagery into accurate 3D building models using deep learning.

## Live Demo
Deployed on Render: *(URL will be available after deployment)*

## Key Metrics (40-Epoch Training)
- **89.4%** Validation IoU (Building Footprint)
- **2.98m** Height MAE (nDSM Elevation)
- **6,999+** Training Scenes (SynRS3D + BONAI)

## Architecture
- **Encoder**: ResNet-18 pretrained backbone
- **Decoder**: Dual-head U-Net (Mask sigmoid + Height regression)
- **3D Pipeline**: Watershed segmentation → Polygon regularization → LoD1 extrusion
- **Server**: FastAPI + Three.js WebGL viewer

## Quick Start (Local)
```bash
pip install -r requirements.txt
python src/demo/server.py
# Open http://localhost:8000
```

## Deploy on Render
1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Blueprint
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` and deploys

## Project Structure
```
├── src/
│   ├── models/          # MultiTaskBuildingNet (dual U-Net)
│   ├── reconstruction/  # predict.py + extrude.py (3D mesh)
│   └── demo/            # FastAPI server
├── web_ui/              # Landing page + Dashboard (HTML/CSS/JS)
├── test_samples/        # Curated satellite test images
├── checkpoints/         # Trained model weights
├── render.yaml          # Render.com deployment config
└── build.sh             # Build script
```

## Tech Stack
PyTorch • OpenCV • Three.js • FastAPI • trimesh
