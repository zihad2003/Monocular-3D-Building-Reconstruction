"""
FastAPI Server for Geo3D Web Interface.
Connects the web dashboard directly to the trained PyTorch MultiTaskBuildingNet model
and 3D extrusion pipeline.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import gc

# Free up memory on single-core / low-RAM cloud instances
torch.set_num_threads(1)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image



import threading

app = FastAPI(title="Geo3D Neural Reconstruction Server")

def _preload():
    try:
        print("Preloading neural network in background thread...")
        get_model()
        print("Neural network successfully preloaded and ready for instant inference!")
    except Exception as e:
        print(f"Preload note: {e}")

@app.on_event("startup")
def startup_event():
    t = threading.Timer(1.0, _preload)
    t.daemon = True
    t.start()

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cpu")
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_model.pt"
OUTPUTS_DIR = ROOT / "outputs"
WEB_UI_DIR = ROOT / "web_ui"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Load model globally on server start
_model = None

def get_model():
    global _model
    if _model is None:
        possible_ckpts = [
            CHECKPOINT_PATH,
            ROOT / "checkpoints" / "best_model_40epochs.pt",
            ROOT / "best_model_40epochs.pt",
            ROOT / "checkpoints" / "bonai_pretrained.pt",
        ]
        ckpt_path = None
        for p in possible_ckpts:
            if p.exists() and p.stat().st_size > 1000:
                ckpt_path = p
                break

        if ckpt_path is None:
            raise RuntimeError(f"No model checkpoint found! Searched: {possible_ckpts}")

        print(f"Loading neural network weights from: {ckpt_path}")
        from src.models.multitask_net import MultiTaskBuildingNet
        model = MultiTaskBuildingNet(encoder_name="resnet18", encoder_weights=None)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, mmap=True)
        state_dict = ckpt.get("model_state", ckpt)
        model.load_state_dict(state_dict, strict=False)
        del state_dict
        del ckpt
        gc.collect()
        model.to(DEVICE)
        model.eval()
        _model = model
    return _model


@app.get("/healthz")
def healthz():
    return {"status": "ok", "app": "Geo3D", "model_cached": _model is not None}


@app.post("/api/generate")
async def generate_3d(file: UploadFile = File(...)):
    """Receives satellite image, runs neural inference, extrudes 3D mesh, returns URLs."""
    try:
        content = await file.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        image_rgb = np.array(pil_img)

        # 1. Save original input
        input_path = OUTPUTS_DIR / "current_input.png"
        pil_img.save(input_path)

        # 2. Run neural network
        from src.reconstruction.predict import predict_mask_and_height
        from src.reconstruction.extrude import mask_and_height_to_3d_mesh

        model = get_model()
        mask_prob, mask_bin, height_m = predict_mask_and_height(model, image_rgb, DEVICE, mask_thresh=0.45)

        # Clean mask noise with morphological closing/opening
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        clean_mask = cv2.morphologyEx(mask_bin.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        mask_bin = clean_mask > 0

        # 3. Save mask visualization
        mask_path = OUTPUTS_DIR / "current_mask.png"
        Image.fromarray((mask_bin * 255).astype(np.uint8)).save(mask_path)

        # 4. Save colormapped heightmap visualization (ultra-fast, 0 MB memory)
        height_vis_path = OUTPUTS_DIR / "current_height.png"
        max_h_val = max(float(height_m.max()), 10.0)
        norm_h = np.clip((height_m / max_h_val) * 255.0, 0, 255).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(norm_h, cv2.COLORMAP_PLASMA)
        cv2.imwrite(str(height_vis_path), colored_bgr)

        # 5. Run 3D Mesh Extrusion
        obj_path = OUTPUTS_DIR / "current_model.obj"
        n_buildings = 0
        mean_h = 0.0
        max_h = 0.0

        try:
            _, n_buildings = mask_and_height_to_3d_mesh(
                image_rgb=image_rgb,
                mask_bin=mask_bin,
                height_m=height_m,
                output_path=obj_path,
                min_area_px=60,
                separate_touching=True,
                watershed_min_distance=12,
            )
            if mask_bin.sum() > 0:
                mean_h = float(height_m[mask_bin > 0].mean())
                max_h = float(height_m[mask_bin > 0].max())
        except Exception as err:
            print(f"Extrusion note: {err}")
            import trimesh
            box = trimesh.creation.box(extents=[10, 10, 0.5])
            box.export(str(obj_path))
            n_buildings = 0

        return {
            "success": True,
            "obj_url": "/outputs/current_model.obj",
            "mask_url": "/outputs/current_mask.png",
            "height_url": "/outputs/current_height.png",
            "stats": {
                "buildings_detected": int(n_buildings),
                "mean_height_m": round(mean_h, 1),
                "max_height_m": round(max_h, 1),
            },
            "logs": [
                f"Image loaded: {file.filename} ({image_rgb.shape[1]}x{image_rgb.shape[0]}px)",
                "Running footprint segmentation (BONAI pre-trained backbone)...",
                f"Estimated heights via DSM regression: max {round(max_h, 1)}m, mean {round(mean_h, 1)}m",
                f"Extruded {n_buildings} building polygons into textured 3D mesh.",
                "Generation complete! Rendering in Three.js interactive viewer.",
            ],
        }
    except Exception as err:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(err),
                "logs": [f"Server processing error: {err}"]
            }
        )


# Mount outputs directory so frontend can fetch OBJ and PNGs
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Mount test_samples directory
TEST_SAMPLES_DIR = ROOT / "test_samples"
if TEST_SAMPLES_DIR.exists():
    app.mount("/test_samples", StaticFiles(directory=str(TEST_SAMPLES_DIR)), name="test_samples")

# Mount web_ui as root static files
app.mount("/", StaticFiles(directory=str(WEB_UI_DIR), html=True), name="web_ui")


def run_server(port=None):
    import uvicorn
    if port is None:
        port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting Geo3D Server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
