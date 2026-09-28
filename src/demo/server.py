"""
FastAPI Server for Geo3D Web Interface.
Connects the web dashboard directly to the trained PyTorch MultiTaskBuildingNet model
and 3D extrusion pipeline.
"""
from __future__ import annotations

import io
import os
import sys
import time
import uuid
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import gc
import psutil

# Single-thread execution to save RAM on free-tier cloud containers
torch.set_num_threads(1)

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

DEVICE = torch.device("cpu")
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_model.pt"
OUTPUTS_DIR = ROOT / "outputs"
WEB_UI_DIR = ROOT / "web_ui"
TEST_SAMPLES_DIR = ROOT / "test_samples"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

_model = None
_model_lock = threading.Lock()


def get_model():
    """Load model with strict=True, reading encoder_name from checkpoint."""
    global _model
    with _model_lock:
        if _model is not None:
            return _model

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

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        encoder_name = ckpt.get("encoder_name", "resnet18") if isinstance(ckpt, dict) else "resnet18"
        state_dict = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt

        model = MultiTaskBuildingNet(encoder_name=encoder_name, encoder_weights=None)
        # Enforce strict=True verification
        model.load_state_dict(state_dict, strict=True)

        del state_dict
        del ckpt
        gc.collect()

        model.to(DEVICE)
        model.eval()
        _model = model

        rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        print(f"Model loaded successfully with strict=True (encoder={encoder_name}). RSS memory: {rss_mb:.2f} MB")
        return _model


def _background_model_loader():
    try:
        get_model()
    except Exception as err:
        print(f"Background model load error: {err}")


def cleanup_old_outputs(max_age_seconds: int = 1800):
    """Delete per-request output directories older than 30 minutes."""
    now = time.time()
    for child in OUTPUTS_DIR.iterdir():
        if child.is_dir():
            try:
                if now - child.stat().st_mtime > max_age_seconds:
                    shutil.rmtree(child, ignore_errors=True)
            except Exception:
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start preloading model in background thread on startup
    loader_thread = threading.Thread(target=_background_model_loader, daemon=True)
    loader_thread.start()
    yield


app = FastAPI(title="Geo3D Neural Reconstruction Server", lifespan=lifespan)

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    import segmentation_models_pytorch as smp
    return {
        "status": "ok",
        "app": "Geo3D",
        "model_loaded": _model is not None,
        "torch_version": torch.__version__,
        "smp_version": smp.__version__,
    }


@app.post("/api/generate")
async def generate_3d(
    file: UploadFile = File(...),
    ground_sample_distance: float = Form(0.3),
    use_tta: bool = Form(False),
):
    """Receives satellite image, runs neural inference, extrudes 3D mesh, returns URLs."""
    # If model is still loading at startup, return 503 so client can backoff and retry
    if _model is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Model is currently loading, server waking up (free tier)",
                "model_loaded": False,
            },
        )

    # 1. Cap upload size (10 MB)
    MAX_FILE_BYTES = 10 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB)")

    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # 2. Cap image dimensions (max side 2048 px)
    if max(pil_img.size) > 2048:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions ({pil_img.size[0]}x{pil_img.size[1]}) exceed maximum allowed side length of 2048 pixels",
        )

    image_rgb = np.array(pil_img)

    # 3. Create unique per-request directory and cleanup old folders
    cleanup_old_outputs(1800)
    req_id = str(uuid.uuid4())
    req_dir = OUTPUTS_DIR / req_id
    req_dir.mkdir(parents=True, exist_ok=True)

    input_path = req_dir / "input.png"
    mask_path = req_dir / "mask.png"
    height_path = req_dir / "height.png"
    obj_path = req_dir / "model.obj"

    pil_img.save(input_path)

    # 4. Neural inference
    from src.reconstruction.predict import predict_mask_and_height
    from src.reconstruction.extrude import mask_and_height_to_3d_mesh

    model = get_model()
    mask_prob, mask_bin, height_m = predict_mask_and_height(
        model,
        image_rgb,
        DEVICE,
        mask_thresh=0.45,
        use_tta=use_tta,
    )

    # If mask is completely empty, reject immediately with 422
    if mask_bin.sum() == 0:
        return JSONResponse(status_code=422, content={"error": "No buildings detected"})

    # Clean mask noise with morphological closing/opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean_mask = cv2.morphologyEx(mask_bin.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    mask_bin = clean_mask > 0

    if mask_bin.sum() == 0:
        return JSONResponse(status_code=422, content={"error": "No buildings detected"})

    # Save mask visualization
    Image.fromarray((mask_bin * 255).astype(np.uint8)).save(mask_path)

    # Save colormapped heightmap visualization (OpenCV plasma, 0 MB overhead)
    max_h_val = max(float(height_m.max()), 10.0)
    norm_h = np.clip((height_m / max_h_val) * 255.0, 0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(norm_h, cv2.COLORMAP_PLASMA)
    cv2.imwrite(str(height_path), colored_bgr)

    # 5. Run 3D Mesh Extrusion
    try:
        _, n_buildings = mask_and_height_to_3d_mesh(
            image_rgb=image_rgb,
            mask_bin=mask_bin,
            height_m=height_m,
            output_path=obj_path,
            min_area_px=60,
            meters_per_pixel=ground_sample_distance,
            separate_touching=True,
            watershed_min_distance=12,
        )
    except ValueError as val_err:
        if "No buildings detected" in str(val_err):
            return JSONResponse(status_code=422, content={"error": "No buildings detected"})
        raise HTTPException(status_code=422, detail=str(val_err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Extrusion failed: {err}")

    if n_buildings == 0:
        return JSONResponse(status_code=422, content={"error": "No buildings detected"})

    mean_h = float(height_m[mask_bin > 0].mean()) if mask_bin.sum() > 0 else 0.0
    max_h = float(height_m[mask_bin > 0].max()) if mask_bin.sum() > 0 else 0.0

    return {
        "success": True,
        "obj_url": f"/outputs/{req_id}/model.obj",
        "mask_url": f"/outputs/{req_id}/mask.png",
        "height_url": f"/outputs/{req_id}/height.png",
        "stats": {
            "buildings_detected": int(n_buildings),
            "mean_height_m": round(mean_h, 1),
            "max_height_m": round(max_h, 1),
            "ground_sample_distance": round(ground_sample_distance, 3),
            "use_tta": use_tta,
        },
        "logs": [
            f"Image loaded: {file.filename} ({image_rgb.shape[1]}x{image_rgb.shape[0]}px)",
            "Running native-resolution sliding window footprint segmentation...",
            f"Estimated heights via DSM regression: max {round(max_h, 1)}m, mean {round(mean_h, 1)}m",
            f"Extruded {n_buildings} building polygons into 3D mesh (GSD={ground_sample_distance}m/px).",
            "Generation complete! Rendering in Three.js interactive viewer.",
        ],
    }


# Mount outputs directory so frontend can fetch OBJ and PNGs
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Mount test_samples directory
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
