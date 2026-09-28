"""Integration tests for FastAPI endpoints: /healthz and /api/generate."""
import io
import re
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from src.demo.server import app, get_model

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "test_samples" / "01_tall_skyscraper_towers.png"


@pytest.fixture(scope="module")
def client():
    # Pre-load model so tests run immediately
    get_model()
    return TestClient(app)


def test_healthz(client):
    res = client.get("/healthz")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "torch_version" in data
    assert "smp_version" in data


def test_generate_valid_image(client):
    assert SAMPLE_PATH.exists()
    with open(SAMPLE_PATH, "rb") as f:
        res = client.post(
            "/api/generate",
            files={"file": ("sample.png", f, "image/png")},
            data={"ground_sample_distance": 0.25, "use_tta": False},
        )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["success"] is True
    assert re.match(r"^/outputs/[a-f0-9\-]+/model\.obj$", data["obj_url"])
    assert re.match(r"^/outputs/[a-f0-9\-]+/mask\.png$", data["mask_url"])
    assert re.match(r"^/outputs/[a-f0-9\-]+/height\.png$", data["height_url"])
    assert data["stats"]["buildings_detected"] > 0
    assert data["stats"]["ground_sample_distance"] == 0.25


def test_generate_empty_image_returns_422(client):
    """An image with no buildings must return HTTP 422 with {"error": "No buildings detected"}."""
    # Green vegetation field image produces 0 building detections
    green_field = Image.new("RGB", (256, 256), (50, 100, 50))
    buf = io.BytesIO()
    green_field.save(buf, format="PNG")
    buf.seek(0)

    res = client.post(
        "/api/generate",
        files={"file": ("field.png", buf, "image/png")},
    )
    assert res.status_code == 422
    assert res.json() == {"error": "No buildings detected"}


def test_generate_oversized_file_returns_413(client):
    """Files over 10 MB must return HTTP 413."""
    oversized = b"x" * (10 * 1024 * 1024 + 100)
    res = client.post(
        "/api/generate",
        files={"file": ("huge.png", io.BytesIO(oversized), "image/png")},
    )
    assert res.status_code == 413


def test_generate_oversized_dimensions_returns_400(client):
    """Image side > 2048px must return HTTP 400."""
    big_img = Image.new("RGB", (2049, 100), (128, 128, 128))
    buf = io.BytesIO()
    big_img.save(buf, format="PNG")
    buf.seek(0)

    res = client.post(
        "/api/generate",
        files={"file": ("wide.png", buf, "image/png")},
    )
    assert res.status_code == 400
    assert "exceed maximum allowed side length of 2048 pixels" in res.json()["detail"]
