"""Tests for sliding-window inference logic and edge case shapes."""
from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.multitask_net import MultiTaskBuildingNet
from src.reconstruction.predict import predict_mask_and_height
from src.reconstruction.extrude import mask_and_height_to_3d_mesh

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_model.pt"


@pytest.fixture(scope="module")
def loaded_model():
    assert CHECKPOINT_PATH.exists()
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    encoder_name = ckpt.get("encoder_name", "resnet18") if isinstance(ckpt, dict) else "resnet18"
    state_dict = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model = MultiTaskBuildingNet(encoder_name=encoder_name, encoder_weights=None)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def test_inference_200x200(loaded_model):
    """Image smaller than tile_size (256) should be padded, not stretched, and returned as (200, 200)."""
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    prob, mask, height = predict_mask_and_height(loaded_model, img, torch.device("cpu"), tile_size=256)
    assert prob.shape == (200, 200)
    assert mask.shape == (200, 200)
    assert height.shape == (200, 200)


def test_inference_non_square(loaded_model):
    """Non-square image should preserve aspect ratio and native resolution."""
    img = np.random.randint(0, 255, (320, 480, 3), dtype=np.uint8)
    prob, mask, height = predict_mask_and_height(loaded_model, img, torch.device("cpu"), tile_size=256)
    assert prob.shape == (320, 480)
    assert mask.shape == (320, 480)
    assert height.shape == (320, 480)


def test_inference_1024x1024(loaded_model):
    """Large 1024x1024 image should slide cleanly with 50% overlap."""
    img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    prob, mask, height = predict_mask_and_height(loaded_model, img, torch.device("cpu"), tile_size=256)
    assert prob.shape == (1024, 1024)
    assert mask.shape == (1024, 1024)
    assert height.shape == (1024, 1024)


def test_empty_input_no_buildings(loaded_model, tmp_path):
    """An image with no buildings (all black/empty) should raise ValueError in extrusion."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    prob, mask, height = predict_mask_and_height(loaded_model, img, torch.device("cpu"), mask_thresh=0.99)
    # Ensure mask has 0 buildings
    mask[:] = 0
    height[:] = 0.0

    out_obj = tmp_path / "empty.obj"
    with pytest.raises(ValueError, match="No buildings detected"):
        mask_and_height_to_3d_mesh(img, mask, height, output_path=out_obj)
