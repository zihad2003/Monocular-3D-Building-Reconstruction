"""Test loading model checkpoint and running forward pass."""
from pathlib import Path
import torch
import pytest

from src.models.multitask_net import MultiTaskBuildingNet

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_model.pt"


def test_model_load_and_forward():
    assert CHECKPOINT_PATH.exists(), f"Checkpoint missing at {CHECKPOINT_PATH}"
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    encoder_name = ckpt.get("encoder_name", "resnet18") if isinstance(ckpt, dict) else "resnet18"
    state_dict = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt

    model = MultiTaskBuildingNet(encoder_name=encoder_name, encoder_weights=None)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        mask_logits, height_map = model(dummy_input)

    assert mask_logits.shape == (1, 1, 256, 256), f"Expected mask shape (1,1,256,256), got {mask_logits.shape}"
    assert height_map.shape == (1, 1, 256, 256), f"Expected height shape (1,1,256,256), got {height_map.shape}"
