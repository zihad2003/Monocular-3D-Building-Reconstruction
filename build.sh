#!/usr/bin/env bash
# Render.com build script for Geo3D
set -o errexit

echo "==> Installing CPU-optimized PyTorch..."
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing remaining dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "==> Creating output directories..."
mkdir -p outputs
mkdir -p checkpoints

# Copy model checkpoint if it exists at root level
if [ -f "best_model_40epochs.pt" ] && [ ! -f "checkpoints/best_model.pt" ]; then
    echo "==> Copying model weights to checkpoints/"
    cp best_model_40epochs.pt checkpoints/best_model.pt
fi

echo "==> Build complete!"
