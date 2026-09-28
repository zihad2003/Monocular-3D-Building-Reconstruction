#!/usr/bin/env bash
# Render.com build script for Geo3D
set -o errexit

echo "==> Current Directory: $(pwd)"

# Locate requirements.txt wherever Render is running from
REQ_FILE="requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
    if [ -f "../requirements.txt" ]; then
        REQ_FILE="../requirements.txt"
    else
        FOUND=$(find . -name "requirements.txt" 2>/dev/null | head -n 1 || true)
        if [ -n "$FOUND" ]; then
            REQ_FILE="$FOUND"
        fi
    fi
fi
echo "==> Using requirements file: $REQ_FILE"

echo "==> Installing CPU-optimized PyTorch..."
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing remaining dependencies..."
pip install --no-cache-dir -r "$REQ_FILE"

echo "==> Creating output directories..."
mkdir -p outputs checkpoints

echo "==> Build complete!"
