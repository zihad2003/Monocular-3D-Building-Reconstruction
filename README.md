# Monocular 3D Building Reconstruction

Single RGB overhead image → footprint mask + height map → LoD1 textured `.obj`.

## Setup

Use **Python 3.11** (3.14 lacks wheels for shapely/pycocotools on Windows):

```bash
cd d:\3d_building
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Train (synthetic smoke-test dataset)

> Synthetic metrics are for pipeline sanity-checks only — not paper results.
> BONAI provides footprints/off-nadir labels, not metric height (see project draft).

```bash
.\.venv\Scripts\python.exe scripts\prepare_synthetic.py --n-train 200 --n-val 40
.\.venv\Scripts\python.exe scripts\train.py --config configs\default.yaml
```

Checkpoint: `checkpoints/best_model.pt`

## Layout

- `src/data` — dataset + synthetic / BONAI preprocess
- `src/models` — dual-decoder multi-task net
- `src/training` — loss, metrics, loop
- `src/reconstruction` — predict + extrude to OBJ
- `scripts/` — CLI entrypoints
- `configs/default.yaml` — hyperparameters

## Note on this machine

Current environment is **CPU-only** PyTorch. Defaults use `resnet18`, batch 4, 8 epochs.
On Colab GPU, switch to `resnet34`, batch 16, ~40 epochs.
