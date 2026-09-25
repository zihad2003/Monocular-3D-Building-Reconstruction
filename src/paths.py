"""Resolve project paths for local Windows and Google Colab."""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def is_colab() -> bool:
    return os.path.isdir("/content")


def resolve_dir(name: str, default_under_root: str | None = None) -> Path:
    """
    Prefer /content/<name> on Colab, else <project_root>/<name>.
    Creates the directory if missing.
    """
    if is_colab():
        path = Path("/content") / name
    else:
        path = project_root() / (default_under_root or name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_root() -> Path:
    root = resolve_dir("data")
    for sub in ("images", "masks", "heights"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def outputs_dir() -> Path:
    return resolve_dir("outputs")


def checkpoints_dir() -> Path:
    return resolve_dir("checkpoints")
