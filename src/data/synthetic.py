"""Synthetic satellite building samples for architecture smoke-tests."""
from __future__ import annotations

import os
import random

import cv2
import numpy as np
from PIL import Image

from src.data.dataset import IMG_SIZE, MAX_HEIGHT_M


def _random_polygon(center, img_size, min_r=25, max_r=70, n_sides_range=(4, 7)):
    cx, cy = center
    n_sides = random.randint(*n_sides_range)
    angles = np.sort(np.random.uniform(0, 2 * np.pi, n_sides))
    radii = np.random.uniform(min_r, max_r, n_sides)
    pts = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1)
    pts = np.clip(pts, 5, img_size - 5)
    return pts.astype(np.int32)


def generate_synthetic_sample(img_size=IMG_SIZE, max_buildings=4):
    base = np.random.randint(60, 140, size=(img_size, img_size, 3), dtype=np.uint8)
    noise = cv2.GaussianBlur(
        np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8),
        (0, 0),
        sigmaX=15,
    )
    image = cv2.addWeighted(base, 0.7, noise, 0.3, 0).astype(np.uint8)

    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    height_map = np.zeros((img_size, img_size), dtype=np.float32)

    n_buildings = random.randint(1, max_buildings)
    sun_dir = np.array([0.6, -0.8])

    placed = 0
    attempts = 0
    while placed < n_buildings and attempts < n_buildings * 5:
        attempts += 1
        cx = random.randint(40, img_size - 40)
        cy = random.randint(40, img_size - 40)
        poly = _random_polygon((cx, cy), img_size)

        building_mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.fillPoly(building_mask, [poly], 1)

        if (mask & building_mask).sum() > 0.15 * building_mask.sum():
            continue

        height_m = float(np.random.uniform(4.0, MAX_HEIGHT_M))

        rooftop_color = np.random.randint(90, 220, size=3)
        ys, xs = np.where(building_mask > 0)
        shade = (xs - cx) * sun_dir[0] + (ys - cy) * sun_dir[1]
        shade = (shade - shade.min()) / (np.ptp(shade) + 1e-6)
        shade = 0.7 + 0.3 * shade
        for c in range(3):
            image[ys, xs, c] = np.clip(rooftop_color[c] * shade, 0, 255).astype(np.uint8)

        shadow_offset = int(0.15 * height_m) + 2
        shadow_poly = poly + np.array([shadow_offset, shadow_offset])
        shadow_layer = image.copy()
        cv2.fillPoly(shadow_layer, [shadow_poly], (20, 20, 20))
        blend_mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.fillPoly(blend_mask, [shadow_poly], 1)
        blend_mask = (blend_mask & (1 - building_mask)).astype(bool)
        image[blend_mask] = cv2.addWeighted(image, 0.45, shadow_layer, 0.55, 0)[blend_mask]

        mask = np.maximum(mask, building_mask)
        height_map[ys, xs] = height_m
        placed += 1

    return image, mask, height_map


def build_synthetic_dataset(n_samples, out_dir, img_size=IMG_SIZE, seed_offset=0):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "heights"), exist_ok=True)

    ids = []
    for i in range(n_samples):
        np.random.seed(1000 + seed_offset + i)
        random.seed(1000 + seed_offset + i)
        image, mask, height_map = generate_synthetic_sample(img_size)
        sample_id = f"synth_{seed_offset + i:05d}"
        Image.fromarray(image).save(os.path.join(out_dir, "images", f"{sample_id}.png"))
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            os.path.join(out_dir, "masks", f"{sample_id}.png")
        )
        np.save(os.path.join(out_dir, "heights", f"{sample_id}.npy"), height_map)
        ids.append(sample_id)
    return ids
