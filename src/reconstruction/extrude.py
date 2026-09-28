"""2D footprint + height map -> textured LoD1 OBJ via Shapely/Trimesh."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from shapely.affinity import translate as shapely_translate
from shapely.geometry import Polygon

from src.data.dataset import MAX_HEIGHT_M
from src.reconstruction.instances import split_touching_instances


def mask_and_height_to_3d_mesh(
    image_rgb,
    mask_bin,
    height_m,
    output_path="outputs/building.obj",
    min_area_px=40,
    simplify_tolerance=1.5,
    meters_per_pixel=0.3,
    max_height_m=MAX_HEIGHT_M,
    min_height_m=2.5,
    separate_touching=True,
    watershed_min_distance=8,
):
    h_img, w_img = mask_bin.shape[:2]
    mask_u8 = (np.asarray(mask_bin) > 0).astype(np.uint8)

    if separate_touching:
        instance_labels = split_touching_instances(
            mask_u8, min_area_px=min_area_px, min_distance=watershed_min_distance
        )
        instance_ids = [l for l in np.unique(instance_labels) if l != 0]
        get_instance_mask = lambda lbl: (instance_labels == lbl).astype(np.uint8) * 255
    else:
        n_labels, cc_labels = cv2.connectedComponents(mask_u8)
        instance_ids = list(range(1, n_labels))
        get_instance_mask = lambda lbl: (cc_labels == lbl).astype(np.uint8) * 255

    meshes = []
    n_buildings = 0

    for lbl in instance_ids:
        inst_mask_u8 = get_instance_mask(lbl)
        if inst_mask_u8.sum() // 255 < min_area_px:
            continue

        contours, _ = cv2.findContours(
            inst_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < min_area_px or len(cnt) < 3:
            continue

        cnt_pts = cnt.reshape(-1, 2).astype(np.float64)
        
        # Architectural Regularization (straighten walls & orthogonalize)
        rect = cv2.minAreaRect(cnt)
        rect_box = cv2.boxPoints(rect)
        rect_w, rect_h = rect[1]
        rect_area = rect_w * rect_h
        area = cv2.contourArea(cnt)

        # If approximately rectangular (ratio > 0.65), snap to crisp rotated bounding box
        if area / max(rect_area, 1e-5) > 0.65 and rect_w > 4 and rect_h > 4:
            poly = Polygon(rect_box)
        else:
            # Simplify polygon to snap into straight orthogonal walls
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
            if len(approx) >= 3:
                poly = Polygon(approx.reshape(-1, 2))
            else:
                poly = Polygon(rect_box)

        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < min_area_px or not poly.is_valid:
            continue

        px = inst_mask_u8 > 0
        if px.sum() == 0:
            continue
        height_val = float(np.clip(height_m[px].mean(), min_height_m, max_height_m))

        minx, miny, maxx, maxy = poly.bounds
        span_x_px = max(maxx - minx, 1e-6)
        span_y_px = max(maxy - miny, 1e-6)

        local_poly = shapely_translate(poly, xoff=-minx, yoff=-miny)
        try:
            mesh = trimesh.creation.extrude_polygon(
                local_poly, height=height_val / meters_per_pixel
            )
        except Exception:
            continue

        mesh.apply_scale(meters_per_pixel)

        world_x = minx * meters_per_pixel
        world_y = (h_img - maxy) * meters_per_pixel
        mesh.apply_translation([world_x, world_y, 0.0])

        meshes.append(mesh)
        n_buildings += 1

    if n_buildings == 0:
        raise ValueError("No buildings detected")

    scene_mesh = trimesh.util.concatenate(meshes) if n_buildings > 1 else meshes[0]

    # --- ORIENTATION & CENTERING FOR 3D ENGINE (Three.js / WebGL Y-Up) ---
    # In OpenCV/Shapely, Z is height. In Three.js, Y is Up and X-Z is ground plane.
    # Rotate -90 degrees around X so height points straight UP (+Y):
    rot_x = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    scene_mesh.apply_transform(rot_x)

    # Center model on ground plane at (0, 0) and sit bottom firmly at Y = 0.0
    bounds = scene_mesh.bounds
    center_x = (bounds[0][0] + bounds[1][0]) / 2.0
    center_z = (bounds[0][2] + bounds[1][2]) / 2.0
    min_y = bounds[0][1]
    scene_mesh.apply_translation([-center_x, -min_y, -center_z])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene_mesh.export(str(output_path))

    print(f"Exported {n_buildings} architectural building(s) -> {output_path}")
    return str(output_path), n_buildings
