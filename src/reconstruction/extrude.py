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
        poly = Polygon(cnt_pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        poly = poly.simplify(simplify_tolerance, preserve_topology=True)
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

        x0, y0 = int(max(minx, 0)), int(max(miny, 0))
        x1, y1 = int(min(maxx, w_img)), int(min(maxy, h_img))
        x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
        crop = image_rgb[y0:y1, x0:x1]
        tex_img = Image.fromarray(crop.astype(np.uint8))

        verts = mesh.vertices
        span_x_m = span_x_px * meters_per_pixel
        span_y_m = span_y_px * meters_per_pixel
        u = np.clip(verts[:, 0] / span_x_m, 0.0, 1.0)
        v = 1.0 - np.clip(verts[:, 1] / span_y_m, 0.0, 1.0)
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.stack([u, v], axis=1), image=tex_img
        )

        world_x = minx * meters_per_pixel
        world_y = (h_img - maxy) * meters_per_pixel
        mesh.apply_translation([world_x, world_y, 0.0])

        meshes.append(mesh)
        n_buildings += 1

    if n_buildings == 0:
        raise ValueError(
            "No valid building footprints found in the predicted mask "
            "(try a lower segmentation threshold or check min_area_px)."
        )

    scene_mesh = trimesh.util.concatenate(meshes) if n_buildings > 1 else meshes[0]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene_mesh.export(str(output_path))

    print(f"Exported {n_buildings} building(s) -> {output_path}")
    return str(output_path), n_buildings
