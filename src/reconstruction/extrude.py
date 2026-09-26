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

        # --- AUTOMATIC WALL TEXTURING ---
        # Unmerge vertices so roof and walls can have different UVs without conflict
        mesh.unmerge_vertices()
        verts = mesh.vertices
        faces = mesh.faces
        face_normals = mesh.face_normals
        
        # Create Texture Atlas (Top: Roof Crop, Bottom: Procedural Wall with Windows)
        ch, cw = crop.shape[:2]
        wall_tex = np.full((ch, cw, 3), [220, 220, 225], dtype=np.uint8) # Concrete gray
        # Draw window grid
        win_w, win_h = max(cw // 8, 2), max(ch // 8, 2)
        pad_x, pad_y = max(cw // 16, 1), max(ch // 16, 1)
        for wy in range(pad_y, ch, win_h + pad_y):
            for wx in range(pad_x, cw, win_w + pad_x):
                end_y, end_x = min(wy + win_h, ch), min(wx + win_w, cw)
                wall_tex[wy:end_y, wx:end_x] = [40, 50, 60] # Dark glass
                
        atlas_np = np.vstack([crop, wall_tex])
        tex_img = Image.fromarray(atlas_np)
        
        span_x_m = span_x_px * meters_per_pixel
        span_y_m = span_y_px * meters_per_pixel
        
        uvs = np.zeros((len(verts), 2))
        
        for i, face in enumerate(faces):
            normal = face_normals[i]
            v_idx = face
            face_verts = verts[v_idx]
            
            if normal[2] > 0.5:
                # Roof: Map to top half of atlas (V: 0.5 to 1.0)
                u = np.clip(face_verts[:, 0] / max(span_x_m, 1e-6), 0.0, 1.0)
                v = 1.0 - np.clip(face_verts[:, 1] / max(span_y_m, 1e-6), 0.0, 1.0)
                uvs[v_idx, 0] = u
                uvs[v_idx, 1] = (v * 0.5) + 0.5
            else:
                # Wall: Map to bottom half of atlas (V: 0.0 to 0.5)
                # Use dominant axis for U to prevent stretching
                if abs(normal[0]) > abs(normal[1]):
                    u = np.clip(face_verts[:, 1] / max(span_y_m, 1e-6), 0.0, 1.0)
                else:
                    u = np.clip(face_verts[:, 0] / max(span_x_m, 1e-6), 0.0, 1.0)
                # Z goes from 0 to height_val
                v = np.clip(face_verts[:, 2] / max(height_val, 1e-6), 0.0, 1.0)
                uvs[v_idx, 0] = u
                uvs[v_idx, 1] = v * 0.5
                
        mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, image=tex_img)

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
