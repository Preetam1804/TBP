"""Projective Camera UV Texture Baking and Multi-Material Atlas Engine.

Projects high-resolution 4K drone keyframes directly onto architectural facades
and drapes 2D aerial orthomosaics over terrain, eliminating blurry per-vertex coloring.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh

logger = logging.getLogger("uav3d.modeling.texturing")


def find_best_keyframe_for_structure(
    structure_center_upright: np.ndarray,
    poses_json_path: Path,
    R_up: np.ndarray,
    d_offset: float,
) -> tuple[str, np.ndarray, np.ndarray, tuple[float, float, float, float], tuple[int, int]]:
    """Find the keyframe that best centers and views the main building facade.

    Returns:
        (best_frame_id, R_cam, t_cam, (fx, fy, cx, cy), (width, height))
    """
    with open(poses_json_path, encoding="utf-8") as f:
        data = json.load(f)

    cam = data["camera"]
    fx, fy, cx, cy = cam["params"]
    w, h = cam["width"], cam["height"]

    # Transform upright center back to raw world frame
    s_world = (structure_center_upright - np.array([0.0, 0.0, d_offset])) @ R_up

    best_frame_id = None
    best_dist = float("inf")
    best_R = np.eye(3)
    best_t = np.zeros(3)

    for p in data.get("poses", []):
        if not p.get("registered", True):
            continue
        q = [p["qx"], p["qy"], p["qz"], p["qw"]]
        t = np.array([p["tx"], p["ty"], p["tz"]])
        R = Rotation.from_quat(q).as_matrix()

        p_cam = R @ s_world + t
        # Must be in front of the camera
        if p_cam[2] > 1.0:
            u = fx * (p_cam[0] / p_cam[2]) + cx
            v = fy * (p_cam[1] / p_cam[2]) + cy
            # Distance from optical center (540, 960)
            dist = np.hypot(u - cx, v - cy)
            if dist < best_dist:
                best_dist = dist
                best_frame_id = p["frame_id"]
                best_R = R
                best_t = t

    if best_frame_id is None and data.get("poses"):
        p0 = data["poses"][0]
        best_frame_id = p0["frame_id"]
        q = [p0["qx"], p0["qy"], p0["qz"], p0["qw"]]
        best_R = Rotation.from_quat(q).as_matrix()
        best_t = np.array([p0["tx"], p0["ty"], p0["tz"]])

    logger.info(f"Selected optimal facade keyframe: {best_frame_id} (principal offset: {best_dist:.1f}px)")
    return best_frame_id, best_R, best_t, (fx, fy, cx, cy), (w, h)


def create_texture_atlas(
    orthomosaic_path: Path,
    facade_image_path: Path,
    atlas_width: int = 2048,
    atlas_height: int = 2048,
) -> tuple[Image.Image, Path]:
    """Create a unified texture atlas: Top half = Orthomosaic, Bottom half = Drone Photo.

    Returns:
        (atlas_pil_image, output_path)
    """
    half_h = atlas_height // 2

    # Top half: Orthomosaic
    if orthomosaic_path.exists():
        ortho_img = Image.open(orthomosaic_path).convert("RGB")
        ortho_resized = ortho_img.resize((atlas_width, half_h), Image.Resampling.LANCZOS)
    else:
        ortho_resized = Image.new("RGB", (atlas_width, half_h), color=(50, 120, 50))

    # Bottom half: High-res Facade keyframe
    if facade_image_path.exists():
        facade_img = Image.open(facade_image_path).convert("RGB")
        facade_resized = facade_img.resize((atlas_width, half_h), Image.Resampling.LANCZOS)
    else:
        facade_resized = Image.new("RGB", (atlas_width, half_h), color=(200, 200, 200))

    atlas = Image.new("RGB", (atlas_width, atlas_height))
    # In OpenGL/glTF UV coordinates:
    # V in [0.0, 0.5] maps to bottom of texture
    # V in [0.5, 1.0] maps to top of texture
    # Image coordinates: top is y=0, bottom is y=atlas_height
    atlas.paste(ortho_resized, (0, 0))            # Top half in pixel space (V in [0.5, 1.0])
    atlas.paste(facade_resized, (0, half_h))       # Bottom half in pixel space (V in [0.0, 0.5])

    return atlas


def bake_lod2_textured_model(
    terrain_vertices: np.ndarray,
    terrain_triangles: np.ndarray,
    terrain_bounds: tuple[float, float, float, float],
    building_vertices: np.ndarray,
    building_triangles: np.ndarray,
    run_dir: Path,
    R_up: np.ndarray,
    d_offset: float,
) -> tuple[Path, Path, Path]:
    """Bake clean UV texture coordinates and export map_3d.obj, map_3d.mtl, and map_3d.glb.

    Returns:
        (map_3d_obj, map_3d_glb, texture_atlas_png)
    """
    map_3d_obj = run_dir / "map_3d.obj"
    map_3d_glb = run_dir / "map_3d.glb"
    map_3d_ply = run_dir / "map_3d.ply"
    texture_path = run_dir / "texture_atlas.png"
    ortho_path = run_dir / "orthomosaic.png"
    poses_path = run_dir / "poses.json"
    images_dir = run_dir / "sfm" / "images"

    x_min, x_max, y_min, y_max = terrain_bounds
    x_span = max(1e-4, x_max - x_min)
    y_span = max(1e-4, y_max - y_min)

    # 1. Terrain UV calculation (mapped to Top Half of Atlas: V in [0.5, 1.0])
    num_terr_verts = len(terrain_vertices)
    u_terr = np.clip((terrain_vertices[:, 0] - x_min) / x_span, 0.0, 1.0)
    # Note: in top-down orthomosaic, y_max is top of image, y_min is bottom
    v_terr_raw = np.clip((terrain_vertices[:, 1] - y_min) / y_span, 0.0, 1.0)
    v_terr = 0.5 + 0.5 * v_terr_raw
    uv_terrain = np.stack([u_terr, v_terr], axis=-1)

    # 2. Building Facade UV calculation (mapped to Bottom Half of Atlas: V in [0.0, 0.5])
    num_bldg_verts = len(building_vertices)
    uv_building = np.zeros((num_bldg_verts, 2), dtype=np.float32)

    if num_bldg_verts > 0 and poses_path.exists() and images_dir.exists():
        struct_center = np.mean(building_vertices, axis=0)
        best_id, R_cam, t_cam, (fx, fy, cx, cy), (w, h) = find_best_keyframe_for_structure(
            struct_center, poses_path, R_up, d_offset
        )
        facade_img_path = images_dir / f"{best_id}.jpg"
        if not facade_img_path.exists():
            # Try finding any image matching frame id
            candidates = list(images_dir.glob(f"*{best_id}*.jpg"))
            if candidates:
                facade_img_path = candidates[0]
            else:
                first_img = next(images_dir.glob("*.jpg"), None)
                facade_img_path = first_img or facade_img_path

        # Create combined texture atlas
        atlas_img = create_texture_atlas(ortho_path, facade_img_path)
        atlas_img.save(str(texture_path), format="PNG")

        # Project 3D building vertices into best camera frame
        bldg_world = (building_vertices - np.array([0.0, 0.0, d_offset])) @ R_up
        p_cam = (bldg_world @ R_cam.T) + t_cam

        # Safe perspective projection
        z_safe = np.maximum(p_cam[:, 2], 0.1)
        u_px = fx * (p_cam[:, 0] / z_safe) + cx
        v_px = fy * (p_cam[:, 1] / z_safe) + cy

        u_norm = np.clip(u_px / float(w), 0.0, 1.0)
        v_norm = np.clip(1.0 - (v_px / float(h)), 0.0, 1.0)

        # Map to bottom half of atlas (V in [0.0, 0.5])
        uv_building[:, 0] = u_norm
        uv_building[:, 1] = v_norm * 0.5
    else:
        # Fallback neutral atlas
        atlas_img = Image.new("RGB", (1024, 1024), color=(180, 180, 180))
        atlas_img.save(str(texture_path), format="PNG")
        uv_building[:, 0] = 0.5
        uv_building[:, 1] = 0.25

    # 3. Combine vertices, UVs, and triangles
    all_vertices = np.concatenate([terrain_vertices, building_vertices], axis=0)
    all_uvs = np.concatenate([uv_terrain, uv_building], axis=0)

    # Offset building triangle indices
    bldg_triangles_offset = building_triangles + num_terr_verts
    all_triangles = np.concatenate([terrain_triangles, bldg_triangles_offset], axis=0)

    # 4. Construct Trimesh with TextureVisuals
    visual = trimesh.visual.TextureVisuals(uv=all_uvs, image=atlas_img)
    full_mesh = trimesh.Trimesh(
        vertices=all_vertices,
        faces=all_triangles,
        visual=visual,
        process=False,
    )

    # 5. Export universal formats
    full_mesh.export(str(map_3d_obj))
    full_mesh.export(str(map_3d_glb))
    logger.info(f"Exported textured 3D deliverables: {map_3d_obj.name} and {map_3d_glb.name}")

    # Also export textured PLY with vertex colors sampled from atlas for Open3D / CloudCompare
    try:
        colors_sampled = visual.to_color().vertex_colors
        mesh_ply = trimesh.Trimesh(
            vertices=all_vertices,
            faces=all_triangles,
            vertex_colors=colors_sampled,
            process=False,
        )
        mesh_ply.export(str(map_3d_ply))
        logger.info(f"Exported textured {map_3d_ply.name}")
    except Exception as exc:
        logger.debug(f"PLY export fallback: {exc}")

    return map_3d_obj, map_3d_glb, texture_path
