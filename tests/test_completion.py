"""Unit tests for UAV 3D reconstruction completion engine.

Verifies:
- Ground plane and elevated structure decomposition
- Continuous terrain inpainting across occlusion gaps
- Volumetric structure completion (bilateral symmetry + vertical wall extrusion)
- 2D Orthomosaic and DSM elevation raster generation
- End-to-end full 3D map generation pipeline
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pytest

from uav3d.modeling.completion import (
    build_full_3d_map,
    complete_structure_volume,
    decompose_scene,
    generate_2d_orthomosaic_and_dsm,
    inpaint_continuous_terrain,
)


def _make_synthetic_cloud() -> o3d.geometry.PointCloud:
    """Create synthetic point cloud with a ground plane and an elevated building."""
    # 500 ground points on z = 0 with slight jitter
    np.random.seed(42)
    gx = np.random.uniform(-10.0, 10.0, 500)
    gy = np.random.uniform(-10.0, 10.0, 500)
    gz = np.random.normal(0.0, 0.02, 500)
    ground_pts = np.stack([gx, gy, gz], axis=-1)
    ground_cols = np.tile(np.array([0.2, 0.7, 0.3]), (500, 1))

    # 300 structure points (facade rising up to z = 3)
    sx = np.random.uniform(-2.0, 2.0, 300)
    sy = np.random.uniform(-0.2, 0.2, 300)
    sz = np.random.uniform(0.5, 3.0, 300)
    struct_pts = np.stack([sx, sy, sz], axis=-1)
    struct_cols = np.tile(np.array([0.8, 0.7, 0.6]), (300, 1))

    all_pts = np.concatenate([ground_pts, struct_pts], axis=0)
    all_cols = np.concatenate([ground_cols, struct_cols], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_cols)
    return pcd


def test_decompose_scene_planar_and_elevated():
    pcd = _make_synthetic_cloud()
    plane, terrain_pcd, struct_pcd = decompose_scene(pcd, distance_threshold=0.20)

    assert len(plane) == 4
    # Ground normal should be close to [0, 0, 1]
    assert abs(plane[2]) > 0.7
    # Terrain points should capture majority of ground
    assert len(terrain_pcd.points) >= 400
    # Structure points should capture the elevated points
    assert len(struct_pcd.points) >= 150


def test_decompose_scene_small_pcd():
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((10, 3)))
    plane, terrain, struct = decompose_scene(pcd)
    assert len(plane) == 4
    assert len(terrain.points) == 10
    assert len(struct.points) == 0


def test_inpaint_continuous_terrain():
    pcd = _make_synthetic_cloud()
    plane, terrain_pcd, _ = decompose_scene(pcd)

    terrain_mesh = inpaint_continuous_terrain(terrain_pcd, plane, grid_res_m=0.50)
    assert len(terrain_mesh.vertices) > 0
    assert len(terrain_mesh.triangles) > 0

    verts = np.asarray(terrain_mesh.vertices)
    assert not np.isnan(verts).any()
    assert not np.isinf(verts).any()


def test_complete_structure_volume():
    pcd = _make_synthetic_cloud()
    plane, _, struct_pcd = decompose_scene(pcd)

    struct_mesh = complete_structure_volume(struct_pcd, plane)
    assert len(struct_mesh.vertices) > 0
    assert len(struct_mesh.triangles) > 0

    verts = np.asarray(struct_mesh.vertices)
    assert not np.isnan(verts).any()


def test_generate_2d_orthomosaic_and_dsm(tmp_path: Path):
    # Create simple colored cube mesh
    mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.ones((len(mesh.vertices), 3)) * 0.5)

    plane = np.array([0.0, 0.0, 1.0, 0.0])
    ortho_path = tmp_path / "orthomosaic.png"
    dsm_path = tmp_path / "dsm_elevation.png"

    generate_2d_orthomosaic_and_dsm(mesh, plane, ortho_path, dsm_path, resolution_px=128)

    assert ortho_path.exists()
    assert dsm_path.exists()

    ortho = cv2.imread(str(ortho_path))
    assert ortho is not None
    assert ortho.shape == (128, 128, 3)

    dsm = cv2.imread(str(dsm_path))
    assert dsm is not None
    assert dsm.shape == (128, 128, 3)


def test_build_full_3d_map(tmp_path: Path):
    pcd = _make_synthetic_cloud()
    input_ply = tmp_path / "cloud_dense.ply"
    o3d.io.write_point_cloud(str(input_ply), pcd)

    map_3d_ply, map_3d_glb, ortho_png, dsm_png = build_full_3d_map(input_ply, tmp_path)

    assert map_3d_ply.exists()
    assert ortho_png.exists()
    assert dsm_png.exists()

    mesh = o3d.io.read_triangle_mesh(str(map_3d_ply))
    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0


def test_generate_viewer_html(tmp_path: Path):
    from uav3d.modeling.dashboard import generate_viewer_html

    viewer_html = generate_viewer_html(tmp_path)
    assert viewer_html.exists()
    content = viewer_html.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "Three.js" in content or "THREE." in content
    assert "map_3d.glb" in content
    assert "orthomosaic.png" in content
    assert "dsm_elevation.png" in content
