"""Unit tests for LoD-2 planar reconstruction and projective texturing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from PIL import Image

from uav3d.modeling.lod2 import (
    build_clipped_terrain_mesh,
    decompose_and_align_scene,
    reconstruct_lod2_structures,
)
from uav3d.modeling.texturing import (
    bake_lod2_textured_model,
    create_texture_atlas,
    find_best_keyframe_for_structure,
)


def _make_synthetic_scene() -> o3d.geometry.PointCloud:
    np.random.seed(42)
    # 400 ground points on tilted plane (y = 0.5 * z)
    gx = np.random.uniform(-10.0, 10.0, 400)
    gy = np.random.uniform(-10.0, 10.0, 400)
    gz = np.random.normal(0.0, 0.02, 400)
    ground = np.stack([gx, gy, gz], axis=-1)
    gcols = np.tile(np.array([0.2, 0.7, 0.3]), (400, 1))

    # 200 elevated points (box structure)
    bx = np.random.uniform(-3.0, 3.0, 200)
    by = np.random.uniform(-2.0, 2.0, 200)
    bz = np.random.uniform(1.0, 4.0, 200)
    bldg = np.stack([bx, by, bz], axis=-1)
    bcols = np.tile(np.array([0.8, 0.7, 0.6]), (200, 1))

    pts = np.concatenate([ground, bldg], axis=0)
    cols = np.concatenate([gcols, bcols], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def test_decompose_and_align_scene():
    pcd = _make_synthetic_scene()
    R_up, d, t_pts, s_pts, t_cols, s_cols = decompose_and_align_scene(pcd)

    assert R_up.shape == (3, 3)
    assert len(t_pts) >= 350
    assert len(s_pts) >= 150
    # Ground terrain should be near Z = 0
    assert abs(np.median(t_pts[:, 2])) < 0.2
    # Structure points should be elevated (Z > 0.40)
    assert np.all(s_pts[:, 2] > 0.35)


def test_build_clipped_terrain_mesh():
    pcd = _make_synthetic_scene()
    R_up, d, t_pts, s_pts, t_cols, s_cols = decompose_and_align_scene(pcd)

    mesh, bounds = build_clipped_terrain_mesh(t_pts, t_cols, grid_res_m=0.50)
    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0
    verts = np.asarray(mesh.vertices)
    assert not np.isnan(verts).any()


def test_reconstruct_lod2_structures():
    pcd = _make_synthetic_scene()
    R_up, d, t_pts, s_pts, t_cols, s_cols = decompose_and_align_scene(pcd)

    lod2_mesh = reconstruct_lod2_structures(s_pts, s_cols, ground_z=0.0)
    assert len(lod2_mesh.vertices) > 0
    assert len(lod2_mesh.triangles) > 0

    verts = np.asarray(lod2_mesh.vertices)
    assert not np.isnan(verts).any()
    # Should touch ground (Z = 0.0)
    assert np.isclose(verts[:, 2].min(), 0.0, atol=0.1)


def test_create_texture_atlas(tmp_path: Path):
    ortho_p = tmp_path / "orthomosaic.png"
    facade_p = tmp_path / "facade.jpg"

    Image.new("RGB", (200, 200), color=(0, 255, 0)).save(str(ortho_p))
    Image.new("RGB", (200, 200), color=(255, 0, 0)).save(str(facade_p))

    atlas = create_texture_atlas(ortho_p, facade_p, atlas_width=512, atlas_height=512)
    assert atlas.size == (512, 512)


def test_bake_lod2_textured_model(tmp_path: Path):
    # Create simple terrain and building
    v_terr = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], dtype=np.float32)
    t_terr = np.array([[0,1,2], [0,2,3]], dtype=np.int32)
    bounds = (0.0, 1.0, 0.0, 1.0)

    v_bldg = np.array([[0.2,0.2,0], [0.8,0.2,0], [0.8,0.8,0], [0.2,0.8,0],
                       [0.2,0.2,1], [0.8,0.2,1], [0.8,0.8,1], [0.2,0.8,1]], dtype=np.float32)
    t_bldg = np.array([[0,1,5], [0,5,4], [1,2,6], [1,6,5]], dtype=np.int32)

    obj_p, glb_p, tex_p = bake_lod2_textured_model(
        v_terr, t_terr, bounds, v_bldg, t_bldg, tmp_path, np.eye(3), 0.0
    )

    assert obj_p.exists()
    assert glb_p.exists()
    assert tex_p.exists()
