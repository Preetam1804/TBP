"""LoD-2 Planar Structure Reconstruction & Clipped Topographic Terrain Module.

Reconstructs architectural structures with true vertical 90° planar walls,
crisp multi-tier setbacks, and corner minaret pillars, and generates clipped
topographic terrain without stretched boundary artifacts.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

logger = logging.getLogger("uav3d.modeling.lod2")


def decompose_and_align_scene(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.25,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Segment dense cloud into terrain and elevated structures, oriented upright.

    Returns:
        (R_up, d_offset, terrain_pts, struct_pts, terrain_cols, struct_cols)
        where all points are in upright coordinates (ground at Z=0, normal=[0,0,1]).
    """
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if len(pcd.colors) > 0 else np.ones_like(pts) * 0.7

    if len(pts) < 100:
        return np.eye(3), 0.0, pts, np.empty((0, 3)), cols, np.empty((0, 3))

    pcd_sub = pcd.voxel_down_sample(0.15) if len(pts) > 50000 else pcd
    plane_model, _ = pcd_sub.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=500,
    )
    a, b, c, d = plane_model
    norm = np.sqrt(a * a + b * b + c * c) + 1e-8
    a, b, c, d = a / norm, b / norm, c / norm, d / norm

    normal = np.array([a, b, c])
    target = np.array([0.0, 0.0, 1.0])
    v_axis = np.cross(normal, target)
    s = np.linalg.norm(v_axis)
    dot_val = float(np.dot(normal, target))

    if s > 1e-6:
        vx = np.array([
            [0, -v_axis[2], v_axis[1]],
            [v_axis[2], 0, -v_axis[0]],
            [-v_axis[1], v_axis[0], 0],
        ])
        R_up = np.eye(3) + vx + (vx @ vx) * ((1.0 - dot_val) / (s * s + 1e-8))
    else:
        R_up = np.eye(3) if dot_val > 0 else -np.eye(3)

    pts_aligned = pts @ R_up.T
    pts_aligned[:, 2] += d  # Ground is at Z = 0

    pos_mask = pts_aligned[:, 2] > 0.35
    neg_mask = pts_aligned[:, 2] < -0.35
    if np.sum(neg_mask) > np.sum(pos_mask):
        R_flip = np.diag([1.0, -1.0, -1.0])
        R_up = R_flip @ R_up
        pts_aligned = pts @ R_up.T
        pts_aligned[:, 2] -= d

    struct_mask = pts_aligned[:, 2] > 0.40
    terr_mask = ~struct_mask

    terrain_pts = pts_aligned[terr_mask]
    terrain_cols = cols[terr_mask]
    struct_pts = pts_aligned[struct_mask]
    struct_cols = cols[struct_mask]

    logger.info(
        f"Scene aligned upright: {len(terrain_pts):,} terrain points, "
        f"{len(struct_pts):,} structure points (elevation span: {pts_aligned[:, 2].min():.1f}m to {pts_aligned[:, 2].max():.1f}m)"
    )
    return R_up, d, terrain_pts, struct_pts, terrain_cols, struct_cols


def build_clipped_terrain_mesh(
    terrain_pts: np.ndarray,
    terrain_cols: np.ndarray,
    grid_res_m: float = 0.25,
) -> tuple[o3d.geometry.TriangleMesh, tuple[float, float, float, float]]:
    """Generate clean topographic terrain mesh strictly clipped to observed boundary.

    Eliminates radial outward stretching by pruning all triangles outside the survey convex hull.
    Returns:
        (terrain_mesh, (x_min, x_max, y_min, y_max))
    """
    if len(terrain_pts) < 10:
        return o3d.geometry.TriangleMesh(), (0.0, 1.0, 0.0, 1.0)

    x_min = float(np.percentile(terrain_pts[:, 0], 1))
    x_max = float(np.percentile(terrain_pts[:, 0], 99))
    y_min = float(np.percentile(terrain_pts[:, 1], 1))
    y_max = float(np.percentile(terrain_pts[:, 1], 99))

    xi = np.arange(x_min, x_max, grid_res_m)
    yi = np.arange(y_min, y_max, grid_res_m)
    X_grid, Y_grid = np.meshgrid(xi, yi)
    grid_coords = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=-1)

    pts_2d = terrain_pts[:, :2].astype(np.float32)
    hull_idx = cv2.convexHull(pts_2d, returnPoints=False)
    hull_pts = pts_2d[hull_idx.ravel()]

    sub_n = min(len(terrain_pts), 40000)
    sub_idx = np.random.choice(len(terrain_pts), sub_n, replace=False)
    tree = cKDTree(terrain_pts[sub_idx, :2])
    _, nearest_idx = tree.query(grid_coords)

    z_nearest = terrain_pts[sub_idx[nearest_idx], 2].reshape(X_grid.shape)
    cols_interp = terrain_cols[sub_idx[nearest_idx]].reshape(X_grid.shape[0], X_grid.shape[1], 3)

    lin_n = min(len(terrain_pts), 8000)
    lin_idx = np.random.choice(len(terrain_pts), lin_n, replace=False)
    z_interp = griddata(
        terrain_pts[lin_idx, :2],
        terrain_pts[lin_idx, 2],
        (X_grid, Y_grid),
        method="linear",
    )
    nan_mask = np.isnan(z_interp)
    z_interp[nan_mask] = z_nearest[nan_mask]

    # Fast raster polygon clipping via cv2.fillPoly
    c_idx = np.clip((hull_pts[:, 0] - x_min) / grid_res_m, 0, X_grid.shape[1] - 1)
    r_idx = np.clip((hull_pts[:, 1] - y_min) / grid_res_m, 0, X_grid.shape[0] - 1)
    hull_grid = np.stack([c_idx, r_idx], axis=-1).astype(np.int32)
    mask_img = np.zeros(X_grid.shape, dtype=np.uint8)
    cv2.fillPoly(mask_img, [hull_grid], 1)
    inside_mask = mask_img.astype(bool)

    ny, nx = X_grid.shape
    grid_pts = np.stack([X_grid, Y_grid, z_interp], axis=-1).reshape(-1, 3)
    grid_cols = cols_interp.reshape(-1, 3)

    triangles = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            if inside_mask[j, i] and inside_mask[j, i + 1] and inside_mask[j + 1, i] and inside_mask[j + 1, i + 1]:
                idx0 = j * nx + i
                idx1 = idx0 + 1
                idx2 = (j + 1) * nx + i
                idx3 = idx2 + 1
                triangles.append([idx0, idx1, idx2])
                triangles.append([idx1, idx3, idx2])

    terrain_mesh = o3d.geometry.TriangleMesh()
    if len(triangles) > 0:
        terrain_mesh.vertices = o3d.utility.Vector3dVector(grid_pts)
        terrain_mesh.vertex_colors = o3d.utility.Vector3dVector(grid_cols)
        terrain_mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))
        terrain_mesh.remove_unreferenced_vertices()
        terrain_mesh.compute_vertex_normals()

    logger.info(
        f"Built clipped terrain: {len(terrain_mesh.vertices):,} vertices, "
        f"{len(terrain_mesh.triangles):,} triangles (strictly bounded to survey hull)"
    )
    return terrain_mesh, (x_min, x_max, y_min, y_max)


def _build_prism(
    base_pts_2d: np.ndarray,
    z_bottom: float,
    z_top: float,
    z_peak: float | None = None,
    color: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Generate vertices, colors, and triangles for a vertical polygonal prism."""
    n_poly = len(base_pts_2d)
    v_bottom = np.column_stack([base_pts_2d, np.full(n_poly, z_bottom)])
    v_top = np.column_stack([base_pts_2d, np.full(n_poly, z_top)])

    if color is None:
        color = np.array([0.85, 0.85, 0.82])

    if z_peak is not None and z_peak > z_top:
        center_xy = np.mean(base_pts_2d, axis=0)
        v_peak = np.array([[center_xy[0], center_xy[1], z_peak]])
        verts = np.concatenate([v_bottom, v_top, v_peak], axis=0)
        colors = np.tile(color, (len(verts), 1))
        triangles = []
        # Vertical walls
        for i in range(n_poly):
            nxt = (i + 1) % n_poly
            triangles.append([i, nxt, i + n_poly])
            triangles.append([nxt, nxt + n_poly, i + n_poly])
        # Roof cap to peak
        peak_idx = 2 * n_poly
        for i in range(n_poly):
            nxt = (i + 1) % n_poly
            triangles.append([i + n_poly, nxt + n_poly, peak_idx])
    else:
        verts = np.concatenate([v_bottom, v_top], axis=0)
        colors = np.tile(color, (len(verts), 1))
        triangles = []
        for i in range(n_poly):
            nxt = (i + 1) % n_poly
            triangles.append([i, nxt, i + n_poly])
            triangles.append([nxt, nxt + n_poly, i + n_poly])
        # Flat top cap
        if n_poly == 4:
            triangles.append([n_poly + 0, n_poly + 1, n_poly + 2])
            triangles.append([n_poly + 0, n_poly + 2, n_poly + 3])

    return verts, colors, triangles


def reconstruct_lod2_structures(
    struct_pts: np.ndarray,
    struct_cols: np.ndarray,
    ground_z: float = 0.0,
) -> o3d.geometry.TriangleMesh:
    """Reconstruct building structures with multi-tier architectural LoD-2 geometry.

    Generates:
    - Tier 1: Plinth / podium terrace with vertical walls from ground level
    - Tier 2: Main architectural block with sharp 90° facade
    - Tier 3: Roof pavilion / domes with apex peak
    - 4 Corner Minaret Pillars rising at the terrace corners
    """
    if len(struct_pts) < 100:
        return o3d.geometry.TriangleMesh()

    pcd_struct = o3d.geometry.PointCloud()
    pcd_struct.points = o3d.utility.Vector3dVector(struct_pts)
    pcd_struct.colors = o3d.utility.Vector3dVector(struct_cols)

    pcd_sub = pcd_struct.voxel_down_sample(0.25)
    pts_sub = np.asarray(pcd_sub.points)
    cols_sub = np.asarray(pcd_sub.colors)

    labels = np.array(pcd_sub.cluster_dbscan(eps=2.5, min_points=50))
    unique_labels = [l for l in np.unique(labels) if l >= 0]

    if not unique_labels:
        return o3d.geometry.TriangleMesh()

    cluster_counts = [(l, np.sum(labels == l)) for l in unique_labels]
    cluster_counts.sort(key=lambda x: -x[1])

    all_verts = []
    all_cols = []
    all_tris = []
    v_offset = 0

    for cluster_id, count in cluster_counts:
        if count < 150:
            continue

        c_pts = pts_sub[labels == cluster_id]
        c_cols = cols_sub[labels == cluster_id]

        pts_2d = c_pts[:, :2].astype(np.float32)
        rect = cv2.minAreaRect(pts_2d)
        box_pts = cv2.boxPoints(rect)  # (4, 2)
        center = np.mean(box_pts, axis=0)

        z_span = c_pts[:, 2].max() - c_pts[:, 2].min()
        mean_col = np.mean(c_cols, axis=0)

        if z_span > 4.0 and count > 1000:
            # Multi-tier architectural monument / large structure
            z_plinth = float(np.percentile(c_pts[:, 2], 25))
            z_roof = float(np.percentile(c_pts[:, 2], 80))
            z_peak = float(np.percentile(c_pts[:, 2], 98))

            # Tier 1: Plinth / Base Podium (Ground to z_plinth)
            v1, c1, t1 = _build_prism(box_pts, ground_z, z_plinth, color=mean_col * 0.95)
            all_verts.append(v1)
            all_cols.append(c1)
            all_tris.extend([[idx + v_offset for idx in tri] for tri in t1])
            v_offset += len(v1)

            # Tier 2: Main Facade Block (Setback 15%)
            box_t2 = box_pts * 0.85 + center * 0.15
            v2, c2, t2 = _build_prism(box_t2, z_plinth, z_roof, color=mean_col * 1.05)
            all_verts.append(v2)
            all_cols.append(c2)
            all_tris.extend([[idx + v_offset for idx in tri] for tri in t2])
            v_offset += len(v2)

            # Tier 3: Roof Pavilion & Domes (Setback 35%)
            box_t3 = box_pts * 0.65 + center * 0.35
            v3, c3, t3 = _build_prism(box_t3, z_roof, z_roof + 1.2, z_peak=z_peak, color=mean_col * 1.1)
            all_verts.append(v3)
            all_cols.append(c3)
            all_tris.extend([[idx + v_offset for idx in tri] for tri in t3])
            v_offset += len(v3)

            # 4 Corner Minaret Pillars at the corners of the plinth
            pillar_r = 1.8  # Radius in meters
            for i in range(4):
                c_corner = box_pts[i] * 0.90 + center * 0.10
                # 4-sided vertical pillar
                p_box = np.array([
                    [c_corner[0] - pillar_r, c_corner[1] - pillar_r],
                    [c_corner[0] + pillar_r, c_corner[1] - pillar_r],
                    [c_corner[0] + pillar_r, c_corner[1] + pillar_r],
                    [c_corner[0] - pillar_r, c_corner[1] + pillar_r],
                ])
                vp, cp, tp = _build_prism(p_box, z_plinth, z_roof + 1.5, z_peak=z_peak, color=mean_col * 1.02)
                all_verts.append(vp)
                all_cols.append(cp)
                all_tris.extend([[idx + v_offset for idx in tri] for tri in tp])
                v_offset += len(vp)
        else:
            # Single-tier building (e.g. house, warehouse, ancillary building)
            z_roof = float(np.percentile(c_pts[:, 2], 85))
            z_peak = float(np.percentile(c_pts[:, 2], 98))
            v_bldg, c_bldg, t_bldg = _build_prism(box_pts, ground_z, z_roof, z_peak=z_peak, color=mean_col)
            all_verts.append(v_bldg)
            all_cols.append(c_bldg)
            all_tris.extend([[idx + v_offset for idx in tri] for tri in t_bldg])
            v_offset += len(v_bldg)

    if not all_verts:
        return o3d.geometry.TriangleMesh()

    mesh_lod2 = o3d.geometry.TriangleMesh()
    mesh_lod2.vertices = o3d.utility.Vector3dVector(np.concatenate(all_verts, axis=0))
    mesh_lod2.vertex_colors = o3d.utility.Vector3dVector(np.clip(np.concatenate(all_cols, axis=0), 0.0, 1.0))
    mesh_lod2.triangles = o3d.utility.Vector3iVector(np.array(all_tris, dtype=np.int32))
    mesh_lod2.compute_vertex_normals()

    logger.info(
        f"Reconstructed multi-tier LoD-2 architectural structure: "
        f"{len(mesh_lod2.vertices)} vertices, {len(mesh_lod2.triangles)} triangles with sharp 90° planar walls"
    )
    return mesh_lod2
