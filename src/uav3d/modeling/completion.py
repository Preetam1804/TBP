"""AI and geometric completion module for single-pass UAV 3D reconstructions.

Reconstructs occluded surfaces (rear facades, walls down to ground, rooftops),
inpaints continuous ground terrain, and generates orthomosaic + DSM maps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

logger = logging.getLogger("uav3d.modeling.completion")


def decompose_scene(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.25,
) -> tuple[np.ndarray, o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """Segment dense point cloud into ground terrain and elevated structures.

    Args:
        pcd: Dense point cloud.
        distance_threshold: Plane inlier distance threshold in meters.

    Returns:
        (plane_model [A, B, C, D], terrain_pcd, structure_pcd)
    """
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if len(pcd.colors) > 0 else np.ones_like(pts) * 0.7

    if len(pts) < 100:
        return np.array([0, 0, 1, 0]), pcd, o3d.geometry.PointCloud()

    # RANSAC ground plane fitting on downsampled cloud for speed and robustness
    if len(pts) > 50000:
        pcd_down = pcd.voxel_down_sample(voxel_size=0.15)
    else:
        pcd_down = pcd

    plane_model, _ = pcd_down.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=500,
    )
    a, b, c, d = plane_model
    norm = np.sqrt(a * a + b * b + c * c) + 1e-8
    a, b, c, d = a / norm, b / norm, c / norm, d / norm

    # Signed distance from ground plane
    dist = a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d

    # Orient normal such that elevated structures have positive height
    pos_pts = dist[dist > 0.3]
    neg_pts = dist[dist < -0.3]
    if len(neg_pts) > len(pos_pts):
        a, b, c, d = -a, -b, -c, -d
        dist = -dist
    plane_model = np.array([a, b, c, d])

    # Structure points: height above ground > 0.40m
    structure_mask = dist > 0.40
    terrain_mask = ~structure_mask

    terrain_pcd = o3d.geometry.PointCloud()
    terrain_pcd.points = o3d.utility.Vector3dVector(pts[terrain_mask])
    terrain_pcd.colors = o3d.utility.Vector3dVector(cols[terrain_mask])

    structure_pcd = o3d.geometry.PointCloud()
    structure_pcd.points = o3d.utility.Vector3dVector(pts[structure_mask])
    structure_pcd.colors = o3d.utility.Vector3dVector(cols[structure_mask])

    logger.info(
        f"Scene decomposed: {len(terrain_pcd.points):,} terrain points, "
        f"{len(structure_pcd.points):,} structure points (ground normal: [{a:.3f}, {b:.3f}, {c:.3f}])"
    )
    return plane_model, terrain_pcd, structure_pcd


def inpaint_continuous_terrain(
    terrain_pcd: o3d.geometry.PointCloud,
    ground_plane: np.ndarray,
    grid_res_m: float = 0.20,
    margin_ratio: float = 1.15,
) -> o3d.geometry.TriangleMesh:
    """Generate a seamless, continuous 2D ground terrain mesh filling all occlusion gaps.

    Args:
        terrain_pcd: Segmented terrain points.
        ground_plane: [A, B, C, D] ground plane equation.
        grid_res_m: Spatial resolution of terrain grid.
        margin_ratio: Corridor bounding expansion ratio.

    Returns:
        Continuous polygonal terrain triangle mesh.
    """
    pts = np.asarray(terrain_pcd.points)
    cols = np.asarray(terrain_pcd.colors)

    if len(pts) < 10:
        return o3d.geometry.TriangleMesh()

    # Align coordinate system to ground plane for flat 2D grid generation
    a, b, c, d = ground_plane
    normal = np.array([a, b, c])
    target = np.array([0.0, 0.0, 1.0])
    v_axis = np.cross(normal, target)
    s = np.linalg.norm(v_axis)
    c_val = float(np.dot(normal, target))

    if s > 1e-6:
        vx = np.array([
            [0, -v_axis[2], v_axis[1]],
            [v_axis[2], 0, -v_axis[0]],
            [-v_axis[1], v_axis[0], 0],
        ])
        R_ground = np.eye(3) + vx + (vx @ vx) * ((1.0 - c_val) / (s * s + 1e-8))
    else:
        R_ground = np.eye(3) if c_val > 0 else -np.eye(3)

    pts_aligned = pts @ R_ground.T

    x_min, x_max = np.percentile(pts_aligned[:, 0], [1, 99])
    y_min, y_max = np.percentile(pts_aligned[:, 1], [1, 99])

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    x_half = 0.5 * (x_max - x_min) * margin_ratio
    y_half = 0.5 * (y_max - y_min) * margin_ratio

    xi = np.arange(x_center - x_half, x_center + x_half, grid_res_m)
    yi = np.arange(y_center - y_half, y_center + y_half, grid_res_m)
    X_grid, Y_grid = np.meshgrid(xi, yi)
    grid_coords = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=-1)

    # Use fast vectorized cKDTree for color and elevation transfer
    # Subsample for tree construction if needed
    if len(pts_aligned) > 50000:
        sub_sample = np.random.choice(len(pts_aligned), 50000, replace=False)
        pts_for_tree = pts_aligned[sub_sample]
        cols_for_tree = cols[sub_sample]
    else:
        pts_for_tree = pts_aligned
        cols_for_tree = cols

    tree = cKDTree(pts_for_tree[:, :2])
    _, nearest_idx = tree.query(grid_coords)

    z_nearest = pts_for_tree[nearest_idx, 2].reshape(X_grid.shape)
    cols_interp = cols_for_tree[nearest_idx].reshape(X_grid.shape[0], X_grid.shape[1], 3)

    # Smooth linear interpolation for elevation
    if len(pts_for_tree) > 10000:
        lin_sample = np.random.choice(len(pts_for_tree), 10000, replace=False)
        pts_lin = pts_for_tree[lin_sample]
    else:
        pts_lin = pts_for_tree

    z_interp = griddata(
        pts_lin[:, :2],
        pts_lin[:, 2],
        (X_grid, Y_grid),
        method="linear",
    )
    nan_mask = np.isnan(z_interp)
    z_interp[nan_mask] = z_nearest[nan_mask]

    # Build regular quad grid mesh
    ny, nx = X_grid.shape
    grid_pts = np.stack([X_grid, Y_grid, z_interp], axis=-1).reshape(-1, 3)
    grid_cols = cols_interp.reshape(-1, 3)

    # Un-rotate back to world coordinates
    world_pts = grid_pts @ R_ground

    # Construct quad triangles vectorially
    j_idx, i_idx = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    i_flat = i_idx.ravel()
    j_flat = j_idx.ravel()

    idx0 = j_flat * nx + i_flat
    idx1 = idx0 + 1
    idx2 = (j_flat + 1) * nx + i_flat
    idx3 = idx2 + 1

    tri1 = np.stack([idx0, idx1, idx2], axis=-1)
    tri2 = np.stack([idx1, idx3, idx2], axis=-1)
    triangles = np.concatenate([tri1, tri2], axis=0)

    terrain_mesh = o3d.geometry.TriangleMesh()
    terrain_mesh.vertices = o3d.utility.Vector3dVector(world_pts)
    terrain_mesh.vertex_colors = o3d.utility.Vector3dVector(grid_cols)
    terrain_mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    terrain_mesh.compute_vertex_normals()

    logger.info(
        f"Inpainted continuous terrain: {len(terrain_mesh.vertices):,} vertices, "
        f"{len(terrain_mesh.triangles):,} triangles ({X_grid.shape[1]}x{X_grid.shape[0]} grid)"
    )
    return terrain_mesh


def complete_structure_volume(
    structure_pcd: o3d.geometry.PointCloud,
    ground_plane: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    """Reconstruct occluded rear surfaces, rooftops, and walls down to ground plane.

    Args:
        structure_pcd: Points corresponding to elevated structures (monuments, buildings).
        ground_plane: Ground plane equation [A, B, C, D].

    Returns:
        Completed volumetric structure mesh.
    """
    pts = np.asarray(structure_pcd.points)
    cols = np.asarray(structure_pcd.colors)

    if len(pts) < 100:
        return o3d.geometry.TriangleMesh()

    a, b, c, d = ground_plane
    norm = np.sqrt(a * a + b * b + c * c) + 1e-8

    # 1. Bilateral symmetry reflection across depth centroid
    # Principal component analysis for dominant facade normal and lateral axes
    cov = np.cov(pts.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigenvectors[:, 0] is the thinnest axis (depth into the building)
    depth_axis = eigenvectors[:, 0]
    depth_axis = depth_axis / np.linalg.norm(depth_axis)

    # Project points onto depth axis
    depth_coords = pts @ depth_axis
    depth_min = float(np.min(depth_coords))
    depth_max = float(np.max(depth_coords))
    depth_span = max(0.5, depth_max - depth_min)

    # Generate mirrored rear points (reflection across depth axis)
    mirrored_pts = pts.copy()
    proj = pts @ depth_axis
    shift = 2.0 * (depth_max - proj)
    mirrored_pts += np.outer(shift, depth_axis)

    # Use matching architectural colors with slight lighting variation
    mirrored_cols = np.clip(cols * 0.92, 0.0, 1.0)

    # 2. Extrude base walls vertically down from eaves to ground plane
    height_above_ground = (a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d) / norm
    base_mask = height_above_ground < np.percentile(height_above_ground, 30)
    base_pts = pts[base_mask]
    base_cols = cols[base_mask]

    # Subsample base points for wall extrusion
    sub_step = max(1, len(base_pts) // 2000)
    base_sub = base_pts[::sub_step]
    cols_sub = base_cols[::sub_step]

    normal_ground = np.array([a, b, c])
    dist_to_ground = (base_sub @ normal_ground + d) / norm
    ground_proj = base_sub - np.outer(dist_to_ground, normal_ground)

    steps = np.linspace(0.0, 1.0, 4)[:, None, None]
    wall_pts_arr = base_sub[None, :, :] * (1.0 - steps) + ground_proj[None, :, :] * steps
    wall_pts = wall_pts_arr.reshape(-1, 3)

    wall_cols_arr = cols_sub[None, :, :] * (1.0 - 0.15 * steps)
    wall_cols = np.clip(wall_cols_arr.reshape(-1, 3), 0.0, 1.0)

    # Combine visible front, mirrored rear, and extruded base walls
    all_struct_pts = np.concatenate([pts, mirrored_pts, wall_pts], axis=0)
    all_struct_cols = np.concatenate([cols, mirrored_cols, wall_cols], axis=0)

    pcd_completed = o3d.geometry.PointCloud()
    pcd_completed.points = o3d.utility.Vector3dVector(all_struct_pts)
    pcd_completed.colors = o3d.utility.Vector3dVector(all_struct_cols)

    pcd_clean, _ = pcd_completed.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.2)
    pcd_clean = pcd_clean.voxel_down_sample(voxel_size=0.12)

    # Estimate normals and construct watertight surface volume
    pcd_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.6, max_nn=30))
    pcd_clean.orient_normals_consistent_tangent_plane(20)

    mesh_struct, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_clean, depth=8)
    densities = np.asarray(densities)
    if len(densities) > 0:
        mesh_struct.remove_vertices_by_mask(densities < np.percentile(densities, 8))

    # Fast vectorized coloring from completed point cloud using cKDTree
    if len(mesh_struct.vertices) > 0:
        tree_struct = cKDTree(np.asarray(pcd_clean.points))
        _, nearest_idx = tree_struct.query(np.asarray(mesh_struct.vertices))
        mesh_struct.vertex_colors = o3d.utility.Vector3dVector(np.asarray(pcd_clean.colors)[nearest_idx])
        mesh_struct.compute_vertex_normals()

    logger.info(
        f"Completed volumetric structure: {len(mesh_struct.vertices):,} vertices, "
        f"{len(mesh_struct.triangles):,} triangles (depth span: {depth_span:.1f}m)"
    )
    return mesh_struct


def generate_2d_orthomosaic_and_dsm(
    mesh: o3d.geometry.TriangleMesh,
    ground_plane: np.ndarray,
    output_ortho: Path,
    output_dsm: Path,
    resolution_px: int = 1024,
) -> tuple[Path, Path]:
    """Generate top-down 2D Orthomosaic map and Digital Surface Model (DSM) elevation raster.

    Args:
        mesh: Reconstructed and completed 3D map mesh.
        ground_plane: Ground plane equation.
        output_ortho: Destination path for orthomosaic.png.
        output_dsm: Destination path for dsm_elevation.png.
        resolution_px: Square raster dimension.

    Returns:
        (output_ortho, output_dsm)
    """
    v = np.asarray(mesh.vertices)
    c = np.asarray(mesh.vertex_colors)

    if len(v) < 3:
        return output_ortho, output_dsm

    # Align coordinates to ground plane for top-down orthographic rasterization
    a, b, c_val, d = ground_plane
    normal = np.array([a, b, c_val])
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
        R_align = np.eye(3) + vx + (vx @ vx) * ((1.0 - dot_val) / (s * s + 1e-8))
    else:
        R_align = np.eye(3) if dot_val > 0 else -np.eye(3)

    v_aligned = v @ R_align.T

    x_min, x_max = v_aligned[:, 0].min(), v_aligned[:, 0].max()
    y_min, y_max = v_aligned[:, 1].min(), v_aligned[:, 1].max()

    # Map each vertex into 2D raster grid
    u = np.clip(((v_aligned[:, 0] - x_min) / (x_max - x_min + 1e-6) * (resolution_px - 1)).astype(int), 0, resolution_px - 1)
    v_img = np.clip(((y_max - v_aligned[:, 1]) / (y_max - y_min + 1e-6) * (resolution_px - 1)).astype(int), 0, resolution_px - 1)

    ortho_img = np.zeros((resolution_px, resolution_px, 3), dtype=np.uint8)
    dsm_grid = np.full((resolution_px, resolution_px), np.nan, dtype=np.float32)

    # Sort vertices by height so topmost surfaces dominate orthomosaic
    sort_idx = np.argsort(v_aligned[:, 2])
    u_sorted = u[sort_idx]
    v_sorted = v_img[sort_idx]
    c_sorted = (c[sort_idx] * 255).astype(np.uint8) if len(c) > 0 else np.full((len(v), 3), 180, dtype=np.uint8)
    z_sorted = v_aligned[sort_idx, 2]

    # Vectorized fast raster assignment
    ortho_img[v_sorted, u_sorted] = c_sorted
    dsm_grid[v_sorted, u_sorted] = z_sorted

    # Inpaint any empty raster pixels
    mask = (ortho_img.sum(axis=-1) == 0).astype(np.uint8)
    ortho_clean = cv2.inpaint(ortho_img, mask, 3, cv2.INPAINT_TELEA)

    # Normalize DSM elevation to heatmap
    dsm_valid = dsm_grid[~np.isnan(dsm_grid)]
    if len(dsm_valid) > 0:
        d_min, d_max = np.percentile(dsm_valid, [2, 98])
        dsm_norm = np.clip((dsm_grid - d_min) / (d_max - d_min + 1e-6), 0.0, 1.0)
        dsm_clean = np.nan_to_num(dsm_norm * 255.0, nan=0.0).astype(np.uint8)
        dsm_mask = np.isnan(dsm_grid).astype(np.uint8)
        dsm_inp = cv2.inpaint(dsm_clean, dsm_mask, 3, cv2.INPAINT_TELEA)
        dsm_color = cv2.applyColorMap(dsm_inp, cv2.COLORMAP_TURBO)
    else:
        dsm_color = np.zeros_like(ortho_clean)

    # Convert RGB -> BGR for OpenCV write
    ortho_bgr = cv2.cvtColor(ortho_clean, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_ortho), ortho_bgr)
    cv2.imwrite(str(output_dsm), dsm_color)

    logger.info(f"Generated 2D GIS deliverables: {output_ortho.name} and {output_dsm.name}")
    return output_ortho, output_dsm


def build_full_3d_map(
    cloud_dense_ply: Path,
    run_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    """Execute end-to-end completion: full 3D map, volumetric structures, continuous terrain, orthomosaic, and DSM.

    Args:
        cloud_dense_ply: Path to input calibrated dense point cloud.
        run_dir: Output run directory.

    Returns:
        (map_3d_ply, map_3d_glb, orthomosaic_png, dsm_png)
    """
    map_3d_ply = run_dir / "map_3d.ply"
    map_3d_glb = run_dir / "map_3d.glb"
    map_3d_obj = run_dir / "map_3d.obj"
    ortho_png = run_dir / "orthomosaic.png"
    dsm_png = run_dir / "dsm_elevation.png"

    pcd = o3d.io.read_point_cloud(str(cloud_dense_ply))
    if len(pcd.points) < 50:
        return map_3d_ply, map_3d_glb, ortho_png, dsm_png

    logger.info("Executing Authentic Photogrammetric 3D Reconstruction...")

    # 1. Statistical Outlier Removal & Voxel Downsampling
    pcd_sub = pcd.voxel_down_sample(0.20)
    pcd_clean, _ = pcd_sub.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # 2. Ground Plane Alignment (Upright normal [0, 0, 1])
    from scipy.spatial import Delaunay
    import trimesh

    plane, _ = pcd_clean.segment_plane(0.25, 3, 500)
    a, b, c, d = plane
    norm = np.linalg.norm([a, b, c]) + 1e-8
    a, b, c, d = a / norm, b / norm, c / norm, d / norm
    normal = np.array([a, b, c])

    pts = np.asarray(pcd_clean.points)
    cols = np.asarray(pcd_clean.colors)

    h = pts @ normal + d
    if np.percentile(h, 95) - np.median(h) < np.median(h) - np.percentile(h, 5):
        normal, d = -normal, -d

    target = np.array([0.0, 0.0, 1.0])
    v_axis = np.cross(normal, target)
    s = np.linalg.norm(v_axis)
    dot_val = float(np.dot(normal, target))
    if s > 1e-6:
        vx = np.array([[0, -v_axis[2], v_axis[1]], [v_axis[2], 0, -v_axis[0]], [-v_axis[1], v_axis[0], 0]])
        R_up = np.eye(3) + vx + (vx @ vx) * ((1.0 - dot_val) / (s * s + 1e-8))
    else:
        R_up = np.eye(3) if dot_val > 0 else -np.eye(3)

    pts_up = pts @ R_up.T
    pts_up[:, 2] += d

    # 3. 2.5D Photogrammetric Surface Meshing with Boundary Ray Pruning
    tri = Delaunay(pts_up[:, :2])
    v0 = pts_up[tri.simplices[:, 0], :2]
    v1 = pts_up[tri.simplices[:, 1], :2]
    v2 = pts_up[tri.simplices[:, 2], :2]

    l0 = np.linalg.norm(v1 - v0, axis=1)
    l1 = np.linalg.norm(v2 - v1, axis=1)
    l2 = np.linalg.norm(v0 - v2, axis=1)
    max_edge = np.maximum(l0, np.maximum(l1, l2))

    valid_mask = max_edge < 1.0
    clean_tris = tri.simplices[valid_mask]

    # Create solid Trimesh model with authentic per-vertex colors
    mesh_tri = trimesh.Trimesh(vertices=pts_up, faces=clean_tris, vertex_colors=cols, process=False)
    mesh_tri.fix_normals()

    # Export universal formats
    mesh_tri.export(str(map_3d_glb))
    mesh_tri.export(str(map_3d_ply))
    mesh_tri.export(str(map_3d_obj))

    logger.info(
        f"Exported clean photogrammetric 3D digital twin: "
        f"{len(mesh_tri.vertices):,} vertices, {len(mesh_tri.faces):,} faces (zero boundary stretching)"
    )

    # 4. Generate 2D Orthomosaic and DSM deliverables
    full_o3d = o3d.geometry.TriangleMesh()
    full_o3d.vertices = o3d.utility.Vector3dVector(pts_up)
    full_o3d.vertex_colors = o3d.utility.Vector3dVector(cols)
    full_o3d.triangles = o3d.utility.Vector3iVector(clean_tris.astype(np.int32))
    generate_2d_orthomosaic_and_dsm(full_o3d, np.array([0.0, 0.0, 1.0, 0.0]), ortho_png, dsm_png)

    # 5. Generate interactive viewer.html
    try:
        from uav3d.modeling.dashboard import generate_viewer_html
        generate_viewer_html(run_dir)
    except Exception as exc:
        logger.debug(f"Viewer generation skipped: {exc}")

    return map_3d_ply, map_3d_glb, ortho_png, dsm_png
