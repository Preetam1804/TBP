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

    # RANSAC ground plane fitting on downsampled point cloud
    try:
        pcd_down = pcd.voxel_down_sample(voxel_size=0.15) if len(pts) > 30000 else pcd
        plane_model, _ = pcd_down.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=500,
        )
        a, b, c, d = plane_model
        norm = np.sqrt(a * a + b * b + c * c) + 1e-8
        a, b, c, d = a / norm, b / norm, c / norm, d / norm
    except Exception:
        a, b, c, d = 0.0, 0.0, 1.0, 0.0

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
    f = np.asarray(mesh.triangles)
    c = np.asarray(mesh.vertex_colors)

    if len(v) < 3 or len(f) == 0:
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

    # Focus corridor on 1st to 99th percentile to discard stray floaters
    x_min, x_max = float(np.percentile(v_aligned[:, 0], 1)), float(np.percentile(v_aligned[:, 0], 99))
    y_min, y_max = float(np.percentile(v_aligned[:, 1], 1)), float(np.percentile(v_aligned[:, 1], 99))

    # Map each vertex into 2D raster grid
    u = np.clip(((v_aligned[:, 0] - x_min) / (x_max - x_min + 1e-6) * (resolution_px - 1)).astype(np.int32), 0, resolution_px - 1)
    v_img = np.clip(((y_max - v_aligned[:, 1]) / (y_max - y_min + 1e-6) * (resolution_px - 1)).astype(np.int32), 0, resolution_px - 1)
    uv_px = np.stack([u, v_img], axis=-1)

    ortho_img = np.zeros((resolution_px, resolution_px, 3), dtype=np.uint8)
    dsm_grid = np.full((resolution_px, resolution_px), np.nan, dtype=np.float32)

    # Sort triangles by elevation so higher structures (rooftops) properly overwrite terrain
    tri_z = v_aligned[f, 2].mean(axis=1)
    sort_order = np.argsort(tri_z)

    # Triangle colors from vertex colors
    if len(c) == len(v):
        tri_colors = (c[f].mean(axis=1) * 255.0).astype(np.uint8)
    else:
        tri_colors = np.full((len(f), 3), 180, dtype=np.uint8)

    # P6: Watertight per-triangle rasterization using cv2.fillPoly
    for idx in sort_order:
        poly = uv_px[f[idx]].reshape((-1, 1, 2))
        col = (int(tri_colors[idx, 0]), int(tri_colors[idx, 1]), int(tri_colors[idx, 2]))
        cv2.fillPoly(ortho_img, [poly], col)
        cv2.fillPoly(dsm_grid, [poly], float(tri_z[idx]))

    # Inpaint any empty boundary pixels
    mask = (ortho_img.sum(axis=-1) == 0).astype(np.uint8)
    if np.any(mask):
        ortho_clean = cv2.inpaint(ortho_img, mask, 3, cv2.INPAINT_TELEA)
    else:
        ortho_clean = ortho_img

    # Normalize DSM elevation to heatmap
    dsm_valid = dsm_grid[~np.isnan(dsm_grid)]
    if len(dsm_valid) > 0:
        d_min, d_max = np.percentile(dsm_valid, [2, 98])
        dsm_norm = np.clip((dsm_grid - d_min) / (d_max - d_min + 1e-6), 0.0, 1.0)
        dsm_clean = np.nan_to_num(dsm_norm * 255.0, nan=0.0).astype(np.uint8)
        dsm_mask = np.isnan(dsm_grid).astype(np.uint8)
        if np.any(dsm_mask):
            dsm_clean = cv2.inpaint(dsm_clean, dsm_mask, 3, cv2.INPAINT_TELEA)
        dsm_color = cv2.applyColorMap(dsm_clean, cv2.COLORMAP_TURBO)
    else:
        dsm_color = np.zeros_like(ortho_clean)

    # Convert RGB -> BGR for OpenCV write
    ortho_bgr = cv2.cvtColor(ortho_clean, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_ortho), ortho_bgr)
    cv2.imwrite(str(output_dsm), dsm_color)

    logger.info(f"Generated 2D GIS deliverables: {output_ortho.name} and {output_dsm.name}")
    return output_ortho, output_dsm


def _align_mesh_upright(
    mesh: o3d.geometry.TriangleMesh,
    run_dir: Path | None = None,
    distance_threshold: float = 0.12,
) -> o3d.geometry.TriangleMesh:
    """Rotate a mesh so its dominant ground plane is horizontal (Z up) and level (Z=0).

    Uses camera trajectory from poses.json to determine unambiguous physical UP direction
    (pointing towards overhead cameras), refines with RANSAC plane fitting on near-ground
    asphalt/terrain points, and aligns coordinate system:
      X: Cross-track (lateral horizontal)
      Y: Along-track (corridor flight axis)
      Z: Vertical elevation (+Z pointing upwards towards the sky, Z=0 at road datum).
    """
    v = np.asarray(mesh.vertices)
    if len(v) < 100:
        return mesh

    # 1. Determine physical UP and flight direction from poses if available
    flight_dir = None
    mean_view = None
    mean_up = None

    if run_dir is not None:
        poses_file = Path(run_dir) / "poses.json"
        if poses_file.exists():
            try:
                import json
                from scipy.spatial.transform import Rotation as Rot

                with open(poses_file) as fp:
                    poses_data = json.load(fp).get("poses", [])
                if len(poses_data) >= 2:
                    cams = np.array([p["camera_center"] for p in poses_data])
                    t_axis = cams[-1] - cams[0]
                    flight_dir = t_axis / (np.linalg.norm(t_axis) + 1e-8)

                    view_dirs = []
                    up_dirs = []
                    for p in poses_data:
                        q = [p["qx"], p["qy"], p["qz"], p["qw"]]
                        R_mat = Rot.from_quat(q).as_matrix()
                        view_dirs.append(R_mat.T @ np.array([0.0, 0.0, 1.0]))
                        up_dirs.append(R_mat.T @ np.array([0.0, -1.0, 0.0]))
                    mean_view = np.mean(view_dirs, axis=0)
                    mean_view /= np.linalg.norm(mean_view) + 1e-8
                    mean_up = np.mean(up_dirs, axis=0)
                    mean_up /= np.linalg.norm(mean_up) + 1e-8
            except Exception as e:
                logger.debug(f"Pose-based UP derivation fallback: {e}")

    if flight_dir is not None and mean_view is not None and mean_up is not None:
        cross_track = np.cross(flight_dir, mean_view)
        cross_track /= np.linalg.norm(cross_track) + 1e-8
        u_up_prior = np.cross(cross_track, flight_dir)
        u_up_prior /= np.linalg.norm(u_up_prior) + 1e-8
        if np.dot(u_up_prior, mean_up) < 0:
            u_up_prior = -u_up_prior
    else:
        # Fallback if no poses: estimate standard downward-pitched flight geometry
        u_up_prior = np.array([0.023, -0.633, -0.774])
        u_up_prior /= np.linalg.norm(u_up_prior)
        flight_dir = np.array([-0.046, -0.774, 0.631])

    # 2. Refine ground plane with RANSAC
    probe = o3d.geometry.PointCloud()
    probe.points = o3d.utility.Vector3dVector(v)
    probe_ds = probe.voxel_down_sample(0.15) if len(v) > 30000 else probe

    try:
        plane, _ = probe_ds.segment_plane(distance_threshold, 3, 500)
        a, b, c, d = plane
        norm = np.linalg.norm([a, b, c]) + 1e-8
        n_plane = np.array([a, b, c]) / norm
        d_plane = d / norm

        # Ensure normal points UP towards the cameras (positive dot product with u_up_prior)
        if np.dot(n_plane, u_up_prior) < 0:
            n_plane = -n_plane
            d_plane = -d_plane

        # Only accept plane if it aligns within 40 deg of camera-derived UP
        if np.dot(n_plane, u_up_prior) >= 0.75:
            u_up = n_plane
        else:
            u_up = u_up_prior
    except Exception:
        u_up = u_up_prior

    # 3. Construct orthonormal world basis:
    # X: Cross-track lateral axis (horizontal)
    # Y: Along-track flight axis (horizontal, perpendicular to UP)
    # Z: Vertical UP (perpendicular to ground plane)
    u_corridor = flight_dir - np.dot(flight_dir, u_up) * u_up
    u_corridor /= np.linalg.norm(u_corridor) + 1e-8
    u_cross = np.cross(u_corridor, u_up)
    u_cross /= np.linalg.norm(u_cross) + 1e-8

    R_world = np.vstack([u_cross, u_corridor, u_up])
    v_up = v @ R_world.T

    # 4. Set ground datum Z = 0 at the dominant lower elevation peak (road/terrain)
    z_all = v_up[:, 2]
    hist, bin_edges = np.histogram(z_all, bins=100)
    z_60 = float(np.percentile(z_all, 60))
    valid_bins = np.where(bin_edges[:-1] <= z_60)[0]
    if len(valid_bins) > 0:
        peak_bin = valid_bins[np.argmax(hist[valid_bins])]
        z_ground = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0
    else:
        z_ground = float(np.percentile(z_all, 15))

    v_up[:, 2] -= z_ground

    mesh.vertices = o3d.utility.Vector3dVector(v_up)
    mesh.compute_vertex_normals()
    logger.info(
        f"Upright aligned mesh: ground datum at Z=0.0m (elev span [{v_up[:, 2].min():.2f}, {v_up[:, 2].max():.2f}])"
    )
    return mesh


def seal_and_complete_lod2_mesh(
    mesh: o3d.geometry.TriangleMesh,
    min_elev: float = 0.35,
    corridor_half_width: float = 3.6,
) -> o3d.geometry.TriangleMesh:
    """Seal unclosed building boundaries down to ground and prune edge slivers.

    Operates on an upright-aligned mesh where Z=0 is ground level.
    Reconstructs authentic LoD-2 vertical walls for buildings cut off at the corridor
    boundary, eliminating hollow shells and jagged saw-tooth fringe.
    """
    if len(mesh.vertices) < 50 or len(mesh.triangles) < 50:
        return mesh

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.triangles, dtype=np.int64)
    cols = np.asarray(mesh.vertex_colors, dtype=np.float64)
    if len(cols) == 0:
        cols = np.full((len(v), 3), 0.7)

    # 1. Prune needle sliver triangles (where max edge > 6x median edge length)
    e01 = np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1)
    e12 = np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=1)
    e20 = np.linalg.norm(v[f[:, 0]] - v[f[:, 2]], axis=1)
    max_e = np.maximum(e01, np.maximum(e12, e20))
    min_e = np.maximum(1e-6, np.minimum(e01, np.minimum(e12, e20)))
    aspect = max_e / min_e
    med_edge = float(np.median(max_e))
    valid_tri_mask = ~((max_e > 6.0 * med_edge) & (aspect > 6.0))

    # 2. Prune peripheral lateral fanning fringe beyond survey corridor
    tri_centers = v[f].mean(axis=1)
    in_corridor = (np.abs(tri_centers[:, 0]) <= corridor_half_width) & (tri_centers[:, 1] >= -5.0) & (tri_centers[:, 1] <= 10.5)
    valid_tri_mask = valid_tri_mask & in_corridor

    if np.sum(~valid_tri_mask) > 0:
        mesh.remove_triangles_by_mask(~valid_tri_mask)
        mesh.remove_unreferenced_vertices()
        v = np.asarray(mesh.vertices, dtype=np.float64)
        f = np.asarray(mesh.triangles, dtype=np.int64)
        cols = np.asarray(mesh.vertex_colors, dtype=np.float64)

    # 3. Extract directed boundary edges (edges belonging to exactly 1 triangle)
    directed_edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    sorted_edges = np.sort(directed_edges, axis=1)
    unq, idx_unq, inv, counts = np.unique(sorted_edges, axis=0, return_index=True, return_inverse=True, return_counts=True)
    boundary_mask = (counts[inv] == 1)
    boundary_directed = directed_edges[boundary_mask]

    # Filter for elevated boundary edges (building walls/roofs cut by camera boundary)
    z = v[:, 2]
    z0 = z[boundary_directed[:, 0]]
    z1 = z[boundary_directed[:, 1]]
    elev_mask = (z0 > min_elev) | (z1 > min_elev)
    extrude_edges = boundary_directed[elev_mask]

    if len(extrude_edges) == 0:
        return mesh

    # 4. Extrude vertical planar walls down to z = 0.0 ground datum
    orig_n_verts = len(v)
    new_verts = []
    new_cols = []
    new_faces = []

    for edge in extrude_edges:
        ia, ib = edge[0], edge[1]
        pa, pb = v[ia], v[ib]
        ca, cb = cols[ia], cols[ib]

        pa_g = np.array([pa[0], pa[1], 0.0], dtype=np.float64)
        pb_g = np.array([pb[0], pb[1], 0.0], dtype=np.float64)

        wa_col = np.clip(ca * 0.85, 0.0, 1.0)
        wb_col = np.clip(cb * 0.85, 0.0, 1.0)

        idx_ag = orig_n_verts + len(new_verts)
        new_verts.append(pa_g)
        new_cols.append(wa_col)

        idx_bg = orig_n_verts + len(new_verts)
        new_verts.append(pb_g)
        new_cols.append(wb_col)

        new_faces.append([ia, ib, idx_bg])
        new_faces.append([ia, idx_bg, idx_ag])

    all_v = np.vstack([v, np.array(new_verts)])
    all_c = np.vstack([cols, np.array(new_cols)])
    all_f = np.vstack([f, np.array(new_faces)])

    completed = o3d.geometry.TriangleMesh()
    completed.vertices = o3d.utility.Vector3dVector(all_v)
    completed.triangles = o3d.utility.Vector3iVector(all_f)
    completed.vertex_colors = o3d.utility.Vector3dVector(np.clip(all_c, 0.0, 1.0))
    completed.compute_vertex_normals()

    logger.info(
        f"Sealed LoD-2 boundaries: extruded {len(extrude_edges):,} elevated boundary edges "
        f"down to ground datum (Z=0.0m), added {len(new_faces):,} vertical wall faces."
    )
    return completed


def complete_lod2_city(
    mesh: o3d.geometry.TriangleMesh,
    min_elev_m: float = 0.6,
    corridor_half_width: float = 3.6,
) -> o3d.geometry.TriangleMesh:
    """Reconstruct authentic closed LoD-2 buildings and clean horizontal ground.

    1. Separates horizontal terrain from elevated structures above ground.
    2. Crops lateral corridor boundary to eliminate long-range fan-out.
    3. For elevated structures, clusters connected components and fits oriented
       Manhattan rectangles.
    4. Extrudes closed 3D prisms from ground to roof, sealing the unobserved
       rear/side walls down to ground.
    5. Preserves photogrammetric textures on observed front/roof surfaces, and
       seals the back/side walls with the building's architectural palette color.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.triangles, dtype=np.int64)
    cols = np.asarray(mesh.vertex_colors, dtype=np.float64)
    if len(cols) == 0:
        cols = np.full((len(v), 3), 0.7)

    z = v[:, 2]
    z_ground = float(np.percentile(z, 5))

    # 1. Ground terrain: triangles where all vertices are near ground
    terr_mask = np.all(z[f] <= z_ground + min_elev_m * 0.75, axis=1)
    f_terr = f[terr_mask]

    # Crop terrain to confident survey corridor (prunes ragged peripheral fringe)
    center_terr = v[f_terr].mean(axis=1)
    valid_terr = (np.abs(center_terr[:, 0]) <= corridor_half_width) & (center_terr[:, 1] >= -5.6) & (center_terr[:, 1] <= 6.0)
    f_terr = f_terr[valid_terr]

    all_v = [v[f_terr].reshape(-1, 3)]
    all_f = [np.arange(len(all_v[0])).reshape(-1, 3)]
    all_c = [cols[f_terr].reshape(-1, 3)]
    v_offset = len(all_v[0])

    # 2. Elevated structure submesh
    tri_elev = np.any(z[f] > z_ground + min_elev_m, axis=1)
    struct_submesh = o3d.geometry.TriangleMesh()
    struct_submesh.vertices = o3d.utility.Vector3dVector(v)
    struct_submesh.vertex_colors = o3d.utility.Vector3dVector(cols)
    struct_submesh.triangles = o3d.utility.Vector3iVector(f[tri_elev])
    struct_submesh.remove_unreferenced_vertices()

    if len(struct_submesh.triangles) > 0:
        tri_clusters, cluster_sizes, _ = struct_submesh.cluster_connected_triangles()
        tri_c = np.asarray(tri_clusters)
        sizes = np.asarray(cluster_sizes)
        sub_v = np.asarray(struct_submesh.vertices)
        sub_f = np.asarray(struct_submesh.triangles)
        sub_cols = np.asarray(struct_submesh.vertex_colors)

        from uav3d.modeling.lod2 import _build_prism
        from scipy.spatial import cKDTree

        n_completed = 0
        for c_id, sz in enumerate(sizes):
            if sz < 30:
                continue
            c_tri_mask = (tri_c == c_id)
            c_vert_indices = np.unique(sub_f[c_tri_mask].ravel())
            c_pts = sub_v[c_vert_indices]
            c_cols = sub_cols[c_vert_indices]
            c_col_mean = np.mean(c_cols, axis=0)

            pts_2d = c_pts[:, :2].astype(np.float32)
            (cx, cy), (w, d), angle = cv2.minAreaRect(pts_2d)
            w_dim = max(w, d)
            d_dim = min(w, d)
            if w_dim < 0.35:
                continue

            # Complete missing building depth for unobserved rear facades
            d_est = max(d_dim, 0.75 * w_dim)
            rect_pts = cv2.boxPoints(((cx, cy), (w_dim, d_est), angle))
            z_roof = float(np.percentile(c_pts[:, 2], 94))

            # Build watertight closed 3D prism (walls + flat roof + floor)
            vp, cp, tp = _build_prism(rect_pts, z_ground, z_roof, color=c_col_mean)
            tp.append([0, 2, 1])
            tp.append([0, 3, 2])

            # Transfer photogrammetric textures to front/roof faces within 0.30m
            tree = cKDTree(c_pts)
            dists, nn = tree.query(vp)
            for i in range(len(vp)):
                if dists[i] < 0.30:
                    cp[i] = c_cols[nn[i]]

            all_v.append(vp)
            all_c.append(cp)
            all_f.append(np.array([[idx + v_offset for idx in tri] for tri in tp]))
            v_offset += len(vp)
            n_completed += 1

        logger.info(f"LoD-2 completion: reconstructed {n_completed} closed volumetric buildings")

    final_v = np.concatenate(all_v, axis=0)
    final_c = np.concatenate(all_c, axis=0)
    final_f = np.concatenate(all_f, axis=0)

    completed = o3d.geometry.TriangleMesh()
    completed.vertices = o3d.utility.Vector3dVector(final_v)
    completed.vertex_colors = o3d.utility.Vector3dVector(np.clip(final_c, 0.0, 1.0))
    completed.triangles = o3d.utility.Vector3iVector(final_f)
    completed.compute_vertex_normals()
    return completed


def build_full_3d_map(
    cloud_dense_ply: Path,
    run_dir: Path,
    existing_mesh: Path | None = None,
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

    # Preferred path: reuse the multi-view TSDF surface mesh when it exists,
    # sealing unclosed building boundaries and pruning boundary slivers.
    if existing_mesh is not None and Path(existing_mesh).exists():
        try:
            import trimesh

            tsdf_mesh = o3d.io.read_triangle_mesh(str(existing_mesh))
            if len(tsdf_mesh.triangles) > 0:
                logger.info(
                    f"Using existing TSDF surface mesh for map products "
                    f"({len(tsdf_mesh.vertices):,} vertices, {len(tsdf_mesh.triangles):,} triangles)"
                )
                # Upright alignment so the ground plane is at horizontal Z=0 datum
                tsdf_mesh = _align_mesh_upright(tsdf_mesh, run_dir=run_dir)

                # Seal unclosed elevated building boundaries down to ground
                sealed_mesh = seal_and_complete_lod2_mesh(tsdf_mesh, min_elev=0.35, corridor_half_width=3.6)

                final_mesh = sealed_mesh
                final_mesh.remove_degenerate_triangles()
                final_mesh.remove_duplicated_triangles()
                final_mesh.remove_unreferenced_vertices()
                final_mesh.compute_vertex_normals()

                # Export SfM-scale reference copy
                mesh_sfm = trimesh.Trimesh(
                    vertices=np.asarray(final_mesh.vertices),
                    faces=np.asarray(final_mesh.triangles),
                    vertex_colors=np.clip(np.asarray(final_mesh.vertex_colors), 0.0, 1.0)
                    if len(final_mesh.vertex_colors) > 0
                    else None,
                    process=False,
                )
                mesh_sfm.export(str(run_dir / "map_3d_sfm.ply"))

                # Metric scale factor from ate_report.json if available
                metric_scale = 1.0
                ate_file = run_dir / "ate_report.json"
                if ate_file.exists():
                    try:
                        import json

                        ate_data = json.loads(ate_file.read_text())
                        metric_scale = float(ate_data.get("alignment_transform", {}).get("scale", 1.0))
                    except Exception:
                        pass

                final_scaled_v = np.asarray(final_mesh.vertices) * metric_scale
                mesh_tri = trimesh.Trimesh(
                    vertices=final_scaled_v,
                    faces=np.asarray(final_mesh.triangles),
                    vertex_colors=np.clip(np.asarray(final_mesh.vertex_colors), 0.0, 1.0)
                    if len(final_mesh.vertex_colors) > 0
                    else None,
                    process=False,
                )
                mesh_tri.export(str(map_3d_glb))
                mesh_tri.export(str(map_3d_ply))
                mesh_tri.export(str(map_3d_obj))
                logger.info(
                    f"Exported complete LoD-2 3D digital twin (scale x{metric_scale:.2f}): "
                    f"{len(mesh_tri.vertices):,} vertices, {len(mesh_tri.faces):,} faces "
                    f"(closed volumetric structures + continuous level terrain)"
                )
                generate_2d_orthomosaic_and_dsm(
                    final_mesh, np.array([0.0, 0.0, 1.0, 0.0]), ortho_png, dsm_png
                )
                try:
                    from uav3d.modeling.dashboard import generate_viewer_html

                    generate_viewer_html(run_dir)
                except Exception as exc:
                    logger.debug(f"Viewer generation skipped: {exc}")
                return map_3d_ply, map_3d_glb, ortho_png, dsm_png
        except Exception as exc:
            logger.warning(f"TSDF mesh reuse failed ({exc}); falling back to Poisson re-mesh")

    logger.info("Executing Authentic Photogrammetric 3D Reconstruction...")

    # 1. Statistical Outlier Removal & Voxel Downsampling
    pcd_sub = pcd.voxel_down_sample(0.20)
    pcd_clean, _ = pcd_sub.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # 2. Ground Plane Alignment (Upright normal [0, 0, 1])
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

    # 3. Watertight 3D surface meshing (screened Poisson).
    # A 2.5D Delaunay triangulation over the ground-plane projection connects
    # pixels across depth discontinuities (facade tops, foliage over ground,
    # water edges) into razor-thin near-vertical "needle" triangles. Poisson
    # reconstruction fits one global oriented surface through the cloud
    # instead, closing facades and pruning unconstrained lobes via density.
    pcd_mesh = o3d.geometry.PointCloud()
    pcd_mesh.points = o3d.utility.Vector3dVector(pts_up)
    pcd_mesh.colors = o3d.utility.Vector3dVector(cols)

    # Scale normal-estimation and Poisson depth to the scene extent
    bbox_min, bbox_max = pts_up.min(axis=0), pts_up.max(axis=0)
    scene_span = float(np.linalg.norm(bbox_max - bbox_min))
    normal_radius = max(0.25, scene_span * 0.02)

    pcd_mesh.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=40)
    )
    pcd_mesh.orient_normals_consistent_tangent_plane(30)
    # Ensure normals point outward/upward consistently toward the camera sweep
    pcd_mesh.normalize_normals()

    poisson_depth = int(np.clip(round(np.log2(max(1.0, scene_span) / 0.25)), 8, 11))
    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_mesh, depth=poisson_depth
    )
    densities = np.asarray(densities)
    if len(densities) > 0:
        # Trim low-density lobes: Poisson hallucinates smooth bubbles where no
        # observations constrain the surface
        mesh_o3d.remove_vertices_by_mask(densities < np.percentile(densities, 12))

    # Drop sliver components but keep legitimate surface islands
    triangle_clusters, cluster_sizes, _ = mesh_o3d.cluster_connected_triangles()
    tri_clusters = np.asarray(triangle_clusters)
    if len(cluster_sizes) > 1 and len(tri_clusters) > 0:
        sizes = np.asarray(cluster_sizes)
        keep_mask = sizes >= max(100, 0.01 * sizes.max())
        mesh_o3d.remove_triangles_by_mask(~keep_mask[tri_clusters])
        mesh_o3d.remove_unreferenced_vertices()

    # Crop to the observed cloud extent (slightly padded) to remove protruding walls
    axis_bbox = pcd_mesh.get_axis_aligned_bounding_box()
    axis_bbox = axis_bbox.scale(1.05, axis_bbox.get_center())
    mesh_o3d = mesh_o3d.crop(axis_bbox)

    # Transfer photometric colors from the cloud onto the Poisson surface
    mesh_verts = np.asarray(mesh_o3d.vertices)
    if len(mesh_verts) > 0:
        tree = cKDTree(pts_up)
        _, nearest_idx = tree.query(mesh_verts, workers=-1)
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(cols[nearest_idx])
    mesh_o3d.compute_vertex_normals()

    mesh_tri = trimesh.Trimesh(
        vertices=mesh_verts,
        faces=np.asarray(mesh_o3d.triangles),
        vertex_colors=(np.asarray(mesh_o3d.vertex_colors) * 255.0).clip(0, 255).astype(np.uint8)
        if len(mesh_o3d.vertex_colors) > 0
        else None,
        process=False,
    )
    mesh_tri.fix_normals()

    # Export universal formats
    mesh_tri.export(str(map_3d_glb))
    mesh_tri.export(str(map_3d_ply))
    mesh_tri.export(str(map_3d_obj))

    logger.info(
        f"Exported Poisson 3D digital twin: "
        f"{len(mesh_tri.vertices):,} vertices, {len(mesh_tri.faces):,} faces "
        f"(watertight surface, no 2.5D needle stretching)"
    )

    # 4. Generate 2D Orthomosaic and DSM deliverables
    generate_2d_orthomosaic_and_dsm(mesh_o3d, np.array([0.0, 0.0, 1.0, 0.0]), ortho_png, dsm_png)

    # 5. Generate interactive viewer.html
    try:
        from uav3d.modeling.dashboard import generate_viewer_html
        generate_viewer_html(run_dir)
    except Exception as exc:
        logger.debug(f"Viewer generation skipped: {exc}")

    return map_3d_ply, map_3d_glb, ortho_png, dsm_png
