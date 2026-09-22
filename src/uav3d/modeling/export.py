"""3D model export: colored point cloud, COLMAP model directory, camera trajectory."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uav3d.models.reconstruction import ReconstructionSummary
from uav3d.sfm.colmap_compat import get_pose_components

logger = logging.getLogger("uav3d.modeling.export")


@dataclass
class ModelExportResult:
    """Paths and counts produced by export_model()."""

    model_ply: Path
    model_dir: Path
    trajectory_ply: Path
    summary_path: Path
    num_points: int
    mesh_ply: Path | None = None
    mesh_decimated_ply: Path | None = None
    cloud_dense_ply: Path | None = None
    map_3d_ply: Path | None = None
    map_3d_glb: Path | None = None
    orthomosaic_png: Path | None = None
    dsm_png: Path | None = None


def export_model(
    reconstruction: Any,
    run_dir: str | Path,
    dataset_label: str,
    num_source_frames: int,
    num_keyframes: int,
    num_models: int,
    mean_reprojection_error_px: float | None,
    scale_note: str = "monocular SfM: metric scale is unknown until aligned to a metric reference (GPS/GT)",
) -> ModelExportResult:
    """Export the reconstruction as viewer-ready 3D model artifacts.

    Writes into the run directory:
      - model.ply            colored 3D point cloud (the 3D model; opens in
                             CloudCompare / MeshLab / Blender)
      - mesh.ply             reconstructed surface triangle mesh (solid 3D surface)
      - mesh_decimated.ply   optimized surface mesh for lightweight viewing
      - cloud_dense.ply      dense colored 3D point cloud from AI depth
      - map_3d.ply           full completed 3D map (terrain + volumetric structures)
      - map_3d.glb           WebGL/3D viewer asset (Three.js / Cesium)
      - orthomosaic.png      top-down 2D orthomosaic aerial map
      - dsm_elevation.png    Digital Surface Model (elevation heatmap)
      - model/               COLMAP sparse model in text format
                             (cameras.txt, images.txt, points3D.txt)
      - trajectory.ply       camera-center flight path (vertices + edges)
      - reconstruction_summary.json  versioned machine-readable manifest

    Args:
        reconstruction: Selected pycolmap.Reconstruction (largest model).
        run_dir: Run output directory (data/outputs/run_<timestamp>).
        dataset_label: Provenance label recorded in the summary.
        num_source_frames: Frames available in the source before pre-striding.
        num_keyframes: Keyframes fed to SfM.
        num_models: Number of reconstructions produced.
        mean_reprojection_error_px: Mean reprojection error of the main model.
        scale_note: Scale interpretation text for the summary.
    """
    run_path = Path(run_dir).resolve()
    run_path.mkdir(parents=True, exist_ok=True)

    # --- 1. Colored point cloud (the 3D model) ---
    model_ply = run_path / "model.ply"
    reconstruction.export_PLY(str(model_ply))
    num_points = int(reconstruction.num_points3D())
    logger.info(f"Exported colored point cloud ({num_points:,} points) to {model_ply}")

    # Clean outlier rays from model.ply if open3d is available
    _filter_point_cloud_outliers(model_ply)

    # --- 2. COLMAP sparse model directory (text format, viewer-friendly) ---
    model_dir = run_path / "model"
    if model_dir.exists():
        import shutil

        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write_text(str(model_dir))
    logger.info(f"Exported COLMAP sparse model (text) to {model_dir}")

    # --- 3. Camera trajectory as PLY (vertices + edges, time-ordered) ---
    trajectory_ply = run_path / "trajectory.ply"
    _write_trajectory_ply(reconstruction, trajectory_ply)
    logger.info(f"Exported camera trajectory to {trajectory_ply}")

    # --- 4. Dense colored point cloud and surface mesh ---
    mesh_ply, mesh_decimated_ply, cloud_dense_ply = _generate_dense_model_and_mesh(run_path)

    # --- 5. Full 3D Map Completion (Volumetric structures + continuous terrain + Orthomosaic + DSM) ---
    map_3d_ply, map_3d_glb, orthomosaic_png, dsm_png = None, None, None, None
    if cloud_dense_ply and cloud_dense_ply.exists():
        try:
            from uav3d.modeling.completion import build_full_3d_map

            m_ply, m_glb, o_png, d_png = build_full_3d_map(cloud_dense_ply, run_path)
            if m_ply.exists():
                map_3d_ply = m_ply
            if m_glb.exists():
                map_3d_glb = m_glb
            if o_png.exists():
                orthomosaic_png = o_png
            if d_png.exists():
                dsm_png = d_png
        except Exception as exc:
            logger.warning(f"Full 3D map completion skipped or non-fatal: {exc}")

    # --- 6. Versioned summary manifest ---
    summary = ReconstructionSummary(
        schema_version="1.0.0",
        dataset_label=dataset_label,
        num_source_frames=num_source_frames,
        num_keyframes=num_keyframes,
        num_registered_frames=int(reconstruction.num_reg_images()),
        num_unregistered_frames=max(0, num_keyframes - int(reconstruction.num_reg_images())),
        registration_rate=float(
            reconstruction.num_reg_images() / num_keyframes if num_keyframes > 0 else 0.0
        ),
        mean_reprojection_error_px=mean_reprojection_error_px,
        num_models=num_models,
        num_points3d=num_points,
        mean_track_length=(
            float(reconstruction.compute_mean_track_length())
            if hasattr(reconstruction, "compute_mean_track_length")
            else None
        ),
        scale_note=scale_note,
        model_ply="model.ply",
        cloud_dense_ply="cloud_dense.ply" if cloud_dense_ply and cloud_dense_ply.exists() else None,
        mesh_ply="mesh.ply" if mesh_ply and mesh_ply.exists() else None,
        mesh_decimated_ply="mesh_decimated.ply" if mesh_decimated_ply and mesh_decimated_ply.exists() else None,
        map_3d_ply="map_3d.ply" if map_3d_ply and map_3d_ply.exists() else None,
        map_3d_glb="map_3d.glb" if map_3d_glb and map_3d_glb.exists() else None,
        orthomosaic_png="orthomosaic.png" if orthomosaic_png and orthomosaic_png.exists() else None,
        dsm_png="dsm_elevation.png" if dsm_png and dsm_png.exists() else None,
        model_dir="model",
        trajectory_ply="trajectory.ply",
        poses_json="poses.json",
        colmap_native_log="colmap_native.log",
    )
    # created_utc is filled by the caller after other stages may have written
    # telemetry/ATE artifacts; set a baseline here.
    from datetime import UTC, datetime

    summary.created_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary_path = run_path / "reconstruction_summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"Wrote reconstruction summary to {summary_path}")

    return ModelExportResult(
        model_ply=model_ply,
        model_dir=model_dir,
        trajectory_ply=trajectory_ply,
        summary_path=summary_path,
        num_points=num_points,
        mesh_ply=mesh_ply,
        mesh_decimated_ply=mesh_decimated_ply,
        cloud_dense_ply=cloud_dense_ply,
        map_3d_ply=map_3d_ply,
        map_3d_glb=map_3d_glb,
        orthomosaic_png=orthomosaic_png,
        dsm_png=dsm_png,
    )


def _filter_point_cloud_outliers(model_ply: Path) -> int:
    """Filter divergent sky rays and statistical outliers from sparse model.ply."""
    try:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(model_ply))
        if len(pcd.points) < 50:
            return len(pcd.points)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.5)
        pcd_clean = pcd.select_by_index(ind)
        o3d.io.write_point_cloud(str(model_ply), pcd_clean)
        return len(pcd_clean.points)
    except Exception as exc:
        logger.debug(f"Outlier filtering skipped: {exc}")
        return 0


def _generate_dense_model_and_mesh(run_path: Path) -> tuple[Path | None, Path | None, Path | None]:
    """Generate dense point cloud (cloud_dense.ply) and surface mesh (mesh.ply).

    Uses DepthAnything-V2-Small if available, or surface Poisson reconstruction.
    Returns:
        (mesh_ply, mesh_decimated_ply, cloud_dense_ply)
    """
    mesh_ply = run_path / "mesh.ply"
    mesh_decimated_ply = run_path / "mesh_decimated.ply"
    cloud_dense_ply = run_path / "cloud_dense.ply"

    try:
        import json
        import cv2
        import numpy as np
        import open3d as o3d
        from scipy.spatial.transform import Rotation

        poses_path = run_path / "poses.json"
        images_dir = run_path / "sfm" / "images"
        if not poses_path.exists() or not images_dir.exists():
            return None, None, None

        with open(poses_path, encoding="utf-8") as f:
            poses_data = json.load(f)

        poses = poses_data.get("poses", [])
        if not poses:
            return None, None, None

        cam_info = poses_data.get("camera", {})
        params = cam_info.get("params", [])
        if len(params) < 4:
            return None, None, None
        fx, fy, cx, cy = params[:4]
        width = cam_info.get("width", 1920)
        height = cam_info.get("height", 1080)

        # 1. Check for AI Depth Model
        model_checkpoint = Path("models/checkpoints/depth_anything_v2_vits.onnx")
        has_onnx = False
        sess = None
        if model_checkpoint.exists():
            try:
                import onnxruntime as ort

                sess = ort.InferenceSession(str(model_checkpoint), providers=["CPUExecutionProvider"])
                has_onnx = True
            except Exception as exc:
                logger.debug(f"ONNX session init skipped: {exc}")

        # Check if pycolmap model exists for sparse point calibration
        rec = None
        model_txt = run_path / "model"
        if model_txt.exists():
            try:
                import pycolmap

                rec = pycolmap.Reconstruction(str(model_txt))
            except Exception:
                rec = None

        # Estimate scene median depth from sparse points
        sparse_points_file = run_path / "model" / "points3D.txt"
        med_z = 15.0
        if sparse_points_file.exists():
            pts_z = []
            with open(sparse_points_file, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        pts_z.append(float(parts[3]))
            if pts_z:
                med_z = float(np.median(pts_z))

        dense_points = []
        dense_colors = []

        if has_onnx and sess is not None:
            logger.info("Generating dense AI depth point cloud and 3D surface mesh...")
            # Preserve true image aspect ratio (e.g. 1080x1920 portrait or 1920x1080 landscape)
            target_h = 384
            w_sub = max(64, int(round(target_h * (width / height))))
            h_sub = target_h
            u = np.linspace(0, width - 1, w_sub)
            v = np.linspace(0, height - 1, h_sub)
            uu, vv = np.meshgrid(u, v)

            for p in poses:
                fid = p.get("frame_id")
                img_name = f"{fid}.jpg"
                img_file = images_dir / img_name
                if not img_file.exists():
                    continue
                bgr = cv2.imread(str(img_file))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # Depth prediction
                resized = cv2.resize(rgb, (518, 518))
                inp = resized.astype(np.float32) / 255.0
                inp = (inp - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                inp = inp.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
                disp = sess.run(None, {"pixel_values": inp})[0][0]

                disp_sub = cv2.resize(disp, (w_sub, h_sub))
                rgb_sub = cv2.resize(rgb, (w_sub, h_sub))

                # Affine depth alignment against sparse triangulated points
                aligned_depth = None
                if rec is not None and rec.find_image_with_name(img_name) is not None:
                    img_obj = rec.find_image_with_name(img_name)
                    cfw = img_obj.cam_from_world() if callable(img_obj.cam_from_world) else img_obj.cam_from_world
                    disp_full = cv2.resize(disp, (width, height))
                    sp_z, sp_d = [], []
                    for p2d in img_obj.points2D:
                        if p2d.has_point3D():
                            X_w = rec.points3D[p2d.point3D_id].xyz
                            X_c = cfw * X_w
                            z = X_c[2]
                            px, py = int(round(p2d.xy[0])), int(round(p2d.xy[1]))
                            if 0 <= px < width and 0 <= py < height:
                                sp_z.append(z)
                                sp_d.append(disp_full[py, px])
                    if len(sp_z) >= 10:
                        sp_z = np.array(sp_z)
                        sp_d = np.array(sp_d)
                        inv_z = 1.0 / sp_z
                        s, t = np.polyfit(sp_d, inv_z, 1)
                        pred_inv = s * disp_sub + t
                        min_depth = max(1.0, float(np.percentile(sp_z, 1)) * 0.7)
                        max_depth = min(100.0, float(np.percentile(sp_z, 99)) * 2.0)
                        pred_z = 1.0 / np.maximum(pred_inv, 1.0 / max_depth)
                        aligned_depth = np.clip(pred_z, min_depth, max_depth)

                if aligned_depth is None:
                    aligned_depth = med_z / (disp_sub + 0.1)
                    aligned_depth = np.clip(aligned_depth, 1.0, 50.0)

                # Sky / infinite distance mask (disparity near zero or excessive depth)
                valid_mask = (disp_sub > 0.15) & (aligned_depth < 60.0)

                # Backproject
                x_c = (uu - cx) * aligned_depth / fx
                y_c = (vv - cy) * aligned_depth / fy
                z_c = aligned_depth

                pts_cam = np.stack([x_c, y_c, z_c], axis=-1).reshape(-1, 3)[valid_mask.reshape(-1)]
                colors = (rgb_sub.reshape(-1, 3).astype(np.float32) / 255.0)[valid_mask.reshape(-1)]

                # Transform to world coordinate frame
                rot = Rotation.from_quat([p["qx"], p["qy"], p["qz"], p["qw"]])
                R = rot.as_matrix()
                t_vec = np.array([p["tx"], p["ty"], p["tz"]])
                pts_world = (pts_cam - t_vec) @ R

                dense_points.append(pts_world)
                dense_colors.append(colors)

        if dense_points:
            all_pts = np.concatenate(dense_points, axis=0)
            all_cols = np.concatenate(dense_colors, axis=0)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_pts)
            pcd.colors = o3d.utility.Vector3dVector(all_cols)

            pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.2)
            pcd_down = pcd_clean.voxel_down_sample(voxel_size=0.08)
            o3d.io.write_point_cloud(str(cloud_dense_ply), pcd_down)
            logger.info(f"Exported dense colored point cloud ({len(pcd_down.points):,} points) to {cloud_dense_ply}")
        else:
            sparse_ply = run_path / "model.ply"
            if sparse_ply.exists():
                pcd_down = o3d.io.read_point_cloud(str(sparse_ply))
            else:
                return None, None, None

        # 2. Surface Mesh Reconstruction (Poisson + Density Trimming + Point Cloud Crop)
        pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=30))
        pcd_down.orient_normals_consistent_tangent_plane(20)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_down, depth=9)
        densities = np.asarray(densities)
        if len(densities) > 0:
            vertices_to_remove = densities < np.percentile(densities, 15)
            mesh.remove_vertices_by_mask(vertices_to_remove)

        # Crop mesh strictly to bounding box of point cloud to remove peripheral boundary walls
        bbox = pcd_down.get_axis_aligned_bounding_box()
        bbox = bbox.scale(1.05, bbox.get_center())
        mesh = mesh.crop(bbox)

        # Filter disconnected floating ribbons and debris (keep primary connected surface)
        triangle_clusters, num_clusters, _ = mesh.cluster_connected_triangles()
        tri_clusters = np.asarray(triangle_clusters)
        if len(tri_clusters) > 0:
            cluster_counts = np.bincount(tri_clusters)
            main_cluster_id = int(np.argmax(cluster_counts))
            mesh.remove_triangles_by_mask(tri_clusters != main_cluster_id)
            mesh.remove_unreferenced_vertices()

        # Color the mesh vertices from the dense point cloud
        pcd_tree = o3d.geometry.KDTreeFlann(pcd_down)
        mesh_colors = []
        for v in np.asarray(mesh.vertices):
            _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
            mesh_colors.append(np.asarray(pcd_down.colors)[idx[0]])
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(mesh_colors))
        mesh.compute_vertex_normals()

        o3d.io.write_triangle_mesh(str(mesh_ply), mesh)
        logger.info(f"Exported 3D surface mesh ({len(mesh.triangles):,} triangles) to {mesh_ply}")

        # 3. Decimated Mesh (lightweight 250k)
        target_triangles = min(250000, max(10000, len(mesh.triangles) // 2))
        mesh_dec = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
        mesh_dec.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(mesh_decimated_ply), mesh_dec)
        logger.info(f"Exported decimated mesh ({len(mesh_dec.triangles):,} triangles) to {mesh_decimated_ply}")

        return mesh_ply, mesh_decimated_ply, cloud_dense_ply
    except Exception as exc:
        logger.warning(f"Dense surface meshing skipped or non-fatal: {exc}")
        return None, None, None


def _registered_image_ids(reconstruction: Any) -> list[int]:
    """Registered image ids; pycolmap exposes this as a method or list by version."""
    ids = getattr(reconstruction, "reg_image_ids", None)
    if ids is None:
        return []
    return [int(i) for i in (ids() if callable(ids) else ids)]


def _write_trajectory_ply(reconstruction: Any, path: Path) -> None:
    """Write registered camera centers, ordered by frame name, as a PLY polyline."""
    # Collect (name, center) for all registered images, ordered lexicographically
    # so the path follows the capture sequence (frame IDs are zero-padded).
    entries: list[tuple[str, list[float]]] = []
    for image_id in _registered_image_ids(reconstruction):
        image = reconstruction.image(image_id)
        _, _, center = get_pose_components(image)
        entries.append((image.name, center))
    entries.sort(key=lambda item: item[0])

    n = len(entries)
    lines = [
        "ply",
        "format ascii 1.0",
        f"comment uav3d camera trajectory ({n} registered frames, ordered by frame id)",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element edge {max(0, n - 1)}",
        "property int vertex_index_1",
        "property int vertex_index_2",
        "end_header",
    ]
    for _, (cx, cy, cz) in entries:
        # Bright cyan flight path, distinct from scene point colors
        lines.append(f"{cx:.6f} {cy:.6f} {cz:.6f} 0 255 255")
    for i in range(n - 1):
        lines.append(f"{i} {i + 1}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_summary(
    summary_path: Path,
    telemetry_json: str | None = None,
    ate_report_json: str | None = None,
    map_3d_ply: str | None = None,
    map_3d_glb: str | None = None,
    orthomosaic_png: str | None = None,
    dsm_png: str | None = None,
) -> None:
    """Patch telemetry/ATE and 3D map artifact paths into an existing summary manifest."""
    if not summary_path.exists():
        return
    summary = ReconstructionSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
    if telemetry_json is not None:
        summary.telemetry_json = telemetry_json
    if ate_report_json is not None:
        summary.ate_report_json = ate_report_json
    if map_3d_ply is not None:
        summary.map_3d_ply = map_3d_ply
    if map_3d_glb is not None:
        summary.map_3d_glb = map_3d_glb
    if orthomosaic_png is not None:
        summary.orthomosaic_png = orthomosaic_png
    if dsm_png is not None:
        summary.dsm_png = dsm_png
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
