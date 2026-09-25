"""Dense reconstruction via multi-view TSDF fusion with robust metric alignment.

Replaces naive per-frame backprojection (which produced independent "depth
sheets" per keyframe that surfaced as a fan of conflicting blades) with:

1. Per-frame robust alignment of DA-V2 relative disparity to the sparse SfM
   depths in INVERSE-DISPARITY space (Theil-Sen + Huber IRLS polish), which is
   the space where the model is linear and far-range noise is not amplified.
2. Aspect-preserving depth inference (the legacy path squeezed 1080x1920 into
   518x518, shifting every object horizontally).
3. Correct pixel-center backprojection grid (the legacy grid mixed working-res
   coordinates with full-res principal points, skewing points ~300 px off-axis).
4. Multi-view TSDF fusion: overlapping frames are averaged in a single voxel
   grid, collapsing per-frame depth sheets into one consistent surface.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("uav3d.modeling.dense")

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MIN_SPARSE_MATCHES = 10
_MIN_SPARSE_FOR_FIT = 8


def theil_sen_affine(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    max_pairs: int = 4000,
) -> tuple[float, float] | None:
    """Robust affine fit y ~ s*x + t via Theil-Sen slope + Huber IRLS polish.

    Returns None when the data cannot constrain the fit.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < _MIN_SPARSE_FOR_FIT:
        return None

    if n * (n - 1) // 2 > max_pairs:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, max_pairs)
        j = rng.integers(0, n, max_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
    else:
        i, j = np.triu_indices(n, k=1)

    dx = x[j] - x[i]
    ok = np.abs(dx) > 1e-9
    if int(ok.sum()) < _MIN_SPARSE_FOR_FIT:
        return None
    slopes = (y[j[ok]] - y[i[ok]]) / dx[ok]
    s = float(np.median(slopes))
    if not np.isfinite(s) or s <= 0.0:
        return None
    t = float(np.median(y - s * x))

    # Huber IRLS polish (2 iterations, MAD-scaled delta)
    resid = y - (s * x + t)
    mad = float(np.median(np.abs(resid - np.median(resid)))) + 1e-9
    for _ in range(2):
        resid = y - (s * x + t)
        delta = 1.345 * mad
        w = np.minimum(1.0, delta / (np.abs(resid) + 1e-12))
        w_sum = w.sum()
        sx, sy = (w * x).sum(), (w * y).sum()
        sxx, sxy = (w * x * x).sum(), (w * x * y).sum()
        denom = w_sum * sxx - sx * sx
        if abs(denom) < 1e-12:
            break
        s = float((w_sum * sxy - sx * sy) / denom)
        t = float((sy - s * sx) / w_sum)
    if not np.isfinite(s) or s <= 0.0:
        return None
    return s, t


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation without scipy.stats (argsort double trick)."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _oriented_disparity(
    model_out: np.ndarray,
    disp_samples: np.ndarray,
    z_sparse: np.ndarray,
) -> np.ndarray | None:
    """Return disparity (larger = closer) from raw DA-V2 output.

    ONNX exports of DA-V2 differ: some emit relative disparity (larger =
    closer), others the shifted inverse-disparity pseudo-depth 1/(d + eps)
    (larger = farther). The orientation is detected empirically per frame via
    rank correlation against sparse SfM depths and inverted when needed. The
    reciprocal of the pseudo-depth form is affine in 1/Z, so this restores a
    quantity that linearly aligns to metric inverse depth.

    Returns None when neither orientation correlates with the sparse depths.
    """
    corr_pos = _rank_corr(disp_samples, 1.0 / z_sparse)
    corr_neg = _rank_corr(disp_samples, z_sparse)
    if max(abs(corr_pos), abs(corr_neg)) < 0.3:
        return None
    if corr_pos >= abs(corr_neg):
        return model_out.astype(np.float32)
    return (1.0 / np.maximum(model_out, 1e-6)).astype(np.float32)


def _extract_sparse_observations(
    rec,  # pycolmap.Reconstruction
    image_name: str,
    width: int,
    height: int,
    infer_w: int,
    infer_h: int,
    max_reproj_error: float = 1.5,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Sparse (disp_pixel_xy, Z) observations for one image.

    Returns (uv_pixels at inference resolution [N,2], camera-frame depths [N]).
    """
    img = rec.find_image_with_name(image_name)
    if img is None:
        return None
    cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world

    uv_px, depths = [], []
    sx = infer_w / float(width)
    sy = infer_h / float(height)
    for p2d in img.points2D:
        if not p2d.has_point3D():
            continue
        p3d = rec.points3D[p2d.point3D_id]
        if getattr(p3d, "error", 0.0) > max_reproj_error:
            continue
        z = float((cfw * p3d.xyz)[2])
        if z <= 0.5:
            continue
        uv_px.append([p2d.xy[0] * sx, p2d.xy[1] * sy])
        depths.append(z)
    if len(depths) < _MIN_SPARSE_MATCHES:
        return None
    return np.asarray(uv_px, dtype=np.float64), np.asarray(depths, dtype=np.float64)


def _sample_disparity(disp: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinearly sample disparity values at floating pixel coordinates."""
    h, w = disp.shape
    u = np.clip(uv[:, 0], 0.0, w - 1.001)
    v = np.clip(uv[:, 1], 0.0, h - 1.001)
    u0 = u.astype(np.int64)
    v0 = v.astype(np.int64)
    fu = u - u0
    fv = v - v0
    d00 = disp[v0, u0]
    d01 = disp[v0, u0 + 1]
    d10 = disp[v0 + 1, u0]
    d11 = disp[v0 + 1, u0 + 1]
    top = d00 * (1 - fu) + d01 * fu
    bot = d10 * (1 - fu) + d11 * fu
    return (top * (1 - fv) + bot * fv).ravel()


def _inlier_rmse_rel(
    resid: np.ndarray,
    rmse_rel_all: float,
) -> float:
    """RMSE of residuals inside the robust inlier band (|r| <= 3*MAD).

    A robust fit correctly rejects gross outliers, so the quality gate must
    not penalize them either — otherwise 25% outliers (exactly what the fit
    is designed to survive) fail the gate.
    """
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med))) + 1e-12
    inlier = np.abs(resid - med) <= 3.0 * mad
    if int(inlier.sum()) < max(_MIN_SPARSE_FOR_FIT, len(resid) // 2):
        return rmse_rel_all
    return float(np.sqrt(np.mean(resid[inlier] ** 2)))


def _explains_variance(x: np.ndarray, y: np.ndarray, resid: np.ndarray) -> bool:
    """Check the affine fit genuinely explains the data (not self-confirming).

    An inlier-RMSE gate alone can pass on unrelated data (the inlier band only
    measures points the fit already agrees with). True disparity vs inverse-
    depth data is strongly linear, so require a healthy inlier correlation
    whenever the samples span a usable range.
    """
    spread = float(np.percentile(x, 95) - np.percentile(x, 5))
    med = max(abs(float(np.median(x))), 1e-9)
    if spread / med < 0.1:
        # Too little range for correlation to be meaningful; RMSE gate decides.
        return True
    med_r = float(np.median(resid))
    mad_r = float(np.median(np.abs(resid - med_r))) + 1e-12
    inlier = np.abs(resid - med_r) <= 3.0 * mad_r
    if int(inlier.sum()) < _MIN_SPARSE_FOR_FIT:
        return False
    xi, yi = x[inlier], y[inlier]
    sx, sy = xi.std(), yi.std()
    if sx < 1e-12 or sy < 1e-12:
        return False
    corr = float(np.corrcoef(xi, yi)[0, 1])
    return corr >= 0.7


def _align_disparity_to_depth(
    disp_samples: np.ndarray,
    z_sparse: np.ndarray,
) -> tuple[float, float, str, float] | None:
    """Fit 1/Z ~ s*disp + t robustly. Returns (s, t, method, rmse_rel)."""
    inv_z = 1.0 / z_sparse
    med_inv = max(float(np.median(inv_z)), 1e-6)
    fit = theil_sen_affine(disp_samples, inv_z)
    if fit is not None:
        s, t = fit
        resid = inv_z - (s * disp_samples + t)
        rmse_rel = _inlier_rmse_rel(resid, np.sqrt(np.mean(resid**2)) / med_inv)
        if rmse_rel < 0.35 and _explains_variance(disp_samples, inv_z, resid):
            return s, t, "theil_sen_huber", rmse_rel
    # Single-scale median fallback (t = 0)
    ok = disp_samples > 1e-6
    if int(ok.sum()) < _MIN_SPARSE_MATCHES:
        return None
    s = float(np.median(inv_z[ok] / disp_samples[ok]))
    if s <= 0.0:
        return None
    resid = inv_z - s * disp_samples
    rmse_rel = _inlier_rmse_rel(resid, np.sqrt(np.mean(resid**2)) / med_inv)
    if rmse_rel < 0.5 and _explains_variance(disp_samples, inv_z, resid):
        return s, 0.0, "median_scale_fallback", rmse_rel
    return None


def _depth_edge_mask(z_map: np.ndarray, rel_threshold: float = 0.18) -> np.ndarray:
    """Mask pixels at depth discontinuities (incl. a 1-px dilated halo).

    Monocular depth models bleed depth across object silhouettes; those
    pixels backproject into fuzzy "curtains" between foreground and
    background. A relative depth jump > ~18% between neighboring pixels is
    a discontinuity, not a surface — drop it and its immediate halo.
    """
    gz = np.abs(np.diff(z_map, axis=1))
    gz = np.pad(gz, ((0, 0), (0, 1)), mode="edge")
    gv = np.abs(np.diff(z_map, axis=0))
    gv = np.pad(gv, ((0, 1), (0, 0)), mode="edge")
    rel_grad = np.maximum(gz, gv) / np.maximum(z_map, 1e-6)
    edges = (rel_grad > rel_threshold).astype(np.uint8)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges > 0


def _predict_disparity(session, rgb: np.ndarray, infer_w: int, infer_h: int) -> np.ndarray:
    """Run DA-V2 ONNX on an RGB image at aspect-preserving inference resolution."""
    resized = cv2.resize(rgb, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
    inp = resized.astype(np.float32) / 255.0
    inp = (inp - _IMAGENET_MEAN) / _IMAGENET_STD
    inp = inp.transpose(2, 0, 1)[None].astype(np.float32)
    disp = session.run(None, {"pixel_values": inp})[0][0]
    return disp


def generate_dense_and_mesh_tsdf(
    run_path: Path,
    voxel_length: float | None = None,
    depth_trunc: float = 120.0,
    work_height: int = 480,
    infer_height: int = 518,
) -> tuple[Path | None, Path | None, Path | None]:
    """Multi-view TSDF fusion of DA-V2 depth maps anchored to sparse SfM depths.

    Args:
        run_path: Run directory containing poses.json, model/ (COLMAP text),
            and sfm/images/.
        voxel_length: TSDF voxel size in scene units; defaults to an
            extent-adaptive value.
        depth_trunc: Depths beyond this (scene units) are ignored.
        work_height: Working resolution height for fusion.
        infer_height: DA-V2 inference height (width follows aspect, /14).

    Returns:
        (mesh_ply, mesh_decimated_ply, cloud_dense_ply) or (None, None, None).
    """
    import json

    import open3d as o3d
    import pycolmap

    poses_path = run_path / "poses.json"
    images_dir = run_path / "sfm" / "images"
    model_dir = run_path / "model"
    if not poses_path.exists() or not images_dir.exists() or not model_dir.exists():
        return None, None, None

    poses_data = json.loads(poses_path.read_text(encoding="utf-8"))
    poses = poses_data.get("poses", [])
    cam_info = poses_data.get("camera", {})
    params = [float(p) for p in cam_info.get("params", [])]
    if not poses or not params:
        return None, None, None
    if len(params) == 3:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        fx, fy, cx, cy = params[:4]
    width = int(cam_info.get("width", 1080))
    height = int(cam_info.get("height", 1920))

    checkpoint = Path("models/checkpoints/depth_anything_v2_vits.onnx")
    if not checkpoint.exists():
        logger.warning("DA-V2 ONNX checkpoint missing; TSDF dense stage skipped")
        return None, None, None
    import onnxruntime as ort

    session = ort.InferenceSession(str(checkpoint), providers=["CPUExecutionProvider"])

    rec = pycolmap.Reconstruction(str(model_dir))

    # Working + inference resolutions (aspect-preserving; /14 for ViT patching)
    scale_w = width / float(height)
    work_w = max(16, round(work_height * scale_w))
    infer_w = max(14, round(infer_height * scale_w / 14.0) * 14)
    infer_h = max(14, round(infer_height / 14.0) * 14)

    # Exact resize mapping between working-res pixels and full-res pixels
    scale_u = width / float(work_w)
    scale_v = height / float(work_height)
    fx_s, fy_s = fx / scale_u, fy / scale_v
    cx_s = (cx + 0.5) / scale_u - 0.5
    cy_s = (cy + 0.5) / scale_v - 0.5

    u_full = (np.arange(work_w, dtype=np.float64) + 0.5) * scale_u - 0.5
    v_full = (np.arange(work_height, dtype=np.float64) + 0.5) * scale_v - 0.5
    uu, vv = np.meshgrid(u_full, v_full)

    # Adaptive voxel size from the sparse scene extent
    if voxel_length is None:
        sp_pts = np.array([p.xyz for p in rec.points3D.values()])
        span = float(np.linalg.norm(sp_pts.max(0) - sp_pts.min(0))) if len(sp_pts) else 50.0
        voxel_length = float(np.clip(span * 0.002, 0.08, 0.25))
    sdf_trunc = 3.0 * voxel_length
    logger.info(
        f"TSDF fusion: voxel={voxel_length:.3f} trunc={sdf_trunc:.2f} "
        f"work={work_w}x{work_height} infer={infer_w}x{infer_h}"
    )

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        work_w, work_height, fx_s, fy_s, cx_s, cy_s
    )

    cloud_pts: list[np.ndarray] = []
    cloud_cols: list[np.ndarray] = []
    n_integrated = 0
    n_skipped = 0

    for pose in poses:
        frame_id = pose.get("frame_id")
        img_file = images_dir / f"{frame_id}.jpg"
        if frame_id is None or not img_file.exists():
            continue
        bgr = cv2.imread(str(img_file))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        disp = _predict_disparity(session, rgb, infer_w, infer_h)

        obs = _extract_sparse_observations(
            rec, f"{frame_id}.jpg", width, height, infer_w, infer_h
        )
        if obs is None:
            n_skipped += 1
            logger.info(f"{frame_id}: <{_MIN_SPARSE_MATCHES} sparse matches; skipped")
            continue
        uv_px, z_sparse = obs
        disp_samples = _sample_disparity(disp, uv_px)
        disparity = _oriented_disparity(disp, disp_samples, z_sparse)
        if disparity is None:
            n_skipped += 1
            logger.info(f"{frame_id}: model output uncorrelated with SfM depths; skipped")
            continue
        disp_work = cv2.resize(disparity, (work_w, work_height), interpolation=cv2.INTER_LINEAR)
        disp_samples = _sample_disparity(disparity, uv_px)
        aligned = _align_disparity_to_depth(disp_samples, z_sparse)
        if aligned is None:
            n_skipped += 1
            logger.info(f"{frame_id}: alignment failed quality gate; skipped")
            continue
        s, t, method, rmse_rel = aligned

        inv_z_map = s * disp_work + t
        z_map = 1.0 / np.clip(inv_z_map, 1.0 / max(depth_trunc * 2.0, 1e-6), 1.0)

        # Depth-discontinuity masking: drop silhouette-bleed pixels before
        # they contaminate the volume or the cloud.
        edge_mask = _depth_edge_mask(z_map).reshape(-1)

        x_cam = (uu - cx) * z_map / fx
        y_cam = (vv - cy) * z_map / fy
        pts_cam = np.stack([x_cam, y_cam, z_map], axis=-1).reshape(-1, 3)

        valid = (
            np.isfinite(pts_cam).all(axis=1)
            & (z_map.reshape(-1) > 1.0)  # near clip: ultra-close junk is noise
            & (z_map.reshape(-1) < depth_trunc)
            & ~edge_mask
        )
        if int(valid.sum()) < 1000:
            n_skipped += 1
            logger.info(f"{frame_id}: too few valid depth pixels; skipped")
            continue

        # cam_from_world (wxyz quaternion) -> world points
        from scipy.spatial.transform import Rotation

        R = Rotation.from_quat(
            [pose["qx"], pose["qy"], pose["qz"], pose["qw"]]
        ).as_matrix()
        t_vec = np.array([pose["tx"], pose["ty"], pose["tz"]], dtype=np.float64)
        pts_world = (pts_cam[valid] - t_vec) @ R

        colors = cv2.resize(rgb, (work_w, work_height), interpolation=cv2.INTER_AREA)
        cloud_pts.append(pts_world.astype(np.float32))
        cloud_cols.append(colors.reshape(-1, 3)[valid])

        # TSDF integrate the depth map directly (streaming; memory bounded)
        depth_img = o3d.geometry.Image(z_map.astype(np.float32))
        color_img = o3d.geometry.Image(colors)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img,
            depth_img,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t_vec
        volume.integrate(rgbd, intrinsic, extrinsic)
        n_integrated += 1
        logger.info(
            f"{frame_id}: fused ({method}, rmse_rel={rmse_rel:.3f}, "
            f"z_median={float(np.median(z_map.reshape(-1)[valid])):.1f})"
        )

    logger.info(f"TSDF fused {n_integrated} frame(s); skipped {n_skipped}")
    if n_integrated < 2:
        logger.warning("Too few frames fused for TSDF meshing")
        return None, None, None

    mesh_ply = run_path / "mesh.ply"
    mesh_decimated_ply = run_path / "mesh_decimated.ply"
    cloud_dense_ply = run_path / "cloud_dense.ply"

    # TSDF mesh (already multi-view consistent; no hallucinated lobes)
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    # Preferred dense cloud: the TSDF mesh vertices — the mesh is the TSDF
    # zero-crossing surface, so every vertex lies ON the fused multi-view
    # surface (extract_point_cloud() would return ALL observed voxels along
    # rays, i.e. the full viewing-cone volume, not the surface). Colors come
    # from the mesh, which averages all frames that observed each voxel.
    pcd = o3d.geometry.PointCloud()
    if len(mesh.vertices) > 1000:
        pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        if len(mesh.vertex_colors) > 0:
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
        pcd = pcd.voxel_down_sample(voxel_size=voxel_length * 0.75)
        logger.info(f"Dense cloud from TSDF surface (mesh vertices): {len(pcd.points):,} points")
    else:
        all_pts = np.concatenate(cloud_pts, axis=0)
        all_cols = np.concatenate(cloud_cols, axis=0).astype(np.float64) / 255.0
        pcd.points = o3d.utility.Vector3dVector(all_pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(all_cols)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_length * 0.75)
        logger.info(f"Dense cloud from backprojection fallback: {len(pcd.points):,} points")

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    o3d.io.write_point_cloud(str(cloud_dense_ply), pcd)
    logger.info(f"Exported dense colored point cloud ({len(pcd.points):,} points)")

    if len(mesh.triangles) == 0:
        logger.warning("TSDF mesh empty; falling back to Poisson on the dense cloud")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_length * 3, max_nn=40)
        )
        pcd.orient_normals_consistent_tangent_plane(20)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        densities = np.asarray(densities)
        if len(densities) > 0:
            mesh.remove_vertices_by_mask(densities < np.percentile(densities, 15))

    # Crop to observed cloud extent and drop sliver components
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.05, bbox.get_center())
    mesh = mesh.crop(bbox)
    triangle_clusters, cluster_sizes, _ = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(triangle_clusters)
    if len(cluster_sizes) > 1 and len(tri_clusters) > 0:
        sizes = np.asarray(cluster_sizes)
        keep = sizes >= max(100, 0.01 * sizes.max())
        mesh.remove_triangles_by_mask(~keep[tri_clusters])
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(mesh_ply), mesh)
    logger.info(
        f"Exported TSDF surface mesh ({len(mesh.triangles):,} triangles) to {mesh_ply}"
    )

    target_tris = min(400_000, max(10_000, len(mesh.triangles) // 2))
    mesh_dec = mesh.simplify_quadric_decimation(target_number_of_triangles=target_tris)
    mesh_dec.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(mesh_decimated_ply), mesh_dec)
    logger.info(
        f"Exported decimated mesh ({len(mesh_dec.triangles):,} triangles) to {mesh_decimated_ply}"
    )

    return mesh_ply, mesh_decimated_ply, cloud_dense_ply
