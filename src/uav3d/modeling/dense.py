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
import open3d as o3d

logger = logging.getLogger("uav3d.modeling.dense")

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MIN_SPARSE_MATCHES = 10
_MIN_SPARSE_FOR_FIT = 8
# The fusion constants below (near clip 1.0, depth_trunc 120, voxel clamp 0.08-0.25) were tuned on a
# scene whose median sparse depth is ~25 SfM units. SfM scale is arbitrary, so the whole scene is
# multiplied by _REF_DEPTH / median_depth before fusion and the outputs are scaled back afterwards.
_REF_DEPTH = 25.0


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
    if corr_pos >= corr_neg:
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
        if z <= 1e-9:
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
        # P2 degeneracy fix: too little range for affine correlation to be trusted;
        # reject affine fit so caller falls back to robust sparse-anchored scale.
        return False
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
    """Fit 1/Z ~ s*disp + t robustly with hard physical validation against sparse geometry."""
    inv_z = 1.0 / z_sparse
    med_inv = max(float(np.median(inv_z)), 1e-6)
    med_sparse_z = float(np.median(z_sparse))
    med_disp = float(np.median(disp_samples))

    fit = theil_sen_affine(disp_samples, inv_z)
    if fit is not None:
        s, t = fit
        implied_inv = s * med_disp + t
        if implied_inv > 1e-6:
            z_implied = 1.0 / implied_inv
            ratio = z_implied / max(med_sparse_z, 1e-6)
            # P1 hard-validation: implied frame depth must match triangulated sparse depth within 25%
            if 0.80 <= ratio <= 1.25:
                resid = inv_z - (s * disp_samples + t)
                rmse_rel = _inlier_rmse_rel(resid, np.sqrt(np.mean(resid**2)) / med_inv)
                if rmse_rel < 0.35 and _explains_variance(disp_samples, inv_z, resid):
                    return s, t, "theil_sen_huber", rmse_rel

    # Single-scale robust median anchor (t = 0.0)
    ok = disp_samples > 1e-6
    if int(ok.sum()) < _MIN_SPARSE_MATCHES:
        return None
    s = float(np.median(inv_z[ok] / disp_samples[ok]))
    if s <= 0.0 or not np.isfinite(s):
        return None
    resid = inv_z - s * disp_samples
    rmse_rel_all = float(np.sqrt(np.mean(resid**2)) / med_inv)
    med_r = float(np.median(resid[ok]))
    mad_r = float(np.median(np.abs(resid[ok] - med_r))) + 1e-12
    inlier_fraction = float(np.mean(np.abs(resid[ok] - med_r) <= 3.0 * mad_r))
    # Require both a dominant inlier set and a positive disparity/depth
    # relationship. This retains robust recovery with genuine outliers while
    # rejecting random depth maps that happen to have a few close samples.
    if inlier_fraction < 0.60 or not _explains_variance(disp_samples[ok], inv_z[ok], resid[ok]):
        return None
    rmse_rel = _inlier_rmse_rel(resid, rmse_rel_all)
    return s, 0.0, "median_scale_anchored", rmse_rel


def _depth_edge_mask(z_map: np.ndarray, rel_threshold: float = 0.15) -> np.ndarray:
    """Mask pixels at depth discontinuities (incl. a 2-px dilated halo).

    Monocular depth models bleed depth across object silhouettes; those
    pixels backproject into fuzzy "curtains" between foreground and
    background. A relative depth jump > ~15% between neighboring pixels is
    a discontinuity, not a surface — drop it and its immediate halo.
    """
    gz = np.abs(np.diff(z_map, axis=1))
    gz = np.pad(gz, ((0, 0), (0, 1)), mode="edge")
    gv = np.abs(np.diff(z_map, axis=0))
    gv = np.pad(gv, ((0, 1), (0, 0)), mode="edge")
    rel_grad = np.maximum(gz, gv) / np.maximum(z_map, 1e-6)
    edges = (rel_grad > rel_threshold).astype(np.uint8)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    return edges > 0


def _predict_disparity(session, rgb: np.ndarray, infer_w: int, infer_h: int) -> np.ndarray:
    """Run DA-V2 ONNX on an RGB image at aspect-preserving inference resolution."""
    resized = cv2.resize(rgb, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
    inp = resized.astype(np.float32) / 255.0
    inp = (inp - _IMAGENET_MEAN) / _IMAGENET_STD
    inp = inp.transpose(2, 0, 1)[None].astype(np.float32)
    disp = session.run(None, {"pixel_values": inp})[0][0]
    return disp


def _median_sparse_depth(rec) -> float:
    """Median camera-frame depth of all sparse observations (SfM units)."""
    depths: list[float] = []
    for img in rec.images.values():
        if not img.has_pose:
            continue
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        for p2d in img.points2D:
            if p2d.has_point3D():
                z = float((cfw * rec.points3D[p2d.point3D_id].xyz)[2])
                if z > 0:
                    depths.append(z)
    return float(np.median(depths)) if depths else 0.0


def _texture_mesh_visibility(
    mesh: o3d.geometry.TriangleMesh,
    frame_data: list[dict],
    work_w: int,
    work_h: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_trunc: float = 65.0,
    max_views_per_vert: int = 3,
) -> o3d.geometry.TriangleMesh:
    """Texture mesh vertices by projecting into visible keyframes with occlusion and angle checks.

    Replaces voxel-averaged RGB with median color from top non-occluded, frontal-facing views.
    """
    v = np.asarray(mesh.vertices)
    if len(v) == 0:
        return mesh

    mesh.compute_vertex_normals()
    n_vec = np.asarray(mesh.vertex_normals)

    n_verts = len(v)
    vert_candidates: list[list[tuple[float, np.ndarray]]] = [[] for _ in range(n_verts)]

    for fd in frame_data:
        if "z_map" not in fd or "rgb_work" not in fd:
            continue

        R = fd["R"]
        t_vec = fd["t_vec"]
        z_map = fd["z_map"]
        rgb_img = fd["rgb_work"]

        # 1. Transform vertices to camera frame: p_c = v @ R.T + t_vec
        p_c = v @ R.T + t_vec
        z_c = p_c[:, 2]

        front_mask = (z_c > 1.0) & (z_c < depth_trunc * 0.95)
        if not np.any(front_mask):
            continue

        valid_idx = np.where(front_mask)[0]
        p_c_val = p_c[valid_idx]
        z_c_val = z_c[valid_idx]

        # 2. Project to pixel coordinates
        u = fx * (p_c_val[:, 0] / z_c_val) + cx
        v_pix = fy * (p_c_val[:, 1] / z_c_val) + cy

        in_bounds = (u >= 2) & (u <= work_w - 3) & (v_pix >= 2) & (v_pix <= work_h - 3)
        if not np.any(in_bounds):
            continue

        valid_idx = valid_idx[in_bounds]
        p_c_val = p_c_val[in_bounds]
        z_c_val = z_c_val[in_bounds]
        u = u[in_bounds]
        v_pix = v_pix[in_bounds]

        # 3. Viewing angle: dot product in camera frame
        dist_cam = np.linalg.norm(p_c_val, axis=1, keepdims=True) + 1e-6
        view_dir = -p_c_val / dist_cam
        n_c = n_vec[valid_idx] @ R.T
        cos_theta = np.sum(n_c * view_dir, axis=1)

        facing = cos_theta > 0.15
        if not np.any(facing):
            continue

        valid_idx = valid_idx[facing]
        z_c_val = z_c_val[facing]
        u_int = np.round(u[facing]).astype(int)
        v_int = np.round(v_pix[facing]).astype(int)
        cos_theta = cos_theta[facing]

        # 4. Depth occlusion test
        z_pred = z_map[v_int, u_int]
        non_occluded = (z_pred > 0.5) & (np.abs(z_c_val - z_pred) / z_c_val <= 0.08)
        if not np.any(non_occluded):
            continue

        valid_idx = valid_idx[non_occluded]
        z_c_val = z_c_val[non_occluded]
        u_int = u_int[non_occluded]
        v_int = v_int[non_occluded]
        cos_theta = cos_theta[non_occluded]

        # Quality score: frontal orientation / distance
        scores = cos_theta / (z_c_val + 1e-3)
        sampled_cols = rgb_img[v_int, u_int].astype(np.float32) / 255.0

        for idx, sc, col in zip(valid_idx, scores, sampled_cols):
            vert_candidates[idx].append((float(sc), col))

    orig_cols = np.asarray(mesh.vertex_colors)
    has_orig = len(orig_cols) == n_verts
    new_cols = np.zeros((n_verts, 3), dtype=np.float64)

    n_retextured = 0
    for i in range(n_verts):
        cands = vert_candidates[i]
        if len(cands) > 0:
            cands.sort(key=lambda x: x[0], reverse=True)
            top_cols = np.array([c[1] for c in cands[:max_views_per_vert]])
            new_cols[i] = np.median(top_cols, axis=0)
            n_retextured += 1
        elif has_orig:
            new_cols[i] = orig_cols[i]
        else:
            new_cols[i] = [0.7, 0.7, 0.7]

    mesh.vertex_colors = o3d.utility.Vector3dVector(new_cols)
    logger.info(
        f"Visibility texturing (P5): {n_retextured:,}/{n_verts:,} vertices textured with median multi-view color"
    )
    return mesh


def generate_dense_and_mesh_tsdf(
    run_path: Path,
    voxel_length: float | None = None,
    depth_trunc: float = 65.0,
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

    unit = _median_sparse_depth(rec)
    k = _REF_DEPTH / unit if unit > 0 else 1.0
    logger.info(f"TSDF scene scale: median sparse depth {unit:.4g} units -> x{k:.4g} normalisation")

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
        span = float(np.linalg.norm(sp_pts.max(0) - sp_pts.min(0))) * k if len(sp_pts) else 50.0
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

    # Two-pass processing:
    # Pass 1: Extract disparities and initial affine depth alignment parameters for all frames
    frame_data: list[dict] = []
    logger.info("Pass 1/2: Extracting neural disparity and computing per-frame metric anchors...")

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

        s_val, t_val = None, None
        method_str = "unanchored"
        rmse_val = 1.0
        disparity_work = None

        if obs is not None:
            uv_px, z_sparse = obs
            z_sparse = z_sparse * k
            disp_samples = _sample_disparity(disp, uv_px)
            disparity = _oriented_disparity(disp, disp_samples, z_sparse)
            if disparity is not None:
                disparity_work = cv2.resize(disparity, (work_w, work_height), interpolation=cv2.INTER_LINEAR)
                disp_samples = _sample_disparity(disparity, uv_px)
                aligned = _align_disparity_to_depth(disp_samples, z_sparse)
                if aligned is not None:
                    s_val, t_val, method_str, rmse_val = aligned
                else:
                    # Defer to temporal interpolation across flightline rather than mixing 1-param fit
                    method_str = "interpolated"

        if disparity_work is None:
            disparity_work = cv2.resize(disp.astype(np.float32), (work_w, work_height), interpolation=cv2.INTER_LINEAR)

        from scipy.spatial.transform import Rotation
        R = Rotation.from_quat(
            [pose["qx"], pose["qy"], pose["qz"], pose["qw"]]
        ).as_matrix()
        t_vec = np.array([pose["tx"], pose["ty"], pose["tz"]], dtype=np.float64) * k

        frame_data.append({
            "frame_id": frame_id,
            "rgb": rgb,
            "disp_work": disparity_work,
            "s": s_val,
            "t": t_val,
            "method": method_str,
            "rmse_rel": rmse_val,
            "R": R,
            "t_vec": t_vec,
        })

    # Temporal smoothing and gap-filling of alignment parameters (s, t) across the flightline
    valid_s = [fd["s"] for fd in frame_data if fd["s"] is not None and fd["s"] > 0]
    valid_t = [fd["t"] for fd in frame_data if fd["t"] is not None]
    median_s = float(np.median(valid_s)) if valid_s else 1.0
    median_t = float(np.median(valid_t)) if valid_t else 0.0

    raw_s = np.array([fd["s"] if fd["s"] is not None and fd["s"] > 0 else np.nan for fd in frame_data], dtype=np.float64)
    raw_t = np.array([fd["t"] if fd["t"] is not None else np.nan for fd in frame_data], dtype=np.float64)

    # 1D linear interpolation for missing parameter values
    n_frames = len(frame_data)
    idx_all = np.arange(n_frames)
    nan_mask_s = np.isnan(raw_s)
    if nan_mask_s.all():
        clean_s = np.full(n_frames, median_s)
    elif nan_mask_s.any():
        clean_s = np.interp(idx_all, idx_all[~nan_mask_s], raw_s[~nan_mask_s])
    else:
        clean_s = raw_s.copy()

    nan_mask_t = np.isnan(raw_t)
    if nan_mask_t.all():
        clean_t = np.full(n_frames, median_t)
    elif nan_mask_t.any():
        clean_t = np.interp(idx_all, idx_all[~nan_mask_t], raw_t[~nan_mask_t])
    else:
        clean_t = raw_t.copy()

    # Apply rolling median filter and enforce global neighbor constraint (±30%)
    smooth_s = clean_s.copy()
    smooth_t = clean_t.copy()
    w_size = 5
    for i in range(n_frames):
        i0 = max(0, i - w_size // 2)
        i1 = min(n_frames, i + w_size // 2 + 1)
        smooth_s[i] = float(np.median(clean_s[i0:i1]))
        smooth_t[i] = float(np.median(clean_t[i0:i1]))
        # Reject and clamp any residual scale outliers beyond ±30% of global median
        ratio = smooth_s[i] / max(median_s, 1e-6)
        if ratio < 0.70 or ratio > 1.30:
            smooth_s[i] = float(np.clip(smooth_s[i], 0.70 * median_s, 1.30 * median_s))

    logger.info(
        f"Temporal depth consistency: scale range [{smooth_s.min():.4f}, {smooth_s.max():.4f}], "
        f"shift range [{smooth_t.min():.4f}, {smooth_t.max():.4f}] across {n_frames} keyframes"
    )

    # Pass 2: TSDF Integration with smooth scale continuity
    logger.info("Pass 2/2: TSDF fusion and multi-view consistency integration...")
    for i, fd in enumerate(frame_data):
        frame_id = fd["frame_id"]
        rgb = fd["rgb"]
        disp_work = fd["disp_work"]
        R = fd["R"]
        t_vec = fd["t_vec"]
        s = smooth_s[i]
        t = smooth_t[i]

        inv_z_map = s * disp_work + t
        # P3: Hard-clamp to depth_trunc (no sky integration)
        z_map = 1.0 / np.clip(inv_z_map, 1.0 / max(depth_trunc, 1e-6), 1.0)

        # Depth discontinuity mask: sever flying silhouette curtains
        edge_mask = _depth_edge_mask(z_map, rel_threshold=0.15).reshape(-1)

        x_cam = (uu - cx_s) * z_map / fx_s
        y_cam = (vv - cy_s) * z_map / fy_s
        pts_cam = np.stack([x_cam, y_cam, z_map], axis=-1).reshape(-1, 3)

        valid = (
            np.isfinite(pts_cam).all(axis=1)
            & (z_map.reshape(-1) > 1.0)
            & (z_map.reshape(-1) < depth_trunc * 0.95)
            & ~edge_mask
        )
        if int(valid.sum()) < 500:
            n_skipped += 1
            logger.info(f"{frame_id}: too few valid depth pixels; skipped")
            continue

        pts_world = (pts_cam[valid] - t_vec) @ R
        colors = cv2.resize(rgb, (work_w, work_height), interpolation=cv2.INTER_AREA)
        cloud_pts.append(pts_world.astype(np.float32))
        cloud_cols.append(colors.reshape(-1, 3)[valid])

        # Zero out silhouette edges and extreme depths so Open3D TSDF ignores flying curtain pixels
        z_map_fused = z_map.copy()
        z_map_fused[edge_mask.reshape(z_map.shape)] = 0.0
        z_map_fused[z_map_fused >= depth_trunc * 0.95] = 0.0

        fd["z_map"] = z_map_fused
        fd["rgb_work"] = colors

        depth_img = o3d.geometry.Image(z_map_fused.astype(np.float32))
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
            f"{frame_id}: fused (smooth_fit, s={s:.3f}, t={t:.3f}, "
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

    # P5: Visibility-checked per-vertex texturing using median color of top views
    try:
        mesh = _texture_mesh_visibility(
            mesh, frame_data, work_w, work_height, fx_s, fy_s, cx_s, cy_s, depth_trunc=depth_trunc
        )
    except Exception as tex_exc:
        logger.warning(f"Visibility texturing non-fatal fallback: {tex_exc}")

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
    pcd_out = o3d.geometry.PointCloud(pcd)
    pcd_out.scale(1.0 / k, center=(0.0, 0.0, 0.0))
    o3d.io.write_point_cloud(str(cloud_dense_ply), pcd_out)
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
    mesh.scale(1.0 / k, center=(0.0, 0.0, 0.0))  # back to the run's SfM units
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
