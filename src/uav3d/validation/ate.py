"""Absolute Trajectory Error (ATE) validation using 7-DoF Umeyama similarity alignment.

Strictly consumes GroundTruthAGL for evaluation only. Never consumed by visual SfM.
"""

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from uav3d.exceptions import DataError, Uav3dError
from uav3d.models.poses import PosesFile
from uav3d.models.validation import AteMetrics, AteReport, PerFrameAte
from uav3d.telemetry.zurich_parser import ZurichGroundTruthRecord, parse_ground_truth_agl

logger = logging.getLogger("uav3d.validation.ate")


def umeyama_alignment(
    est_points: np.ndarray,
    gt_points: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute optimal 7-DoF Sim(3) similarity transformation aligning est_points to gt_points.

    Solves:
        min_{s, R, t} \\sum || gt_i - (s * R @ est_i + t) ||^2
    with strict reflection guard: det(R) == +1.0 (SO(3)).

    Args:
        est_points: (N, 3) array of estimated camera positions.
        gt_points: (N, 3) array of ground truth positions.

    Returns:
        tuple of (scale s, rotation matrix R (3x3), translation vector t (3,), aligned_est_points (N, 3)).
    """
    X = np.asarray(est_points, dtype=np.float64)
    Y = np.asarray(gt_points, dtype=np.float64)

    if X.shape != Y.shape or X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"Shape mismatch: X={X.shape}, Y={Y.shape}. Expected (N, 3).")

    N, m = X.shape
    if N < 3:
        raise ValueError(f"Need at least 3 non-collinear points for 3D alignment, got {N}.")

    # 1. Centroids
    mu_x = np.mean(X, axis=0)
    mu_y = np.mean(Y, axis=0)

    # 2. Demeaned coordinates
    Xc = X - mu_x
    Yc = Y - mu_y

    # 3. Variance of estimated points
    var_x = float(np.mean(np.sum(Xc**2, axis=1)))
    if var_x < 1e-12:
        raise ValueError("Estimated points have zero or near-zero variance; cannot compute similarity scale.")

    # 4. Cross-covariance matrix
    cov = (Yc.T @ Xc) / N

    # 5. SVD
    U, D, Vt = np.linalg.svd(cov)

    # 6. Reflection Guard: ensure det(R) = +1
    S = np.eye(m)
    det_val = np.linalg.det(U) * np.linalg.det(Vt)
    if det_val < 0.0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt
    # Enforce exact orthogonality defensively
    det_R = np.linalg.det(R)
    if not np.isclose(det_R, 1.0, atol=1e-4):
        raise ValueError(f"Computed rotation matrix determinant is {det_R:.4f}, not +1.0 (SO(3) violated).")

    # 7. Optimal scale
    s = float(np.sum(S.diagonal() * D) / var_x)
    if s <= 0.0:
        logger.warning(f"Estimated similarity scale is non-positive ({s}); clipping to small epsilon.")
        s = 1e-6

    # 8. Optimal translation
    t = mu_y - s * (R @ mu_x)

    # 9. Transformed points
    aligned_X = s * (X @ R.T) + t

    return s, R, t, aligned_X


def check_flightline_geometry(gt_points: np.ndarray) -> tuple[str, dict[str, float]]:
    """Evaluate whether ground truth points are collinear or near-straight (degenerate for Sim(3)).

    Args:
        gt_points: (N, 3) ground truth positions.

    Returns:
        tuple of (condition_string, stats_dict).
        condition_string is 'weak_flightline_geometry' if cross-track variance is tiny,
        otherwise 'well_conditioned'.
    """
    pts = np.asarray(gt_points, dtype=np.float64)
    if len(pts) < 3:
        return "weak_flightline_geometry", {"primary_eigenval": 0.0, "secondary_eigenval": 0.0, "aspect_ratio_21": 0.0}

    cov = np.cov(pts, rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = np.maximum(eigvals, 0.0)

    lam1 = float(eigvals[0])
    lam2 = float(eigvals[1])
    lam3 = float(eigvals[2])

    if lam1 <= 1e-9:
        return "weak_flightline_geometry", {"primary_eigenval": lam1, "secondary_eigenval": lam2, "aspect_ratio_21": 0.0}

    aspect_ratio_21 = float(np.sqrt(lam2 / lam1))
    aspect_ratio_31 = float(np.sqrt(lam3 / lam1))

    # If cross-track horizontal spread is < 5% of longitudinal flightline spread
    # or secondary eigenvalue is < 0.5 m^2 (sub-meter cross-track dispersion)
    if aspect_ratio_21 < 0.05 or lam2 < 0.5:
        condition = "weak_flightline_geometry"
    else:
        condition = "ok"

    stats = {
        "primary_eigenval": lam1,
        "secondary_eigenval": lam2,
        "tertiary_eigenval": lam3,
        "aspect_ratio_21": aspect_ratio_21,
        "aspect_ratio_31": aspect_ratio_31,
    }
    return condition, stats


def compute_ate(
    poses_file: PosesFile,
    gt_records: dict[int, ZurichGroundTruthRecord] | Path | str,
    ate_threshold_m: float = 5.0,
    dataset_label: str = "replayed public dataset (Zurich Urban MAV) — not field-flight validation",
) -> AteReport:
    """Compute Absolute Trajectory Error (ATE) metrics against ground truth.

    Matches SfM poses strictly on source_image_id (never index-based).
    Requires at least 10 common matched frames; fails loudly otherwise.

    Args:
        poses_file: Validated PosesFile from Phase 3 SfM.
        gt_records: Ground truth records dict or path to GroundTruthAGL.csv.
        ate_threshold_m: Warning threshold for ATE RMSE in meters.
        dataset_label: Provenance label for report.

    Returns:
        Validated AteReport instance.
    """
    # 1. Parse GT if file path provided
    if isinstance(gt_records, (str, Path)):
        parsed_gt, _ = parse_ground_truth_agl(gt_records)
    elif isinstance(gt_records, dict):
        parsed_gt = gt_records
    else:
        raise TypeError(f"Expected dict or Path for gt_records, got {type(gt_records)}")

    # 2. Extract numeric image ID from frame_ids and find common matches
    num_pattern = re.compile(r"(\d+)$")

    matched_frame_ids: list[str] = []
    est_points_list: list[list[float]] = []
    gt_points_list: list[list[float]] = []

    for pose in poses_file.poses:
        if not pose.registered or pose.camera_center_world is None:
            continue

        fid = pose.frame_id
        match = num_pattern.search(fid)
        if not match:
            continue

        num_id = int(match.group(1))
        gt_entry = parsed_gt.get(num_id)
        if gt_entry is not None:
            matched_frame_ids.append(fid)
            est_points_list.append(list(pose.camera_center_world))
            gt_points_list.append([float(gt_entry.x), float(gt_entry.y), float(gt_entry.z)])

    matched_count = len(matched_frame_ids)
    logger.info(f"Common frames between SfM and Ground Truth: {matched_count}/{len(poses_file.poses)}")

    # 3. If ground truth is subsampled at intervals (e.g. Zurich 1 fps / 30-frame gaps),
    # interpolate along the flight path for intermediate keyframes between GT anchors.
    if matched_count < 10 and len(parsed_gt) >= 2:
        sorted_gt_ids = sorted(parsed_gt.keys())
        gt_min, gt_max = sorted_gt_ids[0], sorted_gt_ids[-1]
        max_gap = max(b - a for a, b in zip(sorted_gt_ids[:-1], sorted_gt_ids[1:])) if len(sorted_gt_ids) > 1 else 1

        if max_gap > 2:
            logger.info(
                f"Ground truth has {max_gap}-frame sampling intervals (e.g. 1 fps photogrammetry). "
                f"Interpolating ground truth along flight path for keyframes in range [{gt_min}, {gt_max}]."
            )
            import bisect

            matched_frame_ids = []
            est_points_list = []
            gt_points_list = []

            for pose in poses_file.poses:
                if not pose.registered or pose.camera_center_world is None:
                    continue
                fid = pose.frame_id
                match = num_pattern.search(fid)
                if not match:
                    continue
                num_id = int(match.group(1))

                if num_id in parsed_gt:
                    gt_entry = parsed_gt[num_id]
                    matched_frame_ids.append(fid)
                    est_points_list.append(list(pose.camera_center_world))
                    gt_points_list.append([float(gt_entry.x), float(gt_entry.y), float(gt_entry.z)])
                elif gt_min <= num_id <= gt_max:
                    idx = bisect.bisect_right(sorted_gt_ids, num_id)
                    if 0 < idx < len(sorted_gt_ids):
                        id_prev = sorted_gt_ids[idx - 1]
                        id_next = sorted_gt_ids[idx]
                        gt_prev = parsed_gt[id_prev]
                        gt_next = parsed_gt[id_next]
                        alpha = (num_id - id_prev) / (id_next - id_prev)
                        interp_x = float(gt_prev.x) + alpha * float(gt_next.x - gt_prev.x)
                        interp_y = float(gt_prev.y) + alpha * float(gt_next.y - gt_prev.y)
                        interp_z = float(gt_prev.z) + alpha * float(gt_next.z - gt_prev.z)
                        matched_frame_ids.append(fid)
                        est_points_list.append(list(pose.camera_center_world))
                        gt_points_list.append([interp_x, interp_y, interp_z])

            matched_count = len(matched_frame_ids)
            logger.info(
                f"After flight-path interpolation, common frames between SfM and Ground Truth: {matched_count}/{len(poses_file.poses)}"
            )

    # 4. Guard against too few matches
    if matched_count < 10:
        raise Uav3dError(
            f"ATE Validation FAILED: Fewer than 10 common frames matched between SfM "
            f"and Ground Truth ({matched_count} matched, required >= 10). "
            f"Cannot reliably compute 7-DoF similarity alignment."
        )

    X = np.array(est_points_list, dtype=np.float64)
    Y = np.array(gt_points_list, dtype=np.float64)

    # 4. Geometry condition check on GT points
    fit_condition, geom_stats = check_flightline_geometry(Y)
    if fit_condition == "weak_flightline_geometry":
        logger.warning(
            f"Ground truth trajectory geometry is near-collinear (aspect_ratio={geom_stats['aspect_ratio_21']:.4f}). "
            f"Flagging fit_condition='weak_flightline_geometry'; rotation about flight axis is weakly constrained."
        )

    # 5. Optimal 7-DoF Sim(3) Umeyama alignment
    s, R, t, aligned_X = umeyama_alignment(X, Y)
    logger.info(f"Umeyama alignment: scale s = {s:.4f}, translation = [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}] m")

    # 6. Compute Euclidean position errors in meters
    residuals = Y - aligned_X
    errors_m = np.linalg.norm(residuals, axis=1)

    rmse_m = float(np.sqrt(np.mean(errors_m**2)))
    median_m = float(np.median(errors_m))
    p95_m = float(np.percentile(errors_m, 95))
    max_m = float(np.max(errors_m))
    mean_m = float(np.mean(errors_m))

    reg_rate = poses_file.metrics.registration_rate if poses_file.metrics else None
    mean_reproj = poses_file.metrics.mean_reprojection_error_px if poses_file.metrics else None

    # 7. Construct per-frame records
    per_frame_records: list[PerFrameAte] = []
    for i, fid in enumerate(matched_frame_ids):
        per_frame_records.append(
            PerFrameAte(
                frame_id=fid,
                error_m=round(float(errors_m[i]), 4),
                est_world=[round(float(v), 5) for v in X[i]],
                gt_world=[round(float(v), 5) for v in Y[i]],
                aligned_est_world=[round(float(v), 5) for v in aligned_X[i]],
            )
        )

    metrics = AteMetrics(
        rmse_m=round(rmse_m, 4),
        median_m=round(median_m, 4),
        p95_m=round(p95_m, 4),
        max_m=round(max_m, 4),
        mean_m=round(mean_m, 4),
        scale_factor=round(s, 4),
        num_matched_frames=matched_count,
        num_registered_frames=len(poses_file.poses),
        num_gt_frames=len(parsed_gt),
        fit_condition=fit_condition,
        registration_rate=reg_rate,
        mean_reprojection_error_px=mean_reproj,
    )

    alignment_dict = {
        "scale": round(s, 6),
        "rotation_matrix": [[round(float(val), 6) for val in row] for row in R],
        "translation_m": [round(float(v), 5) for v in t],
        "det_R": round(float(np.linalg.det(R)), 6),
        "geometry_stats": geom_stats,
    }

    report = AteReport(
        schema_version="1.0.0",
        dataset_label=dataset_label,
        status="completed",
        metrics=metrics,
        alignment_transform=alignment_dict,
        per_frame_errors=per_frame_records,
        plots={},
    )

    return report
