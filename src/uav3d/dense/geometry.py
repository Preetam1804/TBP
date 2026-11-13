"""Projective geometry and error-model helpers for dense reconstruction.

Every formula here is in the pipeline's pose convention
``x_cam = R @ X_world + t`` with pixel-centre coordinates.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from uav3d.dense.cameras import CameraView
from uav3d.exceptions import DataError

logger = logging.getLogger("uav3d.dense.geometry")

# Achievable ZNCC sub-pixel disparity accuracy, in pixels. Real imagery with
# bilinear resampling, JPEG artefacts and exposure drift lands at 0.3-0.5 px;
# 0.3 is the optimistic end used only for sizing decisions, never for scoring.
DEFAULT_DISPARITY_SIGMA_PX = 0.3


def plane_homography(
    K: np.ndarray,
    ref_R: np.ndarray,
    ref_t: np.ndarray,
    src_R: np.ndarray,
    src_t: np.ndarray,
    depth: float,
) -> np.ndarray:
    """Homography mapping reference pixels to source pixels for a fronto-parallel plane.

    The plane is ``n . X_ref = depth`` with ``n = (0, 0, 1)``, i.e. a constant
    Z-depth plane in the reference camera. Derivation::

        X_src = A X_ref + b,  A = R_src R_ref^T,  b = t_src - A t_ref
        H(d)  = K_src [ A + (1/d) b e3^T ] K_ref^-1

    Args:
        K: shared 3x3 camera matrix (all working images are rectified).
        ref_R: reference rotation (cam_from_world).
        ref_t: reference translation.
        src_R: source rotation.
        src_t: source translation.
        depth: plane Z-depth in the reference camera.

    Returns:
        3x3 homography mapping reference pixel coordinates to source pixels.
    """
    A = src_R @ ref_R.T
    b = src_t - A @ ref_t
    M = A + np.outer(b, np.array([0.0, 0.0, 1.0])) / float(depth)
    return K @ M @ np.linalg.inv(K)


def depth_planes(near: float, far: float, count: int) -> np.ndarray:
    """Return ``count`` depth planes uniformly spaced in inverse depth.

    Uniform spacing in inverse depth (i.e. in disparity, for a fixed baseline)
    gives a constant sub-pixel step across the whole range, which is what the
    parabolic refinement in :mod:`uav3d.dense.sweep` assumes.
    """
    if near <= 0 or far <= near:
        raise DataError(f"Invalid depth range: near={near}, far={far}")
    inv = np.linspace(1.0 / far, 1.0 / near, int(count))
    return (1.0 / inv).astype(np.float64)


def plane_count_for_range(
    near: float,
    far: float,
    focal_px: float,
    baseline: float,
    step_px: float,
    min_planes: int,
    max_planes: int,
) -> int:
    """Choose a plane count that keeps the disparity step near ``step_px``.

    The disparity between two views separated by ``baseline`` at depth ``z`` is
    ``f * baseline / z``. Uniform inverse-depth planes therefore differ in
    disparity by ``f * baseline * (1/near - 1/far) / count``.
    """
    if far <= near or near <= 0:
        raise DataError(f"Invalid depth range: near={near}, far={far}")
    span_px = abs(focal_px * baseline) * (1.0 / near - 1.0 / far)
    count = int(np.ceil(span_px / max(step_px, 1e-6)))
    return int(np.clip(count, min_planes, max_planes))


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted percentile ``q`` (0-100) of ``values``."""
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cumulative = np.cumsum(w)
    if cumulative[-1] <= 0:
        return float(np.percentile(values, q))
    target = (q / 100.0) * cumulative[-1]
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(v[min(index, v.size - 1)])


def depth_sigma_array(depths: np.ndarray, focal_px: float, baseline: float, disparity_sigma_px: float = DEFAULT_DISPARITY_SIGMA_PX) -> np.ndarray:
    """Vectorised version of :func:`depth_noise_sigma`."""
    safe = np.maximum(np.asarray(depths, dtype=np.float64), 1e-9)
    if focal_px <= 0 or baseline <= 0:
        return np.full_like(safe, np.inf)
    return safe * safe / (focal_px * abs(baseline)) * disparity_sigma_px


def depth_noise_sigma(depth: float, focal_px: float, baseline: float, disparity_sigma_px: float = DEFAULT_DISPARITY_SIGMA_PX) -> float:
    """Depth standard deviation for triangulation with disparity noise.

    ``sigma_z = z^2 / (f b) * sigma_d`` — the standard stereo error propagation.

    This is the number that decides how fine a voxel is defensible: resolving
    below ``sigma_z`` reproduces noise, not detail.
    """
    if depth <= 0 or focal_px <= 0 or baseline <= 0:
        return float("inf")
    return float(depth * depth / (focal_px * abs(baseline)) * disparity_sigma_px)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform mapping ``src`` onto ``dst``.

    Returns:
        ``(scale, R, t)`` such that ``dst ~= scale * R @ src + t``.
    """
    if src.shape != dst.shape or src.shape[1] != 3:
        raise DataError("umeyama_similarity expects two (N, 3) arrays of equal length")
    if src.shape[0] < 3:
        raise DataError("umeyama_similarity needs at least 3 correspondences")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    s0 = src - mu_src
    d0 = dst - mu_dst
    cov = d0.T @ s0 / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_src = (s0**2).sum() / src.shape[0]
    scale = float(np.trace(np.diag(D) @ S) / var_src) if var_src > 0 else 1.0
    t = mu_dst - scale * R @ mu_src
    return scale, R, t


def sparse_depth_range(
    view: CameraView,
    points: np.ndarray,
    low_pct: float,
    high_pct: float,
    margin: float,
) -> tuple[float, float, float, int]:
    """Derive a per-view depth range from sparse SfM points visible in that view.

    Args:
        view: reference view.
        points: ``(N, 3)`` sparse points in SfM units.
        low_pct / high_pct: percentiles of the observed depth distribution.
        margin: fractional expansion applied on both sides.

    Returns:
        ``(near, far, median_depth, num_points)``. The median is what view
        selection and the error model must use: the midpoint of a long street is
        not a representative depth.
    """
    uv, z = view.project(points)
    inside = (
        (z > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < view.intrinsics.width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < view.intrinsics.height)
    )
    z = z[inside]
    if z.size < 20:
        raise DataError(
            f"Only {z.size} sparse points project into frame {view.frame_id}; "
            "cannot derive a trustworthy depth range for dense MVS."
        )

    # Sparse points sample the surface, not the image. A point at depth z covers
    # an image area proportional to 1/z^2, so weighting by that converts the
    # surface distribution into the pixel-coverage distribution — which is what
    # depth planes, baseline selection and the error model must be based on.
    weights = 1.0 / np.maximum(z, 1e-6) ** 2
    near = _weighted_percentile(z, weights, low_pct)
    far = _weighted_percentile(z, weights, high_pct)
    median = _weighted_percentile(z, weights, 50.0)
    near *= 1.0 - margin
    far *= 1.0 + margin
    return max(near, 1e-6), max(far, near * 1.2), max(median, 1e-6), int(z.size)


def parse_depth_range(spec: str) -> tuple[float, float]:
    """Parse a ``"min:max"`` depth-range override."""
    try:
        lo_s, hi_s = spec.split(":")
        near, far = float(lo_s), float(hi_s)
    except Exception as exc:  # noqa: BLE001 - normalise any parse failure
        raise DataError(f"Invalid depth_range '{spec}'; expected 'min:max', e.g. '5:200'") from exc
    if near <= 0 or far <= near:
        raise DataError(f"Invalid depth_range '{spec}': need 0 < min < max")
    return near, far


def load_sparse_points(run_dir: str | Path) -> np.ndarray | None:
    """Load the SfM sparse point cloud for a run, if it is available.

    Tries ``sparse_points.npz`` first, then the COLMAP model written by the SfM
    stage (``colmap/sparse/*``). Returns ``None`` when neither exists, in which
    case the caller must supply an explicit depth range.
    """
    run_path = Path(run_dir)
    npz = run_path / "sparse_points.npz"
    if npz.exists():
        with np.load(npz) as data:
            if "points" in data:
                return np.asarray(data["points"], dtype=np.float64)

    sparse_root = run_path / "colmap" / "sparse"
    if not sparse_root.exists():
        return None
    candidates = [p for p in sorted(sparse_root.iterdir()) if p.is_dir()] or [sparse_root]
    try:
        import pycolmap
    except ImportError:
        logger.warning("pycolmap unavailable; cannot read the COLMAP sparse model for depth ranges")
        return None

    for candidate in candidates:
        try:
            reconstruction = pycolmap.Reconstruction(str(candidate))
        except Exception:  # noqa: BLE001 - model directories vary by version
            continue
        points = reconstruction.points3D
        if not points:
            continue
        xyz = np.array([p.xyz for p in points.values()], dtype=np.float64)
        logger.info("Loaded %d sparse points from %s", xyz.shape[0], candidate)
        return xyz
    return None


def write_sparse_points(points: np.ndarray, run_dir: str | Path) -> Path:
    """Cache sparse points next to the run outputs for fast reuse."""
    dest = Path(run_dir) / "sparse_points.npz"
    np.savez_compressed(dest, points=np.asarray(points, dtype=np.float32))
    return dest
