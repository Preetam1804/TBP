"""Plane-sweep multi-view stereo on the CPU.

For every reference view the scene is swept with fronto-parallel planes; each
plane is scored by the multi-view zero-mean normalised cross-correlation (ZNCC)
between the reference image and every warped source image. The depth estimate is
the cost minimum, refined to sub-plane accuracy by a parabola fit in inverse
depth.

Memory is O(H*W): the cost volume is never materialised, because the running
minimum and its two neighbours are tracked online. That is what makes a
1600x900 sweep over 100+ planes fit in a few hundred megabytes.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from uav3d.dense.cameras import CameraView
from uav3d.dense.geometry import plane_homography


@dataclass
class DepthEstimate:
    """Per-pixel depth estimate for one reference view."""

    depth: np.ndarray
    cost: np.ndarray
    num_views: np.ndarray
    valid: np.ndarray


def to_unit_float(image: np.ndarray) -> np.ndarray:
    """Convert an image to float32 in ``[0, 1]`` without double-normalising."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) * (1.0 / 255.0)
    out = arr.astype(np.float32)
    peak = float(out.max()) if out.size else 0.0
    if peak > 1.5:
        out *= 1.0 / 255.0
    return out


def zncc_stats(image: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Local mean and variance for ZNCC, with reflection padding."""
    mu = cv2.boxFilter(image, -1, (window, window), normalize=True, borderType=cv2.BORDER_REFLECT)
    var = (
        cv2.boxFilter(image * image, -1, (window, window), normalize=True, borderType=cv2.BORDER_REFLECT)
        - mu * mu
    )
    return mu, np.maximum(var, 1e-6)


def _zncc_cost(
    ref: np.ndarray,
    ref_mu: np.ndarray,
    ref_var: np.ndarray,
    warped: np.ndarray,
    window: int,
) -> np.ndarray:
    """Return ``1 - ZNCC`` in ``[0, 2]`` between the reference and a warped source."""
    mu_w = cv2.boxFilter(warped, -1, (window, window), normalize=True, borderType=cv2.BORDER_REFLECT)
    var_w = (
        cv2.boxFilter(warped * warped, -1, (window, window), normalize=True, borderType=cv2.BORDER_REFLECT)
        - mu_w * mu_w
    )
    cov = (
        cv2.boxFilter(warped * ref, -1, (window, window), normalize=True, borderType=cv2.BORDER_REFLECT)
        - mu_w * ref_mu
    )
    zncc = cov / np.sqrt(np.maximum(var_w, 1e-6) * ref_var)
    return np.clip(1.0 - zncc, 0.0, 2.0).astype(np.float32)


def estimate_depth(
    ref_gray: np.ndarray,
    ref_view: CameraView,
    source_grays: list[tuple[CameraView, np.ndarray]],
    planes: np.ndarray,
    window: int = 5,
    min_views: int = 2,
) -> DepthEstimate:
    """Run a plane sweep for one reference view.

    Args:
        ref_gray: reference image (uint8 or float).
        ref_view: reference camera.
        source_grays: list of ``(view, image)`` source pairs, already ordered by quality.
        planes: depth planes in SfM units, ordered near-to-far or far-to-near.
        window: ZNCC aggregation window in pixels.
        min_views: minimum number of agreeing sources required for a valid pixel.

    Returns:
        :class:`DepthEstimate` with depth, cost, contributing view count and a
        validity mask.
    """
    if len(source_grays) == 0:
        raise ValueError("estimate_depth requires at least one source view")
    if len(planes) < 3:
        raise ValueError("estimate_depth requires at least three depth planes")

    K = ref_view.intrinsics.matrix
    K_inv = np.linalg.inv(K)
    ref = to_unit_float(ref_gray)
    height, width = ref.shape
    ref_mu, ref_var = zncc_stats(ref, window)

    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]] = []
    for view, image in source_grays:
        src = to_unit_float(image)
        if src.shape != ref.shape:
            src = cv2.resize(src, (width, height), interpolation=cv2.INTER_AREA)
        A = view.R @ ref_view.R.T
        b = view.t - A @ ref_view.t
        # K @ A @ K^-1 and K @ b let us build K[A + (1/d) b e3^T]K^-1 cheaply.
        KA = K @ A @ K_inv
        Kb = (K @ b).reshape(3)
        prepared.append((src, KA, Kb, src.shape[1], src.shape[0]))

    ones = np.ones((height, width), np.float32)
    best = np.full((height, width), np.inf, np.float32)
    best_index = np.zeros((height, width), np.int32)
    cost_prev = np.full((height, width), np.inf, np.float32)
    cost_next = np.full((height, width), np.inf, np.float32)
    views_at_best = np.zeros((height, width), np.float32)
    views_max = np.zeros((height, width), np.float32)
    last = np.full((height, width), np.inf, np.float32)

    for plane_index, depth in enumerate(planes):
        acc = np.zeros((height, width), np.float32)
        weight = np.zeros((height, width), np.float32)
        for src, KA, Kb, src_w, src_h in prepared:
            # Homography mapping reference pixels -> source pixels.
            H_ref_to_src = KA + np.outer(Kb, np.array([0.0, 0.0, 1.0])) / float(depth)
            try:
                M = np.linalg.inv(H_ref_to_src)
            except np.linalg.LinAlgError:
                continue
            warped = cv2.warpPerspective(
                src,
                M,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            inside = cv2.warpPerspective(
                ones,
                M,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            cost = _zncc_cost(ref, ref_mu, ref_var, warped, window)
            acc += cost * inside
            weight += inside

        current = np.divide(
            acc,
            np.maximum(weight, 1e-9),
            out=np.full((height, width), 2.0, np.float32),
            where=weight > 0,
        )

        improved = current < best
        # The new best's forward neighbour was scored one iteration ago.
        np.copyto(cost_next, current, where=(best_index == plane_index - 1))
        np.copyto(cost_prev, last, where=improved)
        np.copyto(best, current, where=improved)
        np.copyto(best_index, np.int32(plane_index), where=improved)
        np.copyto(views_at_best, weight, where=improved)
        last = current
        np.maximum(views_max, weight, out=views_max)

    # Fill unusable parabola neighbours so the refinement degrades to zero offset.
    np.copyto(cost_prev, best, where=~np.isfinite(cost_prev))
    np.copyto(cost_next, best, where=~np.isfinite(cost_next))

    depth = refine_depth(best_index, best, cost_prev, cost_next, planes)
    valid = (views_at_best >= float(min_views)) & np.isfinite(depth) & (views_max > 0)
    depth = np.where(valid, depth, 0.0).astype(np.float32)
    return DepthEstimate(
        depth=depth,
        cost=best.astype(np.float32),
        num_views=views_at_best.astype(np.uint8),
        valid=valid,
    )


def refine_depth(
    best_index: np.ndarray,
    best_cost: np.ndarray,
    cost_prev: np.ndarray,
    cost_next: np.ndarray,
    planes: np.ndarray,
) -> np.ndarray:
    """Sub-plane depth from a parabola through the cost minimum and its neighbours.

    The fit is performed in inverse depth, where cost is locally quadratic for a
    fronto-parallel plane sweep.
    """
    inv = 1.0 / np.asarray(planes, dtype=np.float64)
    step = inv[1] - inv[0]
    wi = inv[best_index]

    denom = cost_prev - 2.0 * best_cost + cost_next
    safe = np.abs(denom) > 1e-9
    offset = np.zeros_like(wi)
    offset[safe] = 0.5 * (cost_prev[safe] - cost_next[safe]) / denom[safe]
    np.clip(offset, -0.5, 0.5, out=offset)

    inv_depth = wi + offset * step
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = 1.0 / np.maximum(inv_depth, 1e-12)
    return depth


def apply_uncertainty_gate(
    estimate: DepthEstimate,
    focal_px: float,
    baseline: float,
    max_relative_sigma: float,
    disparity_sigma_px: float = 0.3,
) -> DepthEstimate:
    """Reject pixels whose predicted depth uncertainty exceeds a relative bound.

    ``sigma_z / z = z / (f b) * sigma_d``, so the gate is equivalent to a maximum
    usable range for the given baseline. Pixels beyond it carry no information:
    keeping them adds metres of error, not detail.
    """
    if max_relative_sigma <= 0 or baseline <= 0 or focal_px <= 0:
        return estimate
    depth = np.maximum(estimate.depth, 1e-9)
    sigma = depth * depth / (focal_px * abs(baseline)) * disparity_sigma_px
    valid = estimate.valid & (sigma <= max_relative_sigma * depth)
    return DepthEstimate(
        depth=np.where(valid, estimate.depth, 0.0).astype(np.float32),
        cost=estimate.cost,
        num_views=estimate.num_views,
        valid=valid,
    )


def apply_cost_threshold(estimate: DepthEstimate, threshold: float) -> DepthEstimate:
    """Drop pixels whose best ZNCC cost exceeds ``threshold``."""
    valid = estimate.valid & (estimate.cost <= float(threshold))
    depth = np.where(valid, estimate.depth, 0.0).astype(np.float32)
    return DepthEstimate(
        depth=depth,
        cost=estimate.cost,
        num_views=estimate.num_views,
        valid=valid,
    )
