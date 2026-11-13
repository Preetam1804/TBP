"""Multi-view geometric consistency and depth-discontinuity filtering.

Photometric cost alone accepts ambiguous matches (repeated facades, sky, flat
walls seen at grazing angles). Cross-checking each depth against the depth maps
of neighbouring views removes them before they can contaminate fusion.
"""

from __future__ import annotations

import cv2
import numpy as np

from uav3d.dense.cameras import CameraView
from uav3d.dense.sweep import DepthEstimate


def remove_depth_discontinuities(
    depth: np.ndarray,
    valid: np.ndarray,
    relative_threshold: float = 0.08,
) -> np.ndarray:
    """Invalidate pixels sitting on a depth edge.

    A 3x3 window whose depth range exceeds ``relative_threshold`` is treated as
    an occlusion boundary. Those pixels produce "flying" points that smear
    silhouettes, and they are cheap to drop.
    """
    if not np.any(valid):
        return valid
    d = np.where(valid, depth, np.nan).astype(np.float32)
    filled = np.where(np.isnan(d), np.nanmax(d) if np.isfinite(np.nanmax(d)) else 1.0, d)
    kernel = np.ones((3, 3), np.uint8)
    dmax = cv2.dilate(filled, kernel)
    dmin = cv2.erode(filled, kernel)
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = (dmax - dmin) / np.maximum(dmin, 1e-9)
    return valid & (spread <= float(relative_threshold)) & np.isfinite(spread)


def bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinearly sample ``image`` at scattered coordinates.

    Uses NumPy gather instead of ``cv2.remap`` because remap caps map dimensions
    at ``SHRT_MAX`` and a full-resolution view has more valid pixels than that.
    Out-of-bounds samples return 0.
    """
    height, width = image.shape
    flat = image.ravel()
    inside = (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
    out = np.zeros(u.shape, np.float32)
    if not np.any(inside):
        return out

    uu = u[inside]
    vv = v[inside]
    x0 = np.floor(uu).astype(np.int64)
    y0 = np.floor(vv).astype(np.int64)
    dx = (uu - x0).astype(np.float32)
    dy = (vv - y0).astype(np.float32)
    idx = y0 * width + x0
    top = flat[idx] * (1.0 - dx) + flat[idx + 1] * dx
    bottom = flat[idx + width] * (1.0 - dx) + flat[idx + width + 1] * dx
    out[inside] = top * (1.0 - dy) + bottom * dy
    return out


def cross_view_agreement(
    ref_view: CameraView,
    estimate: DepthEstimate,
    neighbour_views: list[CameraView],
    neighbour_depths: list[np.ndarray],
    tolerance: float,
) -> np.ndarray:
    """Count how many neighbouring depth maps agree with a reference depth map.

    A point is projected into each neighbour; the neighbour's depth at that pixel
    must match the projected depth within ``tolerance`` (relative).
    """
    height, width = estimate.depth.shape
    pixel_coords = ref_view.pixel_grid()
    uv = pixel_coords[estimate.valid.ravel()]
    depth = estimate.depth.ravel()[estimate.valid.ravel()].astype(np.float64)
    if uv.shape[0] == 0:
        return np.zeros((height, width), np.float32)

    points = ref_view.backproject(uv, depth)
    votes = np.zeros(uv.shape[0], np.float32)

    for view, neighbour_depth in zip(neighbour_views, neighbour_depths):
        if neighbour_depth is None:
            continue
        uv_n, z_n = view.project(points)
        inside = (
            (z_n > 0)
            & (uv_n[:, 0] >= 0)
            & (uv_n[:, 0] < view.intrinsics.width - 1)
            & (uv_n[:, 1] >= 0)
            & (uv_n[:, 1] < view.intrinsics.height - 1)
        )
        if not np.any(inside):
            continue
        sample = bilinear_sample(neighbour_depth, uv_n[:, 0], uv_n[:, 1])
        ok = inside & (sample > 0) & (np.abs(sample - z_n) / np.maximum(z_n, 1e-9) <= tolerance)
        votes += ok.astype(np.float32)

    out = np.zeros((height, width), np.float32)
    out.ravel()[estimate.valid.ravel()] = votes
    return out


def filter_estimate(
    ref_view: CameraView,
    estimate: DepthEstimate,
    neighbour_views: list[CameraView],
    neighbour_depths: list[np.ndarray],
    tolerance: float,
    min_agreements: int,
    drop_discontinuities: bool = True,
) -> DepthEstimate:
    """Apply geometric consistency, edge removal and the photometric threshold."""
    valid = estimate.valid.copy()
    if drop_discontinuities:
        valid &= remove_depth_discontinuities(estimate.depth, valid)

    if neighbour_views:
        votes = cross_view_agreement(
            ref_view,
            DepthEstimate(
                depth=np.where(valid, estimate.depth, 0.0).astype(np.float32),
                cost=estimate.cost,
                num_views=estimate.num_views,
                valid=valid,
            ),
            neighbour_views,
            neighbour_depths,
            tolerance,
        )
        valid &= votes >= float(min_agreements)

    depth = np.where(valid, estimate.depth, 0.0).astype(np.float32)
    return DepthEstimate(
        depth=depth,
        cost=estimate.cost,
        num_views=estimate.num_views,
        valid=valid,
    )
