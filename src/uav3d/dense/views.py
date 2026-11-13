"""Source-view selection for multi-view stereo.

The single most important decision in CPU MVS is *which* frames to compare.
Adjacent video frames typically have a baseline/depth ratio of a few percent,
which is far below the 0.05-0.3 range where stereo is well conditioned. Selecting
by baseline ratio (rather than by temporal adjacency) is what turns a marginal
sequence into usable geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from uav3d.dense.cameras import CameraView

# Baseline/depth ratio that maximises depth accuracy while keeping enough
# photometric overlap between views. Well conditioned stereo needs >= 0.05;
# beyond ~0.3 perspective distortion and occlusion start to dominate.
PREFERRED_BASELINE_RATIO = 0.10
BASELINE_RATIO_SPREAD = 0.09


@dataclass(frozen=True)
class SourceSelection:
    """A chosen source view with the quality terms used to pick it."""

    index: int
    score: float
    baseline_ratio: float
    overlap: float
    angle_deg: float


def predicted_overlap(ref: CameraView, other: CameraView, near: float, far: float) -> float:
    """Fraction of a reference-view sample grid that lands inside ``other``.

    The grid is back-projected at three depths spanning the view's depth range,
    so the score reflects the true 3-D frustum intersection rather than just the
    angle between optical axes.
    """
    k = 7
    # Sample pixel centres, inclusive of the last valid pixel, so a view always
    # scores 1.0 against itself.
    xs = np.linspace(0.0, ref.intrinsics.width - 1.0, k)
    ys = np.linspace(0.0, ref.intrinsics.height - 1.0, k)
    gx, gy = np.meshgrid(xs, ys)
    uv = np.stack([gx.ravel(), gy.ravel()], axis=1)
    total = 0
    inside = 0
    for depth in (near, 0.5 * (near + far), far):
        pts = ref.backproject(uv, np.full(uv.shape[0], depth))
        uv_o, z_o = other.project(pts)
        ok = (
            (z_o > 0)
            & (uv_o[:, 0] >= 0)
            & (uv_o[:, 0] < other.intrinsics.width)
            & (uv_o[:, 1] >= 0)
            & (uv_o[:, 1] < other.intrinsics.height)
        )
        inside += int(ok.sum())
        total += uv.shape[0]
    return inside / max(total, 1)


def view_angle_deg(ref: CameraView, other: CameraView) -> float:
    """Angle between two optical axes, in degrees."""
    cos = float(np.clip(ref.forward @ other.forward, -1.0, 1.0))
    return math.degrees(math.acos(cos))


def rank_sources(
    views: list[CameraView],
    ref_index: int,
    near: float,
    far: float,
    reference_depth: float,
    max_angle_deg: float,
    min_baseline_ratio: float,
    max_baseline_ratio: float,
    min_overlap: float,
    window: int,
    num_sources: int,
) -> list[SourceSelection]:
    """Rank candidate source views for one reference view.

    Score combines frustum overlap with a Gaussian preference for a
    baseline/depth ratio near :data:`PREFERRED_BASELINE_RATIO`.

    Ties are broken by index so the result is deterministic.
    """
    ref = views[ref_index]
    median_depth = reference_depth if reference_depth > 0 else 0.5 * (near + far)
    selected: list[SourceSelection] = []

    lo = max(0, ref_index - window)
    hi = min(len(views), ref_index + window + 1)
    for j in range(lo, hi):
        if j == ref_index:
            continue
        other = views[j]
        angle = view_angle_deg(ref, other)
        if angle > max_angle_deg:
            continue
        baseline = float(np.linalg.norm(ref.center - other.center))
        if baseline <= 0.0:
            continue
        ratio = baseline / median_depth
        if ratio < min_baseline_ratio or ratio > max_baseline_ratio:
            continue
        overlap = predicted_overlap(ref, other, near, far)
        if overlap < min_overlap:
            continue
        z = (ratio - PREFERRED_BASELINE_RATIO) / BASELINE_RATIO_SPREAD
        score = overlap * math.exp(-0.5 * z * z)
        selected.append(
            SourceSelection(
                index=j,
                score=score,
                baseline_ratio=ratio,
                overlap=overlap,
                angle_deg=angle,
            )
        )

    selected.sort(key=lambda s: (-s.score, s.index))
    return selected[:num_sources]


def median_baseline(views: list[CameraView], selections: dict[int, list[SourceSelection]]) -> float:
    """Median baseline over all selected source pairs, used for error modelling."""
    values: list[float] = []
    for ref_index, group in selections.items():
        ref = views[ref_index]
        for sel in group:
            values.append(float(np.linalg.norm(ref.center - views[sel.index].center)))
    if not values:
        return 0.0
    return float(np.median(values))


def median_depth_of(medians: dict[int, float]) -> float:
    """Median of the per-view median depths, in SfM units."""
    values = [value for value in medians.values() if value > 0]
    return float(np.median(values)) if values else 0.0
