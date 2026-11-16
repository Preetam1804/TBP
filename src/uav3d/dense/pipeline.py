"""Orchestration of the CPU dense reconstruction stage.

Pipeline:

1. load poses/keyframes, rectify images to one pinhole camera
2. derive a per-view depth range from the sparse SfM points
3. plane-sweep MVS per reference view (this is where depth actually comes from)
4. cross-view geometric consistency + depth-edge removal
5. weighted voxel fusion into a coloured point cloud
6. tiled level-set meshing
7. report, with the metric scale stated explicitly
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from uav3d.dense.cameras import (
    CameraView,
    Intrinsics,
    build_views,
    load_run_inputs,
    undistort_run_images,
)
from uav3d.dense.config import DenseConfig
from uav3d.dense.consistency import filter_estimate
from uav3d.dense.coverage import compute_coverage, write_coverage_report
from uav3d.dense.evaluate import cloud_metrics, estimate_point_spacing, load_ply_points
from uav3d.dense.fusion import (
    VoxelAccumulator,
    backproject_view,
    depth_normals,
    filter_density,
    transfer_vertex_colors,
    write_obj,
    write_ply,
)
from uav3d.dense.geometry import (
    clamp_near_for_plane_budget,
    depth_noise_sigma,
    depth_planes,
    load_sparse_points,
    parse_depth_range,
    plane_count_for_range,
    sparse_depth_range,
    umeyama_similarity,
    write_sparse_points,
)
from uav3d.dense.glb_export import write_glb
from uav3d.dense.mesh import (
    filter_unobserved_faces,
    orient_faces,
    reconstruct_mesh,
    weld_vertices,
)
from uav3d.dense.prior import (
    MonocularPriorModel,
    build_prior,
    edge_mask,
    fill_from_prior,
    sky_mask,
)
from uav3d.dense.sweep import (
    DepthEstimate,
    apply_cost_threshold,
    apply_uncertainty_gate,
    estimate_depth,
)
from uav3d.dense.views import (
    SourceSelection,
    median_baseline,
    median_depth_of,
    rank_sources,
)
from uav3d.exceptions import DataError, PipelineError
from uav3d.models.dense import (
    DenseDepthSummary,
    DenseFusionSummary,
    DenseGeometrySummary,
    DenseMeshSummary,
    DensePriorSummary,
    DenseReport,
    DenseScaleSummary,
)

logger = logging.getLogger("uav3d.dense.pipeline")


@dataclass
class DenseRunResult:
    """Artifacts and metrics produced by a dense run."""

    output_dir: Path
    point_cloud_path: Path
    mesh_path: Path | None
    report_path: Path
    report: DenseReport
    num_points: int
    num_faces: int
    total_seconds: float


class _ImageCache:
    """Small LRU cache of decoded rectified images (BGR uint8)."""

    def __init__(self, paths: list[Path], max_items: int = 48) -> None:
        self._paths = paths
        self._max = max_items
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, index: int) -> np.ndarray:
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        image = cv2.imread(str(self._paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise DataError(f"Failed to read rectified image: {self._paths[index]}")
        self._cache[index] = image
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)
        return image


def _load_or_build_rectified(
    views: list[CameraView],
    output_dir: Path,
    target_width: int,
) -> tuple[list[Path], Intrinsics]:
    """Return cached rectified image paths and the shared pinhole intrinsics.

    The cache is only reused when the stored geometry still matches the
    intrinsics the current run's poses imply. After an SfM re-run (a
    different solved focal, a different working width) the cached
    ``geometry.json`` is stale: reusing it would silently run the whole
    dense stage — plane sweep and fusion — with the previous run's focal
    while the summary reports the new one. On mismatch the cache is
    rebuilt from the current views.
    """
    geometry_path = output_dir / "geometry.json"
    expected = [output_dir / f"{view.index:05d}.png" for view in views]
    if geometry_path.exists() and all(p.exists() for p in expected):
        data = json.loads(geometry_path.read_text(encoding="utf-8"))
        cached = Intrinsics(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )
        current = views[0].intrinsics
        if (
            abs(cached.fx - current.fx) <= 1e-6
            and abs(cached.fy - current.fy) <= 1e-6
            and abs(cached.cx - current.cx) <= 1e-6
            and abs(cached.cy - current.cy) <= 1e-6
            and (cached.width, cached.height) == (current.width, current.height)
        ):
            for view in views:
                view.intrinsics = cached
            return expected, cached
        logger.warning(
            "Cached rectified geometry (f=%.2f) does not match the current "
            "run's intrinsics (f=%.2f); rebuilding the rectified images.",
            cached.fx,
            current.fx,
        )

    paths = undistort_run_images(views, output_dir, target_width=target_width, overwrite=True)
    intr = views[0].intrinsics
    for view in views[1:]:
        if (view.intrinsics.width, view.intrinsics.height) != (intr.width, intr.height):
            raise PipelineError("Rectified images do not share one resolution; check input frames")
    geometry_path.write_text(
        json.dumps(
            {
                "fx": intr.fx,
                "fy": intr.fy,
                "cx": intr.cx,
                "cy": intr.cy,
                "width": intr.width,
                "height": intr.height,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths, intr


def _latlon_to_enu(lat: float, lon: float, alt: float, origin: tuple[float, float, float]) -> np.ndarray:
    """Local east/north/up offsets in metres for a short-baseline site."""
    radius = 6378137.0
    lat0, lon0, alt0 = origin
    east = math.radians(lon - lon0) * radius * math.cos(math.radians(lat0))
    north = math.radians(lat - lat0) * radius
    return np.array([east, north, alt - alt0], dtype=np.float64)


def _resolve_metric_scale(
    config: DenseConfig,
    run_path: Path,
    views: list[CameraView],
) -> tuple[float, str, str, dict[str, float]]:
    """Determine metres per SfM unit, and say honestly where it came from."""
    if config.scale_source == "fixed":
        if config.scale_factor is None:
            raise DataError("scale_source='fixed' requires scale_factor")
        return float(config.scale_factor), "fixed", "fixed_by_user", {}

    if config.scale_source == "gnss":
        telemetry_path = run_path / "telemetry.json"
        if not telemetry_path.exists():
            raise DataError(
                "scale_source='gnss' requires telemetry.json in the run directory "
                "(produced by 'uav3d sync')."
            )
        data = json.loads(telemetry_path.read_text(encoding="utf-8"))
        records = {r.get("frame_id"): r for r in data.get("records", []) if r.get("frame_id")}
        src: list[np.ndarray] = []
        dst: list[np.ndarray] = []
        origin: tuple[float, float, float] | None = None
        for view in views:
            rec = records.get(view.frame_id)
            if rec is None or rec.get("lat") is None or rec.get("lon") is None:
                continue
            lat, lon = float(rec["lat"]), float(rec["lon"])
            alt = float(rec.get("alt") or 0.0)
            if origin is None:
                origin = (lat, lon, alt)
            src.append(view.center)
            dst.append(_latlon_to_enu(lat, lon, alt, origin))
        if len(src) < 10:
            raise DataError(
                f"Only {len(src)} frames have GPS fixes; need at least 10 to establish metric scale."
            )
        scale, _, _ = umeyama_similarity(np.asarray(src), np.asarray(dst))
        predicted = np.asarray(src) * scale
        residual = float(np.sqrt(np.mean(np.sum((predicted - np.asarray(dst)) ** 2, axis=1))))
        logger.info("GNSS scale factor %.4f m/unit (alignment RMS %.2f m)", scale, residual)
        return float(scale), "gnss", "gnss_aligned", {
            "gnss_alignment_rms_m": residual,
            "gnss_frames": float(len(src)),
        }

    return 1.0, "none", "unverified_monocular_scale", {}


def _per_view_ranges(
    config: DenseConfig,
    views: list[CameraView],
    sparse_points: np.ndarray | None,
) -> tuple[dict[int, tuple[float, float]], dict[int, float], list[int]]:
    """Compute a depth range for every view that will become a depth reference."""
    ref_indices = list(range(0, len(views), config.ref_stride))
    if config.max_reference_views:
        ref_indices = ref_indices[: config.max_reference_views]

    if config.depth_range:
        near, far = parse_depth_range(config.depth_range)
        logger.info("Using manual depth range [%.2f, %.2f] for all views", near, far)
        median = float(np.sqrt(near * far))
        return {i: (near, far) for i in ref_indices}, {i: median for i in ref_indices}, ref_indices

    if sparse_points is None:
        raise DataError(
            "No sparse point cloud found for this run, so the depth range cannot be derived.\n"
            "Fix this in one of two ways:\n"
            "  1. Re-run 'uav3d sfm' so the COLMAP model is written, or\n"
            "  2. pass --depth-range MIN:MAX (SfM units, e.g. '8:200')."
        )

    ranges: dict[int, tuple[float, float]] = {}
    medians: dict[int, float] = {}
    fallback: tuple[float, float, float] | None = None
    for index in ref_indices:
        try:
            near, far, median, _ = sparse_depth_range(
                views[index],
                sparse_points,
                config.depth_percentile_low,
                config.depth_percentile_high,
                config.depth_margin,
                config.depth_near_min_ratio,
            )
            ranges[index] = (near, far)
            medians[index] = median
        except DataError as exc:
            if fallback is None:
                fallback = _global_depth_range(config, views[0], sparse_points)
                logger.warning("Using a global depth range for some views: %s", exc)
            ranges[index] = (fallback[0], fallback[1])
            medians[index] = fallback[2]
    return ranges, medians, ref_indices


def _global_depth_range(
    config: DenseConfig,
    view: CameraView,
    sparse_points: np.ndarray,
) -> tuple[float, float, float]:
    """Fallback depth range from every sparse point in front of one camera."""
    uv, z = view.project(sparse_points)
    inside = (
        (z > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < view.intrinsics.width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < view.intrinsics.height)
    )
    zz = z[inside]
    if zz.size < 20:
        raise DataError("Sparse points do not project into the image; cannot derive a depth range")
    lo = float(np.percentile(zz, config.depth_percentile_low)) * (1.0 - config.depth_margin)
    hi = float(np.percentile(zz, config.depth_percentile_high)) * (1.0 + config.depth_margin)
    median = float(np.median(zz))
    lo = max(lo, float(config.depth_near_min_ratio) * median)
    return max(lo, 1e-6), max(hi, lo * 1.2), max(median, 1e-6)


def _select_sources(
    config: DenseConfig,
    views: list[CameraView],
    ref_indices: list[int],
    ranges: dict[int, tuple[float, float]],
    medians: dict[int, float],
) -> dict[int, list[SourceSelection]]:
    selections: dict[int, list[SourceSelection]] = {}
    for index in ref_indices:
        near, far = ranges[index]
        chosen = rank_sources(
            views=views,
            ref_index=index,
            near=near,
            far=far,
            reference_depth=medians[index],
            max_angle_deg=config.max_view_angle_deg,
            min_baseline_ratio=config.min_baseline_ratio,
            max_baseline_ratio=config.max_baseline_ratio,
            min_overlap=config.min_overlap,
            window=config.source_window,
            num_sources=config.num_sources,
        )
        if not chosen:
            logger.warning(
                "View %d (%s) has no usable source views; its depth map will be skipped",
                index,
                views[index].frame_id,
            )
        selections[index] = chosen
    return selections


def _save_depth(path: Path, depth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, depth.astype(np.float16))


def _load_depth(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def _depth_path(depth_dir: Path, index: int) -> Path:
    return depth_dir / f"{index:05d}.npy"


def _neighbour_views(views: list[CameraView], ref_indices: list[int], index: int, count: int) -> list[int]:
    """The ``count`` reference views whose camera centres are closest to ``index``."""
    centre = views[index].center
    candidates = [j for j in ref_indices if j != index]
    candidates.sort(key=lambda j: (float(np.linalg.norm(views[j].center - centre)), j))
    return candidates[:count]


def run_dense(
    run_dir: str | Path,
    config: DenseConfig | None = None,
    output_dir: str | Path | None = None,
    ground_truth_ply: str | Path | None = None,
) -> DenseRunResult:
    """Run the full CPU dense reconstruction on a completed SP-1 run directory."""
    started = time.time()
    config = config or DenseConfig()
    run_path = Path(run_dir).resolve()
    out_dir = Path(output_dir).resolve() if output_dir else run_path / "dense"
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_seconds: dict[str, float] = {}

    # --- 1. inputs ---------------------------------------------------------
    t0 = time.time()
    poses, keyframes = load_run_inputs(run_path)
    views = build_views(poses, keyframes, run_path, config.max_width)
    rectified_paths, intrinsics = _load_or_build_rectified(views, out_dir / "rectified", config.max_width)
    images = _ImageCache(rectified_paths)
    stage_seconds["prepare"] = time.time() - t0
    logger.info(
        "%d registered views at %dx%d (f=%.1f px)",
        len(views),
        intrinsics.width,
        intrinsics.height,
        intrinsics.focal,
    )

    # --- 2. depth ranges + scale ------------------------------------------
    t0 = time.time()
    sparse_points = load_sparse_points(run_path)
    if sparse_points is not None:
        write_sparse_points(sparse_points, run_path)
    ranges, medians, ref_indices = _per_view_ranges(config, views, sparse_points)
    selections = _select_sources(config, views, ref_indices, ranges, medians)
    usable = [i for i in ref_indices if selections[i]]
    if not usable:
        raise PipelineError(
            "No reference view found a usable source view. The flight may be too sparse for "
            "stereo (check min/max baseline ratio and overlap settings)."
        )
    median_base = median_baseline(views, selections)
    median_depth = median_depth_of(medians)
    ratio = median_base / median_depth if median_depth > 0 else 0.0
    sigma = depth_noise_sigma(median_depth, intrinsics.focal, median_base)
    scale, scale_source, scale_status, scale_extra = _resolve_metric_scale(config, run_path, views)
    stage_seconds["view_selection"] = time.time() - t0

    logger.info(
        "baseline %.3f units, median depth %.2f units, ratio %.3f, depth sigma %.4f units",
        median_base,
        median_depth,
        ratio,
        sigma,
    )
    if ratio < 0.03:
        logger.warning(
            "Median baseline/depth ratio is %.3f. Stereo is weak below 0.05, so depth will be "
            "noisy: increase keyframe spacing (keyframes.stride) or fly closer to the subject.",
            ratio,
        )

    # --- 2b. monocular prior prepass (opt-in) ------------------------------
    # A failure here must never fail the run: the prior is an additive fill, so
    # any problem degrades to "no prior" with a warning.
    prior_maps: dict[int, np.ndarray] = {}
    prior_summary: DensePriorSummary | None = None
    if config.prior_enabled:
        t0 = time.time()
        if sparse_points is None:
            logger.warning(
                "Depth prior enabled but no sparse SfM points are available; the prior "
                "cannot be aligned to the SfM frame and is being skipped"
            )
        else:
            try:
                prior_model = MonocularPriorModel(
                    config.prior_model_path,
                    infer_height=config.prior_infer_height,
                )
            except Exception as exc:  # noqa: BLE001 - the prior must never fail the run
                logger.warning("Depth prior unavailable (%s); continuing without it", exc)
            else:
                per_view: dict[str, dict[str, float | bool | None]] = {}
                for index in usable:
                    near, far = ranges[index]
                    try:
                        depth, _scale, _shift, corr, err, inverted = build_prior(
                            prior_model,
                            views[index],
                            images.get(index),
                            sparse_points,
                            near,
                            far,
                            min_rank_corr=config.prior_min_rank_corr,
                            max_relative_error=config.prior_max_relative_error,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Prior failed for view %s: %s", views[index].frame_id, exc)
                        continue
                    accepted = bool(np.any(depth > 0.0) and np.isfinite(err))
                    if accepted:
                        prior_maps[index] = depth
                    per_view[views[index].frame_id] = {
                        "accepted": accepted,
                        "corr": float(corr) if np.isfinite(corr) else None,
                        "error": float(err) if np.isfinite(err) else None,
                        "inverted": bool(inverted),
                    }
                accepted_views = [m for m in per_view.values() if m["accepted"]]
                prior_report = {
                    "model": str(config.prior_model_path),
                    "views_total": len(usable),
                    "views_accepted": len(prior_maps),
                    "median_corr": (
                        float(np.median([m["corr"] for m in accepted_views]))
                        if accepted_views
                        else None
                    ),
                    "median_error": (
                        float(np.median([m["error"] for m in accepted_views]))
                        if accepted_views
                        else None
                    ),
                    "views": per_view,
                }
                prior_path = out_dir / "prior_report.json"
                prior_path.write_text(json.dumps(prior_report, indent=2), encoding="utf-8")
                prior_summary = DensePriorSummary(
                    enabled=True,
                    model_path=str(config.prior_model_path),
                    views_total=len(usable),
                    views_accepted=len(prior_maps),
                    median_corr=prior_report["median_corr"],
                    median_error=prior_report["median_error"],
                )
                stage_seconds["prior"] = time.time() - t0
                if not prior_maps:
                    logger.warning(
                        "Depth prior accepted no views (min_rank_corr=%.2f, "
                        "max_relative_error=%.2f); continuing without prior fill. "
                        "Details in %s",
                        config.prior_min_rank_corr,
                        config.prior_max_relative_error,
                        prior_path,
                    )
                else:
                    logger.info(
                        "Depth prior accepted %d/%d views (median corr %.3f, error %.3f)",
                        len(prior_maps),
                        len(usable),
                        prior_report["median_corr"],
                        prior_report["median_error"],
                    )

    # --- 3. plane-sweep MVS ------------------------------------------------
    t0 = time.time()
    depth_dir = out_dir / "depth"
    in_memory: dict[int, np.ndarray] = {}
    total_pixels = 0
    valid_pixels = 0
    cost_samples: list[float] = []
    plane_counts: list[int] = []
    runtime_per_view: list[float] = []
    depth_limits: dict[int, tuple[float, float]] = {}

    for order, index in enumerate(usable):
        nearest, farthest = ranges[index]
        chosen = selections[index]
        baselines = [float(np.linalg.norm(views[index].center - views[s.index].center)) for s in chosen]
        baseline = float(np.median(baselines)) if baselines else median_base
        clamped = clamp_near_for_plane_budget(
            nearest,
            farthest,
            intrinsics.focal,
            baseline,
            config.plane_step_px,
            config.max_planes,
            relax=config.depth_near_plane_relax,
        )
        if clamped > nearest * (1.0 + 1e-4):
            logger.warning(
                "View %s: depth range [%.3f, %.3f] needs more than %d planes at "
                "%.2f px step; raising near to %.3f so the sweep keeps its step.",
                views[index].frame_id,
                nearest,
                farthest,
                config.max_planes,
                config.plane_step_px,
                clamped,
            )
            nearest = clamped
        # The clamped bounds are what the sweep actually used; the prior fill
        # must gate on the same range, not on the pre-clamp sparse percentiles.
        depth_limits[index] = (nearest, farthest)
        count = plane_count_for_range(
            nearest,
            farthest,
            intrinsics.focal,
            baseline,
            config.plane_step_px,
            config.min_planes,
            config.max_planes,
        )
        planes = depth_planes(nearest, farthest, count)

        ref_gray = cv2.cvtColor(images.get(index), cv2.COLOR_BGR2GRAY)
        source_grays = [
            (views[s.index], cv2.cvtColor(images.get(s.index), cv2.COLOR_BGR2GRAY)) for s in chosen
        ]

        view_started = time.time()
        estimate = estimate_depth(
            ref_gray,
            views[index],
            source_grays,
            planes,
            window=config.window,
            min_views=config.min_views,
        )
        estimate = apply_cost_threshold(estimate, config.cost_threshold)
        estimate = apply_uncertainty_gate(
            estimate, intrinsics.focal, baseline, config.max_relative_sigma
        )
        view_seconds = time.time() - view_started
        runtime_per_view.append(view_seconds)

        _save_depth(_depth_path(depth_dir, index), estimate.depth)
        if config.keep_depth_in_memory:
            in_memory[index] = estimate.depth

        total_pixels += estimate.depth.size
        valid_pixels += int(estimate.valid.sum())
        cost_samples.append(
            float(np.median(estimate.cost[estimate.valid])) if bool(estimate.valid.any()) else 2.0
        )
        plane_counts.append(count)

        done = order + 1
        projected_min = float(np.median(runtime_per_view)) * (len(usable) - done) / 60.0
        logger.info(
            "[%d/%d] %s depth: %d planes, %.1f%% valid, %.1fs (projected %.1f min remaining)",
            done,
            len(usable),
            views[index].frame_id,
            count,
            100.0 * float(estimate.valid.mean()),
            view_seconds,
            projected_min,
        )
    stage_seconds["mvs"] = time.time() - t0

    # --- 4. consistency filtering -----------------------------------------
    t0 = time.time()
    filtered_dir = out_dir / "depth_filtered"
    filtered_in_memory: dict[int, np.ndarray] = {}
    for index in usable:
        raw = in_memory.get(index)
        if raw is None:
            raw = _load_depth(_depth_path(depth_dir, index))
        estimate = DepthEstimate(
            depth=raw,
            cost=np.zeros_like(raw),
            num_views=np.zeros_like(raw, dtype=np.uint8),
            valid=raw > 0,
        )
        neighbours = _neighbour_views(views, usable, index, config.consistency_views)
        neighbour_views: list[CameraView] = []
        neighbour_depths: list[np.ndarray] = []
        for j in neighbours:
            d = in_memory.get(j)
            if d is None:
                d = _load_depth(_depth_path(depth_dir, j))
            if d.shape != raw.shape:
                continue
            neighbour_depths.append(d)
            neighbour_views.append(views[j])
        filtered = filter_estimate(
            views[index],
            estimate,
            neighbour_views,
            neighbour_depths,
            tolerance=config.consistency_tol,
            min_agreements=config.consistency_views,
        )
        _save_depth(_depth_path(filtered_dir, index), filtered.depth)
        filtered_in_memory[index] = filtered.depth
    stage_seconds["consistency"] = time.time() - t0

    # --- 4b. prior fill (opt-in) -------------------------------------------
    # The fill happens AFTER the consistency filter: filled pixels only exist
    # to replace MVS holes, and if they were added before the filter,
    # cross_view_agreement would reject them against neighbour maps that have
    # the same holes. fill_from_prior never touches measured (valid) pixels.
    prior_filled = 0
    prior_filled_by_view: dict[int, int] = {}
    if prior_maps:
        t0 = time.time()
        views_filled = 0
        for index in usable:
            if index not in prior_maps:
                continue
            depth = filtered_in_memory.get(index)
            if depth is None:
                depth = _load_depth(_depth_path(filtered_dir, index))
            estimate = DepthEstimate(
                depth=depth,
                cost=np.zeros_like(depth),
                num_views=(depth > 0).astype(np.uint8),
                valid=depth > 0,
            )
            neighbours = [
                j
                for j in _neighbour_views(views, usable, index, config.consistency_views)
                if j in prior_maps
            ]
            near, far = depth_limits.get(index, ranges[index])
            # Sky pixels are a prior-fill no-go zone: a monocular model gives
            # the sky a smooth pseudo-depth that neighbouring priors agree on,
            # so without this guard a horizon becomes a solid far-plane sheet.
            avoid = edge_mask(images.get(index), config.prior_edge_guard_px)
            if config.prior_sky_guard:
                avoid = avoid | sky_mask(
                    images.get(index),
                    max_row_fraction=config.prior_sky_max_row_fraction,
                )
            filled_estimate, filled = fill_from_prior(
                estimate,
                prior_maps[index],
                views[index],
                [views[j] for j in neighbours],
                [prior_maps[j] for j in neighbours],
                near,
                far,
                min_support=config.prior_min_support,
                tolerance=config.prior_tolerance,
                avoid_edges=avoid,
            )
            if filled:
                filtered_in_memory[index] = filled_estimate.depth
                _save_depth(_depth_path(filtered_dir, index), filled_estimate.depth)
                prior_filled += filled
                prior_filled_by_view[index] = filled
                views_filled += 1
        stage_seconds["prior_fill"] = time.time() - t0
        logger.info(
            "Prior fill added %d depth pixels in %d views", prior_filled, views_filled
        )
    if prior_summary is not None:
        # Count coverage from the final depth maps themselves: the consistency
        # filter rejects some MVS pixels, which the fill may then re-add as
        # prior values, so stage counters must not be summed.
        final_valid = 0
        for index in usable:
            depth = filtered_in_memory.get(index)
            if depth is None:
                depth = _load_depth(_depth_path(filtered_dir, index))
            final_valid += int((depth > 0).sum())
        prior_summary.pixels_filled = prior_filled
        prior_summary.pixels_measured = final_valid - prior_filled
        prior_summary.final_valid_fraction = float(final_valid / max(total_pixels, 1))
        logger.info(
            "Final depth coverage %.1f%% (%d measured + %d prior-filled of %d pixels)",
            100.0 * prior_summary.final_valid_fraction,
            prior_summary.pixels_measured,
            prior_summary.pixels_filled,
            total_pixels,
        )

    # --- 5. fusion ---------------------------------------------------------
    t0 = time.time()
    voxel_units = _resolve_voxel(config, sigma, median_depth, config.num_sources, scale)
    trunc_units = config.trunc_sigmas * max(sigma, 1e-9)
    accumulator = VoxelAccumulator(voxel=voxel_units)
    used_views = 0
    for index in usable:
        depth = filtered_in_memory.get(index)
        if depth is None:
            depth = _load_depth(_depth_path(filtered_dir, index))
        valid = depth > 0
        if not np.any(valid):
            continue
        normals = depth_normals(views[index], depth, valid)
        points, colors, nrm = backproject_view(
            views[index], depth, valid, images.get(index), normals
        )
        if points.shape[0] == 0:
            continue
        accumulator.add(points, colors, nrm)
        used_views += 1
        logger.info(
            "[fusion] %d/%d views, %d points buffered", used_views, len(usable), accumulator.buffered
        )
    cloud = accumulator.to_cloud()
    if len(cloud) == 0:
        raise PipelineError("Fusion produced an empty point cloud; every depth pixel was rejected")
    keep = filter_density(cloud.points, voxel_units, min_count=2 if used_views > 2 else 1)
    points_unscaled = cloud.points[keep]
    colors = cloud.colors[keep]
    normals = cloud.normals[keep] if cloud.normals is not None else None
    stage_seconds["fusion"] = time.time() - t0

    # --- 6. metric scale + export -----------------------------------------
    points = points_unscaled * scale if scale != 1.0 else points_unscaled
    voxel_m = voxel_units * scale if scale_source != "none" else None
    spacing_m = estimate_point_spacing(points) if scale_source != "none" else None
    cloud_path = write_ply(out_dir / "cloud.ply", points, colors, normals)

    # --- 7. meshing --------------------------------------------------------
    mesh_path: Path | None = None
    glb_path: Path | None = None
    mesh_summary: DenseMeshSummary | None = None
    t0 = time.time()
    if config.mesh and normals is not None and points.shape[0] >= 5000:
        # Mesh resolution must be coarser than the point spacing: with ~1 point
        # per voxel the occupancy grid is ragged and the level set fragments.
        spacing = estimate_point_spacing(points)
        mesh_voxel = max(config.mesh_voxel_fraction * voxel_units * scale, 4.0 * spacing)
        tile = (
            config.mesh_tile_units * scale
            if (config.mesh_tile_units and scale_source != "none")
            else _default_tile(points, mesh_voxel)
        )
        try:
            mesh = reconstruct_mesh(
                points,
                normals,
                voxel=mesh_voxel,
                tile_units=tile,
                max_tile_voxels=config.mesh_max_tile_voxels,
            )
            # Marching tetrahedra emits arbitrary winding; orientation must be
            # resolved against the measured normals before the visibility filter,
            # which reads face winding.
            mesh = orient_faces(mesh, points, normals)
            centres = np.array([v.center for v in views], dtype=np.float64)
            mesh = filter_unobserved_faces(mesh, centres)
            mesh = weld_vertices(mesh, tolerance=voxel_units * 1e-3)
            # Colour the mesh from the fused cloud so the export carries the
            # captured surface appearance instead of one flat default material.
            mesh_colors = transfer_vertex_colors(mesh.vertices, points, colors)
            mesh_path = write_obj(
                out_dir / "mesh.obj", mesh.vertices, mesh.faces, mesh_colors
            )
            # Also emit a GLB with normals and a material. trimesh's exporter
            # omits NORMAL when the mesh has none, which makes every viewer
            # shade the surface flat and unreadable.
            glb_path = write_glb(
                out_dir / "mesh.glb",
                mesh.vertices,
                mesh.faces,
                mesh_colors,
                name="dense_mesh",
            )
            logger.info("[mesh] wrote %s", glb_path)
            mesh_summary = DenseMeshSummary(
                num_vertices=mesh.num_vertices,
                num_faces=mesh.num_faces,
                voxel_units=mesh_voxel / scale,
            )
        except (PipelineError, MemoryError) as exc:
            logger.warning("Meshing skipped: %s", exc)
    stage_seconds["mesh"] = time.time() - t0

    # --- 8. evaluation -----------------------------------------------------
    evaluation = None
    if ground_truth_ply is not None:
        t0 = time.time()
        gt = load_ply_points(ground_truth_ply)
        gt_sfm = gt / scale if scale != 1.0 else gt
        evaluation = cloud_metrics(
            points_unscaled,
            gt_sfm,
            views=views,
            thresholds=(0.5, 1.0, 2.0),
            distance_scale=scale,
        )
        evaluation.update(scale_extra)
        stage_seconds["evaluation"] = time.time() - t0

    # --- 8b. coverage ------------------------------------------------------
    # Reporting only: a failure here must never discard a finished dense
    # run, so it degrades to "no coverage report" with a warning.
    coverage: dict | None = None
    coverage_path: Path | None = None
    t0 = time.time()
    try:
        coverage = compute_coverage(
            out_dir,
            scale=scale,
            scale_source=scale_source,
            scale_verified=scale_source != "none",
            cell_size_m=config.coverage_cell_size_m,
            thin_fraction=config.coverage_thin_fraction,
            hole_fraction_warn=config.coverage_hole_fraction_warn,
            hole_fraction_bad=config.coverage_hole_fraction_bad,
            thin_fraction_bad=config.coverage_thin_fraction_bad,
            prior_enabled=bool(prior_summary is not None and prior_summary.enabled),
            prior_pixels_filled=prior_summary.pixels_filled if prior_summary else None,
            prior_pixels_measured=prior_summary.pixels_measured if prior_summary else None,
            per_view_filled_pixels=prior_filled_by_view or None,
            voxel_units=voxel_units,
        )
        coverage_path = write_coverage_report(out_dir, coverage)
        logger.info(
            "Coverage verdict %s: %.1f%% of %d footprint cells empty, %.1f%% thin",
            coverage["verdict"],
            100.0 * coverage["grid"]["empty_fraction"],
            coverage["grid"]["num_cells"],
            100.0 * coverage["grid"]["thin_fraction"],
        )
    except Exception as exc:  # noqa: BLE001 - coverage is reporting, not reconstruction
        logger.warning("Coverage reporting failed (%s); continuing without it", exc)
    stage_seconds["coverage"] = time.time() - t0

    # --- 9. report ---------------------------------------------------------
    artifacts = {"cloud": str(cloud_path)}
    if coverage_path is not None:
        artifacts["coverage_report"] = str(coverage_path)
    if mesh_path is not None:
        artifacts["mesh"] = str(mesh_path)
    if glb_path is not None:
        artifacts["glb"] = str(glb_path)
    if prior_summary is not None:
        artifacts["prior_report"] = str(out_dir / "prior_report.json")
    report = DenseReport(
        run_dir=str(run_path),
        dataset_label=poses.dataset_label,
        created_utc=datetime.now(UTC).isoformat(),
        geometry=DenseGeometrySummary(
            num_registered_views=len(views),
            num_reference_views=len(usable),
            working_width=intrinsics.width,
            working_height=intrinsics.height,
            focal_px=intrinsics.focal,
            median_baseline=median_base,
            median_depth=median_depth,
            median_baseline_ratio=ratio,
            depth_range=(
                float(min(near for near, _ in ranges.values())),
                float(max(far for _, far in ranges.values())),
            ),
        ),
        depth=DenseDepthSummary(
            total_pixels=total_pixels,
            valid_pixels=valid_pixels,
            valid_fraction=float(np.clip(valid_pixels / max(total_pixels, 1), 0.0, 1.0)),
            plane_count_min=int(min(plane_counts)) if plane_counts else 0,
            plane_count_max=int(max(plane_counts)) if plane_counts else 0,
            median_cost=float(np.median(cost_samples)) if cost_samples else 0.0,
        ),
        fusion=DenseFusionSummary(
            voxel_units=voxel_units,
            voxel_meters=voxel_m,
            trunc_units=trunc_units,
            depth_sigma_units=sigma,
            num_points=int(points.shape[0]),
            point_spacing_meters=spacing_m,
        ),
        mesh=mesh_summary,
        prior=prior_summary,
        scale=DenseScaleSummary(value=scale, source=scale_source, status=scale_status),
        evaluation=evaluation,
        coverage=coverage,
        stage_seconds={k: round(v, 3) for k, v in stage_seconds.items()},
        total_seconds=round(time.time() - started, 3),
        artifacts=artifacts,
    )
    report_path = out_dir / "dense_report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    _write_markdown(out_dir / "DENSE_REPORT.md", report)

    return DenseRunResult(
        output_dir=out_dir,
        point_cloud_path=cloud_path,
        mesh_path=mesh_path,
        report_path=report_path,
        report=report,
        num_points=int(points.shape[0]),
        num_faces=int(mesh_summary.num_faces) if mesh_summary else 0,
        total_seconds=report.total_seconds,
    )


def _resolve_voxel(
    config: DenseConfig,
    sigma: float,
    median_depth: float,
    num_sources: int,
    scale: float,
) -> float:
    """Fusion voxel from the depth noise model (in SfM units).

    Averaging N views reduces the random part of depth noise by sqrt(N), so the
    fused resolution is ``sigma / sqrt(N)``. Going finer than the measurement
    noise would encode noise as if it were detail.
    """
    if config.voxel_m is not None:
        return float(config.voxel_m) / max(scale, 1e-12)
    fused_sigma = sigma / math.sqrt(max(1, min(num_sources, 6)))
    voxel = min(max(fused_sigma, 0.3 * sigma), 1.0 * sigma)
    voxel = min(voxel, max(median_depth / 300.0, 1e-3))
    return float(max(voxel, 1e-3))


def _default_tile(points: np.ndarray, voxel: float) -> float:
    """Tile size for meshing: about a quarter of the largest extent."""
    if points.shape[0] == 0:
        return 32.0 * voxel
    extent = np.percentile(points, 99.5, axis=0) - np.percentile(points, 0.5, axis=0)
    return float(max(extent.max() / 4.0, 8.0 * voxel))


def _write_markdown(path: Path, report: DenseReport) -> None:
    """Human-readable summary of a dense run."""
    backend_note = "CPU multi-view stereo"
    if report.prior is not None and report.prior.enabled:
        backend_note += ", monocular prior fill"
    else:
        backend_note += ", no monocular prior"
    lines = [
        "# Dense reconstruction report",
        "",
        f"- Run: `{report.run_dir}`",
        f"- Backend: `{report.backend}` ({backend_note})",
        f"- Views: {report.geometry.num_registered_views} registered, "
        f"{report.geometry.num_reference_views} used as depth references",
        f"- Working resolution: {report.geometry.working_width}x{report.geometry.working_height}",
        f"- Median baseline/depth: {report.geometry.median_baseline_ratio:.3f}",
        f"- Depth pixels valid: {report.depth.valid_fraction * 100:.1f}%",
        f"- Fused points: {report.fusion.num_points}",
        f"- Fusion voxel: {report.fusion.voxel_units:.4f} units"
        + (f" ({report.fusion.voxel_meters:.3f} m)" if report.fusion.voxel_meters else ""),
        f"- Depth sigma: {report.fusion.depth_sigma_units:.4f} units",
        f"- Metric scale: {report.scale.value:.4f} m/unit ({report.scale.status})",
    ]
    if report.mesh is not None:
        lines.append(f"- Mesh: {report.mesh.num_vertices} vertices, {report.mesh.num_faces} triangles")
    if report.prior is not None and report.prior.enabled:
        lines.append(
            f"- Prior: {report.prior.views_accepted}/{report.prior.views_total} views accepted, "
            f"{report.prior.pixels_filled:,} pixels filled from the monocular prior "
            f"(inferred, not measured; measured in final maps: {report.prior.pixels_measured:,}); "
            f"final coverage {report.prior.final_valid_fraction * 100:.1f}%; see prior_report.json"
        )
    if report.coverage:
        cov = report.coverage
        summary = cov.get("per_view_summary") or {}
        fused = cov.get("fused") or {}
        grid = cov.get("grid") or {}
        unit = cov.get("unit") or "SfM units"
        area_unit = "m^2" if unit == "m" else "SfM unit^2"
        lines += [
            "",
            "## Coverage",
            "",
            f"- Verdict: **{cov.get('verdict')}**"
            + (f" — {'; '.join(cov.get('verdict_reasons') or [])}" if cov.get("verdict_reasons") else ""),
            f"- Per-view valid depth (final maps): "
            f"{100.0 * (summary.get('min_valid_fraction') or 0):.1f}%–"
            f"{100.0 * (summary.get('max_valid_fraction') or 0):.1f}% "
            f"(median {100.0 * (summary.get('median_valid_fraction') or 0):.1f}% over "
            f"{summary.get('num_views')} views)"
            + ("; final maps include prior-filled (inferred) pixels" if (cov.get("prior") or {}).get("enabled") else "; all valid pixels are measured"),
            f"- Fused density: {fused.get('points_per_unit_area', 0):,.0f} points/{area_unit} "
            f"over {fused.get('footprint_area', 0):,.1f} {area_unit} "
            f"({fused.get('num_points', 0):,} points; footprint on the PCA PC1–PC2 plane, "
            f"units: {unit})",
            f"- Grid ({grid.get('cell_size')} {unit} cells): "
            f"{grid.get('empty_cells')} of {grid.get('num_cells')} footprint cells empty "
            f"({100.0 * (grid.get('empty_fraction') or 0):.1f}%), "
            f"{grid.get('thin_cells')} thin ({100.0 * (grid.get('thin_fraction') or 0):.1f}%, "
            f"below {grid.get('thin_threshold_points_per_cell', 0):,.0f} points/cell)",
            "- Full per-view fractions, emptiest cells and thresholds: see coverage_report.json",
        ]
    if report.evaluation:
        lines += ["", "## Ground-truth evaluation", ""]
        for key, value in sorted(report.evaluation.items()):
            lines.append(f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}")
    lines += ["", "## Stage timings (seconds)", ""]
    for key, value in report.stage_seconds.items():
        lines.append(f"- {key}: {value:.1f}")
    lines.append(f"- total: {report.total_seconds:.1f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
