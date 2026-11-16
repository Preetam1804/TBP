"""Coverage reporting for the dense / mesh stage (accuracy-work Phase 4).

This module answers one question the accuracy numbers alone do not: *where*
did the reconstruction actually produce surface, and where is it thin or
missing? It is computed from the artefacts a dense run already writes — the
final filtered depth maps, the fused ``cloud.ply`` — plus the run's scale
provenance. Ground truth is never an input here.

What is measured
----------------
* **Per-view valid-depth fraction**, from the final filtered depth maps
  (``depth_filtered/`` when present, otherwise ``depth/``). The raw MVS
  fraction from ``depth/`` is reported alongside it when both exist, so a
  drop between the two is visible as consistency-filter loss rather than
  silently blamed on the matcher.
* **Measured vs inferred pixels.** When the monocular prior filled holes,
  filled pixels are *inferred, never measured*. The per-view split is only
  reported when the caller supplies per-view filled counts (``run_dense``
  tracks them during the fill); re-analysing an old run directory from disk
  cannot recover the split, so those fields are ``None`` there and the
  provenance is labelled ``final_mixed_measured_and_inferred`` instead of
  pretending the split is known. The run-wide split comes from the prior
  summary (``pixels_measured`` / ``pixels_filled``) when available.
* **Fused point density** on the footprint plane and overall points per unit
  area. The footprint plane is the plane spanned by the two largest-variance
  PCA axes of the fused cloud. For a nadir city flight that plane *is* the
  ground plane (the largest spreads are horizontal, the smallest is height);
  for a corridor / canyon flight it is the along-track x cross-track plane,
  and the report says so (``projection: "pca_pc1_pc2"``) rather than claiming
  a gravity-aligned ground plane we did not measure.
* **Holes / thin regions** on a regular grid over that footprint. Only grid
  cells whose centre lies inside the convex hull of the projected cloud are
  counted: bbox corners outside a diagonal footprint are not surface that
  was missed, they were never inside the footprint at all.

Units and scale
---------------
If the run's scale is verified (``fixed`` or ``gnss`` source) the cloud is
converted to metres and a cell is ``cell_size_m`` metres on a side. If the
scale is unverified, everything is reported in SfM units and the report says
so (``unit: "SfM units (scale unverified)"``); no metres are invented.

Verdict thresholds (also written into every report under ``thresholds``)
------------------------------------------------------------------------
* ``holes``  — more than 10% of footprint cells are empty.
* ``thin``   — more than 5% of footprint cells are empty, or more than 25%
  of footprint cells hold less than 10% of the mean per-cell point count.
* ``good``   — otherwise.

The 10% / 5% / 25% cutoffs are deliberately coarse: a single empty cell in
a small footprint is one cell, not a failed reconstruction, while a tenth of
the footprint missing is a real gap in the model. The gate in
``scripts/run_video_pipeline.py`` fails a run (``ok=False``) exactly when
the verdict is not ``good``.

Determinism: every quantity is a pure function of the input files (SVD sign
ambiguity is removed by a fixed sign convention, grid binning is by
``floor``), so re-running on the same directory reproduces the report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError

from uav3d.dense.evaluate import load_ply_points
from uav3d.exceptions import DataError

logger = logging.getLogger("uav3d.dense.coverage")

COVERAGE_SCHEMA_VERSION = "1.0.0"

# --- verdict thresholds (see module docstring for the reasoning) ----------
DEFAULT_CELL_SIZE_M = 1.0
DEFAULT_THIN_FRACTION = 0.10
DEFAULT_HOLE_FRACTION_WARN = 0.05
DEFAULT_HOLE_FRACTION_BAD = 0.10
DEFAULT_THIN_FRACTION_BAD = 0.25
MAX_EMPTIEST_CELLS = 10


def _sign_fixed_basis(vt: np.ndarray) -> np.ndarray:
    """Fix SVD sign ambiguity: each axis points so its largest-|component| is positive."""
    basis = np.array(vt, dtype=np.float64)
    for i in range(basis.shape[0]):
        j = int(np.argmax(np.abs(basis[i])))
        if basis[i, j] < 0:
            basis[i] = -basis[i]
    return basis


def pca_footprint(points: np.ndarray) -> dict:
    """Project ``points`` onto the two largest-variance PCA axes.

    Returns the 2-D projection, the PCA basis/centroid (so a grid cell can be
    mapped back to an approximate 3-D position), axis extents, and the
    convex-hull / bounding-box footprint areas in the input units.
    """
    if points.shape[0] == 0:
        raise DataError("Cannot compute a footprint for an empty point cloud")
    centroid = points.mean(axis=0)
    centred = points - centroid
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    basis = _sign_fixed_basis(vt)
    projected = centred @ basis[:2].T
    extents = projected.max(axis=0) - projected.min(axis=0)
    bbox_area = float(extents[0] * extents[1])
    hull_area: float | None = None
    hull_basis = "bounding_box"
    if points.shape[0] >= 3:
        try:
            hull_area = float(ConvexHull(projected).volume)
            hull_basis = "convex_hull"
        except QhullError:
            hull_area = None
    return {
        "projected": projected,
        "basis": basis,
        "centroid": centroid,
        "extents": extents,
        "bbox_area": bbox_area,
        "hull_area": hull_area,
        "footprint_area": hull_area if hull_area is not None else bbox_area,
        "footprint_area_basis": hull_basis,
    }


def _load_depth_map(path: Path) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    if depth.ndim != 2:
        raise DataError(f"Depth map is not 2-D: {path}")
    return depth


def _depth_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.npy"))


def _view_index(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def per_view_fractions(
    final_dir: Path,
    raw_dir: Path | None = None,
    per_view_filled_pixels: dict[int, int] | None = None,
    prior_enabled: bool = False,
) -> list[dict]:
    """Per-view valid-depth fractions from the final filtered depth maps.

    ``per_view_filled_pixels`` maps a view index to the number of its final
    valid pixels that came from the monocular prior (inferred). When the
    prior was enabled but that map is not supplied, the measured/inferred
    split is unknown and reported as ``None`` — never guessed.
    """
    filled = {int(k): int(v) for k, v in (per_view_filled_pixels or {}).items()}
    raw_by_index: dict[int, float] = {}
    if raw_dir is not None:
        for path in _depth_files(raw_dir):
            index = _view_index(path)
            if index is None:
                continue
            depth = _load_depth_map(path)
            raw_by_index[index] = float(np.mean(np.isfinite(depth) & (depth > 0)))

    rows: list[dict] = []
    for path in _depth_files(final_dir):
        index = _view_index(path)
        depth = _load_depth_map(path)
        valid = np.isfinite(depth) & (depth > 0)
        valid_pixels = int(valid.sum())
        total_pixels = int(depth.size)
        valid_fraction = float(valid_pixels / max(total_pixels, 1))
        if prior_enabled and index is not None and index in filled:
            filled_pixels = min(filled[index], valid_pixels)
            measured_fraction = float((valid_pixels - filled_pixels) / max(total_pixels, 1))
            inferred_fraction = float(filled_pixels / max(total_pixels, 1))
            provenance = "measured_and_inferred_split"
        elif prior_enabled:
            measured_fraction = None
            inferred_fraction = None
            provenance = "final_mixed_measured_and_inferred"
        else:
            measured_fraction = valid_fraction
            inferred_fraction = 0.0
            provenance = "measured"
        rows.append(
            {
                "view_index": index,
                "file": path.name,
                "valid_fraction": valid_fraction,
                "measured_fraction": measured_fraction,
                "inferred_fraction": inferred_fraction,
                "raw_valid_fraction": raw_by_index.get(index),
                "valid_pixels": valid_pixels,
                "total_pixels": total_pixels,
                "provenance": provenance,
            }
        )
    rows.sort(key=lambda r: (r["view_index"] is None, r["view_index"] if r["view_index"] is not None else 0, r["file"]))
    return rows


def grid_coverage(
    projected: np.ndarray,
    cell_size: float,
    thin_fraction: float = DEFAULT_THIN_FRACTION,
    hull_vertices: np.ndarray | None = None,
) -> dict:
    """Regular-grid hole/thin analysis over a 2-D footprint projection.

    Cells are ``cell_size`` on a side over the projection bounding box, but
    only cells whose centre falls inside the convex hull of the projected
    points count towards the footprint (see module docstring).
    """
    if cell_size <= 0:
        raise DataError(f"Grid cell size must be positive, got {cell_size}")
    if projected.shape[0] == 0:
        return {
            "cell_size": float(cell_size),
            "num_cells": 0,
            "num_cells_bbox": 0,
            "empty_cells": 0,
            "empty_fraction": 1.0,
            "thin_cells": 0,
            "thin_fraction": 0.0,
            "thin_threshold_points_per_cell": 0.0,
            "mean_points_per_cell": 0.0,
            "median_points_per_cell": 0.0,
            "emptiest_cells": [],
        }
    origin = projected.min(axis=0)
    extent = projected.max(axis=0) - origin
    nx = max(int(np.ceil(extent[0] / cell_size)), 1)
    ny = max(int(np.ceil(extent[1] / cell_size)), 1)
    ix = np.clip(((projected[:, 0] - origin[0]) / cell_size).astype(int), 0, nx - 1)
    iy = np.clip(((projected[:, 1] - origin[1]) / cell_size).astype(int), 0, ny - 1)
    counts = np.zeros((ny, nx), dtype=np.int64)
    np.add.at(counts, (iy, ix), 1)

    centres_x = origin[0] + (np.arange(nx) + 0.5) * cell_size
    centres_y = origin[1] + (np.arange(ny) + 0.5) * cell_size
    gx, gy = np.meshgrid(centres_x, centres_y)
    centres = np.stack([gx.ravel(), gy.ravel()], axis=1)
    inside = np.ones(centres.shape[0], dtype=bool)
    if hull_vertices is not None and hull_vertices.shape[0] >= 3:
        try:
            inside = Delaunay(hull_vertices).find_simplex(centres) >= 0
        except QhullError:
            inside = np.ones(centres.shape[0], dtype=bool)
    inside = inside.reshape(ny, nx)
    footprint_counts = counts[inside]
    num_cells = int(footprint_counts.size)
    empty_cells = int(np.sum(footprint_counts == 0))
    mean_per_cell = float(footprint_counts.mean()) if num_cells else 0.0
    thin_threshold = float(thin_fraction * mean_per_cell)
    thin_cells = int(np.sum(footprint_counts < thin_threshold)) if num_cells else 0
    order = np.argsort(footprint_counts, kind="stable")
    # footprint_counts is the masked (hull-only) view, so recover grid
    # coordinates from the inside mask, not from the flattened order.
    inside_flat = np.flatnonzero(inside.ravel())
    emptiest: list[dict] = []
    for rank in order[:MAX_EMPTIEST_CELLS]:
        flat = int(inside_flat[int(rank)])
        j, i = flat // nx, flat % nx
        emptiest.append(
            {
                "cell_i": i,
                "cell_j": j,
                "centre_pc1": float(centres_x[i]),
                "centre_pc2": float(centres_y[j]),
                "count": int(counts[j, i]),
            }
        )
    return {
        "cell_size": float(cell_size),
        "num_cells": num_cells,
        "num_cells_bbox": int(nx * ny),
        "empty_cells": empty_cells,
        "empty_fraction": float(empty_cells / max(num_cells, 1)),
        "thin_cells": thin_cells,
        "thin_fraction": float(thin_cells / max(num_cells, 1)),
        "thin_threshold_points_per_cell": thin_threshold,
        "mean_points_per_cell": mean_per_cell,
        "median_points_per_cell": float(np.median(footprint_counts)) if num_cells else 0.0,
        "emptiest_cells": emptiest,
    }


def decide_verdict(
    empty_fraction: float,
    thin_fraction: float,
    num_points: int,
    hole_fraction_warn: float = DEFAULT_HOLE_FRACTION_WARN,
    hole_fraction_bad: float = DEFAULT_HOLE_FRACTION_BAD,
    thin_fraction_bad: float = DEFAULT_THIN_FRACTION_BAD,
) -> tuple[str, list[str]]:
    """Overall coverage verdict plus the human-readable reasons behind it."""
    reasons: list[str] = []
    if num_points == 0:
        return "holes", ["fused cloud is empty: no surface was recovered at all"]
    if empty_fraction > hole_fraction_bad:
        reasons.append(
            f"{100.0 * empty_fraction:.1f}% of footprint cells are empty "
            f"(threshold {100.0 * hole_fraction_bad:.0f}%)"
        )
        return "holes", reasons
    verdict = "good"
    if empty_fraction > hole_fraction_warn:
        reasons.append(
            f"{100.0 * empty_fraction:.1f}% of footprint cells are empty "
            f"(warn above {100.0 * hole_fraction_warn:.0f}%)"
        )
        verdict = "thin"
    if thin_fraction > thin_fraction_bad:
        reasons.append(
            f"{100.0 * thin_fraction:.1f}% of footprint cells are thin "
            f"(threshold {100.0 * thin_fraction_bad:.0f}%)"
        )
        verdict = "thin"
    if verdict == "good":
        reasons.append(
            f"{100.0 * empty_fraction:.1f}% empty cells, "
            f"{100.0 * thin_fraction:.1f}% thin cells — within thresholds"
        )
    return verdict, reasons


def coverage_gate_ok(coverage: dict | None) -> bool:
    """True when there is no coverage verdict to fail on, or it is ``good``."""
    if not coverage:
        return True
    return coverage.get("verdict", "good") == "good"


def compute_coverage(
    dense_dir: str | Path,
    *,
    scale: float = 1.0,
    scale_source: str = "none",
    scale_verified: bool = False,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
    thin_fraction: float = DEFAULT_THIN_FRACTION,
    hole_fraction_warn: float = DEFAULT_HOLE_FRACTION_WARN,
    hole_fraction_bad: float = DEFAULT_HOLE_FRACTION_BAD,
    thin_fraction_bad: float = DEFAULT_THIN_FRACTION_BAD,
    prior_enabled: bool = False,
    prior_pixels_filled: int | None = None,
    prior_pixels_measured: int | None = None,
    per_view_filled_pixels: dict[int, int] | None = None,
    voxel_units: float | None = None,
) -> dict:
    """Compute the coverage report for one dense run directory.

    ``dense_dir`` must contain ``cloud.ply`` and, for the per-view part,
    ``depth_filtered/`` (preferred) or ``depth/``. ``scale_verified`` must
    be True only when the scale comes from a verified source (``fixed`` or
    ``gnss``); with an unverified scale all areas/densities are reported in
    SfM units and labelled as such.
    """
    dense_path = Path(dense_dir)
    cloud_path = dense_path / "cloud.ply"
    if not cloud_path.exists():
        raise DataError(f"No fused cloud found for coverage reporting: {cloud_path}")
    points_raw = load_ply_points(cloud_path)
    if scale <= 0:
        raise DataError(f"Scale must be positive, got {scale}")

    unit = "m" if scale_verified else "SfM units (scale unverified)"
    # Work in reporting units: metres when verified, SfM units otherwise.
    # Either way a cell is cell_size_m reporting units on a side — when the
    # scale is unverified that means 1 SfM unit per cell by default, and no
    # conversion is applied because the scale is not trusted.
    points = points_raw * scale if scale_verified else points_raw
    cell_size = float(cell_size_m)

    filtered_dir = dense_path / "depth_filtered"
    raw_dir = dense_path / "depth"
    if _depth_files(filtered_dir):
        final_dir, final_source = filtered_dir, "depth_filtered"
    elif _depth_files(raw_dir):
        final_dir, final_source = raw_dir, "depth"
    else:
        final_dir, final_source = filtered_dir, "depth_filtered"
    per_view = per_view_fractions(
        final_dir,
        raw_dir=raw_dir if _depth_files(raw_dir) else None,
        per_view_filled_pixels=per_view_filled_pixels,
        prior_enabled=prior_enabled,
    )
    if per_view:
        valid = np.array([row["valid_fraction"] for row in per_view], dtype=np.float64)
        per_view_summary = {
            "num_views": len(per_view),
            "source": final_source,
            "min_valid_fraction": float(valid.min()),
            "max_valid_fraction": float(valid.max()),
            "median_valid_fraction": float(np.median(valid)),
            "mean_valid_fraction": float(valid.mean()),
        }
    else:
        per_view_summary = {
            "num_views": 0,
            "source": final_source,
            "min_valid_fraction": None,
            "max_valid_fraction": None,
            "median_valid_fraction": None,
            "mean_valid_fraction": None,
        }

    footprint = pca_footprint(points)
    projected = footprint["projected"]
    hull_vertices = None
    if footprint["hull_area"] is not None and points.shape[0] >= 3:
        try:
            hull_vertices = projected[ConvexHull(projected).vertices]
        except QhullError:
            hull_vertices = None
    grid = grid_coverage(projected, cell_size, thin_fraction=thin_fraction, hull_vertices=hull_vertices)
    # Map the emptiest cell centres back onto the PCA plane through the
    # centroid so a hole can be located approximately in 3-D.
    basis = footprint["basis"]
    centroid = footprint["centroid"]
    for cell in grid["emptiest_cells"]:
        centre3d = (
            centroid
            + cell["centre_pc1"] * basis[0]
            + cell["centre_pc2"] * basis[1]
        )
        cell["centre_approx"] = [float(v) for v in centre3d]

    footprint_area = float(footprint["footprint_area"])
    num_points = int(points.shape[0])
    density = float(num_points / footprint_area) if footprint_area > 0 else 0.0
    verdict, reasons = decide_verdict(
        grid["empty_fraction"],
        grid["thin_fraction"],
        num_points,
        hole_fraction_warn=hole_fraction_warn,
        hole_fraction_bad=hole_fraction_bad,
        thin_fraction_bad=thin_fraction_bad,
    )
    thresholds = {
        "cell_size": float(cell_size),
        "cell_size_unit": unit,
        "thin_fraction_of_mean_cell_density": float(thin_fraction),
        "hole_fraction_warn": float(hole_fraction_warn),
        "hole_fraction_bad": float(hole_fraction_bad),
        "thin_fraction_bad": float(thin_fraction_bad),
    }
    prior_block: dict = {
        "enabled": bool(prior_enabled),
        "pixels_filled": prior_pixels_filled,
        "pixels_measured": prior_pixels_measured,
        "note": (
            "Filled pixels are inferred from the monocular prior, not measured; "
            "per-view measured/inferred splits are only known when per-view "
            "filled counts were recorded during the run."
            if prior_enabled
            else "Prior fill was not enabled; all valid depth pixels are measured."
        ),
    }
    return {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "dense_dir": str(dense_path),
        "unit": unit,
        "scale": {
            "value": float(scale),
            "source": scale_source,
            "verified": bool(scale_verified),
        },
        "per_view": per_view,
        "per_view_summary": per_view_summary,
        "prior": prior_block,
        "fused": {
            "num_points": num_points,
            "footprint_area": footprint_area,
            "footprint_area_basis": footprint["footprint_area_basis"],
            "bbox_area": float(footprint["bbox_area"]),
            "points_per_unit_area": density,
            "projection": "pca_pc1_pc2",
            "projection_note": (
                "Footprint plane is spanned by the two largest-variance PCA axes "
                "of the fused cloud; for a nadir flight this is the ground plane."
            ),
            "voxel_units": voxel_units,
        },
        "grid": grid,
        "verdict": verdict,
        "verdict_reasons": reasons,
        "thresholds": thresholds,
    }


def compute_coverage_for_run_dir(
    dense_dir: str | Path,
    **overrides,
) -> dict:
    """Re-analyse an existing dense directory from disk.

    Scale and prior provenance are read from ``dense_report.json`` and
    ``prior_report.json`` when present; the per-view measured/inferred split
    cannot be recovered from disk, so per-view rows are labelled mixed when
    the prior was enabled. ``overrides`` are passed through to
    :func:`compute_coverage`.
    """
    dense_path = Path(dense_dir)
    scale = 1.0
    scale_source = "none"
    scale_verified = False
    prior_enabled = False
    prior_pixels_filled: int | None = None
    prior_pixels_measured: int | None = None
    report_path = dense_path / "dense_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        scale_info = report.get("scale") or {}
        scale = float(scale_info.get("value", 1.0))
        scale_source = str(scale_info.get("source", "none"))
        scale_verified = scale_source in ("fixed", "gnss")
        prior_info = report.get("prior") or {}
        prior_enabled = bool(prior_info.get("enabled", False))
        if prior_enabled:
            prior_pixels_filled = prior_info.get("pixels_filled")
            prior_pixels_measured = prior_info.get("pixels_measured")
    prior_path = dense_path / "prior_report.json"
    if prior_path.exists():
        prior_enabled = True
    return compute_coverage(
        dense_path,
        scale=scale,
        scale_source=scale_source,
        scale_verified=scale_verified,
        prior_enabled=prior_enabled,
        prior_pixels_filled=prior_pixels_filled,
        prior_pixels_measured=prior_pixels_measured,
        **overrides,
    )


def write_coverage_report(dense_dir: str | Path, coverage: dict) -> Path:
    """Write ``coverage_report.json`` next to ``dense_report.json``."""
    path = Path(dense_dir) / "coverage_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    return path
