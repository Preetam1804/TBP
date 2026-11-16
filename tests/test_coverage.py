"""Tests for Phase 4 coverage reporting (uav3d/dense/coverage.py).

Covers: per-view valid-depth fractions on synthetic depth maps, grid hole
detection on a synthetic cloud with a known hole, JSON report round-trip,
the coverage verdict gate, measured-vs-inferred labelling, and summarise()
printing coverage lines (plus backwards compatibility when coverage data is
absent).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from uav3d.dense.config import DenseConfig
from uav3d.dense.coverage import (
    compute_coverage,
    compute_coverage_for_run_dir,
    coverage_gate_ok,
    decide_verdict,
    grid_coverage,
    per_view_fractions,
    write_coverage_report,
)
from uav3d.dense.fusion import write_ply
from uav3d.exceptions import DataError
from uav3d.models.dense import DenseReport

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_run_video_pipeline():
    spec = importlib.util.spec_from_file_location(
        "run_video_pipeline", REPO_ROOT / "scripts" / "run_video_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_test_dense_pipeline():
    spec = importlib.util.spec_from_file_location(
        "test_dense_pipeline_mod", REPO_ROOT / "tests" / "test_dense_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_cloud(dense_dir: Path, points: np.ndarray) -> None:
    write_ply(dense_dir / "cloud.ply", points.astype(np.float64))


def _write_depth_maps(dense_dir: Path, subdir: str, maps: dict[int, np.ndarray]) -> None:
    target = dense_dir / subdir
    target.mkdir(parents=True, exist_ok=True)
    for index, depth in maps.items():
        np.save(target / f"{index:05d}.npy", depth.astype(np.float16))


def _lattice_cloud(
    nx: int = 14,
    ny: int = 10,
    per_cell: int = 25,
    hole: tuple[int, int, int, int] | None = None,
    seed: int = 7,
) -> np.ndarray:
    """Points clustered tightly around 1-unit cell centres on a flat
    ``nx`` x ``ny`` plane, except the ``hole`` rectangle (i0, i1, j0, j1)
    in cell coordinates, which gets none. The rectangular footprint makes
    the PCA axes line up with the cell axes, and the tight clusters keep
    the hole aligned to whole grid cells."""
    rng = np.random.default_rng(seed)
    points: list[np.ndarray] = []
    for i in range(nx):
        for j in range(ny):
            if hole and hole[0] <= i < hole[1] and hole[2] <= j < hole[3]:
                continue
            cell = rng.uniform(-0.05, 0.05, size=(per_cell, 2)) + np.array([i + 0.5, j + 0.5])
            points.append(np.hstack([cell, np.zeros((per_cell, 1))]))
    return np.vstack(points)


def _grid_cloud(
    size: int = 10,
    per_cell: int = 25,
    hole: tuple[int, int, int, int] | None = None,
    seed: int = 7,
) -> np.ndarray:
    """Points on a flat ``size`` x ``size`` plane, ``per_cell`` jittered
    points per 1-unit cell, except the ``hole`` rectangle
    (i0, i1, j0, j1) in cell coordinates, which gets none."""
    rng = np.random.default_rng(seed)
    points: list[np.ndarray] = []
    for i in range(size):
        for j in range(size):
            if hole and hole[0] <= i < hole[1] and hole[2] <= j < hole[3]:
                continue
            cell = rng.uniform(size=(per_cell, 2)) + np.array([i, j], dtype=float)
            z = np.zeros((per_cell, 1))
            points.append(np.hstack([cell, z]))
    return np.vstack(points)


def _dense_dir_with_cloud(tmp_path: Path, points: np.ndarray) -> Path:
    dense_dir = tmp_path / "dense"
    dense_dir.mkdir(parents=True)
    _write_cloud(dense_dir, points)
    return dense_dir


# --- per-view fractions -----------------------------------------------------


def test_per_view_fractions_on_synthetic_depth_maps(tmp_path: Path) -> None:
    dense_dir = _dense_dir_with_cloud(tmp_path, _grid_cloud())
    full = np.full((10, 20), 5.0)
    half = np.zeros((10, 20))
    half[:, :10] = 5.0
    empty = np.zeros((10, 20))
    _write_depth_maps(dense_dir, "depth_filtered", {0: full, 1: half, 2: empty})
    # Raw MVS maps: view 1 was 75% valid before the consistency filter.
    raw_half = np.zeros((10, 20))
    raw_half[:, :15] = 5.0
    _write_depth_maps(dense_dir, "depth", {0: full, 1: raw_half, 2: empty})

    coverage = compute_coverage(dense_dir)
    rows = {row["view_index"]: row for row in coverage["per_view"]}
    assert rows[0]["valid_fraction"] == pytest.approx(1.0)
    assert rows[1]["valid_fraction"] == pytest.approx(0.5)
    assert rows[2]["valid_fraction"] == pytest.approx(0.0)
    assert rows[1]["raw_valid_fraction"] == pytest.approx(0.75)
    assert rows[0]["raw_valid_fraction"] == pytest.approx(1.0)
    assert rows[0]["provenance"] == "measured"
    assert rows[0]["measured_fraction"] == pytest.approx(1.0)
    assert rows[0]["inferred_fraction"] == pytest.approx(0.0)
    summary = coverage["per_view_summary"]
    assert summary["num_views"] == 3
    assert summary["min_valid_fraction"] == pytest.approx(0.0)
    assert summary["max_valid_fraction"] == pytest.approx(1.0)
    assert summary["median_valid_fraction"] == pytest.approx(0.5)
    assert summary["source"] == "depth_filtered"


def test_per_view_fractions_fall_back_to_raw_depth_dir(tmp_path: Path) -> None:
    dense_dir = _dense_dir_with_cloud(tmp_path, _grid_cloud())
    _write_depth_maps(dense_dir, "depth", {0: np.full((4, 4), 3.0)})
    coverage = compute_coverage(dense_dir)
    assert coverage["per_view_summary"]["source"] == "depth"
    assert coverage["per_view"][0]["valid_fraction"] == pytest.approx(1.0)


def test_per_view_fractions_ignore_invalid_depths(tmp_path: Path) -> None:
    dense_dir = _dense_dir_with_cloud(tmp_path, _grid_cloud())
    depth = np.full((4, 4), 3.0)
    depth[0, 0] = np.nan
    depth[0, 1] = np.inf
    depth[0, 2] = -1.0
    depth[0, 3] = 0.0
    _write_depth_maps(dense_dir, "depth_filtered", {0: depth})
    coverage = compute_coverage(dense_dir)
    assert coverage["per_view"][0]["valid_fraction"] == pytest.approx(12 / 16)


# --- measured vs inferred ---------------------------------------------------


def test_prior_fill_labelled_inferred_never_measured(tmp_path: Path) -> None:
    dense_dir = _dense_dir_with_cloud(tmp_path, _grid_cloud())
    # 100 pixels, 60 valid in the final map, 25 of them prior-filled.
    depth = np.zeros((10, 10))
    depth[:6, :] = 5.0
    _write_depth_maps(dense_dir, "depth_filtered", {0: depth})
    rows = per_view_fractions(
        dense_dir / "depth_filtered", per_view_filled_pixels={0: 25}, prior_enabled=True
    )
    assert rows[0]["valid_fraction"] == pytest.approx(0.6)
    assert rows[0]["measured_fraction"] == pytest.approx(0.35)
    assert rows[0]["inferred_fraction"] == pytest.approx(0.25)
    assert rows[0]["provenance"] == "measured_and_inferred_split"


def test_prior_enabled_without_per_view_counts_reports_mixed(tmp_path: Path) -> None:
    """Re-analysing an old run from disk cannot recover the per-view split,
    so it must be labelled mixed with None splits — never guessed."""
    dense_dir = _dense_dir_with_cloud(tmp_path, _grid_cloud())
    depth = np.zeros((10, 10))
    depth[:6, :] = 5.0
    _write_depth_maps(dense_dir, "depth_filtered", {0: depth})
    rows = per_view_fractions(dense_dir / "depth_filtered", prior_enabled=True)
    assert rows[0]["provenance"] == "final_mixed_measured_and_inferred"
    assert rows[0]["measured_fraction"] is None
    assert rows[0]["inferred_fraction"] is None
    assert rows[0]["valid_fraction"] == pytest.approx(0.6)


# --- grid holes -------------------------------------------------------------


def test_grid_hole_detection_on_synthetic_cloud_with_known_hole(tmp_path: Path) -> None:
    # 14x10 lattice footprint, 4x4 hole in the middle: 16 of ~140 cells empty.
    points = _lattice_cloud(nx=14, ny=10, per_cell=25, hole=(5, 9, 3, 7))
    dense_dir = _dense_dir_with_cloud(tmp_path, points)
    coverage = compute_coverage(dense_dir)
    grid = coverage["grid"]
    # The hull restriction drops bbox cells whose centres fall outside the
    # convex hull. With tight lattice clusters the hull is inset from the
    # footprint edge, so the outer ring loses ~23 cells; on a real fused
    # cloud (points spread to the edge) the loss is only a few corner cells.
    # What must be exact is the hole itself: 16 empty cells.
    assert 110 <= grid["num_cells"] <= 140
    assert grid["empty_cells"] == 16
    assert grid["empty_fraction"] == pytest.approx(16 / grid["num_cells"])
    assert grid["thin_cells"] == 16  # empty cells are also below the thin threshold
    assert coverage["verdict"] == "holes"
    assert coverage_gate_ok(coverage) is False
    emptiest = grid["emptiest_cells"]
    assert emptiest and all(cell["count"] == 0 for cell in emptiest)
    # The emptiest cells sit inside the hole (cell coords 5..8 in x, 3..6 in y).
    assert all(5 <= cell["cell_i"] <= 8 and 3 <= cell["cell_j"] <= 6 for cell in emptiest)
    assert all(len(cell["centre_approx"]) == 3 for cell in emptiest)


def test_grid_coverage_full_cloud_is_good(tmp_path: Path) -> None:
    points = _grid_cloud(size=10, per_cell=25, hole=None)
    dense_dir = _dense_dir_with_cloud(tmp_path, points)
    coverage = compute_coverage(dense_dir)
    assert coverage["grid"]["empty_cells"] == 0
    assert coverage["grid"]["empty_fraction"] == pytest.approx(0.0)
    assert coverage["verdict"] == "good"
    assert coverage_gate_ok(coverage) is True
    assert coverage["fused"]["num_points"] == 2500
    assert coverage["fused"]["points_per_unit_area"] == pytest.approx(25.0, rel=0.05)


def test_grid_coverage_direct_on_projected_points() -> None:
    rng = np.random.default_rng(3)
    projected = rng.uniform(size=(1000, 2)) * 10.0
    grid = grid_coverage(projected, cell_size=1.0)
    assert grid["num_cells_bbox"] == 100
    assert grid["empty_fraction"] == pytest.approx(0.0)


def test_decide_verdict_thresholds() -> None:
    assert decide_verdict(0.0, 0.0, 100)[0] == "good"
    assert decide_verdict(0.06, 0.0, 100)[0] == "thin"  # > 5% empty
    assert decide_verdict(0.05, 0.0, 100)[0] == "good"  # at the warn line, not over it
    assert decide_verdict(0.11, 0.0, 100)[0] == "holes"  # > 10% empty
    assert decide_verdict(0.0, 0.26, 100)[0] == "thin"  # > 25% thin
    assert decide_verdict(0.0, 0.0, 0)[0] == "holes"  # empty cloud


# --- scale / units ----------------------------------------------------------


def test_unverified_scale_reports_sfm_units(tmp_path: Path) -> None:
    points = _grid_cloud(size=10, per_cell=25)
    dense_dir = _dense_dir_with_cloud(tmp_path, points)
    coverage = compute_coverage(dense_dir, scale=1.0, scale_verified=False)
    assert coverage["unit"] == "SfM units (scale unverified)"
    assert coverage["scale"]["verified"] is False
    assert coverage["fused"]["footprint_area"] == pytest.approx(100.0, rel=0.05)


def test_verified_scale_reports_metres(tmp_path: Path) -> None:
    points = _grid_cloud(size=10, per_cell=25)
    dense_dir = _dense_dir_with_cloud(tmp_path, points)
    coverage = compute_coverage(
        dense_dir, scale=2.0, scale_source="fixed", scale_verified=True
    )
    assert coverage["unit"] == "m"
    assert coverage["scale"]["verified"] is True
    # Points scale by 2, so areas scale by 4 and density by 1/4.
    assert coverage["fused"]["footprint_area"] == pytest.approx(400.0, rel=0.05)
    assert coverage["fused"]["points_per_unit_area"] == pytest.approx(2500 / 400, rel=0.05)


def test_missing_cloud_raises(tmp_path: Path) -> None:
    dense_dir = tmp_path / "dense"
    dense_dir.mkdir()
    with pytest.raises(DataError, match="No fused cloud"):
        compute_coverage(dense_dir)


# --- JSON round-trip --------------------------------------------------------


def test_coverage_json_report_round_trip(tmp_path: Path) -> None:
    points = _grid_cloud(size=10, per_cell=25, hole=(3, 7, 3, 7))
    dense_dir = _dense_dir_with_cloud(tmp_path, points)
    _write_depth_maps(dense_dir, "depth_filtered", {0: np.full((4, 4), 3.0)})
    coverage = compute_coverage(dense_dir)
    path = write_coverage_report(dense_dir, coverage)
    assert path.name == "coverage_report.json"
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded == coverage
    assert reloaded["schema_version"] == "1.0.0"
    assert reloaded["thresholds"]["hole_fraction_bad"] == pytest.approx(0.10)
    # The coverage dict must fit the dense report schema (coverage key).
    report = DenseReport.model_validate(
        {
            "run_dir": str(tmp_path),
            "created_utc": "2026-11-16T00:00:00+00:00",
            "geometry": {
                "num_registered_views": 1,
                "num_reference_views": 1,
                "working_width": 4,
                "working_height": 4,
                "focal_px": 100.0,
                "median_baseline": 1.0,
                "median_depth": 5.0,
                "median_baseline_ratio": 0.2,
                "depth_range": (1.0, 10.0),
            },
            "depth": {
                "total_pixels": 16,
                "valid_pixels": 16,
                "valid_fraction": 1.0,
                "plane_count_min": 8,
                "plane_count_max": 8,
                "median_cost": 0.1,
            },
            "fusion": {
                "voxel_units": 0.1,
                "trunc_units": 0.25,
                "depth_sigma_units": 0.1,
                "num_points": 2500,
            },
            "scale": {"value": 1.0, "source": "none", "status": "unverified_monocular_scale"},
            "total_seconds": 1.0,
            "coverage": coverage,
        }
    )
    assert report.model_validate_json(report.model_dump_json()).coverage == coverage


def test_compute_coverage_for_run_dir_reads_scale_from_report(tmp_path: Path) -> None:
    points = _grid_cloud(size=10, per_cell=25)
    dense_dir = _dense_dir_with_cloud(tmp_path, points)
    (dense_dir / "dense_report.json").write_text(
        json.dumps(
            {
                "scale": {"value": 3.0, "source": "gnss", "status": "gnss_aligned"},
                "prior": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    coverage = compute_coverage_for_run_dir(dense_dir)
    assert coverage["unit"] == "m"
    assert coverage["scale"]["source"] == "gnss"


def test_coverage_is_deterministic(tmp_path: Path) -> None:
    points = _grid_cloud(size=10, per_cell=25, hole=(3, 7, 3, 7))
    dense_dir = _dense_dir_with_cloud(tmp_path, points)
    first = compute_coverage(dense_dir)
    second = compute_coverage(dense_dir)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


# --- pipeline wiring --------------------------------------------------------


def test_run_dense_writes_coverage_report(tmp_path: Path) -> None:
    """End-to-end wiring: run_dense writes coverage_report.json, embeds the
    coverage key in dense_report.json, and adds a Coverage section to the
    markdown report."""
    from uav3d.dense.pipeline import run_dense

    helpers = _load_test_dense_pipeline()
    run_dir = tmp_path / "run_test"
    helpers._write_run(run_dir)
    config = DenseConfig(
        max_width=192,
        num_sources=3,
        plane_step_px=3.0,
        min_planes=8,
        max_planes=32,
        mesh=False,
    )
    result = run_dense(run_dir=run_dir, config=config)
    coverage_path = result.output_dir / "coverage_report.json"
    assert coverage_path.exists()
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    assert coverage["verdict"] in ("good", "thin", "holes")
    assert coverage["per_view_summary"]["num_views"] > 0
    assert coverage["fused"]["num_points"] == result.num_points
    assert coverage["unit"] == "SfM units (scale unverified)"

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["coverage"] == coverage
    assert report["artifacts"]["coverage_report"] == str(coverage_path)
    markdown = (result.output_dir / "DENSE_REPORT.md").read_text(encoding="utf-8")
    assert "## Coverage" in markdown
    assert f"Verdict: **{coverage['verdict']}**" in markdown


# --- summarise() ------------------------------------------------------------


def _dense_dict(*, valid_fraction: float = 0.5, coverage: dict | None = None) -> dict:
    dense: dict = {
        "geometry": {
            "num_registered_views": 10,
            "num_reference_views": 5,
            "working_width": 192,
            "working_height": 144,
        },
        "depth": {"valid_fraction": valid_fraction, "median_cost": 0.2},
        "fusion": {"num_points": 1000, "voxel_units": 0.01, "point_spacing_meters": None},
        "scale": {"status": "unverified_monocular_scale", "source": "none"},
    }
    if coverage is not None:
        dense["coverage"] = coverage
    return dense


def _coverage_dict(*, verdict: str = "good", empty: float = 0.0, thin: float = 0.05) -> dict:
    return {
        "unit": "SfM units (scale unverified)",
        "verdict": verdict,
        "verdict_reasons": [f"{100.0 * empty:.1f}% empty cells"],
        "per_view_summary": {
            "num_views": 5,
            "min_valid_fraction": 0.4,
            "max_valid_fraction": 0.6,
            "median_valid_fraction": 0.5,
        },
        "prior": {"enabled": False},
        "fused": {"num_points": 1000, "footprint_area": 10.0, "points_per_unit_area": 100.0},
        "grid": {
            "num_cells": 10,
            "empty_cells": int(round(empty * 10)),
            "empty_fraction": empty,
            "thin_cells": 1,
            "thin_fraction": thin,
        },
    }


def test_summarise_without_coverage_is_backwards_compatible(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Old run dirs have no coverage key: summarise must still print and
    return usable instead of crashing."""
    module = _load_run_video_pipeline()
    info = {"width": 192, "height": 144, "frames": 10, "fps": 10.0, "seconds": 1.0}
    ok = module.summarise(
        Path("video.mp4"), info, tmp_path, poses=None, dense=_dense_dict(coverage=None)
    )
    assert ok is True
    assert "no coverage report" in capsys.readouterr().out


def test_summarise_prints_coverage_lines(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_run_video_pipeline()
    info = {"width": 192, "height": 144, "frames": 10, "fps": 10.0, "seconds": 1.0}
    ok = module.summarise(
        Path("video.mp4"),
        info,
        tmp_path,
        poses=None,
        dense=_dense_dict(coverage=_coverage_dict(verdict="good")),
    )
    out = capsys.readouterr().out
    assert ok is True
    assert "COVERAGE" in out
    assert "per-view valid 40.0%-60.0%" in out
    assert "fused density 100 points/SfM unit^2" in out
    assert "verdict good" in out
    assert "WARNING" not in out


def test_summarise_coverage_holes_fails_the_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_run_video_pipeline()
    info = {"width": 192, "height": 144, "frames": 10, "fps": 10.0, "seconds": 1.0}
    ok = module.summarise(
        Path("video.mp4"),
        info,
        tmp_path,
        poses=None,
        dense=_dense_dict(coverage=_coverage_dict(verdict="holes", empty=0.2, thin=0.3)),
    )
    out = capsys.readouterr().out
    assert ok is False
    assert "WARNING" in out
    assert "coverage verdict is 'holes'" in out
    assert "completed WITH WARNINGS" in out


def test_summarise_with_no_dense_at_all(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_run_video_pipeline()
    info = {"width": 192, "height": 144, "frames": 10, "fps": 10.0, "seconds": 1.0}
    ok = module.summarise(Path("video.mp4"), info, tmp_path, poses=None, dense=None)
    assert ok is True
    assert "COVERAGE" not in capsys.readouterr().out
