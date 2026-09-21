"""3D model export: colored point cloud, COLMAP model directory, camera trajectory."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uav3d.models.reconstruction import ReconstructionSummary
from uav3d.sfm.colmap_compat import get_pose_components

logger = logging.getLogger("uav3d.modeling.export")


@dataclass
class ModelExportResult:
    """Paths and counts produced by export_model()."""

    model_ply: Path
    model_dir: Path
    trajectory_ply: Path
    summary_path: Path
    num_points: int


def export_model(
    reconstruction: Any,
    run_dir: str | Path,
    dataset_label: str,
    num_source_frames: int,
    num_keyframes: int,
    num_models: int,
    mean_reprojection_error_px: float | None,
    scale_note: str = "monocular SfM: metric scale is unknown until aligned to a metric reference (GPS/GT)",
) -> ModelExportResult:
    """Export the reconstruction as viewer-ready 3D model artifacts.

    Writes into the run directory:
      - model.ply            colored 3D point cloud (the 3D model; opens in
                             CloudCompare / MeshLab / Blender)
      - model/               COLMAP sparse model in text format
                             (cameras.txt, images.txt, points3D.txt)
      - trajectory.ply       camera-center flight path (vertices + edges)
      - reconstruction_summary.json  versioned machine-readable manifest

    Args:
        reconstruction: Selected pycolmap.Reconstruction (largest model).
        run_dir: Run output directory (data/outputs/run_<timestamp>).
        dataset_label: Provenance label recorded in the summary.
        num_source_frames: Frames available in the source before pre-striding.
        num_keyframes: Keyframes fed to SfM.
        num_models: Number of reconstructions produced.
        mean_reprojection_error_px: Mean reprojection error of the main model.
        scale_note: Scale interpretation text for the summary.
    """
    run_path = Path(run_dir).resolve()
    run_path.mkdir(parents=True, exist_ok=True)

    # --- 1. Colored point cloud (the 3D model) ---
    model_ply = run_path / "model.ply"
    reconstruction.export_PLY(str(model_ply))
    num_points = int(reconstruction.num_points3D())
    logger.info(f"Exported colored point cloud ({num_points:,} points) to {model_ply}")

    # --- 2. COLMAP sparse model directory (text format, viewer-friendly) ---
    model_dir = run_path / "model"
    if model_dir.exists():
        import shutil

        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write_text(str(model_dir))
    logger.info(f"Exported COLMAP sparse model (text) to {model_dir}")

    # --- 3. Camera trajectory as PLY (vertices + edges, time-ordered) ---
    trajectory_ply = run_path / "trajectory.ply"
    _write_trajectory_ply(reconstruction, trajectory_ply)
    logger.info(f"Exported camera trajectory to {trajectory_ply}")

    # --- 4. Versioned summary manifest ---
    summary = ReconstructionSummary(
        schema_version="1.0.0",
        dataset_label=dataset_label,
        num_source_frames=num_source_frames,
        num_keyframes=num_keyframes,
        num_registered_frames=int(reconstruction.num_reg_images()),
        num_unregistered_frames=max(0, num_keyframes - int(reconstruction.num_reg_images())),
        registration_rate=float(
            reconstruction.num_reg_images() / num_keyframes if num_keyframes > 0 else 0.0
        ),
        mean_reprojection_error_px=mean_reprojection_error_px,
        num_models=num_models,
        num_points3d=num_points,
        mean_track_length=(
            float(reconstruction.compute_mean_track_length())
            if hasattr(reconstruction, "compute_mean_track_length")
            else None
        ),
        scale_note=scale_note,
        model_ply="model.ply",
        model_dir="model",
        trajectory_ply="trajectory.ply",
        poses_json="poses.json",
        colmap_native_log="colmap_native.log",
    )
    # created_utc is filled by the caller after other stages may have written
    # telemetry/ATE artifacts; set a baseline here.
    from datetime import UTC, datetime

    summary.created_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary_path = run_path / "reconstruction_summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"Wrote reconstruction summary to {summary_path}")

    return ModelExportResult(
        model_ply=model_ply,
        model_dir=model_dir,
        trajectory_ply=trajectory_ply,
        summary_path=summary_path,
        num_points=num_points,
    )


def _registered_image_ids(reconstruction: Any) -> list[int]:
    """Registered image ids; pycolmap exposes this as a method or list by version."""
    ids = getattr(reconstruction, "reg_image_ids", None)
    if ids is None:
        return []
    return [int(i) for i in (ids() if callable(ids) else ids)]


def _write_trajectory_ply(reconstruction: Any, path: Path) -> None:
    """Write registered camera centers, ordered by frame name, as a PLY polyline."""
    # Collect (name, center) for all registered images, ordered lexicographically
    # so the path follows the capture sequence (frame IDs are zero-padded).
    entries: list[tuple[str, list[float]]] = []
    for image_id in _registered_image_ids(reconstruction):
        image = reconstruction.image(image_id)
        _, _, center = get_pose_components(image)
        entries.append((image.name, center))
    entries.sort(key=lambda item: item[0])

    n = len(entries)
    lines = [
        "ply",
        "format ascii 1.0",
        f"comment uav3d camera trajectory ({n} registered frames, ordered by frame id)",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element edge {max(0, n - 1)}",
        "property int vertex_index_1",
        "property int vertex_index_2",
        "end_header",
    ]
    for _, (cx, cy, cz) in entries:
        # Bright cyan flight path, distinct from scene point colors
        lines.append(f"{cx:.6f} {cy:.6f} {cz:.6f} 0 255 255")
    for i in range(n - 1):
        lines.append(f"{i} {i + 1}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_summary(
    summary_path: Path,
    telemetry_json: str | None = None,
    ate_report_json: str | None = None,
) -> None:
    """Patch telemetry/ATE artifact paths into an existing summary manifest."""
    if not summary_path.exists():
        return
    summary = ReconstructionSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
    if telemetry_json is not None:
        summary.telemetry_json = telemetry_json
    if ate_report_json is not None:
        summary.ate_report_json = ate_report_json
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
