"""Report and visualization generators for SP-1 ATE validation."""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np

from uav3d.config import PipelineConfig
from uav3d.models.validation import AteReport

logger = logging.getLogger("uav3d.validation.reports")


def generate_validation_plots(
    report: AteReport,
    output_dir: Path | str,
) -> dict[str, str]:
    """Generate 3D trajectory plot and per-frame ATE error profile plot using matplotlib.

    Args:
        report: Validated AteReport instance.
        output_dir: Directory where plot PNGs will be saved.

    Returns:
        Dictionary mapping plot identifiers to relative filenames.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not report.per_frame_errors:
        logger.warning("No per-frame records available to generate plots.")
        return {}

    records = report.per_frame_errors
    gt_pts = np.array([r.gt_world for r in records], dtype=np.float64)
    est_aligned_pts = np.array([r.aligned_est_world for r in records], dtype=np.float64)
    errors_m = np.array([r.error_m for r in records], dtype=np.float64)
    frame_indices = np.arange(len(records))

    plot_paths: dict[str, str] = {}

    # 1. 3D Trajectory Comparison Plot
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        gt_pts[:, 0],
        gt_pts[:, 1],
        gt_pts[:, 2],
        "o-",
        color="#1f77b4",
        label="Ground Truth (AGL)",
        linewidth=2.0,
        markersize=5,
    )
    ax.plot(
        est_aligned_pts[:, 0],
        est_aligned_pts[:, 1],
        est_aligned_pts[:, 2],
        "s--",
        color="#ff7f0e",
        label="Aligned SfM Trajectory",
        linewidth=1.8,
        markersize=4,
    )

    # Start and End markers
    ax.scatter(gt_pts[0, 0], gt_pts[0, 1], gt_pts[0, 2], color="green", s=100, label="Start (GT)", zorder=5)
    ax.scatter(gt_pts[-1, 0], gt_pts[-1, 1], gt_pts[-1, 2], color="red", s=100, label="End (GT)", zorder=5)

    ax.set_title(
        f"Trajectory Comparison: Ground Truth vs Aligned SfM (Sim3)\n"
        f"ATE RMSE: {report.metrics.rmse_m:.3f} m | Scale: {report.metrics.scale_factor:.3f}",
        fontsize=12,
        pad=15,
    )
    ax.set_xlabel("X (meters)", fontsize=10, labelpad=8)
    ax.set_ylabel("Y (meters)", fontsize=10, labelpad=8)
    ax.set_zlabel("Z (meters)", fontsize=10, labelpad=8)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    traj_filename = "trajectory_3d.png"
    traj_path = out_dir / traj_filename
    fig.tight_layout()
    fig.savefig(str(traj_path))
    plt.close(fig)
    plot_paths["trajectory_3d"] = traj_filename
    logger.info(f"Generated 3D trajectory plot: {traj_path}")

    # 2. Per-Frame ATE Error Profile Plot
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    ax.plot(frame_indices, errors_m, "o-", color="#d62728", linewidth=1.8, markersize=5, label="ATE Position Error (m)")
    ax.axhline(
        report.metrics.median_m,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.5,
        label=f"Median Error ({report.metrics.median_m:.3f} m)",
    )
    ax.axhline(
        report.metrics.p95_m,
        color="#9467bd",
        linestyle=":",
        linewidth=1.5,
        label=f"P95 Error ({report.metrics.p95_m:.3f} m)",
    )
    ax.axhline(
        5.0,
        color="#7f7f7f",
        linestyle="-.",
        linewidth=1.2,
        alpha=0.8,
        label="ATE Warning Threshold (5.0 m)",
    )

    ax.set_title(
        f"Per-Frame Absolute Trajectory Error (ATE)\n"
        f"Condition: {report.metrics.fit_condition} | Max: {report.metrics.max_m:.3f} m",
        fontsize=12,
    )
    ax.set_xlabel("Keyframe Index", fontsize=10)
    ax.set_ylabel("Error (meters)", fontsize=10)
    ax.set_xticks(frame_indices[:: max(1, len(frame_indices) // 10)])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)

    error_filename = "ate_error_profile.png"
    error_path = out_dir / error_filename
    fig.tight_layout()
    fig.savefig(str(error_path))
    plt.close(fig)
    plot_paths["ate_error_profile"] = error_filename
    logger.info(f"Generated ATE error profile plot: {error_path}")

    return plot_paths


def _read_json_if_present(path: Path) -> dict | None:
    """Load a JSON artifact from a run directory, returning None when absent or unreadable."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(f"Could not read {path}: {exc}")
        return None


def generate_sp1_report(
    run_dir: Path | str,
    report: AteReport,
    config: PipelineConfig,
    plots: dict[str, str],
) -> Path:
    """Generate markdown SP-1 metrics and validation report (SP1_REPORT.md).

    Args:
        run_dir: Run directory where SP1_REPORT.md will be written.
        report: Validated AteReport instance.
        config: PipelineConfig used during the run.
        plots: Dictionary of relative plot image paths.

    Returns:
        Path to the written SP1_REPORT.md file.
    """
    out_dir = Path(run_dir).resolve()
    report_file = out_dir / "SP1_REPORT.md"

    m = report.metrics
    reg_rate_str = f"{m.registration_rate * 100:.1f}%" if m.registration_rate is not None else "N/A"
    reproj_str = f"{m.mean_reprojection_error_px:.3f} px" if m.mean_reprojection_error_px is not None else "N/A"

    # --- Values taken from the run artifacts, never hardcoded -------------------------------
    poses_data = _read_json_if_present(out_dir / "poses.json") or {}
    keyframes_data = _read_json_if_present(out_dir / "keyframes.json") or {}
    telemetry_data = _read_json_if_present(out_dir / "telemetry.json") or {}
    sfm_metrics = poses_data.get("metrics") or {}
    camera_block = poses_data.get("camera") or {}

    num_total = int(sfm_metrics.get("num_images_total") or m.num_registered_frames)
    num_registered = int(sfm_metrics.get("num_images_registered") or m.num_registered_frames)
    num_dropped = max(0, num_total - num_registered)

    camera_model = camera_block.get("model") or "unknown"
    camera_params = camera_block.get("params") or []
    camera_params_str = "[" + ", ".join(f"{float(v):.4g}" for v in camera_params) + "]" if camera_params else "N/A"
    camera_resolution = (
        f"{camera_block.get('width', '?')}x{camera_block.get('height', '?')}" if camera_block else "N/A"
    )

    keyframe_meta = keyframes_data.get("selection") or {}
    stride_str = str(keyframe_meta.get("final_stride", config.keyframes.stride))
    num_keyframes = len(keyframes_data.get("keyframes") or []) or num_total
    min_blur_mode = keyframe_meta.get("min_blur_mode", config.keyframes.min_blur_mode)
    effective_min_blur = keyframe_meta.get("effective_min_blur", config.keyframes.min_blur)

    crs_label = (
        telemetry_data.get("crs_recommendation")
        or telemetry_data.get("recommended_crs")
        or (telemetry_data.get("sync") or {}).get("crs_recommendation")
        or "not computed"
    )
    sync_status = (
        telemetry_data.get("sync_status")
        or (telemetry_data.get("sync") or {}).get("sync_status")
        or "N/A"
    )
    telemetry_records = telemetry_data.get("records") or telemetry_data.get("telemetry") or []
    telemetry_matched = len(telemetry_records)

    # --- KPI thresholds come from the configuration actually used --------------------------
    reg_target = config.validation.min_registration_rate
    reproj_target = config.validation.max_reprojection_error_px
    ate_target = config.validation.ate_threshold_meters

    kpi_reg_pass = m.registration_rate is not None and m.registration_rate >= reg_target
    kpi_reproj_pass = m.mean_reprojection_error_px is not None and m.mean_reprojection_error_px <= reproj_target
    kpi_ate_pass = m.rmse_m <= ate_target

    content = f"""# SP-1 Metrics & Trajectory Validation Report

> **Dataset Provenance**: `{report.dataset_label}`
> **Evaluation Mode**: Ground Truth consumed exclusively for post-reconstruction validation. Never consumed by visual SfM or keyframing stages.

---

## 1. Executive Summary & KPI Status

| Primary Metric / KPI | Benchmark Target | Achieved Pipeline Value | Status |
| :--- | :--- | :--- | :--- |
| **SfM Keyframe Registration Rate** | $\\ge {reg_target * 100:.1f}\\%$ | **{reg_rate_str}** ({num_registered}/{num_total}) | {'**PASSED**' if kpi_reg_pass else '**FAILED**'} |
| **Mean Reprojection Error** | $\\le {reproj_target:.3f}\\text{{ px}}$ | **{reproj_str}** | {'**PASSED**' if kpi_reproj_pass else '**FAILED**'} |
| **Absolute Trajectory Error (RMSE)** | $\\le {ate_target:.3f}\\text{{ m}}$ | **{m.rmse_m:.3f} m** | {'**PASSED**' if kpi_ate_pass else '**FAILED**'} |
| **Sim(3) Alignment Condition** | `well_conditioned` | **{m.fit_condition}** | {'Normal' if m.fit_condition == 'well_conditioned' else 'Warning (Collinear corridor)'} |

---

## 2. Trajectory Alignment & ATE Metrics (7-DoF Sim3)

Similarity transformation optimal parameters ($C_{{gt}} \\approx s \\cdot R \\cdot C_{{est}} + t$):
- **Scale Factor ($s$)**: `{m.scale_factor:.4f}`
- **Rotation Determinant (SO3 Reflection Guard)**: `{report.alignment_transform.get('det_R', 1.0):.6f}`
- **Translation Vector ($t$)**: `{report.alignment_transform.get('translation_m')}` meters
- **Fit Condition**: `{m.fit_condition}`

### Metric Distribution Table

| Statistic | Position Error (meters) |
| :--- | :--- |
| **RMSE** | **{m.rmse_m:.4f} m** |
| **Median Error** | **{m.median_m:.4f} m** |
| **Mean Error** | **{m.mean_m:.4f} m** |
| **95th Percentile (P95)** | **{m.p95_m:.4f} m** |
| **Maximum Error** | **{m.max_m:.4f} m** |
| **Common Matched Frames** | **{m.num_matched_frames} / {m.num_registered_frames}** |

---

## 3. Visualizations

### 3D Trajectory Alignment
![3D Trajectory]({plots.get('trajectory_3d', 'trajectory_3d.png')})

### Per-Frame Absolute Trajectory Error Profile
![ATE Error Profile]({plots.get('ate_error_profile', 'ate_error_profile.png')})

---

## 4. Pipeline Stage Breakdown

| Stage | Key Operations & Parameters | Artifacts Produced |
| :--- | :--- | :--- |
| **Phase 1: Ingestion** | Zurich image sequence, discrete timestamp mapping | `SourceMetadata`, verified calibration |
| **Phase 2: Keyframing** | Laplacian variance quality gate ({min_blur_mode}, threshold {effective_min_blur:g}), overlap guard (stride={stride_str}) | `keyframes.json` ({num_keyframes} keyframes), `sfm/images/`, `contact_sheet.jpg` |
| **Phase 3: Visual SfM** | pycolmap incremental mapping, `{camera_model}` intrinsics {camera_params_str} at {camera_resolution} ({'locked' if not config.sfm.refine_intrinsics else 'refined'}), sequential matching | `poses.json`, `database.db`, `sparse/0/` |
| **Phase 4: Telemetry Sync** | Source-image-id join (status `{sync_status}`), `{telemetry_matched}` telemetry records, CRS `{crs_label}` | `telemetry.json`, `poses_enriched.json` |
| **Phase 5: Validation** | 7-DoF Umeyama alignment, PCA flightline condition check | `ate_report.json`, plots, `SP1_REPORT.md` |

---

## 5. Known Limitations & Engineering Assumptions

1. **Rolling Shutter Distortion Uncorrected**:
   - The GoPro Hero4 camera sensor utilizes a rolling shutter. During rapid translational and yaw accelerations, slight spatial distortion occurs that is not parameterized in the global pinhole/OPENCV projection model.
2. **GPS Multipath & Consumer Receiver Noise**:
   - Single-frequency onboard GPS logs (`OnbordGPS.csv`) are subject to urban canyon multipath and atmospheric drift ($\\\\pm 2$–$5$ meters).
3. **Stride-Based Keyframing**:
   - Keyframe selection combines Laplacian variance blur filtering and uniform temporal striding with an overlap safety guard. Dense optical flow feature velocity gating is deferred to future optimization.
4. **Flightline Collinearity**:
   - As flagged by the fit condition check (`{m.fit_condition}`), linear flight trajectories exhibit small cross-track dispersion, making rotation about the primary flight vector ill-conditioned. The Umeyama alignment reports the global least-squares fit across the corridor.
5. **Unregistered Frames**:
   - {num_dropped} frame(s) dropped ({num_registered}/{num_total} registered in the largest model).

---

## 6. SP-2 Handoff Specification

SP-1 deliverables are fully prepared for SP-2 (Monocular Depth & TSDF Fusion):
- `sfm/images/`: Keyframe image directory ready for depth network batch inference.
- `poses.json`: Verified camera trajectory ($C = -R^T t$, quaternion `wxyz`) and locked camera intrinsic block.
- `telemetry.json`: Georeferenced WGS84 coordinates and recommended CRS (`{crs_label}`).
"""
    report_file.write_text(content.strip() + "\n", encoding="utf-8")
    logger.info(f"Wrote SP-1 report to {report_file}")
    return report_file
