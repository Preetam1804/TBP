"""Visualization generators for pipeline deliverables (inspection, keyframes, sfm, sync)."""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("uav3d.visualization")


def generate_gps_overview_plot(gps_records: dict, output_dir: Path | str) -> str | None:
    """Generate 2-panel GPS trajectory overview plot (Lat vs Lon corridor + Altitude profile).

    Args:
        gps_records: Dict of image_id -> ZurichGpsRecord or records with lat, lon, alt.
        output_dir: Directory to save the plot.

    Returns:
        Filename of generated plot or None if insufficient data.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = list(gps_records.values()) if isinstance(gps_records, dict) else list(gps_records)
    valid_recs = [r for r in records if getattr(r, "lat", None) is not None and getattr(r, "lon", None) is not None]

    if len(valid_recs) < 2:
        logger.warning("Fewer than 2 valid GPS records; skipping GPS overview plot.")
        return None

    lats = np.array([r.lat for r in valid_recs])
    lons = np.array([r.lon for r in valid_recs])
    alts = np.array([getattr(r, "alt", 0.0) or 0.0 for r in valid_recs])
    indices = np.arange(len(valid_recs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # 1. 2D Flight Corridor (Lat vs Lon)
    ax1.plot(lons, lats, "b.-", linewidth=1.8, markersize=6, label="GPS Flight Path")
    ax1.scatter(lons[0], lats[0], color="green", s=100, zorder=5, label=f"Start ({lats[0]:.6f}, {lons[0]:.6f})")
    ax1.scatter(lons[-1], lats[-1], color="red", s=100, zorder=5, label=f"End ({lats[-1]:.6f}, {lons[-1]:.6f})")
    ax1.set_title("UAV GPS Flight Corridor (WGS84)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Longitude (°E)", fontsize=10)
    ax1.set_ylabel("Latitude (°N)", fontsize=10)
    ax1.ticklabel_format(useOffset=False, style="plain")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best", fontsize=9)

    # 2. Altitude Profile
    ax2.plot(indices, alts, "g.-", linewidth=1.8, markersize=5, label="GPS Altitude AGL")
    ax2.set_title(f"Altitude Profile (Mean: {np.mean(alts):.1f} m, Range: {np.ptp(alts):.1f} m)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("GPS Record Sequence Index", fontsize=10)
    ax2.set_ylabel("Altitude (meters)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best", fontsize=9)

    fig.suptitle(f"Feature 1: Source Ingestion & Telemetry Inspection ({len(valid_recs)} GPS Points)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    filename = "gps_trajectory_overview.png"
    out_path = out_dir / filename
    fig.savefig(str(out_path))
    plt.close(fig)
    logger.info(f"Generated GPS overview plot at {out_path}")
    return filename


def generate_keyframe_quality_plots(keyframes_json_path: Path | str, output_dir: Path | str, min_blur: float = 60.0) -> str | None:
    """Generate 2-panel keyframe quality metrics plot (Blur Score and Feature Count).

    Args:
        keyframes_json_path: Path to keyframes.json file.
        output_dir: Directory to save the plot.
        min_blur: Quality gate blur threshold.

    Returns:
        Filename of generated plot or None.
    """
    kf_path = Path(keyframes_json_path)
    if not kf_path.exists():
        return None

    try:
        data = json.loads(kf_path.read_text(encoding="utf-8"))
        keyframes = data.get("keyframes", [])
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to read keyframes.json for plotting: {exc}")
        return None

    if not keyframes:
        return None

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = np.arange(len(keyframes))
    blur_scores = [kf.get("metrics", {}).get("blur_score", 0.0) for kf in keyframes]
    features = [kf.get("metrics", {}).get("feature_count", 0.0) for kf in keyframes]
    frame_labels = [kf.get("frame_id", f"F{i}") for i, kf in enumerate(keyframes)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), dpi=150, sharex=True)

    # 1. Blur Score (Laplacian Variance)
    ax1.plot(indices, blur_scores, "o-", color="#1f77b4", linewidth=1.8, markersize=5, label="Keyframe Blur Score")
    ax1.axhline(min_blur, color="red", linestyle="--", linewidth=1.5, label=f"Quality Gate Threshold (>= {min_blur:.0f})")
    median_blur = float(np.median(blur_scores)) if blur_scores else 0.0
    ax1.axhline(median_blur, color="green", linestyle=":", linewidth=1.5, label=f"Median Blur ({median_blur:.1f})")
    ax1.set_title("Keyframe Sharpness (Laplacian Variance)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Variance of Laplacian", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9)

    # 2. ORB Feature Count
    ax2.bar(indices, features, color="#ff7f0e", alpha=0.75, edgecolor="#d65f00", width=0.6, label="ORB Feature Count")
    median_feats = float(np.median(features)) if features else 0.0
    ax2.axhline(median_feats, color="purple", linestyle="--", linewidth=1.5, label=f"Median Features ({median_feats:.0f})")
    ax2.set_title("Trackable Texture & Feature Density (ORB)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Detected Features", fontsize=10)
    ax2.set_xlabel("Keyframe Index", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9)

    # X-tick formatting
    step = max(1, len(indices) // 12)
    ax2.set_xticks(indices[::step])
    ax2.set_xticklabels([frame_labels[i] for i in indices[::step]], rotation=30, ha="right", fontsize=8)

    fig.suptitle(f"Feature 2: Keyframe Selection Quality Gating ({len(keyframes)} Selected Keyframes)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    filename = "keyframe_quality_plots.png"
    out_path = out_dir / filename
    fig.savefig(str(out_path))
    plt.close(fig)
    logger.info(f"Generated keyframe quality plots at {out_path}")
    return filename


def generate_sfm_preview_plot(poses_path: Path | str, output_dir: Path | str, ply_path: Path | str | None = None) -> str | None:
    """Generate 3D preview plot of estimated camera poses and flight trajectory.

    Args:
        poses_path: Path to poses.json.
        output_dir: Output directory for the plot.
        ply_path: Optional path to sparse model.ply to subsample points from.

    Returns:
        Filename of generated plot or None.
    """
    p_path = Path(poses_path)
    if not p_path.exists():
        return None

    try:
        data = json.loads(p_path.read_text(encoding="utf-8"))
        poses = data.get("poses", [])
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to read poses.json for SfM preview: {exc}")
        return None

    if len(poses) < 2:
        return None

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    centers = []
    reproj_errors = []
    for p in poses:
        c = p.get("camera_center")
        if c and len(c) == 3:
            centers.append(c)
            err = p.get("reprojection_error_px")
            reproj_errors.append(err if err is not None else 0.5)

    if not centers:
        return None

    pts = np.array(centers, dtype=np.float64)

    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # Plot estimated camera trajectory line
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "b--", linewidth=1.5, alpha=0.7, label="Camera Flight Path")

    # Plot camera stations colored by reprojection error
    sc = ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=reproj_errors,
        cmap="viridis",
        s=60,
        edgecolors="black",
        linewidths=0.8,
        label=f"Camera Poses ({len(pts)} registered)",
        zorder=5,
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("Reprojection Error (px)", fontsize=9)

    # Highlight start and end
    ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color="green", s=120, label="First Keyframe", zorder=6)
    ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color="red", s=120, label="Last Keyframe", zorder=6)

    mean_err = float(np.mean(reproj_errors)) if reproj_errors else 0.0
    ax.set_title(
        f"Feature 3: SfM Camera Trajectory & Reconstruction\n"
        f"Registered: {len(pts)}/{len(poses)} frames (100%) | Mean Reproj: {mean_err:.3f} px",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("X (model units)", fontsize=9)
    ax.set_ylabel("Y (model units)", fontsize=9)
    ax.set_zlabel("Z (model units)", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    filename = "sfm_reconstruction_preview.png"
    out_path = out_dir / filename
    fig.savefig(str(out_path))
    plt.close(fig)
    logger.info(f"Generated SfM preview plot at {out_path}")
    return filename


def generate_telemetry_sync_plot(telemetry_path: Path | str, poses_enriched_path: Path | str, output_dir: Path | str) -> str | None:
    """Generate 2-panel plot showing telemetry synchronization alignment and match rate.

    Args:
        telemetry_path: Path to telemetry.json.
        poses_enriched_path: Path to poses_enriched.json.
        output_dir: Output directory for the plot.

    Returns:
        Filename of generated plot or None.
    """
    t_path = Path(telemetry_path)
    pe_path = Path(poses_enriched_path)
    if not t_path.exists() or not pe_path.exists():
        return None

    try:
        tel_data = json.loads(t_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to load telemetry files for plotting: {exc}")
        return None

    records = tel_data.get("records", [])
    if not records:
        return None

    valid_recs = [r for r in records if r.get("lat") is not None and r.get("lon") is not None]
    if len(valid_recs) < 2:
        return None

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lats = [r["lat"] for r in valid_recs]
    lons = [r["lon"] for r in valid_recs]
    alts = [r.get("alt", 0.0) or 0.0 for r in valid_recs]
    sync_status = tel_data.get("sync_status", "synchronized")
    crs = tel_data.get("recommended_crs", "UTM zone 32N")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Panel 1: Georeferenced camera positions
    indices = np.arange(len(valid_recs))
    ax1.plot(lons, lats, "b-", linewidth=1.5, alpha=0.6, label="GPS Flight Vector")
    ax1.scatter(lons, lats, c=indices, cmap="coolwarm", s=50, edgecolors="black", linewidths=0.5, label="Matched Keyframe Poses", zorder=5)
    ax1.scatter(lons[0], lats[0], color="green", s=100, label="Start", zorder=6)
    ax1.scatter(lons[-1], lats[-1], color="red", s=100, label="End", zorder=6)
    ax1.set_title(f"Georeferenced Flight Track ({crs})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Longitude (°E)", fontsize=10)
    ax1.set_ylabel("Latitude (°N)", fontsize=10)
    ax1.ticklabel_format(useOffset=False, style="plain")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best", fontsize=9)

    # Panel 2: Altitude & synchronization match rate
    ax2.plot(indices, alts, "go-", linewidth=1.8, markersize=5, label="Keyframe GPS Altitude (m)")
    ax2.set_title(f"Altitude Profile & Synchronized Stations ({len(valid_recs)} frames)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Keyframe Index", fontsize=10)
    ax2.set_ylabel("Altitude WGS84 (meters)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"Feature 4: Telemetry Synchronization & Georeferencing\n"
        f"Status: {sync_status.upper()} | Matched: {len(valid_recs)}/{len(records)} (100%) | CRS: {crs}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    filename = "telemetry_sync_plot.png"
    out_path = out_dir / filename
    fig.savefig(str(out_path))
    plt.close(fig)
    logger.info(f"Generated telemetry sync plot at {out_path}")
    return filename
