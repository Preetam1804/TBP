"""Command-line interface for the uav3d pipeline."""

import argparse
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from uav3d import __version__
from uav3d.calibration import estimate_video_calibration, scale_calibration
from uav3d.config import PipelineConfig, load_config
from uav3d.exceptions import Uav3dError
from uav3d.keyframes.selector import KeyframeSelector
from uav3d.logging_config import setup_logging, timed_stage
from uav3d.models.keyframes import KeyframesFile
from uav3d.models.poses import PosesFile
from uav3d.sfm import SfmRunner, export_poses
from uav3d.sources.base import ImageSource
from uav3d.sources.video import VideoFileSource
from uav3d.sources.zurich import ZurichSequenceSource
from uav3d.telemetry import TelemetrySynchronizer
from uav3d.validation import (
    compute_ate,
    generate_sp1_report,
    generate_validation_plots,
)

console = Console()
err_console = Console(stderr=True)



def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach standard --config, --data-dir, and --video flags to a subcommand parser."""
    parser.add_argument(
        "--config",
        type=str,
        default="configs/zurich_baseline.yaml",
        help="Path to YAML pipeline configuration file (default: configs/zurich_baseline.yaml)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory root (overrides config data_dir and UAV3D_DATA_DIR env var)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to an MP4/video file to process directly without telemetry",
    )


def _build_source(cfg: PipelineConfig, data_dir: str, args: argparse.Namespace) -> ImageSource:
    """Build and initialize ImageSource adapter from configuration or CLI arguments."""
    video_cli = getattr(args, "video", None)
    if video_cli:
        video_path = Path(video_cli).resolve()
        if not video_path.exists():
            raise Uav3dError(f"Specified video file does not exist: {video_path}")
        return VideoFileSource(video_path=video_path)

    source_type = cfg.source.type.lower()
    if source_type == "zurich":
        dataset_path = Path(args.data_dir) if args.data_dir else Path(cfg.source.dataset_dir)
        if not dataset_path.is_absolute():
            dataset_path = (Path(data_dir).parent / dataset_path).resolve() if not dataset_path.exists() else dataset_path.resolve()

        return ZurichSequenceSource(
            dataset_dir=dataset_path,
            start_id=cfg.source.start_id,
            count=cfg.source.count,
            stride=cfg.source.stride,
        )
    if source_type == "video":
        video_path = Path(cfg.source.video_path or (Path(data_dir) / "interim" / "synthetic.mp4")).resolve()
        if not video_path.exists():
            raise Uav3dError(f"Video file not found at: {video_path}")
        return VideoFileSource(video_path=video_path)

    raise Uav3dError(f"Unsupported source type '{source_type}'. Must be 'zurich' or 'video'.")


def _handle_inspect_command(args: argparse.Namespace) -> int:
    """Execute the inspect pipeline command."""
    logger = setup_logging()
    cfg = load_config(args.config if os.path.exists(args.config) else None)
    data_dir = args.data_dir or os.environ.get("UAV3D_DATA_DIR") or cfg.data_dir

    with timed_stage("inspect", logger=logger):
        source = _build_source(cfg, data_dir, args)
        # Report the type of the source actually opened, not the configured default.
        source_type = "video" if isinstance(source, VideoFileSource) else cfg.source.type.lower()
        dataset_label = _resolve_dataset_label(cfg, source)
        meta = source.metadata()
        calib = source.calibration()

        # Build Rich table
        table = Table(title=f"Source Inspection: [bold cyan]{meta.name}[/bold cyan]", show_header=True)
        table.add_column("Property", style="bold white", width=26)
        table.add_column("Value", style="green")

        table.add_row("Source Type", source_type)
        table.add_row("Frame Count (in slice)", str(meta.frame_count))
        table.add_row("FPS", f"{meta.fps:.2f}" if meta.fps is not None else "N/A (discrete sequence)")
        table.add_row("Resolution", f"{meta.width} x {meta.height}")
        table.add_row("First Frame ID", meta.first_frame_id or "None")
        table.add_row("Last Frame ID", meta.last_frame_id or "None")
        table.add_row(
            "First Timestamp (µs)",
            f"{meta.first_timestamp_us:,} µs" if meta.first_timestamp_us is not None else "None (unmapped)",
        )
        table.add_row(
            "Last Timestamp (µs)",
            f"{meta.last_timestamp_us:,} µs" if meta.last_timestamp_us is not None else "None (unmapped)",
        )
        table.add_row(
            "Camera Model",
            f"{calib.model} (source: {calib.source})",
        )
        table.add_row(
            "Camera Parameters",
            f"{[round(p, 2) for p in calib.params]}",
        )

        if isinstance(source, ZurichSequenceSource):
            table.add_row("Telemetry GPS Records", f"{source.total_gps_count:,}")
            table.add_row("Ground Truth Records (EVAL)", f"{source.total_gt_count:,} [yellow](eval only)[/yellow]")

            # Calculate coordinate ranges if GPS records exist
            lats = [rec.lat for rec in source._gps_records.values() if rec.lat is not None]
            lons = [rec.lon for rec in source._gps_records.values() if rec.lon is not None]
            alts = [rec.alt for rec in source._gps_records.values() if rec.alt is not None]

            if lats and lons:
                table.add_row("Latitude Range", f"[{min(lats):.6f}, {max(lats):.6f}]")
                table.add_row("Longitude Range", f"[{min(lons):.6f}, {max(lons):.6f}]")
            if alts:
                table.add_row("Altitude Range (m)", f"[{min(alts):.2f}, {max(alts):.2f}]")

        # Surface the calibration provenance explicitly: for uncalibrated video the
        # focal length is an assumption, not a measurement.
        if isinstance(source, VideoFileSource):
            table.add_row(
                "Intrinsics Provenance",
                "[yellow]assumed FoV (no embedded calibration); focal refinement gated[/yellow]",
            )

        console.print(table)
        console.print(
            Panel.fit(
                f"[bold yellow]Dataset Label:[/bold yellow] {dataset_label}\n"
                "[dim]Ground truth positions (GroundTruthAGL.csv) are stored for evaluation only and never used in SfM.[/dim]",
                border_style="yellow",
            )
        )
    return 0


def _handle_keyframes_command(args: argparse.Namespace) -> int:
    """Execute the keyframes extraction and filtering command."""
    logger = setup_logging()
    cfg = load_config(args.config if os.path.exists(args.config) else None)
    data_dir = args.data_dir or os.environ.get("UAV3D_DATA_DIR") or cfg.data_dir

    with timed_stage("keyframes", logger=logger):
        source = _build_source(cfg, data_dir, args)
        timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        run_dir = Path(data_dir) / "outputs" / f"run_{timestamp_str}"

        dataset_label = _resolve_dataset_label(cfg, source)
        selector = KeyframeSelector(config=cfg.keyframes)
        summary = selector.select(source=source, output_dir=run_dir, dataset_label=dataset_label)

        table = Table(title=f"Keyframe Extraction Summary: [bold cyan]{run_dir.name}[/bold cyan]", show_header=True)
        table.add_column("Metric / Property", style="bold white", width=28)
        table.add_column("Value", style="green")

        table.add_row("Dataset Label", dataset_label)
        table.add_row("Frames Evaluated", str(summary.considered_count))
        table.add_row("Keyframes Selected", f"[bold green]{summary.selected_count}[/bold green]")
        table.add_row("Decode Failures", str(summary.decode_failed_count))
        table.add_row("Initial Stride", str(summary.initial_stride))
        table.add_row("Final Stride", f"{summary.final_stride} (reductions: {summary.stride_reductions})")
        table.add_row(
            "Blur Score (med / p95)",
            f"{summary.blur_dist.median_val:.1f} / {summary.blur_dist.p95_val:.1f} (min: {summary.blur_dist.min_val:.1f})",
        )
        table.add_row(
            "ORB Features (med / p95)",
            f"{summary.features_dist.median_val:.0f} / {summary.features_dist.p95_val:.0f} (min: {summary.features_dist.min_val:.0f})",
        )
        table.add_row(
            "Clipping Ratio (med / p95)",
            f"{summary.clipping_dist.median_val*100:.2f}% / {summary.clipping_dist.p95_val*100:.2f}%",
        )
        table.add_row("Keyframes JSON", str(summary.keyframes_json_path))
        table.add_row("SfM Images Directory", str(summary.sfm_images_dir))
        table.add_row("Contact Sheet", str(summary.contact_sheet_path))

        console.print(table)
    return 0


def _handle_sfm_command(args: argparse.Namespace) -> int:
    """Execute pycolmap incremental visual Structure-from-Motion."""
    logger = setup_logging()
    cfg = load_config(args.config if os.path.exists(args.config) else None)
    data_dir = args.data_dir or os.environ.get("UAV3D_DATA_DIR") or cfg.data_dir

    with timed_stage("sfm", logger=logger):
        # 1. Determine run directory
        if getattr(args, "run_dir", None):
            run_dir = Path(args.run_dir).resolve()
        else:
            outputs_dir = Path(data_dir) / "outputs"
            if not outputs_dir.exists():
                raise Uav3dError(f"Outputs directory {outputs_dir} does not exist. Run 'uav3d keyframes' first.")
            run_dirs = sorted(
                [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
                key=lambda p: p.name,
            )
            if not run_dirs:
                raise Uav3dError(f"No run directories found in {outputs_dir}. Run 'uav3d keyframes' first.")
            run_dir = run_dirs[-1]

        logger.info(f"Target SfM run directory: {run_dir}")
        images_dir = run_dir / "sfm" / "images"
        keyframes_json_path = run_dir / "keyframes.json"

        if not images_dir.exists():
            raise Uav3dError(f"Keyframe images directory {images_dir} does not exist in {run_dir}.")
        if not keyframes_json_path.exists():
            raise Uav3dError(f"keyframes.json does not exist in {run_dir}. Run 'uav3d keyframes' first.")

        # 2. Load keyframes metadata
        kf_file = KeyframesFile.model_validate_json(keyframes_json_path.read_text(encoding="utf-8"))
        all_image_names = [f"{kf.frame_id}.jpg" for kf in kf_file.keyframes]
        timestamps_map = {kf.frame_id: kf.timestamp_us for kf in kf_file.keyframes}

        # 3. Obtain camera calibration
        # Read the actual image dimensions in images_dir so the calibration matches
        # the pixels that SfM will actually see.
        actual_width, actual_height = None, None
        if all_image_names:
            first_img_path = images_dir / all_image_names[0]
            if first_img_path.exists():
                import cv2

                sample_mat = cv2.imread(str(first_img_path))
                if sample_mat is not None:
                    actual_height, actual_width = sample_mat.shape[:2]

        source = _build_source(cfg, data_dir, args)
        dataset_label = _resolve_dataset_label(cfg, source)
        is_video_run = (
            isinstance(source, VideoFileSource)
            or any(name.startswith("video_") for name in all_image_names)
            or cfg.source.type.lower() == "video"
        )

        focal_override = getattr(args, "focal_px", None) or cfg.sfm.video_focal_px

        if is_video_run:
            # Video files carry no trustworthy intrinsics, so the focal length is
            # either supplied explicitly or assumed from a configured FoV. The
            # model is SIMPLE_PINHOLE so bundle adjustment cannot split fx and fy
            # apart (which previously produced physically impossible cameras).
            w = actual_width or 1920
            h = actual_height or 1080
            calibration = estimate_video_calibration(
                width=w,
                height=h,
                fov_deg=cfg.sfm.video_fov_deg,
                focal_px=focal_override,
                source_note="cli",
            )
            cfg.sfm.camera_model = calibration.model
            if focal_override is not None:
                cfg.sfm.refine_intrinsics = False
                logger.info(
                    f"Video run with explicit focal length: SIMPLE_PINHOLE f={focal_override:.1f} px "
                    f"({calibration.width}x{calibration.height}); intrinsics locked (not refined)."
                )
            else:
                cfg.sfm.refine_intrinsics = True
                logger.info(
                    f"Video run with assumed {cfg.sfm.video_fov_deg:.1f} deg horizontal FoV: "
                    f"SIMPLE_PINHOLE f={calibration.params[0]:.1f} px ({w}x{h}). Focal refinement will be "
                    f"attempted and kept only if the result stays within "
                    f"[{cfg.sfm.video_min_fov_deg:.0f}, {cfg.sfm.video_max_fov_deg:.0f}] deg."
                )
        else:
            calibration = source.calibration()
            if actual_width and actual_height and (actual_width, actual_height) != (
                calibration.width,
                calibration.height,
            ):
                # Frames on disk differ from the calibration resolution: rescale the
                # real calibration instead of discarding it for an invented pinhole.
                calibration = scale_calibration(calibration, actual_width, actual_height)
            logger.info(f"Using camera calibration: {calibration.model} {calibration.params}")

        # 4. Optional CLI overrides
        if getattr(args, "overlap", None) is not None:
            cfg.sfm.seq_overlap = args.overlap

        # 5. Execute SfM
        runner = SfmRunner(config=cfg.sfm)
        result = runner.run(images_dir=images_dir, output_dir=run_dir, calibration=calibration)

        # 6. Export poses.json
        poses_path = run_dir / "poses.json"
        poses_file = export_poses(
            reconstruction=result.reconstruction,
            all_image_names=all_image_names,
            camera_calibration=calibration,
            output_path=poses_path,
            dataset_label=dataset_label,
            scale_factor=1.0,
            timestamps_map=timestamps_map,
            intrinsics_locked=not result.intrinsics_refined,
            initial_params=result.initial_params if result.intrinsics_refined else None,
            horizontal_fov_deg=result.final_fov_deg,
        )

        # 7. Print summary table
        table = Table(
            title=f"SfM Reconstruction Summary: [bold cyan]{run_dir.name}[/bold cyan]",
            show_header=True,
        )
        table.add_column("Metric / Property", style="bold white", width=28)
        table.add_column("Value", style="green")

        table.add_row("Input Keyframes", str(result.total_images))
        table.add_row("Registered Frames", f"[bold green]{result.num_reg_images}[/bold green] / {result.total_images}")
        rate_style = "bold green" if result.registration_rate >= 0.85 else "bold red"
        table.add_row("Registration Rate", f"[{rate_style}]{result.registration_rate*100:.1f}%[/{rate_style}] (target: >= 85%)")
        reproj_style = "bold green" if result.mean_reprojection_error_px <= 1.5 else "bold yellow"
        table.add_row(
            "Mean Reprojection Error",
            f"[{reproj_style}]{result.mean_reprojection_error_px:.3f} px[/{reproj_style}] (target: <= 1.5 px)",
        )
        table.add_row(
            "Reconstructed 3D Points",
            f"{result.reconstruction.num_points3D():,}" if hasattr(result.reconstruction, "num_points3D") else "N/A",
        )
        table.add_row(
            "Mean Track Length",
            f"{result.reconstruction.compute_mean_track_length():.2f}" if hasattr(result.reconstruction, "compute_mean_track_length") else "N/A",
        )
        table.add_row("Models Formed", str(result.num_models))
        table.add_row("Total Wall Time", f"{result.total_wall_time_s:.2f}s")
        table.add_row("Peak Memory (RSS)", f"{result.peak_rss_mb:.1f} MB")
        table.add_row("Poses File", str(poses_path))
        table.add_row("COLMAP Database", str(result.database_path))
        table.add_row("Sparse Reconstruction", str(result.sparse_dir))
        final_params = poses_file.camera.params if poses_file.camera else calibration.params
        fov_note = f", hFoV={result.final_fov_deg:.1f} deg" if result.final_fov_deg else ""
        if result.refinement_rejected:
            table.add_row(
                "Intrinsics (refinement REJECTED)",
                f"[green]{final_params}[/green] (locked at supplied values{fov_note})",
            )
        elif result.intrinsics_refined:
            table.add_row(
                "Intrinsics (REFINED by BA)",
                f"[yellow]{final_params}[/yellow] "
                f"(initial {result.initial_params}{fov_note})",
            )
        else:
            table.add_row(
                "Intrinsics (LOCKED)",
                f"[green]{final_params}[/green]{fov_note}",
            )

        console.print(table)

        if result.refinement_rejected:
            console.print(
                "[yellow]Focal-length refinement produced an implausible camera and was discarded:[/yellow] "
                + "; ".join(result.intrinsics_violations)
                + "\n[dim]The reconstruction was redone with the supplied intrinsics locked. Pass "
                "--focal-px with a known focal length for a measured camera instead of an assumed one.[/dim]"
            )
        elif result.intrinsics_violations:
            console.print(
                "[yellow]Intrinsics warnings (allow_unbounded_intrinsics=True):[/yellow] "
                + "; ".join(result.intrinsics_violations)
            )

        if result.registration_rate < 0.85:
            console.print(
                "[yellow]Warning: Registration rate below 85% KPI. Consider increasing sequential overlap or adjusting keyframe stride.[/yellow]"
            )
        else:
            console.print(
                f"[bold green]SfM Quality Gate Passed:[/bold green] Registration {result.registration_rate*100:.1f}% >= 85%, "
                f"Reprojection Error {result.mean_reprojection_error_px:.3f} px <= 1.5 px."
            )

    return 0


def _handle_sync_command(args: argparse.Namespace) -> int:
    """Execute telemetry synchronization and emit georef-ready outputs."""
    logger = setup_logging()
    cfg = load_config(args.config if os.path.exists(args.config) else None)
    data_dir = args.data_dir or os.environ.get("UAV3D_DATA_DIR") or cfg.data_dir

    with timed_stage("sync", logger=logger):
        # 1. Determine run directory
        if getattr(args, "run_dir", None):
            run_dir = Path(args.run_dir).resolve()
        else:
            outputs_dir = Path(data_dir) / "outputs"
            if not outputs_dir.exists():
                raise Uav3dError(f"Outputs directory {outputs_dir} does not exist. Run 'uav3d sfm' first.")
            run_dirs = sorted(
                [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
                key=lambda p: p.name,
            )
            if not run_dirs:
                raise Uav3dError(f"No run directories found in {outputs_dir}. Run 'uav3d sfm' first.")
            run_dir = run_dirs[-1]

        logger.info(f"Target Sync run directory: {run_dir}")
        poses_path = run_dir / "poses.json"
        keyframes_path = run_dir / "keyframes.json"

        if not poses_path.exists():
            raise Uav3dError(f"poses.json does not exist in {run_dir}. Run 'uav3d sfm' first.")
        if not keyframes_path.exists():
            raise Uav3dError(f"keyframes.json does not exist in {run_dir}. Run 'uav3d keyframes' first.")

        # 2. Load inputs
        poses_file = PosesFile.model_validate_json(poses_path.read_text(encoding="utf-8"))
        kf_file = KeyframesFile.model_validate_json(keyframes_path.read_text(encoding="utf-8"))

        # 3. Load source telemetry
        source = _build_source(cfg, data_dir, args)
        raw_telemetry = None
        if hasattr(source, "_gps_records"):
            raw_telemetry = source._gps_records

        # 4. Perform synchronization
        # Resolve the mode from the source actually in use rather than relying on
        # frame-id heuristics, so a video run can never be reported as a
        # dataset image-ID join it did not perform.
        sync_mode = getattr(args, "mode", "auto")
        if sync_mode == "auto":
            sync_mode = "zurich" if isinstance(source, ZurichSequenceSource) else "video"
        logger.info(f"Telemetry synchronization mode: {sync_mode}")
        synchronizer = TelemetrySynchronizer(config=cfg.sync)
        tel_file, poses_enriched = synchronizer.sync(
            poses=poses_file,
            keyframes=kf_file,
            raw_telemetry=raw_telemetry,
            manifest_path=Path(data_dir) / "interim" / "manifest.json",
            dataset_label=_resolve_dataset_label(cfg, source),
            mode=sync_mode,
        )

        # 5. Write outputs
        telemetry_path = run_dir / "telemetry.json"
        telemetry_path.write_text(tel_file.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Wrote synchronized telemetry to {telemetry_path}")

        poses_enriched_path = run_dir / "poses_enriched.json"
        poses_enriched_path.write_text(poses_enriched.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Wrote enriched poses to {poses_enriched_path}")

        # 6. Display Summary Table
        table = Table(
            title=f"Telemetry Synchronization Summary: [bold cyan]{run_dir.name}[/bold cyan]",
            show_header=True,
        )
        table.add_column("Metric / Property", style="bold white", width=28)
        table.add_column("Value", style="green")

        total_poses = len(poses_file.poses)
        matched = tel_file.summary.get("matched_telemetry_frames", 0) if tel_file.summary else 0
        match_rate = tel_file.summary.get("match_rate", 0.0) if tel_file.summary else 0.0

        table.add_row("Run Directory", str(run_dir))
        table.add_row("Poses Evaluated", str(total_poses))
        table.add_row("Matched Telemetry Frames", f"{matched} / {total_poses} ({match_rate*100:.1f}%)")
        table.add_row("Sync Status", f"[bold]{tel_file.sync_status}[/bold]")
        table.add_row("Offset Estimation", str(tel_file.offset_estimation))
        table.add_row(
            "Clock Offset (µs)",
            f"{tel_file.offset_us} µs" if tel_file.offset_us is not None else "N/A",
        )
        table.add_row("Recommended CRS", str(tel_file.recommended_crs))
        table.add_row("Distance Consistency Note", str(tel_file.distance_consistency_note))
        table.add_row("Telemetry JSON", str(telemetry_path))
        table.add_row("Enriched Poses JSON", str(poses_enriched_path))

        console.print(table)
    return 0


def _handle_validate_command(args: argparse.Namespace) -> int:
    """Execute the trajectory validation (ATE) pipeline command against ground truth."""
    logger = setup_logging()
    cfg = load_config(args.config if os.path.exists(args.config) else None)
    data_dir = args.data_dir or os.environ.get("UAV3D_DATA_DIR") or cfg.data_dir

    with timed_stage("validate", logger=logger):
        # 1. Resolve run directory
        if args.run_dir:
            run_dir = Path(args.run_dir).resolve()
        else:
            outputs_dir = Path(data_dir) / "outputs"
            if not outputs_dir.exists():
                raise Uav3dError(f"Outputs directory {outputs_dir} does not exist. Run 'uav3d sfm' first.")
            run_dirs = sorted(
                [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
                key=lambda p: p.name,
            )
            if not run_dirs:
                raise Uav3dError(f"No run directories found in {outputs_dir}. Run 'uav3d sfm' first.")
            run_dir = run_dirs[-1]

        logger.info(f"Target validation run directory: {run_dir}")
        poses_path = run_dir / "poses.json"
        if not poses_path.exists():
            raise Uav3dError(f"poses.json does not exist in {run_dir}. Run 'uav3d sfm' first.")

        # 2. Load poses.json (needed before ground truth resolution, because the
        # run's frame-id namespace determines whether auto-discovery is safe)
        poses_file = PosesFile.model_validate_json(poses_path.read_text(encoding="utf-8"))

        frame_ids = [p.frame_id for p in poses_file.poses]
        is_dataset_run = bool(frame_ids) and all(fid.startswith("zurich_") for fid in frame_ids)
        run_namespace = "zurich_dataset_sequence" if is_dataset_run else "video_or_other"
        logger.info(f"Run frame-id namespace: {run_namespace} (sample: {frame_ids[:1]})")

        # 3. Resolve ground truth file
        # Auto-discovery is only permitted for dataset-sequence runs. For any other
        # run (e.g. an arbitrary video) picking up a CSV that happens to sit under
        # the data directory would silently grade the trajectory against unrelated
        # ground truth, so an explicit --gt-file is required.
        gt_file = None
        gt_selection = "explicit_cli_argument"
        if getattr(args, "gt_file", None):
            gt_file = Path(args.gt_file).resolve()
            if not gt_file.exists():
                raise Uav3dError(f"Ground truth file passed with --gt-file does not exist: {gt_file}")
        elif not is_dataset_run:
            raise Uav3dError(
                f"Run '{run_dir.name}' has frame ids in the '{run_namespace}' namespace, which carries no "
                "ground-truth definition. Refusing to auto-select a GroundTruthAGL.csv under the data "
                "directory because it would belong to a different dataset. Pass --gt-file explicitly if "
                "you know the correct ground truth for this run."
            )
        else:
            # Dataset run: ask the source adapter where its own ground truth lives
            # rather than guessing fixed directory layouts.
            gt_selection = "auto_discovered_from_source"
            source = _build_source(cfg, data_dir, args)
            source_gt = getattr(source, "gt_file", None)
            if isinstance(source_gt, Path) and source_gt.is_file():
                gt_file = source_gt
            else:
                candidates = [
                    Path(data_dir) / "raw" / "zurich_mav" / "Log Files" / "GroundTruthAGL.csv",
                    Path(data_dir) / "zurich_mav" / "Log Files" / "GroundTruthAGL.csv",
                    Path(data_dir) / "Log Files" / "GroundTruthAGL.csv",
                ]
                for c in candidates:
                    if c.exists():
                        gt_file = c
                        break

        if gt_file is None or not gt_file.exists():
            raise Uav3dError(
                f"Ground truth file GroundTruthAGL.csv not found in standard paths under {data_dir}. "
                "Please specify path with --gt-file."
            )

        logger.info(f"Using ground truth file: {gt_file} (selection: {gt_selection})")
        if gt_selection.startswith("auto_discovered"):
            console.print(
                "[yellow]Note:[/yellow] ground truth was auto-discovered under the data directory "
                f"({gt_file}); pass --gt-file to pin it explicitly."
            )

        # 4. Compute ATE metrics and alignment
        ate_report = compute_ate(
            poses_file=poses_file,
            gt_records=gt_file,
            ate_threshold_m=cfg.validation.ate_threshold_meters,
            # Prefer the provenance recorded by the run itself: it is the label that
            # was attached when the poses were produced.
            dataset_label=poses_file.dataset_label or _resolve_dataset_label(cfg, None),
            gt_provenance={
                "file": str(gt_file),
                "selection": gt_selection,
                "run_frame_id_namespace": run_namespace,
                "alignment": cfg.validation.alignment,
            },
        )

        # 5. Export ate_report.json
        ate_json_path = run_dir / "ate_report.json"
        ate_json_path.write_text(ate_report.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Wrote ATE report to {ate_json_path}")

        # 6. Generate plots
        plots = generate_validation_plots(ate_report, output_dir=run_dir)

        # 7. Generate SP1_REPORT.md
        sp1_report_path = generate_sp1_report(run_dir=run_dir, report=ate_report, config=cfg, plots=plots)
        logger.info(f"Generated SP-1 Markdown report at {sp1_report_path}")

        # 8. Display Rich summary table
        table = Table(
            title=f"SP-1 Trajectory ATE Validation: [bold cyan]{run_dir.name}[/bold cyan]",
            show_header=True,
        )
        table.add_column("Metric / Parameter", style="bold white", width=28)
        table.add_column("Value", style="green")

        m = ate_report.metrics
        gt_info = ate_report.ground_truth or {}
        table.add_row("Run Directory", str(run_dir))
        table.add_row("Dataset Provenance", ate_report.dataset_label)
        table.add_row("Ground Truth File", str(gt_info.get("file", "unknown")))
        table.add_row("Ground Truth Selection", str(gt_info.get("selection", "unknown")))
        table.add_row("Matched Frames (SfM & GT)", f"{m.num_matched_frames} / {m.num_registered_frames}")

        cond_color = "yellow" if m.fit_condition != "ok" else "green"
        table.add_row("Geometry Condition Check", f"[{cond_color}][bold]{m.fit_condition}[/bold][/{cond_color}]")
        table.add_row("Similarity Scale Factor (s)", f"{m.scale_factor:.4f}")
        table.add_row("ATE RMSE (GT meters)", f"[bold]{m.rmse_m:.4f} m[/bold]")
        table.add_row("ATE Median / P95", f"{m.median_m:.4f} m / {m.p95_m:.4f} m")
        table.add_row("ATE Max Error", f"{m.max_m:.4f} m")

        if m.registration_rate is not None:
            reg_style = "bold green" if m.registration_rate >= 0.85 else "bold red"
            table.add_row("Registration Rate (KPI >= 85%)", f"[{reg_style}]{m.registration_rate * 100:.1f}%[/{reg_style}]")

        if m.mean_reprojection_error_px is not None:
            reproj_style = "bold green" if m.mean_reprojection_error_px <= 1.5 else "bold red"
            table.add_row("Reprojection Error (KPI <= 1.5px)", f"[{reproj_style}]{m.mean_reprojection_error_px:.3f} px[/{reproj_style}]")

        if ate_report.status == "completed":
            table.add_row("Validation Status", f"[bold green]{ate_report.status.upper()}[/bold green]")
        else:
            table.add_row("Validation Status", f"[bold yellow]{ate_report.status.upper()}[/bold yellow]")
        table.add_row("Report Artifacts", f"ate_report.json, SP1_REPORT.md, {', '.join(plots.values())}")

        console.print(table)
    return 0


def _resolve_dataset_label(cfg: PipelineConfig, source: object | None) -> str:
    """Return the provenance label for a source, downgrading locally generated data.

    ``ZurichSequenceSource`` exposes ``dataset_label()``, which appends a warning
    suffix when the dataset directory carries a synthetic-sample marker. This
    keeps sample-data runs from ever being labelled as real dataset results.
    """
    labeller = getattr(source, "dataset_label", None)
    if callable(labeller):
        return str(labeller(cfg.dataset_label))
    return cfg.dataset_label


def create_parser() -> argparse.ArgumentParser:
    """Create root argument parser with all pipeline subcommands."""
    parser = argparse.ArgumentParser(
        prog="uav3d",
        description="Single-pass UAV video to 3D pipeline (SIH 26158 SP-1)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show package version and exit",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="Pipeline Commands",
        description="Select a pipeline stage to execute",
        required=True,
    )

    # 1. inspect
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect raw video/sequence metadata, calibration, and timestamps",
    )
    _add_common_arguments(inspect_parser)

    # 2. keyframes
    keyframes_parser = subparsers.add_parser(
        "keyframes",
        help="Extract and select sharp keyframes from ingested video/images",
    )
    _add_common_arguments(keyframes_parser)

    # 3. sfm
    sfm_parser = subparsers.add_parser(
        "sfm",
        help="Run pycolmap visual Structure-from-Motion camera pose estimation",
    )
    _add_common_arguments(sfm_parser)
    sfm_parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Explicit run directory containing sfm/images and keyframes.json (defaults to latest run)",
    )
    sfm_parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Override sequential matching overlap window",
    )
    sfm_parser.add_argument(
        "--focal-px",
        type=float,
        default=None,
        help=(
            "Known focal length in pixels for video runs. Locks intrinsics instead of "
            "assuming a field of view and refining the focal length"
        ),
    )

    # 4. sync
    sync_parser = subparsers.add_parser(
        "sync",
        help="Synchronize onboard telemetry logs with keyframes",
    )
    _add_common_arguments(sync_parser)
    sync_parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Explicit run directory containing poses.json and keyframes.json (defaults to latest run)",
    )
    sync_parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "zurich", "video"],
        default="auto",
        help="Telemetry synchronization mode (auto, zurich, or video)",
    )

    # 5. validate
    validate_parser = subparsers.add_parser(
        "validate",
        help="Evaluate estimated camera poses against ground truth (ATE metric)",
    )
    _add_common_arguments(validate_parser)
    validate_parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Explicit run directory containing poses.json (defaults to latest run)",
    )
    validate_parser.add_argument(
        "--gt-file",
        type=str,
        default=None,
        help="Path to GroundTruthAGL.csv (defaults to dataset location)",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint with robust error handling."""
    parser = create_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0

    try:
        if args.command == "inspect":
            return _handle_inspect_command(args)
        if args.command == "keyframes":
            return _handle_keyframes_command(args)
        if args.command == "sfm":
            return _handle_sfm_command(args)
        if args.command == "sync":
            return _handle_sync_command(args)
        if args.command == "validate":
            return _handle_validate_command(args)
        parser.print_help()
        return 1
    except Uav3dError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1
    except Exception as exc:
        err_console.print(f"[bold red]Unexpected failure:[/bold red] {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
