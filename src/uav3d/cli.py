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

console = Console()
err_console = Console(stderr=True)

SUBCOMMANDS = ("inspect", "keyframes", "sfm", "sync", "validate")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach standard --config and --data-dir flags to a subcommand parser."""
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


def _build_source(cfg: PipelineConfig, data_dir: str, args: argparse.Namespace) -> ImageSource:
    """Build and initialize ImageSource adapter from configuration."""
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
        video_path = Path(cfg.source.video_path or (Path(data_dir) / "interim" / "synthetic.mp4"))
        return VideoFileSource(video_path=video_path)

    raise Uav3dError(f"Unsupported source type '{source_type}'. Must be 'zurich' or 'video'.")


def _handle_inspect_command(args: argparse.Namespace) -> int:
    """Execute the inspect pipeline command."""
    logger = setup_logging()
    cfg = load_config(args.config if os.path.exists(args.config) else None)
    data_dir = args.data_dir or os.environ.get("UAV3D_DATA_DIR") or cfg.data_dir

    with timed_stage("inspect", logger=logger):
        source_type = cfg.source.type.lower()
        source = _build_source(cfg, data_dir, args)
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

        console.print(table)
        console.print(
            Panel.fit(
                "[bold yellow]Dataset Label:[/bold yellow] replayed public dataset\n"
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

        selector = KeyframeSelector(config=cfg.keyframes)
        summary = selector.select(source=source, output_dir=run_dir, dataset_label=cfg.dataset_label)

        table = Table(title=f"Keyframe Extraction Summary: [bold cyan]{run_dir.name}[/bold cyan]", show_header=True)
        table.add_column("Metric / Property", style="bold white", width=28)
        table.add_column("Value", style="green")

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
        source = _build_source(cfg, data_dir, args)
        calibration = source.calibration()
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
            dataset_label=cfg.dataset_label,
            scale_factor=1.0,
            timestamps_map=timestamps_map,
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
        table.add_row(
            "Intrinsics Locked",
            f"{poses_file.camera.params if poses_file.camera else calibration.params}",
        )

        console.print(table)

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
        sync_mode = getattr(args, "mode", "auto")
        synchronizer = TelemetrySynchronizer(config=cfg.sync)
        tel_file, poses_enriched = synchronizer.sync(
            poses=poses_file,
            keyframes=kf_file,
            raw_telemetry=raw_telemetry,
            dataset_label=cfg.dataset_label,
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


def _handle_stub_command(command_name: str, args: argparse.Namespace) -> int:
    """Execute stub for unimplemented pipeline stage."""
    logger = setup_logging()
    cfg = load_config(args.config if os.path.exists(args.config) else None)
    data_dir = args.data_dir or os.environ.get("UAV3D_DATA_DIR") or cfg.data_dir

    with timed_stage(command_name, logger=logger):
        console.print(
            f"[yellow]Stage '[bold]{command_name}[/bold]' is not implemented yet.[/yellow] "
            f"(Target data directory: [cyan]{data_dir}[/cyan])"
        )
    return 0


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
        if args.command in SUBCOMMANDS:
            return _handle_stub_command(args.command, args)
        parser.print_help()
        return 1
    except Uav3dError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]Unexpected failure:[/bold red] {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
