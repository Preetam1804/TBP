"""Command-line interface for the uav3d pipeline."""

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from uav3d import __version__
from uav3d.config import load_config
from uav3d.exceptions import Uav3dError
from uav3d.logging_config import setup_logging, timed_stage
from uav3d.sources.video import VideoFileSource
from uav3d.sources.zurich import ZurichSequenceSource

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


def _handle_inspect_command(args: argparse.Namespace) -> int:
    """Execute the inspect pipeline command."""
    logger = setup_logging()
    cfg = load_config(args.config if os.path.exists(args.config) else None)
    data_dir = args.data_dir or os.environ.get("UAV3D_DATA_DIR") or cfg.data_dir

    with timed_stage("inspect", logger=logger):
        # Resolve dataset directory for Zurich or video path
        source_type = cfg.source.type.lower()
        if source_type == "zurich":
            # If data_dir given on CLI or env, override dataset_dir
            dataset_path = Path(args.data_dir) if args.data_dir else Path(cfg.source.dataset_dir)
            if not dataset_path.is_absolute():
                dataset_path = (Path(data_dir).parent / dataset_path).resolve() if not dataset_path.exists() else dataset_path.resolve()

            source = ZurichSequenceSource(
                dataset_dir=dataset_path,
                start_id=cfg.source.start_id,
                count=cfg.source.count,
                stride=cfg.source.stride,
            )
        elif source_type == "video":
            video_path = Path(cfg.source.video_path or (Path(data_dir) / "interim" / "synthetic.mp4"))
            source = VideoFileSource(video_path=video_path)
        else:
            raise Uav3dError(f"Unsupported source type '{source_type}'. Must be 'zurich' or 'video'.")

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

    # 4. sync
    sync_parser = subparsers.add_parser(
        "sync",
        help="Synchronize onboard telemetry logs with keyframes",
    )
    _add_common_arguments(sync_parser)

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
