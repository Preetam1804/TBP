"""Command-line interface for the uav3d pipeline."""

import argparse
import os
import sys
from collections.abc import Sequence

from rich.console import Console

from uav3d import __version__
from uav3d.config import load_config
from uav3d.exceptions import Uav3dError
from uav3d.logging_config import setup_logging, timed_stage

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


def _handle_stub_command(command_name: str, args: argparse.Namespace) -> int:
    """Execute stub for unimplemented pipeline stage."""
    logger = setup_logging()
    # Validate and load config fail-fast
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
