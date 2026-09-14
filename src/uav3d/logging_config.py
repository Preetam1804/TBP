"""Logging setup with Rich console output and rotating file handler."""

import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

DEFAULT_LOG_FILE = Path("logs/uav3d.log")
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5

console = Console()


def get_log_level() -> int:
    """Retrieve logging level from UAV3D_LOG_LEVEL environment variable or default to INFO."""
    env_level = os.environ.get("UAV3D_LOG_LEVEL", "INFO").upper().strip()
    return getattr(logging, env_level, logging.INFO)


def setup_logging(
    log_level: int | str | None = None,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure and return root uav3d logger with Rich console and rotating file handlers."""
    logger = logging.getLogger("uav3d")

    # Avoid duplicate handlers on re-entry
    if logger.handlers:
        return logger

    if log_level is None:
        level = get_log_level()
    elif isinstance(log_level, str):
        level = getattr(logging, log_level.upper().strip(), logging.INFO)
    else:
        level = log_level

    logger.setLevel(level)

    # 1. Rich Console Handler
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    rich_handler.setLevel(level)
    logger.addHandler(rich_handler)

    # 2. Rotating File Handler
    target_file = Path(log_file) if log_file else DEFAULT_LOG_FILE
    try:
        target_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            target_file,
            maxBytes=DEFAULT_MAX_BYTES,
            backupCount=DEFAULT_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning(f"Could not initialize log file at {target_file}: {exc}")

    return logger


@contextmanager
def timed_stage(stage_name: str, logger: logging.Logger | None = None) -> Generator[None, None, None]:
    """Context manager to log the start, finish, and elapsed duration of a pipeline stage."""
    active_logger = logger or logging.getLogger("uav3d")
    active_logger.info(f"Stage '{stage_name}' started")
    start_time = time.perf_counter()
    try:
        yield
    except Exception as exc:
        duration = time.perf_counter() - start_time
        active_logger.error(f"Stage '{stage_name}' failed after {duration:.3f}s: {exc}")
        raise
    else:
        duration = time.perf_counter() - start_time
        active_logger.info(f"Stage '{stage_name}' completed in {duration:.3f}s")
