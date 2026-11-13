"""Persist and reload the source camera calibration for a pipeline run.

The keyframes stage is the only stage that reads the source. It is also the only
stage that can describe the camera, since the intrinsics come from the source
(either dataset metadata or the video estimate). The SfM and dense stages work
entirely from files inside the run directory, so they must not need the original
video to be present -- re-running SfM on an archived run would otherwise fail
with "Video file not found".

``calibration.json`` is that hand-off: written next to ``keyframes.json``,
validated on read, and used whenever a stage cannot build a source.
"""

from __future__ import annotations

import json
from pathlib import Path

from uav3d.exceptions import DataError
from uav3d.models.frames import CameraCalibration

CALIBRATION_FILENAME = "calibration.json"


def write_calibration(calibration: CameraCalibration, run_dir: str | Path) -> Path:
    """Write ``calibration`` into ``run_dir`` and return its path."""
    dest = Path(run_dir) / CALIBRATION_FILENAME
    payload = {
        "schema_version": "1.0.0",
        "model": calibration.model,
        "width": int(calibration.width),
        "height": int(calibration.height),
        "params": [float(value) for value in calibration.params],
        "source": calibration.source,
    }
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


def read_calibration(run_dir: str | Path) -> CameraCalibration | None:
    """Load ``calibration`` from ``run_dir``, or ``None`` if it was never written."""
    path = Path(run_dir) / CALIBRATION_FILENAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"Malformed calibration file {path}: {exc}") from exc
    try:
        return CameraCalibration(
            model=str(payload["model"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            params=[float(value) for value in payload["params"]],
            source=str(payload.get("source", "run_calibration")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError(f"Incomplete calibration file {path}: {exc}") from exc
