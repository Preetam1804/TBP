"""Video file source adapter utilizing OpenCV VideoCapture."""

import json
import logging
import math
import re
from collections.abc import Iterator
from pathlib import Path

import cv2

from uav3d.exceptions import DataError, Uav3dError
from uav3d.models.frames import CameraCalibration, FrameRecord
from uav3d.sources.base import ImageSource, SourceMetadata

logger = logging.getLogger("uav3d.sources.video")

# A plausible focal length, relative to the frame width. Values outside this
# band are almost certainly a units mistake (e.g. millimetres instead of
# pixels) rather than a real camera.
_FOCAL_MIN_REL_WIDTH = 0.2
_FOCAL_MAX_REL_WIDTH = 5.0


class VideoFileSource(ImageSource):
    """Ingestion source for video files (e.g. MP4, AVI, MOV) using OpenCV."""

    def __init__(
        self,
        video_path: str | Path,
        start_frame: int = 0,
        end_frame: int | None = None,
        focal_px: float | None = None,
    ) -> None:
        self.video_path = Path(video_path).resolve()
        if not self.video_path.exists():
            raise DataError(f"Video file does not exist: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise DataError(f"Could not open video file with OpenCV: {self.video_path}")

        try:
            self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
            self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        finally:
            cap.release()

        # Explicit override (config `source.focal_px` or CLI `--focal-px`) is
        # the caller's stated ground truth, so a bad value fails loudly here
        # instead of silently falling back to the guess. The sidecar path
        # below is the lenient one: it is only ever warned about and ignored.
        if focal_px is not None:
            self._validate_focal_px(focal_px, origin="explicit focal_px override")
        self._focal_px_override = float(focal_px) if focal_px is not None else None

        if self._frame_count > 0:
            if start_frame < 0 or start_frame >= self._frame_count:
                raise DataError(
                    f"start_frame {start_frame} outside video range 0..{self._frame_count - 1}"
                )
            end = self._frame_count if end_frame is None else int(end_frame)
            end = min(end, self._frame_count)
            if end <= start_frame:
                raise DataError(f"end_frame {end} must be greater than start_frame {start_frame}")
            self._start_frame = int(start_frame)
            self._end_frame = end
        else:
            self._start_frame = 0
            self._end_frame = None

        self._frame_cache: dict[str, FrameRecord] = {}

    def metadata(self) -> SourceMetadata:
        """Return metadata summary for the video (respecting the requested frame range)."""
        count = self._end_frame - self._start_frame
        first_id = f"video_{self._start_frame:06d}" if count > 0 else None
        last_id = f"video_{self._end_frame - 1:06d}" if count > 0 else None
        first_ts = int((self._start_frame / self._fps) * 1_000_000) if count > 0 else None
        last_ts = int(((self._end_frame - 1) / self._fps) * 1_000_000) if count > 0 else None

        return SourceMetadata(
            name=self.video_path.name,
            frame_count=count,
            fps=round(self._fps, 2),
            width=self._width,
            height=self._height,
            first_frame_id=first_id,
            last_frame_id=last_id,
            first_timestamp_us=first_ts,
            last_timestamp_us=last_ts,
        )

    def _validate_focal_px(self, focal_px: float, origin: str) -> None:
        """Check a focal length against the width-relative sane range.

        Raises DataError for non-numeric, non-finite, non-positive, or
        out-of-range values. Used for the explicit override, where a bad
        value must fail the run; the sidecar path catches this and only
        warns instead (see :meth:`_sidecar_focal_px`).
        """
        if isinstance(focal_px, bool) or not isinstance(focal_px, (int, float)):
            raise DataError(f"{origin}: focal_px must be a number, got {focal_px!r}")
        value = float(focal_px)
        if not math.isfinite(value) or value <= 0:
            raise DataError(f"{origin}: focal_px must be positive and finite, got {focal_px!r}")
        lo = _FOCAL_MIN_REL_WIDTH * self._width
        hi = _FOCAL_MAX_REL_WIDTH * self._width
        if not lo <= value <= hi:
            raise DataError(
                f"{origin}: focal_px {value} px is outside the sane range "
                f"{lo:.1f}..{hi:.1f} px (0.2*width..5*width for {self._width} px wide "
                "frames); check the units (pixels, not millimetres)"
            )

    def _sidecar_path(self) -> Path:
        """Path of the optional ``<video-stem>.meta.json`` sidecar next to the video."""
        return self.video_path.with_suffix(".meta.json")

    def _sidecar_focal_px(self) -> float | None:
        """Read ``focal_px`` from the sidecar, or ``None`` if absent/invalid.

        Never raises: a missing file, malformed JSON, a non-numeric value,
        or an out-of-range value is logged as a warning and the caller falls
        back to the next precedence level (the 0.8*width guess).
        """
        sidecar = self._sidecar_path()
        if not sidecar.exists():
            return None
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring unreadable focal sidecar %s: %s", sidecar.name, exc)
            return None
        if not isinstance(payload, dict) or "focal_px" not in payload:
            logger.warning(
                "Ignoring focal sidecar %s: no 'focal_px' field", sidecar.name
            )
            return None
        try:
            self._validate_focal_px(payload["focal_px"], origin=f"sidecar {sidecar.name}")
        except DataError as exc:
            logger.warning("Ignoring invalid focal in sidecar %s: %s", sidecar.name, exc)
            return None
        return float(payload["focal_px"])

    def calibration(self) -> CameraCalibration:
        """Return camera calibration for the video stream.

        Focal precedence (highest first):

        1. explicit override — ``VideoFileSource(focal_px=...)``, from config
           ``source.focal_px`` or CLI ``--focal-px``; source
           ``"video_config_override"``.
        2. sidecar — ``focal_px`` in ``<video-stem>.meta.json`` next to the
           video; source ``"video_meta_sidecar:<filename>"``. Invalid values
           are warned about and ignored, never fatal.
        3. guess — ``0.8 * width`` (FOV ~60 deg); source
           ``"video_estimate_<video-filename>"`` (unchanged default).
        """
        if self._focal_px_override is not None:
            focal = self._focal_px_override
            source = "video_config_override"
        else:
            sidecar_focal = self._sidecar_focal_px()
            if sidecar_focal is not None:
                focal = sidecar_focal
                source = f"video_meta_sidecar:{self._sidecar_path().name}"
            else:
                # Generic pinhole camera model based on frame dimensions (FOV ~60 deg)
                focal = float(self._width) * 0.8
                source = f"video_estimate_{self.video_path.name}"
        fx = fy = float(focal)
        cx = float(self._width) / 2.0
        cy = float(self._height) / 2.0
        return CameraCalibration(
            model="PINHOLE",
            width=self._width,
            height=self._height,
            params=[fx, fy, cx, cy],
            source=source,
        )

    def iter_frames(self) -> Iterator[FrameRecord]:
        """Lazily iterate over video frame records using PTS microsecond timestamps."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise DataError(f"Could not open video file: {self.video_path}")

        try:
            if self._start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(self._start_frame))
            frame_idx = self._start_frame
            while self._end_frame is None or frame_idx < self._end_frame:
                # Retrieve current position timestamp in milliseconds before grabbing
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                ret = cap.grab()
                if not ret:
                    break

                # Convert to integer microseconds
                if pos_msec > 0:
                    pts_us = int(pos_msec * 1000)
                else:
                    pts_us = int((frame_idx / self._fps) * 1_000_000)

                frame_id = f"video_{frame_idx:06d}"
                record = FrameRecord(
                    frame_id=frame_id,
                    timestamp_us=pts_us,
                    timestamp_source="video_pts",
                    file_path=str(self.video_path),
                    width=self._width,
                    height=self._height,
                )
                self._frame_cache[frame_id] = record
                yield record
                frame_idx += 1
        finally:
            cap.release()

    def get_frame(self, frame_id: str) -> FrameRecord:
        """Retrieve a specific frame by zero-padded ID (e.g. 'video_000042')."""
        if frame_id in self._frame_cache:
            return self._frame_cache[frame_id]

        match = re.search(r"(\d+)$", frame_id)
        if not match:
            raise Uav3dError(f"Invalid frame_id format: '{frame_id}'")

        idx = int(match.group(1))
        end = self._end_frame if self._end_frame is not None else self._frame_count
        if idx < self._start_frame or idx >= end:
            raise DataError(
                f"Frame index {idx} outside selected range "
                f"{self._start_frame}..{max(self._start_frame, end - 1)} (total: {self._frame_count})"
            )

        pts_us = int((idx / self._fps) * 1_000_000)
        record = FrameRecord(
            frame_id=frame_id,
            timestamp_us=pts_us,
            timestamp_source="video_pts",
            file_path=str(self.video_path),
            width=self._width,
            height=self._height,
        )
        self._frame_cache[frame_id] = record
        return record
