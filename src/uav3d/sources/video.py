"""Video file source adapter utilizing OpenCV VideoCapture."""

import logging
import re
from collections.abc import Iterator
from pathlib import Path

import cv2

from uav3d.exceptions import DataError, Uav3dError
from uav3d.models.frames import CameraCalibration, FrameRecord
from uav3d.sources.base import ImageSource, SourceMetadata

logger = logging.getLogger("uav3d.sources.video")


class VideoFileSource(ImageSource):
    """Ingestion source for video files (e.g. MP4, AVI, MOV) using OpenCV."""

    def __init__(self, video_path: str | Path) -> None:
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

        self._frame_cache: dict[str, FrameRecord] = {}

    def metadata(self) -> SourceMetadata:
        """Return metadata summary for the video."""
        first_id = "video_000000" if self._frame_count > 0 else None
        last_id = f"video_{max(0, self._frame_count - 1):06d}" if self._frame_count > 0 else None
        first_ts = 0 if self._frame_count > 0 else None
        last_ts = int(((self._frame_count - 1) / self._fps) * 1_000_000) if self._frame_count > 0 else None

        return SourceMetadata(
            name=self.video_path.name,
            frame_count=self._frame_count,
            fps=round(self._fps, 2),
            width=self._width,
            height=self._height,
            first_frame_id=first_id,
            last_frame_id=last_id,
            first_timestamp_us=first_ts,
            last_timestamp_us=last_ts,
        )

    def calibration(self) -> CameraCalibration:
        """Return generic or estimated camera calibration for video stream."""
        # Generic pinhole camera model based on frame dimensions (FOV ~60 deg)
        fx = fy = float(self._width) * 0.8
        cx = float(self._width) / 2.0
        cy = float(self._height) / 2.0
        return CameraCalibration(
            model="PINHOLE",
            width=self._width,
            height=self._height,
            params=[fx, fy, cx, cy],
            source=f"video_estimate_{self.video_path.name}",
        )

    def iter_frames(self) -> Iterator[FrameRecord]:
        """Lazily iterate over video frame records using PTS microsecond timestamps."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise DataError(f"Could not open video file: {self.video_path}")

        try:
            frame_idx = 0
            while True:
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
        if idx < 0 or idx >= self._frame_count:
            raise DataError(f"Frame index {idx} out of range (total: {self._frame_count})")

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
