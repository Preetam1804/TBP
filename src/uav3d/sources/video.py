"""Video file source adapter utilizing OpenCV VideoCapture."""

import logging
import re
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from uav3d.exceptions import DataError, Uav3dError
from uav3d.models.frames import CameraCalibration, FrameRecord
from uav3d.sources.base import ImageSource, SourceMetadata

logger = logging.getLogger("uav3d.sources.video")

# Approximate horizontal field of view ratio used to derive focal length from frame width.
# Single source of truth for both the source-level calibration and the SfM auto-configuration.
DEFAULT_FOCAL_RATIO = 0.9

# Default bound on the in-memory FrameRecord cache (records are tiny; this prevents unbounded
# growth for long video files).
DEFAULT_CACHE_LIMIT = 8192


class VideoFileSource(ImageSource):
    """Ingestion source for video files (e.g. MP4, AVI, MOV) using OpenCV.

    The source selects a contiguous slice of frames (``start_frame`` + ``max_frames`` +
    ``stride``) so that long recordings are never fully decoded for a dev run, and
    ``iter_frames`` performs a grab-only scan (no pixel decoding) over that slice.
    """

    def __init__(
        self,
        video_path: str | Path,
        start_frame: int = 0,
        max_frames: int | None = None,
        stride: int = 1,
        cache_limit: int = DEFAULT_CACHE_LIMIT,
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

        self.start_frame = max(0, start_frame)
        self.stride = max(1, stride)
        self.cache_limit = max(16, cache_limit)

        available = max(0, self._frame_count - self.start_frame)
        self.max_frames = available if max_frames is None else min(int(max_frames), available)
        self._selected_indices: list[int] = list(
            range(self.start_frame, self.start_frame + self.max_frames, self.stride)
        )
        self._frame_cache: OrderedDict[str, FrameRecord] = OrderedDict()

        logger.info(
            f"Video slice: frames {self.start_frame}..{self._selected_indices[-1] if self._selected_indices else self.start_frame} "
            f"(stride={self.stride}, selected={len(self._selected_indices)}/{self._frame_count} of {self.video_path.name})"
        )

    # ------------------------------------------------------------------ helpers
    def _cache_put(self, record: FrameRecord) -> None:
        """Insert a record into the bounded LRU cache."""
        self._frame_cache[record.frame_id] = record
        self._frame_cache.move_to_end(record.frame_id)
        while len(self._frame_cache) > self.cache_limit:
            self._frame_cache.popitem(last=False)

    def _timestamp_for(self, index: int, pos_msec: float | None) -> tuple[int, str]:
        """Return (timestamp_us, source) using a single, validated rule.

        The deterministic baseline is ``index / fps``; a container presentation timestamp is
        accepted only when it agrees with that baseline within one frame duration plus 10 ms.
        OpenCV reports unreliable/zero PTS values before a frame has been decoded and after a
        seek, so unvalidated values are never used (and the fallback is recorded explicitly).
        """
        estimated_us = round(index / self._fps * 1_000_000.0)
        if pos_msec is not None and pos_msec > 0:
            pts_us = round(pos_msec * 1000.0)
            tolerance_us = int(1_000_000.0 / max(self._fps, 1e-6)) + 10_000
            if abs(pts_us - estimated_us) <= tolerance_us:
                return pts_us, "video_pts"
            logger.debug(
                f"Frame {index}: container PTS {pts_us} µs deviates from index/fps estimate "
                f"{estimated_us} µs beyond tolerance; using the deterministic estimate"
            )
        return estimated_us, "video_index_fps_estimate"

    def _make_record(self, index: int, pos_msec: float | None) -> FrameRecord:
        """Create a FrameRecord for a frame index using the validated timestamp rule."""
        timestamp_us, timestamp_source = self._timestamp_for(index, pos_msec)

        return FrameRecord(
            frame_id=f"video_{index:06d}",
            timestamp_us=timestamp_us,
            timestamp_source=timestamp_source,
            file_path=str(self.video_path),
            width=self._width,
            height=self._height,
        )

    # ------------------------------------------------------------------ protocol
    def metadata(self) -> SourceMetadata:
        """Return metadata summary for the selected video slice."""
        has_frames = bool(self._selected_indices)
        first_id = f"video_{self.start_frame:06d}" if has_frames else None
        last_id = f"video_{self._selected_indices[-1]:06d}" if has_frames else None
        first_ts = round(self.start_frame / self._fps * 1_000_000.0) if has_frames else None
        last_ts = (
            round(self._selected_indices[-1] / self._fps * 1_000_000.0) if has_frames else None
        )

        return SourceMetadata(
            name=self.video_path.name,
            frame_count=len(self._selected_indices),
            fps=round(self._fps, 2),
            width=self._width,
            height=self._height,
            first_frame_id=first_id,
            last_frame_id=last_id,
            first_timestamp_us=first_ts,
            last_timestamp_us=last_ts,
        )

    def calibration(self) -> CameraCalibration:
        """Return generic camera calibration estimated from the frame dimensions."""
        fx = fy = float(self._width) * DEFAULT_FOCAL_RATIO
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
        """Yield FrameRecords for the selected slice using a grab-only sequential scan.

        Grabbing advances the decoder without decoding full frames, so this pass is cheap
        even for long recordings; pixel decoding happens only where a stage needs it.
        """
        if not self._selected_indices:
            return

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise DataError(f"Could not open video file: {self.video_path}")

        try:
            index = 0
            for target in self._selected_indices:
                while index < target:
                    if not cap.grab():
                        logger.warning(f"Video ended early at frame {index}; expected frame {target}")
                        return
                    index += 1

                if not cap.grab():
                    logger.warning(f"Could not grab frame {target} from {self.video_path.name}")
                    return
                index += 1

                # Grab-only scans do not decode pixels, and OpenCV's POS_MSEC is unreliable
                # before decoding, so this pass uses the deterministic index/fps rule.
                record = self._make_record(target, None)
                self._cache_put(record)
                yield record
        finally:
            cap.release()

    def iter_frames_with_pixels(
        self,
        indices: list[int] | None = None,
    ) -> Iterator[tuple[FrameRecord, np.ndarray]]:
        """Yield (FrameRecord, BGR image) for the requested frame indices in one sequential pass.

        Args:
            indices: Frame indices to decode; defaults to the whole selected slice.
        """
        requested = sorted(set(self._selected_indices if indices is None else indices))
        if not requested:
            return

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise DataError(f"Could not open video file: {self.video_path}")

        try:
            index = 0
            for target in requested:
                while index < target:
                    if not cap.grab():
                        logger.warning(f"Video ended early at frame {index}; expected frame {target}")
                        return
                    index += 1

                ok, frame = cap.read()
                index += 1
                if not ok or frame is None:
                    logger.warning(f"Failed to decode frame {target} from {self.video_path.name}")
                    return
                # After a successful read, POS_MSEC is the PTS of the frame just decoded.
                pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))

                record = self._make_record(target, pos_msec)
                self._cache_put(record)
                yield record, frame
        finally:
            cap.release()

    def get_frame(self, frame_id: str) -> FrameRecord:
        """Retrieve a specific frame by zero-padded ID (e.g. 'video_000042')."""
        if frame_id in self._frame_cache:
            self._frame_cache.move_to_end(frame_id)
            return self._frame_cache[frame_id]

        match = re.search(r"(\d+)$", frame_id)
        if not match:
            raise Uav3dError(f"Invalid frame_id format: '{frame_id}'")

        idx = int(match.group(1))
        if idx < 0 or idx >= self._frame_count:
            raise DataError(f"Frame index {idx} out of range (total: {self._frame_count})")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise DataError(f"Could not open video file: {self.video_path}")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, _frame = cap.read()
            if not ok:
                raise DataError(f"Could not read frame {idx} from {self.video_path}")
            pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        finally:
            cap.release()

        record = self._make_record(idx, pos_msec)
        self._cache_put(record)
        return record
