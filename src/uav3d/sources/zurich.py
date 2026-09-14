"""Zurich Urban MAV dataset adapter implementing ImageSource protocol."""

import logging
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from uav3d.exceptions import DataError, Uav3dError
from uav3d.models.frames import CameraCalibration, FrameRecord
from uav3d.sources.base import ImageSource, SourceMetadata
from uav3d.telemetry.zurich_parser import (
    ZurichGpsRecord,
    ZurichGroundTruthRecord,
    parse_ground_truth_agl,
    parse_onboard_gps,
)

logger = logging.getLogger("uav3d.sources.zurich")

# Zurich Urban MAV camera default dimensions (GoPro Hero4 Black)
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080


class ZurichSequenceSource(ImageSource):
    """Adapter for the ETH RPG Zurich Urban MAV dataset.

    Provides sequential access to calibrated frames with telemetry mapping
    and evaluation ground truth.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        start_id: int = 0,
        count: int = 250,
        stride: int = 1,
    ) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        self.start_id = max(0, start_id)
        # Cap subset count to 300 per specification
        self.count = min(300, max(1, count))
        self.stride = max(1, stride)

        # 1. Validate paths and discover directory layout
        self.images_dir, self.gps_file, self.gt_file, self.calib_file = self._resolve_dataset_paths()

        # 2. Parse telemetry and ground truth defensively
        self._gps_records: dict[int, ZurichGpsRecord] = {}
        if self.gps_file and self.gps_file.exists():
            self._gps_records, _ = parse_onboard_gps(self.gps_file)

        self._gt_records: dict[int, ZurichGroundTruthRecord] = {}
        if self.gt_file and self.gt_file.exists():
            self._gt_records, _ = parse_ground_truth_agl(self.gt_file)

        # 3. Discover and numerically sort all available images
        self._all_image_ids, self._image_paths = self._discover_images()

        # 4. Filter subset by start_id, count, and stride
        self._selected_ids = self._select_subset()

        # Cache frame records
        self._frame_cache: dict[str, FrameRecord] = {}

    def _resolve_dataset_paths(self) -> tuple[Path, Path | None, Path | None, Path | None]:
        """Validate dataset structure against directory traversal and find component files."""
        if not self.dataset_dir.exists():
            raise DataError(f"Zurich dataset directory does not exist: {self.dataset_dir}")

        missing_components: list[str] = []

        # Find Images directory
        image_candidates = [
            self.dataset_dir / "MAV Images",
            self.dataset_dir / "MAV_Images",
            self.dataset_dir / "images",
            self.dataset_dir / "Images",
            self.dataset_dir,  # in case images are directly inside
        ]
        images_dir: Path | None = None
        for candidate in image_candidates:
            if candidate.is_dir() and any(candidate.glob("*.jp*g")):
                images_dir = candidate.resolve()
                break

        if images_dir is None:
            # Check if any subfolder contains images
            subfolders_with_jpg = [d for d in self.dataset_dir.iterdir() if d.is_dir() and any(d.glob("*.jp*g"))]
            if subfolders_with_jpg:
                images_dir = subfolders_with_jpg[0].resolve()
            else:
                missing_components.append("Images folder (e.g. 'MAV Images/' or '*.jpg' files)")

        # Verify no path traversal escape
        if images_dir and not images_dir.is_relative_to(self.dataset_dir):
            raise DataError(f"Security error: image path {images_dir} escapes dataset directory {self.dataset_dir}")

        # Find Log directory or CSVs
        gps_candidates = [
            self.dataset_dir / "Log Files" / "OnbordGPS.csv",
            self.dataset_dir / "Log_Files" / "OnbordGPS.csv",
            self.dataset_dir / "logs" / "OnbordGPS.csv",
            self.dataset_dir / "OnbordGPS.csv",
        ]
        gps_file: Path | None = None
        for cand in gps_candidates:
            if cand.is_file():
                gps_file = cand.resolve()
                break

        gt_candidates = [
            self.dataset_dir / "Log Files" / "GroundTruthAGL.csv",
            self.dataset_dir / "Log_Files" / "GroundTruthAGL.csv",
            self.dataset_dir / "logs" / "GroundTruthAGL.csv",
            self.dataset_dir / "GroundTruthAGL.csv",
        ]
        gt_file: Path | None = None
        for cand in gt_candidates:
            if cand.is_file():
                gt_file = cand.resolve()
                break

        # Find Calibration file
        calib_candidates = [
            self.dataset_dir / "calibration_data.npz",
            self.dataset_dir / "calibration" / "calibration_data.npz",
            self.dataset_dir / "Calibration" / "calibration_data.npz",
        ]
        calib_file: Path | None = None
        for cand in calib_candidates:
            if cand.is_file():
                calib_file = cand.resolve()
                break

        if images_dir is None:
            raise DataError(
                f"Zurich dataset directory '{self.dataset_dir}' is missing required components: "
                f"{', '.join(missing_components)}"
            )

        return images_dir, gps_file, gt_file, calib_file

    def _discover_images(self) -> tuple[list[int], dict[int, Path]]:
        """Discover image files and sort them numerically by numeric ID."""
        pattern = re.compile(r"(\d+)")
        image_map: dict[int, Path] = {}

        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for img_path in self.images_dir.glob(ext):
                match = pattern.findall(img_path.stem)
                if match:
                    numeric_id = int(match[-1])
                    image_map[numeric_id] = img_path.resolve()

        if not image_map:
            raise DataError(f"No numeric image files found in {self.images_dir}")

        sorted_ids = sorted(image_map.keys())
        return sorted_ids, image_map

    def _select_subset(self) -> list[int]:
        """Select a contiguous slice of image IDs based on start_id, count, and stride."""
        available_ids = [img_id for img_id in self._all_image_ids if img_id >= self.start_id]
        if not available_ids:
            raise DataError(
                f"No images found with id >= start_id ({self.start_id}). "
                f"Available range: [{self._all_image_ids[0]}, {self._all_image_ids[-1]}]"
            )

        strided = available_ids[:: self.stride]
        return strided[: self.count]

    def _build_frame_record(self, numeric_id: int) -> FrameRecord:
        """Construct a FrameRecord with zero-padded frame_id and non-fabricated timestamp."""
        frame_id = f"zurich_{numeric_id:06d}"
        img_path = self._image_paths[numeric_id]

        gps = self._gps_records.get(numeric_id)
        if gps is not None and gps.timestamp_us is not None:
            timestamp_us: int | None = gps.timestamp_us
            timestamp_source: str | None = "dataset_log"
        else:
            timestamp_us = None
            timestamp_source = "missing"

        # Obtain relative path from dataset_dir or absolute path
        try:
            rel_path = str(img_path.relative_to(self.dataset_dir.parent))
        except ValueError:
            rel_path = str(img_path)

        return FrameRecord(
            frame_id=frame_id,
            timestamp_us=timestamp_us,
            timestamp_source=timestamp_source,
            file_path=rel_path,
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
        )

    def metadata(self) -> SourceMetadata:
        """Return summary metadata for the selected slice."""
        first_id = f"zurich_{self._selected_ids[0]:06d}" if self._selected_ids else None
        last_id = f"zurich_{self._selected_ids[-1]:06d}" if self._selected_ids else None

        first_ts = None
        last_ts = None
        for img_id in self._selected_ids:
            if img_id in self._gps_records:
                first_ts = self._gps_records[img_id].timestamp_us
                break
        for img_id in reversed(self._selected_ids):
            if img_id in self._gps_records:
                last_ts = self._gps_records[img_id].timestamp_us
                break

        return SourceMetadata(
            name=f"zurich_urban_mav_{self.dataset_dir.name}",
            frame_count=len(self._selected_ids),
            fps=None,  # Discrete irregular image sequence
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            first_frame_id=first_id,
            last_frame_id=last_id,
            first_timestamp_us=first_ts,
            last_timestamp_us=last_ts,
        )

    def calibration(self) -> CameraCalibration:
        """Load intrinsic camera calibration from calibration_data.npz or provide default GoPro Hero4."""
        if self.calib_file and self.calib_file.exists():
            try:
                npz = np.load(self.calib_file)
                # Standard keys in calibration npz files
                if "camera_matrix" in npz:
                    K = npz["camera_matrix"]
                    fx, fy = float(K[0, 0]), float(K[1, 1])
                    cx, cy = float(K[0, 2]), float(K[1, 2])
                    params = [fx, fy, cx, cy]
                    model = "PINHOLE"
                    if "distortion_coefficients" in npz:
                        d = npz["distortion_coefficients"].flatten()
                        params.extend([float(v) for v in d[:4]])
                        model = "OPENCV"
                    return CameraCalibration(
                        model=model,
                        width=DEFAULT_WIDTH,
                        height=DEFAULT_HEIGHT,
                        params=params,
                        source=str(self.calib_file.name),
                    )
            except (OSError, ValueError, KeyError) as exc:
                logger.warning(f"Could not load calibration from {self.calib_file}: {exc}")

        # Default calibrated intrinsics for GoPro Hero4 1080p from Zurich MAV dataset
        # K = [fx: 1060.0, fy: 1060.0, cx: 960.0, cy: 540.0]
        return CameraCalibration(
            model="PINHOLE",
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            params=[1060.0, 1060.0, 960.0, 540.0],
            source="zurich_default_intrinsics",
        )

    def iter_frames(self) -> Iterator[FrameRecord]:
        """Lazily iterate over FrameRecords without loading images into memory."""
        for num_id in self._selected_ids:
            if num_id not in self._frame_cache:
                self._frame_cache[f"zurich_{num_id:06d}"] = self._build_frame_record(num_id)
            yield self._frame_cache[f"zurich_{num_id:06d}"]

    def get_frame(self, frame_id: str) -> FrameRecord:
        """Retrieve a FrameRecord by zero-padded string frame_id (e.g. 'zurich_000123')."""
        if frame_id in self._frame_cache:
            return self._frame_cache[frame_id]

        match = re.search(r"(\d+)$", frame_id)
        if not match:
            raise Uav3dError(f"Invalid frame_id format: '{frame_id}'")

        numeric_id = int(match.group(1))
        if numeric_id not in self._image_paths:
            raise DataError(f"Frame '{frame_id}' not found in dataset {self.dataset_dir}")

        record = self._build_frame_record(numeric_id)
        self._frame_cache[frame_id] = record
        return record

    def get_gps(self, frame_id: str) -> ZurichGpsRecord | None:
        """Retrieve raw GPS telemetry for a frame if logged."""
        match = re.search(r"(\d+)$", frame_id)
        if not match:
            return None
        return self._gps_records.get(int(match.group(1)))

    def get_ground_truth(self, frame_id: str) -> ZurichGroundTruthRecord | None:
        """Retrieve ground truth position (EVALUATION ONLY)."""
        match = re.search(r"(\d+)$", frame_id)
        if not match:
            return None
        return self._gt_records.get(int(match.group(1)))

    @property
    def total_gps_count(self) -> int:
        """Count of loaded GPS records."""
        return len(self._gps_records)

    @property
    def total_gt_count(self) -> int:
        """Count of loaded Ground Truth records."""
        return len(self._gt_records)
