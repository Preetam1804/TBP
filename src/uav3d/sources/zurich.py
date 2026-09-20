"""Zurich Urban MAV dataset adapter implementing ImageSource protocol."""

import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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

# Documented fallback intrinsics for the GoPro Hero4 at 1080p (used only when explicitly allowed).
DEFAULT_INTRINSICS_PINHOLE = [1060.0, 1060.0, 960.0, 540.0]

# Accepted key spellings inside calibration_data.npz (the file is not documented column-by-column).
_CAMERA_MATRIX_KEYS = ("camera_matrix", "camera_mat", "cameramatrix", "k", "mtx", "intrinsics")
_DISTORTION_KEYS = ("distortion_coefficients", "distortion", "dist_coeffs", "dist", "d")

# COLMAP camera model -> parameterisation, derived from the distortion coefficient count.
_MODEL_BY_DISTORTION_COUNT = {
    0: "PINHOLE",
    4: "OPENCV",
    5: "FULL_OPENCV",
    8: "FULL_OPENCV",
}


def _find_npz_key(keys: tuple[str, ...], available: list[str]) -> str | None:
    """Return the first available key matching a known spelling (case/format tolerant)."""
    normalised = {name.lower().replace("-", "_").replace(" ", "_"): name for name in available}
    for key in keys:
        if key in normalised:
            return normalised[key]
    for key in keys:
        for normalised_name, original in normalised.items():
            if key in normalised_name:
                return original
    return None


def build_camera_calibration(
    camera_matrix: Any,
    distortion: Any,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    source: str = "calibration_data.npz",
) -> CameraCalibration:
    """Build a CameraCalibration from a 3x3 intrinsic matrix and distortion coefficients.

    Maps the OpenCV convention (k1, k2, p1, p2[, k3, k4, k5, k6]) onto the closest COLMAP
    camera model, and fails loudly for coefficient counts that cannot be represented.

    Args:
        camera_matrix: 3x3 intrinsic matrix (list/ndarray).
        distortion: Distortion coefficient sequence (may be empty or all zeros for pinhole).
        width: Image width in pixels.
        height: Image height in pixels.
        source: Provenance label recorded in the output schema.

    Returns:
        CameraCalibration with COLMAP-compatible params.
    """
    k = np.asarray(camera_matrix, dtype=np.float64)
    if k.shape != (3, 3):
        raise DataError(f"Camera matrix must be 3x3, got shape {k.shape}")

    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise DataError(
            f"Calibration contains non-positive focal lengths (fx={fx}, fy={fy}); the file is not a valid camera matrix"
        )

    coeffs = np.asarray(distortion, dtype=np.float64).flatten() if distortion is not None else np.zeros(0)
    coeffs = np.array([c for c in coeffs if np.isfinite(c)], dtype=np.float64)

    if coeffs.size == 0 or np.allclose(coeffs, 0.0):
        model = "PINHOLE"
        params = [fx, fy, cx, cy]
    elif coeffs.size == 4:
        model = "OPENCV"
        params = [fx, fy, cx, cy, *coeffs.tolist()]
    elif coeffs.size == 5:
        # OpenCV orders k1, k2, p1, p2, k3 -> COLMAP FULL_OPENCV expects k1, k2, p1, p2, k3, k4, k5, k6
        model = "FULL_OPENCV"
        params = [fx, fy, cx, cy, *coeffs.tolist(), 0.0, 0.0, 0.0]
    elif coeffs.size == 8:
        # OpenCV orders k1, k2, p1, p2, k3, k4, k5, k6 -> COLMAP FULL_OPENCV is identical
        model = "FULL_OPENCV"
        params = [fx, fy, cx, cy, *coeffs.tolist()]
    else:
        raise DataError(
            f"Unsupported distortion coefficient count {coeffs.size} in {source}; "
            "expected 0 (pinhole), 4 (k1,k2,p1,p2), 5 (+k3) or 8 (k1..k6)"
        )

    logger.info(f"Camera calibration from {source}: model={model}, params={[round(p, 4) for p in params]}")
    return CameraCalibration(
        model=model,
        width=width,
        height=height,
        params=params,
        source=source,
    )


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
        allow_default_intrinsics: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        self.start_id = max(0, start_id)
        # Cap subset count to 300 per specification
        self.count = min(300, max(1, count))
        self.stride = max(1, stride)
        self.allow_default_intrinsics = allow_default_intrinsics

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

        # Obtain relative path from repo cwd, dataset_dir, or absolute path
        try:
            rel_path = str(img_path.relative_to(Path.cwd()))
        except ValueError:
            try:
                rel_path = str(img_path.relative_to(self.dataset_dir))
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
        """Load intrinsic calibration from calibration_data.npz.

        The npz key names are not formally documented, so several known spellings are accepted
        and the available keys are logged. If no camera matrix can be found the run fails loudly
        (``DataError``) unless ``allow_default_intrinsics`` was explicitly enabled, in which case
        the documented GoPro Hero4 1080p fallback is used and clearly labelled as such.
        """
        if not self.calib_file or not self.calib_file.exists():
            if self.allow_default_intrinsics:
                logger.warning(
                    "No calibration_data.npz found; falling back to the documented GoPro Hero4 defaults "
                    f"{DEFAULT_INTRINSICS_PINHOLE} (source.allow_default_intrinsics=true)"
                )
                return CameraCalibration(
                    model="PINHOLE",
                    width=DEFAULT_WIDTH,
                    height=DEFAULT_HEIGHT,
                    params=list(DEFAULT_INTRINSICS_PINHOLE),
                    source="zurich_default_intrinsics",
                )
            raise DataError(
                f"No calibration_data.npz found in {self.dataset_dir}. Provide the dataset calibration file, "
                "or set source.allow_default_intrinsics: true to use the documented GoPro Hero4 defaults."
            )

        try:
            with np.load(self.calib_file) as npz:
                available_keys = list(npz.files)
                logger.info(f"calibration_data.npz keys: {available_keys}")

                matrix_key = _find_npz_key(_CAMERA_MATRIX_KEYS, available_keys)
                if matrix_key is None:
                    raise DataError(
                        f"No camera matrix found in {self.calib_file.name}; available keys: {available_keys}. "
                        f"Expected one of {list(_CAMERA_MATRIX_KEYS)}"
                    )
                camera_matrix = npz[matrix_key]

                dist_key = _find_npz_key(_DISTORTION_KEYS, available_keys)
                distortion = npz[dist_key] if dist_key is not None else None
                if dist_key is None:
                    logger.info(
                        f"No distortion coefficients found in {self.calib_file.name} "
                        f"(keys: {available_keys}); using a pinhole model"
                    )
                else:
                    logger.info(f"Using '{matrix_key}' and '{dist_key}' from {self.calib_file.name}")

            return build_camera_calibration(
                camera_matrix=camera_matrix,
                distortion=distortion,
                width=DEFAULT_WIDTH,
                height=DEFAULT_HEIGHT,
                source=str(self.calib_file.name),
            )
        except DataError:
            raise
        except (OSError, ValueError, KeyError) as exc:
            raise DataError(f"Could not load calibration from {self.calib_file}: {exc}") from exc

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
    def gps_records(self) -> dict[int, ZurichGpsRecord]:
        """Onboard GPS telemetry keyed by dataset image id (public accessor)."""
        return self._gps_records

    @property
    def ground_truth_records(self) -> dict[int, ZurichGroundTruthRecord]:
        """Ground truth positions keyed by dataset image id (EVALUATION ONLY)."""
        return self._gt_records

    @property
    def total_gps_count(self) -> int:
        """Count of loaded GPS records."""
        return len(self._gps_records)

    @property
    def total_gt_count(self) -> int:
        """Count of loaded Ground Truth records."""
        return len(self._gt_records)
