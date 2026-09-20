"""Quality metric computation for keyframe selection."""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("uav3d.keyframes.quality")


@dataclass(frozen=True)
class FrameQualityMetrics:
    """Computed quality metrics for a single frame."""

    blur_score: float
    brightness_mean: float
    clipping_ratio: float
    feature_count: int
    # Normalisation of the above metrics: they are computed on a grayscale frame resized to
    # `analysis_width_px` width, so absolute thresholds are only comparable at equal width.
    analysis_width_px: int
    analysis_height_px: int

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to float dictionary."""
        return asdict(self)


def compute_frame_quality(
    image_input: str | Path | np.ndarray,
    target_width: int = 640,
    orb_features: int = 2000,
) -> FrameQualityMetrics | None:
    """Compute blur, brightness, clipping, and feature count on a downscaled grayscale frame.

    Args:
        image_input: Path to the image file or in-memory BGR/grayscale numpy array.
        target_width: Approximate width to downscale for fast evaluation (default 640px).
        orb_features: Maximum ORB features to detect.

    Returns:
        FrameQualityMetrics if successfully decoded, None if unreadable or corrupted.
    """
    if isinstance(image_input, np.ndarray):
        img = image_input
        if img is None or img.size == 0:
            return None
    else:
        path_obj = Path(image_input)
        if not path_obj.exists():
            logger.warning(f"Image file does not exist: {path_obj}")
            return None

        img = cv2.imread(str(path_obj))
        if img is None or img.size == 0:
            logger.warning(f"Failed to decode image: {path_obj}")
            return None

    # Convert to grayscale if 3-channel
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # Downscale preserving aspect ratio for fast Laplacian and ORB evaluation
    h, w = gray.shape[:2]
    if w > target_width:
        scale = target_width / float(w)
        target_height = max(1, int(h * scale))
        gray_small = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_AREA)
    else:
        gray_small = gray

    # 1. Blur score: variance of Laplacian
    laplacian = cv2.Laplacian(gray_small, cv2.CV_64F)
    blur_score = float(laplacian.var())

    # 2. Brightness mean: average grayscale intensity
    brightness_mean = float(np.mean(gray_small))

    # 3. Clipping ratio: fraction of under-exposed (< 5) or over-exposed (> 250) pixels
    clipped_count = np.count_nonzero((gray_small < 5) | (gray_small > 250))
    clipping_ratio = float(clipped_count / gray_small.size)

    # 4. Feature count: number of ORB keypoints
    orb = cv2.ORB_create(nfeatures=orb_features)
    keypoints = orb.detect(gray_small, None)
    feature_count = len(keypoints)

    return FrameQualityMetrics(
        blur_score=blur_score,
        brightness_mean=brightness_mean,
        clipping_ratio=clipping_ratio,
        feature_count=feature_count,
        analysis_width_px=int(gray_small.shape[1]),
        analysis_height_px=int(gray_small.shape[0]),
    )
