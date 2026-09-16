"""Camera calibration mathematics shared by sources, CLI, and the SfM runner.

This module exists so that the "assumed intrinsics" question has exactly one
answer in the codebase. Previously the video source and the CLI each carried a
different magic multiplier (0.8x and 0.9x image width), which produced
inconsistent calibrations between ``uav3d inspect`` and ``uav3d sfm``.
"""

import logging
import math

from uav3d.exceptions import ConfigError
from uav3d.models.frames import CameraCalibration

logger = logging.getLogger("uav3d.calibration")

#: Assumed horizontal field of view (degrees) for an uncalibrated video source.
#: 60 deg is a deliberately conservative consumer-drone/action-camera default.
DEFAULT_VIDEO_FOV_DEG = 60.0

#: Plausible horizontal FoV window used to validate post-bundle-adjustment
#: focal lengths for video runs. Narrower than 35 deg (>= 2.4x zoom) or wider
#: than 100 deg (fisheye-like) is not credible for a rectilinear UAV camera.
MIN_VIDEO_FOV_DEG = 35.0
MAX_VIDEO_FOV_DEG = 100.0

#: Maximum allowed principal-point drift from image center, as a fraction of
#: the corresponding image dimension, when principal point refinement is on.
MAX_PRINCIPAL_POINT_OFFSET_FRACTION = 0.10


#: COLMAP camera models -> required number of parameters.
COLMAP_PARAM_COUNTS: dict[str, int] = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12,
}


def camera_params_expected_count(model: str) -> int | None:
    """Return the parameter count COLMAP requires for a camera model.

    Args:
        model: Camera model name, with or without a ``CameraModelId.`` prefix.

    Returns:
        Expected parameter count, or None for models this table does not know
        (callers should then defer validation to COLMAP).
    """
    normalized = model.upper().replace("CAMERAMODELID.", "").strip()
    return COLMAP_PARAM_COUNTS.get(normalized)


def focal_from_fov(width: int, fov_deg: float) -> float:
    """Convert a horizontal field of view to a focal length in pixels."""
    if not 0.0 < fov_deg < 180.0:
        raise ConfigError(f"Horizontal field of view must be in (0, 180) degrees, got {fov_deg}")
    return (float(width) / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def horizontal_fov_deg(params: list[float] | tuple[float, ...], width: int, fy: float | None = None) -> float:
    """Compute the horizontal field of view (degrees) implied by intrinsics.

    Args:
        params: COLMAP-style parameter list. ``params[0]`` is fx for both
            SIMPLE_PINHOLE and PINHOLE models; ``params[1]`` is fy only for
            models that carry a separate focal length.
        width: Image width in pixels used for the sensor half-width.
        fy: Optional explicit fy; enables an isotropic-focal check by callers.

    Returns:
        Horizontal field of view in degrees (NaN if fx is not positive).
    """
    fx = float(params[0])
    if fx <= 0.0:
        return float("nan")
    return math.degrees(2.0 * math.atan((float(width) / 2.0) / fx))


def estimate_video_calibration(
    width: int,
    height: int,
    fov_deg: float = DEFAULT_VIDEO_FOV_DEG,
    focal_px: float | None = None,
    source_note: str = "assumed_default_fov",
) -> CameraCalibration:
    """Build a calibration for a video file that carries no usable intrinsics.

    Always returns a **SIMPLE_PINHOLE** model (single shared focal length), so
    that bundle adjustment cannot independently drift fx and fy apart, which is
    physically impossible for a real camera and was previously observed in
    exported PosesFile camera blocks.

    Args:
        width: Video width in pixels.
        height: Video height in pixels.
        fov_deg: Assumed horizontal FoV, used when ``focal_px`` is None.
        focal_px: Explicit focal length in pixels (overrides ``fov_deg``).
        source_note: Provenance string recorded in the calibration.

    Returns:
        CameraCalibration with params ``[f, cx, cy]``.
    """
    if focal_px is not None:
        if focal_px <= 0.0:
            raise ConfigError(f"Explicit focal length must be positive, got {focal_px}")
        f = float(focal_px)
        origin = f"user_supplied_focal_{source_note}"
    else:
        f = focal_from_fov(width, fov_deg)
        origin = f"{source_note}_fov{float(fov_deg):.1f}deg"

    return CameraCalibration(
        model="SIMPLE_PINHOLE",
        width=int(width),
        height=int(height),
        params=[f, float(width) / 2.0, float(height) / 2.0],
        source=f"video_estimate_{origin}",
    )


def scale_calibration(calibration: CameraCalibration, width: int, height: int) -> CameraCalibration:
    """Rescale an existing calibration to a new image resolution.

    Used when the frames on disk are not the resolution the calibration was
    captured at (e.g. a resized sequence). Focal lengths and principal point
    scale with the respective axis; the camera model and its distortion
    coefficients are preserved, which is strictly better than discarding a
    real calibration and inventing a pinhole guess.

    Args:
        calibration: Source calibration in its native resolution.
        width: Target image width in pixels.
        height: Target image height in pixels.

    Returns:
        A new CameraCalibration at the requested resolution.
    """
    if (width, height) == (calibration.width, calibration.height):
        return calibration

    sx = float(width) / float(calibration.width)
    sy = float(height) / float(calibration.height)

    params = list(calibration.params)
    model = calibration.model.upper()

    if model == "SIMPLE_PINHOLE":
        # params = [f, cx, cy]
        params[0] = params[0] * sx
        params[1] = params[1] * sx
        params[2] = params[2] * sy
    elif model in {"PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE", "RADIAL", "SIMPLE_RADIAL"}:
        # params = [fx, fy, cx, cy, <extra...>] for all of these models
        params[0] = params[0] * sx
        params[1] = params[1] * sy
        params[2] = params[2] * sx
        params[3] = params[3] * sy
    else:
        raise ConfigError(
            f"Cannot rescale calibration for unsupported camera model '{calibration.model}'. "
            "Provide a calibration matching the input resolution."
        )

    logger.info(
        f"Rescaled {calibration.model} calibration from {calibration.width}x{calibration.height} "
        f"to {width}x{height} (sx={sx:.4f}, sy={sy:.4f})"
    )
    return CameraCalibration(
        model=calibration.model,
        width=int(width),
        height=int(height),
        params=params,
        source=f"{calibration.source} (rescaled to {width}x{height})",
    )


def check_intrinsics_sanity(
    params: list[float],
    model: str,
    width: int,
    height: int,
    min_fov_deg: float = MIN_VIDEO_FOV_DEG,
    max_fov_deg: float = MAX_VIDEO_FOV_DEG,
) -> list[str]:
    """Return a list of human-readable violations for a refined calibration.

    An empty list means the intrinsics are physically plausible. Used as a
    post-bundle-adjustment gate: focal lengths must stay inside the configured
    FoV window and a refined principal point must stay near the image center.

    Args:
        params: Final COLMAP camera parameters.
        model: Camera model name (e.g. SIMPLE_PINHOLE, PINHOLE).
        width: Working image width in pixels.
        height: Working image height in pixels.
        min_fov_deg: Lower credible horizontal FoV bound.
        max_fov_deg: Upper credible horizontal FoV bound.

    Returns:
        List of violation strings (empty when sane).
    """
    violations: list[str] = []
    normalized = model.upper().replace("CAMERAMODELID.", "")

    if normalized == "SIMPLE_PINHOLE":
        fx = fy = float(params[0])
        cx, cy = float(params[1]), float(params[2])
    else:
        fx, fy = float(params[0]), float(params[1])
        cx, cy = float(params[2]), float(params[3])

    for label, value in (("fx", fx), ("fy", fy)):
        if not math.isfinite(value) or value <= 0.0:
            violations.append(f"{label}={value} is not a positive finite focal length")

    if violations:
        return violations

    fov = horizontal_fov_deg(params, width)
    if math.isfinite(fov) and not (min_fov_deg <= fov <= max_fov_deg):
        violations.append(
            f"implied horizontal FoV {fov:.1f} deg is outside the credible window "
            f"[{min_fov_deg:.1f}, {max_fov_deg:.1f}] deg (fx={fx:.1f} px, fy={fy:.1f} px)"
        )

    if normalized in {"PINHOLE", "OPENCV", "FULL_OPENCV", "SIMPLE_RADIAL", "RADIAL", "OPENCV_FISHEYE"}:
        rel_diff = abs(fx - fy) / max(fx, fy)
        if rel_diff > 0.05:
            violations.append(
                f"fx={fx:.1f} px and fy={fy:.1f} px differ by {rel_diff * 100:.1f}% "
                "(> 5%); a rectilinear camera must have near-isotropic focal lengths"
            )

    max_dx = MAX_PRINCIPAL_POINT_OFFSET_FRACTION * float(width)
    max_dy = MAX_PRINCIPAL_POINT_OFFSET_FRACTION * float(height)
    if abs(cx - width / 2.0) > max_dx or abs(cy - height / 2.0) > max_dy:
        violations.append(
            f"principal point ({cx:.1f}, {cy:.1f}) is more than "
            f"{MAX_PRINCIPAL_POINT_OFFSET_FRACTION * 100:.0f}% away from image center "
            f"({width / 2.0:.1f}, {height / 2.0:.1f})"
        )

    return violations
