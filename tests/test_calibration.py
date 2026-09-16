"""Tests for calibration assumptions, rescaling, and the post-BA plausibility gate."""

import numpy as np
import pytest

from uav3d.calibration import (
    camera_params_expected_count,
    check_intrinsics_sanity,
    estimate_video_calibration,
    focal_from_fov,
    horizontal_fov_deg,
    scale_calibration,
)
from uav3d.exceptions import ConfigError
from uav3d.models.frames import CameraCalibration


class TestFocalGeometry:
    """Verify focal length <-> field of view conversions."""

    def test_focal_from_fov_60_degrees(self) -> None:
        """A 60 deg horizontal FoV on a 1920 px frame is f = 960 / tan(30 deg)."""
        f = focal_from_fov(1920, 60.0)
        assert f == pytest.approx(960.0 / np.tan(np.radians(30.0)), rel=1e-9)
        assert f == pytest.approx(1662.77, abs=0.01)

    def test_fov_round_trip(self) -> None:
        """Converting focal -> FoV -> focal returns the original value."""
        for fov in (35.0, 60.0, 90.0):
            f = focal_from_fov(1920, fov)
            assert horizontal_fov_deg([f, 960.0, 540.0], 1920) == pytest.approx(fov, abs=1e-9)

    def test_invalid_fov_rejected(self) -> None:
        """Non-physical field of view values raise ConfigError."""
        with pytest.raises(ConfigError):
            focal_from_fov(1920, 0.0)
        with pytest.raises(ConfigError):
            focal_from_fov(1920, 181.0)


class TestVideoCalibrationEstimate:
    """Verify the single source of truth for uncalibrated video intrinsics."""

    def test_default_is_simple_pinhole_with_shared_focal(self) -> None:
        """Video estimates must use SIMPLE_PINHOLE so fx == fy can never drift apart."""
        calib = estimate_video_calibration(width=1920, height=1080)
        assert calib.model == "SIMPLE_PINHOLE"
        assert len(calib.params) == 3
        assert calib.params[1] == 960.0
        assert calib.params[2] == 540.0
        assert calib.params[0] == pytest.approx(focal_from_fov(1920, 60.0))

    def test_explicit_focal_override(self) -> None:
        """An explicit focal length wins over the assumed FoV and is recorded."""
        calib = estimate_video_calibration(width=1920, height=1080, focal_px=1430.0)
        assert calib.params[0] == 1430.0
        assert "user_supplied_focal" in calib.source

    def test_explicit_focal_must_be_positive(self) -> None:
        """A non-positive focal length is rejected instead of silently accepted."""
        with pytest.raises(ConfigError):
            estimate_video_calibration(width=1920, height=1080, focal_px=0.0)

    def test_video_source_matches_shared_estimate(self, fake_video_file) -> None:
        """Regression: inspect and sfm must not disagree about the same video file.

        The two paths previously used different magic multipliers (0.8x and 0.9x
        image width), so `uav3d inspect` and `uav3d sfm` reported different
        intrinsics for one file.
        """
        from uav3d.sources.video import VideoFileSource

        source = VideoFileSource(video_path=fake_video_file)
        expected = estimate_video_calibration(width=source.metadata().width, height=source.metadata().height)
        assert source.calibration().model == expected.model
        assert source.calibration().params == pytest.approx(expected.params)


class TestCalibrationRescaling:
    """Verify that a real calibration is rescaled, not discarded, on size mismatch."""

    def test_opencv_rescale_scales_focal_and_principal_point(self) -> None:
        """Halving resolution halves fx, fy, cx and cy; distortion stays untouched."""
        original = CameraCalibration(
            model="OPENCV",
            width=1920,
            height=1080,
            params=[1060.0, 1060.0, 960.0, 540.0, -0.05, 0.01, 0.0, 0.0],
            source="calibration_data.npz",
        )
        scaled = scale_calibration(original, 960, 540)
        assert scaled.width == 960
        assert scaled.height == 540
        assert scaled.model == "OPENCV"
        assert scaled.params[:4] == pytest.approx([530.0, 530.0, 480.0, 270.0])
        assert scaled.params[4:] == [-0.05, 0.01, 0.0, 0.0]

    def test_simple_pinhole_rescale(self) -> None:
        """SIMPLE_PINHOLE keeps a single focal length while cx/cy scale per axis."""
        original = estimate_video_calibration(width=1920, height=1080)
        scaled = scale_calibration(original, 640, 360)
        assert scaled.model == "SIMPLE_PINHOLE"
        assert scaled.params[0] == pytest.approx(original.params[0] / 3.0)
        assert scaled.params[1] == pytest.approx(320.0)
        assert scaled.params[2] == pytest.approx(180.0)

    def test_identical_size_is_a_noop(self) -> None:
        """No rescaling happens when the resolution already matches."""
        original = estimate_video_calibration(width=1920, height=1080)
        assert scale_calibration(original, 1920, 1080) is original

    def test_unknown_model_is_rejected(self) -> None:
        """Rescaling an unknown model fails loudly rather than producing garbage."""
        original = CameraCalibration(model="THIN_PRISM_FISHEYE_X", width=1920, height=1080, params=[1.0], source="x")
        with pytest.raises(ConfigError):
            scale_calibration(original, 960, 540)


class TestIntrinsicsSanityGate:
    """Verify the post-BA gate that rejected nothing before this change."""

    def test_healthy_intrinsics_pass(self) -> None:
        """A realistic 60 deg / 1920 px camera produces no violations."""
        assert check_intrinsics_sanity([1662.77, 960.0, 540.0], "SIMPLE_PINHOLE", 1920, 1080) == []
        assert check_intrinsics_sanity([1662.0, 1662.0, 960.0, 540.0], "PINHOLE", 1920, 1080) == []

    def test_regression_video_run_with_wild_focal_is_rejected(self) -> None:
        """Regression: fx=213 px, fy=745 px on a 1920 px frame must not pass.

        These are the exact values exported by the video path before the gate
        existed, where they were reported alongside a flattering reprojection
        error and a green 'quality gate passed' line.
        """
        violations = check_intrinsics_sanity([213.0029, 745.2103, 960.0, 540.0], "PINHOLE", 1920, 1080)
        assert violations
        joined = " ".join(violations)
        assert "FoV" in joined
        assert "isotropic" in joined

    def test_non_positive_focal_is_rejected(self) -> None:
        """Zero or NaN focal lengths are caught before they reach the output file."""
        assert check_intrinsics_sanity([0.0, 960.0, 540.0], "SIMPLE_PINHOLE", 1920, 1080)
        assert check_intrinsics_sanity([float("nan"), 960.0, 540.0], "SIMPLE_PINHOLE", 1920, 1080)

    def test_drifted_principal_point_is_rejected(self) -> None:
        """A principal point far from the image center is not credible."""
        violations = check_intrinsics_sanity([1662.0, 1662.0, 1900.0, 900.0], "PINHOLE", 1920, 1080)
        assert any("principal point" in v for v in violations)

    def test_fov_window_is_configurable(self) -> None:
        """A narrow FoV is only a violation when it falls outside the configured window."""
        params = [3600.0, 960.0, 540.0]  # ~30 deg horizontal FoV
        assert check_intrinsics_sanity(params, "SIMPLE_PINHOLE", 1920, 1080, min_fov_deg=25.0) == []
        assert check_intrinsics_sanity(params, "SIMPLE_PINHOLE", 1920, 1080, min_fov_deg=35.0)


class TestCameraParamCounts:
    """Verify the COLMAP parameter-count table used to fail fast on bad calibrations."""

    def test_known_models(self) -> None:
        """Parameter counts match the COLMAP camera models used by the pipeline."""
        assert camera_params_expected_count("SIMPLE_PINHOLE") == 3
        assert camera_params_expected_count("PINHOLE") == 4
        assert camera_params_expected_count("OPENCV") == 8
        assert camera_params_expected_count("CameraModelId.OPENCV") == 8

    def test_unknown_model_defers_to_colmap(self) -> None:
        """Unknown models return None so validation is delegated to COLMAP."""
        assert camera_params_expected_count("SOMETHING_NEW") is None
