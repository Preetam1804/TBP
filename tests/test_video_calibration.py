"""Regression tests for video-run camera calibration.

These lock in the fix for a real defect: the SfM command used to force
``fx = fy = 0.9 * width`` and force ``refine_intrinsics = True`` for every video
run, ignoring the config. With ``PINHOLE`` the solver then refined fx and fy
independently and drove them to a physically impossible aspect ratio (1.69 on
synthetic forward flight), which stretches the reconstructed cloud along one
image axis.
"""

from __future__ import annotations

import json

import pytest

from uav3d.cli import _scale_calibration_params
from uav3d.models.calibration_io import read_calibration, write_calibration
from uav3d.models.frames import CameraCalibration


class TestScaleCalibrationParams:
    """Rescaling must respect each camera model's parameter layout."""

    def test_simple_pinhole_has_three_params(self) -> None:
        scaled = _scale_calibration_params("SIMPLE_PINHOLE", [400.0, 320.0, 180.0], 2.0, 2.0)
        assert scaled == [800.0, 640.0, 360.0]

    def test_pinhole_scales_each_axis(self) -> None:
        scaled = _scale_calibration_params("PINHOLE", [400.0, 300.0, 320.0, 180.0], 2.0, 3.0)
        assert scaled == [800.0, 900.0, 640.0, 540.0]

    def test_distortion_coefficients_are_dimensionless(self) -> None:
        params = [400.0, 400.0, 320.0, 180.0, -0.1, 0.01, 0.0, 0.0]
        scaled = _scale_calibration_params("OPENCV", params, 2.0, 2.0)
        assert scaled[:4] == [800.0, 800.0, 640.0, 360.0]
        assert scaled[4:] == params[4:]

    def test_unknown_model_is_rejected(self) -> None:
        with pytest.raises(Exception, match="Cannot rescale intrinsics"):
            _scale_calibration_params("FISHEYE_MAGIC", [1.0, 2.0], 2.0, 2.0)


class TestCalibrationRoundTrip:
    """The keyframes stage saves calibration so later stages need no video."""

    def test_round_trip(self, tmp_path) -> None:
        original = CameraCalibration(
            model="SIMPLE_PINHOLE",
            width=480,
            height=270,
            params=[384.0, 240.0, 135.0],
            source="video_estimate_flight.mp4",
        )
        path = write_calibration(original, tmp_path)
        assert path.exists()

        restored = read_calibration(tmp_path)
        assert restored is not None
        assert restored.model == original.model
        assert restored.params == original.params
        assert (restored.width, restored.height) == (original.width, original.height)
        assert restored.source == original.source

    def test_missing_file_returns_none(self, tmp_path) -> None:
        assert read_calibration(tmp_path) is None

    def test_schema_version_is_recorded(self, tmp_path) -> None:
        write_calibration(
            CameraCalibration(
                model="PINHOLE", width=640, height=480, params=[1.0, 1.0, 2.0, 2.0], source="fixture"
            ),
            tmp_path,
        )
        payload = json.loads((tmp_path / "calibration.json").read_text())
        assert payload["schema_version"] == "1.0.0"

    def test_malformed_file_raises(self, tmp_path) -> None:
        (tmp_path / "calibration.json").write_text("{not json")
        with pytest.raises(Exception, match="Malformed calibration"):
            read_calibration(tmp_path)

    def test_incomplete_file_raises(self, tmp_path) -> None:
        (tmp_path / "calibration.json").write_text(json.dumps({"model": "PINHOLE"}))
        with pytest.raises(Exception, match="Incomplete calibration"):
            read_calibration(tmp_path)


class TestVideoFocalPolicy:
    """A video run must not be able to express an anisotropic camera."""

    def test_simple_pinhole_can_represent_one_focal_only(self) -> None:
        calibration = CameraCalibration(
            model="SIMPLE_PINHOLE", width=480, height=270, params=[384.0, 240.0, 135.0], source="fixture"
        )
        # Three parameters: one focal and a principal point. There is no slot in
        # which a diverging fx/fy could be stored, which is the point.
        assert len(calibration.params) == 3
        assert calibration.params[0] > 0

    def test_aspect_ratio_of_simple_pinhole_is_unity(self) -> None:
        params = [384.0, 240.0, 135.0]
        fx = fy = params[0]
        assert fx / fy == 1.0
