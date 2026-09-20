"""Tests for SfM export, conventions, locked intrinsics, and unregistered frame handling."""

from pathlib import Path

import numpy as np
import pytest

from uav3d.exceptions import Uav3dError
from uav3d.models.frames import CameraCalibration
from uav3d.models.poses import PosesFile
from uav3d.sfm.colmap_compat import (
    compute_camera_center,
    get_peak_rss_mb,
    quaternion_to_rotation_matrix,
)
from uav3d.sfm.colmap_export import export_poses
from uav3d.sfm.runner import SfmRunner


class TestPoseConventions:
    """Verify coordinate conventions: cam_from_world, wxyz quaternion, C = -R.T @ t."""

    def test_quaternion_identity(self) -> None:
        q_identity = [1.0, 0.0, 0.0, 0.0]
        R = quaternion_to_rotation_matrix(q_identity)
        assert np.allclose(R, np.eye(3), atol=1e-6)

    def test_camera_center_formula_spot_checks(self) -> None:
        """Spot-check C = -R.T @ t across three distinct test poses."""
        # Case 1: Pure translation along Z, identity rotation
        q1 = [1.0, 0.0, 0.0, 0.0]
        t1 = [0.0, 0.0, 10.0]
        c1 = compute_camera_center(q1, t1)
        assert np.allclose(c1, [0.0, 0.0, -10.0], atol=1e-6)

        # Case 2: 90 deg rotation around Y axis: qw=cos(45)=sqrt(2)/2, qy=sin(45)=sqrt(2)/2
        val = np.sqrt(2.0) / 2.0
        q2 = [val, 0.0, val, 0.0]
        t2 = [5.0, 2.0, -3.0]
        R2 = quaternion_to_rotation_matrix(q2)
        expected_c2 = (-R2.T @ np.array(t2).reshape(3, 1)).flatten()
        c2 = compute_camera_center(q2, t2)
        assert np.allclose(c2, expected_c2, atol=1e-6)

        # Case 3: Downward-looking UAV drone pose: 180 deg around X axis
        # qw=0, qx=1, qy=0, qz=0 => R = diag(1, -1, -1)
        q3 = [0.0, 1.0, 0.0, 0.0]
        t3 = [15.0, -20.0, 35.0]
        R3 = quaternion_to_rotation_matrix(q3)
        expected_c3 = (-R3.T @ np.array(t3).reshape(3, 1)).flatten()
        c3 = compute_camera_center(q3, t3)
        assert np.allclose(c3, expected_c3, atol=1e-6)

    def test_quaternion_wxyz_ordering(self) -> None:
        """Confirm quaternion component 0 is scalar (w)."""
        # 90-degree rotation about Z axis
        # qw = cos(pi/4) = 0.7071, qz = sin(pi/4) = 0.7071
        s = np.sin(np.pi / 4.0)
        c = np.cos(np.pi / 4.0)
        q = [c, 0.0, 0.0, s]
        R = quaternion_to_rotation_matrix(q)
        # Vector along X [1, 0, 0] rotated 90 deg about Z becomes [0, 1, 0]
        v_rot = R @ np.array([1.0, 0.0, 0.0])
        assert np.allclose(v_rot, [0.0, 1.0, 0.0], atol=1e-5)


class TestSfmExport:
    """Test export_poses logic using minimal mock-free pycolmap data structures."""

    def test_export_poses_with_unregistered_frames(self, tmp_path: Path) -> None:
        class DummyRotation:
            def __init__(self, R):
                self._R = R
                # Pycolmap/Eigen xyzw convention for identity
                self.quat = np.array([0.0, 0.0, 0.0, 1.0])

            def matrix(self):
                return self._R

        class DummyRigid3d:
            def __init__(self, quat, trans):
                self.rotation = DummyRotation(np.eye(3))
                self.rotation.quat = np.array(quat, dtype=np.float64)
                self.translation = np.array(trans, dtype=np.float64)

        class DummyImage:
            def __init__(self, name: str, quat: list[float], trans: list[float], has_pose: bool = True, num_pts: int = 15):
                self.name = name
                self.cam_from_world = DummyRigid3d(quat, trans)
                self.has_pose = has_pose
                self._num_pts = num_pts

            def num_points3D(self) -> int:
                return self._num_pts

        class DummyCamera:
            def __init__(self, params: list[float]):
                self.params = params
                self.width = 1920
                self.height = 1080
                self.model = "OPENCV"

        class DummyReconstruction:
            def __init__(self, images: dict[int, DummyImage], cameras: dict[int, DummyCamera]):
                self.images = images
                self.cameras = cameras

            def compute_mean_reprojection_error(self) -> float:
                return 0.42

            def compute_mean_track_length(self) -> float:
                return 3.5

            def num_points3D(self) -> int:
                return 150

        # Construct 2 registered images (quat in [x, y, z, w] xyzw identity)
        img1 = DummyImage("zurich_000000.jpg", [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 5.0])
        img2 = DummyImage("zurich_000002.jpg", [0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 5.0])
        cam = DummyCamera([1060.0, 1060.0, 960.0, 540.0, -0.05, 0.01, 0.0, 0.0])

        rec = DummyReconstruction(images={1: img1, 2: img2}, cameras={1: cam})

        # Define 3 input images: zurich_000000, zurich_000002, and zurich_000004 (unregistered)
        all_images = ["zurich_000000.jpg", "zurich_000002.jpg", "zurich_000004.jpg"]

        calib = CameraCalibration(
            model="OPENCV",
            width=1920,
            height=1080,
            params=[1060.0, 1060.0, 960.0, 540.0, -0.05, 0.01, 0.0, 0.0],
            source="calibration_data.npz",
        )

        out_json = tmp_path / "poses.json"
        poses_file = export_poses(
            reconstruction=rec,
            all_image_names=all_images,
            camera_calibration=calib,
            output_path=out_json,
            timestamps_map={"zurich_000000": 1000000, "zurich_000002": 1200000},
        )

        # Verify output file exists and strictly parses
        assert out_json.exists()
        reloaded = PosesFile.model_validate_json(out_json.read_text(encoding="utf-8"))
        assert reloaded == poses_file

        # Verify registered poses
        assert len(poses_file.poses) == 2
        p0 = poses_file.poses[0]
        assert p0.frame_id == "zurich_000000"
        assert p0.timestamp_us == 1000000
        assert p0.registered is True
        assert np.allclose(p0.camera_center_world, [0.0, 0.0, -5.0], atol=1e-5)

        # Verify unregistered frame is explicitly captured
        assert len(poses_file.unregistered_frames) == 1
        unreg = poses_file.unregistered_frames[0]
        assert unreg.frame_id == "zurich_000004"
        assert unreg.reason == "not_registered_in_largest_model"

        # Verify metrics block
        assert poses_file.metrics is not None
        assert poses_file.metrics.num_images_total == 3
        assert poses_file.metrics.num_images_registered == 2
        assert np.isclose(poses_file.metrics.registration_rate, 2.0 / 3.0, atol=1e-3)

        # Verify camera block
        assert poses_file.camera is not None
        assert poses_file.camera.model == "OPENCV"
        assert len(poses_file.camera.params) == 8


class TestSfmRunnerLogic:
    """Test SfM runner assertions and safety gates."""

    def test_zero_images_raises_error(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_images"
        empty_dir.mkdir()
        runner = SfmRunner()
        calib = CameraCalibration(
            model="OPENCV",
            width=1920,
            height=1080,
            params=[1060.0, 1060.0, 960.0, 540.0, 0.0, 0.0, 0.0, 0.0],
            source="test",
        )
        with pytest.raises(Uav3dError) as exc:
            runner.run(images_dir=empty_dir, output_dir=tmp_path, calibration=calib)
        assert "no image files found" in str(exc.value).lower()

    def test_peak_rss_returns_positive_float(self) -> None:
        rss = get_peak_rss_mb()
        assert isinstance(rss, float)
        assert rss >= 0.0
