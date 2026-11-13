"""Unit tests for the dense-stage projective geometry and error model."""

import numpy as np
import pytest

from uav3d.dense.cameras import Intrinsics
from uav3d.dense.geometry import (
    depth_noise_sigma,
    depth_planes,
    depth_sigma_array,
    parse_depth_range,
    plane_count_for_range,
    plane_homography,
    umeyama_similarity,
)
from uav3d.exceptions import DataError

K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480).matrix


def _pose(rotation: np.ndarray, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build cam_from_world (R, t) from a rotation and a camera centre."""
    return rotation, -rotation @ centre


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _project(K: np.ndarray, R: np.ndarray, t: np.ndarray, points: np.ndarray) -> np.ndarray:
    cam = points @ R.T + t
    uv = cam[:, :2] / cam[:, 2:3]
    return np.stack([K[0, 0] * uv[:, 0] + K[0, 2], K[1, 1] * uv[:, 1] + K[1, 2]], axis=1)


@pytest.mark.parametrize("depth", [4.0, 17.5, 90.0])
def test_plane_homography_maps_plane_points(depth: float) -> None:
    """H(d) must map reference pixels to source pixels for points on the plane at depth d."""
    R_ref, t_ref = _pose(_rot_y(0.2), np.array([1.0, -2.0, 0.5]))
    R_src, t_src = _pose(_rot_y(-0.35), np.array([3.5, -1.0, 1.5]))

    uv_ref = np.array([[50.0, 60.0], [320.0, 240.0], [600.0, 400.0], [120.0, 450.0]])
    rays = np.stack(
        [
            (uv_ref[:, 0] - 320.0) / 500.0,
            (uv_ref[:, 1] - 240.0) / 500.0,
            np.ones(len(uv_ref)),
        ],
        axis=1,
    )
    points_cam = rays * depth
    points_world = (points_cam - t_ref) @ R_ref

    expected = _project(K, R_src, t_src, points_world)

    H = plane_homography(K, R_ref, t_ref, R_src, t_src, depth)
    homog = np.concatenate([uv_ref, np.ones((len(uv_ref), 1))], axis=1) @ H.T
    actual = homog[:, :2] / homog[:, 2:3]

    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_plane_homography_moves_far_points_less() -> None:
    """A larger plane depth must produce a smaller pixel displacement."""
    R_ref, t_ref = _pose(np.eye(3), np.zeros(3))
    R_src, t_src = _pose(np.eye(3), np.array([1.0, 0.0, 0.0]))
    uv = np.array([[600.0, 240.0]])
    homog = np.concatenate([uv, np.ones((1, 1))], axis=1)

    near = plane_homography(K, R_ref, t_ref, R_src, t_src, 5.0)
    far = plane_homography(K, R_ref, t_ref, R_src, t_src, 50.0)
    near_uv = (homog @ near.T)[:, :2] / (homog @ near.T)[:, 2:3]
    far_uv = (homog @ far.T)[:, :2] / (homog @ far.T)[:, 2:3]

    assert abs(near_uv[0, 0] - uv[0, 0]) > abs(far_uv[0, 0] - uv[0, 0])


def test_depth_planes_are_uniform_in_inverse_depth() -> None:
    planes = depth_planes(5.0, 100.0, 40)
    inverse = 1.0 / planes
    steps = np.diff(inverse)
    assert np.allclose(steps, steps[0], rtol=1e-9)
    assert planes.min() == pytest.approx(5.0)
    assert planes.max() == pytest.approx(100.0)


def test_depth_planes_rejects_bad_range() -> None:
    with pytest.raises(DataError):
        depth_planes(10.0, 10.0, 10)
    with pytest.raises(DataError):
        depth_planes(0.0, 10.0, 10)


def test_plane_count_scales_with_disparity_span() -> None:
    """More baseline means a wider disparity span, which needs more planes."""
    narrow = plane_count_for_range(10.0, 100.0, 1000.0, 1.0, 0.5, 8, 400)
    wide = plane_count_for_range(10.0, 100.0, 1000.0, 8.0, 0.5, 8, 400)
    assert wide > narrow
    assert plane_count_for_range(10.0, 100.0, 1000.0, 1.0, 0.5, 8, 400) >= 8
    assert plane_count_for_range(1e6, 1e7, 1e6, 1e6, 0.5, 8, 400) == 400


def test_depth_noise_sigma_matches_stereo_propagation() -> None:
    """sigma_z = z^2 / (f b) * sigma_d."""
    sigma = depth_noise_sigma(depth=50.0, focal_px=1000.0, baseline=5.0, disparity_sigma_px=0.3)
    assert sigma == pytest.approx(50.0**2 / (1000.0 * 5.0) * 0.3)
    # Four times the standoff means sixteen times the depth error.
    assert depth_noise_sigma(100.0, 1000.0, 5.0, 0.3) == pytest.approx(4.0 * sigma)


def test_depth_sigma_array_matches_scalar() -> None:
    depths = np.array([5.0, 20.0, 80.0])
    array = depth_sigma_array(depths, 900.0, 2.5, 0.4)
    for depth, value in zip(depths, array):
        assert value == pytest.approx(depth_noise_sigma(float(depth), 900.0, 2.5, 0.4))


def test_umeyama_recovers_known_similarity() -> None:
    rng = np.random.default_rng(3)
    src = rng.normal(size=(40, 3)) * 4.0
    scale_true = 2.5
    R_true = _rot_y(0.4)
    t_true = np.array([1.0, -2.0, 3.0])
    dst = scale_true * src @ R_true.T + t_true

    scale, R, t = umeyama_similarity(src, dst)
    assert scale == pytest.approx(scale_true, rel=1e-6)
    np.testing.assert_allclose(R, R_true, atol=1e-6)
    np.testing.assert_allclose(t, t_true, atol=1e-6)


def test_parse_depth_range() -> None:
    assert parse_depth_range("5:200") == (5.0, 200.0)
    with pytest.raises(DataError):
        parse_depth_range("200:5")
    with pytest.raises(DataError):
        parse_depth_range("nonsense")
    with pytest.raises(DataError):
        parse_depth_range("0:10")
