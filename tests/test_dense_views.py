"""Unit tests for dense-view selection and boundary handling."""

import numpy as np
import pytest

from uav3d.dense.cameras import CameraView, Intrinsics
from uav3d.dense.views import (
    PREFERRED_BASELINE_RATIO,
    median_baseline,
    median_depth_of,
    predicted_overlap,
    rank_sources,
    view_angle_deg,
)

INTRINSICS = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)


def _view(index: int, centre: np.ndarray, yaw: float = 0.0) -> CameraView:
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return CameraView(
        index=index,
        frame_id=f"frame_{index:03d}",
        image_path=__import__("pathlib").Path(f"/tmp/{index}.jpg"),
        R=R,
        t=-R @ centre,
        intrinsics=INTRINSICS,
    )


def test_camera_center_roundtrip() -> None:
    centre = np.array([3.0, -4.0, 12.0])
    view = _view(0, centre, yaw=0.3)
    np.testing.assert_allclose(view.center, centre, atol=1e-12)


def test_predicted_overlap_self_is_one() -> None:
    centre = np.array([0.0, 0.0, 10.0])
    view = _view(0, centre)
    assert predicted_overlap(view, view, 5.0, 40.0) == pytest.approx(1.0)


def test_predicted_overlap_backwards_is_zero() -> None:
    forward = _view(0, np.array([0.0, 0.0, 0.0]))
    backward = _view(1, np.array([0.05, 0.0, 0.0]), yaw=np.pi)
    assert predicted_overlap(forward, backward, 5.0, 40.0) == pytest.approx(0.0)


def test_view_angle_is_zero_for_parallel_axes() -> None:
    a = _view(0, np.zeros(3))
    b = _view(1, np.array([1.0, 0.0, 0.0]))
    assert view_angle_deg(a, b) == pytest.approx(0.0, abs=1e-9)


def test_rank_sources_prefers_informative_baselines() -> None:
    """With a long camera line, the best source should sit near the preferred ratio."""
    depth = 40.0
    step = 0.5
    views = [_view(i, np.array([i * step, 0.0, 0.0])) for i in range(41)]
    central = 20
    chosen = rank_sources(
        views=views,
        ref_index=central,
        near=20.0,
        far=80.0,
        reference_depth=depth,
        max_angle_deg=45.0,
        min_baseline_ratio=0.005,
        max_baseline_ratio=0.30,
        min_overlap=0.2,
        window=20,
        num_sources=3,
    )
    assert len(chosen) == 3
    best = chosen[0]
    assert best.baseline_ratio == pytest.approx(PREFERRED_BASELINE_RATIO, abs=0.06)
    # Scores must be non-increasing.
    scores = [c.score for c in chosen]
    assert scores == sorted(scores, reverse=True)


def test_rank_sources_rejects_unusable_geometry() -> None:
    """A rapid slew pushes the angle over the limit, leaving nothing to fuse."""
    views = [_view(i, np.array([i * 1.0, 0.0, 0.0]), yaw=i * 0.5) for i in range(10)]
    chosen = rank_sources(
        views=views,
        ref_index=0,
        near=10.0,
        far=50.0,
        reference_depth=30.0,
        max_angle_deg=5.0,
        min_baseline_ratio=0.0,
        max_baseline_ratio=10.0,
        min_overlap=0.0,
        window=9,
        num_sources=4,
    )
    assert all(c.angle_deg <= 5.0 for c in chosen)


def test_median_helpers() -> None:
    views = [_view(i, np.array([i * 2.0, 0.0, 0.0])) for i in range(5)]
    assert median_depth_of({0: 10.0, 1: 20.0, 2: 60.0}) == pytest.approx(20.0)
    assert median_depth_of({}) == 0.0
    selections = {
        0: [type("S", (), {"index": 1})()],
        2: [type("S", (), {"index": 3})()],
    }
    # Baselines are 2.0 and 2.0 -> median 2.0
    assert median_baseline(views, selections) == pytest.approx(2.0)
