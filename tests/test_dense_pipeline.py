"""End-to-end test of the CPU dense pipeline on an analytically rendered plane.

The scene is a slanted textured wall seen by a laterally translating camera, so
ground-truth depth is known in closed form and the recovered cloud can be scored
against it directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from uav3d.dense.cameras import Intrinsics
from uav3d.dense.config import DenseConfig
from uav3d.dense.evaluate import load_ply_points
from uav3d.dense.pipeline import run_dense

WIDTH, HEIGHT = 192, 144
FOCAL = 150.0
WALL_Z = 20.0
STEP_M = 1.0
FRAMES = 10
TEXTURE_METRES = (48.0, 36.0)
_TEXTURE_SHAPE = (720, 960)


def _texture_image() -> np.ndarray:
    """Band-limited random texture, so no view resamples fine detail differently."""
    rng = np.random.default_rng(5)
    coarse = rng.normal(size=(90, 120))
    texture = cv2.resize(coarse, (_TEXTURE_SHAPE[1], _TEXTURE_SHAPE[0]), interpolation=cv2.INTER_CUBIC)
    texture = cv2.GaussianBlur(texture, (0, 0), 2.0)
    texture -= texture.min()
    texture /= max(texture.max(), 1e-9)
    return (texture * 255.0).astype(np.uint8)


_TEXTURE = _texture_image()


def _render(centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Render a fronto-parallel textured wall from a camera looking along +z."""
    u, v = np.meshgrid(
        np.arange(WIDTH, dtype=np.float32) + 0.5, np.arange(HEIGHT, dtype=np.float32) + 0.5
    )
    world_x = centre[0] + (u - WIDTH / 2.0) / FOCAL * WALL_Z
    world_y = centre[1] + (v - HEIGHT / 2.0) / FOCAL * WALL_Z

    # Texture spans TEXTURE_METRES metres, anchored at the origin.
    px = (world_x / TEXTURE_METRES[0] + 0.5) * (_TEXTURE_SHAPE[1] - 1)
    py = (world_y / TEXTURE_METRES[1] + 0.5) * (_TEXTURE_SHAPE[0] - 1)
    image = cv2.remap(
        _TEXTURE, px.astype(np.float32), py.astype(np.float32), interpolation=cv2.INTER_LINEAR
    )
    depth = np.full((HEIGHT, WIDTH), WALL_Z, dtype=np.float64)
    return image, depth


def _write_run(run_dir: Path) -> None:
    images = run_dir / "sfm" / "images"
    images.mkdir(parents=True, exist_ok=True)

    poses, keyframes = [], []
    for index in range(FRAMES):
        centre = np.array([index * STEP_M, 0.0, 0.0])
        image, _ = _render(centre)
        name = f"frame_{index:04d}.png"
        path = images / name
        assert cv2.imwrite(str(path), image)
        poses.append(
            {
                "frame_id": f"frame_{index:04d}",
                "timestamp_us": 1_000_000 + index * 100_000,
                "qw": 1.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "tx": float(-centre[0]),
                "ty": 0.0,
                "tz": 0.0,
                "camera_center": centre.tolist(),
                "registered": True,
            }
        )
        keyframes.append(
            {
                "frame_id": f"frame_{index:04d}",
                "timestamp_us": 1_000_000 + index * 100_000,
                "file_path": f"sfm/images/{name}",
                "selected": True,
            }
        )

    (run_dir / "poses.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "dataset_label": "synthetic dense pipeline test",
                "coordinate_frame": "cam_from_world",
                "camera": {
                    "camera_id": 1,
                    "model": "PINHOLE",
                    "width": WIDTH,
                    "height": HEIGHT,
                    "params": [FOCAL, FOCAL, WIDTH / 2.0, HEIGHT / 2.0],
                    "scale_factor": 1.0,
                },
                "metrics": {
                    "num_images_total": FRAMES,
                    "num_images_registered": FRAMES,
                    "registration_rate": 1.0,
                    "mean_reprojection_error_px": 0.4,
                },
                "poses": poses,
                "unregistered_frames": [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "keyframes.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "dataset_label": "synthetic dense pipeline test",
                "keyframes": keyframes,
            }
        ),
        encoding="utf-8",
    )

    rng = np.random.default_rng(11)
    xs = rng.uniform(-3.0, 6.0, 400)
    ys = rng.uniform(-3.0, 3.0, 400)
    zs = np.full_like(xs, WALL_Z)
    np.savez(
        run_dir / "sparse_points.npz",
        points=np.stack([xs, ys, zs], axis=1),
        rgb=rng.integers(0, 255, (400, 3)),
    )


def test_dense_pipeline_recovers_plane_depth(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_test"
    _write_run(run_dir)

    config = DenseConfig(
        max_width=WIDTH,
        num_sources=4,
        plane_step_px=2.0,
        min_planes=16,
        max_planes=64,
        cost_threshold=0.5,
        min_views=2,
        mesh=True,
    )
    result = run_dense(run_dir=run_dir, config=config)

    assert result.point_cloud_path.exists()
    points = load_ply_points(result.point_cloud_path)
    assert points.shape[0] > 500, f"only {points.shape[0]} points survived"

    # Every point should sit on the analytic wall.
    residual = np.abs(points[:, 2] - WALL_Z)
    assert float(np.median(residual)) < 0.6
    assert float(np.percentile(residual, 90)) < 2.0

    assert result.report_path.exists()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["schema_version"]
    assert isinstance(report["scale"]["status"], str) and report["scale"]["status"]
    assert report["fusion"]["num_points"] == points.shape[0]
    assert 0.0 <= report["depth"]["valid_fraction"] <= 1.0

    if result.mesh_path is not None:
        assert result.mesh_path.exists()
        assert report["mesh"]["num_faces"] > 0


def test_dense_pipeline_is_deterministic(tmp_path: Path) -> None:
    """Two runs on identical inputs must produce byte-identical clouds."""
    run_dir = tmp_path / "run_test"
    _write_run(run_dir)
    config = DenseConfig(
        max_width=WIDTH,
        num_sources=3,
        plane_step_px=3.0,
        min_planes=8,
        max_planes=32,
        mesh=False,
    )
    first = run_dense(run_dir=run_dir, config=config, output_dir=tmp_path / "a")
    second = run_dense(run_dir=run_dir, config=config, output_dir=tmp_path / "b")
    assert first.point_cloud_path.read_bytes() == second.point_cloud_path.read_bytes()


def test_dense_pipeline_respects_manual_depth_range(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_test"
    _write_run(run_dir)
    config = DenseConfig(
        max_width=WIDTH,
        num_sources=3,
        plane_step_px=3.0,
        min_planes=8,
        max_planes=32,
        depth_range="5:60",
        mesh=False,
    )
    result = run_dense(run_dir=run_dir, config=config)
    assert tuple(result.report.geometry.depth_range) == pytest.approx((5.0, 60.0))
    assert result.point_cloud_path.exists()


def test_intrinsics_roundtrip() -> None:
    intrinsics = Intrinsics(fx=FOCAL, fy=FOCAL, cx=WIDTH / 2.0, cy=HEIGHT / 2.0, width=WIDTH, height=HEIGHT)
    matrix = intrinsics.matrix
    assert matrix[0, 0] == FOCAL
    assert matrix[1, 2] == HEIGHT / 2.0
    scaled = intrinsics.scaled(0.5)
    assert scaled.fx == pytest.approx(FOCAL * 0.5)
