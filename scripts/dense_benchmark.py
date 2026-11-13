"""End-to-end benchmark of the CPU dense stage against exact synthetic ground truth.

Builds a street-canyon scene, writes it out as a complete SP-1 run directory
(``poses.json``, ``keyframes.json``, ``sfm/images``, ``sparse_points.npz``),
runs :func:`uav3d.dense.run_dense`, and scores the result against the analytic
surface.

Why synthetic: the Zurich sequence ships no dense ground truth, so "did it get
better?" cannot be answered on it directly. Here the correct answer is known to
the millimetre, and the camera geometry (focal length, frame spacing, standoff,
forward-looking corridor flight) matches the real clip.

Usage::

    python scripts/dense_benchmark.py --frames 40 --width 480 --height 12
    python scripts/dense_benchmark.py --frames 40 --width 320 --height 30 --pitch -0.06
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from synthetic_street import build_street, flight, render, sample_surface  # noqa: E402

from uav3d.dense.config import DenseConfig  # noqa: E402
from uav3d.dense.evaluate import cloud_metrics  # noqa: E402
from uav3d.dense.pipeline import run_dense  # noqa: E402


def rotation_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a rotation matrix to a (qw, qx, qy, qz) unit quaternion."""
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz])
    return tuple((q / np.linalg.norm(q)).tolist())


def small_rotation(rng: np.random.Generator, degrees: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(degrees) * rng.normal()
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float64
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def build_run(
    out_dir: Path,
    frames: int,
    width: int,
    height: int,
    step: float,
    camera_height: float,
    pitch: float,
    focal: float,
    noise: float,
    pose_noise_m: float,
    pose_noise_deg: float,
    seed: int,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Render the scene and write a complete SP-1 run directory."""
    rng = np.random.default_rng(seed)
    boxes = build_street()
    poses = flight(frames, step, camera_height, pitch)

    images_dir = out_dir / "sfm" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    pose_records = []
    for i, (R, t) in enumerate(poses):
        image, _, _ = render(R, t, focal, focal, width, height, boxes)
        # Sensor noise plus a per-frame exposure drift, as in a real camera.
        gain = 1.0 + rng.normal(scale=0.03)
        noisy = np.clip(image.astype(np.float32) * gain + rng.normal(scale=noise, size=image.shape), 0, 255)
        frame_id = f"syn_{i:06d}"
        cv2.imwrite(str(images_dir / f"{frame_id}.jpg"), noisy.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Perturb the pose that we hand to the pipeline, if requested.
        R_used = small_rotation(rng, pose_noise_deg) @ R
        t_used = t + rng.normal(scale=pose_noise_m, size=3) if pose_noise_m > 0 else t
        qw, qx, qy, qz = rotation_to_quaternion(R_used)
        pose_records.append(
            dict(
                frame_id=frame_id,
                timestamp_us=1_459_425_600_000_000 + i * 33_333,
                qw=qw, qx=qx, qy=qy, qz=qz,
                tx=float(t_used[0]), ty=float(t_used[1]), tz=float(t_used[2]),
                camera_center=[float(c) for c in (-R_used.T @ t_used)],
                registered=True,
            )
        )

    poses_doc = {
        "schema_version": "1.0.0",
        "dataset_label": "synthetic benchmark (analytic ground truth)",
        "coordinate_frame": "cam_from_world",
        "camera": {
            "camera_id": 1,
            "model": "PINHOLE",
            "width": width,
            "height": height,
            "params": [focal, focal, width / 2.0, height / 2.0],
            "scale_factor": 1.0,
        },
        "metrics": {
            "num_images_total": frames,
            "num_images_registered": frames,
            "registration_rate": 1.0,
            "mean_reprojection_error_px": 0.0,
        },
        "poses": pose_records,
        "unregistered_frames": [],
    }
    (out_dir / "poses.json").write_text(json.dumps(poses_doc, indent=2), encoding="utf-8")

    keyframes_doc = {
        "schema_version": "1.0.0",
        "dataset_label": "synthetic benchmark (analytic ground truth)",
        "keyframes": [
            dict(
                frame_id=rec["frame_id"],
                timestamp_us=rec["timestamp_us"],
                file_path=f"sfm/images/{rec['frame_id']}.jpg",
                selected=True,
            )
            for rec in pose_records
        ],
    }
    (out_dir / "keyframes.json").write_text(json.dumps(keyframes_doc, indent=2), encoding="utf-8")

    gt_points = sample_surface(boxes, 300_000, rng)
    sparse = gt_points[rng.choice(len(gt_points), size=4000, replace=False)]
    np.savez_compressed(out_dir / "sparse_points.npz", points=sparse.astype(np.float32))
    return gt_points, poses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--focal", type=float, default=None, help="Defaults to width * 1060 / 1920")
    parser.add_argument("--step", type=float, default=1.36, help="Camera translation per frame (m)")
    parser.add_argument("--camera-height", type=float, default=12.0)
    parser.add_argument("--pitch", type=float, default=-0.12)
    parser.add_argument("--sensor-noise", type=float, default=1.5, help="Gaussian noise, 8-bit levels")
    parser.add_argument("--pose-noise-m", type=float, default=0.0)
    parser.add_argument("--pose-noise-deg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workdir", type=Path, default=REPO / "data" / "benchmark")
    parser.add_argument("--keep", action="store_true", help="Keep the generated run directory")
    parser.add_argument("--tag", type=str, default="run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("uav3d").setLevel(logging.INFO)
    focal = args.focal if args.focal else args.width * 1060.0 / 1920.0

    run_dir = args.workdir / args.tag
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    t0 = time.time()
    gt_points, _ = build_run(
        run_dir,
        args.frames,
        args.width,
        args.height,
        args.step,
        args.camera_height,
        args.pitch,
        focal,
        args.sensor_noise,
        args.pose_noise_m,
        args.pose_noise_deg,
        args.seed,
    )
    print(f"rendered {args.frames} frames in {time.time() - t0:.1f}s -> {run_dir}")

    config = DenseConfig(
        max_width=args.width,
        source_window=10,
        num_sources=5,
        cost_threshold=0.50,
        max_relative_sigma=0.02,
        plane_step_px=0.6,
        min_planes=24,
        max_planes=96,
        consistency_tol=0.03,
        consistency_views=2,
        mesh=True,
    )
    result = run_dense(run_dir, config=config)

    from uav3d.dense.cameras import build_views, load_run_inputs

    poses_doc, keyframes_doc = load_run_inputs(run_dir)
    views = build_views(poses_doc, keyframes_doc, run_dir, args.width)
    metrics = cloud_metrics(
        _load_cloud(result.point_cloud_path),
        gt_points,
        views=views,
        thresholds=(0.25, 0.5, 1.0, 2.0),
    )
    print("\n=== dense benchmark ===")
    for key in (
        "accuracy_median", "accuracy_p90",
        "precision_at_0.25m", "precision_at_0.5m", "precision_at_1.0m", "precision_at_2.0m",
        "completeness_at_0.5m", "completeness_at_1.0m", "completeness_at_2.0m",
        "fscore_at_0.5m", "fscore_at_1.0m", "observable_fraction", "ground_truth_spacing",
    ):
        print(f"  {key:24s} {metrics[key]:.4f}")
    print(f"  points {result.num_points}, faces {result.num_faces}, {result.total_seconds:.1f}s")
    (run_dir / "benchmark_metrics.json").write_text(
        json.dumps({"metrics": metrics, "config": config.model_dump()}, indent=2), encoding="utf-8"
    )

    if not args.keep:
        pass  # generated run directory is kept by default; it is gitignored under data/
    return 0


def _load_cloud(path: Path) -> np.ndarray:
    from uav3d.dense.evaluate import load_ply_points

    return load_ply_points(path)


if __name__ == "__main__":
    raise SystemExit(main())
