"""Generate synthetic MP4 video and ground-truth frame manifest for video pipeline tests.

The rendered scene is a persistent 3D landmark cloud observed by a moving, yawing camera, so
that visual SfM can triangulate it (a per-frame random overlay would produce no consistent
geometry, and the video -> 3D path could never be validated).

World frame: X/Y ground plane, Z up. Camera convention matches the rest of the pipeline:
``x_cam = R_cw @ x_world + t_cw`` with ``R_cw = diag(1, -1, -1)`` (nadir-looking GoPro).
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uav3d.logging_config import setup_logging  # noqa: E402
from uav3d.sources.video import DEFAULT_FOCAL_RATIO  # noqa: E402

logger = setup_logging()

# Camera mount: nadir with a slight forward tilt (world Z up).
CAM_FROM_WORLD_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
CAMERA_ALTITUDE_M = 25.0
FLIGHT_STEP_M = 0.9


def camera_center_at(index: int) -> np.ndarray:
    """Camera centre (world metres) for a flight frame index."""
    return np.array([index * FLIGHT_STEP_M, index * FLIGHT_STEP_M * 0.25, CAMERA_ALTITUDE_M])


def synthetic_gps_at(index: int) -> tuple[float, float, float]:
    """Synthetic WGS84 GPS sample along the same flight line (demo telemetry, not real GPS)."""
    lat = 47.376900 + index * FLIGHT_STEP_M * 9.0e-6
    lon = 8.541700 + index * FLIGHT_STEP_M * 1.2e-5
    alt = 430.0 + index * 0.05
    return lat, lon, alt


def _rotation_z(angle_rad: float) -> np.ndarray:
    """Return the 3x3 rotation matrix about the world Z axis."""
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_scene(num_landmarks: int = 900, seed: int = 7) -> tuple[np.ndarray, list[np.ndarray]]:
    """Create a persistent landmark cloud plus a distinct texture patch per landmark.

    Returns:
        tuple of (landmarks (N, 3) float array, list of BGR texture patches).
    """
    rng = np.random.default_rng(seed)

    travel = np.linspace(-12.0, 70.0, num_landmarks)
    across = rng.uniform(-16.0, 16.0, size=num_landmarks)
    height = rng.uniform(-3.0, 14.0, size=num_landmarks)
    points = np.column_stack([travel, across, height])

    patterns: list[np.ndarray] = []
    for _ in range(num_landmarks):
        patch = rng.integers(25, 235, size=(24, 24, 3), dtype=np.uint8)
        patch[11:13, :, :] = 250  # high-contrast cross: invariant SIFT keypoints
        patch[:, 11:13, :] = 5
        patterns.append(patch)
    return points, patterns


def render_scene_frame(
    index: int,
    width: int,
    height: int,
    points: np.ndarray,
    patterns: list[np.ndarray],
    yaw_deg_per_frame: float = 0.35,
) -> np.ndarray:
    """Render one frame of the persistent scene for the camera pose at ``index``."""
    fx = fy = float(width) * DEFAULT_FOCAL_RATIO
    cx, cy = float(width) / 2.0, float(height) / 2.0

    # Camera trajectory: straight flight with a gentle yaw sway for rotational diversity.
    camera_center = camera_center_at(index)
    yaw_rad = np.radians(index * yaw_deg_per_frame)

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        frame[row, :, 0] = int(45 + 35 * (row / height))
        frame[row, :, 1] = int(60 + 45 * (row / height))
        frame[row, :, 2] = int(35 + 25 * (row / height))

    # Yawing the camera is equivalent to counter-rotating the world points about the camera centre.
    relative = (points - camera_center) @ _rotation_z(-yaw_rad).T
    points_cam = relative @ CAM_FROM_WORLD_ROTATION.T

    in_front = points_cam[:, 2] > 2.0
    depths = points_cam[in_front, 2]
    u = (fx * points_cam[in_front, 0] / depths + cx).astype(int)
    v = (fy * points_cam[in_front, 1] / depths + cy).astype(int)
    sizes = np.clip((24.0 * CAMERA_ALTITUDE_M / depths).astype(int), 10, 40)

    for (px, py), size, pattern in zip(zip(u, v, strict=True), sizes, iter(patterns), strict=False):
        half = size // 2
        if not (half <= px < width - half and half <= py < height - half):
            continue
        patch = cv2.resize(pattern, (size, size), interpolation=cv2.INTER_NEAREST)
        frame[py - half : py - half + size, px - half : px - half + size] = patch

    cv2.putText(
        frame,
        f"SYNTHETIC UAV FRAME {index:06d}",
        (int(width * 0.03), int(height * 0.06)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.5, width / 1600.0),
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    return frame


def make_synthetic_video(
    output_video_path: Path | str = "data/interim/synthetic.mp4",
    output_manifest_path: Path | str = "data/interim/manifest.json",
    num_frames: int = 100,
    fps: float = 5.0,
    zurich_dir: Path | str | None = "data/raw/zurich_mav",
    width: int = 1920,
    height: int = 1080,
) -> None:
    """Write a synthetic MP4 plus a manifest.json mapping every frame to its pose/telemetry."""
    video_out = Path(output_video_path).resolve()
    manifest_out = Path(output_manifest_path).resolve()

    video_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    # Prefer real Zurich frames when the dataset is present (replayed-dataset runs).
    zurich_images: list[Path] = []
    if zurich_dir:
        z_path = Path(zurich_dir)
        if z_path.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                zurich_images.extend(sorted(z_path.rglob(ext)))

    points, patterns = build_scene()
    if zurich_images:
        sample = cv2.imread(str(zurich_images[0]))
        if sample is not None:
            height, width = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(video_out), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a video writer for {video_out}")

    manifest: list[dict[str, float | int | str]] = []
    base_timestamp_us = 1459425600000000  # realistic epoch base in microseconds
    ground_truth_rows: list[str] = []

    logger.info(f"Generating synthetic video with {num_frames} frames at {fps} fps to {video_out}")

    try:
        for idx in range(num_frames):
            if idx < len(zurich_images):
                img = cv2.imread(str(zurich_images[idx]))
                source_image_id = f"zurich_{idx:06d}"
            else:
                img = render_scene_frame(idx, width, height, points, patterns)
                source_image_id = f"synthetic_{idx:06d}"

            writer.write(img)

            video_pts_us = int((idx / fps) * 1_000_000)
            source_timestamp_us = base_timestamp_us + video_pts_us

            entry: dict[str, float | int | str] = {
                "video_frame_index": idx,
                "source_image_id": source_image_id,
                "source_timestamp_us": source_timestamp_us,
                "video_pts_us": video_pts_us,
            }
            if not zurich_images:
                # Synthetic scene: attach a demo GPS track and an evaluation-only ground truth.
                lat, lon, alt = synthetic_gps_at(idx)
                entry.update({"lat": lat, "lon": lon, "alt": alt})
                center = camera_center_at(idx)
                entry["camera_center_world"] = [float(center[0]), float(center[1]), float(center[2])]
                ground_truth_rows.append(f"{idx}, {center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}")
            manifest.append(entry)
    finally:
        writer.release()

    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Wrote {len(manifest)} manifest entries to {manifest_out}")

    if ground_truth_rows:
        gt_path = video_out.parent / "ground_truth_agl.csv"
        gt_path.write_text(
            "# image_id, x, y, z  (synthetic scene coordinates, EVALUATION ONLY)\n"
            + "\n".join(ground_truth_rows)
            + "\n",
            encoding="utf-8",
        )
        logger.info(f"Wrote {len(ground_truth_rows)} evaluation-only ground truth rows to {gt_path}")


if __name__ == "__main__":
    make_synthetic_video()
