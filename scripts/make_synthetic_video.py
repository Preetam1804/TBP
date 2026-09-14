"""Generate synthetic MP4 video and ground-truth frame manifest for video pipeline tests."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uav3d.logging_config import setup_logging

logger = setup_logging()


def generate_synthetic_frame(index: int, width: int = 1920, height: int = 1080) -> np.ndarray:
    """Generate a high-detail synthetic aerial frame with corners and text for SfM."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Background gradient
    for y in range(height):
        img[y, :, 0] = int(40 + 60 * (y / height))
        img[y, :, 1] = int(50 + 70 * (y / height))
        img[y, :, 2] = int(30 + 40 * (y / height))

    # Draw moving grid and feature blocks simulating drone motion
    dx = int((index * 7) % width)
    dy = int((index * 5) % height)

    for x in range(0, width, 80):
        cv2.line(img, ((x + dx) % width, 0), ((x + dx) % width, height), (70, 90, 70), 1)
    for y in range(0, height, 80):
        cv2.line(img, (0, (y + dy) % height), (width, (y + dy) % height), (70, 90, 70), 1)

    # Feature points (rectangles, circles)
    np.random.seed(42 + index)
    for _ in range(30):
        rx = np.random.randint(50, width - 50)
        ry = np.random.randint(50, height - 50)
        color = (int(np.random.randint(100, 255)), int(np.random.randint(100, 255)), int(np.random.randint(100, 255)))
        cv2.circle(img, (rx, ry), 8, color, -1)

    # Frame text watermark
    cv2.putText(
        img,
        f"SYNTHETIC UAV FRAME {index:06d}",
        (60, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return img


def make_synthetic_video(
    output_video_path: Path | str = "data/interim/synthetic.mp4",
    output_manifest_path: Path | str = "data/interim/manifest.json",
    num_frames: int = 100,
    fps: float = 5.0,
    zurich_dir: Path | str | None = "data/raw/zurich_mav",
) -> None:
    """Create a 100-frame MP4 at 5 fps plus manifest.json mapping frames."""
    video_out = Path(output_video_path).resolve()
    manifest_out = Path(output_manifest_path).resolve()

    video_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    # Check if real Zurich images exist to use
    zurich_images: list[Path] = []
    if zurich_dir:
        z_path = Path(zurich_dir)
        if z_path.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                zurich_images.extend(sorted(z_path.rglob(ext)))

    width = 1920
    height = 1080
    if zurich_images:
        sample = cv2.imread(str(zurich_images[0]))
        if sample is not None:
            height, width = sample.shape[:2]

    # Initialize OpenCV VideoWriter using MP4V codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out), fourcc, fps, (width, height))
    if not writer.isOpened():
        # Fallback to avc1 or XVID if mp4v fails
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(video_out), fourcc, fps, (width, height))

    manifest: list[dict[str, int | str]] = []
    base_timestamp_us = 1459425600000000  # Base realistic timestamp in microseconds

    logger.info(f"Generating synthetic video with {num_frames} frames at {fps} fps to {video_out}")

    for idx in range(num_frames):
        if idx < len(zurich_images):
            img = cv2.imread(str(zurich_images[idx]))
            source_image_id = f"zurich_{idx:06d}"
        else:
            img = generate_synthetic_frame(idx, width=width, height=height)
            source_image_id = f"synthetic_{idx:06d}"

        writer.write(img)

        # Video presentation timestamp in microseconds
        video_pts_us = int((idx / fps) * 1_000_000)
        source_timestamp_us = base_timestamp_us + video_pts_us

        manifest.append(
            {
                "video_frame_index": idx,
                "source_image_id": source_image_id,
                "source_timestamp_us": source_timestamp_us,
                "video_pts_us": video_pts_us,
            }
        )

    writer.release()

    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Wrote {len(manifest)} manifest entries to {manifest_out}")


if __name__ == "__main__":
    make_synthetic_video()
