"""Pytest fixtures for Zurich dataset and synthetic video testing."""

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def fake_zurich_dataset(tmp_path: Path) -> Path:
    """Create a miniature, fully-valid Zurich dataset structure."""
    dataset_dir = tmp_path / "zurich_mav"
    images_dir = dataset_dir / "MAV Images"
    logs_dir = dataset_dir / "Log Files"
    images_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    # 1. Create 5 sample frame images (1080p black frames with noise)
    img_data = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for i in range(5):
        # Frame names matching Zurich format (e.g. frame_00000.jpg, frame_00001.jpg)
        img_path = images_dir / f"frame_{i:05d}.jpg"
        cv2.imwrite(str(img_path), img_data)

    # 2. Create OnbordGPS.csv (image 4 intentionally omitted to test unmapped telemetry)
    gps_csv = logs_dir / "OnbordGPS.csv"
    gps_csv.write_text(
        "# timestamp, image_id, lat, lon, alt\n"
        "1459425600000000, 0, 47.376900, 8.541700, 450.00\n"
        "1459425600100000, 1, 47.376910, 8.541710, 450.25\n"
        "1459425600200000, 2, 47.376920, 8.541720, 450.50\n"
        "1459425600300000, 3, 47.376930, 8.541730, 450.75\n",
        encoding="utf-8",
    )

    # 3. Create GroundTruthAGL.csv (all 5 frames present)
    gt_csv = logs_dir / "GroundTruthAGL.csv"
    gt_csv.write_text(
        "# image_id, x, y, z\n"
        "0, 10.0, 20.0, 30.0\n"
        "1, 10.2, 20.2, 30.2\n"
        "2, 10.4, 20.4, 30.4\n"
        "3, 10.6, 20.6, 30.6\n"
        "4, 10.8, 20.8, 30.8\n",
        encoding="utf-8",
    )

    # 4. Create calibration_data.npz
    calib_npz = dataset_dir / "calibration_data.npz"
    k_matrix = np.array([[1060.0, 0.0, 960.0], [0.0, 1060.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs = np.array([-0.05, 0.01, 0.0, 0.0], dtype=np.float64)
    np.savez(str(calib_npz), camera_matrix=k_matrix, distortion_coefficients=dist_coeffs)

    return dataset_dir


@pytest.fixture
def fake_video_file(tmp_path: Path) -> Path:
    """Create a miniature 10-frame synthetic MP4 file."""
    video_path = tmp_path / "mini_test.mp4"
    width, height = 320, 240
    fps = 10.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    for i in range(10):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (i * 20, 100, 150)
        writer.write(frame)
    writer.release()
    return video_path
