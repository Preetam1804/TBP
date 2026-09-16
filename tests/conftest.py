"""Pytest fixtures for Zurich dataset and synthetic video testing."""

from pathlib import Path

import cv2
import numpy as np
import pytest

# Camera used by the reconstruction-capable mini scene fixture.
MINI_W, MINI_H = 640, 480
MINI_FX = MINI_FY = 500.0
MINI_CX, MINI_CY = MINI_W / 2.0, MINI_H / 2.0


def _render_mini_scene_frames(num_frames: int, seed: int = 7) -> list[np.ndarray]:
    """Render frames of a static 3D point cloud observed by a translating camera.

    Unlike a moving 2D pattern, this gives real parallax so incremental SfM can
    actually triangulate the scene, which is what makes an end-to-end test of the
    sfm/sync/validate stages possible without shipping image data in the repo.
    """
    rng = np.random.default_rng(seed)
    num_landmarks = 420
    # Landmarks distributed on the far side of the OpenCV camera axis (+Z) along
    # a corridor that the camera flies through from x=0 to x≈5.
    xs = rng.uniform(-3.0, 8.0, num_landmarks)
    ys = rng.uniform(-3.2, 3.2, num_landmarks)
    zs = rng.uniform(4.5, 8.5, num_landmarks)
    radii = [int(r) for r in rng.integers(3, 7, num_landmarks)]
    palette = rng.integers(60, 240, (num_landmarks, 3)).tolist()

    frames: list[np.ndarray] = []
    for i in range(num_frames):
        img = np.full((MINI_H, MINI_W, 3), 30, dtype=np.uint8)
        cam_center = np.array([0.6 * i, 0.0, 0.0], dtype=np.float64)
        rvec = np.zeros(3, dtype=np.float64)
        tvec = -cam_center

        pts = np.column_stack([xs, ys, zs])
        projected, _ = cv2.projectPoints(pts, rvec, tvec, np.array([[MINI_FX, 0, MINI_CX], [0, MINI_FY, MINI_CY], [0, 0, 1.0]]), None)
        uv = projected.reshape(-1, 2)

        for j in range(num_landmarks):
            u, v = round(uv[j, 0]), round(uv[j, 1])
            if 8 <= u < MINI_W - 8 and 8 <= v < MINI_H - 8:
                bgr = (palette[j][0], palette[j][1], palette[j][2])
                cv2.circle(img, (u, v), radii[j], bgr, -1, cv2.LINE_AA)
                cv2.circle(img, (u, v), max(1, radii[j] // 2), (255 - bgr[0], 255 - bgr[1], 255 - bgr[2]), -1)
        frames.append(img)
    return frames


@pytest.fixture
def mini_recon_dataset(tmp_path: Path) -> Path:
    """Build a small Zurich-layout dataset that is genuinely reconstructable.

    Contains 10 frames of a static 3D scene, matching calibration, onboard GPS
    rows for every frame and evaluation ground truth for every frame, so the
    complete inspect -> keyframes -> sfm -> sync -> validate chain can be
    exercised end to end inside a temporary directory.
    """
    dataset_dir = tmp_path / "mini_zurich"
    images_dir = dataset_dir / "MAV Images"
    logs_dir = dataset_dir / "Log Files"
    images_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    num_frames = 10
    frames = _render_mini_scene_frames(num_frames)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(images_dir / f"frame_{i:05d}.jpg"), frame)

    k_matrix = np.array([[MINI_FX, 0.0, MINI_CX], [0.0, MINI_FY, MINI_CY], [0.0, 0.0, 1.0]], dtype=np.float64)
    np.savez(
        str(dataset_dir / "calibration_data.npz"),
        camera_matrix=k_matrix,
        distortion_coefficients=np.zeros(4, dtype=np.float64),
    )

    base_ts = 1459425600000000
    with open(logs_dir / "OnbordGPS.csv", "w", encoding="utf-8") as f:
        f.write("# timestamp, image_id, lat, lon, alt\n")
        for i in range(num_frames):
            f.write(
                f"{base_ts + i * 200000}, {i}, {47.3769 + i * 0.000015:.6f}, "
                f"{8.5417 + i * 0.000018:.6f}, {450.0 + i * 0.1:.2f}\n"
            )

    with open(logs_dir / "GroundTruthAGL.csv", "w", encoding="utf-8") as f:
        f.write("# image_id, x, y, z\n")
        for i in range(num_frames):
            f.write(f"{i}, {0.6 * i:.4f}, 0.0, 0.0\n")

    return dataset_dir


@pytest.fixture
def fake_zurich_dataset(tmp_path: Path) -> Path:
    """Create a miniature, fully-valid Zurich dataset structure."""
    dataset_dir = tmp_path / "zurich_mav"
    images_dir = dataset_dir / "MAV Images"
    logs_dir = dataset_dir / "Log Files"
    images_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    # 1. Create 5 sample frame images (1080p with normal mid-tone values and contrast edges)
    img_data = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    for x in range(0, 1920, 60):
        img_data[:, x : x + 30] = 180
    for i in range(5):
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
