"""Generate realistic 3D synthetic aerial sample dataset for Zurich Urban MAV."""

import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_sample_zurich")


def generate_zurich_sample_dataset(
    dataset_dir: str | Path = "data/raw/zurich_mav",
    num_frames: int = 50,
) -> None:
    """Populate data/raw/zurich_mav with 50 frames observing a true 3D scene.

    Ensures that visual SfM (pycolmap) can extract consistent SIFT features,
    triangulate 3D points, lock known intrinsics, and estimate camera poses
    matching the GroundTruthAGL and GPS trajectory.
    """
    base = Path(dataset_dir).resolve()
    img_dir = base / "MAV Images"
    log_dir = base / "Log Files"
    img_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Camera calibration (GoPro Hero4 1080p from Zurich dataset)
    W, H = 1920, 1080
    fx, fy = 1060.0, 1060.0
    cx, cy = 960.0, 540.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([-0.05, 0.01, 0.0, 0.0], dtype=np.float64)

    # Save calibration_data.npz
    np.savez(str(base / "calibration_data.npz"), camera_matrix=K, distortion_coefficients=dist)
    logger.info(f"Saved calibration_data.npz to {base}")

    # Generate persistent 3D world scene along flight path
    # Flight path: from (0,0) to (75, 60)
    np.random.seed(42)
    num_landmarks = 450

    dir_vec = np.array([1.5, 1.2]) / np.linalg.norm([1.5, 1.2])
    perp_vec = np.array([-dir_vec[1], dir_vec[0]])

    t_vals = np.linspace(-8.0, 88.0, num_landmarks)
    across_vals = np.random.uniform(-18.0, 18.0, size=num_landmarks)
    height_vals = np.random.uniform(-2.0, 16.0, size=num_landmarks)

    X = t_vals * dir_vec[0] + across_vals * perp_vec[0]
    Y = t_vals * dir_vec[1] + across_vals * perp_vec[1]
    Z = height_vals

    pts3D = np.column_stack([X, Y, Z])

    # Distinct 28x28 textured pattern for each landmark
    patterns = []
    for _ in range(num_landmarks):
        pat = np.random.randint(30, 235, (28, 28, 3), dtype=np.uint8)
        # Distinct high-contrast cross in center for invariant SIFT keypoints
        pat[13:15, :, :] = 255
        pat[:, 13:15, :] = 0
        patterns.append(pat)

    # Camera orientation (looking nadir / slightly pitched forward)
    # Camera X = World X, Camera Y = -World Y, Camera Z = -World Z
    R_cw = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)

    logger.info(f"Rendering {num_frames} synthetic aerial frames...")
    for i in range(num_frames):
        # Camera center in world from GroundTruthAGL formula
        pos_w = np.array([i * 1.5, i * 1.2, 30.0 + i * 0.05], dtype=np.float64)
        t_cw = -R_cw @ pos_w

        # Base terrain background
        img = np.zeros((H, W, 3), dtype=np.uint8)
        for y in range(H):
            img[y, :, 0] = int(55 + 25 * (y / H))
            img[y, :, 1] = int(70 + 35 * (y / H))
            img[y, :, 2] = int(45 + 20 * (y / H))

        # Project 3D landmarks
        for pt, pat in zip(pts3D, patterns):
            p_c = R_cw @ pt + t_cw
            if p_c[2] > 2.0:
                u = int(fx * p_c[0] / p_c[2] + cx)
                v = int(fy * p_c[1] / p_c[2] + cy)
                sz = int(28 * 30.0 / p_c[2])
                sz = max(16, min(40, sz))
                half = sz // 2
                if half <= u < W - half and half <= v < H - half:
                    scaled_pat = cv2.resize(pat, (sz, sz), interpolation=cv2.INTER_NEAREST)
                    img[v - half : v - half + sz, u - half : u - half + sz] = scaled_pat

        # Subtle frame identifier
        cv2.putText(
            img,
            f"ZURICH MAV FRAME {i:05d}",
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )

        frame_path = img_dir / f"frame_{i:05d}.jpg"
        cv2.imwrite(str(frame_path), img)

    logger.info(f"Rendered {num_frames} frames to {img_dir}")

    # Generate OnbordGPS.csv
    gps_path = log_dir / "OnbordGPS.csv"
    base_ts = 1459425600000000
    with open(gps_path, "w", encoding="utf-8") as f:
        f.write("# timestamp, image_id, lat, lon, alt\n")
        for i in range(num_frames):
            if i == 49:
                # Leave frame 49 without GPS row to test unmapped telemetry in real data
                continue
            ts = base_ts + i * 200000
            lat = 47.376900 + i * 0.000015
            lon = 8.541700 + i * 0.000018
            alt = 450.0 + i * 0.1
            f.write(f"{ts}, {i}, {lat:.6f}, {lon:.6f}, {alt:.2f}\n")
    logger.info(f"Generated {gps_path}")

    # Generate GroundTruthAGL.csv
    gt_path = log_dir / "GroundTruthAGL.csv"
    with open(gt_path, "w", encoding="utf-8") as f:
        f.write("# image_id, x, y, z\n")
        for i in range(num_frames):
            f.write(f"{i}, {i*1.5:.2f}, {i*1.2:.2f}, {30.0 + i*0.05:.2f}\n")
    logger.info(f"Generated {gt_path}")


if __name__ == "__main__":
    generate_zurich_sample_dataset()
