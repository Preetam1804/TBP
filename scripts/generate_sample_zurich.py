"""Generate a realistic 3D synthetic aerial sample dataset for the Zurich Urban MAV layout.

The default output directory is deliberately *not* the path the baseline config reads for the
real dataset (``data/raw/zurich_mav``): writing synthetic frames into a real dataset directory
would silently mix 50 synthetic frames into the real image set and overwrite its calibration.
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_sample_zurich")

DEFAULT_DATASET_DIR = "data/synthetic/zurich_mav"


class DatasetPathError(RuntimeError):
    """Raised when the requested output directory would clobber existing data."""


def _ensure_writable_output_dir(base: Path, force: bool) -> None:
    """Refuse to write into a directory that already holds dataset content, unless forced."""
    if not base.exists():
        return

    existing_entries = [p for p in base.iterdir() if p.name not in {".gitkeep"}]
    if not existing_entries:
        return

    looks_like_real_dataset = any(
        entry.name == "calibration_data.npz" or entry.name == "MAV Images" or entry.name == "Log Files"
        for entry in existing_entries
    )
    if force:
        logger.warning(f"--force: overwriting existing content in {base}")
        return

    detail = " (this looks like a real Zurich dataset directory)" if looks_like_real_dataset else ""
    raise DatasetPathError(
        f"Refusing to write synthetic data into non-empty directory {base}{detail}. "
        "Choose an empty --dataset-dir (e.g. data/synthetic/zurich_mav), or pass --force to overwrite."
    )


def generate_zurich_sample_dataset(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    num_frames: int = 50,
    force: bool = False,
    overwrite_calibration: bool = False,
) -> None:
    """Populate ``dataset_dir`` with ``num_frames`` frames observing a true 3D scene.

    Ensures that visual SfM (pycolmap) can extract consistent SIFT features,
    triangulate 3D points, lock known intrinsics, and estimate camera poses
    matching the GroundTruthAGL and GPS trajectory.

    Args:
        dataset_dir: Output directory (default ``data/synthetic/zurich_mav``).
        num_frames: Number of frames to render.
        force: Allow writing into a non-empty/real dataset directory.
        overwrite_calibration: Allow replacing an existing ``calibration_data.npz``.
    """
    base = Path(dataset_dir).resolve()
    _ensure_writable_output_dir(base, force)

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

    # Save calibration_data.npz — never silently replace a real dataset calibration.
    calib_path = base / "calibration_data.npz"
    if calib_path.exists() and not overwrite_calibration:
        raise DatasetPathError(
            f"{calib_path} already exists. Refusing to overwrite an existing calibration file; "
            "pass --overwrite-calibration if you really mean to regenerate the synthetic one."
        )
    np.savez(str(calib_path), camera_matrix=K, distortion_coefficients=dist)
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
        for pt, pat in zip(pts3D, patterns, strict=True):
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


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", nargs="?", default=DEFAULT_DATASET_DIR, help=f"Output directory (default: {DEFAULT_DATASET_DIR})")
    parser.add_argument("--num-frames", type=int, default=50, help="Number of frames to render (default: 50)")
    parser.add_argument("--force", action="store_true", help="Allow writing into a non-empty directory")
    parser.add_argument(
        "--overwrite-calibration",
        action="store_true",
        help="Allow replacing an existing calibration_data.npz",
    )
    args = parser.parse_args()

    try:
        generate_zurich_sample_dataset(
            dataset_dir=args.dataset_dir,
            num_frames=args.num_frames,
            force=args.force,
            overwrite_calibration=args.overwrite_calibration,
        )
    except DatasetPathError as exc:
        logger.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
