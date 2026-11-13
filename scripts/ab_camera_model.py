"""A/B the camera model on the synthetic video: shared focal vs. free fx/fy.

Forward flight has a focal-length/depth ambiguity, so a *shared* focal trades
off against the scene scale without harming shape. Free fx/fy is different: a
non-physical aspect ratio warps the back-projection, and no single scalar can
undo that. This script measures both so the claim can be checked, not assumed.

Runs SfM + dense twice with the same inputs and scores both against the
analytic depth. Slow (a few minutes), validation only.

Usage:
    python scripts/ab_camera_model.py --video data/benchmark/flyover.mp4 \
        --truth data/benchmark/flyover --work data/benchmark/ab
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_street import flight  # noqa: E402

from uav3d.config import load_config  # noqa: E402
from uav3d.models.frames import CameraCalibration  # noqa: E402
from uav3d.sfm import SfmRunner, export_poses  # noqa: E402


def score_depth(run_dir: Path, truth_path: Path, args: argparse.Namespace) -> dict:
    depth_dir = run_dir / "dense" / "depth"
    if not depth_dir.exists():
        return {"error": "dense stage produced no depth maps"}
    truth_stack = np.load(truth_path)
    poses = json.loads((run_dir / "poses.json").read_text())
    order = sorted(int(p["frame_id"].split("_")[-1]) for p in poses["poses"] if p["registered"])
    ratios = []
    for index in order:
        npy = depth_dir / f"{index:05d}.npy"
        if not npy.exists():
            continue
        est = np.load(npy).astype(np.float32)
        truth = truth_stack[index]
        valid = (est > 0) & np.isfinite(est) & (truth > 0) & np.isfinite(truth)
        if valid.sum() < 500:
            continue
        a, b = est[valid].astype(np.float64), truth[valid].astype(np.float64)
        keep = b < 400.0 * np.median(b)
        if keep.sum() >= 500:
            ratios.append(float(np.median(b[keep] / a[keep])))
    if not ratios:
        return {"error": "no overlapping depth pixels"}
    scale = float(np.median(ratios))

    errors, lateral = [], []
    for index in order:
        npy = depth_dir / f"{index:05d}.npy"
        if not npy.exists():
            continue
        est = np.load(npy).astype(np.float32) * scale
        truth = truth_stack[index]
        valid = (est > 0) & np.isfinite(est) & (truth > 0) & np.isfinite(truth)
        if valid.sum() < 500:
            continue
        a, b = est[valid].astype(np.float64), truth[valid].astype(np.float64)
        keep = b < 400.0 * np.median(b)
        errors.append(a[keep] - b[keep])

        # Lateral check: back-project both at their own depth and compare the
        # horizontal footprint of the reconstructed scene. An anisotropic focal
        # stretches x by fx/fy while leaving depth almost untouched, so depth
        # alone cannot detect it -- this is the term that can.
        rr, tt = flight(args.frames, args.step, args.camera_height, args.pitch)[index]
        h, w = truth.shape
        ys, xs = np.nonzero(valid)
        cam = poses["camera"]
        params = cam["params"]
        if cam["model"].upper() == "SIMPLE_PINHOLE":
            fx = fy = float(params[0])
            cx, cy = float(params[1]), float(params[2])
        else:
            fx, fy = float(params[0]), float(params[1])
            cx, cy = float(params[2]), float(params[3])
        x_cam = (xs - cx) / fx * est[valid]
        lateral.append((np.percentile(np.abs(x_cam), 90), fx, fy))

    if not errors:
        return {"error": "no scored pixels"}
    absolute = np.abs(np.concatenate(errors))
    return {
        "scale": scale,
        "median_m": float(np.median(absolute)),
        "p90_m": float(np.percentile(absolute, 90)),
        "within_0.5m": float(100.0 * np.mean(absolute <= 0.5)),
        "within_1m": float(100.0 * np.mean(absolute <= 1.0)),
        "fx": lateral[0][1] if lateral else None,
        "fy": lateral[0][2] if lateral else None,
    }


def run_variant(name: str, model: str, refine: bool, args: argparse.Namespace) -> dict:
    run_dir = args.work / name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    images = run_dir / "sfm" / "images"
    images.mkdir(parents=True)

    import cv2

    cap = cv2.VideoCapture(str(args.video))
    index = 0
    names = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        name_i = f"video_{index:06d}"
        cv2.imwrite(str(images / f"{name_i}.jpg"), frame)
        names.append(f"{name_i}.jpg")
        index += 1
    cap.release()

    keyframes = {
        "schema_version": "1.0.0",
        "dataset_label": "synthetic video A/B",
        "keyframes": [
            {
                "frame_id": Path(n).stem,
                "timestamp_us": i * 100_000,
                "file_path": f"sfm/images/{n}",
                "score": 1.0,
                "selected": True,
                "selection_reason": "ab_test",
            }
            for i, n in enumerate(names)
        ],
    }
    (run_dir / "keyframes.json").write_text(json.dumps(keyframes, indent=2))

    cfg = load_config(None)
    cfg.sfm.camera_model = model
    cfg.sfm.refine_intrinsics = refine
    cfg.sfm.max_image_size = args.width
    cfg.sfm.seq_overlap = 10
    cfg.sfm.min_num_matches = 15
    cfg.sfm.init_min_inliers = 15
    cfg.sfm.init_min_tri_angle = 2.0
    cfg.sfm.abs_pose_min_inliers = 10

    if model == "SIMPLE_PINHOLE":
        params = [float(args.width) * 0.8, args.width / 2.0, args.height / 2.0]
    else:
        params = [float(args.width) * 0.8, float(args.width) * 0.8, args.width / 2.0, args.height / 2.0]

    calibration = CameraCalibration(
        model=model, width=args.width, height=args.height, params=params, source="ab_test"
    )
    result = SfmRunner(config=cfg.sfm).run(images_dir=images, output_dir=run_dir, calibration=calibration)
    export_poses(
        reconstruction=result.reconstruction,
        all_image_names=names,
        camera_calibration=calibration,
        output_path=run_dir / "poses.json",
        dataset_label="synthetic video A/B",
        scale_factor=1.0,
        timestamps_map={Path(n).stem: i * 100_000 for i, n in enumerate(names)},
    )
    from uav3d.dense.geometry import write_sparse_points

    points = result.reconstruction.points3D
    if points:
        xyz = np.array([p.xyz for p in points.values()], dtype=np.float64)
        write_sparse_points(xyz, run_dir)

    subprocess.run(
        [sys.executable, "-m", "uav3d.cli", "dense", "--run-dir", str(run_dir),
         "--max-width", str(args.width), "--ref-stride", "2", "--no-mesh"],
        check=True, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        cwd=str(Path(__file__).resolve().parent.parent),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    metrics = score_depth(run_dir, Path(str(args.truth) + ".depth.npy"), args)
    metrics["model"] = model
    metrics["refine"] = refine
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("data/benchmark/flyover.mp4"))
    parser.add_argument("--truth", type=Path, default=Path("data/benchmark/flyover"))
    parser.add_argument("--work", type=Path, default=Path("data/benchmark/ab"))
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--step", type=float, default=1.36)
    parser.add_argument("--camera-height", type=float, default=12.0)
    parser.add_argument("--pitch", type=float, default=-0.12)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--skip", default="", help="Comma-separated variants to skip")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    variants = [
        ("simple_locked", "SIMPLE_PINHOLE", False),
        ("simple_refined", "SIMPLE_PINHOLE", True),
        ("pinhole_refined", "PINHOLE", True),
    ]
    rows = []
    for name, model, refine in variants:
        if name in skip:
            continue
        print(f"\n=== {name}: {model} refine={refine} ===", flush=True)
        rows.append(run_variant(name, model, refine, args))

    print(f"\n{'variant':<20}{'fx':>9}{'fy':>9}{'fx/fy':>8}{'scale':>9}"
          f"{'med m':>9}{'p90 m':>9}{'<=0.5m':>9}{'<=1m':>8}")
    for r in rows:
        if "error" in r:
            print(f"{r.get('model','?'):<20} {r['error']}")
            continue
        ratio = (r["fx"] / r["fy"]) if r.get("fx") and r.get("fy") else 1.0
        print(f"{r['model']:<20}{r['fx']:>9.1f}{(r['fy'] or 0):>9.1f}{ratio:>8.3f}"
              f"{r['scale']:>9.3f}{r['median_m']:>9.3f}{r['p90_m']:>9.3f}"
              f"{r['within_0.5m']:>8.1f}%{r['within_1m']:>7.1f}%")
    (args.work / "ab_results.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
