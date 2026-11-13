"""Score the dense depth maps against analytic ground truth, gauge-free.

Each depth map is in that reference frame's own camera coordinates, so it can
be compared to the analytic Z-depth of the same frame after a single scalar.
The scale is resolved the standard monocular way: SfM is scale-free, so one
global scalar maps units to metres. Two independent scale estimates are
reported -- from the trajectory (Umeyama on camera centres) and from depth
itself (median ratio over the reference frames) -- so a bad trajectory fit
cannot silently flatter the depth numbers.

This deliberately avoids aligning the point cloud in 3-D. A straight flight
leaves rotation about the flight axis unconstrained, so a similarity fit on
camera centres alone can roll the whole cloud by tens of degrees and report a
huge error on a perfectly good reconstruction.

Validation only; the production pipeline never imports this module.

Usage:
    python scripts/eval_video_depth.py --run-dir data/outputs/run_x \
        --truth data/benchmark/flyover
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_street import flight  # noqa: E402


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity mapping ``src -> dst`` (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1.0
    rot = u @ np.diag(sign) @ vt
    scale = float((d * sign).sum() / ((s_c**2).sum() / len(src)))
    return scale, rot, mu_d - scale * rot @ mu_s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True, help="Video path minus suffix")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--step", type=float, default=1.36)
    parser.add_argument("--camera-height", type=float, default=12.0)
    parser.add_argument("--pitch", type=float, default=-0.12)
    parser.add_argument("--true-focal", type=float, default=265.0)
    parser.add_argument("--max-relative-depth", type=float, default=400.0,
                        help="Ignore truth pixels deeper than this fraction of median depth")
    args = parser.parse_args()

    poses_path = args.run_dir / "poses.json"
    depth_dir = args.run_dir / "dense" / "depth"
    truth_path = Path(str(args.truth) + ".depth.npy")
    for path in (poses_path, depth_dir, truth_path):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    poses = json.loads(poses_path.read_text())
    depth_stack = np.load(truth_path)
    truth_rt = flight(args.frames, args.step, args.camera_height, args.pitch)

    order = sorted(int(p["frame_id"].split("_")[-1]) for p in poses["poses"] if p["registered"])
    centres = {int(p["frame_id"].split("_")[-1]): np.array(p["camera_center"], float)
               for p in poses["poses"] if p["registered"]}
    est = np.array([centres[i] for i in order])
    true_c = np.array([-truth_rt[i][0].T @ truth_rt[i][1] for i in order])

    traj_scale, _, _ = umeyama(est, true_c)
    flight_len = args.step * (args.frames - 1)

    # A straight flight makes the similarity alignment degenerate: the camera
    # centres form a rank-1 point set, so rotation about the flight axis is
    # unconstrained and a 3-D RMS would measure that free parameter, not the
    # reconstruction. Test the shape gauge-free instead: is the recovered path a
    # straight line, and are the frame-to-frame steps uniform (a constant-speed
    # flight must produce constant spacing)?
    centred = est - est.mean(0)
    _, sv, vt = np.linalg.svd(centred, full_matrices=False)
    direction = vt[0]
    off_axis = centred - np.outer(centred @ direction, direction)
    line_rms = float(np.sqrt((off_axis**2).sum(1).mean()))
    steps = np.linalg.norm(np.diff(est, axis=0), axis=1) * traj_scale

    # Per-frame depth comparison in each frame's own camera frame.
    ratios, errors, truth_values, per_frame = [], [], [], []
    for index in order:
        npy = depth_dir / f"{index:05d}.npy"
        if not npy.exists():
            continue
        est_depth = np.load(npy).astype(np.float32)
        truth_depth = depth_stack[index]
        valid = (est_depth > 0) & np.isfinite(est_depth) & (truth_depth > 0) & np.isfinite(truth_depth)
        if valid.sum() < 500:
            continue
        a = est_depth[valid].astype(np.float64)
        b = truth_depth[valid].astype(np.float64)
        # Reject the analytic "sky" (infinite depth) before it dominates.
        keep = b < args.max_relative_depth * np.median(b)
        a, b = a[keep], b[keep]
        if len(a) < 500:
            continue
        frame_scale = float(np.median(b / a))
        ratios.append(frame_scale)
        per_frame.append((index, len(a), frame_scale, float(np.median(np.abs(a * frame_scale - b)))))

    if not ratios:
        print("no overlapping valid depth pixels; cannot score")
        return 1

    scale = float(np.median(ratios))
    for index, _, _, err in per_frame:
        npy = depth_dir / f"{index:05d}.npy"
        est_depth = np.load(npy).astype(np.float32) * scale
        truth_depth = depth_stack[index]
        valid = (est_depth > 0) & np.isfinite(est_depth) & (truth_depth > 0) & np.isfinite(truth_depth)
        if valid.sum() < 500:
            continue
        a = est_depth[valid].astype(np.float64)
        b = truth_depth[valid].astype(np.float64)
        keep = b < args.max_relative_depth * np.median(b)
        errors.append(a[keep] - b[keep])
        truth_values.append(b[keep])

    errors = np.concatenate(errors)
    truth_all = np.concatenate(truth_values)
    absolute = np.abs(errors)
    relative = absolute / np.maximum(np.abs(truth_all), 1e-6)

    camera = poses.get("camera") or {}
    print(f"registered frames        : {len(order)}")
    print(f"scored depth pixels      : {len(errors):,} over {len(per_frame)} frames")
    print(f"trajectory scale         : {traj_scale:.4f} m/unit")
    print(f"depth scale              : {scale:.4f} m/unit")
    print(f"scale agreement          : {100.0 * abs(scale / traj_scale - 1.0):.1f}% apart")
    print(f"path straightness        : {1000.0 * line_rms:.2f} mm off the best-fit line")
    print(f"step spacing             : {steps.mean():.4f} +- {steps.std():.4f} m "
          f"(true {args.step:.4f} m, error {100.0 * abs(steps.mean() / args.step - 1.0):.2f}%)")
    if camera.get("params"):
        print(f"recovered focal          : {camera['params'][0]:.1f} px "
              f"(true {args.true_focal:.1f} px, ratio {camera['params'][0] / args.true_focal:.2f}x)")
    print()
    print(f"depth error median       : {np.median(absolute):.3f} m")
    print(f"depth error p90          : {np.percentile(absolute, 90):.3f} m")
    print(f"relative error median    : {100.0 * np.median(relative):.1f}%")
    for t in (0.25, 0.5, 1.0, 2.0):
        print(f"  within {t:>4.2f} m        : {100.0 * np.mean(absolute <= t):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
