"""Measure the 3-D shape error caused by an anisotropic camera model.

Per-view depth is measured along each camera's own optical axis, so it survives
a wrong focal length: focal and scene scale trade off and one global scalar
undoes it. What a wrong *aspect ratio* does is different -- it places each pixel
on the wrong ray, so the reconstructed points are stretched in one image axis
even though every point's depth is right.

This back-projects each frame's estimated depth with (a) the recovered
intrinsics and (b) the true intrinsics, then Procrustes-aligns the point set to
the analytic 3-D coordinates. Shape RMS is reported after similarity alignment,
so the focal/scale ambiguity cannot masquerade as shape error.

Validation only.

Usage:
    python scripts/eval_shape_anisotropy.py --work data/benchmark/ab
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_street import flight  # noqa: E402


def similarity_rms(src: np.ndarray, dst: np.ndarray) -> float:
    """RMS after optimal similarity alignment of ``src`` onto ``dst``."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(d_c.T @ s_c / len(src))
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1.0
    rot = u @ np.diag(sign) @ vt
    scale = float((d * sign).sum() / ((s_c**2).sum() / len(src)))
    aligned = scale * (rot @ s_c.T).T
    return float(np.sqrt(((aligned - d_c) ** 2).sum(1).mean()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("data/benchmark/ab"))
    parser.add_argument("--truth", type=Path, default=Path("data/benchmark/flyover"))
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--step", type=float, default=1.36)
    parser.add_argument("--camera-height", type=float, default=12.0)
    parser.add_argument("--pitch", type=float, default=-0.12)
    parser.add_argument("--true-focal", type=float, default=265.0)
    parser.add_argument("--samples", type=int, default=4000)
    args = parser.parse_args()

    truth_stack = np.load(Path(str(args.truth) + ".depth.npy"))
    truth_rt = flight(args.frames, args.step, args.camera_height, args.pitch)
    rng = np.random.default_rng(0)

    print(f"{'variant':<20}{'fx':>8}{'fy':>8}{'fx/fy':>7}{'shape RMS m':>13}")
    for run_dir in sorted(p for p in args.work.iterdir() if (p / "poses.json").exists()):
        poses = json.loads((run_dir / "poses.json").read_text())
        cam = poses["camera"]
        params = cam["params"]
        if cam["model"].upper() == "SIMPLE_PINHOLE":
            fx = fy = float(params[0])
            cx, cy = float(params[1]), float(params[2])
        else:
            fx, fy = float(params[0]), float(params[1])
            cx, cy = float(params[2]), float(params[3])

        own = []
        for record in poses["poses"]:
            if not record["registered"]:
                continue
            index = int(record["frame_id"].split("_")[-1])
            npy = run_dir / "dense" / "depth" / f"{index:05d}.npy"
            if not npy.exists():
                continue
            depth = np.load(npy).astype(np.float32)
            truth_depth = truth_stack[index]
            valid = (depth > 0) & np.isfinite(depth) & (truth_depth > 0) & np.isfinite(truth_depth)
            ys, xs = np.nonzero(valid)
            if len(xs) < args.samples:
                continue
            pick = rng.choice(len(xs), args.samples, replace=False)
            ys, xs = ys[pick], xs[pick]
            z = depth[ys, xs].astype(np.float64)
            h, w = depth.shape

            # Scale estimated depth to metres with this frame's own median ratio.
            ratio = float(np.median(truth_depth[ys, xs].astype(np.float64) / z))
            z_m = z * ratio

            rot, trans = truth_rt[index]
            rays = np.array([
                (xs - w / 2.0) / args.true_focal,
                (ys - h / 2.0) / args.true_focal,
                np.ones_like(z),
            ])
            truth_pts = (rot.T @ (rays * z_m)) + (-rot.T @ trans)[:, None]
            truth_pts = truth_pts.T

            def backproject(fx_, fy_, cx_, cy_):
                return np.stack([
                    (xs - cx_) / fx_ * z_m,
                    (ys - cy_) / fy_ * z_m,
                    z_m,
                ], axis=1)

            own.append(similarity_rms(backproject(fx, fy, cx, cy), truth_pts))

        if own:
            ratio = fx / fy
            print(f"{run_dir.name:<20}{fx:>8.1f}{fy:>8.1f}{ratio:>7.3f}{np.median(own):>11.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
