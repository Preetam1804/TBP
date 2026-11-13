"""Render the analytic street scene into a video file.

Exists so the *video* ingestion path (``--video``) can be exercised end to end
without a real recording: inspect → keyframes → sfm → dense, with exact
per-pixel ground truth available for scoring.

Validation only; the production pipeline never imports this module.

Usage:
    python scripts/render_street_video.py --frames 60 --out data/benchmark/flyover.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_street import build_street, flight, render  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--focal", type=float, default=265.0)
    parser.add_argument("--step", type=float, default=1.36, help="Metres advanced per frame")
    parser.add_argument("--camera-height", type=float, default=12.0)
    parser.add_argument("--pitch", type=float, default=-0.12)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--sensor-noise", type=float, default=2.0, help="Gaussian noise, grey levels")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/benchmark/flyover.mp4"),
        help="Destination video (mp4v codec)",
    )
    args = parser.parse_args()

    boxes = build_street()
    poses = flight(args.frames, args.step, args.camera_height, args.pitch)
    rng = np.random.default_rng(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer for {args.out}")

    depth_stack = []
    for index, (rotation, translation) in enumerate(poses):
        gray, depth, hit = render(
            rotation, translation, args.focal, args.focal, args.width, args.height, boxes
        )
        noise = rng.normal(scale=args.sensor_noise, size=gray.shape)
        frame = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        depth_stack.append(depth.astype(np.float32))
        if index % 10 == 0:
            print(f"  rendered {index + 1}/{args.frames}")

    writer.release()
    np.save(args.out.with_suffix(".depth.npy"), np.stack(depth_stack))
    print(f"wrote {args.out} ({args.frames} frames, {args.width}x{args.height} @ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
