"""Render the analytic street scene into a loop/return-flight video.

Same scene, camera and true focal (265 px) as ``render_street_video.py``,
but the flight revisits ground: a forward pass along the street, a
180-degree U-turn at the far end, and a return pass over the same stretch
on the opposite side of the street. This is the loop-closure test clip:
sequential matching should drift / miss the revisit, exhaustive matching
should close the loop.

Flight geometry (all heights 12 m, same -0.12 "pitch" roll as the flyover):
  * forward leg: 45 frames, x = 0 .. 59.84 m, y = 0, heading +x.
    Poses are ``synthetic_street.flight`` poses (unchanged convention).
  * turn: 14 frames, a semicircle of radius 3.5 m centred at
    (59.84, 3.5), heading sweeping 0 -> pi through +y. The camera rig is
    rotated rigidly about the world z-axis (``R = R_fwd @ rot_z(-beta)``),
    which is what a fixed-mounted camera on a yawing drone does; the
    return images are therefore upside-down relative to the forward leg
    (SIFT is rotation-invariant, so matching is unaffected in principle).
  * return leg: 45 frames, x = 59.84 .. 0 m, y = 7, heading -x, with the
    same rigidly-rotated mount (``R = R_fwd @ rot_z(-pi)``).

Total 104 frames at 480x270. Ground-truth artefacts, same format as the
flyover plus poses: ``loop.mp4``, ``loop.depth.npy`` (per-frame GT depth),
``loop.poses.npz`` (R, t, centres per frame; ``R``/``t`` are cam_from_world).

Usage:
    python scripts/render_loop_video.py --out data/benchmark/loop.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_street import build_street, flight, render, rot_z  # noqa: E402

FORWARD_FRAMES = 45
TURN_FRAMES = 14
RETURN_FRAMES = 45


def loop_flight(
    forward_frames: int = FORWARD_FRAMES,
    turn_frames: int = TURN_FRAMES,
    return_frames: int = RETURN_FRAMES,
    step: float = 1.36,
    height: float = 12.0,
    pitch: float = -0.12,
    turn_radius: float = 3.5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward leg + U-turn + return leg, as (R, t) cam_from_world pairs."""
    forward = flight(forward_frames, step, height, pitch)
    r_fwd, _ = forward[0]
    x_end = (forward_frames - 1) * step
    centre_turn = np.array([x_end, turn_radius, height])
    poses: list[tuple[np.ndarray, np.ndarray]] = list(forward)

    for j in range(1, turn_frames + 1):
        beta = np.pi * j / (turn_frames + 1)
        centre = centre_turn + turn_radius * np.array(
            [np.sin(beta), -np.cos(beta), 0.0]
        )
        rotation = r_fwd @ rot_z(-beta)
        poses.append((rotation, -rotation @ centre))

    r_ret = r_fwd @ rot_z(-np.pi)
    for k in range(return_frames):
        centre = np.array([x_end - k * step, 2.0 * turn_radius, height])
        poses.append((r_ret.copy(), -r_ret @ centre))
    return poses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-forward", type=int, default=FORWARD_FRAMES)
    parser.add_argument("--frames-turn", type=int, default=TURN_FRAMES)
    parser.add_argument("--frames-return", type=int, default=RETURN_FRAMES)
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
        default=Path("data/benchmark/loop.mp4"),
        help="Destination video (mp4v codec)",
    )
    args = parser.parse_args()

    boxes = build_street()
    poses = loop_flight(
        args.frames_forward,
        args.frames_turn,
        args.frames_return,
        args.step,
        args.camera_height,
        args.pitch,
    )
    rng = np.random.default_rng(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer for {args.out}")

    depth_stack = []
    rotations, translations, centres = [], [], []
    for index, (rotation, translation) in enumerate(poses):
        gray, depth, hit = render(
            rotation, translation, args.focal, args.focal, args.width, args.height, boxes
        )
        noise = rng.normal(scale=args.sensor_noise, size=gray.shape)
        frame = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        depth_stack.append(depth.astype(np.float32))
        rotations.append(rotation)
        translations.append(translation)
        centres.append(-rotation.T @ translation)
        if index % 10 == 0:
            print(f"  rendered {index + 1}/{len(poses)}")

    writer.release()
    np.save(args.out.with_suffix(".depth.npy"), np.stack(depth_stack))
    np.savez(
        args.out.with_suffix(".poses.npz"),
        R=np.stack(rotations),
        t=np.stack(translations),
        centres=np.stack(centres),
    )
    print(f"wrote {args.out} ({len(poses)} frames, {args.width}x{args.height} @ {args.fps} fps)")
    print(f"  forward x: 0..{(args.frames_forward - 1) * args.step:.2f} m, "
          f"return y=7.0 m, turn radius 3.5 m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
