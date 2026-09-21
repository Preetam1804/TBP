"""Generate a deterministic synthetic UAV video with genuine 3D parallax for pipeline tests.

Renders a fixed 3D scene (textured ground grid + ground patches + vertical "trees")
from a camera flying *over* the scene at ~8 m altitude. Because the structure is
projected in true 3D (cv2.projectPoints) rather than 2D-warped, successive frames
exhibit real perspective parallax and are suitable SfM input.

Usage:
    uv run python scripts/make_parallax_video.py --out /tmp/parallax.mp4 \
        --frames 60 --width 640 --height 360 --fps 10

Output is fully deterministic (fixed RNG seed, fixed camera path).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SEED = 20260917
GRID_HALF = 10.0  # ground grid spans [-10, 10] x/z
GRID_STEP = 1.0
N_PATCHES = 60  # colored ground patches (1 m x 1 m)
N_TREES = 25  # vertical features with 1-3 m height
CAM_ALT = 8.0  # camera altitude above the ground plane


def _scene():
    """Deterministic scene definition (independent of image size)."""
    rng = np.random.default_rng(SEED)

    grid_x = np.arange(-GRID_HALF, GRID_HALF + 1e-6, GRID_STEP)
    grid_z = np.arange(-GRID_HALF, GRID_HALF + 1e-6, GRID_STEP)

    # Ground patches: 1m x 1m squares anchored at random grid cells
    cell_ix = rng.integers(0, len(grid_x) - 1, size=N_PATCHES)
    cell_iz = rng.integers(0, len(grid_z) - 1, size=N_PATCHES)
    patch_colors = rng.integers(60, 230, size=(N_PATCHES, 3))

    # Trees: base at random ground positions, height 1-3 m
    tree_base = np.column_stack(
        [rng.uniform(-8.0, 8.0, size=N_TREES), np.full(N_TREES, 0.0), rng.uniform(-8.0, 8.0, size=N_TREES)]
    )
    tree_h = rng.uniform(1.0, 3.0, size=N_TREES)
    tree_colors = rng.integers(90, 230, size=(N_TREES, 3))

    return grid_x, grid_z, cell_ix, cell_iz, patch_colors, tree_base, tree_h, tree_colors


def _camera(index: int, total: int, focal: float, width: int, height: int):
    """Camera extrinsics (R, t) and intrinsics (K) for a frame.

    The camera flies over the scene at CAM_ALT altitude along a gentle S-curve
    with yaw, looking down. The scene always stays below/in front of the camera.
    """
    t = index / max(1, total - 1)  # 0..1
    tx = 3.0 * np.sin(2.0 * np.pi * t)
    ty = -CAM_ALT
    tz = 9.0 - 16.0 * t  # fly from z=+9 toward z=-7 (over the scene)

    yaw = 0.6 * np.sin(2.0 * np.pi * t)
    roll = 0.15 * np.sin(2.0 * np.pi * t + 1.0)
    pitch = 0.35  # radians, looking down toward the ground

    R, _ = cv2.Rodrigues(np.array([pitch, yaw, roll], dtype=np.float64))
    tvec = np.array([tx, ty, tz], dtype=np.float64)
    K = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return R, tvec, K


def _project(pts_world: np.ndarray, R: np.ndarray, tvec: np.ndarray, K: np.ndarray):
    """Project 3D points; return (px, py) array and in-front-of-camera mask."""
    cam = (pts_world - tvec) @ R.T
    front = cam[:, 2] > 0.1  # reject points at/beyond the near plane
    with np.errstate(divide="ignore", invalid="ignore"):
        img = (K @ cam.T).T
        px = img[:, 0] / img[:, 2]
        py = img[:, 1] / img[:, 2]
    px[~np.isfinite(px)] = -1e9
    py[~np.isfinite(py)] = -1e9
    return np.column_stack([px, py]), front


def render_frame(index: int, total: int, width: int, height: int, focal: float) -> np.ndarray:
    """Render one deterministic frame of the flying synthetic scene."""
    grid_x, grid_z, cell_ix, cell_iz, patch_colors, tree_base, tree_h, tree_colors = _scene()
    R, tvec, K = _camera(index, total, focal, width, height)

    img = np.full((height, width, 3), (40, 52, 66), dtype=np.uint8)

    # --- ground grid lines (per-segment depth check) ---
    gx = np.arange(-GRID_HALF, GRID_HALF + 1e-6, GRID_STEP)
    gz = np.arange(-GRID_HALF, GRID_HALF + 1e-6, GRID_STEP)
    def seg(x1, z1, x2, z2, color):
        p, front = _project(np.array([[x1, 0.0, z1], [x2, 0.0, z2]]), R, tvec, K)
        if front.all():
            cv2.line(img, (int(p[0][0]), int(p[0][1])), (int(p[1][0]), int(p[1][1])), color, 1, cv2.LINE_AA)

    for z in gz:
        seg(gx[0], z, gx[-1], z, (58, 74, 58))
    for x in gx:
        seg(x, gz[0], x, gz[-1], (58, 74, 58))

    # --- ground patches (1m x 1m colored squares) ---
    for i in range(N_PATCHES):
        x0, z0 = grid_x[cell_ix[i]], grid_z[cell_iz[i]]
        x1, z1 = x0 + GRID_STEP, z0 + GRID_STEP
        pts = np.array(
            [[x0, 0.0, z0], [x1, 0.0, z0], [x1, 0.0, z1], [x0, 0.0, z1]],
            dtype=np.float64,
        )
        proj, front = _project(pts, R, tvec, K)
        if front.all():
            poly = proj.astype(np.int32)
            col = tuple(int(c) for c in patch_colors[i])
            cv2.fillPoly(img, [poly], col, lineType=cv2.LINE_AA)
            cv2.polylines(img, [poly], True, (20, 30, 20), 1, cv2.LINE_AA)

    # --- trees (vertical lines with a crown) ---
    for i in range(N_TREES):
        base = tree_base[i]
        top = base.copy()
        top[1] += tree_h[i]
        proj, front = _project(np.stack([base, top]), R, tvec, K)
        if front.all():
            (bx, by), (tx, ty_) = proj[0], proj[1]
            col = tuple(int(c) for c in tree_colors[i])
            cv2.line(img, (int(bx), int(by)), (int(tx), int(ty_)), (60, 80, 60), 2, cv2.LINE_AA)
            cv2.circle(img, (int(tx), int(ty_)), 6, col, -1, cv2.LINE_AA)

    # --- watermark (screen-fixed, unique text) ---
    cv2.putText(img, f"PARALLAX {index:05d}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def make_parallax_video(
    out_path: str | Path,
    frames: int = 60,
    width: int = 640,
    height: int = 360,
    fps: float = 10.0,
    focal_ratio: float = 1.0,
) -> Path:
    """Write the synthetic parallax video and return its path."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out}")

    focal = float(width) * focal_ratio
    for i in range(frames):
        writer.write(render_frame(i, frames, width, height, focal))
    writer.release()

    meta = {
        "frames": frames,
        "width": width,
        "height": height,
        "fps": fps,
        "focal_px": focal,
        "seed": SEED,
        "cam_alt_m": CAM_ALT,
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="data/interim/parallax.mp4")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--focal-ratio", type=float, default=1.0, help="Focal length as a multiple of width")
    args = parser.parse_args()
    path = make_parallax_video(args.out, args.frames, args.width, args.height, args.fps, args.focal_ratio)
    print(
        f"Wrote synthetic parallax video: {path} "
        f"({args.frames} frames, {args.width}x{args.height} @ {args.fps} fps, seed={SEED})"
    )


if __name__ == "__main__":
    main()
