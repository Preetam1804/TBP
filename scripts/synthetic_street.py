"""Analytic street-canyon ray tracer used to validate the dense stage.

The scene is a set of axis-aligned boxes (ground, two facades, balconies) with a
procedural multi-scale texture. Rendering returns an exact Z-depth ground truth
per pixel, which is what makes the benchmark in ``dense_benchmark.py`` an
actual measurement instead of a visual impression.

This module exists only for validation and tests; the production pipeline never
imports it.
"""

from __future__ import annotations

import numpy as np


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _hash3(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    h = (
        (ix.astype(np.int64) * 73856093)
        ^ (iy.astype(np.int64) * 19349663)
        ^ (iz.astype(np.int64) * 83492791)
        ^ (np.int64(seed) * 2654435761)
    )
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF).astype(np.float32) / 65535.0


def texture(points: np.ndarray, seed: int) -> np.ndarray:
    """Multi-scale procedural albedo in ``[0, 1]`` with useful high frequencies."""
    out = np.zeros(len(points), np.float32)
    amp, total = 1.0, 0.0
    for scale, falloff in ((0.5, 0.8), (0.18, 0.6), (0.07, 0.4)):
        q = np.floor(points / scale)
        out += amp * _hash3(q[:, 0], q[:, 1], q[:, 2], seed)
        total += amp
        amp *= falloff
    return np.clip(0.30 + 0.70 * (out / total), 0.0, 1.0)


def build_street(length: float = 160.0, half_width: float = 9.0, height: float = 24.0) -> list[dict]:
    """Ground plane, two facades with balconies — a street canyon."""
    boxes: list[dict] = [
        dict(
            lo=np.array([-20.0, -half_width - 8.0, -1.2]),
            hi=np.array([length + 20.0, half_width + 8.0, 0.0]),
            seed=11,
        )
    ]
    for sign in (-1, 1):
        y_in = sign * half_width
        y_out = sign * (half_width + 0.8)
        boxes.append(
            dict(
                lo=np.array([-20.0, min(y_in, y_out), 0.0]),
                hi=np.array([length + 20.0, max(y_in, y_out), height]),
                seed=21,
            )
        )
        for i in range(int(length / 6.0)):
            x0 = i * 6.0
            y_balcony = y_in - sign * 0.7
            boxes.append(
                dict(
                    lo=np.array([x0 + 0.5, min(y_in, y_balcony), 5.5]),
                    hi=np.array([x0 + 3.0, max(y_in, y_balcony), 5.9]),
                    seed=31 + i,
                )
            )
            y_window = y_in - sign * 0.25
            boxes.append(
                dict(
                    lo=np.array([x0 + 0.5, min(y_in, y_window), 8.5]),
                    hi=np.array([x0 + 3.0, max(y_in, y_window), 10.5]),
                    seed=41 + i,
                )
            )
    return boxes


def trace(origins: np.ndarray, directions: np.ndarray, boxes: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray-box intersection returning distance, hit axis and face sign."""
    n = len(origins)
    best = np.full(n, np.inf)
    axis_out = np.zeros(n, np.int8)
    sign_out = np.zeros(n, np.int8)
    inv = 1.0 / np.where(np.abs(directions) < 1e-12, 1e-12, directions)
    rows = np.arange(n)
    for box in boxes:
        t0 = (box["lo"] - origins) * inv
        t1 = (box["hi"] - origins) * inv
        tmin = np.minimum(t0, t1)
        tmax = np.maximum(t0, t1)
        axis = np.argmax(tmin, axis=1)
        near = tmin[rows, axis]
        far = tmax.min(axis=1)
        update = (far >= near) & (near > 1e-6) & (near < best)
        best = np.where(update, near, best)
        axis_out = np.where(update, axis, axis_out)
        sign_out = np.where(update, -np.sign(directions[rows, axis]), sign_out)
    return best, axis_out, sign_out


def render(
    R: np.ndarray,
    t: np.ndarray,
    fx: float,
    fy: float,
    width: int,
    height: int,
    boxes: list[dict],
    supersample: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one view. Returns ``(gray_uint8, depth, hit_mask)``.

    ``supersample`` anti-aliases by area integration, which a real lens does
    optically. Without it the point-sampled texture aliases and multi-view
    correlation collapses — an artefact of the renderer, not of the scene.
    """
    if supersample < 1:
        supersample = 1
    out_w, out_h = width, height
    width, height = width * supersample, height * supersample
    # Same field of view at `supersample` times the sampling density.
    cx, cy = width / 2.0, height / 2.0
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    rays = np.stack(
        [(u - cx) / (fx * supersample), (v - cy) / (fy * supersample), np.ones_like(u)], axis=-1
    ).reshape(-1, 3)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    world_dirs = rays @ R
    origin = np.broadcast_to(-R.T @ t, (len(rays), 3))
    distance, axis, sign = trace(origin, world_dirs, boxes)
    hit = np.isfinite(distance)
    points = origin + world_dirs * np.where(hit, distance, 0.0)[:, None]
    face = axis * 2 + (sign > 0)
    gray = np.zeros(len(distance), np.float32)
    for ax in range(3):
        for s in (0, 1):
            mask = hit & (face == ax * 2 + s)
            if mask.any():
                gray[mask] = texture(points[mask], 100 * ax + 10 * s + 7)
    depth = (distance * rays[:, 2]).reshape(height, width)
    gray_img = gray.reshape(height, width)
    hit_img = hit.reshape(height, width)

    if supersample > 1:
        s = supersample
        gray_img = gray_img.reshape(out_h, s, out_w, s).mean(axis=(1, 3))
        blocks = depth.reshape(out_h, s, out_w, s)
        d_min = blocks.min(axis=(1, 3))
        d_max = blocks.max(axis=(1, 3))
        depth_img = blocks.mean(axis=(1, 3))
        with np.errstate(invalid="ignore"):
            consistent = (d_max - d_min) <= 0.01 * np.maximum(d_min, 1e-9)
        consistent &= np.isfinite(d_min)
        hit_img = hit_img.reshape(out_h, s, out_w, s).all(axis=(1, 3)) & consistent
    else:
        depth_img = depth

    return np.clip(gray_img * 255.0, 0, 255).astype(np.uint8), depth_img, hit_img


def sample_surface(boxes: list[dict], count: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly sample points on the visible box surfaces (ground truth cloud)."""
    areas = []
    for box in boxes:
        size = box["hi"] - box["lo"]
        areas.append(2.0 * (size[0] * size[1] + size[1] * size[2] + size[0] * size[2]))
    areas = np.asarray(areas)
    picks = rng.choice(len(boxes), size=count, p=areas / areas.sum())
    out = np.empty((count, 3))
    for i, box in enumerate(boxes):
        sel = picks == i
        n = int(sel.sum())
        if n == 0:
            continue
        size = box["hi"] - box["lo"]
        pts = rng.random((n, 3)) * size + box["lo"]
        # Push each sample onto a random face of the box.
        face = rng.integers(0, 6, n)
        axis = face // 2
        value = np.where(face % 2 == 0, box["lo"][axis], box["hi"][axis])
        pts[np.arange(n), axis] = value
        out[sel] = pts
    return out


def flight(
    count: int,
    step: float,
    height: float,
    pitch: float,
    lateral: float = 0.0,
    yaw: float = 0.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Camera poses looking down +x from a forward-moving platform.

    Returns ``(R, t)`` pairs in ``cam_from_world`` convention.
    """
    base = rot_z(yaw) @ rot_y(-np.pi / 2) @ rot_x(pitch)
    poses = []
    for i in range(count):
        centre = np.array([i * step, lateral, height])
        poses.append((base.copy(), -base @ centre))
    return poses
