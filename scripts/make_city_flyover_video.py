"""Generate a realistic synthetic single-pass UAV city flyover video with 3D ground truth.

Unlike scripts/make_parallax_video.py (abstract grid + trees, geometry too trivial to
judge model quality), this renders a small textured city district - buildings with
windowed facades, parapet roofs with AC units, roads with lane markings, grass/plaza
blocks - projected in true 3D (ray-cast ground plane + painter-sorted textured quads),
so frames exhibit genuine perspective parallax and rich SIFT texture.

The camera flies a realistic single straight pass (gentle lateral weave + yaw
oscillation), oblique forward-down view, ~45-55 m AGL - the exact profile the
SIH 26158 single-pass pipeline is built for.

Because the scene is fully known, the script also exports ground truth so you can
verify the reconstruction quantitatively instead of eyeballing it:

  <out>.mp4                      the test video
  <out>.GroundTruthAGL.csv       per-frame camera centers (image_id = frame index,
                                 matches trailing digits of `video_XXXXXX` frame ids,
                                 so `uav3d reconstruct --gt-file <this>` runs ATE)
  <out>.gt_model.ply             dense colored point cloud of the true scene surfaces
                                 (compare against the pipeline's model.ply in
                                 CloudCompare via ICP + cloud-to-cloud distance)
  <out>.gt_scene.json            building footprints + heights (LoD2 reference)
  <out>.meta.json                camera/scene parameters

Usage:
    uv run python scripts/make_city_flyover_video.py \
        --out data/raw/test_videos/city_flyover.mp4 --frames 600

Output is fully deterministic (fixed RNG seed, fixed camera path).
World frame: ground plane y=0, +y points DOWN (same convention as
make_parallax_video.py); camera altitude is -y. Units are meters.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SEED = 20260922
GROUND_HALF = 150.0  # ground plane spans [-150, 150] m in x and z
TEX_RES = 2048  # ground texture resolution (pixels)
ROAD_WIDTH = 8.0  # m
ROAD_XS = (-90.0, -30.0, 30.0, 90.0)  # north-south roads (constant x)
ROAD_ZS = (-60.0, 0.0, 60.0)  # east-west roads (constant z)


@dataclass(frozen=True)
class Building:
    """Axis-aligned box: base at y=0 (ground), roof at y=-height (up is -y)."""

    cx: float  # center x (east)
    cz: float  # center z (north)
    w: float  # footprint width along x (m)
    d: float  # footprint depth along z (m)
    h: float  # height (m)
    wall_rgb: tuple[int, int, int]
    roof_rgb: tuple[int, int, int]
    seed: int


def _buildings() -> list[Building]:
    """Deterministic city blocks lining both sides of the flight corridor (x axis)."""
    rng = np.random.default_rng(SEED)
    wall_palette = [
        (178, 172, 160),  # beige concrete
        (150, 152, 158),  # gray
        (164, 120, 96),  # brick
        (188, 182, 172),  # light stone
        (132, 140, 150),  # blue-gray
    ]
    roof_palette = [(96, 96, 100), (110, 108, 104), (88, 92, 96), (120, 116, 110)]

    buildings: list[Building] = []
    # Bands on both sides of the corridor plus occasional in-corridor blocks, so
    # the oblique camera sees facades frontally and rooftops from above (a pure
    # side-lined corridor only yields grazing slivers, useless for testing).
    rows = ((-50.0, -14.0), (14.0, 50.0))
    xs = range(-110, 111, 20)
    bseed = 0
    for band_lo, band_hi in rows:
        for x in xs:
            if rng.random() < 0.15:  # occasional empty lot
                continue
            w = float(rng.uniform(10.0, 18.0))
            d = float(rng.uniform(10.0, 18.0))
            h = float(rng.uniform(12.0, 42.0))
            cz = float(rng.uniform(band_lo + d / 2, band_hi - d / 2))
            cx = float(x + rng.uniform(-3.0, 3.0))
            buildings.append(
                Building(
                    cx=cx, cz=cz, w=w, d=d, h=h,
                    wall_rgb=wall_palette[bseed % len(wall_palette)],
                    roof_rgb=roof_palette[bseed % len(roof_palette)],
                    seed=bseed,
                )
            )
            bseed += 1
    # Low in-corridor buildings the camera flies directly over (rooftop parallax test)
    for x, z, h in ((-70.0, 4.0, 9.0), (-15.0, -5.0, 12.0), (45.0, 5.0, 8.0), (85.0, -4.0, 11.0)):
        buildings.append(
            Building(
                cx=x, cz=z, w=13.0, d=10.0, h=h,
                wall_rgb=wall_palette[bseed % len(wall_palette)],
                roof_rgb=roof_palette[bseed % len(roof_palette)],
                seed=bseed,
            )
        )
        bseed += 1
    # A few landmarks close to the corridor for strong near-field parallax
    buildings.append(Building(cx=-40.0, cz=12.0, w=14.0, d=10.0, h=16.0,
                              wall_rgb=(170, 160, 148), roof_rgb=(100, 98, 94), seed=bseed))
    buildings.append(Building(cx=35.0, cz=-13.0, w=12.0, d=12.0, h=22.0,
                              wall_rgb=(156, 148, 140), roof_rgb=(92, 92, 96), seed=bseed + 1))
    return buildings


def _ground_texture() -> np.ndarray:
    """Build the deterministic ground texture (TEX_RES x TEX_RES x 3, RGB)."""
    rng = np.random.default_rng(SEED + 1)
    m_per_px = (2.0 * GROUND_HALF) / TEX_RES

    # Base: mottled grass/dirt via blurred noise at ~1.2 m blob scale, boosted contrast
    noise = rng.integers(0, 255, size=(TEX_RES // 4, TEX_RES // 4, 3), dtype=np.uint8)
    tex = cv2.resize(noise, (TEX_RES, TEX_RES), interpolation=cv2.INTER_LINEAR)
    tex = cv2.GaussianBlur(tex, (0, 0), 1.5)
    grass = np.array([92, 122, 78], dtype=np.float64)
    dirt = np.array([132, 112, 88], dtype=np.float64)
    mix = np.clip((tex.astype(np.float64) - 127.5) * 1.6 + 127.5, 0, 255) / 255.0
    img = grass * mix + dirt * (1.0 - mix)

    # Plaza/asphalt blocks between roads (light concrete with noise)
    block_xs = [-120.0, -60.0, 0.0, 60.0, 120.0]
    block_zs = [-95.0, -30.0, 30.0, 95.0]
    for bx in block_xs:
        for bz in block_zs:
            if rng.random() < 0.5:
                continue
            half_w = float(rng.uniform(14.0, 22.0))
            half_d = float(rng.uniform(14.0, 22.0))
            x0 = int((bx - half_w + GROUND_HALF) / m_per_px)
            x1 = int((bx + half_w + GROUND_HALF) / m_per_px)
            z0 = int((bz - half_d + GROUND_HALF) / m_per_px)
            z1 = int((bz + half_d + GROUND_HALF) / m_per_px)
            base = np.array(rng.uniform(120.0, 165.0, size=3))
            block_noise = rng.normal(0.0, 9.0, size=(z1 - z0, x1 - x0, 1))
            img[z0:z1, x0:x1] = np.clip(base + block_noise, 0, 255)

    # Roads: dark asphalt strips with dashed white center lines
    asphalt = np.array([62, 62, 66])
    half_road = ROAD_WIDTH / 2.0
    for rx in ROAD_XS:
        x0 = int((rx - half_road + GROUND_HALF) / m_per_px)
        x1 = int((rx + half_road + GROUND_HALF) / m_per_px)
        img[:, x0:x1] = asphalt
    for rz in ROAD_ZS:
        z0 = int((rz - half_road + GROUND_HALF) / m_per_px)
        z1 = int((rz + half_road + GROUND_HALF) / m_per_px)
        img[z0:z1, :] = asphalt
    # Dashed center lines (2 m dash, 3 m gap)
    dash, gap = 2.0, 3.0
    line_px = max(1, int(0.15 / m_per_px))
    for rx in ROAD_XS:
        cxp = int((rx + GROUND_HALF) / m_per_px)
        s = 0.0
        while s < 2.0 * GROUND_HALF:
            e = min(s + dash, 2.0 * GROUND_HALF)
            r0, r1 = int(s / m_per_px), int(e / m_per_px)
            img[r0:r1, cxp - line_px : cxp + line_px] = (225, 225, 220)
            s = e + gap
    for rz in ROAD_ZS:
        czp = int((rz + GROUND_HALF) / m_per_px)
        s = 0.0
        while s < 2.0 * GROUND_HALF:
            e = min(s + dash, 2.0 * GROUND_HALF)
            c0, c1 = int(s / m_per_px), int(e / m_per_px)
            img[czp - line_px : czp + line_px, c0:c1] = (225, 225, 220)
            s = e + gap

    # Parked cars / street clutter: small bright/dark rectangles near roads
    for _ in range(240):
        along = float(rng.uniform(-GROUND_HALF + 4, GROUND_HALF - 4))
        side = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(2.2, 3.2))
        if rng.random() < 0.5:
            rx = float(rng.choice(ROAD_XS))
            px, pz = rx + side, along
            wp, dp = 2.0, 4.4  # car width/length (x,z)
        else:
            rz = float(rng.choice(ROAD_ZS))
            px, pz = along, rz + side
            wp, dp = 4.4, 2.0
        col = rng.integers(40, 240, size=3)
        x0 = int((px - wp / 2 + GROUND_HALF) / m_per_px)
        x1 = int((px + wp / 2 + GROUND_HALF) / m_per_px)
        z0 = int((pz - dp / 2 + GROUND_HALF) / m_per_px)
        z1 = int((pz + dp / 2 + GROUND_HALF) / m_per_px)
        img[z0:z1, x0:x1] = col

    return np.clip(img, 0, 255).astype(np.uint8)

    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def camera_pose(index: int, total: int, straight: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, C): world->camera rotation and camera center in world (m).

    If straight=False: single pass along +x from x=-100 to +100, gentle lateral weave
    and yaw oscillation, oblique forward-down view at 44-56 m AGL.
    If straight=True: dead-straight collinear pass along +x at constant y=-50m, cz=0,
    pitch=50.4 deg, fixed yaw, zero roll.
    """
    t = index / max(1, total - 1)
    cx = -100.0 + 200.0 * t

    if straight:
        cz = 0.0
        alt = 50.0
        pitch = 0.88  # ~50.4 deg down
        yaw = -np.pi / 2.0  # base: face +x exactly
        roll = 0.0
    else:
        cz = 10.0 * np.sin(2.0 * np.pi * t) + 3.0 * np.sin(6.0 * np.pi * t)
        alt = 50.0 + 6.0 * np.sin(2.0 * np.pi * t + 0.7)
        pitch = 0.88 + 0.05 * np.sin(2.0 * np.pi * t + 2.0)  # ~50 deg down
        yaw = -np.pi / 2.0 + 0.13 * np.sin(2.0 * np.pi * t + 1.2)  # base: face +x
        roll = 0.05 * np.sin(4.0 * np.pi * t)

    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(pitch), -np.sin(pitch)], [0.0, np.sin(pitch), np.cos(pitch)]]
    )
    ry = np.array(
        [[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]]
    )
    rz = np.array(
        [[np.cos(roll), -np.sin(roll), 0.0], [np.sin(roll), np.cos(roll), 0.0], [0.0, 0.0, 1.0]]
    )
    R = rz @ rx @ ry
    C = np.array([cx, -alt, cz], dtype=np.float64)
    return R, C


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_ground_and_sky(
    img: np.ndarray, R: np.ndarray, C: np.ndarray, K: np.ndarray, texture: np.ndarray
) -> None:
    """Ray-cast the ground plane (sampled from texture) and sky gradient, in place."""
    height, width = img.shape[:2]
    f = K[0, 0]
    cx, cy = K[0, 2], K[1, 2]

    us = np.arange(width, dtype=np.float64)
    vs = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    dirs_cam = np.stack([(uu - cx) / f, (vv - cy) / f, np.ones_like(uu)], axis=-1)  # (H,W,3)
    dirs_world = dirs_cam @ R  # R^T @ d, row-vector form

    dy = dirs_world[..., 1]
    ground_mask = dy > 1e-9

    # Sky: horizon haze fading to blue overhead (dy <= 0 -> ray points up), BGR order
    sky_t = np.clip(-dy / 0.6, 0.0, 1.0)[..., None]
    horizon = np.array([218.0, 208.0, 196.0])
    zenith = np.array([215.0, 150.0, 110.0])
    sky_col = horizon * (1.0 - sky_t) + zenith * sky_t
    img[:] = sky_col.astype(np.uint8)

    if not ground_mask.any():
        return

    alt = -C[1]
    t_hit = np.where(ground_mask, alt / np.maximum(dy, 1e-9), 0.0)
    hit_x = C[0] + t_hit * dirs_world[..., 0]
    hit_z = C[2] + t_hit * dirs_world[..., 2]

    scale = (TEX_RES - 1) / (2.0 * GROUND_HALF)
    tx = np.clip((hit_x + GROUND_HALF) * scale, 0, TEX_RES - 1).astype(np.int32)
    tz = np.clip((hit_z + GROUND_HALF) * scale, 0, TEX_RES - 1).astype(np.int32)
    cols = texture[tz, tx]  # (H,W,3) uint8

    # Mild distance haze so the far field does not alias into hard noise (BGR)
    dist = t_hit
    haze = np.clip((dist - 120.0) / 400.0, 0.0, 0.45)[..., None]
    haze_col = np.array([218.0, 208.0, 196.0])
    out = cols.astype(np.float64) * (1.0 - haze) + haze_col * haze
    img[ground_mask] = out[ground_mask].astype(np.uint8)


def _project(points: np.ndarray, R: np.ndarray, C: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points; return (pixels (N,2), in-front mask)."""
    cam = (points - C) @ R.T
    front = cam[:, 2] > 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        px = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
        py = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    px[~np.isfinite(px)] = -1e9
    py[~np.isfinite(py)] = -1e9
    return np.column_stack([px, py]), front


def _facade_quads(b: Building) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    """Textured quads for one building: wall base + window grid + roof + parapet + AC units.

    Returns a list of (4x3 world corners, RGB color). Windows are offset 5 cm
    outward from the wall so they paint over the wall base quad.
    """
    quads: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    rng = np.random.default_rng(SEED + 100 + b.seed)
    x0, x1 = b.cx - b.w / 2, b.cx + b.w / 2
    z0, z1 = b.cz - b.d / 2, b.cz + b.d / 2
    y_top, y_base = -b.h, 0.0

    walls = [
        # (corner a, corner c, horizontal length, outward normal)
        (np.array([x0, 0.0, z0]), np.array([x1, 0.0, z0]), b.w, np.array([0.0, 0.0, -1.0])),  # north
        (np.array([x1, 0.0, z1]), np.array([x0, 0.0, z1]), b.w, np.array([0.0, 0.0, 1.0])),  # south
        (np.array([x0, 0.0, z1]), np.array([x0, 0.0, z0]), b.d, np.array([-1.0, 0.0, 0.0])),  # west
        (np.array([x1, 0.0, z0]), np.array([x1, 0.0, z1]), b.d, np.array([1.0, 0.0, 0.0])),  # east
    ]
    # Sun from the south-east: fixed per-face shading for stable SIFT appearance
    shades = [0.82, 1.00, 0.74, 0.92]
    for (a, c, length, normal), shade in zip(walls, shades, strict=True):
        wall_col = tuple(int(min(255, v * shade)) for v in b.wall_rgb)
        quads.append(
            (
                np.array(
                    [[a[0], y_base, a[2]], [c[0], y_base, c[2]], [c[0], y_top, c[2]], [a[0], y_top, a[2]]]
                ),
                wall_col,
            )
        )
        # Window grid: ~1.7 m x 1.9 m windows on a 3.0 m x 3.2 m pitch
        axis = (c - a) / max(length, 1e-9)
        off = normal * 0.05
        n_cols = max(1, int((length - 1.2) / 3.0))
        n_rows = max(1, int((b.h - 2.0) / 3.2))
        win_w = min(1.7, (length - 1.2) / n_cols * 0.6)
        for col_i in range(n_cols):
            u = 0.6 + (length - 1.2) * (col_i + 0.5) / n_cols
            for row_i in range(n_rows):
                y_c = -(1.4 + (b.h - 2.6) * (row_i + 0.5) / n_rows)
                lit = rng.random() < 0.22
                if lit:
                    wc = tuple(int(v) for v in rng.integers(200, 250, size=3))
                else:
                    wc = tuple(int(v * shade) for v in (58, 78, 96))  # dark glass
                jitter = rng.integers(-12, 13, size=3)
                wc = tuple(int(np.clip(wc[k] + jitter[k], 0, 255)) for k in range(3))
                p = a + axis * u + off
                hw = axis * (win_w / 2)
                quads.append(
                    (
                        np.array(
                            [
                                [p[0] - hw[0], y_c - 0.95, p[2] - hw[2]],
                                [p[0] + hw[0], y_c - 0.95, p[2] + hw[2]],
                                [p[0] + hw[0], y_c + 0.95, p[2] + hw[2]],
                                [p[0] - hw[0], y_c + 0.95, p[2] - hw[2]],
                            ]
                        ),
                        wc,
                    )
                )

    # Roof slab + parapet ring
    quads.append(
        (np.array([[x0, y_top, z0], [x1, y_top, z0], [x1, y_top, z1], [x0, y_top, z1]]), b.roof_rgb)
    )
    par_col = tuple(max(0, v - 34) for v in b.roof_rgb)
    pw = 0.35  # parapet width
    for corners in (
        [(x0, z0), (x1, z0), (x1, z0 + pw), (x0, z0 + pw)],
        [(x0, z1 - pw), (x1, z1 - pw), (x1, z1), (x0, z1)],
        [(x0, z0 + pw), (x0 + pw, z0 + pw), (x0 + pw, z1 - pw), (x0, z1 - pw)],
        [(x1 - pw, z0 + pw), (x1, z0 + pw), (x1, z1 - pw), (x1 - pw, z1 - pw)],
    ):
        quads.append(
            (
                np.array([[qx, y_top - 0.9, qz] for qx, qz in corners]),
                par_col,
            )
        )
    # AC units: small bright boxes on the roof (3D features on an otherwise flat plane)
    n_ac = int(rng.integers(1, 4))
    for _ in range(n_ac):
        ax = float(rng.uniform(x0 + 1.5, x1 - 1.5))
        az = float(rng.uniform(z0 + 1.5, z1 - 1.5))
        s = float(rng.uniform(1.2, 2.2))
        ah = float(rng.uniform(0.8, 1.6))
        ac_col = tuple(int(v) for v in rng.integers(150, 220, size=3))
        top = y_top - ah
        quads.append(
            (
                np.array([[ax - s, top, az - s], [ax + s, top, az - s], [ax + s, top, az + s], [ax - s, top, az + s]]),
                ac_col,
            )
        )
        side_col = tuple(max(0, v - 40) for v in ac_col)
        quads.append(
            (
                np.array([[ax - s, top, az + s], [ax + s, top, az + s], [ax + s, y_top, az + s], [ax - s, y_top, az + s]]),
                side_col,
            )
        )
        quads.append(
            (
                np.array([[ax + s, top, az - s], [ax + s, top, az + s], [ax + s, y_top, az + s], [ax + s, y_top, az - s]]),
                side_col,
            )
        )
    return quads
def render_frame(
    index: int,
    total: int,
    width: int,
    height: int,
    focal: float,
    texture_bgr: np.ndarray,
    buildings: list[Building],
    straight: bool = False,
) -> np.ndarray:
    """Render one frame (BGR image). `texture_bgr` is the ground texture in BGR order."""
    R, C = camera_pose(index, total, straight=straight)
    K = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    img = np.zeros((height, width, 3), dtype=np.uint8)
    _render_ground_and_sky(img, R, C, K, texture_bgr)

    # Collect visible building quads with depth, paint far-to-near
    draw_list: list[tuple[float, np.ndarray, tuple[int, int, int]]] = []
    for b in buildings:
        center = np.array([b.cx, -b.h / 2.0, b.cz])
        dist = float(np.linalg.norm(center - C))
        if dist > 400.0:
            continue
        # behind-camera quick reject on the building centroid
        rel = (center - C) @ R.T
        reach = 2.0 * max(b.w, b.d, b.h)
        if rel[2] < 0.0 and dist > reach:
            continue
        for corners, col in _facade_quads(b):
            px, front = _project(corners, R, C, K)
            # near-plane clip: quads must lie fully in front of the camera,
            # otherwise projected coords blow up and fill the whole frame
            if not front.all():
                continue
            if px[:, 0].max() < 0 or px[:, 0].min() > width or px[:, 1].max() < 0 or px[:, 1].min() > height:
                continue
            depth = float(np.linalg.norm(corners.mean(axis=0) - C))
            draw_list.append((depth, px.astype(np.int32), col))

    draw_list.sort(key=lambda item: item[0], reverse=True)
    for _, px, col in draw_list:
        cv2.fillPoly(img, [px], (int(col[2]), int(col[1]), int(col[0])), lineType=cv2.LINE_AA)

    cv2.putText(
        img,
        f"CITY FLYOVER {index:05d}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return img


# ---------------------------------------------------------------------------
# Ground-truth export
# ---------------------------------------------------------------------------


def _write_gt_trajectory(csv_path: Path, total: int, straight: bool = False) -> None:
    """Write per-frame camera centers in GroundTruthAGL.csv layout (EVALUATION ONLY).

    image_id is the video frame index; `uav3d reconstruct --gt-file` matches it
    against the trailing digits of `video_XXXXXX` frame ids, so ATE just works.
    """
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["# image_id", "x", "y", "z"])
        for i in range(total):
            _, C = camera_pose(i, total, straight=straight)
            writer.writerow([i, f"{C[0]:.6f}", f"{C[1]:.6f}", f"{C[2]:.6f}"])


def _write_gt_point_cloud(ply_path: Path, texture: np.ndarray, buildings: list[Building], spacing: float = 1.0) -> int:
    """Sample all true scene surfaces into a colored binary PLY (RGB order)."""
    pts: list[np.ndarray] = []
    cols: list[np.ndarray] = []

    # Ground, colored from the texture
    g = np.arange(-GROUND_HALF, GROUND_HALF + 1e-6, spacing)
    gx, gz = np.meshgrid(g, g)
    scale = (TEX_RES - 1) / (2.0 * GROUND_HALF)
    tx = np.clip((gx + GROUND_HALF) * scale, 0, TEX_RES - 1).astype(np.int32)
    tz = np.clip((gz + GROUND_HALF) * scale, 0, TEX_RES - 1).astype(np.int32)
    pts.append(np.column_stack([gx.ravel(), np.zeros(gx.size), gz.ravel()]))
    cols.append(texture[tz, tx].reshape(-1, 3))

    for b in buildings:
        for corners, col in _facade_quads(b):
            q0, q1, q2, q3 = corners  # quad corners in order
            edge1 = float(np.linalg.norm(q1 - q0))
            edge2 = float(np.linalg.norm(q3 - q0))
            n1 = max(2, int(edge1 / spacing) + 1)
            n2 = max(2, int(edge2 / spacing) + 1)
            s1 = np.linspace(0.0, 1.0, n1)
            s2 = np.linspace(0.0, 1.0, n2)
            # bilinear interpolation over the quad
            p = (
                q0[None, None, :] * (1 - s1)[None, :, None] * (1 - s2)[:, None, None]
                + q1[None, None, :] * s1[None, :, None] * (1 - s2)[:, None, None]
                + q2[None, None, :] * s1[None, :, None] * s2[:, None, None]
                + q3[None, None, :] * (1 - s1)[None, :, None] * s2[:, None, None]
            )
            pts.append(p.reshape(-1, 3))
            cols.append(np.tile(np.array(col, dtype=np.uint8), (n1 * n2, 1)))

    P = np.concatenate(pts, axis=0).astype(np.float32)
    Col = np.concatenate(cols, axis=0).astype(np.uint8)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {P.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(ply_path, "wb") as f:
        f.write(header.encode("ascii"))
        rec = np.empty(
            P.shape[0],
            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
        )
        rec["x"], rec["y"], rec["z"] = P[:, 0], P[:, 1], P[:, 2]
        rec["red"], rec["green"], rec["blue"] = Col[:, 0], Col[:, 1], Col[:, 2]
        f.write(rec.tobytes())
    return int(P.shape[0])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def make_city_flyover_video(
    out_path: str | Path,
    frames: int = 600,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    focal_ratio: float = 0.9,
    straight: bool = False,
) -> Path:
    """Write the video plus all ground-truth artifacts; return the video path.

    focal_ratio defaults to 0.9 so the rendered intrinsics exactly match the
    pipeline's auto SIMPLE_PINHOLE prior (f = 0.9 * width) - one less failure
    source when judging reconstruction quality.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    texture = _ground_texture()
    texture_bgr = texture[..., ::-1].copy()  # cv2 renders/writes BGR; GT PLY keeps RGB
    buildings = _buildings()
    focal = float(width) * focal_ratio

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out}")

    for i in range(frames):
        writer.write(render_frame(i, frames, width, height, focal, texture_bgr, buildings, straight=straight))
        if (i + 1) % 60 == 0 or i + 1 == frames:
            print(f"  rendered {i + 1}/{frames} frames", flush=True)
    writer.release()

    gt_csv = out.with_suffix(".GroundTruthAGL.csv")
    _write_gt_trajectory(gt_csv, frames, straight=straight)

    gt_ply = out.with_suffix(".gt_model.ply")
    n_pts = _write_gt_point_cloud(gt_ply, texture, buildings)

    gt_scene = out.with_suffix(".gt_scene.json")
    gt_scene.write_text(
        json.dumps(
            {
                "description": "LoD2 reference: axis-aligned building boxes (meters, world y=0 ground, up=-y)",
                "buildings": [
                    {"cx": b.cx, "cz": b.cz, "width_x": b.w, "depth_z": b.d, "height": b.h}
                    for b in buildings
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    flight_desc = (
        "single dead-straight pass along +x, x=-100..+100 m, constant 50 m AGL, fixed pitch ~50 deg down, cz=0"
        if straight
        else "single pass along +x, x=-100..+100 m, 44-56 m AGL, ~38 deg oblique down, gentle weave"
    )

    meta = {
        "frames": frames,
        "width": width,
        "height": height,
        "fps": fps,
        "focal_px": focal,
        "focal_ratio": focal_ratio,
        "seed": SEED,
        "flight": flight_desc,
        "straight": straight,
        "gt_points": n_pts,
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="data/raw/test_videos/city_flyover.mp4")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--focal-ratio", type=float, default=0.9, help="Focal length as a multiple of width")
    parser.add_argument("--straight", action="store_true", help="Dead-straight collinear flight path (no weave/yaw oscillation)")
    args = parser.parse_args()
    path = make_city_flyover_video(args.out, args.frames, args.width, args.height, args.fps, args.focal_ratio, straight=args.straight)
    print(
        f"Wrote city flyover video: {path} "
        f"({args.frames} frames, {args.width}x{args.height} @ {args.fps} fps, seed={SEED}, straight={args.straight})\n"
        f"Ground truth: {path.with_suffix('.GroundTruthAGL.csv')}, "
        f"{path.with_suffix('.gt_model.ply')}, {path.with_suffix('.gt_scene.json')}"
    )


if __name__ == "__main__":
    main()