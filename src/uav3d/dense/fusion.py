"""Multi-view point fusion, denoising and PLY export.

Fusion averages repeated observations of the same surface voxel, which is what
converts per-view depth noise into a single, denser estimate. It is done with a
sort-merge accumulator so memory stays bounded even when 150 views each
contribute a million pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from uav3d.dense.cameras import CameraView


@dataclass
class PointCloud:
    """A fused, coloured point cloud."""

    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray | None = None
    weights: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.points.shape[0])


def depth_normals(view: CameraView, depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Estimate world-space normals from a depth map's local surface tangents.

    Returns an ``(H, W, 3)`` array; entries for invalid pixels are zero.
    """
    height, width = depth.shape
    xs = np.arange(width, dtype=np.float64) + 0.5
    ys = np.arange(height, dtype=np.float64) + 0.5
    gx, gy = np.meshgrid(xs, ys)
    d = np.where(valid, depth, 1.0).astype(np.float64)
    cam_points = np.stack(
        [
            (gx - view.intrinsics.cx) / view.intrinsics.fx * d,
            (gy - view.intrinsics.cy) / view.intrinsics.fy * d,
            d,
        ],
        axis=-1,
    )
    du = np.zeros_like(cam_points)
    dv = np.zeros_like(cam_points)
    du[:, 1:-1] = cam_points[:, 2:] - cam_points[:, :-2]
    dv[1:-1, :] = cam_points[2:, :] - cam_points[:-2, :]
    normals = np.cross(du, dv)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-12)
    # World-space orientation, flipped to face the camera.
    world_normals = normals @ view.R
    to_camera = view.center[None, None, :] - ((cam_points - view.t) @ view.R)
    flip = np.sum(world_normals * to_camera, axis=-1) < 0
    world_normals[flip] *= -1.0
    world_normals[~valid] = 0.0
    return world_normals.astype(np.float32)


def backproject_view(
    view: CameraView,
    estimate_depth: np.ndarray,
    valid: np.ndarray,
    color_image: np.ndarray | None = None,
    normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Back-project a filtered depth map into world points (and colour/normals)."""
    flat_valid = valid.ravel()
    if not np.any(flat_valid):
        return np.zeros((0, 3), np.float64), None, None
    uv = view.pixel_grid()[flat_valid]
    depth = estimate_depth.ravel()[flat_valid].astype(np.float64)
    points = view.backproject(uv, depth)
    colors = None
    if color_image is not None:
        h, w = color_image.shape[:2]
        u = np.clip(uv[:, 0].astype(np.int32), 0, w - 1)
        v = np.clip(uv[:, 1].astype(np.int32), 0, h - 1)
        colors = color_image[v, u][:, ::-1].astype(np.float32)  # BGR -> RGB
        if colors.max(initial=0.0) > 1.5:
            colors *= 1.0 / 255.0
    nrm = None
    if normals is not None:
        nrm = normals.reshape(-1, 3)[flat_valid]
    return points, colors, nrm


def _hash_keys(ijk: np.ndarray) -> np.ndarray:
    """Combine integer voxel coordinates into one collision-resistant int64 key."""
    x = ijk[:, 0].astype(np.int64)
    y = ijk[:, 1].astype(np.int64)
    z = ijk[:, 2].astype(np.int64)
    return (x * 73856093) ^ (y * 19349663) ^ (z * 83492791)


@dataclass
class VoxelAccumulator:
    """Weighted voxel accumulator with bounded memory.

    Points are buffered and periodically merged by voxel key. The result is the
    weighted mean position and colour of every occupied voxel.
    """

    voxel: float
    max_buffered: int = 3_000_000
    keys: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    sums: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float64))
    counts: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))
    colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float64))
    nsums: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float64))
    _buf_pts: list[np.ndarray] = field(default_factory=list)
    _buf_cols: list[np.ndarray] = field(default_factory=list)
    _buf_nrm: list[np.ndarray] = field(default_factory=list)
    _buf_n: int = 0

    @property
    def buffered(self) -> int:
        """Number of points waiting to be merged."""
        return self._buf_n

    def add(
        self,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        normals: np.ndarray | None = None,
    ) -> None:
        """Buffer points (with optional colour and normals) for fusion."""
        if points.shape[0] == 0:
            return
        self._buf_pts.append(np.asarray(points, dtype=np.float64))
        self._buf_cols.append(
            np.asarray(colors, dtype=np.float64) if colors is not None else np.zeros_like(points, dtype=np.float64)
        )
        self._buf_nrm.append(
            np.asarray(normals, dtype=np.float64) if normals is not None else np.zeros_like(points, dtype=np.float64)
        )
        self._buf_n += points.shape[0]
        if self._buf_n >= self.max_buffered:
            self.flush()

    def flush(self) -> None:
        """Merge the buffer into the accumulated voxel table."""
        if self._buf_n == 0:
            return
        pts = np.concatenate(self._buf_pts, axis=0)
        cols = np.concatenate(self._buf_cols, axis=0)
        nrms = np.concatenate(self._buf_nrm, axis=0)
        self._buf_pts.clear()
        self._buf_cols.clear()
        self._buf_nrm.clear()
        self._buf_n = 0

        ijk = np.floor(pts / self.voxel).astype(np.int64)
        keys = _hash_keys(ijk)
        uniq, inv = np.unique(keys, return_inverse=True)
        counts = np.bincount(inv, minlength=uniq.shape[0]).astype(np.float64)
        sums = np.stack(
            [np.bincount(inv, weights=pts[:, c], minlength=uniq.shape[0]) for c in range(3)], axis=1
        )
        col_sums = np.stack(
            [np.bincount(inv, weights=cols[:, c], minlength=uniq.shape[0]) for c in range(3)], axis=1
        )
        nrm_sums = np.stack(
            [np.bincount(inv, weights=nrms[:, c], minlength=uniq.shape[0]) for c in range(3)], axis=1
        )

        if self.keys.shape[0] == 0:
            self.keys, self.sums, self.counts, self.colors, self.nsums = uniq, sums, counts, col_sums, nrm_sums
            return

        all_keys = np.concatenate([self.keys, uniq])
        all_counts = np.concatenate([self.counts, counts])
        all_sums = np.concatenate([self.sums, sums], axis=0)
        all_cols = np.concatenate([self.colors, col_sums], axis=0)
        all_nrms = np.concatenate([self.nsums, nrm_sums], axis=0)
        uniq2, inv2 = np.unique(all_keys, return_inverse=True)
        n = uniq2.shape[0]
        self.counts = np.bincount(inv2, weights=all_counts, minlength=n)
        self.sums = np.stack([np.bincount(inv2, weights=all_sums[:, c], minlength=n) for c in range(3)], axis=1)
        self.colors = np.stack([np.bincount(inv2, weights=all_cols[:, c], minlength=n) for c in range(3)], axis=1)
        self.nsums = np.stack([np.bincount(inv2, weights=all_nrms[:, c], minlength=n) for c in range(3)], axis=1)
        self.keys = uniq2

    def to_cloud(self) -> PointCloud:
        """Finalise and return the fused cloud."""
        self.flush()
        safe = np.maximum(self.counts, 1.0)[:, None]
        points = self.sums / safe
        colors = np.clip(self.colors / safe, 0.0, 1.0)
        normals = self.nsums / safe
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norm, 1e-12)
        return PointCloud(
            points=points,
            colors=colors,
            normals=normals.astype(np.float32),
            weights=self.counts.astype(np.float32),
        )


def filter_density(points: np.ndarray, voxel: float, min_count: int = 2) -> np.ndarray:
    """Keep points whose (coarser) voxel contains at least ``min_count`` samples.

    Cheap replacement for a k-NN statistical outlier filter: isolated flying
    points land alone in their own neighbourhood.
    """
    if points.shape[0] == 0 or min_count <= 1:
        return np.ones(points.shape[0], bool)
    ijk = np.floor(points / (voxel * 2.0)).astype(np.int64)
    keys = _hash_keys(ijk)
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return counts[inv] >= int(min_count)


def statistical_filter(points: np.ndarray, k: int = 8, std_ratio: float = 2.0) -> np.ndarray:
    """Remove points whose mean neighbour distance is an outlier.

    Optional: O(n log n) and memory hungry, so the pipeline uses the voxel
    density filter by default.
    """
    if points.shape[0] <= k:
        return np.ones(points.shape[0], bool)
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k + 1)
    mean_d = distances[:, 1:].mean(axis=1)
    threshold = mean_d.mean() + std_ratio * mean_d.std()
    return mean_d <= threshold


def write_ply(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
    normals: np.ndarray | None = None,
) -> Path:
    """Write a binary little-endian PLY point cloud (optionally coloured/normalled)."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = int(points.shape[0])
    has_colors = colors is not None and colors.shape[0] == n
    has_normals = normals is not None and normals.shape[0] == n

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}", "property float x",
              "property float y", "property float z"]
    fields: list[tuple[str, str]] = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if has_normals:
        header += ["property float nx", "property float ny", "property float nz"]
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if has_colors:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    header.append("end_header")

    records = np.zeros(n, dtype=np.dtype(fields))
    records["x"] = points[:, 0].astype(np.float32)
    records["y"] = points[:, 1].astype(np.float32)
    records["z"] = points[:, 2].astype(np.float32)
    if has_normals:
        records["nx"] = normals[:, 0].astype(np.float32)
        records["ny"] = normals[:, 1].astype(np.float32)
        records["nz"] = normals[:, 2].astype(np.float32)
    if has_colors:
        rgb = np.clip(np.asarray(colors, dtype=np.float64) * 255.0, 0, 255).astype(np.uint8)
        records["red"] = rgb[:, 0]
        records["green"] = rgb[:, 1]
        records["blue"] = rgb[:, 2]

    with open(dest, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        fh.write(records.tobytes())
    return dest


def write_obj(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    """Write a triangle mesh as Wavefront OBJ."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="ascii") as fh:
        fh.write("# uav3d dense reconstruction\n")
        for p in vertices:
            fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for f in faces:
            fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")
    return dest
