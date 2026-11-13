"""Camera intrinsics, poses and image pre-rectification for the dense stage.

Conventions (identical to the rest of the pipeline):

* pose is ``cam_from_world``: ``x_cam = R @ X_world + t``
* camera centre ``C = -R.T @ t``
* quaternions are ``(qw, qx, qy, qz)``
* pixel coordinates refer to pixel *centres* at integer-plus-half offsets
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from uav3d.exceptions import DataError
from uav3d.models.keyframes import KeyframesFile
from uav3d.models.poses import PosesFile
from uav3d.sfm.colmap_compat import quaternion_to_rotation_matrix

logger = logging.getLogger("uav3d.dense.cameras")

# COLMAP camera model -> (num params before distortion, distortion kind)
_MODEL_LAYOUT: dict[str, tuple[str, str]] = {
    "SIMPLE_PINHOLE": ("f", "none"),
    "PINHOLE": ("fx_fy", "none"),
    "SIMPLE_RADIAL": ("f", "simple_radial"),
    "RADIAL": ("f", "radial"),
    "OPENCV": ("fx_fy", "opencv"),
    "FULL_OPENCV": ("fx_fy", "full_opencv"),
}


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics for a single working resolution."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def matrix(self) -> np.ndarray:
        """Return the 3x3 camera matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def focal(self) -> float:
        """Return the mean focal length in pixels."""
        return 0.5 * (self.fx + self.fy)

    def scaled(self, scale: float) -> Intrinsics:
        """Return intrinsics scaled by ``scale`` (used for resized working images)."""
        return Intrinsics(
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=self.cx * scale,
            cy=self.cy * scale,
            width=max(1, int(round(self.width * scale))),
            height=max(1, int(round(self.height * scale))),
        )


def _split_params(model: str, params: list[float]) -> tuple[Intrinsics, np.ndarray | None]:
    """Split a COLMAP parameter vector into pinhole intrinsics and distortion coefficients."""
    model = model.upper()
    if model not in _MODEL_LAYOUT:
        raise DataError(
            f"Unsupported camera model '{model}' for dense reconstruction. "
            f"Supported: {', '.join(sorted(_MODEL_LAYOUT))}."
        )
    layout, kind = _MODEL_LAYOUT[model]
    p = [float(v) for v in params]

    if layout == "fx_fy":
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        rest = p[4:]
    else:
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
        rest = p[3:]

    if kind == "none":
        dist = None
    elif kind == "simple_radial":
        dist = np.array([rest[0], 0.0, 0.0, 0.0], dtype=np.float64)
    elif kind == "radial":
        dist = np.array([rest[0], rest[1], 0.0, 0.0], dtype=np.float64)
    elif kind == "opencv":
        dist = np.array([rest[0], rest[1], rest[2], rest[3]], dtype=np.float64)
    else:  # full_opencv
        dist = np.array([rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6], rest[7]], dtype=np.float64)

    # Width/height are filled in by the caller, which owns the PosesFile block.
    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=0, height=0), dist


def _build_remap(
    intrinsics: Intrinsics,
    dist: np.ndarray | None,
    target: Intrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Build undistortion + resize maps sampling the native image into ``target``."""
    K = intrinsics.matrix
    if dist is None or np.allclose(dist, 0.0):
        # Pure resize: sample the native grid on a scaled lattice.
        scale = target.fx / intrinsics.fx
        xs = (np.arange(target.width, dtype=np.float32) + 0.5) / scale - 0.5
        ys = (np.arange(target.height, dtype=np.float32) + 0.5) / scale - 0.5
        mapx, mapy = np.meshgrid(xs, ys)
        return mapx.astype(np.float32), mapy.astype(np.float32)

    # Undistort at native resolution, expressed in the target working resolution.
    scale = target.fx / intrinsics.fx
    K_native_new = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    mapx_native, mapy_native = cv2.initUndistortRectifyMap(
        K, dist, None, K_native_new, (intrinsics.width, intrinsics.height), cv2.CV_32FC1
    )
    # Convert native pixel coordinates into target coordinates.
    mapx = ((mapx_native + 0.5) * scale - 0.5).astype(np.float32)
    mapy = ((mapy_native + 0.5) * scale - 0.5).astype(np.float32)
    return mapx, mapy


@dataclass
class CameraView:
    """A registered keyframe with its pose, intrinsics and image location."""

    index: int
    frame_id: str
    image_path: Path
    R: np.ndarray
    t: np.ndarray
    intrinsics: Intrinsics

    @property
    def center(self) -> np.ndarray:
        """Return the camera centre in world coordinates, ``C = -R.T @ t``."""
        return -self.R.T @ self.t

    @property
    def forward(self) -> np.ndarray:
        """Return the world-space optical axis direction."""
        return self.R.T @ np.array([0.0, 0.0, 1.0])

    def project(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world points to pixel coordinates.

        Args:
            points_world: ``(N, 3)`` array of world points.

        Returns:
            ``(uv, z)`` where ``uv`` is ``(N, 2)`` pixels and ``z`` is ``(N,)``
            camera-frame depth. Callers must mask ``z <= 0`` themselves.
        """
        pc = points_world @ self.R.T + self.t
        z = pc[:, 2]
        safe = np.where(np.abs(z) < 1e-12, 1e-12, z)
        u = self.intrinsics.fx * pc[:, 0] / safe + self.intrinsics.cx
        v = self.intrinsics.fy * pc[:, 1] / safe + self.intrinsics.cy
        return np.stack([u, v], axis=1), z

    def backproject(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Back-project pixels with Z-depth into world coordinates.

        Args:
            uv: ``(N, 2)`` pixel coordinates.
            depth: ``(N,)`` camera-frame Z depth.

        Returns:
            ``(N, 3)`` world points.
        """
        x = (uv[:, 0] - self.intrinsics.cx) / self.intrinsics.fx * depth
        y = (uv[:, 1] - self.intrinsics.cy) / self.intrinsics.fy * depth
        pc = np.stack([x, y, depth], axis=1)
        return (pc - self.t) @ self.R

    def pixel_grid(self) -> np.ndarray:
        """Return pixel-centre coordinates for the working resolution as ``(H*W, 2)``."""
        xs = np.arange(self.intrinsics.width, dtype=np.float64) + 0.5
        ys = np.arange(self.intrinsics.height, dtype=np.float64) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        return np.stack([gx.ravel(), gy.ravel()], axis=1)


def load_run_inputs(run_dir: str | Path) -> tuple[PosesFile, KeyframesFile]:
    """Load ``poses.json`` and ``keyframes.json`` from a pipeline run directory."""
    run_path = Path(run_dir)
    poses_path = run_path / "poses.json"
    keyframes_path = run_path / "keyframes.json"
    if not poses_path.exists():
        raise DataError(f"poses.json not found in {run_path}. Run 'uav3d sfm' first.")
    if not keyframes_path.exists():
        raise DataError(f"keyframes.json not found in {run_path}. Run 'uav3d keyframes' first.")
    poses = PosesFile.model_validate_json(poses_path.read_text(encoding="utf-8"))
    keyframes = KeyframesFile.model_validate_json(keyframes_path.read_text(encoding="utf-8"))
    return poses, keyframes


def build_views(
    poses: PosesFile,
    keyframes: KeyframesFile,
    run_dir: str | Path,
    target_width: int,
) -> list[CameraView]:
    """Build registered :class:`CameraView` objects at the requested working width.

    Frames that are unregistered, lack an image on disk, or are not present in
    ``keyframes.json`` are skipped with a warning.
    """
    run_path = Path(run_dir)
    paths = {k.frame_id: k.file_path for k in keyframes.keyframes}

    if poses.camera is None:
        raise DataError("poses.json has no 'camera' block; cannot derive dense intrinsics.")

    native, dist = _split_params(poses.camera.model, poses.camera.params)
    native = Intrinsics(
        fx=native.fx,
        fy=native.fy,
        cx=native.cx,
        cy=native.cy,
        width=int(poses.camera.width),
        height=int(poses.camera.height),
    )
    scale = min(1.0, float(target_width) / float(native.width))
    working = native.scaled(scale)

    views: list[CameraView] = []
    for record in poses.poses:
        if not record.registered:
            continue
        rel = paths.get(record.frame_id)
        if rel is None:
            logger.warning("Frame %s has a pose but no keyframe record; skipping", record.frame_id)
            continue
        image_path = run_path / rel
        if not image_path.exists():
            logger.warning("Image missing for %s at %s; skipping", record.frame_id, image_path)
            continue
        R = quaternion_to_rotation_matrix([record.qw, record.qx, record.qy, record.qz])
        t = np.array([record.tx, record.ty, record.tz], dtype=np.float64)
        views.append(
            CameraView(
                index=len(views),
                frame_id=record.frame_id,
                image_path=image_path,
                R=R,
                t=t,
                intrinsics=working,
            )
        )

    if len(views) < 3:
        raise DataError(
            f"Only {len(views)} usable views found; dense MVS needs at least 3 registered frames."
        )

    # Stash the distortion model on the view objects for the rectification pass.
    for view in views:
        view._native_intrinsics = native  # type: ignore[attr-defined]
        view._distortion = dist  # type: ignore[attr-defined]

    return views


def undistort_run_images(
    views: list[CameraView],
    output_dir: str | Path,
    target_width: int,
    overwrite: bool = False,
) -> list[Path]:
    """Rectify every view into a pinhole camera and cache the result as PNG.

    Undistortion and resizing happen in a single bilinear resample, so the cached
    images share one pinhole camera matrix for every view. That lets the plane
    sweep assume a single ``K``.

    ``view.intrinsics`` is rewritten to describe the cached image exactly.

    Returns:
        List of cached image paths, in the same order as ``views``.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for view in views:
        dest = out_dir / f"{view.index:05d}.png"
        written.append(dest)
        if dest.exists() and not overwrite:
            # Intrinsics were restored by the caller from the cached geometry pass.
            continue
        native: Intrinsics = view._native_intrinsics  # type: ignore[attr-defined]
        dist = view._distortion  # type: ignore[attr-defined]
        image = cv2.imread(str(view.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise DataError(f"Failed to read image for {view.frame_id}: {view.image_path}")

        # Intrinsics describing the image actually on disk, then the target scale.
        actual = native_scaled(native, image.shape)
        scale = min(1.0, float(target_width) / float(actual.width))
        target = actual.scaled(scale)

        mapx, mapy = _build_remap(actual, dist, target)
        rectified = cv2.remap(
            image,
            mapx,
            mapy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        cv2.imwrite(str(dest), rectified)
        view.intrinsics = target
    return written


def native_scaled(native: Intrinsics, shape: tuple[int, ...]) -> Intrinsics:
    """Return ``native`` resized to match an image of ``shape`` (H, W, ...)."""
    h, w = int(shape[0]), int(shape[1])
    if (w, h) == (native.width, native.height):
        return native
    return Intrinsics(
        fx=native.fx * w / native.width,
        fy=native.fy * h / native.height,
        cx=native.cx * w / native.width,
        cy=native.cy * h / native.height,
        width=w,
        height=h,
    )
