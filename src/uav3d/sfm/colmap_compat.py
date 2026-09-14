"""Compatibility shim and cross-platform utilities for pycolmap."""

import logging
import sys
from typing import Any

import numpy as np

logger = logging.getLogger("uav3d.sfm.colmap_compat")


def quaternion_to_rotation_matrix(q: list[float] | np.ndarray) -> np.ndarray:
    """Convert a unit quaternion in (qw, qx, qy, qz) wxyz order to a 3x3 rotation matrix.

    Args:
        q: 4-element sequence [qw, qx, qy, qz].

    Returns:
        3x3 rotation matrix R.
    """
    qw, qx, qy, qz = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # Normalize quaternion defensively
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm > 0.0:
        qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def compute_camera_center(qvec: list[float] | np.ndarray, tvec: list[float] | np.ndarray) -> list[float]:
    """Compute camera center in world coordinates from cam_from_world (R, t).

    COLMAP uses cam_from_world convention: x_cam = R @ X_world + t.
    Therefore, the camera optical center in world coordinates is:
        C = -R^T @ t.

    Args:
        qvec: Orientation quaternion in [qw, qx, qy, qz] order.
        tvec: Translation vector [tx, ty, tz].

    Returns:
        Camera center [X, Y, Z] in world coordinate frame.
    """
    R = quaternion_to_rotation_matrix(qvec)
    t = np.array(tvec, dtype=np.float64).reshape(3, 1)
    center = -R.T @ t
    return [float(center[0, 0]), float(center[1, 0]), float(center[2, 0])]


def is_image_registered(image: Any) -> bool:
    """Check if an image was registered in the reconstruction."""
    if hasattr(image, "has_pose"):
        has_pose = bool(image.has_pose() if callable(image.has_pose) else image.has_pose)
    elif hasattr(image, "is_registered"):
        has_pose = bool(image.is_registered() if callable(image.is_registered) else image.is_registered)
    else:
        has_pose = hasattr(image, "cam_from_world") or hasattr(image, "qvec")

    # In addition to pose, check if the image has any 3D triangulated points
    if hasattr(image, "num_points3D"):
        num_pts = image.num_points3D() if callable(image.num_points3D) else image.num_points3D
        return has_pose and num_pts > 0
    if hasattr(image, "has_registered_points"):
        has_reg = image.has_registered_points() if callable(image.has_registered_points) else image.has_registered_points
        return has_pose and bool(has_reg)

    return has_pose


def get_pose_components(image: Any) -> tuple[list[float], list[float], list[float]]:
    """Extract (qvec_wxyz, tvec_xyz, camera_center_world) from a pycolmap.Image.

    Handles both modern pycolmap (>= 3.12, image.cam_from_world) and legacy
    pycolmap (image.qvec, image.tvec) seamlessly.

    In pycolmap >= 3.12, `rotation.quat` is in Eigen [x, y, z, w] format.
    This function explicitly converts it to [qw, qx, qy, qz] (wxyz) order.

    Args:
        image: pycolmap.Image object.

    Returns:
        tuple of (qvec_wxyz, tvec_xyz, camera_center_world).
    """
    if hasattr(image, "cam_from_world"):
        cfw = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        # pycolmap 3.12 Rotation3d.quat docstring: "Quaternion in [x,y,z,w] format"
        quat_xyzw = cfw.rotation.quat
        qvec_wxyz = [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]
        trans = cfw.translation
        tvec_xyz = [float(trans[0]), float(trans[1]), float(trans[2])]
        R = cfw.rotation.matrix()
        t = np.array(tvec_xyz, dtype=np.float64).reshape(3, 1)
        center = -R.T @ t
        camera_center_world = [float(center[0, 0]), float(center[1, 0]), float(center[2, 0])]
        return qvec_wxyz, tvec_xyz, camera_center_world

    # Fallback to legacy pycolmap attributes (legacy qvec was [qw, qx, qy, qz])
    if hasattr(image, "qvec") and hasattr(image, "tvec"):
        qvec = image.qvec() if callable(image.qvec) else image.qvec
        tvec = image.tvec() if callable(image.tvec) else image.tvec
        qvec_wxyz = [float(qvec[0]), float(qvec[1]), float(qvec[2]), float(qvec[3])]
        tvec_xyz = [float(tvec[0]), float(tvec[1]), float(tvec[2])]
        camera_center_world = compute_camera_center(qvec_wxyz, tvec_xyz)
        return qvec_wxyz, tvec_xyz, camera_center_world

    raise AttributeError(
        f"Image object {image} does not have recognized pose attributes (cam_from_world or qvec/tvec)"
    )


def get_peak_rss_mb() -> float:
    """Return process peak Resident Set Size (RSS) in megabytes across platforms."""
    # 1. Unix / macOS via resource module
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # On Linux ru_maxrss is in kilobytes; on macOS in bytes
        if sys.platform == "darwin":
            return float(usage.ru_maxrss) / (1024.0 * 1024.0)
        return float(usage.ru_maxrss) / 1024.0
    except (ImportError, AttributeError):
        pass

    # 2. Windows via ctypes and psapi OpenProcess
    if sys.platform == "win32":
        try:
            import ctypes
            import os
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(pmc)
            # PROCESS_QUERY_INFORMATION (0x0400) | PROCESS_VM_READ (0x0010) = 0x0410
            handle = ctypes.windll.kernel32.OpenProcess(0x0410, False, os.getpid())
            if handle:
                try:
                    if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                        return float(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0)
                finally:
                    ctypes.windll.kernel32.CloseHandle(handle)
        except Exception as exc:
            logger.debug(f"Could not query Windows peak RSS: {exc}")

    return 0.0
