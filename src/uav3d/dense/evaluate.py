"""Evaluation of a dense reconstruction against a reference point cloud.

Two properties matter and they must not be conflated:

* **accuracy** — are the reconstructed points where the surface really is?
* **completeness** — what fraction of the *photographed* surface did we recover?

Completeness computed over an entire ground-truth city (including streets the
drone never looked at) measures the flight plan, not the algorithm. This module
therefore reports completeness over the *observable* subset: ground-truth points
that fall inside at least one camera frustum.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from uav3d.dense.cameras import CameraView
from uav3d.exceptions import DataError

logger = logging.getLogger("uav3d.dense.evaluate")

DEFAULT_THRESHOLDS = (0.5, 1.0, 2.0)


def load_ply_points(path: str | Path) -> np.ndarray:
    """Read vertex positions from an ASCII or binary little-endian PLY file."""
    src = Path(path)
    if not src.exists():
        raise DataError(f"Point cloud not found: {src}")
    with open(src, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise DataError(f"{src} is not a PLY file")
        fmt = "ascii"
        count = 0
        props: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            line = fh.readline()
            if not line:
                raise DataError(f"{src}: unexpected end of header")
            token = line.strip()
            if token.startswith(b"format"):
                fmt = token.split()[1].decode("ascii")
            elif token.startswith(b"element vertex"):
                count = int(token.split()[2])
                in_vertex = True
            elif token.startswith(b"element") and not token.startswith(b"element vertex"):
                in_vertex = False
            elif token.startswith(b"property") and in_vertex:
                parts = token.split()
                props.append((parts[1].decode("ascii"), parts[2].decode("ascii")))
            elif token == b"end_header":
                break

        coord_idx = [i for i, (_, name) in enumerate(props) if name in ("x", "y", "z")]
        if len(coord_idx) < 3:
            raise DataError(f"{src}: PLY has no x/y/z vertex properties")

        if fmt == "ascii":
            data = np.loadtxt(fh, max_rows=count, ndmin=2)
            return np.asarray(data[:, coord_idx[:3]], dtype=np.float64)

        type_map = {
            "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
            "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
            "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
            "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
        }
        dtype = np.dtype([(name, "<" + type_map.get(kind, "f4")) for kind, name in props])
        raw = np.frombuffer(fh.read(dtype.itemsize * count), dtype=dtype, count=count)
        return np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float64)


def observable_mask(gt_points: np.ndarray, views: list[CameraView]) -> np.ndarray:
    """True where a ground-truth point is inside at least one camera frustum.

    This is the correction that makes completeness meaningful for a corridor
    flight: surface outside every frustum was never photographed and no
    reconstruction method can recover it.
    """
    mask = np.zeros(gt_points.shape[0], dtype=bool)
    for view in views:
        uv, z = view.project(gt_points)
        inside = (
            (z > 0)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < view.intrinsics.width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < view.intrinsics.height)
        )
        mask |= inside
    return mask


def _nearest_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if query.shape[0] == 0 or reference.shape[0] == 0:
        return np.full(query.shape[0], np.inf)
    tree = cKDTree(reference)
    distances, _ = tree.query(query, k=1)
    return distances


def cloud_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    views: list[CameraView] | None = None,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    visible: np.ndarray | None = None,
    distance_scale: float = 1.0,
) -> dict[str, float]:
    """Compute accuracy and (observable) completeness for a reconstructed cloud.

    ``distance_scale`` converts internal (SfM unit) distances into metres, so
    thresholds can always be expressed in metres.
    """
    if predicted.shape[0] == 0:
        raise DataError("Predicted point cloud is empty; nothing to evaluate")

    if visible is None:
        visible = observable_mask(ground_truth, views) if views else np.ones(ground_truth.shape[0], bool)
    gt_visible = ground_truth[visible]

    acc = _nearest_distances(predicted, ground_truth) * distance_scale
    comp = (
        _nearest_distances(gt_visible, predicted) * distance_scale
        if gt_visible.shape[0]
        else np.zeros(0)
    )

    # Ground-truth sampling bias: a point cloud can only be as accurate as its
    # own spacing. Reported so the reader can judge small accuracy numbers.
    gt_spacing = estimate_point_spacing(gt_visible) * distance_scale if gt_visible.shape[0] > 1 else 0.0

    metrics: dict[str, float] = {
        "num_predicted": float(predicted.shape[0]),
        "num_ground_truth": float(ground_truth.shape[0]),
        "num_ground_truth_observable": float(gt_visible.shape[0]),
        "observable_fraction": float(gt_visible.shape[0] / max(ground_truth.shape[0], 1)),
        "accuracy_median": float(np.median(acc)),
        "accuracy_mean": float(np.mean(acc)),
        "accuracy_p90": float(np.percentile(acc, 90)),
        "ground_truth_spacing": float(gt_spacing),
    }
    for tau in thresholds:
        metrics[f"precision_at_{tau}m"] = float(np.mean(acc <= tau))
        metrics[f"completeness_at_{tau}m"] = float(np.mean(comp <= tau)) if comp.size else 0.0
        p = metrics[f"precision_at_{tau}m"]
        c = metrics[f"completeness_at_{tau}m"]
        metrics[f"fscore_at_{tau}m"] = float(2 * p * c / (p + c)) if (p + c) > 0 else 0.0
    return metrics


def estimate_point_spacing(points: np.ndarray, sample: int = 20000) -> float:
    """Median nearest-neighbour distance, used to report cloud resolution."""
    n = min(sample, points.shape[0])
    if n < 2:
        return 0.0
    tree = cKDTree(points[:n])
    distances, _ = tree.query(points[:n], k=2)
    return float(np.median(distances[:, 1]))
