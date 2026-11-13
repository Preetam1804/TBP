"""CPU multi-view-stereo dense reconstruction ("SP-3").

Consumes the outputs of SP-1 (``poses.json`` + ``keyframes.json``) and produces
per-view depth maps, a fused point cloud and a mesh, all without CUDA.

The stage deliberately does **not** use a monocular depth prior. Depth is recovered
from multi-view photometric consistency (plane-sweep stereo), which is the only
signal in a posed image sequence that carries real geometric information.

Exports are resolved lazily so that importing :mod:`uav3d.dense.config` (for
example from the global configuration loader) does not pull in OpenCV, SciPy and
pycolmap.
"""

from typing import Any

__all__ = [
    "CameraView",
    "DenseConfig",
    "DenseRunResult",
    "Intrinsics",
    "run_dense",
    "undistort_run_images",
]

_LAZY: dict[str, str] = {
    "CameraView": "uav3d.dense.cameras",
    "Intrinsics": "uav3d.dense.cameras",
    "undistort_run_images": "uav3d.dense.cameras",
    "DenseConfig": "uav3d.dense.config",
    "DenseRunResult": "uav3d.dense.pipeline",
    "run_dense": "uav3d.dense.pipeline",
}


def __getattr__(name: str) -> Any:
    """Resolve public names on first access (PEP 562)."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
