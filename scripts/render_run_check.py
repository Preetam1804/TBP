"""Render verification viewpoints for the latest reconstruction run.

Usage:
    uv run python scripts/render_run_check.py [run_dir]

Defaults to the newest run under data/outputs. Renders 4 viewpoints of
map_3d.obj to run_dir/debug/ for visual quality checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> int:
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        candidates = sorted((Path("data/outputs")).glob("run_*"))
        if not candidates:
            print("No runs found under data/outputs")
            return 1
        run_dir = candidates[-1]

    model_obj = run_dir / "map_3d.obj"
    if not model_obj.exists():
        print(f"Missing {model_obj}")
        return 1

    out_dir = run_dir / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_model(str(model_obj))
    print(f"Loaded {len(mesh.meshes)} submesh(es) from {model_obj}")
    for m in mesh.materials:
        m.shader = "defaultUnlit"

    renderer = o3d.visualization.rendering.OffscreenRenderer(1600, 900)
    scene = renderer.scene
    scene.set_background([0.05, 0.07, 0.10, 1.0])
    scene.add_model("map_3d", mesh)

    bbox = scene.bounding_box
    center = bbox.get_center()
    extent = bbox.get_extent()
    print(f"BBox center: {center}, extent: {extent}")

    views = [
        {"name": "check_oblique", "eye": center + np.array([extent[0] * 0.9, -extent[1] * 1.2, extent[2] * 1.4]), "lookat": center, "up": np.array([0.0, 0.0, 1.0])},
        {"name": "check_topdown", "eye": center + np.array([0.0, 0.0, extent[2] * 1.8 + 15.0]), "lookat": center, "up": np.array([0.0, 1.0, 0.0])},
        {"name": "check_axis_side", "eye": center + np.array([0.0, -extent[1] * 1.6, extent[2] * 0.35]), "lookat": center, "up": np.array([0.0, 0.0, 1.0])},
        {"name": "check_monument", "eye": center + np.array([extent[0] * 0.25, -extent[1] * 0.35, extent[2] * 0.25]), "lookat": center + np.array([0.0, 0.0, extent[2] * 0.05]), "up": np.array([0.0, 0.0, 1.0])},
    ]

    # Warm-up: Filament uploads the model lazily; render a throwaway frame and
    # wait so the first real view doesn't capture an empty scene.
    renderer.render_to_image()
    import time

    time.sleep(1.0)

    for view in views:
        renderer.setup_camera(55.0, view["lookat"], view["eye"], view["up"])
        img = renderer.render_to_image()
        out = out_dir / f"render_{view['name']}.png"
        o3d.io.write_image(str(out), img)
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
