"""Regenerate dense cloud + TSDF mesh + full 3D map for an existing run.

Usage:
    uv run python scripts/regen_dense_map.py <run_dir>
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from uav3d.modeling.completion import build_full_3d_map
from uav3d.modeling.dense import generate_dense_and_mesh_tsdf


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) != 2:
        print("Usage: regen_dense_map.py <run_dir>")
        return 1
    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}")
        return 1

    t0 = time.perf_counter()
    mesh_ply, mesh_dec, cloud_dense = generate_dense_and_mesh_tsdf(run_dir)
    if mesh_ply is None:
        print("TSDF fusion produced no mesh; see logs above.")
        return 2
    print(
        f"Dense stage done in {time.perf_counter() - t0:.1f}s: "
        f"{mesh_ply.name}, {mesh_dec.name}, {cloud_dense.name}"
    )

    t1 = time.perf_counter()
    map_ply, map_glb, ortho, dsm = build_full_3d_map(
        cloud_dense, run_dir, existing_mesh=mesh_ply
    )
    print(
        f"Map stage done in {time.perf_counter() - t1:.1f}s: "
        f"{map_ply.name}, {map_glb.name}, {ortho.name}, {dsm.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
