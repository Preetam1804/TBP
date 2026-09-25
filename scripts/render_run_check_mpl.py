"""Render verification viewpoints for a reconstruction run (GL-free fallback).

Uses matplotlib Poly3DCollection on a decimated mesh copy, so it works where
Open3D's offscreen renderer is unavailable (e.g. Open3D 0.19 on Windows).

Usage:
    uv run python scripts/render_run_check_mpl.py <run_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def main() -> int:
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        candidates = sorted(Path("data/outputs").glob("run_*"))
        if not candidates:
            print("No runs found under data/outputs")
            return 1
        run_dir = candidates[-1]

    mesh_ply = run_dir / "mesh.ply"
    if not mesh_ply.exists():
        print(f"Missing {mesh_ply}")
        return 1

    out_dir = run_dir / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(mesh_ply))
    target = 60_000
    if len(mesh.triangles) > target:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
    mesh.compute_vertex_normals()

    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    cols = np.asarray(mesh.vertex_colors)
    if len(cols) == 0:
        cols = np.full((len(v), 3), 0.7)

    tri_pts = v[f]  # (T, 3, 3)
    face_cols = cols[f].mean(axis=1)  # (T, 3)

    center = v.mean(axis=0)
    extent = v.max(axis=0) - v.min(axis=0)
    span = float(np.linalg.norm(extent))
    print(f"{len(v):,} verts, {len(f):,} tris, center={np.round(center, 1)}, extent={np.round(extent, 1)}")

    views = [
        ("check_oblique", (0.9, -1.2, 0.9)),
        ("check_topdown", (0.0, 0.0, 2.0)),
        ("check_axis_side", (0.0, -1.8, 0.35)),
        ("check_monument", (0.25, -0.4, 0.25)),
    ]
    z_up = extent.argmax() != 2  # keep Z-up on screen

    for name, k in views:
        eye = center + extent * np.array(k) + np.array([0.0, 0.0, span * 0.02])
        fig = plt.figure(figsize=(12, 7), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        pc = Poly3DCollection(tri_pts, facecolors=np.clip(face_cols, 0, 1), edgecolors="none")
        ax.add_collection3d(pc)

        r = span * 0.55
        ax.set_xlim(center[0] - r, center[0] + r)
        ax.set_ylim(center[1] - r, center[1] + r)
        ax.set_zlim(center[2] - r, center[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=25 if k[2] < 1.0 else 89, azim=-60)
        ax.set_axis_off()
        ax.set_title(name, fontsize=10)
        fig.tight_layout()
        out = out_dir / f"render_{name}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
