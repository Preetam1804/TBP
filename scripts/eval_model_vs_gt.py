"""Score a reconstructed point cloud against the synthetic city's true surface cloud.

Uses the run's ate_report.json similarity transform (camera positions -> GT frame).
Metrics are in GT metres and use one fixed trajectory-derived alignment, so model
geometry is evaluated without fitting the point cloud itself to the answer:

* median / p90 distance from model points to the nearest GT surface point
* gt_coverage_pct: share of GT points within `--cov` m of any model point (completeness)

Only use clouds in the COLMAP/SfM world frame, such as model.ply and cloud_dense.ply.
The map-generation outputs are rotated to an upright display frame and need that
frame transform applied before this comparison.

Usage:
    uv run python scripts/eval_model_vs_gt.py RUN_DIR GT_PLY [model.ply|cloud_dense.ply]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def evaluate(run_dir: Path, gt_ply: Path, cloud_name: str = "model.ply", cov: float = 3.0) -> dict:
    ate = json.loads((run_dir / "ate_report.json").read_text())
    tr = ate["alignment_transform"]
    s, R, t = float(tr["scale"]), np.array(tr["rotation_matrix"]), np.array(tr["translation_m"])
    est = np.asarray(o3d.io.read_point_cloud(str(run_dir / cloud_name)).points)
    gt = np.asarray(o3d.io.read_point_cloud(str(gt_ply)).points)
    if len(est) == 0 or len(gt) == 0:
        raise ValueError(f"Empty point cloud: estimate={len(est)}, ground_truth={len(gt)}")
    if len(est) > 150_000:
        est = est[np.random.default_rng(0).choice(len(est), 150_000, replace=False)]
    base = (s * (R @ est.T)).T + t
    p = base
    d, _ = cKDTree(gt).query(p, k=1, workers=-1)
    gt_sub = gt[np.random.default_rng(1).choice(len(gt), min(40_000, len(gt)), replace=False)]
    dc, _ = cKDTree(p).query(gt_sub, k=1, workers=-1)
    return {
        "cloud": cloud_name,
        "points": len(est),
        "model_to_gt_median_m": round(float(np.median(d)), 2),
        "p90_m": round(float(np.percentile(d, 90)), 2),
        "gt_to_model_median_m": round(float(np.median(dc)), 2),
        "gt_to_model_p90_m": round(float(np.percentile(dc, 90)), 2),
        "gt_coverage_pct": round(100.0 * float((dc < cov).mean()), 1),
    }


if __name__ == "__main__":
    run = Path(sys.argv[1])
    gt = Path(sys.argv[2])
    for name in sys.argv[3:] or ["model.ply"]:
        print(evaluate(run, gt, name))
