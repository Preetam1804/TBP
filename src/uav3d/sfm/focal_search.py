"""Focal-length selection for forward-flight video with unknown intrinsics.

Forward-flight drone video has almost no camera rotation, which makes the focal
length (nearly) unobservable: bundle adjustment reprojection error is identical
for any focal over a wide range (measured on the Taj Mahal clip: 0.9021 px for
f = 400..3500). Two failure modes follow:

* With an arbitrary focal prior COLMAP's incremental mapper can fall into a
  *bent* local solution (higher reprojection error, camera pitch drifting by
  >50 deg on a constant-pitch flight, ill-conditioned BA that takes ~4x longer)
  while a different focal gives a straight, better-fitting solution.
* Refining the focal with such data runs away (f -> 13 000 px).

This module therefore (1) picks the focal prior whose trial reconstruction on a
small, evenly spaced subset of keyframes fits best, and (2) tests whether the
focal is observable at all; if not, the caller should keep it fixed instead of
refining it.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pycolmap

from uav3d.config import SfmConfig
from uav3d.sfm.native_log import NativeStderrCapture

logger = logging.getLogger("uav3d.sfm.focal_search")


@dataclass
class FocalCandidate:
    """Outcome of one trial reconstruction at a fixed focal length."""

    focal_px: float
    registered: int
    mean_reproj_px: float | None
    num_points: int

    def as_dict(self) -> dict:
        return {
            "focal_px": round(self.focal_px, 2),
            "registered": self.registered,
            "mean_reproj_px": None if self.mean_reproj_px is None else round(self.mean_reproj_px, 4),
            "num_points": self.num_points,
        }


@dataclass
class FocalSearchResult:
    """Selected focal length and the evidence behind the choice."""

    prior_focal_px: float
    selected_focal_px: float
    focal_observable: bool
    candidates: list[FocalCandidate] = field(default_factory=list)
    observability_reproj_increase: dict[str, float] = field(default_factory=dict)

    def note(self) -> str:
        if self.focal_observable:
            return (
                f"focal length selected by trial reconstruction ({self.selected_focal_px:.0f} px, prior "
                f"{self.prior_focal_px:.0f} px) and refined during bundle adjustment; the data constrains it"
            )
        return (
            f"focal length fixed at {self.selected_focal_px:.0f} px (prior {self.prior_focal_px:.0f} px): "
            "forward-flight video has too little camera rotation to observe it, so reprojection error is "
            "insensitive to focal length. Depth-axis proportions (e.g. ground-plane tilt) depend on this "
            "assumed focal; a measured focal length would remove the ambiguity"
        )

    def as_dict(self) -> dict:
        return {
            "prior_focal_px": round(self.prior_focal_px, 2),
            "selected_focal_px": round(self.selected_focal_px, 2),
            "focal_observable": self.focal_observable,
            "candidates": [c.as_dict() for c in self.candidates],
            "observability_reproj_increase": self.observability_reproj_increase,
        }


def _subset_names(image_files: list[str], max_images: int) -> list[str]:
    n = len(image_files)
    if n <= max_images:
        return list(image_files)
    idx = np.unique(np.linspace(0, n - 1, max_images).round().astype(int))
    return [image_files[i] for i in idx]


def _set_db_focal(database_path: Path, focal: float) -> None:
    db = pycolmap.Database()
    db.open(str(database_path))
    try:
        for cam in db.read_all_cameras():
            cam.focal_length = focal
            db.update_camera(cam)
    finally:
        db.close()


def _trial_mapping(
    base_db: Path,
    image_dir: Path,
    names: list[str],
    focal: float,
    cfg: SfmConfig,
    workdir: Path,
    native_log_path: str | Path,
) -> tuple[FocalCandidate, Path | None]:
    """Map the subset at a fixed focal; return its candidate record and model dir."""
    vdir = workdir / f"f{round(focal)}"
    vdir.mkdir(parents=True, exist_ok=True)
    db_path = vdir / "database.db"
    shutil.copy(base_db, db_path)
    _set_db_focal(db_path, focal)

    opts = pycolmap.IncrementalPipelineOptions()
    opts.image_names = names
    opts.multiple_models = False
    opts.init_num_trials = cfg.init_num_trials
    opts.min_model_size = cfg.min_model_size
    opts.ignore_watermarks = cfg.ignore_watermarks
    opts.mapper.init_min_tri_angle = cfg.init_min_tri_angle
    opts.mapper.init_min_num_inliers = cfg.init_min_inliers
    opts.mapper.abs_pose_min_num_inliers = cfg.abs_pose_min_inliers
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    opts.mapper.abs_pose_refine_focal_length = False
    opts.mapper.abs_pose_refine_extra_params = False

    sparse = vdir / "sparse"
    sparse.mkdir(exist_ok=True)
    try:
        with NativeStderrCapture(native_log_path):
            recs = pycolmap.incremental_mapping(str(db_path), str(image_dir), str(sparse), opts)
    except Exception as exc:  # noqa: BLE001 - trial failures must never abort the real run
        logger.warning(f"  focal trial f={focal:.0f} failed: {exc}")
        return FocalCandidate(focal, 0, None, 0), None
    if not recs:
        return FocalCandidate(focal, 0, None, 0), None

    best_idx = max(recs, key=lambda k: recs[k].num_reg_images())
    rec = recs[best_idx]
    cand = FocalCandidate(
        focal_px=focal,
        registered=rec.num_reg_images(),
        mean_reproj_px=float(rec.compute_mean_reprojection_error()),
        num_points=rec.num_points3D(),
    )
    return cand, sparse / str(best_idx)


def continue_to_focal(rec: pycolmap.Reconstruction, focal: float, rounds: int = 3) -> float:
    """Warm-started fixed-intrinsics bundle adjustment at ``focal``; returns the mean reprojection error.

    When the focal length is unobservable (forward flight) the reprojection error is the same for
    every focal, so a model found at one focal can be moved to another without loss of fit. This
    is how a model mapped at a *surrogate* focal (one where COLMAP's mapper finds the good basin)
    is carried to the focal measured from vanishing points.
    """
    for cam in rec.cameras.values():
        cam.focal_length = focal
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = False
    opts.refine_principal_point = False
    opts.refine_extra_params = False
    for _ in range(rounds):
        pycolmap.bundle_adjustment(rec, opts)
    return float(rec.compute_mean_reprojection_error())


def _ba_reproj_at_focal(model_dir: Path, focal: float, rounds: int = 3) -> float:
    return continue_to_focal(pycolmap.Reconstruction(str(model_dir)), focal, rounds)


def search_focal(
    database_path: Path,
    image_dir: Path,
    image_files: list[str],
    prior_focal_px: float,
    cfg: SfmConfig,
    native_log_path: str | Path,
    probe_observability: bool = True,
) -> FocalSearchResult:
    """Choose a focal length for an exhaustively matched keyframe set.

    With ``probe_observability=False`` (focal already known from vanishing points) the returned
    focal is only the *surrogate* at which the mapper should run before ``continue_to_focal``.

    Candidates are tried from the prior downward (factors in ``cfg.focal_search_factors``).
    The search stops once a candidate improves on an earlier one and the following
    candidate no longer improves it (two consecutive candidates in the same regime).
    Among candidates whose fit is within ``cfg.focal_search_tolerance`` of the best
    (relative reprojection error, same registration count) the one nearest the prior wins.
    """
    names = _subset_names(image_files, cfg.focal_search_max_images)
    tol = cfg.focal_search_tolerance
    logger.info(
        f"Focal search: {len(names)}-image trial subset, prior f={prior_focal_px:.0f} px, "
        f"factors={list(cfg.focal_search_factors)}"
    )

    def key(c: FocalCandidate) -> tuple[int, float]:
        return (-c.registered, c.mean_reproj_px if c.mean_reproj_px is not None else float("inf"))

    candidates: list[FocalCandidate] = []
    models: dict[float, Path | None] = {}
    with tempfile.TemporaryDirectory(prefix="uav3d_focal_", ignore_cleanup_errors=True) as tmp:
        workdir = Path(tmp)
        improved = False
        for factor in cfg.focal_search_factors:
            focal = prior_focal_px * float(factor)
            cand, model_dir = _trial_mapping(database_path, image_dir, names, focal, cfg, workdir, native_log_path)
            logger.info(
                f"  f={focal:7.1f} px -> registered={cand.registered}/{len(names)} "
                f"reproj={'n/a' if cand.mean_reproj_px is None else f'{cand.mean_reproj_px:.3f} px'} "
                f"points={cand.num_points}"
            )
            prev_best = min(candidates, key=key) if candidates else None
            candidates.append(cand)
            models[focal] = model_dir
            if prev_best is not None:
                if key(cand) < key(prev_best) and not _within(cand, prev_best, tol):
                    improved = True
                elif improved:
                    break  # same regime twice after an improvement: nothing better below

        valid = [c for c in candidates if c.mean_reproj_px is not None]
        if not valid:
            logger.warning("Focal search: no trial reconstruction succeeded; keeping the prior focal.")
            return FocalSearchResult(prior_focal_px, prior_focal_px, True, candidates)

        best = min(valid, key=key)
        near_best = [c for c in valid if c.registered == best.registered and _within(c, best, tol)]
        chosen = min(near_best, key=lambda c: abs(c.focal_px - prior_focal_px))
        prior_cand = next((c for c in candidates if c.focal_px == prior_focal_px), None)
        if (
            prior_cand is not None
            and chosen is not prior_cand
            and prior_cand.mean_reproj_px
            and prior_cand.registered == best.registered
            and (prior_cand.mean_reproj_px - best.mean_reproj_px) < cfg.focal_search_min_gain * prior_cand.mean_reproj_px
        ):
            logger.info(
                f"Focal search: best candidate improves on the prior by less than "
                f"{cfg.focal_search_min_gain:.0%}; keeping the prior f={prior_focal_px:.1f} px"
            )
            chosen = prior_cand
        logger.info(f"Focal search: selected f={chosen.focal_px:.1f} px (best reproj {best.mean_reproj_px:.3f} px)")

        # Observability: does moving the focal +/-25% (warm-started BA) change the fit?
        increases: dict[str, float] = {}
        chosen_model = models.get(chosen.focal_px)
        observable = True
        if probe_observability and chosen_model is not None and chosen.mean_reproj_px:
            for label, mult in (("x0.8", 0.8), ("x1.25", 1.25)):
                try:
                    err = _ba_reproj_at_focal(chosen_model, chosen.focal_px * mult)
                    increases[label] = round((err - chosen.mean_reproj_px) / chosen.mean_reproj_px, 4)
                except Exception as exc:  # noqa: BLE001 - observability probe is best-effort
                    logger.warning(f"  observability BA at {label} failed: {exc}")
            if len(increases) == 2:
                observable = max(increases.values()) > cfg.focal_observability_threshold
            logger.info(
                f"Focal observability: reproj change for focal x0.8/x1.25 = {increases} "
                f"(threshold {cfg.focal_observability_threshold:.0%}) -> "
                f"{'observable, focal will be refined' if observable else 'NOT observable, focal stays fixed'}"
            )

    return FocalSearchResult(
        prior_focal_px=prior_focal_px,
        selected_focal_px=chosen.focal_px,
        focal_observable=observable,
        candidates=candidates,
        observability_reproj_increase=increases,
    )


def _within(a: FocalCandidate, b: FocalCandidate, tol: float) -> bool:
    if a.mean_reproj_px is None or b.mean_reproj_px is None:
        return False
    return a.registered == b.registered and abs(a.mean_reproj_px - b.mean_reproj_px) <= tol * b.mean_reproj_px
