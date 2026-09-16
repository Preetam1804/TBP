"""SfM runner executing pycolmap incremental pipeline with locked intrinsics."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap

from uav3d.calibration import (
    camera_params_expected_count,
    check_intrinsics_sanity,
    horizontal_fov_deg,
)
from uav3d.config import SfmConfig
from uav3d.exceptions import Uav3dError
from uav3d.models.frames import CameraCalibration
from uav3d.sfm.colmap_compat import get_peak_rss_mb

logger = logging.getLogger("uav3d.sfm.runner")


@dataclass
class SfmStageProfile:
    """Execution timing and peak memory profiling for a stage."""

    stage_name: str
    wall_time_s: float
    peak_rss_mb: float


@dataclass
class SfmRunResult:
    """Output summary of an SfM execution run."""

    reconstruction: pycolmap.Reconstruction
    database_path: Path
    sparse_dir: Path
    num_models: int
    num_reg_images: int
    total_images: int
    registration_rate: float
    mean_reprojection_error_px: float
    profiles: list[SfmStageProfile]
    total_wall_time_s: float
    peak_rss_mb: float
    intrinsics_refined: bool
    initial_params: list[float]
    final_params: list[float]
    final_fov_deg: float | None
    intrinsics_violations: list[str]
    refinement_rejected: bool


class SfmRunner:
    """Executes incremental visual Structure-from-Motion using pycolmap."""

    def __init__(self, config: SfmConfig | None = None) -> None:
        self.config = config or SfmConfig()

    @staticmethod
    def _first_camera(reconstruction: pycolmap.Reconstruction) -> Any:
        """Return the first camera of a reconstruction."""
        return next(iter(reconstruction.cameras.values()))

    @staticmethod
    def _camera_model_name(reconstruction: pycolmap.Reconstruction, fallback: str) -> str:
        """Return the COLMAP model name of the reconstruction's first camera."""
        camera = next(iter(reconstruction.cameras.values()), None)
        if camera is None:
            return fallback
        return str(getattr(camera, "model", fallback)).replace("CameraModelId.", "")

    def _run_mapping(
        self,
        database_path: Path,
        image_path: Path,
        output_dir: Path,
        refine: bool,
    ) -> dict[int, Any]:
        """Run COLMAP incremental mapping with intrinsics either refined or locked.

        Args:
            database_path: COLMAP database containing extracted features and matches.
            image_path: Directory holding the images referenced by the database.
            output_dir: Directory to write the sparse reconstruction into.
            refine: When False, bundle adjustment and absolute pose refinement are
                forbidden from touching the intrinsics.

        Returns:
            Mapping of model index to pycolmap.Reconstruction.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        pipeline_opts = pycolmap.IncrementalPipelineOptions()
        if not refine:
            pipeline_opts.ba_refine_focal_length = False
            pipeline_opts.ba_refine_principal_point = False
            pipeline_opts.ba_refine_extra_params = False
            pipeline_opts.mapper.abs_pose_refine_focal_length = False
            pipeline_opts.mapper.abs_pose_refine_extra_params = False

        pipeline_opts.mapper.init_min_tri_angle = self.config.init_min_tri_angle
        pipeline_opts.mapper.init_min_num_inliers = self.config.init_min_inliers
        pipeline_opts.mapper.abs_pose_min_num_inliers = self.config.abs_pose_min_inliers
        pipeline_opts.min_model_size = self.config.min_model_size
        pipeline_opts.ignore_watermarks = self.config.ignore_watermarks

        logger.info(f"Incremental mapping (refine_intrinsics={refine}) -> {output_dir}")
        try:
            reconstructions = pycolmap.incremental_mapping(
                database_path=str(database_path),
                image_path=str(image_path),
                output_path=str(output_dir),
                options=pipeline_opts,
            )
        except Exception as exc:
            raise Uav3dError(f"Incremental mapping failed with exception: {exc}") from exc
        return reconstructions or {}

    def run(
        self,
        images_dir: str | Path,
        output_dir: str | Path,
        calibration: CameraCalibration,
    ) -> SfmRunResult:
        """Run the complete incremental SfM pipeline.

        1. Feature extraction with locked camera parameters.
        2. Database camera record injection / update.
        3. Sequential feature matching.
        4. Incremental mapping with bundle adjustment refinement disabled.
        5. Largest model selection and camera intrinsic verification.

        Args:
            images_dir: Directory containing input keyframe images.
            output_dir: Run directory (e.g. data/outputs/run_<timestamp>).
            calibration: Camera calibration containing known intrinsics.

        Returns:
            SfmRunResult containing selected reconstruction and profiling metrics.
        """
        img_path = Path(images_dir).resolve()
        out_path = Path(output_dir).resolve()

        colmap_dir = out_path / "colmap"
        colmap_dir.mkdir(parents=True, exist_ok=True)
        database_path = colmap_dir / "database.db"
        if database_path.exists():
            database_path.unlink()

        sparse_dir = colmap_dir / "sparse"
        if sparse_dir.exists():
            import shutil
            shutil.rmtree(sparse_dir)
        sparse_dir.mkdir(parents=True, exist_ok=True)

        # Discover image files (deduplicated to prevent duplicate entries on Windows)
        image_files = sorted(
            {p.name for p in img_path.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        )
        total_images = len(image_files)
        if total_images == 0:
            raise Uav3dError(f"No image files found in {img_path}")

        logger.info(f"Starting SfM pipeline for {total_images} images in {img_path}")
        profiles: list[SfmStageProfile] = []
        overall_start = time.perf_counter()

        # -------------------------------------------------------------
        # Stage 1: Feature Extraction
        # -------------------------------------------------------------
        t0 = time.perf_counter()

        # The calibration owns the parameter vector, so its model is authoritative.
        # Using a different model from the config with those parameters produced
        # COLMAP's opaque "Invalid camera parameters" failure.
        camera_model = calibration.model.upper()
        if camera_model != self.config.camera_model.upper():
            logger.warning(
                f"Config camera_model='{self.config.camera_model}' does not match the calibration model "
                f"'{calibration.model}'; using the calibration model for extraction so the parameter "
                "vector stays valid."
            )

        expected_num_params = camera_params_expected_count(camera_model)
        if expected_num_params is not None and len(calibration.params) != expected_num_params:
            raise Uav3dError(
                f"Calibration for model {camera_model} provides {len(calibration.params)} parameters "
                f"but {expected_num_params} are required: {calibration.params}. "
                "Fix the calibration source before running SfM."
            )

        logger.info(f"Stage 1/4: Feature extraction (model={camera_model}, max_size={self.config.max_image_size})")

        sift_opts = pycolmap.SiftExtractionOptions()
        sift_opts.max_image_size = self.config.max_image_size

        reader_opts = pycolmap.ImageReaderOptions()
        reader_opts.camera_model = camera_model
        reader_opts.camera_params = ",".join(str(p) for p in calibration.params)

        camera_mode = pycolmap.CameraMode.SINGLE if self.config.single_camera else pycolmap.CameraMode.AUTO

        try:
            pycolmap.extract_features(
                database_path=str(database_path),
                image_path=str(img_path),
                image_names=image_files,
                camera_mode=camera_mode,
                camera_model=camera_model,
                reader_options=reader_opts,
                sift_options=sift_opts,
            )
        except Exception as exc:
            raise Uav3dError(f"Feature extraction failed: {exc}") from exc

        profiles.append(
            SfmStageProfile(
                stage_name="feature_extraction",
                wall_time_s=time.perf_counter() - t0,
                peak_rss_mb=get_peak_rss_mb(),
            )
        )

        # -------------------------------------------------------------
        # Stage 2: Database Camera Record Injection & Verification
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        logger.info("Stage 2/4: Injecting known camera calibration into database")
        db = pycolmap.Database()
        db.open(str(database_path))

        db_cams = db.read_all_cameras()
        if not db_cams:
            logger.info("No cameras found in DB; creating camera record with injected intrinsics")
            new_cam = pycolmap.Camera(
                model=camera_model,
                width=calibration.width,
                height=calibration.height,
                params=calibration.params,
            )
            db.write_camera(new_cam)
        else:
            for cam in db_cams:
                cam.params = np.array(calibration.params, dtype=np.float64)
                cam.width = calibration.width
                cam.height = calibration.height
                db.update_camera(cam)
            logger.info(f"Updated {len(db_cams)} database camera record(s) with injected intrinsics: {calibration.params}")

        db.close()
        profiles.append(
            SfmStageProfile(
                stage_name="camera_injection",
                wall_time_s=time.perf_counter() - t0,
                peak_rss_mb=get_peak_rss_mb(),
            )
        )

        # -------------------------------------------------------------
        # Stage 3: Sequential Feature Matching
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        logger.info(f"Stage 3/4: Sequential feature matching (overlap={self.config.seq_overlap})")

        seq_opts = pycolmap.SequentialMatchingOptions()
        seq_opts.overlap = self.config.seq_overlap

        tvg_opts = pycolmap.TwoViewGeometryOptions()
        if self.config.ignore_watermarks:
            tvg_opts.detect_watermark = False

        try:
            pycolmap.match_sequential(
                str(database_path),
                matching_options=seq_opts,
                verification_options=tvg_opts,
            )
        except Exception as exc:
            raise Uav3dError(f"Sequential feature matching failed: {exc}") from exc

        profiles.append(
            SfmStageProfile(
                stage_name="feature_matching",
                wall_time_s=time.perf_counter() - t0,
                peak_rss_mb=get_peak_rss_mb(),
            )
        )

        # -------------------------------------------------------------
        # Stage 4: Incremental Mapping
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        logger.info("Stage 4/4: Incremental reconstruction mapping")

        refinement_rejected = False
        intrinsics_violations: list[str] = []

        if self.config.refine_intrinsics:
            logger.info("Intrinsics refinement enabled for this run (bounded by the plausibility gate).")

        reconstructions = self._run_mapping(
            database_path=database_path,
            image_path=img_path,
            output_dir=sparse_dir,
            refine=self.config.refine_intrinsics,
        )
        mapping_time_s = time.perf_counter() - t0

        if not reconstructions:
            raise Uav3dError(
                f"COLMAP incremental mapping produced 0 valid reconstructions from {total_images} images. "
                "Check keyframe overlap, feature matches, or scene texture."
            )

        num_models = len(reconstructions)
        selected_rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())

        # -------------------------------------------------------------
        # Post-BA camera verification
        #
        # Locked mode: intrinsics must be bit-for-bit unchanged.
        # Refined mode: intrinsics are free, so they are instead gated on
        # physical plausibility (focal length -> FoV window, isotropy,
        # principal point location).
        #
        # If a refinement run fails the gate, the run is redone with the
        # supplied intrinsics locked. That is strictly better than exporting a
        # camera that cannot exist (e.g. fx=213 px, fy=745 px on a 1920 px
        # frame), and it is reported rather than hidden.
        # -------------------------------------------------------------
        expected_params = [float(p) for p in calibration.params]
        initial_params = list(expected_params)

        if self.config.refine_intrinsics:
            model_name = self._camera_model_name(selected_rec, camera_model)
            final_candidate = [float(p) for p in self._first_camera(selected_rec).params]
            intrinsics_violations = check_intrinsics_sanity(
                params=final_candidate,
                model=model_name,
                width=int(self._first_camera(selected_rec).width),
                height=int(self._first_camera(selected_rec).height),
                min_fov_deg=self.config.video_min_fov_deg,
                max_fov_deg=self.config.video_max_fov_deg,
            )
            if intrinsics_violations:
                message = (
                    "Refined camera intrinsics failed the plausibility gate: " + "; ".join(intrinsics_violations)
                )
                if self.config.allow_unbounded_intrinsics:
                    logger.warning(message + " (allow_unbounded_intrinsics=True: keeping refined values)")
                    logger.info(f"Camera post-BA parameters (refined, unbounded): {final_candidate}")
                else:
                    logger.error(
                        message + " Falling back to the supplied intrinsics with refinement disabled."
                    )
                    refinement_rejected = True
                    locked_dir = sparse_dir.parent / "sparse_locked"
                    locked = self._run_mapping(
                        database_path=database_path,
                        image_path=img_path,
                        output_dir=locked_dir,
                        refine=False,
                    )
                    if not locked:
                        raise Uav3dError(
                            "Intrinsics refinement was rejected as implausible and the locked re-run "
                            "produced no reconstruction. Provide a calibrated focal length "
                            "(sfm.video_focal_px or --focal-px)."
                        )
                    selected_rec = max(locked.values(), key=lambda r: r.num_reg_images())
                    num_models = len(locked)
                    sparse_dir = locked_dir
                    mapping_time_s += time.perf_counter() - t0 - mapping_time_s

        profiles.append(
            SfmStageProfile(
                stage_name="incremental_mapping",
                wall_time_s=mapping_time_s,
                peak_rss_mb=get_peak_rss_mb(),
            )
        )

        final_params = [float(p) for p in self._first_camera(selected_rec).params]

        if not self.config.refine_intrinsics or refinement_rejected:
            logger.info(
                f"Camera post-BA parameters (locked): {final_params} (expected: {initial_params})"
            )
            if not np.allclose(np.array(final_params), np.array(expected_params), atol=1e-3):
                raise Uav3dError(
                    "Camera intrinsics were modified during bundle adjustment despite "
                    f"refine_intrinsics=False! Expected: {expected_params}, Got: {final_params}"
                )
        else:
            logger.info(f"Camera post-BA parameters (refined): {final_params} (initial: {initial_params})")

        final_fov_deg: float | None = None
        fov = horizontal_fov_deg(final_params, calibration.width)
        final_fov_deg = round(fov, 2) if np.isfinite(fov) else None

        num_reg = selected_rec.num_reg_images()
        if num_reg == 0:
            raise Uav3dError("Selected reconstruction contains 0 registered images.")
        reg_rate = float(num_reg / max(1, total_images))
        mean_reproj = (
            float(selected_rec.compute_mean_reprojection_error())
            if hasattr(selected_rec, "compute_mean_reprojection_error")
            else 0.0
        )

        overall_time = time.perf_counter() - overall_start
        peak_rss = get_peak_rss_mb()

        logger.info(
            f"SfM Completed successfully: {num_reg}/{total_images} frames registered ({reg_rate*100:.1f}%), "
            f"reproj error={mean_reproj:.3f} px, wall_time={overall_time:.2f}s, peak_rss={peak_rss:.1f} MB"
        )

        return SfmRunResult(
            reconstruction=selected_rec,
            database_path=database_path,
            sparse_dir=sparse_dir,
            num_models=num_models,
            num_reg_images=num_reg,
            total_images=total_images,
            registration_rate=reg_rate,
            mean_reprojection_error_px=mean_reproj,
            profiles=profiles,
            total_wall_time_s=overall_time,
            peak_rss_mb=peak_rss,
            intrinsics_refined=bool(self.config.refine_intrinsics) and not refinement_rejected,
            initial_params=initial_params,
            final_params=final_params,
            final_fov_deg=final_fov_deg,
            intrinsics_violations=intrinsics_violations,
            refinement_rejected=refinement_rejected,
        )
