"""SfM runner executing pycolmap incremental pipeline with locked intrinsics."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap

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


class SfmRunner:
    """Executes incremental visual Structure-from-Motion using pycolmap."""

    def __init__(self, config: SfmConfig | None = None) -> None:
        self.config = config or SfmConfig()

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
            list({p.name for p in img_path.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}})
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
        logger.info(f"Stage 1/4: Feature extraction (model={self.config.camera_model}, max_size={self.config.max_image_size})")

        sift_opts = pycolmap.SiftExtractionOptions()
        sift_opts.max_image_size = self.config.max_image_size

        reader_opts = pycolmap.ImageReaderOptions()
        reader_opts.camera_model = self.config.camera_model
        reader_opts.camera_params = ",".join(str(p) for p in calibration.params)

        camera_mode = pycolmap.CameraMode.SINGLE if self.config.single_camera else pycolmap.CameraMode.AUTO

        try:
            pycolmap.extract_features(
                database_path=str(database_path),
                image_path=str(img_path),
                image_names=image_files,
                camera_mode=camera_mode,
                camera_model=self.config.camera_model,
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
                model=self.config.camera_model,
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
        # Stage 4: Incremental Mapping with Locked Intrinsics
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        logger.info("Stage 4/4: Incremental reconstruction mapping")

        pipeline_opts = pycolmap.IncrementalPipelineOptions()

        # Lock camera intrinsics during bundle adjustment if configured
        if not self.config.refine_intrinsics:
            pipeline_opts.ba_refine_focal_length = False
            pipeline_opts.ba_refine_principal_point = False
            pipeline_opts.ba_refine_extra_params = False
            pipeline_opts.mapper.abs_pose_refine_focal_length = False
            pipeline_opts.mapper.abs_pose_refine_extra_params = False
            logger.info("Intrinsics refinement locked: ba_refine_* and abs_pose_refine_* disabled")

        # Configurable mapper thresholds
        pipeline_opts.mapper.init_min_tri_angle = self.config.init_min_tri_angle
        pipeline_opts.mapper.init_min_num_inliers = self.config.init_min_inliers
        pipeline_opts.mapper.abs_pose_min_num_inliers = self.config.abs_pose_min_inliers
        pipeline_opts.min_model_size = self.config.min_model_size
        pipeline_opts.ignore_watermarks = self.config.ignore_watermarks

        try:
            reconstructions = pycolmap.incremental_mapping(
                database_path=str(database_path),
                image_path=str(img_path),
                output_path=str(sparse_dir),
                options=pipeline_opts,
            )
        except Exception as exc:
            raise Uav3dError(f"Incremental mapping failed with exception: {exc}") from exc

        profiles.append(
            SfmStageProfile(
                stage_name="incremental_mapping",
                wall_time_s=time.perf_counter() - t0,
                peak_rss_mb=get_peak_rss_mb(),
            )
        )

        # -------------------------------------------------------------
        # Model Selection & Locked Intrinsics Assertion
        # -------------------------------------------------------------
        if not reconstructions:
            raise Uav3dError(
                f"COLMAP incremental mapping produced 0 valid reconstructions from {total_images} images. "
                "Check keyframe overlap, feature matches, or scene texture."
            )

        num_models = len(reconstructions)
        logger.info(f"Reconstruction produced {num_models} model(s)")

        # Select largest reconstruction by number of registered images
        selected_rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())
        num_reg = selected_rec.num_reg_images()

        if num_reg == 0:
            raise Uav3dError("Selected reconstruction contains 0 registered images.")

        reg_rate = float(num_reg / max(1, total_images))
        mean_reproj = (
            float(selected_rec.compute_mean_reprojection_error())
            if hasattr(selected_rec, "compute_mean_reprojection_error")
            else 0.0
        )

        # Assert and verify locked intrinsics post-bundle adjustment
        for cam_id, cam in selected_rec.cameras.items():
            actual_params = np.array(cam.params, dtype=np.float64)
            expected_params = np.array(calibration.params, dtype=np.float64)

            logger.info(
                f"Camera #{cam_id} post-BA parameters: {actual_params.tolist()} "
                f"(expected: {expected_params.tolist()})"
            )

            if not self.config.refine_intrinsics:
                if not np.allclose(actual_params, expected_params, atol=1e-3):
                    raise Uav3dError(
                        f"Camera intrinsics were modified during bundle adjustment despite refine_intrinsics=False! "
                        f"Expected: {expected_params.tolist()}, Got: {actual_params.tolist()}"
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
        )
