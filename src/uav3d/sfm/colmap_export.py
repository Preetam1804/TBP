"""Export COLMAP reconstruction to validated PosesFile schema."""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from uav3d.models.frames import CameraCalibration
from uav3d.models.poses import (
    PoseRecord,
    PosesFile,
    SfmCameraBlock,
    SfmMetrics,
    UnregisteredFrameRecord,
)
from uav3d.sfm.colmap_compat import get_pose_components, is_image_registered

logger = logging.getLogger("uav3d.sfm.colmap_export")


def compute_sample_reprojection_errors(
    reconstruction: Any,
    sample_size: int = 100,
) -> list[float]:
    """Compute per-observation reprojection errors on a sample of 3D points.

    Projects each 3D point into the cameras observing it, measuring Euclidean
    distance in pixels between projected image coordinate and measured 2D feature.

    Args:
        reconstruction: pycolmap.Reconstruction instance.
        sample_size: Maximum number of points to evaluate.

    Returns:
        List of reprojection errors in pixels.
    """
    if not hasattr(reconstruction, "points3D") or not reconstruction.points3D:
        return []

    errors: list[float] = []
    points = list(reconstruction.points3D.values())

    # Deterministic sampling
    step = max(1, len(points) // sample_size)
    sampled_points = points[::step][:sample_size]

    for pt in sampled_points:
        if not hasattr(pt, "track") or not pt.track.elements:
            continue
        xyz_world = np.array(pt.xyz, dtype=np.float64)

        for el in pt.track.elements:
            image_id = el.image_id
            point2D_idx = el.point2D_idx

            if not reconstruction.exists_image(image_id):
                continue
            image = reconstruction.image(image_id)
            if not is_image_registered(image):
                continue

            if not reconstruction.exists_camera(image.camera_id):
                continue
            camera = reconstruction.camera(image.camera_id)

            # Transform world point to camera coordinates
            if hasattr(image, "cam_from_world"):
                cfw = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
                p_cam = cfw * xyz_world
            else:
                qvec, tvec, _ = get_pose_components(image)
                from uav3d.sfm.colmap_compat import quaternion_to_rotation_matrix

                R = quaternion_to_rotation_matrix(qvec)
                t = np.array(tvec, dtype=np.float64).reshape(3, 1)
                p_cam = (R @ xyz_world.reshape(3, 1) + t).flatten()

            # Point must be strictly in front of camera
            if p_cam[2] <= 1e-4:
                continue

            # Project into image plane using camera model
            try:
                if hasattr(camera, "img_from_cam"):
                    proj_uv = np.array(camera.img_from_cam(p_cam), dtype=np.float64)
                else:
                    # Generic pinhole fallback
                    fx, fy = float(camera.params[0]), float(camera.params[1])
                    cx, cy = float(camera.params[2]), float(camera.params[3])
                    proj_uv = np.array([fx * p_cam[0] / p_cam[2] + cx, fy * p_cam[1] / p_cam[2] + cy])

                # Get observed 2D point
                if hasattr(image, "points2D") and point2D_idx < len(image.points2D):
                    obs_uv = np.array(image.points2D[point2D_idx].xy, dtype=np.float64)
                    err = float(np.linalg.norm(proj_uv - obs_uv))
                    errors.append(err)
            except Exception as exc:
                logger.debug(f"Reprojection calculation error: {exc}")

    return errors


def export_poses(
    reconstruction: Any,
    all_image_names: list[str],
    camera_calibration: CameraCalibration,
    output_path: Path | str,
    dataset_label: str = "replayed public dataset",
    scale_factor: float = 1.0,
    timestamps_map: dict[str, int | None] | None = None,
    intrinsics_locked: bool = True,
    initial_params: list[float] | None = None,
    horizontal_fov_deg: float | None = None,
) -> PosesFile:
    """Export COLMAP reconstruction poses to validated PosesFile schema.

    Args:
        reconstruction: pycolmap.Reconstruction object with registered images.
        all_image_names: Complete list of image names or frame IDs fed into SfM.
        camera_calibration: Verified camera calibration parameters.
        output_path: Destination path for poses.json.
        dataset_label: Dataset provenance label.
        scale_factor: Resolution scale factor used during feature extraction.
        timestamps_map: Optional mapping of frame_id -> timestamp_us.
        intrinsics_locked: Whether bundle adjustment was forbidden from refining intrinsics.
        initial_params: Pre-BA intrinsics (recorded when refinement was enabled).
        horizontal_fov_deg: Horizontal FoV implied by the final intrinsics.

    Returns:
        Validated PosesFile instance.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    timestamps_map = timestamps_map or {}

    # 1. Map registered images
    registered_poses: list[PoseRecord] = []
    registered_frame_ids: set[str] = set()

    # Determine image list from reconstruction
    rec_images = []
    if hasattr(reconstruction, "images"):
        imgs = reconstruction.images
        if callable(imgs):
            imgs = imgs()
        if hasattr(imgs, "values"):
            rec_images = list(imgs.values())
        elif isinstance(imgs, (list, tuple)):
            rec_images = list(imgs)

    for img in rec_images:
        if not is_image_registered(img):
            continue

        frame_id = Path(img.name).stem
        registered_frame_ids.add(frame_id)
        registered_frame_ids.add(img.name)

        qvec_wxyz, tvec_xyz, camera_center_world = get_pose_components(img)
        num_obs = (
            img.num_points3D()
            if callable(getattr(img, "num_points3D", None))
            else getattr(img, "num_points3D", None)
        )
        ts = timestamps_map.get(frame_id)

        record = PoseRecord(
            frame_id=frame_id,
            timestamp_us=ts,
            qw=qvec_wxyz[0],
            qx=qvec_wxyz[1],
            qy=qvec_wxyz[2],
            qz=qvec_wxyz[3],
            tx=tvec_xyz[0],
            ty=tvec_xyz[1],
            tz=tvec_xyz[2],
            camera_center=camera_center_world,
            camera_center_world=camera_center_world,
            num_observations=num_obs,
            registered=True,
        )
        registered_poses.append(record)

    # Sort registered poses by frame_id for determinism
    registered_poses.sort(key=lambda p: p.frame_id)

    # 2. Identify and record unregistered frames
    unregistered_records: list[UnregisteredFrameRecord] = []
    for raw_name in all_image_names:
        fid = Path(raw_name).stem
        if fid not in registered_frame_ids and raw_name not in registered_frame_ids:
            unregistered_records.append(
                UnregisteredFrameRecord(
                    frame_id=fid,
                    reason="not_registered_in_largest_model",
                )
            )

    # 3. Calculate summary metrics
    total_count = len(all_image_names)
    reg_count = len(registered_poses)
    reg_rate = float(reg_count / max(1, total_count))

    # Built-in mean reprojection error
    if hasattr(reconstruction, "compute_mean_reprojection_error"):
        mean_reproj_err = float(reconstruction.compute_mean_reprojection_error())
    else:
        sample_errs = compute_sample_reprojection_errors(reconstruction, sample_size=100)
        mean_reproj_err = float(np.mean(sample_errs)) if sample_errs else 0.0

    # Mean track length
    mean_track = (
        float(reconstruction.compute_mean_track_length())
        if hasattr(reconstruction, "compute_mean_track_length")
        else None
    )

    num_pts = (
        reconstruction.num_points3D()
        if callable(getattr(reconstruction, "num_points3D", None))
        else getattr(reconstruction, "num_points3D", None)
    )

    # Compute total observations count
    total_obs = None
    if hasattr(reconstruction, "points3D"):
        pts_map = reconstruction.points3D
        if hasattr(pts_map, "values"):
            total_obs = sum(
                len(p.track.elements)
                for p in pts_map.values()
                if hasattr(p, "track") and hasattr(p.track, "elements")
            )

    metrics = SfmMetrics(
        num_images_total=total_count,
        num_images_registered=reg_count,
        registration_rate=round(reg_rate, 4),
        mean_reprojection_error_px=round(mean_reproj_err, 4),
        mean_track_length=round(mean_track, 2) if mean_track is not None else None,
        num_points3D=num_pts,
        num_observations=total_obs,
    )

    # 4. Verified Camera Block
    # Fetch camera params from reconstruction if available to confirm verified values
    verified_params = camera_calibration.params
    cam_model = camera_calibration.model
    w, h = camera_calibration.width, camera_calibration.height

    if hasattr(reconstruction, "cameras") and reconstruction.cameras:
        first_cam = next(iter(reconstruction.cameras.values()))
        verified_params = [float(p) for p in first_cam.params]
        w, h = int(first_cam.width), int(first_cam.height)
        if hasattr(first_cam, "model"):
            cam_model = str(first_cam.model).replace("CameraModelId.", "")

    camera_block = SfmCameraBlock(
        camera_id=1,
        model=cam_model,
        width=w,
        height=h,
        params=verified_params,
        scale_factor=scale_factor,
        intrinsics_locked=intrinsics_locked,
        initial_params=initial_params,
        horizontal_fov_deg=horizontal_fov_deg,
        intrinsics_source=camera_calibration.source,
    )

    # 5. Build and validate PosesFile schema
    poses_file = PosesFile(
        schema_version="1.0.0",
        dataset_label=dataset_label,
        coordinate_frame="cam_from_world",
        camera=camera_block,
        metrics=metrics,
        poses=registered_poses,
        unregistered_frames=unregistered_records,
    )

    # Serialize to JSON with formatting
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(poses_file.model_dump_json(indent=2))

    logger.info(
        f"Exported poses to {out_file}: {reg_count}/{total_count} registered ({reg_rate*100:.1f}%), "
        f"mean reproj error={mean_reproj_err:.3f} px, {len(unregistered_records)} unregistered"
    )

    return poses_file
