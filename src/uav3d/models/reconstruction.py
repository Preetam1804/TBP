"""Schema for the exported 3D reconstruction model summary."""

from pydantic import BaseModel, ConfigDict, Field


class ReconstructionSummary(BaseModel):
    """Versioned summary of an exported 3D model (sparse reconstruction).

    This is the machine-readable manifest for everything the `reconstruct`
    command writes to a run directory: the colored point cloud, the COLMAP
    sparse model directory, the camera trajectory, and the pose file.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(default="1.0.0", description="Schema specification version")
    dataset_label: str = Field(default="replayed public dataset", description="Dataset provenance label")
    num_source_frames: int = Field(default=0, ge=0, description="Frames available in the source (pre-stride)")
    num_keyframes: int = Field(default=0, ge=0, description="Keyframes fed to SfM")
    num_registered_frames: int = Field(default=0, ge=0, description="Frames registered in the main model")
    num_unregistered_frames: int = Field(default=0, ge=0, description="Keyframes not registered")
    registration_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Registered / keyframes")
    mean_reprojection_error_px: float | None = Field(default=None, description="Mean reprojection error of main model")
    num_models: int = Field(default=0, ge=0, description="Number of COLMAP reconstructions kept")
    num_points3d: int = Field(default=0, ge=0, description="Reconstructed 3D point count")
    mean_track_length: float | None = Field(default=None, description="Mean 3D-point track length")
    scale_note: str = Field(
        default="monocular SfM: metric scale is unknown until aligned to a metric reference (GPS/GT)",
        description="Scale interpretation of the exported model",
    )
    intrinsics_note: str | None = Field(
        default=None, description="How the camera focal length was chosen and how far it can be trusted"
    )
    focal_search: dict | None = Field(
        default=None, description="Focal-search evidence (candidates, selected focal, observability) if it ran"
    )
    model_ply: str = Field(default="model.ply", description="Path (relative to run dir) of the colored point cloud")
    cloud_dense_ply: str | None = Field(default=None, description="Path (relative to run dir) of dense point cloud")
    mesh_ply: str | None = Field(default=None, description="Path (relative to run dir) of surface triangle mesh")
    mesh_decimated_ply: str | None = Field(default=None, description="Path (relative to run dir) of decimated mesh")
    map_3d_ply: str | None = Field(default=None, description="Path (relative to run dir) of completed 3D map mesh")
    map_3d_glb: str | None = Field(default=None, description="Path (relative to run dir) of WebGL 3D map asset")
    orthomosaic_png: str | None = Field(default=None, description="Path (relative to run dir) of 2D top-down orthomosaic")
    dsm_png: str | None = Field(default=None, description="Path (relative to run dir) of Digital Surface Model heatmap")
    model_dir: str = Field(default="model", description="Path (relative to run dir) of the COLMAP sparse model directory")
    trajectory_ply: str = Field(default="trajectory.ply", description="Path (relative to run dir) of the camera path")
    poses_json: str = Field(default="poses.json", description="Path (relative to run dir) of the pose file")
    telemetry_json: str | None = Field(default=None, description="Path (relative to run dir) if telemetry sync ran")
    ate_report_json: str | None = Field(default=None, description="Path (relative to run dir) if ATE validation ran")
    colmap_native_log: str = Field(default="colmap_native.log", description="Captured raw COLMAP C++ log file")
    created_utc: str = Field(default="", description="ISO-8601 UTC timestamp of the export")
