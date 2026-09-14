"""Models for camera poses output from visual SfM."""

from pydantic import BaseModel, ConfigDict, Field


class PoseRecord(BaseModel):
    """Represents a camera pose in COLMAP convention (cam_from_world).

    Orientation quaternion uses (qw, qx, qy, qz) wxyz order.
    Translation vector is (tx, ty, tz).
    Camera center in world coordinates is: C = -R.T @ t.
    """

    model_config = ConfigDict(extra="forbid")

    frame_id: str = Field(..., description="Frame identifier associated with this pose (e.g. zurich_000123)")
    timestamp_us: int | None = Field(default=None, description="Capture timestamp in integer microseconds")
    qw: float = Field(..., description="Quaternion w (scalar component)")
    qx: float = Field(..., description="Quaternion x")
    qy: float = Field(..., description="Quaternion y")
    qz: float = Field(..., description="Quaternion z")
    tx: float = Field(..., description="Translation vector x")
    ty: float = Field(..., description="Translation vector y")
    tz: float = Field(..., description="Translation vector z")
    camera_center: list[float] | None = Field(
        default=None,
        description="Optional precomputed camera center in world coordinates [X, Y, Z] = -R.T @ t",
    )
    camera_center_world: list[float] | None = Field(
        default=None,
        description="Precomputed camera center in world coordinates [X, Y, Z] = -R.T @ t",
    )
    num_observations: int | None = Field(
        default=None,
        description="Number of triangulated 2D-3D point observations for this image",
    )
    registered: bool = Field(
        default=True,
        description="Whether this frame was successfully registered in the reconstruction",
    )

    @property
    def qvec(self) -> list[float]:
        """Return quaternion in wxyz order."""
        return [self.qw, self.qx, self.qy, self.qz]

    @property
    def tvec(self) -> list[float]:
        """Return translation vector [tx, ty, tz]."""
        return [self.tx, self.ty, self.tz]


class UnregisteredFrameRecord(BaseModel):
    """Represents a frame that could not be registered during SfM reconstruction."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str = Field(..., description="Unregistered frame identifier")
    reason: str = Field(
        default="not_registered_in_largest_model",
        description="Reason why frame was not registered (e.g. low_inliers, no_matches, disconnected)",
    )


class SfmMetrics(BaseModel):
    """Reconstruction quality and registration summary metrics."""

    model_config = ConfigDict(extra="forbid")

    num_images_total: int = Field(..., description="Total number of input keyframes considered", ge=0)
    num_images_registered: int = Field(..., description="Number of successfully registered frames", ge=0)
    registration_rate: float = Field(..., description="Fraction of frames registered (0.0 to 1.0)", ge=0.0, le=1.0)
    mean_reprojection_error_px: float = Field(..., description="Mean reprojection error in pixels", ge=0.0)
    mean_track_length: float | None = Field(default=None, description="Average 3D point track observation length")
    num_points3D: int | None = Field(default=None, description="Total number of reconstructed 3D world points")
    num_observations: int | None = Field(default=None, description="Total number of 2D-3D point observations")


class SfmCameraBlock(BaseModel):
    """Camera intrinsic parameters and image resolution used during SfM."""

    model_config = ConfigDict(extra="forbid")

    camera_id: int = Field(default=1, description="COLMAP camera identifier")
    model: str = Field(..., description="Camera model (e.g. OPENCV, PINHOLE)")
    width: int = Field(..., description="Working image width in pixels", gt=0)
    height: int = Field(..., description="Working image height in pixels", gt=0)
    params: list[float] = Field(..., description="Final verified camera intrinsic parameters")
    scale_factor: float = Field(default=1.0, description="Image scale factor applied relative to native source resolution")


class PosesFile(BaseModel):
    """File-level schema for serialized SfM camera poses output."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0.0", description="Schema specification version")
    dataset_label: str = Field(default="replayed public dataset", description="Dataset provenance label")
    coordinate_frame: str = Field(
        default="cam_from_world",
        description="Coordinate convention, defaults to COLMAP cam_from_world",
    )
    camera: SfmCameraBlock | None = Field(
        default=None,
        description="Camera intrinsics and working resolution verified post-BA",
    )
    metrics: SfmMetrics | None = Field(
        default=None,
        description="Reconstruction metrics (registration rate, reprojection error, track lengths)",
    )
    poses: list[PoseRecord] = Field(..., description="List of camera poses")
    unregistered_frames: list[UnregisteredFrameRecord] | list[str] = Field(
        default_factory=list,
        description="List of unregistered frames with reasons",
    )
