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

    @property
    def qvec(self) -> list[float]:
        """Return quaternion in wxyz order."""
        return [self.qw, self.qx, self.qy, self.qz]

    @property
    def tvec(self) -> list[float]:
        """Return translation vector [tx, ty, tz]."""
        return [self.tx, self.ty, self.tz]


class PosesFile(BaseModel):
    """File-level schema for serialized SfM camera poses output."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0.0", description="Schema specification version")
    dataset_label: str = Field(default="replayed public dataset", description="Dataset provenance label")
    coordinate_frame: str = Field(
        default="cam_from_world",
        description="Coordinate convention, defaults to COLMAP cam_from_world",
    )
    poses: list[PoseRecord] = Field(..., description="List of camera poses")
