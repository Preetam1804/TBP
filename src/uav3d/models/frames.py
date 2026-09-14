"""Models for frame records and camera calibration."""

from pydantic import BaseModel, ConfigDict, Field


class FrameRecord(BaseModel):
    """Represents a single ingested video/sequence frame."""

    model_config = ConfigDict(extra="forbid")

    frame_id: int = Field(..., description="Contiguous 0-indexed frame identifier")
    timestamp_us: int = Field(..., description="Capture timestamp in integer microseconds")
    file_path: str = Field(..., description="Relative or absolute path to the frame image file")
    width: int = Field(..., description="Frame image width in pixels", gt=0)
    height: int = Field(..., description="Frame image height in pixels", gt=0)


class CameraCalibration(BaseModel):
    """Represents camera intrinsic calibration parameters."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description="Camera model name (e.g. PINHOLE, RADIAL, OPENCV)")
    width: int = Field(..., description="Sensor width in pixels", gt=0)
    height: int = Field(..., description="Sensor height in pixels", gt=0)
    params: list[float] = Field(..., description="Intrinsic camera parameters list")
    source: str = Field(..., description="Calibration source (e.g. calibration_data.npz)")
