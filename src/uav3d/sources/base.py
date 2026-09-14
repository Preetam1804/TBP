"""Base interfaces and metadata models for image ingestion sources."""

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from uav3d.models.frames import CameraCalibration, FrameRecord


class SourceMetadata(BaseModel):
    """Metadata describing an image sequence or video source."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Source name or sequence identifier")
    frame_count: int = Field(..., description="Number of frames available in selected slice", ge=0)
    fps: float | None = Field(default=None, description="Playback frames per second, if known")
    width: int = Field(..., description="Frame width in pixels", gt=0)
    height: int = Field(..., description="Frame height in pixels", gt=0)
    first_frame_id: str | None = Field(default=None, description="ID of the first frame in slice")
    last_frame_id: str | None = Field(default=None, description="ID of the last frame in slice")
    first_timestamp_us: int | None = Field(default=None, description="First valid microsecond timestamp")
    last_timestamp_us: int | None = Field(default=None, description="Last valid microsecond timestamp")


@runtime_checkable
class ImageSource(Protocol):
    """Protocol defining the interface for image and video ingestion sources."""

    def metadata(self) -> SourceMetadata:
        """Return summary metadata for the source."""
        ...

    def calibration(self) -> CameraCalibration:
        """Return intrinsic camera calibration parameters."""
        ...

    def iter_frames(self) -> Iterator[FrameRecord]:
        """Lazily iterate over FrameRecord items without loading full image pixels."""
        ...

    def get_frame(self, frame_id: str) -> FrameRecord:
        """Retrieve a specific FrameRecord by unique zero-padded frame identifier."""
        ...
