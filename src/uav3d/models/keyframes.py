"""Models for keyframe selection output."""

from pydantic import BaseModel, ConfigDict, Field


class KeyframeRecord(BaseModel):
    """Represents a selected keyframe."""

    model_config = ConfigDict(extra="forbid")

    frame_id: int = Field(..., description="Unique frame identifier")
    timestamp_us: int = Field(..., description="Frame timestamp in integer microseconds")
    file_path: str = Field(..., description="Relative or absolute path to keyframe image")
    score: float | None = Field(default=None, description="Optional keyframe selection quality score")


class KeyframesFile(BaseModel):
    """File-level schema for serialized keyframe selection output."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0.0", description="Schema specification version")
    dataset_label: str = Field(default="replayed public dataset", description="Dataset provenance label")
    keyframes: list[KeyframeRecord] = Field(..., description="List of selected keyframe records")
