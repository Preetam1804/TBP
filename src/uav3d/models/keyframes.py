"""Models for keyframe selection output."""

from pydantic import BaseModel, ConfigDict, Field


class KeyframeRecord(BaseModel):
    """Represents a selected keyframe."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str = Field(..., description="Unique frame identifier (e.g. zurich_000123)")
    timestamp_us: int | None = Field(default=None, description="Frame timestamp in integer microseconds, or None")
    file_path: str = Field(..., description="Relative or absolute path to keyframe image")
    score: float | None = Field(default=None, description="Optional keyframe selection quality score")
    selected: bool = Field(default=True, description="Whether this frame was selected as keyframe")
    selection_reason: str | None = Field(default=None, description="Reason for selection or rejection")
    metrics: dict[str, float] | None = Field(default=None, description="Computed quality metrics for this frame")


class KeyframesFile(BaseModel):
    """File-level schema for serialized keyframe selection output."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0.0", description="Schema specification version")
    dataset_label: str = Field(default="replayed public dataset", description="Dataset provenance label")
    keyframes: list[KeyframeRecord] = Field(..., description="List of selected keyframe records")
