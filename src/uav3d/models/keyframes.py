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


class KeyframeSelectionMeta(BaseModel):
    """Provenance of the keyframe selection run (parameters actually applied)."""

    model_config = ConfigDict(extra="forbid")

    considered_count: int = Field(..., ge=0, description="Frames evaluated before gating")
    selected_count: int = Field(..., ge=0, description="Frames selected after gating and capping")
    initial_stride: int = Field(..., ge=1, description="Configured stride at the start of selection")
    final_stride: int = Field(..., ge=1, description="Stride actually applied after any overlap-guard reduction")
    min_blur_mode: str = Field(..., description="Blur gate mode applied (absolute or adaptive)")
    effective_min_blur: float = Field(..., ge=0.0, description="Blur threshold actually applied")
    max_clipping: float = Field(..., ge=0.0, description="Clipping gate threshold applied")
    max_gap: int = Field(..., ge=1, description="Maximum allowed index gap between consecutive keyframes")
    rejected_by_blur: int = Field(default=0, ge=0, description="Candidates rejected by the blur gate")
    rejected_by_clipping: int = Field(default=0, ge=0, description="Candidates rejected by the clipping gate")
    decode_failed_count: int = Field(default=0, ge=0, description="Frames that could not be decoded")
    max_frames: int = Field(..., gt=0, description="Selection cap applied")
    cap_applied_by_subsampling: bool = Field(
        default=False,
        description="True when max_frames was enforced by uniform subsampling of the candidates",
    )


class KeyframesFile(BaseModel):
    """File-level schema for serialized keyframe selection output."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0.0", description="Schema specification version")
    dataset_label: str = Field(default="replayed public dataset", description="Dataset provenance label")
    selection: KeyframeSelectionMeta | None = Field(
        default=None, description="Selection parameters actually applied in this run"
    )
    keyframes: list[KeyframeRecord] = Field(..., description="List of selected keyframe records")
