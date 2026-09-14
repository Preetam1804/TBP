"""Validation schemas for trajectory evaluation and ATE metrics reporting."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PerFrameAte(BaseModel):
    """Per-frame absolute trajectory error and coordinates."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str = Field(..., description="Frame identifier")
    error_m: float = Field(..., description="Absolute position error in meters against ground truth", ge=0.0)
    est_world: list[float] = Field(..., description="Raw estimated camera center [X, Y, Z] in SfM world frame")
    gt_world: list[float] = Field(..., description="Ground truth camera position [X, Y, Z] in meters")
    aligned_est_world: list[float] = Field(..., description="Similarity-aligned estimated camera center in GT metric frame")


class AteMetrics(BaseModel):
    """Summary Absolute Trajectory Error (ATE) metrics computed via 7-DoF Umeyama alignment."""

    model_config = ConfigDict(extra="forbid")

    rmse_m: float = Field(..., description="Root Mean Square Error in meters", ge=0.0)
    median_m: float = Field(..., description="Median trajectory position error in meters", ge=0.0)
    p95_m: float = Field(..., description="95th percentile trajectory position error in meters", ge=0.0)
    max_m: float = Field(..., description="Maximum trajectory position error in meters", ge=0.0)
    mean_m: float = Field(..., description="Mean trajectory position error in meters", ge=0.0)
    scale_factor: float = Field(..., description="Recovered Sim(3) similarity scale factor s from SfM to GT", gt=0.0)
    num_matched_frames: int = Field(..., description="Count of common frames evaluated between SfM and Ground Truth", ge=0)
    num_registered_frames: int = Field(..., description="Total registered SfM frames considered", ge=0)
    num_gt_frames: int = Field(..., description="Total ground truth frames available", ge=0)
    fit_condition: str = Field(
        ...,
        description="Condition check on GT trajectory geometry: 'well_conditioned' or 'weak_flightline_geometry'",
    )
    registration_rate: float | None = Field(default=None, description="SfM registration rate carried from poses.json")
    mean_reprojection_error_px: float | None = Field(default=None, description="Mean reprojection error carried from poses.json")


class AteReport(BaseModel):
    """File-level serialization schema for ate_report.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0.0", description="Schema specification version")
    dataset_label: str = Field(
        default="replayed public dataset (Zurich Urban MAV) — not field-flight validation",
        description="Dataset provenance label",
    )
    status: str = Field(default="completed", description="Validation execution status")
    metrics: AteMetrics = Field(..., description="Summary ATE metrics")
    alignment_transform: dict[str, Any] = Field(
        ...,
        description="Optimal Sim(3) alignment transform parameters (scale, rotation, translation)",
    )
    per_frame_errors: list[PerFrameAte] = Field(default_factory=list, description="List of per-frame error records")
    plots: dict[str, str] = Field(default_factory=dict, description="Mapping of plot names to relative file paths")
