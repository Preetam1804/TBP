"""Schemas for dense reconstruction outputs."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DenseGeometrySummary(BaseModel):
    """Camera and image geometry used by the dense stage."""

    model_config = ConfigDict(extra="forbid")

    num_registered_views: int = Field(..., ge=0)
    num_reference_views: int = Field(..., ge=0)
    working_width: int = Field(..., gt=0)
    working_height: int = Field(..., gt=0)
    focal_px: float = Field(..., gt=0.0)
    median_baseline: float = Field(..., ge=0.0, description="Median source baseline in SfM units")
    median_depth: float = Field(..., gt=0.0, description="Median mid-range depth in SfM units")
    median_baseline_ratio: float = Field(
        ..., ge=0.0, description="Median baseline / depth ratio used for view selection"
    )
    depth_range: tuple[float, float] = Field(
        ..., description="(near, far) depth bounds applied to every view, in SfM units"
    )


class DenseDepthSummary(BaseModel):
    """Aggregate statistics of the per-view depth maps."""

    model_config = ConfigDict(extra="forbid")

    total_pixels: int = Field(..., ge=0)
    valid_pixels: int = Field(..., ge=0)
    valid_fraction: float = Field(..., ge=0.0, le=1.0)
    plane_count_min: int = Field(..., ge=0)
    plane_count_max: int = Field(..., ge=0)
    median_cost: float = Field(..., ge=0.0)


class DenseFusionSummary(BaseModel):
    """Fusion resolution and error model."""

    model_config = ConfigDict(extra="forbid")

    voxel_units: float = Field(..., gt=0.0)
    voxel_meters: float | None = Field(default=None, description="Voxel size in metres when scale is known")
    trunc_units: float = Field(..., gt=0.0)
    depth_sigma_units: float = Field(..., gt=0.0)
    num_points: int = Field(..., ge=0)
    point_spacing_meters: float | None = Field(default=None)


class DenseMeshSummary(BaseModel):
    """Mesh extraction summary."""

    model_config = ConfigDict(extra="forbid")

    num_vertices: int = Field(..., ge=0)
    num_faces: int = Field(..., ge=0)
    voxel_units: float = Field(..., gt=0.0)


class DenseScaleSummary(BaseModel):
    """Metric scale provenance — never silently assumed."""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(..., gt=0.0, description="Metres per SfM unit")
    source: str = Field(..., description="none, fixed, or gnss")
    status: str = Field(..., description="unverified_monocular_scale, fixed_by_user, gnss_aligned")


class DenseReport(BaseModel):
    """Top-level ``dense_report.json`` schema."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0.0")
    dataset_label: str = Field(default="replayed public dataset")
    run_dir: str = Field(..., description="Source run directory")
    created_utc: str = Field(..., description="Creation timestamp, ISO-8601 UTC")
    backend: str = Field(default="cpu_plane_sweep_mvs", description="Dense reconstruction backend identifier")
    geometry: DenseGeometrySummary
    depth: DenseDepthSummary
    fusion: DenseFusionSummary
    mesh: DenseMeshSummary | None = Field(default=None)
    scale: DenseScaleSummary
    evaluation: dict[str, Any] | None = Field(default=None, description="Ground-truth metrics when supplied")
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    total_seconds: float = Field(..., ge=0.0)
    artifacts: dict[str, str] = Field(default_factory=dict)
