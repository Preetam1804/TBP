"""Configuration management for uav3d pipeline."""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from uav3d.exceptions import ConfigError


class IngestConfig(BaseModel):
    """Configuration for dataset ingestion."""

    model_config = ConfigDict(extra="forbid")

    source_type: str = Field(default="zurich", description="Ingestion source type: zurich or video")
    start_frame: int = Field(default=0, ge=0, description="First frame index to ingest (video sources)")
    max_frames: int = Field(
        default=300,
        gt=0,
        description="Maximum contiguous frames to ingest (enforced for video sources)",
    )


class KeyframeConfig(BaseModel):
    """Configuration for keyframe selection."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["laplacian_variance"] = Field(
        default="laplacian_variance",
        description="Selection algorithm (only deterministic Laplacian-variance gating is implemented)",
    )
    stride: int = Field(default=2, ge=1, description="Temporal subsampling stride")
    max_frames: int = Field(
        default=250,
        gt=0,
        description="Maximum number of keyframes to select (selection is uniformly subsampled, not truncated)",
    )
    min_blur: float = Field(default=60.0, ge=0.0, description="Minimum Laplacian variance blur score threshold")
    max_clipping: float = Field(default=0.05, ge=0.0, le=1.0, description="Maximum allowed pixel clipping ratio")
    max_gap: int = Field(default=15, ge=1, description="Maximum allowed source frame index gap between consecutive keyframes")
    orb_features: int = Field(default=2000, gt=0, description="Maximum ORB feature count cap")
    sharpness_threshold: float | None = Field(default=None, description="Optional alias for min_blur")
    target_count: int | None = Field(default=None, description="Optional alias for max_frames")

    @model_validator(mode="after")
    def _resolve_aliases(self) -> "KeyframeConfig":
        """Resolve the documented alias fields onto their canonical equivalents."""
        if self.sharpness_threshold is not None:
            if self.min_blur != type(self).model_fields["min_blur"].default and self.min_blur != self.sharpness_threshold:
                raise ValueError(
                    "keyframes.sharpness_threshold conflicts with keyframes.min_blur; set only one of them"
                )
            self.min_blur = self.sharpness_threshold
        if self.target_count is not None:
            if self.max_frames != type(self).model_fields["max_frames"].default and self.max_frames != self.target_count:
                raise ValueError("keyframes.target_count conflicts with keyframes.max_frames; set only one of them")
            self.max_frames = self.target_count
        return self


class SfmConfig(BaseModel):
    """Configuration for visual Structure-from-Motion."""

    model_config = ConfigDict(extra="forbid")

    camera_model: str = Field(default="OPENCV", description="COLMAP camera model")
    single_camera: bool = Field(default=True, description="Share a single camera intrinsic model across all frames")
    seq_overlap: int = Field(default=10, ge=1, description="Sequential matching overlap window")
    max_image_size: int = Field(default=1600, gt=0, description="Maximum image dimension for feature extraction")
    refine_intrinsics: bool = Field(default=False, description="Whether to refine camera intrinsics during bundle adjustment")
    feature_type: Literal["SIFT"] = Field(default="SIFT", description="Feature extractor type (SIFT only)")
    matcher_type: Literal["sequential"] = Field(
        default="sequential",
        description="COLMAP matcher type (sequential only; exhaustive matching is not wired)",
    )
    min_num_matches: int = Field(
        default=15, ge=1, description="Minimum number of verified inlier matches required per image pair"
    )
    init_min_inliers: int = Field(default=15, ge=1, description="Minimum inliers for initial pair")
    init_min_tri_angle: float = Field(default=2.0, ge=0.0, description="Minimum triangulation angle for initial pair in degrees")
    abs_pose_min_inliers: int = Field(default=10, ge=1, description="Minimum inliers for absolute pose registration")
    min_model_size: int = Field(default=3, ge=2, description="Minimum registered images to keep reconstruction")
    ignore_watermarks: bool = Field(default=True, description="Whether to ignore watermark detection in sequential matching")


class SyncConfig(BaseModel):
    """Configuration for telemetry time synchronization."""

    model_config = ConfigDict(extra="forbid")

    max_delta_us: int = Field(default=500000, ge=0, description="Max tolerance delta in microseconds (default 500ms)")
    strategy: Literal["nearest"] = Field(
        default="nearest",
        description="Sync strategy (only nearest-neighbour timestamp matching is implemented)",
    )
    estimate_offset: bool = Field(default=True, description="Whether to estimate constant clock offset for Mode B")
    min_match_rate: float = Field(default=0.8, ge=0.0, le=1.0, description="Minimum match rate required for synchronized status")


class ValidationConfig(BaseModel):
    """Configuration for trajectory evaluation and ATE validation."""

    model_config = ConfigDict(extra="forbid")

    alignment: Literal["sim3"] = Field(
        default="sim3",
        description="Trajectory alignment type (only 7-DoF Umeyama Sim(3) is implemented)",
    )
    ate_threshold_meters: float = Field(
        default=5.0,
        gt=0.0,
        description="Absolute Trajectory Error RMSE warning threshold in metres (reported as PASS/FAIL)",
    )


class SourceConfig(BaseModel):
    """Configuration for image sequence or video source."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(default="zurich", description="Source type: 'zurich' or 'video'")
    dataset_dir: str = Field(default="data/raw/zurich_mav", description="Path to Zurich dataset root")
    video_path: str | None = Field(default=None, description="Path to video file if type is 'video'")
    start_id: int = Field(default=0, ge=0, description="Starting numeric image/frame identifier")
    count: int = Field(default=250, gt=0, le=300, description="Contiguous frame count (<= 300)")
    stride: int = Field(default=1, ge=1, description="Frame sampling stride")
    allow_default_intrinsics: bool = Field(
        default=False,
        description=(
            "Permit falling back to documented GoPro Hero4 default intrinsics when "
            "calibration_data.npz is missing or unreadable (default: fail loudly)"
        ),
    )


class PipelineConfig(BaseModel):
    """Root configuration for uav3d pipeline."""

    model_config = ConfigDict(extra="forbid")

    data_dir: str = Field(default="./data", description="Root data directory")
    dataset_label: str = Field(default="replayed public dataset", description="Dataset provenance label")
    source: SourceConfig = Field(default_factory=SourceConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    keyframes: KeyframeConfig = Field(default_factory=KeyframeConfig)
    sfm: SfmConfig = Field(default_factory=SfmConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load and validate pipeline configuration from a YAML file.

    Fails fast with ConfigError if file is missing, contains invalid YAML,
    or contains unrecognized/invalid configuration keys (extra='forbid').
    """
    raw_data: dict[str, Any] = {}

    if path is not None:
        path_obj = Path(path)
        if not path_obj.exists():
            raise ConfigError(f"Configuration file not found: {path_obj}")
        try:
            with open(path_obj, encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded is not None:
                    if not isinstance(loaded, dict):
                        raise ConfigError(f"Configuration in {path_obj} must be a dictionary/mapping, got {type(loaded).__name__}")
                    raw_data = loaded
        except yaml.YAMLError as exc:
            raise ConfigError(f"Failed to parse YAML configuration in {path_obj}: {exc}") from exc
        except OSError as exc:
            raise ConfigError(f"IO error reading configuration file {path_obj}: {exc}") from exc

    # Honor UAV3D_DATA_DIR environment variable if data_dir not explicitly set in config
    env_data_dir = os.environ.get("UAV3D_DATA_DIR")
    if env_data_dir and "data_dir" not in raw_data:
        raw_data["data_dir"] = env_data_dir

    try:
        return PipelineConfig(**raw_data)
    except ValidationError as exc:
        raise ConfigError(f"Configuration validation failed: {exc}") from exc
