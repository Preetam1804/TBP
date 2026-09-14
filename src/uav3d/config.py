"""Configuration management for uav3d pipeline."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from uav3d.exceptions import ConfigError


class IngestConfig(BaseModel):
    """Configuration for dataset ingestion."""

    model_config = ConfigDict(extra="forbid")

    source_type: str = Field(default="zurich", description="Ingestion source type: zurich or video")
    start_frame: int = Field(default=0, ge=0, description="Start index for processing")
    max_frames: int = Field(default=300, gt=0, description="Maximum contiguous frames to ingest")


class KeyframeConfig(BaseModel):
    """Configuration for keyframe selection."""

    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(default="laplacian_variance", description="Selection algorithm")
    stride: int = Field(default=2, ge=1, description="Temporal subsampling stride")
    max_frames: int = Field(default=250, gt=0, description="Maximum number of keyframes to select")
    min_blur: float = Field(default=60.0, ge=0.0, description="Minimum Laplacian variance blur score threshold")
    max_clipping: float = Field(default=0.05, ge=0.0, le=1.0, description="Maximum allowed pixel clipping ratio")
    max_gap: int = Field(default=15, ge=1, description="Maximum allowed source frame index gap between consecutive keyframes")
    orb_features: int = Field(default=2000, gt=0, description="Maximum ORB feature count cap")
    sharpness_threshold: float | None = Field(default=None, description="Optional alias for min_blur")
    target_count: int | None = Field(default=None, description="Optional alias for max_frames")


class SfmConfig(BaseModel):
    """Configuration for visual Structure-from-Motion."""

    model_config = ConfigDict(extra="forbid")

    camera_model: str = Field(default="PINHOLE", description="COLMAP camera model")
    feature_type: str = Field(default="SIFT", description="Feature extractor type")
    matcher_type: str = Field(default="exhaustive", description="COLMAP matcher type")
    min_num_matches: int = Field(default=15, ge=1, description="Minimum inlier matches for pair")


class SyncConfig(BaseModel):
    """Configuration for telemetry time synchronization."""

    model_config = ConfigDict(extra="forbid")

    max_delta_us: int = Field(default=50000, ge=0, description="Max tolerance delta in microseconds")
    strategy: str = Field(default="nearest", description="Sync strategy (nearest or linear)")


class ValidationConfig(BaseModel):
    """Configuration for trajectory evaluation and ATE validation."""

    model_config = ConfigDict(extra="forbid")

    alignment: str = Field(default="sim3", description="Trajectory alignment type: sim3, se3, or none")
    ate_threshold_meters: float = Field(default=5.0, gt=0.0, description="Absolute Trajectory Error warning threshold")


class SourceConfig(BaseModel):
    """Configuration for image sequence or video source."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(default="zurich", description="Source type: 'zurich' or 'video'")
    dataset_dir: str = Field(default="data/raw/zurich_mav", description="Path to Zurich dataset root")
    video_path: str | None = Field(default=None, description="Path to video file if type is 'video'")
    start_id: int = Field(default=0, ge=0, description="Starting numeric image/frame identifier")
    count: int = Field(default=250, gt=0, le=300, description="Contiguous frame count (<= 300)")
    stride: int = Field(default=1, ge=1, description="Frame sampling stride")


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
            with open(path_obj, "r", encoding="utf-8") as f:
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
