"""Models for telemetry synchronization output."""

from pydantic import BaseModel, ConfigDict, Field


class TelemetryRecord(BaseModel):
    """Represents a synchronized telemetry sample mapped to a timestamp/frame."""

    model_config = ConfigDict(extra="forbid")

    timestamp_us: int = Field(..., description="Timestamp in integer microseconds")
    frame_id: int | None = Field(default=None, description="Mapped frame identifier, if synchronized")
    lat: float | None = Field(default=None, description="WGS84 latitude in degrees")
    lon: float | None = Field(default=None, description="WGS84 longitude in degrees")
    alt: float | None = Field(default=None, description="Altitude in meters")
    easting: float | None = Field(default=None, description="Projected Easting coordinate in meters (UTM/local)")
    northing: float | None = Field(default=None, description="Projected Northing coordinate in meters (UTM/local)")
    source: str | None = Field(default=None, description="Source log name (e.g. OnbordGPS.csv)")


class TelemetryFile(BaseModel):
    """File-level schema for serialized telemetry synchronization output."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0.0", description="Schema specification version")
    dataset_label: str = Field(default="replayed public dataset", description="Dataset provenance label")
    records: list[TelemetryRecord] = Field(..., description="List of synchronized telemetry records")
