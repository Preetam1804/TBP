"""Telemetry parsing utilities for uav3d."""

from uav3d.telemetry.synchronizer import (
    TelemetrySynchronizer,
    haversine_distance_m,
    recommend_crs,
)
from uav3d.telemetry.zurich_parser import (
    ZurichGpsRecord,
    ZurichGroundTruthRecord,
    parse_ground_truth_agl,
    parse_onboard_gps,
)

__all__ = [
    "TelemetrySynchronizer",
    "ZurichGpsRecord",
    "ZurichGroundTruthRecord",
    "haversine_distance_m",
    "parse_ground_truth_agl",
    "parse_onboard_gps",
    "recommend_crs",
]
