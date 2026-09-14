"""Defensive parsers for Zurich Urban MAV onboard GPS and ground truth logs."""

import csv
import logging
from pathlib import Path
from typing import NamedTuple

from uav3d.exceptions import DataError

logger = logging.getLogger("uav3d.telemetry")


class ZurichGpsRecord(NamedTuple):
    """Parsed Zurich onboard GPS entry."""

    timestamp_us: int
    image_id: int
    lat: float | None
    lon: float | None
    alt: float | None
    raw_fields: dict[str, str]


class ZurichGroundTruthRecord(NamedTuple):
    """Ground-truth camera pose from GroundTruthAGL.csv.

    EVALUATION ONLY — must never be consumed by visual SfM or reconstruction stages.
    """

    image_id: int
    x: float
    y: float
    z: float
    raw_fields: dict[str, str]
    is_evaluation_only: bool = True


def parse_onboard_gps(csv_path: Path | str) -> tuple[dict[int, ZurichGpsRecord], int]:
    """Parse OnbordGPS.csv defensively into a mapping of image_id -> ZurichGpsRecord.

    Rules:
    - Reads and logs header columns dynamically (never hardcodes column counts).
    - Normalizes timestamps to integer microseconds.
    - Fails loudly (DataError) on duplicate image IDs.
    - Skips and counts malformed rows with warnings.

    Returns:
        tuple of (records_by_image_id, skipped_row_count)
    """
    path = Path(csv_path)
    if not path.exists():
        raise DataError(f"Onboard GPS log file not found: {path}")

    records: dict[int, ZurichGpsRecord] = {}
    skipped_count = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return records, 0

        # Clean header strings
        headers = [h.strip().lstrip("#").strip() for h in header]
        logger.info(f"Parsing {path.name} with columns: {headers}")

        # Determine column indexes defensively
        ts_idx = 0
        img_idx = 1
        lat_idx: int | None = None
        lon_idx: int | None = None
        alt_idx: int | None = None

        for idx, col_name in enumerate(headers):
            lower = col_name.lower()
            if "time" in lower and idx < 2:
                ts_idx = idx
            elif ("image" in lower or "frame" in lower or "id" in lower) and idx < 2:
                img_idx = idx
            elif "lat" in lower and lat_idx is None:
                lat_idx = idx
            elif "lon" in lower and lon_idx is None:
                lon_idx = idx
            elif ("alt" in lower or "height" in lower or "elev" in lower) and alt_idx is None:
                alt_idx = idx

        # Fallback heuristic for standard Zurich GPS layout if not named in header
        if lat_idx is None and len(headers) > 2:
            lat_idx = 2
        if lon_idx is None and len(headers) > 3:
            lon_idx = 3
        if alt_idx is None and len(headers) > 4:
            alt_idx = 4

        line_num = 1
        for row in reader:
            line_num += 1
            if not row or all(not cell.strip() for cell in row):
                continue

            if len(row) < 2:
                logger.warning(f"Skipping malformed row at line {line_num} in {path.name}: too few columns ({len(row)})")
                skipped_count += 1
                continue

            try:
                # Parse timestamp into integer microseconds
                raw_ts = float(row[ts_idx].strip())
                timestamp_us = int(raw_ts)

                # Parse image_id
                image_id = int(float(row[img_idx].strip()))
            except (ValueError, IndexError) as exc:
                logger.warning(f"Skipping malformed row at line {line_num} in {path.name}: {exc}")
                skipped_count += 1
                continue

            # Detect duplicate image IDs — must fail loudly per specification
            if image_id in records:
                raise DataError(
                    f"Duplicate image_id {image_id} detected at line {line_num} in {path.name} "
                    f"(previously registered with timestamp {records[image_id].timestamp_us} µs)"
                )

            # Optional GPS coordinate parsing
            lat = None
            lon = None
            alt = None
            if lat_idx is not None and lat_idx < len(row):
                try:
                    lat = float(row[lat_idx].strip())
                except ValueError:
                    lat = None

            if lon_idx is not None and lon_idx < len(row):
                try:
                    lon = float(row[lon_idx].strip())
                except ValueError:
                    lon = None

            if alt_idx is not None and alt_idx < len(row):
                try:
                    alt = float(row[alt_idx].strip())
                except ValueError:
                    alt = None

            raw_dict: dict[str, str] = {}
            for col_i, cell in enumerate(row):
                key = headers[col_i] if col_i < len(headers) else f"col_{col_i}"
                raw_dict[key] = cell.strip()

            records[image_id] = ZurichGpsRecord(
                timestamp_us=timestamp_us,
                image_id=image_id,
                lat=lat,
                lon=lon,
                alt=alt,
                raw_fields=raw_dict,
            )

    logger.info(f"Loaded {len(records)} GPS records from {path.name} ({skipped_count} rows skipped)")
    return records, skipped_count


def parse_ground_truth_agl(csv_path: Path | str) -> tuple[dict[int, ZurichGroundTruthRecord], int]:
    """Parse GroundTruthAGL.csv (image_id -> XYZ coordinates).

    STRICTLY FOR EVALUATION ONLY. Must never be used as input to SfM.

    Returns:
        tuple of (gt_records_by_image_id, skipped_row_count)
    """
    path = Path(csv_path)
    if not path.exists():
        raise DataError(f"Ground truth file not found: {path}")

    records: dict[int, ZurichGroundTruthRecord] = {}
    skipped_count = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return records, 0

        headers = [h.strip().lstrip("#").strip() for h in header]
        logger.info(f"Parsing Ground Truth (EVALUATION ONLY) {path.name} with columns: {headers}")

        img_idx = 0
        x_idx = 1
        y_idx = 2
        z_idx = 3

        for idx, col_name in enumerate(headers):
            lower = col_name.lower()
            if ("image" in lower or "frame" in lower or "id" in lower) and img_idx == 0:
                img_idx = idx
            elif lower in ("x", "pos_x", "easting") and x_idx == 1:
                x_idx = idx
            elif lower in ("y", "pos_y", "northing") and y_idx == 2:
                y_idx = idx
            elif lower in ("z", "pos_z", "alt", "height") and z_idx == 3:
                z_idx = idx

        line_num = 1
        for row in reader:
            line_num += 1
            if not row or all(not cell.strip() for cell in row):
                continue

            if len(row) < 4:
                logger.warning(f"Skipping malformed GT row at line {line_num} in {path.name}: too few columns")
                skipped_count += 1
                continue

            try:
                image_id = int(float(row[img_idx].strip()))
                x = float(row[x_idx].strip())
                y = float(row[y_idx].strip())
                z = float(row[z_idx].strip())
            except (ValueError, IndexError) as exc:
                logger.warning(f"Skipping malformed GT row at line {line_num} in {path.name}: {exc}")
                skipped_count += 1
                continue

            if image_id in records:
                raise DataError(f"Duplicate image_id {image_id} detected in Ground Truth file {path.name} at line {line_num}")

            raw_dict: dict[str, str] = {}
            for col_i, cell in enumerate(row):
                key = headers[col_i] if col_i < len(headers) else f"col_{col_i}"
                raw_dict[key] = cell.strip()

            records[image_id] = ZurichGroundTruthRecord(
                image_id=image_id,
                x=x,
                y=y,
                z=z,
                raw_fields=raw_dict,
                is_evaluation_only=True,
            )

    logger.info(f"Loaded {len(records)} Ground Truth records from {path.name} ({skipped_count} rows skipped)")
    return records, skipped_count
