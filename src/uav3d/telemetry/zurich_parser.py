"""Defensive parsers for Zurich Urban MAV onboard GPS and ground truth logs.

Real-data format notes (ETH RPG ``AGZ/readme.txt``):

* ``OnbordGPS.csv``  — ``1: timestamp, 2: imgid, 3: lat (1E7 degrees),
  4: lon (1E7 degrees), 5: alt (1E3 meters, i.e. millimetres above MSL), 6..: accuracy/velocity``.
* ``GroundTruthAGL.csv`` — ``1: imgid, 2: x_gt, 3: y_gt, 4: z_gt`` in WGS 84 / UTM zone 32N metres.
* Column meanings are documented by position; the shipped datasets may or may not carry a header
  line, and the timestamp column is ambiguous between the IJRR paper text ("milliseconds") and the
  dataset website ("microseconds"). Both cases are therefore detected from the data itself instead
  of being hardcoded.
"""

import csv
import logging
import math
from pathlib import Path
from typing import NamedTuple

from uav3d.exceptions import DataError

logger = logging.getLogger("uav3d.telemetry")

# Documented Zuricher GPS scaling factors.
GPS_LAT_LON_SCALE = 1e7
GPS_ALTITUDE_SCALE = 1e3

# Plausible Unix-epoch window used to disambiguate the timestamp unit: 2000-01-01 .. 2100-01-01.
_EPOCH_MIN_US = 946_684_800_000_000
_EPOCH_MAX_US = 4_102_444_800_000_000

# Physical sanity bands (degrees / metres).
_LAT_LIMIT = 90.0
_LON_LIMIT = 180.0
_ALTITUDE_BAND_M = (-2000.0, 20000.0)

_POSITIONAL_GPS_COLUMNS = ["timestamp", "image_id", "lat", "lon", "alt"]
_POSITIONAL_GT_COLUMNS = ["image_id", "x", "y", "z"]


class ZurichGpsRecord(NamedTuple):
    """Parsed Zurich onboard GPS entry (coordinates normalised to degrees/metres)."""

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


def _is_number(text: str) -> bool:
    """Return True if a cell is a (possibly signed/scientific) number."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value)


def normalize_timestamp_us(raw_value: float) -> tuple[int, str]:
    """Normalize a logged timestamp to integer microseconds.

    The dataset documentation is inconsistent (milliseconds vs microseconds), so the unit is
    inferred from the value itself: prefer the interpretation that lands inside a plausible
    Unix-epoch window, otherwise fall back to an order-of-magnitude rule for boot-relative clocks.

    Returns:
        tuple of (timestamp_us, human-readable description of the detected unit).
    """
    # Candidates are (multiplier_to_microseconds, unit_label), ordered so that the most common
    # PX4/GoPro encodings win when several interpretations are epoch-plausible.
    for scale, label in ((1.0, "microseconds"), (1e3, "milliseconds"), (1e-3, "nanoseconds"), (1e6, "seconds")):
        candidate = raw_value * scale
        if _EPOCH_MIN_US <= candidate <= _EPOCH_MAX_US:
            return round(candidate), label

    # No epoch-plausible interpretation: the clock is likely boot-relative (PX4 autopilot clock).
    magnitude = abs(raw_value)
    if magnitude >= 1e14:
        return round(raw_value), "microseconds (boot-relative clock)"
    if magnitude >= 1e11:
        return round(raw_value * 1e3), "milliseconds (boot-relative clock)"
    if magnitude >= 1e8:
        return round(raw_value), "microseconds (boot-relative clock)"
    return round(raw_value * 1e6), "seconds (boot-relative clock)"


def _normalize_latitude(value: float) -> float:
    """Normalize a latitude that may use the documented 1E7 fixed-point scaling."""
    if abs(value) <= _LAT_LIMIT:
        return value
    scaled = value / GPS_LAT_LON_SCALE
    if abs(scaled) <= _LAT_LIMIT:
        return scaled
    raise DataError(f"Invalid latitude {value}: neither degrees nor 1E7-scaled degrees are plausible")


def _normalize_longitude(value: float) -> float:
    """Normalize a longitude that may use the documented 1E7 fixed-point scaling."""
    if abs(value) <= _LON_LIMIT:
        return value
    scaled = value / GPS_LAT_LON_SCALE
    if abs(scaled) <= _LON_LIMIT:
        return scaled
    raise DataError(f"Invalid longitude {value}: neither degrees nor 1E7-scaled degrees are plausible")


def _normalize_altitude(value: float) -> float:
    """Normalize an altitude that may be logged in millimetres (documented 1E3 scaling)."""
    low, high = _ALTITUDE_BAND_M
    if low <= value <= high:
        return value
    scaled = value / GPS_ALTITUDE_SCALE
    if low <= scaled <= high:
        return scaled
    raise DataError(f"Invalid altitude {value}: neither metres nor millimetres are plausible")


def _read_header_and_first_data_row(reader: "csv._reader", path: Path, positional_columns: list[str]) -> tuple[list[str], list[str] | None]:
    """Detect a header row and return (header_names, first_data_row_or_None).

    Skips blank rows and pure-comment rows. A first non-empty row without a leading '#'
    whose cells are all numeric is treated as data (the files are documented column-by-column,
    so a header line is optional) and returned as the first data row.
    """
    for row in reader:
        if not row or all(not cell.strip() for cell in row):
            continue
        if row[0].lstrip().startswith("#"):
            headers = [h.strip().lstrip("#").strip() for h in row]
            logger.info(f"{path.name}: header detected -> {headers}")
            return headers, None
        if all(_is_number(cell.strip()) for cell in row if cell.strip()):
            logger.info(
                f"{path.name}: no header row detected; using positional columns {positional_columns}"
            )
            return list(positional_columns), row
        # Non-numeric, non-comment first row: treat it as a header (names as written).
        headers = [h.strip().lstrip("#").strip() for h in row]
        logger.info(f"{path.name}: header detected -> {headers}")
        return headers, None

    return list(positional_columns), None


def _column_index(headers: list[str], keywords: tuple[str, ...], default: int) -> int:
    """Find the first column whose name contains any keyword, else return the default index."""
    for idx, name in enumerate(headers):
        lower = name.lower()
        if any(keyword in lower for keyword in keywords):
            return idx
    return default


def parse_onboard_gps(csv_path: Path | str) -> tuple[dict[int, ZurichGpsRecord], int]:
    """Parse OnbordGPS.csv defensively into a mapping of image_id -> ZurichGpsRecord.

    Rules:
    - Header is optional; columns are resolved by name when present, by documented position otherwise.
    - Latitude/longitude are normalised from the documented 1E7 fixed-point encoding to degrees,
      altitude from millimetres to metres, timestamps to integer microseconds (units auto-detected).
    - Applied scalings are recorded per row in ``raw_fields`` for traceability.
    - Fails loudly (DataError) on duplicate image IDs or physically impossible coordinates.
    - Skips and counts malformed rows with warnings.

    Returns:
        tuple of (records_by_image_id, skipped_row_count)
    """
    path = Path(csv_path)
    if not path.exists():
        raise DataError(f"Onboard GPS log file not found: {path}")

    records: dict[int, ZurichGpsRecord] = {}
    skipped_count = 0

    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        headers, first_data_row = _read_header_and_first_data_row(reader, path, _POSITIONAL_GPS_COLUMNS)

        ts_idx = _column_index(headers, ("time", "stamp"), 0)
        img_idx = _column_index(headers, ("image", "img", "frame", "id"), 1)
        lat_idx = _column_index(headers, ("lat",), 2)
        lon_idx = _column_index(headers, ("lon",), 3)
        alt_idx = _column_index(headers, ("alt", "height", "elev"), 4)

        rows = [first_data_row, *reader] if first_data_row is not None else list(reader)
        timestamp_unit: str | None = None
        lat_scale_applied = False
        lon_scale_applied = False
        alt_scale_applied = False

        line_num = 1
        for row in rows:
            line_num += 1
            if not row or all(not cell.strip() for cell in row):
                continue
            if row[0].lstrip().startswith("#"):
                continue

            if len(row) < 2:
                logger.warning(f"Skipping malformed row at line {line_num} in {path.name}: too few columns ({len(row)})")
                skipped_count += 1
                continue

            try:
                raw_ts = float(row[ts_idx].strip())
                timestamp_us, unit = normalize_timestamp_us(raw_ts)
                if timestamp_unit is None:
                    timestamp_unit = unit
                    logger.info(f"{path.name}: timestamps interpreted as {unit} (e.g. {raw_ts} -> {timestamp_us} us)")

                # image_id may be logged with a decimal point (e.g. "12.0")
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

            lat = lon = alt = None
            try:
                if lat_idx < len(row) and row[lat_idx].strip():
                    raw_lat = float(row[lat_idx].strip())
                    lat = _normalize_latitude(raw_lat)
                    lat_scale_applied = lat_scale_applied or raw_lat != lat
                if lon_idx < len(row) and row[lon_idx].strip():
                    raw_lon = float(row[lon_idx].strip())
                    lon = _normalize_longitude(raw_lon)
                    lon_scale_applied = lon_scale_applied or raw_lon != lon
                if alt_idx < len(row) and row[alt_idx].strip():
                    raw_alt = float(row[alt_idx].strip())
                    alt = _normalize_altitude(raw_alt)
                    alt_scale_applied = alt_scale_applied or raw_alt != alt
            except (ValueError, DataError) as exc:
                logger.warning(f"Skipping malformed row at line {line_num} in {path.name}: {exc}")
                skipped_count += 1
                continue

            raw_dict: dict[str, str] = {}
            for col_i, cell in enumerate(row):
                key = headers[col_i] if col_i < len(headers) else f"col_{col_i}"
                raw_dict[key] = cell.strip()

            # Traceability: record the normalisation applied to this row.
            raw_dict["_timestamp_us"] = str(timestamp_us)
            raw_dict["_timestamp_unit_detected"] = timestamp_unit or "unknown"
            if lat_scale_applied:
                raw_dict["_lat_scale"] = f"1/{GPS_LAT_LON_SCALE:.0e}"
            if lon_scale_applied:
                raw_dict["_lon_scale"] = f"1/{GPS_LAT_LON_SCALE:.0e}"
            if alt_scale_applied:
                raw_dict["_alt_scale"] = f"1/{GPS_ALTITUDE_SCALE:.0e}"

            records[image_id] = ZurichGpsRecord(
                timestamp_us=timestamp_us,
                image_id=image_id,
                lat=lat,
                lon=lon,
                alt=alt,
                raw_fields=raw_dict,
            )

    if lat_scale_applied or lon_scale_applied or alt_scale_applied:
        logger.info(
            f"{path.name}: normalised fixed-point GPS encoding "
            f"(lat/lon 1E7 applied: {lat_scale_applied}/{lon_scale_applied}, alt 1E3 applied: {alt_scale_applied})"
        )

    logger.info(f"Loaded {len(records)} GPS records from {path.name} ({skipped_count} rows skipped)")
    return records, skipped_count


def parse_ground_truth_agl(csv_path: Path | str) -> tuple[dict[int, ZurichGroundTruthRecord], int]:
    """Parse GroundTruthAGL.csv (image_id -> XYZ in WGS 84 / UTM zone 32N metres).

    STRICTLY FOR EVALUATION ONLY. Must never be used as input to SfM.

    Returns:
        tuple of (gt_records_by_image_id, skipped_row_count)
    """
    path = Path(csv_path)
    if not path.exists():
        raise DataError(f"Ground truth file not found: {path}")

    records: dict[int, ZurichGroundTruthRecord] = {}
    skipped_count = 0

    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        headers, first_data_row = _read_header_and_first_data_row(reader, path, _POSITIONAL_GT_COLUMNS)
        logger.info(f"Parsing Ground Truth (EVALUATION ONLY) {path.name} with columns: {headers}")

        img_idx = _column_index(headers, ("image", "img", "frame", "id"), 0)
        x_idx = _column_index(headers, ("x",), 1)
        y_idx = _column_index(headers, ("y",), 2)
        z_idx = _column_index(headers, ("z", "alt", "height"), 3)

        rows = [first_data_row, *reader] if first_data_row is not None else list(reader)

        line_num = 1
        for row in rows:
            line_num += 1
            if not row or all(not cell.strip() for cell in row):
                continue
            if row[0].lstrip().startswith("#"):
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
