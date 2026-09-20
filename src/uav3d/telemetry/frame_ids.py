"""Helpers for mapping pipeline frame ids onto dataset telemetry image ids.

Frame ids are ``<prefix>_<zero-padded index>`` (e.g. ``zurich_000042``, ``video_000007``).
Joining them to telemetry by "trailing digits" is only safe while the derived numeric ids are
unique and belong to a single prefix, so the join is validated before it is used instead of
silently pairing unrelated frames (e.g. ``left_000012`` with GPS row 12 of another sequence).
"""

import re

from uav3d.exceptions import DataError

_NUMERIC_SUFFIX = re.compile(r"^(.+?)_(\d+)$")


def numeric_suffix(frame_id: str) -> int | None:
    """Return the trailing numeric index of a frame id, or None if it has none."""
    match = _NUMERIC_SUFFIX.match(frame_id)
    return int(match.group(2)) if match else None


def frame_prefix(frame_id: str) -> str | None:
    """Return the prefix part of a frame id (everything before the numeric suffix)."""
    match = _NUMERIC_SUFFIX.match(frame_id)
    return match.group(1) if match else None


def validate_frame_id_mapping(frame_ids: list[str] | tuple[str, ...], *, context: str) -> dict[int, str]:
    """Validate that numeric suffixes uniquely identify frames and return numeric_id -> frame_id.

    Raises:
        DataError: if any frame id lacks a numeric suffix, if two frame ids map to the same
            number, or if more than one prefix is present (ambiguous joining).
    """
    mapping: dict[int, str] = {}
    prefixes: dict[str, str] = {}

    for frame_id in frame_ids:
        number = numeric_suffix(frame_id)
        if number is None:
            raise DataError(
                f"{context}: frame id '{frame_id}' has no numeric suffix; telemetry cannot be joined safely"
            )
        if number in mapping and mapping[number] != frame_id:
            raise DataError(
                f"{context}: frame ids '{mapping[number]}' and '{frame_id}' both map to telemetry id "
                f"{number}; frame ids must be unique per numeric index"
            )
        mapping[number] = frame_id
        prefix = frame_prefix(frame_id)
        if prefix is not None:
            prefixes.setdefault(prefix, frame_id)

    if len(prefixes) > 1:
        listed = ", ".join(sorted(prefixes))
        raise DataError(
            f"{context}: mixed frame id prefixes ({listed}); refusing to join telemetry rows by "
            "numeric suffix across unrelated sequences"
        )

    return mapping
