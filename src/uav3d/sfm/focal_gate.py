"""Focal-drift gate for SfM on forward flight.

Forward flight has a fundamental focal-length / depth ambiguity: image
velocity is ``f * d_inv``, so focal length and scene depth trade off almost
freely along the flight axis. With ``refine_intrinsics=True`` COLMAP's bundle
adjustment is free to spend that ambiguity on the focal length — on the
city flyover it solved f ~ 1159 px against a 1024 px prior (+13%), which
absorbs a depth stretch into the camera instead of the geometry.

This module is the gate the handoff asked for: it compares the focal length
COLMAP actually solved with the focal length the run started from and flags
(never silently clamps) a drift beyond the configured tolerance. The caller
decides what a flagged drift means — warn and record it, or fail the run —
but the drift is always visible in the result, the CLI table and the pipeline
summary.
"""

from __future__ import annotations

from dataclasses import dataclass

# COLMAP camera models whose first parameter is the single shared focal,
# and models whose first two parameters are (fx, fy).
_SINGLE_FOCAL_MODELS = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}
_FX_FY_MODELS = {"PINHOLE", "OPENCV", "FULL_OPENCV"}


def describe_focal_source(source: str) -> str:
    """Human-readable origin of a prior focal, for auditability in reports.

    The ``CameraCalibration.source`` string records the origin; this turns it
    into the short label the SfM summary and the pipeline report print next
    to the focal-drift gate verdict, so a locked-focal run is auditable:

    - ``"video_config_override"`` -> config/CLI override (highest precedence)
    - ``"video_meta_sidecar:<file>"`` -> sidecar ``<video-stem>.meta.json``
    - ``"video_estimate_<file>"`` -> the 0.8*width guess (lowest precedence)

    Any other source string (e.g. a Zurich calibration file) is returned
    unchanged.
    """
    if source == "video_config_override":
        return "config/CLI override"
    if source.startswith("video_meta_sidecar:"):
        return f"sidecar meta.json ({source.split(':', 1)[1]})"
    if source.startswith("video_estimate"):
        return "0.8*width guess"
    return source


def solved_focal_px(model: str, params: list[float]) -> float:
    """Extract the (mean) focal length in pixels from a COLMAP parameter vector.

    SIMPLE_PINHOLE / SIMPLE_RADIAL / RADIAL store one shared focal as
    ``params[0]``; PINHOLE / OPENCV / FULL_OPENCV store ``(fx, fy, ...)`` and
    the mean is the focal the dense stage actually uses
    (``Intrinsics.focal`` is ``0.5 * (fx + fy)``).
    """
    if not params:
        raise ValueError("camera params are empty; cannot read a focal length")
    name = model.upper()
    if name in _SINGLE_FOCAL_MODELS:
        return float(params[0])
    if name in _FX_FY_MODELS:
        if len(params) < 2:
            raise ValueError(f"{model} needs at least fx and fy, got {params!r}")
        return 0.5 * (float(params[0]) + float(params[1]))
    raise ValueError(f"unsupported camera model for focal extraction: {model!r}")


def focal_drift_rel(prior_focal_px: float, solved_focal_px_value: float) -> float:
    """Relative focal drift ``(solved - prior) / prior`` (signed)."""
    if prior_focal_px <= 0:
        raise ValueError(f"prior focal must be positive, got {prior_focal_px}")
    return (float(solved_focal_px_value) - float(prior_focal_px)) / float(prior_focal_px)


@dataclass(frozen=True)
class FocalDriftResult:
    """Outcome of the focal-drift gate for one SfM run."""

    prior_focal_px: float
    solved_focal_px: float
    rel_drift: float
    max_rel_drift: float
    exceeded: bool

    @property
    def message(self) -> str:
        """One-line, human-readable verdict for logs and reports."""
        direction = "grew" if self.rel_drift >= 0 else "shrank"
        verdict = "EXCEEDS" if self.exceeded else "within"
        return (
            f"solved focal {self.solved_focal_px:.1f} px {direction} "
            f"{100.0 * abs(self.rel_drift):.1f}% from prior "
            f"{self.prior_focal_px:.1f} px "
            f"({verdict} the {100.0 * self.max_rel_drift:.0f}% gate)"
        )


def check_focal_drift(
    prior_focal_px: float,
    solved_focal_px_value: float,
    max_rel_drift: float = 0.10,
) -> FocalDriftResult:
    """Compare a solved focal length against its prior and apply the gate.

    ``max_rel_drift`` is the largest absolute relative drift that passes.
    The gate never clamps or rewrites the focal — it only reports.
    """
    if max_rel_drift < 0:
        raise ValueError(f"max_rel_drift must be >= 0, got {max_rel_drift}")
    rel = focal_drift_rel(prior_focal_px, solved_focal_px_value)
    return FocalDriftResult(
        prior_focal_px=float(prior_focal_px),
        solved_focal_px=float(solved_focal_px_value),
        rel_drift=float(rel),
        max_rel_drift=float(max_rel_drift),
        exceeded=bool(abs(rel) > float(max_rel_drift)),
    )
