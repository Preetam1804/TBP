"""Configuration for the CPU dense multi-view-stereo stage."""

from pydantic import BaseModel, ConfigDict, Field


class DenseConfig(BaseModel):
    """Tunables for CPU plane-sweep MVS, fusion and meshing.

    Defaults are sized for a 150-frame 1080p UAV clip on a 2-core CPU: roughly
    30-50 minutes end to end, which is inside the one-hour budget.
    """

    model_config = ConfigDict(extra="forbid")

    # --- working resolution -------------------------------------------------
    max_width: int = Field(
        default=1600,
        gt=0,
        description="Working image width; frames are undistorted and resized to this width",
    )
    ref_stride: int = Field(
        default=1,
        ge=1,
        description="Use every Nth registered frame as a depth-map reference (2 halves runtime)",
    )
    max_reference_views: int | None = Field(
        default=None,
        gt=0,
        description="Optional cap on the number of reference views processed",
    )

    # --- view selection -----------------------------------------------------
    source_window: int = Field(
        default=12,
        ge=1,
        description="Candidate source frames considered on each side of a reference frame",
    )
    num_sources: int = Field(
        default=5,
        ge=2,
        le=16,
        description="Number of source views fused per reference view",
    )
    min_baseline_ratio: float = Field(
        default=0.015,
        gt=0.0,
        description="Minimum baseline/depth ratio for a source view to be considered",
    )
    max_baseline_ratio: float = Field(
        default=0.30,
        gt=0.0,
        description="Maximum baseline/depth ratio; beyond this, view overlap collapses",
    )
    max_view_angle_deg: float = Field(
        default=45.0,
        gt=0.0,
        le=90.0,
        description="Maximum angle between reference and source optical axes",
    )
    min_overlap: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Minimum predicted frustum overlap for a source view",
    )

    # --- plane sweep --------------------------------------------------------
    window: int = Field(default=5, ge=3, description="ZNCC aggregation window (pixels, odd)")
    plane_step_px: float = Field(
        default=0.7,
        gt=0.0,
        description="Target disparity spacing between depth planes in pixels",
    )
    min_planes: int = Field(default=32, ge=8, description="Lower bound on the number of depth planes")
    max_planes: int = Field(default=160, ge=8, description="Upper bound on the number of depth planes")
    cost_threshold: float = Field(
        default=0.40,
        gt=0.0,
        le=2.0,
        description="ZNCC cost above which a depth pixel is discarded (0 = perfect match)",
    )
    min_views: int = Field(
        default=2,
        ge=1,
        description="Minimum number of source views that must agree before a pixel is valid",
    )
    depth_percentile_low: float = Field(
        default=2.0,
        ge=0.0,
        le=50.0,
        description="Lower percentile of sparse-point depths used as the near plane",
    )
    depth_percentile_high: float = Field(
        default=98.0,
        ge=50.0,
        le=100.0,
        description="Upper percentile of sparse-point depths used as the far plane",
    )
    depth_margin: float = Field(
        default=0.25,
        ge=0.0,
        description="Fractional margin applied outside the sparse depth percentiles",
    )
    depth_near_min_ratio: float = Field(
        default=0.28,
        gt=0.0,
        lt=1.0,
        description=(
            "Floor for the near plane as a fraction of the median sparse depth. "
            "SfM outlier tracks triangulate a handful of points almost on top of "
            "the camera; their 1/z^2 coverage weight then drags the 2nd-percentile "
            "near plane to near-zero and starves the plane sweep of resolution in "
            "the range the camera actually sees. Keeping the near plane at least "
            "this far from the median makes every plane budget defensible."
        ),
    )
    depth_near_plane_relax: float = Field(
        default=1.5,
        ge=1.0,
        description=(
            "When the required plane count exceeds max_planes the near plane is "
            "moved outward until the disparity step is at most "
            "plane_step_px * depth_near_plane_relax. This guarantees the sweep "
            "never silently steps wider than intended."
        ),
    )
    depth_range: str | None = Field(
        default=None,
        description="Manual override 'min:max' in SfM units, applied to every view",
    )

    max_relative_sigma: float = Field(
        default=0.02,
        ge=0.0,
        description=(
            "Reject pixels whose predicted depth sigma exceeds this fraction of depth. "
            "0 disables the gate. Bounds how far away depth is still measurable at all."
        ),
    )

    # --- multi-view consistency --------------------------------------------
    consistency_tol: float = Field(
        default=0.02,
        gt=0.0,
        description="Relative depth tolerance for cross-view geometric consistency",
    )
    consistency_views: int = Field(
        default=2,
        ge=1,
        description="Number of neighbouring depth maps that must agree on a point",
    )

    # --- monocular depth prior (opt-in) -------------------------------------
    prior_enabled: bool = Field(
        default=False,
        description=(
            "Fill MVS-invalid pixels with a Depth Anything V2 prior aligned to the SfM frame. "
            "Only pixels rejected by the MVS/consistency gates are filled, and only where "
            "neighbouring views' priors agree. Default off: measured MVS output is unchanged."
        ),
    )
    prior_model_path: str = Field(
        default="models/checkpoints/depth_anything_v2_vits.onnx",
        description="Path to the Depth Anything V2 ONNX checkpoint",
    )
    prior_infer_height: int = Field(
        default=518,
        gt=14,
        description="Internal inference height for the monocular model",
    )
    prior_min_rank_corr: float = Field(
        default=0.35,
        gt=0.0,
        le=1.0,
        description="Minimum rank correlation of the inverse-depth fit for a view's prior to be accepted",
    )
    prior_max_relative_error: float = Field(
        default=0.35,
        gt=0.0,
        description="Maximum median relative depth error of the fit for a view's prior to be accepted",
    )
    prior_min_support: int = Field(
        default=2,
        ge=1,
        description="Neighbouring views whose priors must agree before a hole pixel is filled",
    )
    prior_tolerance: float = Field(
        default=0.06,
        gt=0.0,
        description="Relative depth agreement tolerance between the prior and neighbouring priors",
    )
    prior_edge_guard_px: int = Field(
        default=1,
        ge=0,
        description="Distance from strong image edges where prior filling is forbidden",
    )
    prior_sky_guard: bool = Field(
        default=True,
        description=(
            "Forbid prior filling in sky pixels (bright, desaturated, smooth, in the "
            "top of the frame). A monocular model assigns the sky a smooth pseudo-depth "
            "that neighbouring priors agree on, so the agreement gate alone cannot "
            "tell sky from a real surface — without this guard a visible horizon "
            "becomes a solid sheet at the far plane."
        ),
    )
    prior_sky_max_row_fraction: float = Field(
        default=0.35,
        gt=0.0,
        le=1.0,
        description="Fraction of image height (from the top) in which sky pixels may occur",
    )

    # --- fusion -------------------------------------------------------------
    voxel_m: float | None = Field(
        default=None,
        gt=0.0,
        description="Metric fusion voxel; derived from the depth noise model when None",
    )
    trunc_sigmas: float = Field(
        default=2.5,
        gt=0.0,
        description="TSDF truncation in units of the estimated depth noise sigma",
    )
    outlier_k: int = Field(default=8, ge=1, description="Neighbours used for statistical outlier removal")
    outlier_std: float = Field(
        default=2.0,
        gt=0.0,
        description="Standard-deviation multiplier for statistical outlier removal",
    )
    radius_outlier_units: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional radius filter in SfM units (points with no neighbour inside are dropped)",
    )

    # --- meshing ------------------------------------------------------------
    mesh: bool = Field(default=True, description="Reconstruct a triangle mesh from the fused points")
    mesh_voxel_fraction: float = Field(
        default=1.5,
        gt=0.0,
        description="Mesh voxel size as a multiple of the fusion voxel",
    )
    mesh_tile_units: float | None = Field(
        default=None,
        gt=0.0,
        description="Spatial tile size for meshing (SfM units); derived when None",
    )
    mesh_max_tile_voxels: int = Field(
        default=6_000_000,
        gt=0,
        description="Memory guard: maximum voxels per meshing tile",
    )

    # --- metric scale -------------------------------------------------------
    scale_source: str = Field(
        default="none",
        description="Metric scale source: 'none' (SfM units), 'fixed', or 'gnss'",
    )
    scale_factor: float | None = Field(
        default=None,
        gt=0.0,
        description="Metric scale (metres per SfM unit) when scale_source is 'fixed'",
    )

    # --- coverage reporting (Phase 4) --------------------------------------
    # Thresholds are documented in uav3d/dense/coverage.py; changing them
    # changes the verdict gate in scripts/run_video_pipeline.py, so a run's
    # coverage_report.json always records the thresholds it was judged by.
    coverage_cell_size_m: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Grid cell size for hole/thin analysis, in metres when the scale "
            "is verified, otherwise in SfM units (scale unverified)"
        ),
    )
    coverage_thin_fraction: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="A grid cell holding less than this fraction of the mean per-cell point count is thin",
    )
    coverage_hole_fraction_warn: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Empty-cell fraction above this makes the coverage verdict 'thin'",
    )
    coverage_hole_fraction_bad: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Empty-cell fraction above this makes the coverage verdict 'holes'",
    )
    coverage_thin_fraction_bad: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Thin-cell fraction above this makes the coverage verdict 'thin'",
    )

    # --- runtime ------------------------------------------------------------
    keep_depth_in_memory: bool = Field(
        default=True,
        description="Keep depth maps in RAM for the consistency pass (halves disk I/O)",
    )
