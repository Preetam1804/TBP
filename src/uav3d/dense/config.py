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

    # --- runtime ------------------------------------------------------------
    keep_depth_in_memory: bool = Field(
        default=True,
        description="Keep depth maps in RAM for the consistency pass (halves disk I/O)",
    )
