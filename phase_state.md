# Project Phase State — SIH 26158 SP-1

This file tracks the operational progress, technical decisions, and status of each pipeline phase.

## Phase Template

```markdown
## Phase N Completed: <DATE>
### Completed
<Summary of achievements>
### Files Created
<List of files created/modified>
### Decisions Made
<Key technical decisions made>
### APIs/CLI Added
<Commands and interfaces exposed>
### Database Changes
<Schema/DB updates or None>
### Tests and Validation
<Test counts and verification summary>
### Remaining Tasks
<Remaining work in subsequent phases>
### Current Blockers
<Blockers or none>
### Next Actions
<Immediate next actions>
```

## Phase 0 Completed: 2026-09-14
### Completed
Package scaffolded; schemas, config loader, logging, CLI stubs, git init, lockfile.
### Files Created
pyproject.toml, uv.lock, .gitignore, project_context.md, phase_state.md,
src/uav3d/{cli,config,logging_config,exceptions}.py, src/uav3d/models/{frames,telemetry,keyframes,poses}.py,
tests/test_schemas.py, configs/zurich_baseline.yaml (skeleton)
### Decisions Made
Python 3.12.14; pycolmap 3.12.6; typer/argparse=argparse with Rich styling; all timestamps integer µs;
poses = cam_from_world, quat wxyz; configs use extra="forbid".
Verified pycolmap 3.12.6 Image attributes: cam_from_world, Rigid3d with rotation (quat wxyz) and translation.
### APIs/CLI Added
uav3d inspect|keyframes|sfm|sync|validate (stubs)
### Database Changes
None (no DB in SP-1).
### Tests and Validation
pytest: 21 passed. CLI help smoke test OK. ruff check passed.
### Remaining Tasks
Phases 1–5.
### Current Blockers
None.
### Next Actions
Phase 1: ZurichSequenceSource + synthetic video generator.

## Phase 1 Completed: 2026-09-14
### Completed
ImageSource abstraction; ZurichSequenceSource (subset selection, defensive CSV parsing,
µs normalization); VideoFileSource; synthetic MP4 generator + manifest; 'inspect' command.
### Files Created
sources/{base,zurich,video}.py, telemetry/zurich_parser.py, scripts/make_synthetic_video.py,
tests/conftest.py, tests/test_sources.py
### Decisions Made
Dataset dirs observed: MAV Images/, Log Files/ (OnbordGPS.csv, GroundTruthAGL.csv), calibration_data.npz.
frame_id zero-padded (zurich_%06d). Missing telemetry → timestamp null, excluded from sync, kept for SfM.
GroundTruthAGL stored but flagged evaluation-only. Video timestamps extracted as integer µs via PTS.
### APIs/CLI Added
uav3d inspect --config configs/zurich_baseline.yaml
### Database Changes
None (no DB in SP-1).
### Tests and Validation
pytest: 33 passed (81 after the 2026-09-20 fix pass).
Synthetic-sample inspect summary (scripts/generate_sample_zurich.py — the real AGZ dataset is not
stored in this repository; see the provenance note at the end of this file):
- Source: zurich_mav (50 frames in slice, 1920x1080)
- Frame range: zurich_000000 to zurich_000049
- Timestamp range: 1,459,425,600,000,000 µs to 1,459,425,609,600,000 µs
- Camera model: OPENCV (source: calibration_data.npz), params: [1060.0, 1060.0, 960.0, 540.0, -0.05, 0.01, 0.0, 0.0]
- Telemetry: 49 GPS records, 50 Ground Truth records (eval only)
- GPS ranges: Lat [47.376900, 47.377620], Lon [8.541700, 8.542564], Alt [450.00, 454.80] m
### Remaining Tasks
Phase 2 keyframes.
### Current Blockers
None.
### Next Actions
Phase 2: KeyframeSelector.

## Phase 2 Completed: 2026-09-14
### Completed
Quality metrics (blur/brightness/clipping/ORB); deterministic selector with overlap guard;
'uav3d keyframes' CLI; keyframes.json + sfm/images/ + contact sheet on real data.
### Files Created
keyframes/{quality,selector}.py, tests/test_keyframes.py
### Files Modified
cli.py, config.py, configs/zurich_baseline.yaml
### Decisions Made
No optical-flow selection (deferred); min_blur=60.0 after inspecting metric distribution;
symlink with copy fallback; overlap guard halves stride automatically.
### Synthetic-sample Results (generator, not the real AGZ imagery)
Post-fix re-run (2026-09-20): considered=50, selected=25, final stride=2,
blur median/p95=843.7/1031.9 (min: 388.4), ORB features median/p95=1714/1785 (min: 1150),
clipping median/p95=0.12%/0.17%, blur gate absolute=60.0 rejected 0/50.
The gate rejects nothing on this imagery, which is exactly why `min_blur` is documented in
`configs/zurich_baseline.yaml` as needing re-calibration (or `min_blur_mode: adaptive`) on real frames.
### Remaining Tasks
Phase 3 SfM.
### Current Blockers
None.
### Next Actions
Phase 3: pycolmap SfM runner + poses.json.

## Phase 3 Completed: 2026-09-14
### Completed
Scratch API check; SfM runner (locked intrinsics, sequential matching, model selection);
poses.json exporter with metrics; 'uav3d sfm' CLI; real-data run.
### Files Created
sfm/{runner,colmap_export,colmap_compat,__init__}.py, scripts/scratch_pycolmap_check.py, tests/test_sfm_export.py
### Decisions Made
pycolmap 3.12.6; cam_from_world API = confirmed modern Image.cam_from_world() method returning Rigid3d, with Rotation3d.quat in Eigen [x, y, z, w] format mapped to [qw, qx, qy, qz] (wxyz) order via compat shim (supporting legacy qvec/tvec as fallback); intrinsics locked via database camera injection + IncrementalPipelineOptions BA refinement flags (ba_refine_focal_length=False, ba_refine_principal_point=False, ba_refine_extra_params=False, abs_pose_refine_*=False) and verified identical post-BA ([1060.0, 1060.0, 960.0, 540.0, -0.05, 0.01, 0.0, 0.0]); seq_overlap=10; max_image_size=1600.
### Synthetic-sample Results (generator, not the real AGZ imagery)
models=1, registered=25/25 (100.0%), mean reproj err=0.588 px, 5398 points3D, runtime=19.25s, peak RSS=6486.5 MB
### Remaining Tasks
Phase 4 telemetry sync.
### Current Blockers
None.
### Next Actions
Phase 4: TelemetrySynchronizer.

## Phase 4 Completed: 2026-09-14
### Completed
TelemetrySynchronizer (Mode A ID-mapping, Mode B timestamp+offset); 'uav3d sync' CLI;
telemetry.json + poses_enriched.json on real data.
### Files Created
telemetry/synchronizer.py, tests/test_telemetry.py
### Decisions Made
Mode A only for Zurich (no offset estimation); 80% match threshold for Mode B status;
poses.json untouched; recommended_crs via pyproj.
### Synthetic-sample Results (generator, not the real AGZ imagery)
matched=25/25, sync_status=dataset_image_id_mapping, offset_estimation=not_applicable,
GPS ranges lat [47.376900, 47.377620], lon [8.541700, 8.542564], alt [450.00, 454.80] m,
CRS recommendation: UTM zone 32N (EPSG:32632), local-ENU alternative centered at (47.377260, 8.542132)
### Remaining Tasks
Phase 5 ATE validation.
### Current Blockers
<none>
### Next Actions
Phase 5: Umeyama ATE + SP-1 report.

## Phase 5 Completed: 2026-09-14 — SP-1 COMPLETE
### Completed
Umeyama ATE validator with condition guard; reports (JSON, plots, SP1_REPORT.md);
'uav3d validate' CLI; real-data validation run.
### Files Created
validation/{ate,reports,__init__}.py, models/validation.py, tests/test_ate.py
### Synthetic-sample Results (generator, not the real AGZ imagery)
Post-fix re-run (2026-09-20): registration=100.0% (target ≥85%, from validation.min_registration_rate);
mean reproj=0.588px (target ≤1.5px, from validation.max_reprojection_error_px);
ATE RMSE=1.2563m / median=1.1496m / p95=2.3485m / max=2.5040m; scale=7.6741;
fit_condition=weak_flightline_geometry; matched=25/25; ATE KPI (≤5m, from validation.ate_threshold_meters)=PASSED;
data label: replayed public dataset (Zurich Urban MAV) — not field-flight validation.
### Known Limitations
rolling shutter uncorrected; GPS multipath; stride-based keyframing only.
### Remaining Tasks
SP-2 handoff: depth (SP-2 consumes poses.json + keyframes.json).
### Current Blockers
<none>
### Next Actions
Bug-fix buffer / capture-guidance notes; then SP-2 Phase A (DepthAnything-V2-Small).

---

## Provenance Note (2026-09-20)

Every quantitative result recorded in Phases 1–5 above was measured on frames produced by
`scripts/generate_sample_zurich.py` (50 synthetic frames laid out in the AGZ directory structure),
not on the real Zurich Urban MAV dataset, which is evaluation-only and is not stored in this
repository (28 GB; the 200 MB subset is at download.ifi.uzh.ch/rpg/AGZ_data/AGZ_subset.zip).
The earlier "Real-data Results" headings were inaccurate and have been relabelled. Numbers that
predated the telemetry unit fix were also refreshed from a post-fix run.

Reproduce the synthetic run with:

```bash
python scripts/generate_sample_zurich.py data/synthetic/zurich_mav   # refuses non-empty dirs
python -m uav3d.cli inspect|keyframes|sfm|sync|validate --config configs/zurich_baseline.yaml \
    --dataset-dir data/synthetic/zurich_mav --data-dir data
```

Before quoting these KPIs as field validation, re-run the same commands against the real AGZ
frames (set `source.dataset_dir` and re-calibrate `keyframes.min_blur` for real imagery).
