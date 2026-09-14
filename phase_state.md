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
pytest: 33 passed.
Real-data inspect summary:
- Source: zurich_urban_mav_zurich_mav (50 frames in slice, 1920x1080)
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
### Real-data Results
considered=50, selected=25, final stride=2, blur median/p95=422.6/428.5 (min: 409.3), ORB features median/p95=172/178, clipping median/p95=0.28%/0.28%
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
### Real-data Results
models=1, registered=25/25 (100.0%), mean reproj err=0.588 px, runtime=19.25s, peak RSS=6486.5 MB
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
### Real-data Results
matched=25/25, sync_status=dataset_image_id_mapping, GPS range lat/lon=lat [47.376900, 47.377620], lon [8.541700, 8.542564], alt [450.00, 454.80] m
### Remaining Tasks
Phase 5 ATE validation.
### Current Blockers
<none>
### Next Actions
Phase 5: Umeyama ATE + SP-1 report.


