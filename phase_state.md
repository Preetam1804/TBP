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

## Phase 5 Completed: 2026-09-14 — SP-1 COMPLETE
### Completed
Umeyama ATE validator with condition guard; reports (JSON, plots, SP1_REPORT.md);
'uav3d validate' CLI; real-data validation run.
### Files Created
validation/{ate,reports,__init__}.py, models/validation.py, tests/test_ate.py
### Real-data Results (KPI vs target)
registration=100.0% (target ≥85%); mean reproj=0.588px (target ≤1.5);
ATE RMSE=1.2564m / median=1.1499m / p95=2.3486m; scale=7.6738; fit_condition=weak_flightline_geometry;
matched=25/25; data label: replayed public dataset (Zurich Urban MAV) — not field-flight validation.
### Known Limitations
rolling shutter uncorrected; GPS multipath; stride-based keyframing only.
### Remaining Tasks
SP-2 handoff: depth (SP-2 consumes poses.json + keyframes.json).
### Current Blockers
<none>
### Next Actions
Bug-fix buffer / capture-guidance notes; then SP-2 Phase A (DepthAnything-V2-Small).

## SP-1.5 Completed: 2026-09-21 — Single-pass `reconstruct` + COLMAP hang fix
### Completed
- New `uav3d reconstruct` subcommand: video → keyframes → SfM → 3D model in one pass.
  Video-only input (`--video`; no telemetry/GT). Auto PINHOLE intrinsics
  (fx=fy=0.9*width, overridable via --calibration-file). Sync/ATE stages skip cleanly (INFO).
- Fix for the ~2.5 h hang on the user's field video:
  * `sfm.multiple_models: false` (new default) — COLMAP stops after the main
    reconstruction instead of burning hours re-attempting secondary-model
    initialization over leftover frames ("Discarding reconstruction due to
    insufficient size" tail eliminated; verified 0 discard lines in native log).
  * `sfm.init_num_trials: 200` (new, pipeline-level) caps init attempts.
  * `--max-frames N` is a hard budget: ceil pre-stride before quality evaluation,
    enforced in both `keyframes` and `reconstruct`.
- Native COLMAP glog (fd 2) captured to `colmap_native.log` via fd-2 redirect
  (sfm/native_log.py). Env-var log control (GOOGLE_LOG_LEVEL) verified NOT to work
  on pycolmap 3.12.6; Python logging unaffected.
- 3D model export: model.ply (colored points), model/ (COLMAP text model),
  trajectory.ply (camera path, vertices+edges, frame-id order),
  reconstruction_summary.json (versioned manifest, schema 1.0.0).
### Files Created
modeling/{__init__,export}.py, models/reconstruction.py, sfm/native_log.py,
tests/test_reconstruct.py (6 tests)
### Files Modified
cli.py (reconstruct subcommand/handler; --max-frames on keyframes;
_resolve_calibration/_find_ground_truth helpers), sfm/runner.py (option plumbing +
native-log capture + default log path + 0-registered-frame diagnostics),
keyframes/selector.py (frame_budget param + per-frame seek metrics),
configs/zurich_baseline.yaml, tests/conftest.py (parallax_video fixture),
tests/test_schemas.py (hermetic smoke runs; no developer data needed),
tests/test_keyframes.py (missing `import pytest` — pre-existing NameError),
README.md (usage).
### Verification (sandbox, CPU only)
uv run pytest: 72 passed, 3 skipped (all green).
Live smoke: 60-frame 640x360 parallax video, --max-frames 60 →
30/30 registered (100.0%), mean reproj 0.658 px, 180 3D points, 1 model,
0 "discard" lines in colmap_native.log, wall 6.4 s, all artifacts present.
### Decisions Made
- multiple_models=False default; config-exposed for power users.
- Trajectory PLY is a polyline (n vertices + n-1 edges) ordered by frame id,
  not a point cloud.
- MVS (patch_match_stereo) not wired: requires CUDA; dense mesh stays SP-2.
- Budgeted keyframes too sparse for init → fail fast with actionable error
  (no silent re-stride).
### Known Limitations
- Monocular scale unknown until metric alignment (documented in manifest).
- Windows run: user's local copy has diverged (SP-2 depth stages); they must
  run THIS branch for the fixes.
### Remaining Tasks
SP-2 (Depth Anything V2-Small) can consume poses.json + keyframes.json unchanged.
### Current Blockers
<none>
### Next Actions
Hand the exact command to the user; await real-video results.
