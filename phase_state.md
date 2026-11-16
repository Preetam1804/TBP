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

## Accuracy completion 2026-11-16 (post-SP-1: focal gate, matcher, sky guard, coverage)
### Completed
- Phase 1 sky guard for the monocular-prior fill (`dense/prior.py::sky_mask`, `DenseConfig.prior_sky_guard`, default True) — PR #5.
- Phase 2 focal-drift gate (`src/uav3d/sfm/focal_gate.py`, `SfmConfig.focal_drift_max_rel=0.10`, `focal_drift_action=warn`): flags solved-vs-prior focal drift, never clamps — PR #5.
- Phase 3 matcher wiring (`SfmRunner.match_features()`): `sequential` (default) vs `exhaustive` loop-closure mode; unknown matcher raises — PR #5.
- Two stale-cache bugs fixed: rectified-geometry cache rebuilds on intrinsics mismatch; `load_sparse_points` prefers the fresh COLMAP model — PR #5.
- Phase 4 coverage reporting (this session): `src/uav3d/dense/coverage.py` — per-view valid-depth fractions (measured vs inferred), fused density, PCA-footprint grid with hole/thin cells, `coverage_report.json` + DENSE_REPORT.md section + `dense_report.json` key, surfaced in `run_video_pipeline.summarise()` with a WARNING gate.
- Loop-closure experiment (this session): synthetic 104-frame loop/return flight; exhaustive matcher did NOT help over sequential on a continuous revisit (sparse median 0.457 m vs 0.456 m, 2.4x wall time). Free focal recovered the true focal (264.6 vs 265 px) thanks to viewpoint variation.
- Focal locked-vs-free decision (this session): keep `refine_intrinsics: true` in `configs/video_flyover.yaml`. On pure forward flight neither setting is accurate without a true focal (free 2.9-3.5 m, locked-wrong 3.1 m median sparse error); only the true focal locked gives a sub-metre model (0.49 m). Free refinement is unstable across runs (367-492 px solved) and once collapsed registration to 28/60 with 9 models.
- Full-pass end-to-end (this session): `run_video_pipeline.py` on the full 60-frame synthetic flyover (`data/outputs/run_20261116_093559`, 193 s): 60/60 registered, reproj 0.281 px, solved focal 436 px (+13.5%, gate exceeded), dense 54.0% valid, 396,554 fused points, mesh 983,840 verts / 1,501,121 tris, coverage verdict good (0/45 cells empty). Scored vs GT: fused median accuracy 3.61 m, prec@0.5 m 0.000 — the dense stage faithfully densified a focal-stretched reconstruction.
- Reporting-integrity fix: SfM summary "Quality Gate Passed" no longer prints when registration/reprojection/focal gates fail (`sfm_gate_failures()`, tested).
- Test-suite data skips: the 5 pre-existing data-dependent failures now skip gracefully when the Zurich dataset / historical run dir is absent (missing `pytest` import fixed; smoke test pinned to the historical run dir).
### Files Created
src/uav3d/dense/coverage.py, tests/test_coverage.py, scripts/render_loop_video.py, docs/loop_focal_experiments.md
### Files Modified
src/uav3d/dense/{config,pipeline}.py, src/uav3d/models/dense.py, scripts/run_video_pipeline.py, src/uav3d/cli.py, tests/test_{keyframes,schemas,sfm_export}.py, docs/HANDOFF_NEXT_SESSION.md (§13), README.md, phase_state.md (this section)
### Decisions Made
- `refine_intrinsics` stays `true` (evidence-based, §13.3 of the handoff). Monocular prior stays opt-in (failed promotion gate).
- Coverage verdict is grid-driven, not per-view-driven; footprint plane is PCA PC1-PC2, labelled as such; units are SfM units until scale is verified.
- `city_textured.glb` is not in this clone (untracked, on the user's
  machine) and was not touched; it remains the best measured city model.
  Synthetic-flyover GLBs live under `data/outputs/run_20261116_093559/`.
### Tests and Validation
Full suite: **201 passed, 6 skipped** (the 6 skips are the
Zurich-dataset smoke tests, absent from a fresh clone; 189 passed / 6
skipped excluding the two open3d-only modules). The open3d modules
(`test_completion.py`, `test_lod2.py`, 12 tests) pass in this container
only after installing system `libegl1` — on a container without it,
exclude them as before.
### Remaining Tasks
Real drone video + true focal for real-data validation (user-dependent). Discontinuous-revisit loop closure untested. Monocular prior stays opt-in.
### Current Blockers
No real drone video and no true focal in this checkout — every accuracy number above is synthetic ground truth only.
### Next Actions
User supplies a real single-pass drone video + focal (EXIF / sensor spec / calibration capture); re-run the full pass and score against any available ground truth.

## City focal determination + full-pass runs 2026-11-16 (post-SP-1)
### Completed
- Known-focal path: CLI `--focal-px` > config `source.focal_px` > sidecar `<stem>.meta.json` > 0.8*width guess; focal source recorded in `calibration.json` and printed with the gate verdict. 40 new tests (`tests/test_known_focal.py`).
- Focal of `city_flyover.mp4` verified: free-focal SfM brackets 1152 px (1112-1163, mean ~1138); locked 1152 chosen.
- Full 600-frame pass, locked vs free focal: both 250/250 registered, reproj 0.369 px, ~2.98M fused points, textured GLBs (95%+ coloured); free refinement moved focal only +0.4% from the correct prior. Trajectory vs GT positions (Sim(3), eval only): RMS ~0.10 m both runs.
- GT verdict: `gt_model.ply` / `gt_scene.json` do NOT match this video (box-edge projection at chance level; PLY renders show a different city layout) — no cloud-to-GT scoring for this video. The 0.66 m figure in the old city analysis used a method `CITY_MODEL_EVAL.md` declares meaningless.
### Files Created
tests/test_known_focal.py; docs/HANDOFF_NEXT_SESSION.md (§14)
### Files Modified
src/uav3d/{config.py,sources/video.py,cli.py,sfm/focal_gate.py}, scripts/run_video_pipeline.py, configs/video_flyover.yaml (comment only), README.md, docs/HANDOFF_NEXT_SESSION.md (§14), phase_state.md (this section)
### Decisions Made
- Locked focal 1152 px for `city_flyover.mp4` (evidence-based, handoff §14.2). `refine_intrinsics` stays `true` in the tracked config; locked runs use a `/tmp` config copy.
- Sidecar discovery is next-to-video only; the city meta lives in `data/raw/test_videos/`, so it did not supply the prior — explicit override was used.
- No cloud-to-GT accuracy claim for this video; trajectory-vs-GT-positions + internal consistency only.
### Tests and Validation
Full suite: **241 passed, 6 skipped** (including the two open3d-only modules, after installing system `libegl1` in this container).
### Remaining Tasks
Render a GT-scored benchmark video from `gt_model.ply` with a known camera (the honest way to get a cloud-to-GT number), or validate on real drone footage with a real focal. Coverage verdict on the city runs is "holes" (14/90 footprint cells empty).
### Current Blockers
No cloud-to-GT ground truth that matches `city_flyover.mp4`; absolute metric scale unverified (monocular).
### Next Actions
User decides whether to push/merge branch `accuracy-completion` (commit ee20838 + this uncommitted work); render-from-GT benchmark if a GT-scored claim is needed.
