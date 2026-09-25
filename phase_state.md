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

## 2026-09-22 — Taj Mahal field-video triage & fixes (branch feat/lod2-textured-3d-map)
### Symptoms Reported
- Only 19/98 keyframes registered → "25 m sliver" of the reflecting pool.
- Mesh rendered as spiky "toothbrush" (thousands of near-vertical needle triangles).
### Root Causes Found (run_20260922_024230)
1. **Degenerate focal runaway**: BA refined PINHOLE fx→21324 while fy stayed 1314
   (prior 972). Forward flight + independent fx/fy = classic focal/depth ambiguity;
   insane intrinsics froze PnP registration and COLMAP discarded a 29-frame model.
2. **Sequential-only matching**: runner.py ignored its own matcher_type config; video
   forward flight provides no loop closures, so the monument closeups never bridged
   back to the far view. Three fragments (19+13+16 frames) instead of one model.
3. **2.5D Delaunay meshing**: build_full_3d_map() triangulated ALL points in the
   ground-plane projection; facade tops ↔ ground ↔ water edges connected by
   near-vertical slivers (the "toothbrush"), despite a good Poisson mesh already
   existing in export.py.
### Fixes Implemented
- config.py: SfmConfig gains video_matcher_type (default exhaustive),
  exhaustive_max_images (120), loop_detection(+period), vocab_tree_path.
- sfm/runner.py: real matcher dispatch (_run_matching_stage): exhaustive /
  vocabtree / sequential+optional loop-closure; watermark TwoViewGeometryOptions
  preserved; stage profile named per matcher.
- cli.py `_resolve_calibration`: video runs now auto-configure SIMPLE_PINHOLE
  (shared focal: f, cx, cy) instead of PINHOLE — fx/fy cannot diverge.
- sfm/colmap_export.py + modeling/export.py: 3-param (SIMPLE_PINHOLE) camera
  support in projection/backprojection paths.
- modeling/completion.py build_full_3d_map: 2.5D Delaunay replaced by screened
  Poisson (scene-adaptive depth 8-11), normal estimation, density trimming (12%),
  small-component pruning, bbox crop 1.05, KDTree vertex coloring, no needles.
- configs/zurich_baseline.yaml: new sfm keys documented; keyframe budget note.
### Verification
- uv run pytest: 87 passed (2 initial regressions fixed: cluster_connected_triangles
  returns per-cluster sizes, not count; n=1 runs stay sequential).
- ruff: net-better than baseline in touched files; no new violations in new code.
- Full rerun on data/taj_mahal_drone_1080p.mp4 launched (--max-frames 120 so
  exhaustive matching engages; prior run selected 196 keyframes > 120 threshold).
### Notes
- refine_intrinsics stays True for video runs (SIMPLE_PINHOLE has only one focal;
  runaway is structurally prevented rather than clamped).

## 2026-09-22 — Dense-stage root cause + TSDF fusion (fan-of-blades fix)
### Symptom
Post-fix model still showed a fan of flat "blades" radiating from the flight path.
### Root Causes (3 bugs + 1 missing stage)
1. Branch gap: SP-2 depth/fusion machinery existed only on `main`; this branch's
   reconstruct fell back to naive per-frame backprojection (each keyframe = its
   own depth sheet; no multi-view averaging).
2. ONNX orientation: the exported depth_anything_v2_vits.onnx emits the shifted
   inverse-disparity pseudo-depth 1/(d+eps) (larger = FARTHER), not disparity.
   Affine-fitting it as disparity is mathematically wrong (empirically: spearman
   -0.77 vs 1/Z, +0.77 vs Z).
3. Legacy backprojection grid bug: linspace(0,W-1,w_sub) pixel centers mixed
   working-res coords with full-res cx/cy/f (points skewed ~300 px off-axis);
   also 518x518 squeeze distorted 1080x1920 aspect.
4. Quality gate self-confirmation: inlier-RMSE gate passed garbage; now requires
   inlier rank/Pearson correlation >= 0.7 OR fails.
### Fixes (src/uav3d/modeling/dense.py NEW; export.py dispatch + fallback grid fix)
- Robust disparity-space alignment: Theil-Sen + Huber IRLS (no sklearn dep),
  inlier-band RMSE gate + explains-variance correlation gate, median-scale fallback.
- Automatic model-output orientation detection per frame (rank correlation).
- Aspect-preserving inference (294x518 for 1080x1920, /14 ViT patching), exact
  pixel-center grids, per-frame log of method + rmse_rel + z_median.
- Multi-view ScalableTSDFVolume fusion (Open3D 0.19; pyproject pinned <0.20 for
  the known 0.20 Windows integrate no-op), streaming per frame, adaptive voxel
  size, mesh crop + component pruning, Poisson fallback if TSDF empty.
- scripts/regen_dense_map.py to rebuild dense+map for an existing run.
- tests/test_dense.py: 10 tests (Theil-Sen exactness, 30% outliers, gates,
  bilinear sampling); 87+10 all pass. ruff clean on touched files.
### Results (run_20260922_041900, same SfM poses)
- 40/40 frames fused, 0 skipped, rmse_rel 0.8-1.9%, z_median ramp 43.6->15.0 m
  matches drone approach; dense cloud 1,573,698 pts; TSDF mesh 768,543 tris
  (sliver p95 4.8 vs 12.9 needles era).
- Cross-frame depth consistency (3 leave-one-out pairs): median rel err 20.5%
  overall, 7.9% for geometrically similar pair = monocular DA-V2 noise floor,
  not sheet divergence; TSDF averaging is the mitigation.
- GL-free verification renders via scripts/render_run_check_mpl.py (0.19 has no
  headless EGL on Windows; 0.20 renderer unavailable due to the pin).

## 2026-09-22 — map_3d.glb fix + error-free dense cloud (user priority)
### GLB bug (cyan tent in CloudCompare)
build_full_3d_map re-meshed the wedge cloud with Poisson, capping the open
observed footprint into a fake solid tent and dropping vertex colors.
Fix: build_full_3d_map(existing_mesh=...) now reuses the honest TSDF mesh
(ground-plane uprighted via _align_mesh_upright, colors preserved); Poisson
re-mesh remains only as fallback. export.py/regen_dense_map.py pass mesh.ply.
Verified: map_3d.glb now has COLOR_0 + 415,960 verts (was colorless cone).
### Dense-cloud quality pass (cloud_dense.ply)
1. Depth-discontinuity masking (_depth_edge_mask): relative z-jump >18% between
   neighboring pixels (dilated 1px) dropped before fusion -> no more silhouette
   "curtains". Near clip raised 0.5 -> 1.0 m. Work res 384 -> 480 rows.
2. cloud_dense.ply now = TSDF mesh VERTICES (zero-crossing surface, multi-view
   averaged colors, voxel-uniform density). NOTE: volume.extract_point_cloud()
   returns ALL observed voxels along rays (the full viewing-cone volume, 77%
   off-surface) — verified empirically, do not use for surface clouds.
   extract_voxel_point_cloud() likewise unsuitable. Mesh vertices are correct.
3. Result: 681,529 pts (down from 1.57M noisy), 0.00% camera-proximity junk,
   clean color diversity, extent 92.8 x 126.9 x 82.4 (scene frame).
### Known geometry facts (not bugs)
- Wedge/triangle footprint in any viewer = the actually-observed corridor of
  this single straight flight pass; TSDF crops to it honestly.
- dense->sparse distance median ~50 m is expected: sparse points only exist on
  textured facades (27k pts), the cloud covers gardens/water too.
### Verification
- 97/97 pytest pass; ruff clean on touched files (F821 trimesh-before-import
  caught and fixed in the TSDF-reuse branch).
- debug/cloud_quality_check.png (top-down + oblique scatter) and refreshed
  debug/render_check_*.png provide visual QA.
## 2026-09-22 — Test videos with 3D ground truth for model-quality validation
### Problem
User's field single-pass video reconstructs poorly; needed proper test videos
to tell "pipeline broken" apart from "video unsuitable for SfM".
### Added
- scripts/make_city_flyover_video.py: deterministic (seed 20260922) synthetic
  city district (29 textured buildings: windowed facades, parapet roofs, AC
  units; roads/lane markings/cars/plazas; ray-cast textured ground plane,
  painter-sorted quads, near-plane clipping). 600 f, 1280x720 @ 30 fps single
  pass, 44-56 m AGL, ~50 deg oblique. Exports: mp4 + GroundTruthAGL.csv
  (image_id=frame idx -> `reconstruct --gt-file` ATE works) + gt_model.ply
  (156,648 colored surface pts) + gt_scene.json (LoD2 footprints) + meta.json.
  Rendered intrinsics f=0.9*width match the auto SIMPLE_PINHOLE prior.
- data/raw/test_videos/: city_flyover.mp4 (+GT), zurich_real_50f.mp4 (wraps the
  50 local zurich_mav sample images; NOTE: those are generate_sample_zurich.py
  placeholders, not real GoPro footage - real dataset via ETH RPG link).
- TEST_VIDEOS.md: usage, pass criteria, public datasets (TartanAir 2,
  UrbanScene3D, MatrixCity, Zurich MAV), field-capture checklist.
### Verification (run_20260922_074818, --max-frames 100, CPU)
50/50 registered (100%), mean reproj 0.480 px, 10,903 pts, ATE RMSE 2.5115 m /
median 2.0283 m / p95 4.0871 m, scale 16.51, fit_condition ok, 328 s wall.
Model-vs-GT top-down overlay (debug/city_gt_vs_model.png): corridor block
layout recovered; median model->GT surface distance 2.72 m after ATE
similarity alignment (straight-pass rotation-about-flight-axis conditioning
limits tighter alignment).
### Rendering pitfalls fixed
near-plane clipping (front.all() required; partial-behind quads blow up and
fill the frame); BGR vs RGB channel order for cv2 VideoWriter vs GT PLY.
