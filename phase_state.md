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
### Evidence
<Commands run and their observed results>
### Remaining Tasks
<Remaining work in subsequent phases>
### Current Blockers
<Blockers or none>
### Next Actions
<Immediate next actions>
```

> **Reading note (2026-09-16):** the metric tables for Phases 1, 3, 4 and 5 were produced by
> running the pipeline over `data/raw/zurich_mav` populated by
> `scripts/generate_sample_zurich.py` — a **locally generated synthetic stand-in dataset**, not the
> real ETH Zurich Urban MAV dataset. Those rows are now labelled *Sample-data results*. They were
> previously labelled "Real-data Results", which was wrong: the real dataset (81,169 images) is not
> present in this working tree and `data/` is git-ignored. See Phase 6 for the provenance-labelling
> mechanism that now prevents this class of misreporting.

---

## Phase 0 Completed: 2026-09-14
### Completed
Package scaffolded; schemas, config loader, logging, CLI skeleton, git init, lockfile.
### Files Created
pyproject.toml, uv.lock, .gitignore, project_context.md, phase_state.md,
src/uav3d/{cli,config,logging_config,exceptions}.py, src/uav3d/models/{frames,telemetry,keyframes,poses}.py,
tests/test_schemas.py, configs/zurich_baseline.yaml (skeleton)
### Decisions Made
Python 3.12.14; pycolmap 3.12.6; argparse with Rich styling; all timestamps integer µs;
poses = cam_from_world, quat wxyz; configs use extra="forbid".
Verified pycolmap 3.12.6 Image attributes: cam_from_world, Rigid3d with rotation (quat wxyz) and translation.
### APIs/CLI Added
uav3d inspect|keyframes|sfm|sync|validate (skeleton)
### Database Changes
None (no DB in SP-1).
### Tests and Validation
Claim at the time: pytest 21 passed. CLI help smoke test OK.
### Remaining Tasks
Phases 1–5.
### Current Blockers
None.
### Next Actions
Phase 1: ZurichSequenceSource + synthetic video generator.

## Phase 1 Completed: 2026-09-14
### Completed
ImageSource abstraction; ZurichSequenceSource (subset selection, defensive CSV parsing,
µs normalization); VideoFileSource; synthetic clip generator + manifest; 'inspect' command.
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
Claim at the time: pytest 33 passed.
### Sample-data results (synthetic stand-in dataset, not the published dataset)
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
'uav3d keyframes' CLI; keyframes.json + sfm/images/ + contact sheet.
### Files Created
keyframes/{quality,selector}.py, tests/test_keyframes.py
### Files Modified
cli.py, config.py, configs/zurich_baseline.yaml
### Decisions Made
No optical-flow selection (deferred); min_blur=60.0 after inspecting metric distribution;
symlink with copy fallback; overlap guard halves stride automatically.
### Sample-data results
considered=50, selected=25, final stride=2; blur median/p95 ≈ 422–844/428–1032 (varies with the
generator revision), ORB features median/p95 ≈ 172/178, clipping ≈0.28%.
### Remaining Tasks
Phase 3 SfM.
### Current Blockers
None.
### Next Actions
Phase 3: pycolmap SfM runner + poses.json.

## Phase 3 Completed: 2026-09-14
### Completed
Scratch API check; SfM runner (locked intrinsics, sequential matching, model selection);
poses.json exporter with metrics; 'uav3d sfm' CLI.
### Files Created
sfm/{runner,colmap_export,colmap_compat,__init__}.py, scripts/check_pycolmap_api.py (originally
named scratch_pycolmap_check.py), tests/test_sfm_export.py
### Decisions Made
pycolmap 3.12.6; cam_from_world API = confirmed modern Image.cam_from_world() method returning Rigid3d,
with Rotation3d.quat in Eigen [x, y, z, w] format mapped to [qw, qx, qy, qz] (wxyz) order via compat
shim (supporting legacy qvec/tvec as fallback); intrinsics locked via database camera injection +
IncrementalPipelineOptions BA refinement flags and verified identical post-BA; seq_overlap=10;
max_image_size=1600.
### Sample-data results
models=1, registered=25/25 (100.0%), mean reproj err≈0.588–0.589 px, runtime≈19–34 s,
peak RSS≈1.0 GB, intrinsics unchanged post-BA.
### Remaining Tasks
Phase 4 telemetry sync.
### Current Blockers
None.
### Next Actions
Phase 4: TelemetrySynchronizer.

## Phase 4 Completed: 2026-09-14
### Completed
TelemetrySynchronizer (Mode A ID-mapping, Mode B timestamp+offset); 'uav3d sync' CLI;
telemetry.json + poses_enriched.json.
### Files Created
telemetry/synchronizer.py, tests/test_telemetry.py
### Decisions Made
Mode A only for Zurich (no offset estimation); poses.json untouched; recommended_crs via pyproj.
### Sample-data results
matched=25/25, sync_status=dataset_image_id_mapping, GPS range lat [47.376900, 47.377620],
lon [8.541700, 8.542564], alt [450.00, 454.80] m.
### Remaining Tasks
Phase 5 ATE validation.
### Current Blockers
None.
### Next Actions
Phase 5: Umeyama ATE + SP-1 report.

## Phase 5 Completed: 2026-09-14 — SP-1 pipeline complete
### Completed
Umeyama ATE validator with condition guard; reports (JSON, plots, SP1_REPORT.md);
'uav3d validate' CLI.
### Files Created
validation/{ate,reports,__init__}.py, models/validation.py, tests/test_ate.py
### Sample-data results (KPI vs target)
registration=100.0% (target ≥85%); mean reproj≈0.589 px (target ≤1.5);
ATE RMSE=1.2562 m / median=1.1496 m / p95=2.3482 m; scale=7.6740;
fit_condition=weak_flightline_geometry; matched=25/25.
These are sample-data numbers, not field-flight or published-dataset validation.
### Known Limitations
rolling shutter uncorrected; GPS multipath; stride-based keyframing only.
### Remaining Tasks
SP-2 handoff: depth (SP-2 consumes poses.json + keyframes.json).
### Current Blockers
None.
### Next Actions
Hardening pass (Phase 6), then SP-2 Phase A (DepthAnything-V2-Small).

---

## Phase 6 Completed: 2026-09-16 — audit and hardening
### Completed
An independent audit of the finished pipeline (fresh environment, full re-execution of every stage
on both input paths) found and fixed three correctness defects, three reporting-integrity defects,
and several tooling gaps. See `SP1_STATUS_ANALYSIS.md` for the full audit.

**P0-1 — video runs exported a physically impossible camera.**
The CLI discarded the source calibration for video input, assumed `fx = fy = 0.9 × width` and forced
`refine_intrinsics=True` with no bounds or validation. A run on the sample clip exported
`[213.003, 745.210, 960.0, 540.0]` — focal 8× off and `fx ≠ fy` by 3.5× — while still reporting a
flattering 0.339 px reprojection error and a green "quality gate passed" line. Fixes:
- new `src/uav3d/calibration.py` is the single source of truth for assumed intrinsics (the video
  source and the CLI previously used different 0.8×/0.9× multipliers, so `inspect` and `sfm`
  disagreed about the same file);
- video estimates always use `SIMPLE_PINHOLE`, so BA cannot drift `fx` and `fy` apart;
- post-BA plausibility gate: implied horizontal FoV must sit inside
  `sfm.video_min_fov_deg`/`video_max_fov_deg`, focal lengths must be near-isotropic, and the
  principal point must stay near the image centre;
- if a refined camera fails the gate, the reconstruction is redone with the supplied intrinsics
  locked and the rejection is reported (a warning, a table row, and `intrinsics_locked=true` in
  `poses.json`) instead of exporting the bad camera or aborting;
- `poses.json` camera block now records `intrinsics_locked`, `initial_params`, `horizontal_fov_deg`
  and `intrinsics_source`; the summary table no longer labels a refined camera "Locked";
- the SfM runner no longer trusts `camera_model` from config over the calibration's own model, and
  it fails fast with the parameter count if a calibration cannot be used for its model (COLMAP
  previously reported only "Invalid camera parameters").

**P0-2 — `validate` could grade a run against unrelated ground truth.**
With no `--gt-file` it probed fixed paths under the data directory and joined ground truth by the
trailing integer of the frame id, so a video run silently produced `video_000000 → GT image_id 0`,
25/25 "matched" and a green COMPLETED report. Fixes:
- dataset runs take ground truth from the dataset adapter (`auto_discovered_from_source`);
- any non-`zurich_*` run refuses to validate without an explicit `--gt-file`;
- `AteReport.ground_truth` records the file, the selection method, the run's frame-id namespace,
  the matching rule and the threshold outcome; the CLI table and `SP1_REPORT.md` display them;
- `validation.ate_threshold_meters` is now actually used (it was previously ignored in favour of a
  hard-coded 5.0) and the report's ATE KPI row is computed, not hard-coded "PASSED";
- the registration-rate row prints `registered/total keyframes` instead of `registered/registered`.

**P0-3 — the video path had no valid acceptance evidence.**
`scripts/make_synthetic_video.py` always padded its clip to 100 frames with procedurally generated
images that share no scene content with the rest of the clip, so half the frames were unregistrable
by construction. It now assembles the clip from frames that exist on disk (padding is opt-in via
`--pad-synthetic`), and the video path is exercised end to end for the first time (evidence below).

**Reporting integrity.** `scripts/generate_sample_zurich.py` now writes
`dataset_provenance.json`; when present, `ZurichSequenceSource.dataset_label()` rewrites every
downstream `dataset_label` to a synthetic-sample warning, so sample numbers cannot be reported as
published-dataset results. The generator also refuses to overwrite a non-empty directory without
`--force`. Stale "ruff passed" and test-count claims in this file were corrected.

**Tooling.** `[tool.ruff]` + `[tool.pytest.ini_options]` added to pyproject.toml and all 36 initial
findings resolved (ruff clean); CI workflow added; three permanently-failing, machine-specific tests
replaced by a self-contained end-to-end test and a mini reconstruction fixture; new tests for the
calibration gate, sync statuses, camera provenance and dataset labelling; README rewritten with
install, pipeline diagram, CLI reference, output inventory and the trust boundaries.

### Files Created
src/uav3d/calibration.py, .github/workflows/ci.yml, tests/test_calibration.py
### Files Modified
cli.py, config.py, sfm/runner.py, sfm/colmap_export.py, sources/video.py, sources/zurich.py,
telemetry/synchronizer.py, validation/ate.py, validation/reports.py, models/poses.py,
models/validation.py, pyproject.toml, .gitignore, configs/zurich_baseline.yaml, README.md,
phase_state.md, scripts/{generate_sample_zurich,make_synthetic_video,check_pycolmap_api}.py,
tests/conftest.py, tests/{test_schemas,test_sources,test_telemetry,test_ate,test_sfm_export,test_keyframes}.py
### Decisions Made
- Assumed video intrinsics are a first-class, single-sourced, auditable assumption; refinement is
  opt-out-able and always gated.
- Refinement output is data, not truth: an implausible result is discarded and reported.
- Evaluation refuses to guess its ground truth; provenance is recorded in the artifact.
- Provenance labelling is derived from the dataset directory, not from operator memory.
### APIs/CLI Added
`uav3d sfm --focal-px <pixels>`; config keys `sfm.video_fov_deg`, `sfm.video_focal_px`,
`sfm.video_min_fov_deg`, `sfm.video_max_fov_deg`, `sfm.allow_unbounded_intrinsics`;
`poses.json` camera provenance fields; `ate_report.json` `ground_truth` block;
`scripts/make_synthetic_video.py --frames/--fps/--zurich-dir/--pad-synthetic`;
`scripts/generate_sample_zurich.py --dataset-dir/--frames/--force`.
### Database Changes
None (no DB in SP-1).
### Tests and Validation
pytest: 99 passed (~30 s), ruff: clean. Previously 68 tests with 3 permanent failures and 34 lint
findings. The three failures are fixed rather than skipped, and a self-contained integration test now
covers all five stages on a 10-frame reconstructable mini scene (no dependency on `data/`).
### Evidence
Sample-data image-sequence path (unchanged by the pass, re-verified):
registration 25/25 (100.0%), mean reproj 0.589 px, ATE RMSE 1.2562 m, scale 7.6740, GT
`auto_discovered_from_source`, status `completed`.
Video path (first valid end-to-end run; clip assembled from the 50 sample frames):
`inspect` 50 frames @5 fps, SIMPLE_PINHOLE f=1662.77 px (60° assumed FoV);
`keyframes` 50 → 25 selected (stride 2);
`sfm` 25/25 registered (100.0%), mean reproj 0.339 px, focal refinement attempted
(f = 15498 px → 7.1° FoV) and **rejected**, intrinsics locked at the assumed FoV;
`sync` 25/25 matched via the synthetic manifest (`sync_status=manifest`);
`validate` refused to auto-select ground truth (non-`zurich_*` namespace) and, with an explicit
`--gt-file`, reported ATE RMSE 0.0124 m over 25/25 matched frames.
### Remaining Tasks
- Re-run the pipeline on the real ETH Zurich Urban MAV dataset and replace the sample-data numbers.
- Optional: read focal length from container/EXIF metadata when available (only sample has no EXIF).
- Optional: resolve the duplicate `[project.optional-dependencies].dev` / `[dependency-groups].dev` lists
  if a single toolchain is standardised on.
### Current Blockers
The real ETH Zurich Urban MAV dataset is not available in this environment, so published-dataset
numbers still cannot be produced here.
### Next Actions
SP-2 Phase A: monocular depth (DepthAnything-V2-Small) consuming `poses.json` + `keyframes.json`,
whose schemas are already stable and self-describing.
