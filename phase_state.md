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
