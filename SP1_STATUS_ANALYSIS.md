# SP-1 Status Analysis — `uav3d` (repo `TBP`)

**Analysis date:** 2026-09-16
**Branch analysed:** `arena/01a0aa75-tbp` @ `6c8de8d` ("fix(sfm): auto-detect resolution and configure pinhole model with refine_intrinsics for video runs") — the only commit in the repository history.
**Method:** static reading of every module + **independent execution of the whole pipeline** in a clean environment (fresh venv, generated data, all 5 CLI stages), plus test/lint/coverage measurement.

> **Update 2026-09-16 — the defects below have been fixed.** All P0/P1/P2 items in §7 and the
> documentation issues in §6 were addressed in a hardening pass; the original audit text is kept
> below unchanged as a record of what was found. A status table for each finding is in §10.
> Verified state after the pass: `ruff check .` clean, `pytest` **99 passed** (and green with no
> `data/` directory present at all, i.e. the suite is now self-contained).

---

## 1. Verdict in one paragraph

SP-1 is **functionally complete and genuinely runs end-to-end**: all five stages (`inspect → keyframes → sfm → sync → validate`) execute on real code (no stubs/mocks in `src/`), produce schema-versioned JSON + plots + a Markdown report, and the numbers the phase log claims are reproducible. The code is well-organised, typed, documented, and 68 tests cover the tricky logic (Umeyama with reflection guard, clock-offset sync, pose conventions, overlap guard, GT import isolation). The remaining work is **hardening, not building**: 3 failing tests, a degenerate camera-calibration path for video runs (the newest commit), stale/optimistic entries in `phase_state.md`, no CI/lint config, and no demonstration on the actual ETH Zurich dataset in this checkout.

---

## 2. Scope contract (from `project_context.md`)

| In scope (SP-1) | Delivered? |
|---|---|
| Video + image-sequence ingestion | ✅ |
| Zurich MAV dataset adapter | ✅ |
| Keyframe selection | ✅ (stride + quality gate; optical flow deferred) |
| Telemetry mapping / sync | ✅ (Mode A ID-join, Mode B timestamp+offset) |
| pycolmap visual SfM, pose export | ✅ |
| ATE validation vs ground truth | ✅ (7-DoF Umeyama Sim3) |
| **Out of scope** (SP-2+: depth, fusion, mesh, texturing, frontend, DB, auth, cloud) | ✅ correctly absent |

---

## 3. Repository inventory

```
README.md (9 lines, attribution only)      phase_state.md (157 lines — the real project log)
project_context.md (30 lines — spec)       pyproject.toml + uv.lock (pinned deps)
configs/zurich_baseline.yaml (48 lines, fully-validated via pydantic extra="forbid")

src/uav3d/          2,154 statements across 27 modules
  cli.py            689   argparse + Rich, 5 subcommands, per-stage summary tables & KPI gates
  config.py         139   pydantic models (SfmCfg/SyncCfg/KeyframeCfg/...), fail-fast YAML loader
  logging_config.py  95   rich logging + timed_stage context manager
  sources/          video.py, zurich.py, base.py            (ImageSource protocol)
  keyframes/        quality.py (blur/brightness/clipping/ORB), selector.py (stride + overlap guard + contact sheet)
  telemetry/        zurich_parser.py, synchronizer.py (693 lines — largest module)
  sfm/              runner.py, colmap_export.py, colmap_compat.py (wxyz/COLMAP-version shim)
  validation/       ate.py (Umeyama + flightline condition), reports.py (JSON, 2 PNGs, SP1_REPORT.md)
  models/           frames, keyframes, poses, telemetry, validation (pydantic schemas)
scripts/            generate_sample_zurich.py, make_synthetic_video.py, scratch_pycolmap_check.py
tests/              6 files, 68 tests
```

Everything in `src/` is real implementation: I found **no TODO/FIXME/placeholder/NotImplemented stub** (the only `NotImplementedError` is caught, in the symlink fallback; the only "mock" strings are test fixture names).

---

## 4. What has been done — phase by phase

| Phase | Deliverable | Evidence it exists |
|---|---|---|
| **0** | Scaffold, schemas, config, logging, CLI skeleton, lockfile | `pyproject.toml`, `uv.lock`, models, `configs/zurich_baseline.yaml` |
| **1** | `ImageSource` abstraction; `ZurichSequenceSource` (defensive CSV parsing, µs normalisation, subset/stride); `VideoFileSource` (PTS in µs); synthetic generators; `inspect` | `sources/*`, `telemetry/zurich_parser.py`, `scripts/*` |
| **2** | Quality metrics + deterministic `KeyframeSelector` with auto stride-halving overlap guard; `keyframes.json`, `sfm/images/` (symlinks), contact sheet; `keyframes` CLI | `keyframes/*`, `debug/contact_sheet.jpg` produced |
| **3** | pycolmap 3.12 incremental SfM with locked intrinsics (DB camera injection + BA flags), sequential matching, largest-model selection, post-BA intrinsic assertion; `poses.json`; `sfm` CLI | `sfm/*`, `colmap/database.db` + `colmap/sparse/0/*.bin` produced |
| **4** | `TelemetrySynchronizer` Mode A (dataset image-ID join) and Mode B (timestamp nearest + clock-offset estimation + synthetic-manifest exact mapping + unsynchronized fallback); haversine/CRS helpers; `telemetry.json`, `poses_enriched.json`; `sync` CLI | `telemetry/synchronizer.py`, both JSONs produced |
| **5** | 7-DoF Umeyama ATE with det(R)=+1 reflection guard and collinearity/"weak_flightline_geometry" condition check; `ate_report.json`, 2 plots, `SP1_REPORT.md`; `validate` CLI | `validation/*`, all 4 artifacts produced |

**Design decisions that are actually enforced in code (verified):**
- All timestamps integer µs; poses as COLMAP `cam_from_world` with `qw,qx,qy,qz`; `camera_center = -Rᵀt` (`colmap_compat.py` shim handles both the 3.12 `cam_from_world()`/`Rigid3d` API and legacy `qvec/tvec`).
- Ground truth is **never** an SfM input — there is even a test asserting `GroundTruthAGL` is imported only inside `validation/` (`test_ate.py::TestImportIsolation`).
- Every output JSON carries `schema_version` + the `dataset_label` provenance string.
- `sync` does not mutate `poses.json` (asserted by a test).

---

## 5. Independent verification performed this session

Environment: Python 3.11.2, pycolmap 3.12.6, opencv 4.10 (`-headless`, because this sandbox has no `libGL`), numpy 2.4.6. Editable install of the package, `uv` not available here.

### 5a. Zurich-source pipeline (config `configs/zurich_baseline.yaml`)

| Step | Result (this session) | Recorded in `phase_state.md` | Match |
|---|---|---|---|
| `inspect` | 50 frames 1920×1080, ts 1,459,425,600,000,000 → 1,459,425,609,600,000 µs, OPENCV [1060,1060,960,540,−0.05,0.01,0,0], 49 GPS / 50 GT records, lat [47.376900, 47.377620], lon [8.541700, 8.542564], alt [450.00, 454.80] | identical | ✅ **exact** |
| `keyframes` | 50 evaluated → 25 selected, final stride 2, 0 decode failures | 50 → 25, stride 2 | ✅ (blur median 843.7 vs recorded 422.6 — only difference) |
| `sfm` | **25/25 registered (100.0%)**, mean reproj **0.588 px**, 1 model, 5,396 pts, 30.4 s, 964 MB RSS, intrinsics unchanged post-BA [1060,1060,960,540,−0.05,0.01,0,0] | 25/25, 0.588 px | ✅ **exact** (runtime/RSS differ: different hardware) |
| `sync` | 25/25 matched, `sync_status=dataset_image_id_mapping`, offset N/A, CRS = UTM 32N / EPSG:32632, implied scale ≈ 8.58 m/unit | 25/25, same status/CRS | ✅ |
| `validate` | ATE **RMSE 1.2562 m**, median 1.1496, p95 2.3483, max 2.5040, scale **7.6740**, condition `weak_flightline_geometry` | RMSE 1.2564, scale 7.6738 | ✅ (4th decimal) |
| Determinism | `keyframes.json` byte-identical (md5 `88d90d58…`) across two runs | "deterministic runs" claim | ✅ verified |

Both KPIs pass (registration ≥85%, reproj ≤1.5 px) and ATE < 5 m threshold, exactly as reported.

### 5b. Video-source pipeline (the newest commit's feature — previously unverified in the log)

Ran `make_synthetic_video.py` (100 frames @5 fps) then the video path:

- `inspect --video` → 100 frames, 5.00 fps, 1920×1080, PINHOLE [1536,1536,960,540] ✅
- `keyframes --video` → 100 evaluated → 50 selected (stride 2) ✅
- `sfm` → **only 25/50 registered (50.0%) — below the 85% KPI**; mean reproj 0.339 px; **exported camera `[213.003, 745.210, 960, 540]`**
- `sync` (no telemetry) → graceful fallback, 0/25 matched (status string is misleading, see §7)
- `validate` (no GT for video) → silently used the Zurich `GroundTruthAGL.csv` found under `data/` and printed **"Validation Status: COMPLETED"** with RMSE 0.0056 m

So the video path *runs*, but it is the weakest part of the codebase (details in §7).

### 5c. Tests, coverage, lint

```
pytest (with generated data present):  65 passed,  3 failed   (68 collected, 32 s)
pytest (fresh clone, no data/):        62 passed,  6 failed
coverage (src/uav3d):                  80%  (2,154 stmts / 433 missed)
ruff check .:                          34 findings
```
Coverage per module: models/schemas 100%, `validation/ate.py` 92%, `runner.py` 89%, `config.py` 93%, `validation/reports.py` 97%, sources 85–86%, `keyframes/` 75–87%, `synchronizer.py` 80%, `cli.py` 63%, `sfm/colmap_export.py` 57%, `sfm/colmap_compat.py` 53%.

---

## 6. Documentation integrity issues (fix the log before sign-off)

1. **"Real-data Results" are not real-dataset results.** Every number recorded in Phases 1/3/4/5 reproduces *exactly* from a run over `data/raw/zurich_mav` populated by the project's own `scripts/generate_sample_zurich.py` (50 synthetic renders) — including the giveaway quirks: reproj error `0.588 px`, GPS max `47.377620` (i.e. frame 49's deliberately missing GPS row) and last timestamp `1,459,425,609,600,000 µs`. The real ETH Zurich Urban MAV dataset (81,169 JPEGs) is **not present in this checkout** and `data/` is git-ignored. Either relabel these rows as *sample/synthetic-data runs* or re-run on the actual dataset before claiming SP-1 field readiness. (The `dataset_label` string "replayed public dataset" is at least consistently written into every artifact.)
2. **"pytest: 33/33 passed" / "ruff check passed"** (Phase 1–2 log) do not hold in a clean checkout: 6 tests fail without generated data, 3 fail even with data, and ruff reports 34 findings (including a real `F821`). The test/lint counts in the log are stale.
3. **README is 9 lines** and documents no usage, no install, no pipeline diagram, no CLI examples — all of which exist only inside `phase_state.md`. There is no `LICENSE` (the README *requests* attribution to IJRR 2017, which is good) and no `.github/workflows` CI.

---

## 7. Bugs / risks found (severity ordered)

**P0-1 · Video runs produce a degenerate (unvalidated) camera — `cli.py:255-275` + `sfm/runner.py:211-218`**
For any video run the CLI throws away the source calibration, sets `fx = fy = 0.9 × width` (=1728 for 1920×1080) and force-enables `refine_intrinsics = True`. On my run BA then drifted to `[213.003, 745.210, 960, 540]` — focal off by 8× and **asymmetric by 3.5× between fx and fy**, which is physically implausible for a real camera, yet the pipeline reports a *good-looking* 0.339 px reprojection error (because error was absorbed into the intrinsics and half the frames never registered). There is no FOV clamp, no `fx ≈ fy` / principal-point sanity check, and no post-refinement validation. Also note two different heuristics coexist for the same quantity (`VideoFileSource.calibration()` uses `0.8 × width`, the CLI uses `0.9 × width`), and no attempt is made to read a focal length from container metadata/EXIF.
*Consequence: the "single-pass UAV video" headline path (the actual SIH deliverable) is the one that is not yet trustworthy.*

**P0-2 · `validate` can silently grade a run against unrelated ground truth — `cli.py:470-491`**
With no `--gt-file`, it probes three fixed paths under `data_dir` (including `data/raw/zurich_mav/Log Files/GroundTruthAGL.csv`) and then joins GT rows to poses by *parsing the trailing integer out of the frame id*. On my video run that produced `video_000000 → GT image_id 0` (25/25 "matched") and a green **COMPLETED** report. It happened to be legitimate only because the synthetic video was itself built from those 50 sample images. For any other video (or any run whose ids merely look numeric) this fabricates a passing ATE report. Needs: dataset-identity check / explicit `--gt-file` requirement / explicit "no ground truth available" status.

**P1-1 · Misleading status strings**
* `sync` on a telemetry-less video run reports `sync_status = dataset_image_id_mapping` while showing `0 / 25 (0.0%)` matched — should be `unsynchronized`/`not_applicable`.
* `inspect --video` prints `Source Type: zurich` (it echoes `cfg.source.type` instead of the source actually opened).
* The SfM summary row is labelled **"Intrinsics Locked"** even when `refine_intrinsics=True` and the values were changed by BA (this is exactly what printed the fabricated `[213.003, 745.210 …]` row above).

**P1-2 · Failing / environment-coupled tests**
* `tests/test_keyframes.py:193` — `NameError: name 'pytest' is not defined` (file never imports pytest, so the intended `pytest.skip` crashes and the test can never skip).
* `tests/test_schemas.py:180` — CLI smoke test hard-codes `--run-dir data/outputs/run_20260914_112220`, a git-ignored directory from the author's machine; `sync`/`validate` smoke tests therefore always fail on a clean checkout. (5 more tests fail without `data/`, i.e. there is no self-contained integration test; `SfmRunner.run()` itself is only unit-tested for the zero-image error path.)

**P2 · Housekeeping**
* `.gitignore` lacks `.coverage`/`htmlcov/`; the `!data/.gitkeep` negations point at files that don't exist; `[project.optional-dependencies].dev` and `[dependency-groups].dev` duplicate the same pins; ruff has no config in `pyproject.toml` (hence style noise: 14×F401, 5×RUF059, 5×BLE001, 3×I001, UP035/SIM102/SIM114/S110/RUF022).
* `sfm/colmap_compat.py` and `colmap_export.py` are only ~53–57% covered — and they are precisely the version-sensitive code the project flagged as a risk.
* `scripts/scratch_pycolmap_check.py` (a scratch probe) is shipped in the repo; `generate_sample_zurich.py` writes into `data/raw/zurich_mav` — the same path a real dataset would occupy, which is how the "real-data" confusion above becomes easy to make.

---

## 8. What is definitely NOT done

| Item | Status |
|---|---|
| Re-run + re-label metrics on the real Zurich dataset (81k images) | ❌ not demonstrable here |
| Video-run calibration strategy (metadata focal, FOV bounds, sanity gate) | ❌ |
| Ground-truth ↔ run identity protection in `validate` | ❌ |
| CI (lint + tests on push), lint config, LICENSE, usage README | ❌ |
| Self-contained end-to-end integration test (tiny generated fixture) | ❌ |
| Optical-flow / feature-velocity keyframing (explicitly deferred) | ❌ by design |
| Rolling-shutter correction, GPS multipath mitigation | ❌ (documented as known limitations) |
| SP-2 (depth, TSDF fusion, mesh) | ❌ out of SP-1 scope; handoff spec exists in `SP1_REPORT.md` §6 |

---

## 9. Recommended next actions (in order)

1. **Fix the video intrinsics path (P0-1).** Prefer, in order: (a) read focal length from the container/EXIF if present; (b) otherwise refine focal *with bounds* (e.g. FOV 30°–120°, `fx == fy` shared, principal point fixed at centre) and reject the model if post-BA intrinsics leave those bounds; (c) report both pre- and post-BA intrinsics and never label a refined camera "Locked".
2. **Make `validate` refuse to guess (P0-2):** require `--gt-file` (or a dataset-identity match) for non-Zurich runs; add an explicit `status = "no_ground_truth"` path.
3. **Repair the test suite (P1-2):** add `import pytest`, replace the hard-coded run dir with a `tmp_path`/auto-discovered run, mark data-dependent tests, and add one tiny end-to-end test (5 synthetic frames → all 5 stages).
4. **Correct `phase_state.md`** (§6): relabel the Phase 1/3/4/5 numbers as sample-data runs, refresh test/lint counts, record the video-path finding from §5b as the current open item.
5. **Housekeeping (P1-1, P2):** honest status strings, `.gitignore`/ruff config/CI, README usage + pipeline diagram, LICENSE note.
6. Then start SP-2 Phase A (monocular depth consuming `poses.json` + `keyframes.json`), which the exported schemas already support.

---

## Appendix — exact commands used for this analysis

```bash
python3 -m venv .venv && .venv/bin/pip install -e . pytest ruff coverage     # opencv-python-headless substituted (no libGL in sandbox)
.venv/bin/python scripts/generate_sample_zurich.py                          # data/raw/zurich_mav (50 frames)
.venv/bin/python -m uav3d.cli inspect    --config configs/zurich_baseline.yaml
.venv/bin/python -m uav3d.cli keyframes  --config configs/zurich_baseline.yaml
.venv/bin/python -m uav3d.cli sfm        --run-dir data/outputs/run_<ts>
.venv/bin/python -m uav3d.cli sync       --run-dir data/outputs/run_<ts>
.venv/bin/python -m uav3d.cli validate   --run-dir data/outputs/run_<ts>
.venv/bin/python scripts/make_synthetic_video.py                            # data/interim/synthetic.mp4 (50 frames @5fps)
.venv/bin/python -m uav3d.cli inspect|keyframes --video data/interim/synthetic.mp4
.venv/bin/coverage run --source=src/uav3d -m pytest -q && .venv/bin/coverage report -m
.venv/bin/ruff check . --output-format=concise
```

*Audit phase was analysis-only: no source file in `src/`, `tests/`, `configs/` or `scripts/` was modified while producing §1–§9. Generated data (`data/`, `.venv/`, caches) stays outside Git.*

---

## 10. Fix status (hardening pass, 2026-09-16)

| # | Finding | Status | How |
| :--- | :--- | :--- | :--- |
| 1 | Video runs export an unvalidated/degenerate camera | **Fixed** | New `src/uav3d/calibration.py` (single source of truth, SIMPLE_PINHOLE so fx/fy cannot split) + post-BA plausibility gate + locked-intrinsics fallback that is reported; `--focal-px` for a known focal length |
| 2 | `validate` can grade against unrelated ground truth | **Fixed** | Non-`zurich_*` runs refuse to validate without `--gt-file`; dataset runs resolve GT from the adapter; `ate_report.json.ground_truth` records file, selection, namespace, threshold outcome |
| 3 | Video path had no valid acceptance evidence | **Fixed** | Clip generator no longer pads with unrelated frames by default; video path now runs 5/5 stages at 25/25 registered (100%) |
| 4 | "Real-data Results" were sample-data results | **Fixed** | `dataset_provenance.json` marker + automatic label rewriting; `phase_state.md` rows relabelled; generator refuses to overwrite a non-empty dataset dir |
| 5 | Stale/false test + lint claims in the log | **Fixed** | `phase_state.md` rewritten with current counts (99 tests, ruff clean) and an explicit reading note |
| 6 | Misleading status strings (sync, inspect, "Locked") | **Fixed** | Mode auto-detect requires the `zurich_` prefix; empty telemetry → `unsynchronized`; partial coverage → `partial_telemetry_coverage`; `inspect` echoes the real source type; SfM row distinguishes LOCKED / REFINED / refinement-REJECTED |
| 7 | `test_keyframes.py` missing `import pytest` (F821) | **Fixed** | Import added; the test now builds its own clip instead of skipping |
| 8 | Hard-coded run dir; no self-contained integration test | **Fixed** | `TestEndToEndPipeline` + `mini_recon_dataset` fixture run all five stages on a 10-frame reconstructable scene; suite is green with no `data/` present |
| 9 | No CI, no lint config | **Fixed** | `[tool.ruff]`/`[tool.pytest.ini_options]` in pyproject; 36 findings resolved; `.github/workflows/ci.yml` runs ruff + tests on 3.11/3.12 |
| 10 | `.gitignore` gaps / dead `!data/*` negations | **Fixed** | Coverage artifacts ignored; `data/.gitkeep`, `data/raw/.gitkeep`, `data/interim/.gitkeep` now exist |
| 11 | Duplicated dev dependency pins | **Documented** | Kept for pip *and* uv, with a comment requiring them to stay in sync |
| 12 | Low coverage in pycolmap-facing modules | **Partly fixed** | New tests cover calibration math, param-count gate, camera provenance, sync statuses, GT trust boundary; `colmap_compat` remains the least-covered module (version-specific fallbacks) |
| 13 | README 9 lines, no usage; no LICENSE | **Fixed / deliberate** | README rewritten (install, diagram, CLI table, outputs, trust boundaries, conventions). No LICENSE file added — choosing a licence is the owner's decision; README states this |
| 14 | Scratch script committed; generator writes into the real dataset path | **Fixed** | Renamed to `scripts/check_pycolmap_api.py` with usage docs; generator now writes a provenance marker and refuses non-empty dirs without `--force` |
| 15 | Single squashed commit, no incremental history | **Out of scope** | History is the repository owner's to restructure |

### Additional defects found while fixing

- **`camera_model` vs calibration mismatch** produced COLMAP's opaque *"Invalid camera parameters"*.
  The runner now treats the calibration's model as authoritative (with a warning) and fails fast,
  naming the model and the required parameter count.
- **`ZurichSequenceSource.calibration()` hard-coded 1920×1080**, so any dataset at another
  resolution got wrong principal-point geometry. It now probes the images on disk.
- **`validation.ate_threshold_meters` was ignored** (hard-coded 5.0 in the CLI, unused inside
  `compute_ate`). It is now honoured, and breaching it sets
  `status = "completed_threshold_exceeded"`.
- **`SP1_REPORT.md` hard-coded the ATE row as PASSED** and printed the registration rate as
  `registered/registered`. Both are now computed from the actual threshold and counts.

### Post-fix verification snapshot

```
ruff check .                 All checks passed!
pytest -q                    99 passed   (also 99 passed with data/ removed entirely)
sample-data image sequence   25/25 registered · 0.589 px · ATE RMSE 1.2562 m · scale 7.6740
sample-data video clip       25/25 registered · 0.339 px · refinement rejected (7.1° FoV) and reported
                             · sync_status=manifest · validate refuses auto-GT, ATE 0.0124 m with --gt-file
```
