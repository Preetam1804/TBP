# Pre-flight audit — uav3d (SIH 26158 SP-1)

**Scope requested:** "check until phase 2.2".
**Scope audited:** Phases 0–2 in full (config/CLI/schemas, ingestion, keyframing — image-sequence **and** video
paths), plus every place the Phase-1/2 outputs are consumed downstream (SfM, sync, validate, report) so that
"will eventually break" is answered with evidence and not speculation.
**Branch:** `arena/01a0bd25-tbp` @ `6c8de8d` · **Audit date:** 2026-09-20

**Method:** full read of `src/`, `tests/`, `scripts/`; `pytest` / `ruff` / `uv lock --check`; end-to-end CLI runs
on the repo's own synthetic Zurich generator; a synthetic 60-frame MP4 through the `--video` path; synthetic
"real-format" CSVs built from the ETH dataset's own `readme.txt` field spec; direct inspection of the COLMAP
`database.db` produced by the runner.

**Environment used:** Python 3.11.2 · pycolmap 3.12.6 · opencv-python 4.10.0.84 (swapped to
`opencv-python-headless` — see §F) · numpy 2.4.6 · pydantic 2.13.5 · ruff 0.16.8 · pytest 9.1.1 · uv 0.12.17.

---

## 0. Verdict at a glance

| # | Finding | Class | Severity |
| :-- | :--- | :--- | :--- |
| A1 | Test suite fails on a clean checkout (6 failures) | breaks now | **P0** |
| A2 | `ruff check .` fails (34 findings 1 × `F821` real bug) | breaks now | P1 |
| A3 | Tests are CWD-dependent (7 failures outside repo root) | breaks now | P1 |
| B1 | `OnbordGPS.csv` lat/lon are **1E7-scaled ints**, alt is **mm** — parser never unscales | breaks on real data | **P0** |
| B2 | Headerless CSV ⇒ first telemetry row silently eaten as "header" | breaks on real data | **P0** |
| B3 | `--data-dir` / `UAV3D_DATA_DIR` resolve the dataset path wrongly (or override it entirely) | breaks on real data | **P0** |
| B4 | `calibration_data.npz` keys assumed; silent fallback to hardcoded intrinsics | breaks on real data | **P0** |
| B5 | Image filename ⇒ image-id assumption, silent dict overwrite on collisions | breaks on real data | P1 |
| B6 | Keyframe gates (`min_blur`, `max_clipping`) tuned on synthetic frames only | breaks on real data | P1 |
| B7 | Sample generator writes into the real dataset path and overwrites its `calibration_data.npz` | data corruption | P1 |
| C1 | `keyframes.max_frames` **truncates the tail** instead of subsampling | wrong results | **P0** |
| C2 | Video path ignores every ingest cap/stride; 3 full decodes of the whole video | scalability | P1 |
| C3 | Video sync can never reach telemetry (manifest unreachable from CLI) | dead feature | P1 |
| C4 | Video SfM unvalidated — repo's own synthetic video fails with 0 models | unproven track | P1 |
| C5 | Two different timestamp rules for the same video frame (`POS_MSEC` vs `idx/fps`) | inconsistency | P2 |
| D1 | A missing `--config` file is silently ignored (baseline defaults used) | silent misconfig | **P0** |
| D2 | 10 config keys are accepted but never read (silent no-ops) | silent misconfig | P1 |
| D3 | "Latest run dir" selection is name-ordering; stages can pick an unrelated run | silent misconfig | P2 |
| D4 | PINHOLE focal guess differs between source (0.8·w) and CLI (0.9·w) | dead/misleading code | P2 |
| E1 | `phase_state.md` "Real-data Results" reproduce **exactly** from the synthetic generator | provenance | **P0** |
| E2 | `SP1_REPORT.md` hardcodes "ATE PASSED", "0 frames dropped", "UTM zone 32N" | reporting integrity | **P1** |
| E3 | Duplicate detection is strict for CSVs but silent for images (asymmetric) | data integrity | P2 |
| — | **Verified good** (do not re-derive): full E2E on synthetic data, intrinsics lock, keypoint/camera scale handling with `max_image_size`, video keyframe seek accuracy, `uv.lock` in sync | ✅ | — |

---

## A. Breaking right now (this checkout, no dataset needed)

### A1 — `pytest` fails on a clean checkout: 6 failures

```
$ python -m pytest -q
FAILED tests/test_keyframes.py::TestCliKeyframesCommand::test_keyframe_selection_from_video
FAILED tests/test_schemas.py::TestCliSmoke::test_subcommand_stub_execution[inspect]
FAILED tests/test_schemas.py::TestCliSmoke::test_subcommand_stub_execution[keyframes]
FAILED tests/test_schemas.py::TestCliSmoke::test_subcommand_stub_execution[sfm]
FAILED tests/test_schemas.py::TestCliSmoke::test_subcommand_stub_execution[sync]
FAILED tests/test_schemas.py::TestCliSmoke::test_subcommand_stub_execution[validate]
6 failed, 62 passed in 2.47s
```

Two independent root causes:

1. **`NameError`, not a skip.** `tests/test_keyframes.py:193` calls `pytest.skip(...)` but the module never
   imports `pytest` (unlike every other test module). This failure is unconditional — the guard intended to
   skip when `data/interim/synthetic.mp4` is absent instead crashes:
   ```
   video_path = Path("data/interim/synthetic.mp4")
   if not video_path.exists():
       pytest.skip("synthetic.mp4 not found")   # NameError: name 'pytest' is not defined
   ```
   `ruff` catches it as `F821` (A2) — so a lint gate would have caught a failing test.
2. **Non-hermetic CLI smoke tests.** `tests/test_schemas.py:177-181` invokes subcommands against
   `configs/zurich_baseline.yaml` and a hardcoded `data/outputs/run_20260914_112220`. On a machine without
   `data/raw/zurich_mav` and that run directory, each command correctly returns 1 (dataset/poses missing) and
   the test asserts `== 0`. These tests only pass on the author's machine with data present — they are the
   reason "pytest: N passed" in `phase_state.md` cannot be reproduced by anyone else (or by CI).

`phase_state.md` records "pytest: 21 passed" (Phase 0) and "33 passed" (Phase 1) — a fresh clone cannot
reproduce a green suite at all (the 6 failures above are deterministic). **Fix:**
`import pytest` in `test_keyframes.py`; rewrite the smoke tests to use `tmp_path` + a generated mini dataset
(the fixtures already exist) or assert `main(...) in (0, 1)` while asserting the specific *error message*.

### A2 — `ruff check .` fails: 34 findings

```
$ ruff check .            # 34 errors (20 auto-fixable)
tests/test_keyframes.py:193:13: F821 Undefined name `pytest`
src/uav3d/sfm/colmap_export.py:3:8: F401 `json` imported but unused
src/uav3d/sfm/runner.py:7:20: F401 `typing.Any` imported but unused
src/uav3d/telemetry/synchronizer.py:19:30: F401 `DataError` imported but unused
src/uav3d/sfm/runner.py:90:23: C414 Unnecessary `list()` call within `sorted()`
tests/test_keyframes.py:193:13: F821 Undefined name `pytest`   # ← the only correctness bug
```
Full set includes `F401` × 12 (unused imports in `colmap_export`, `runner`, `synchronizer`, `ate`, `reports`,
`test_sfm_export`, `test_telemetry`), `I001`, `C414`, `SIM102/114`, `UP035`, `RUF022/RUF059`, `BLE001`, `S110`.
Only one is a correctness bug (`F821`, which is A1).

Note the *rule selection ambiguity*: the repo ships **no `[tool.ruff]` config** (and no CI), yet the reported
findings include non-default rule families (`S`, `BLE`, `SIM`, `C4`, `UP`, `RUF`). "ruff check passed" in
`phase_state.md` therefore describes an unspecified local invocation, not what a reader or CI gets. Add a
`[tool.ruff]` block (line-length, `select`) and pin `ruff` so "lint passes" is a checkable claim.
`ruff format --check .` would also reformat 31 of 41 files.

### A3 — tests depend on CWD

```
$ cd /tmp && python -m pytest /home/user/TBP/tests -q
7 failed, 61 passed    # extra failure: test_load_zurich_baseline (relative "configs/zurich_baseline.yaml")
```
There is no `[tool.pytest.ini_options]`, no `testpaths`, and tests use repo-relative paths (`configs/…`,
`data/interim/…`). **Fix:** add `[tool.pytest.ini_options] testpaths = ["tests"]` plus a session fixture that
`chdir`s to the package root (or make every path an absolute `conftest`-provided constant).

---

## B. Will break as soon as the real Zurich data is loaded

The pipeline has (to date) only ever been exercised against `scripts/generate_sample_zurich.py`. The real
dataset differs from that generator in ways the code does not handle. All field specs below are quoted from the
dataset's own `readme.txt` (ETH RPG, `AGZ/readme.txt`, reproduced publicly):
`1: Timestamp, 2: imgid, 3: lat (Latitude in 1E7 degrees), 4: lon (Longitude in 1E7 degrees), 5: alt (Altitude
in 1E3 meters (millimeters) above MSL) …` and, for ground truth, `1: imgid, 2: x_gt, 3: y_gt, 4: z_gt, … All
values regarding positions are in the WGS 84 / UTM zone 32N coordinate system`.

### B1 — GPS is stored scaled by 1E7 (lat/lon) and 1E3 (alt); the parser stores raw values — **P0**

`src/uav3d/telemetry/zurich_parser.py:132,138,144` do `float(row[...])` with no unit handling:

```
$ cat real_format_gps.csv          # values in the exact layout of the real file
1459425600000, 0, 473769000, 85417000, 450000, 0.1
1459425600200, 1, 473769150, 85417018, 450100, 0.1

$ python -c "parse_onboard_gps(...) + recommend_crs(...)"
image_id=1 ts_us=1459425600200 lat=473769150.0 lon=85417018.0 alt=450100.0
recommended CRS -> UTM zone 14236202N (EPSG:14268802) or local ENU centered at (473769225.000000, 85417027.000000)
```

Consequences with real data, all silent:

* `telemetry.json` / `poses_enriched.json` carry lat ≈ 4.7e8 "degrees", alt ≈ 4.5e5 "m".
* `recommend_crs()` (synchronizer.py:41-61) computes a nonsense UTM zone and prints it in the CLI summary and in
  `SP1_REPORT.md` §6.
* `haversine_distance_m()` returns ~1e7× inflated steps → `distance_consistency_note` reports an "implied scale
  factor ~1e8 m/unit" and *still* says "Distance consistency confirmed" (the claim is unconditional,
  synchronizer.py:115).
* Nothing raises. The only reason it hasn't bitten yet is that the sample generator writes already-unscaled
  floats.

**Fix:** normalize on ingest with an explicit, logged rule and fail loudly when out of range, e.g.
`lat = v/1e7 if abs(v) > 90 else v`, `lon = v/1e7 if abs(v) > 180 else v`, `alt = v/1e3 if abs(v) > 1e4 else v`,
recording the applied scale in `raw_fields` (traceability). Add a fixture that mirrors the real readme layout so
this is regression-tested — the current fixture writes pre-scaled decimals and therefore cannot catch it.
(The ATE path is unaffected: `GroundTruthAGL.csv` really is metric UTM 32N, so the Phase-5 design is sound.)

### B2 — Headerless CSVs: the first data row is consumed as the header — **P0**

`zurich_parser.py:60` and `:184` always `next(reader)` as a header. The dataset documents columns by *number*
and ships `write_ros_bag.py` that reads them positionally, i.e. the files plausibly start with data. Result with
a headerless file (same demo as B1): image id `0` is missing from the parsed dict (`KeyError: 0`), so frame 0's
telemetry is dropped, and every subsequent row is parsed through positional fallbacks only. Note the header is
also what selects the lat/lon/alt columns; with a numeric first row the name-based detection silently no-ops.

**Fix:** peek at the first row — if every non-empty cell parses as a float, treat the file as headerless and
re-inject the row (`csv.DictReader`/`islice` pattern), and log "no header detected; using positional columns".

### B3 — `--data-dir` and `UAV3D_DATA_DIR` do not point at the dataset — **P0**

`cli.py:73-75`:

```python
dataset_path = Path(args.data_dir) if args.data_dir else Path(cfg.source.dataset_dir)
if not dataset_path.is_absolute():
    dataset_path = (Path(data_dir).parent / dataset_path).resolve() if not dataset_path.exists() else dataset_path.resolve()
```

Measured behaviour (config: `data_dir: ./data`, `source.dataset_dir: data/raw/zurich_mav`):

| Invocation | Resolved dataset dir | Result |
| :--- | :--- | :--- |
| `uav3d inspect` (default config) | `./data/raw/zurich_mav` | ✅ correct |
| `UAV3D_DATA_DIR=/tmp/uav3d_probe uav3d inspect` | `/tmp/data/raw/zurich_mav` | ❌ `... /tmp/data/raw/zurich_mav` does not exist |
| `uav3d inspect --data-dir /tmp/uav3d_probe` | `/tmp/uav3d_probe` | ❌ treats the *data dir* as the *dataset dir*; ignores `source.dataset_dir` |

The env-var branch rebases the config path on the data dir's **parent** (correct only when `data_dir == ./data`
and CWD is the repo root). The CLI branch ignores `source.dataset_dir` entirely, contradicting the flag's own
help text (`"Data directory root"`) — and it silently rules out the natural
`--data-dir /mnt/dataset_root` + `source.dataset_dir: raw/zurich_mav` layout documented in `project_context.md`
and `configs/zurich_baseline.yaml`.

**Fix (small, matches the docs):** make `source.dataset_dir` authoritative; resolve relative values against CWD
(or `data_dir` when it is absolute *and* the config path is data-dir-relative); let `--data-dir`/env affect
**outputs only**. Add a unit test for all three invocation styles.

### B4 — Calibration file keys are assumed; the fallback is silent and self-confirming — **P0**

`sources/zurich.py:235-269`: only `camera_matrix` + `distortion_coefficients` keys are recognized; anything else
logs a warning and returns **hardcoded** `[1060.0, 1060.0, 960.0, 540.0]` with `model="PINHOLE"` and
`source="zurich_default_intrinsics"`. Why this is dangerous rather than cosmetic:

* `uav3d sfm` passes that calibration into `SfmRunner`, which locks it (`refine_intrinsics: false`) and then
  asserts post-BA that the reconstruction's intrinsics **equal the calibration it injected** (`runner.py:270-286`).
  The check compares against itself, so a wrong-but-consistent calibration can never be caught.
* The demo run in this audit (`phase_state.md`) shows `Camera model: OPENCV (source: calibration_data.npz)` —
  which is only reachable if the npz has exactly those two keys. The real file was never opened, so this remains
  an unverified assumption on the critical path.
* A GoPro Hero 4 at 1080p also has real distortion; silently degrading OPENCV(8) → PINHOLE(4) changes the
  reprojection error and the KPI.

**Fix:** `files = np.load(path).files` → accept several known spellings (`camera_matrix`/`K`/`mtx`,
`distortion_coefficients`/`dist`/`D`), and if nothing matches **raise `DataError`** instead of substituting
defaults (or require an explicit `allow_default_intrinsics: true` in config). Print `npz.files` in
`uav3d inspect` so the real file's schema is visible in the log.

### B5 — Image-name → image-id mapping is an unchecked guess, and collisions are silent — **P1**

`sources/zurich.py:146-162` takes `int(re.findall(r"(\d+)", stem)[-1])` into a `dict[int, Path]`. If two images
map to the same integer (`frame_0001.jpg` vs `cam2_0001.jpg`, or a re-named/mixed folder), the earlier one is
silently dropped — while `parse_onboard_gps` deliberately raises on duplicate image ids. Also unverified: that
the file's numeric stem equals the `imgid` column used for the Mode-A join (`synchronizer.py:211-231`). Given
the 81 k-image dataset, a mismatch would degrade to "many frames without telemetry" without an error.
**Fix:** raise on colliding ids, and add a startup consistency check (sample N stems ∩ GPS ids; warn if the
intersection is empty).

### B6 — Quality gates were calibrated on synthetic imagery — **P1**

`configs/zurich_baseline.yaml` uses `min_blur: 60.0`, said to be set "after inspecting metric distribution" on
"real data". Reproducing that run shows the distribution came from the generator, and it is not even the same
population as the numbers recorded in `phase_state.md`:

| blur median/p95 | ORB features median/p95 | clipping |
| :--- | :--- | :--- |
| `phase_state.md` Phase 2 ("real data") | 422.6 / 428.5 | 172 / 178 | 0.28 % |
| this audit, same generator, current code | 843.7 / 1031.9 | 1714 / 1785 | 0.12 % |

The configured `min_blur: 60.0` sits 7× below the median of the population quoted in `phase_state.md`
(422.6) and 14× below the current generator's median (843.7) — for real urban photos it is therefore likely to
reject nothing at all, and the "quality gate" claim in the report is untested on non-synthetic frames. Note also that `blur_score`/`clipping_ratio` are computed at 640 px width
(`quality.py:65-86`), so absolute values are resolution-dependent: re-tune on real frames and store the metric's
normalization in the output JSON.

### B7 — the sample generator can corrupt a real dataset directory — **P1**

`scripts/generate_sample_zurich.py:13-14` defaults to `data/raw/zurich_mav` — the same path the baseline config
reads — and unconditionally `np.savez`s `calibration_data.npz` (line 37) and drops `frame_%05d.jpg` files into
`MAV Images/`. Run it once in a directory that also holds (or later receives) the real dataset and you get a
silently mixed 50-synthetic + N-real frame set with the real calibration overwritten.
**Fix:** default to `data/synthetic/zurich_mav`, refuse to write into a non-empty directory unless `--force`,
and never overwrite an existing `calibration_data.npz`.

---

## C. Will break with bigger/longer inputs (Phase 2 → Phase 3 coupling)

### C1 — `max_frames` truncates the trajectory instead of subsampling — **P0**

`keyframes/selector.py:242-247`:

```python
if len(selected_records) > self.max_frames:
    selected_records = selected_records[: self.max_frames]      # keeps the FIRST N only
```

Measured (`count: 50, stride: 1, max_frames: 8`):

```
Capping selected keyframes from 50 to max_frames=8
selected: ['zurich_000000', ..., 'zurich_000007']     # 16 % of the flight, the end is gone
```

With the sanctioned dev slice (150–300 frames) at `stride: 1` the default cap of 250 silently discards the last
50 keyframes; `stride: 2` + `max_frames: 250` never triggers, so the bug hides until someone raises the slice
size or lowers the stride. Downstream, SfM/ATE then describe only the beginning of the flight while the report
still claims a full-trajectory KPI. **Fix:** if selection exceeds the cap, uniformly subsample
(`np.linspace(0, n-1, cap).round()`), keep `max_gap`-style continuity, and log first/last frame ids of the kept
subset.

### C2 — the video path has no cap and decodes the whole video three times — **P1**

`VideoFileSource` takes no `count`/`start`/`stride`; `_build_source`'s video branch (`cli.py:63-68`) passes
none; and
`ingest.*` is dead config (D2). Concretely, for one `uav3d keyframes --video` run:

1. `selector.select()` does `list(source.iter_frames())` → `VideoFileSource.iter_frames` **decodes every frame**
   (`grab()`) to build `FrameRecord`s, *and* caches all of them in `self._frame_cache`.
2. The same selector re-opens the file and decodes every frame again for quality metrics (`selector.py:141-160`).
3. A third pass seeks to the selected frames and writes JPEGs (`selector.py:263-283`).

A 20-minute 1080p/30 fps flight is ~36 000 frames ≈ 3 full decodes (~GBs of I/O, minutes to hours) before
`max_frames` is even consulted, and the "≤300 frames" engineering constraint advertised in
`project_context.md` is not enforced for video at all. **Fix:** honour `ingest.max_frames`/`start_frame` (or add
`source.count/stride`) inside `VideoFileSource`, do one sequential pass that computes metrics and stores the
selected frames, and bound `_frame_cache`.

### C3 — video runs can never synchronize with telemetry — **P1**

`TelemetrySynchronizer.sync(..., manifest_path=...)` implements a manifest mode (`synchronizer.py:326-345,
535+`), but no caller supplies it: `cli.py` has no `--manifest` flag and passes neither `manifest_path` nor
telemetry for video (`_handle_sync_command` → `raw_telemetry = source._gps_records` when the attribute exists,
which `VideoFileSource` never provides). Every `uav3d sync --video` therefore ends in
`sync_status="unsynchronized"` with all-null lat/lon, even though `scripts/make_synthetic_video.py` writes the
matching `manifest.json` right next to the video. **Fix:** add `--manifest` (and/or auto-discover
`<video_dir>/manifest.json`), and wire it through `_handle_sync_command`. Note `_sync_mode_b_with_manifest`
matches on `source_image_id`/`video_frame_index` while `make_synthetic_video` writes `zurich_%06d`/`synthetic_%06d`
ids — verify the join keys end-to-end before advertising the feature.

### C4 — the video track has no passing end-to-end case — **P1**

Running the repo's own generator output through the documented commands:

```
keyframes --video /tmp/uav3d_probe/vid/synthetic.mp4   → 30/60 selected, 0 decode failures  ✅
sfm       --video (same run dir)                       → ERROR: 0 valid reconstructions (no good initial pair)
```
The synthetic video scene is not 3D-consistent, so this specific failure is a *scene* limitation — but it means
the `--video` branch of `_handle_sfm_command` (auto PINHOLE + `refine_intrinsics=True`, `cli.py:248-270`) has
never produced poses. Since the SIH deliverable is "video → 3D", either fix the synthetic video generator so it
renders a consistent scene (it currently draws a *different* random pattern per frame plus a translating grid),
or run a short real clip, and record the result. Add it as a test with a hard assertion on
`registered/total >= 0.85`.

### C5 — two conflicting timestamp rules for video frames — **P2**

`VideoFileSource.iter_frames()` derives `timestamp_us` from `CAP_PROP_POS_MSEC` (`video.py:75-105`) while
`VideoFileSource.get_frame()` synthesizes `idx / fps` (`video.py:108-131`). `metadata().fps` is rounded to 2 dp
and is a *container* estimate. For VFR clips the two rules disagree by more than `sync.max_delta_us`, and Mode-B
matching quality then depends on which accessor produced the record. Pick one rule (PTS if monotonic, else
`idx/fps`), and expose the choice in `timestamp_source`.

---

## D. Silent misconfiguration and config drift

### D1 — a typo'd `--config` silently falls back to defaults — **P0**

All six handlers use `load_config(args.config if os.path.exists(args.config) else None)`
(`cli.py:94,160,203,352,443,551`). Demonstrated:

```
$ uav3d inspect --config /tmp/does_not_exist.yaml
ERROR ... Zurich dataset directory does not exist: /home/user/TBP/data/raw/zurich_mav
```
The run proceeded with `configs/zurich_baseline.yaml`'s defaults (the path that "does not exist" is the
*default*, not the typo) — i.e. the user's config was ignored without a word, and the failure surfaced later at
an unrelated place. This directly contradicts the project constraint "fail fast ... extra=forbid".
**Fix:** `--config` should be `default=None`; when `None`, use the packaged baseline *only if it exists*; when
the user provided a path, `load_config` must raise `ConfigError` if it is missing (it already does — the CLI is
the thing swallowing it).

### D2 — 10 config keys are accepted and never read — **P1**

Verified by attribute-usage search across `src/uav3d`:

| Key | Status |
| :--- | :--- |
| `ingest.source_type`, `ingest.start_frame`, `ingest.max_frames` | **entire block unused** |
| `source.count`, `source.stride` | used for Zurich only — ignored for video |
| `keyframes.strategy`, `keyframes.sharpness_threshold`, `keyframes.target_count` | unused (aliases never resolved) |
| `sfm.feature_type`, `sfm.matcher_type`, `sfm.min_num_matches` | unused (SIFT + sequential are hardcoded) |
| `sync.strategy` | unused (nearest is hardcoded) |
| `validation.alignment` | unused (Sim(3) is hardcoded) |
| `validation.ate_threshold_meters` | unused — `cli.py:497` hardcodes `ate_threshold_m=5.0`, and `compute_ate`'s `ate_threshold_m` parameter is itself never referenced in the function body |

Because `extra="forbid"` is on, these keys *look* like a contract. A judge or teammate switching
`sfm.matcher_type: exhaustive`, `keyframes.strategy: optical_flow`, or `validation.alignment: se3` gets no
error and no behaviour change. **Fix:** either implement, or remove the fields and document the fixed choices;
at minimum, log the effective (hardcoded) value at stage start.

### D3 — "latest run" is name-order, not author-order — **P2**

`sfm` / `sync` / `validate` pick `sorted(...run_*)[-1]` (`cli.py:215-220, 364-369, 455-460`). Run dirs are named
`run_<UTC %Y%m%d_%H%M%S>`, so (a) two invocations inside the same second share a directory and overwrite
artifacts, and (b) a run started earlier but still running loses to a later one — a `sync` can be applied to a
different run's `poses.json` while `logger.info` reports it. **Fix:** write a `run.json`/`manifest` marker with
the stage timestamps and select on it; or require `--run-dir` for stages 4–5 and only auto-pick when it is
unambiguous.

### D4 — two different focal-length guesses — **P2**

`VideoFileSource.calibration()` returns `fx = fy = 0.8·width` (`video.py:58-67`), while `_handle_sfm_command`'s
auto-config overrides with `0.9·width` (`cli.py:254-270`). For `sfm` the CLI always wins, so the source value is
dead code that will mislead whoever reads it (and `CameraCalibration.source` records `video_estimate_*` although
a different value is used). Pick one constant, put it in `SfmConfig` (`video_fov_approx`), and record it in
`poses.json`.

---

## E. Integrity of the recorded results (submission risk)

### E1 — `phase_state.md`'s "Real-data Results" are synthetic-generator outputs — **P0**

Reproducing `scripts/generate_sample_zurich.py` today yields, item for item, the numbers recorded as real data:

| Claim in `phase_state.md` | Reproduced from the generator |
| :--- | :--- |
| `Timestamp range: 1,459,425,600,000,000 … 1,459,425,609,600,000 µs` | base `1459425600000000` = 2016-03-31 **12:00:00 UTC** exactly, + `i·200000` |
| `GPS ranges: Lat [47.376900, 47.377620], Lon [8.541700, 8.542564]` | `47.3769 + i·0.000015`, `8.5417 + i·0.000018` (i ≤ 48) — identical to 6 dp |
| `Telemetry: 49 GPS records, 50 Ground Truth records` | generator writes 49 GPS rows (skips frame 49) and 50 GT rows |
| `Camera model: OPENCV (source: calibration_data.npz), params [… -0.05, 0.01 …]` | generator's `np.savez` writes exactly those keys/values |
| `ATE RMSE=1.2564 / median 1.1499 / p95 2.3486, scale=7.6738, reproj 0.588 px` | this audit's run: **1.2562 / 1.1496 / 2.3483, scale 7.6739, 0.588–0.589 px** |
| `registration=100.0%` on 25 images | 25/25 |

A real GPS log cannot have coordinates on a synthetic 6-decimal grid, a timestamp series with a round
12:00:00Z epoch, or a camera calibration byte-identical to the generator's `np.savez`. Additionally the run
covers 50 frames while `project_context.md` mandates 150–300-frame dev slices, and the reported Phase-2 blur/ORB
distribution does not even match the current generator — i.e. it came from an earlier revision of the renderer.
**Conclusion:** none of the recorded "Real-data Results" (Phases 1–5) were produced from the ETH dataset, and
`SP-1 … COMPLETE` with a 100 % registration KPI is unsupported. **Fix:** run the real 150–300-frame slice and
rewrite Phases 1–5 with those numbers, or relabel every block as `synthetic smoke run` and stop quoting KPIs
from it. (Section B explains why the real run is expected to fail until B1–B4 are fixed — do B first.)

### E2 — the generated report asserts things it has not measured — **P1**

`validation/reports.py`:

* `:185` — ATE row is **always** `**PASSED**`, regardless of `rmse_m` and regardless of
  `validation.ate_threshold_meters` (D2).
* `:183` and `:244` — `({m.num_registered_frames}/{m.num_registered_frames})` and
  `"0 frames dropped"` are hardcoded, so a run with dropped frames still reports 100 % and zero losses.
* `:253` — `recommended UTM CRS (UTM zone 32N)` is hardcoded text, not `telemetry.json`'s `recommended_crs`.
* `AteReport.plots` is always `{}` because the CLI writes `ate_report.json` *before* generating plots
  (`cli.py:503-507`), while `SP1_REPORT.md` renders `trajectory_3d.png` / `ate_error_profile.png` from the
  returned dict (works only because the dict is passed directly).

**Fix:** compare `rmse_m` against the configured threshold, interpolate `num_images_total`,
`num_unregistered`, and the CRS from the artifacts, and populate `plots` after plot generation (re-write the
JSON, or generate plots first).

### E3 — asymmetric duplicate handling — **P2**

`parse_onboard_gps` raises `DataError` on duplicate image ids (`zurich_parser.py:121-124`) — strict, good —
while `_discover_images` silently overwrites on numeric-stem collisions (B5) and `_resolve_dataset_paths`
falls back to `subfolders_with_jpg[0]` (`zurich.py:91-93`), i.e. the *first* directory returned by
`Path.iterdir()` when the canonical folder names are absent. That order is filesystem-dependent, which violates
the "deterministic runs" constraint — sort the candidates and log which folder/layout was chosen.

---

## F. Verified working (baseline facts worth keeping)

* **End-to-end on the synthetic dataset works**: `inspect` → `keyframes` (25/50, stride 2, 0 decode failures) →
  `sfm` (25/25 registered, mean reprojection 0.589 px, 5 397 points, 39.2 s, peak RSS 964 MB) → `sync`
  (25/25 matched, `dataset_image_id_mapping`, UTM 32N) → `validate` (ATE RMSE 1.2562 m, median 1.1496 m,
  p95 2.3483 m, scale 7.6739, `weak_flightline_geometry`). All five commands exit 0; artifacts and plots are
  written.
* **Intrinsics lock is real**: `colmap/database.db` → `cameras`: `(1, model 4 = OPENCV, 1920, 1080,
  [1060.0, 1060.0, 960.0, 540.0, -0.05, 0.01, 0.0, 0.0])` — identical to the injected calibration, so the
  post-BA assertion is not vacuous for *changed* values (it is only vacuous for *wrong initial* values, B4).
* **`max_image_size` does not corrupt the intrinsics**: contrary to the usual COLMAP folklore, pycolmap 3.12.6
  rescales keypoints back to native pixel coordinates while keeping the DB camera at native size/params
  (verified with `max_image_size=400` → 283 features, x up to 1894.76 on a 1920-px image, camera stays
  `1920x1080, f=1060`). The DB injection in `runner.py` is therefore consistent; `poses.json.scale_factor=1.0`
  is correct.
* **Video keyframe extraction is frame-accurate** on the repo's synthetic MP4 (mp4v): for every sampled output,
  mean-abs-diff vs the true source frame ≈ 0.2, vs the nearest wrong frame ≈ 1.7 — so `CAP_PROP_POS_FRAMES`
  seeking is fine here (still codec-dependent; C2/C5 remain).
* **`uv lock --check` passes** — `uv.lock` is consistent with `pyproject.toml` (36 packages).
* Zurich source behaviours asserted by the suite hold: numeric ordering + zero padding, subsetting by
  `start_id/count/stride`, missing telemetry ⇒ `timestamp_us=None` with `timestamp_source="missing"` (no
  fabrication), `calibration()` picking up the fixture npz.
* Portability note: `opencv-python==4.10.*` links against `libGL.so.1`;
  `import cv2` dies on a headless box/CI (`ImportError: libGL.so.1`). The pipeline never uses GUI functions
  (matplotlib is forced to `Agg`), so `opencv-python-headless` is the correct dependency for reproducibility.
  Also note `uv` itself is not installed in this sandbox and there is no `Makefile`/CI, so the documented
  `uv sync; uv run pytest` workflow is unverified by tooling.

---

## G. Suggested fix order

**P0 (do before touching the real dataset)**
1. `zurich_parser.py` — B1 unscale lat/lon/alt (auto-detect + log + `raw_fields` provenance) and B2 headerless
   detection; extend the fixture to the readme's real layout.
2. `cli.py` — D1 `--config` must fail when missing; B3 make `source.dataset_dir` authoritative so
   `--data-dir`/`UAV3D_DATA_DIR` work as documented.
3. `sources/zurich.py` — B4 enumerate `npz.files`, accept known aliases, raise instead of silently defaulting.
4. `keyframes/selector.py` — C1 uniform subsampling instead of tail truncation.
5. `phase_state.md` — E1 relabel or re-run; then regenerate Phases 1–5 numbers.
6. Test hygiene — A1 `import pytest`, hermetic smoke tests, `[tool.pytest.ini_options]`/`[tool.ruff]` config so
   the "tests + lint pass" claims are executable by anyone.

**P1** — C2 (video caps/single-pass), C3 (`--manifest` wiring), C4 (consistent synthetic video + real-clip run),
B5 (duplicate ids/layout logging), B6 (re-tune `min_blur` on real frames), B7 (generator path guard),
E2 (report interpolation), D2 (dead config → implement or delete).

**P2** — A3 (CWD robust tests), C5 (single timestamp rule), D3 (run-dir selection), D4 (one FOV constant),
E3 (deterministic folder choice), plus `opencv-python-headless`, a `Makefile`/CI workflow, a `LICENSE`, and a
README usage section (`uv sync`, `uav3d inspect|keyframes|sfm|sync|validate`, dataset placement).

---

## Appendix — how to reproduce this audit

```bash
python -m venv /tmp/uav3d-venv && /tmp/uav3d-venv/bin/pip install \
  "pycolmap==3.12.*" "opencv-python-headless==4.10.*" numpy scipy "pydantic>=2" pyyaml pyproj rich \
  matplotlib pytest ruff uv && /tmp/uav3d-venv/bin/pip install -e . --no-deps
python -m pytest -q                                   # → 6 failed, 62 passed
ruff check .                                          # → 34 errors
uv lock --check                                       # → OK

# synthetic dataset (the source of every "real-data" number in phase_state.md)
python -c "import sys; sys.path.insert(0,'scripts'); \
  from generate_sample_zurich import generate_zurich_sample_dataset as g; g('/tmp/uav3d_probe/raw/zurich_mav', 50)"
UAV3D_DATA_DIR=/tmp/uav3d_probe python -m uav3d.cli inspect      # → dataset path bug (B3)
python -m uav3d.cli keyframes --config <abs-path config>          # → works
python -m uav3d.cli sfm       --config <abs-path config>          # → 25/25, 0.589 px
python -m uav3d.cli sync      --config <abs-path config>          # → 25/25 Mode A
python -m uav3d.cli validate  --config <abs-path config>          # → 1.2562 m

# real-format telemetry demo (B1/B2)
printf '1459425600000, 0, 473769000, 85417000, 450000, 0.1\n' > /tmp/real_format_gps.csv
python -c "from uav3d.telemetry.zurich_parser import parse_onboard_gps as p; \
  from uav3d.telemetry.synchronizer import recommend_crs as r; \
  recs,_=p('/tmp/real_format_gps.csv'); print(recs); print(r([x.lat for x in recs.values()],[x.lon for x in recs.values()]))"
```

---

## Fix Status (2026-09-20)

All findings from sections A–E have been fixed, verified and committed on this branch
(`d8aab11`, `5a995f1`, `326311f`, `fd13ccf`, `b06ff98`, plus the B7/E1/docs commit). Nothing is
deferred except the P2 items that need a human decision (see "Left to the team" below).

| Finding | Status | Fix / evidence |
| :--- | :--- | :--- |
| A1 `pytest` fails on a clean checkout | **Fixed** | `import pytest` restored; non-hermetic CLI smoke tests replaced by fixture-based success + failure-path tests. `python -m pytest` → 81 passed. |
| A2 no ruff config, 34 findings | **Fixed** | `[tool.ruff]`/`[tool.ruff.lint]` pinned; dead imports, B905/SIM102/SIM108/RUF059/F821 cleared. `ruff check .` → All checks passed. |
| A3 tests depend on CWD | **Fixed** | Repo-root fixture + autouse chdir in `tests/conftest.py`; repo-relative config paths. |
| B1 GPS stored in raw 1E7 / 1E3 units | **Fixed** | `zurich_parser` normalises lat/lon (1E7) and altitude (1E3) with per-row provenance in `raw_fields`; impossible coordinates raise `DataError`. Regression tests in `TestRealZurichTelemetryFormat`. |
| B2 first CSV row eaten as header | **Fixed** | Header detection peeks the first non-empty, non-comment row and treats all-numeric rows as data (positional columns). |
| B3 `--data-dir`/env ignored `source.dataset_dir` | **Fixed** | `_resolve_data_dir` (outputs only) + `_resolve_dataset_dir` (dataset) with documented fallbacks and a new `--dataset-dir`. Verified with 5 invocation styles → 50 frames / 49 GPS rows each. |
| B4 npz keys assumed; silent PINHOLE fallback | **Fixed** | Keys discovered from `npz.files` (`camera_matrix`/`K`/`mtx`, `distortion_coefficients`/`dist`/`D`), mapped to PINHOLE/OPENCV/FULL_OPENCV by coefficient count; missing/invalid calibration raises unless `source.allow_default_intrinsics: true`. |
| B5 duplicate ids / stem guessing | **Fixed** | Numeric-suffix joins go through `validate_frame_id_mapping` (rejects duplicate numbers, mixed prefixes, missing suffix); duplicate image ids raise instead of overwriting; layout is logged. |
| B6 gates calibrated on synthetic imagery | **Fixed** | Metrics record their normalisation (`analysis_width_px`); gate reports rejections and warns when a threshold rejects nothing or >50%; `keyframes.min_blur_mode: adaptive` (+`blur_percentile`) added; `configs/zurich_baseline.yaml` documents that `min_blur` must be re-calibrated on real frames. |
| B7 generator can corrupt a real dataset | **Fixed** | Default output is `data/synthetic/zurich_mav`; refuses non-empty/real dataset dirs and existing `calibration_data.npz` (exit 1) unless `--force`/`--overwrite-calibration`. Verified against a real-format directory: refusal + files untouched. |
| C1 `max_frames` truncated the tail | **Fixed** | Uniform subsampling across all candidates; verified 50 → 8 frames spanning `zurich_000000`..`zurich_000049`. |
| C2 video ignored caps; 3 full decodes | **Fixed** | `ingest.start_frame`/`max_frames`/`stride` bound the slice; grab-only scan + pixel iterator decode each selected frame once; bounded LRU cache. |
| C3 manifest unreachable from CLI | **Fixed** | `sync` auto-discovers `<video_dir>/manifest.json` and accepts `--manifest`; `--mode auto` picks video mode. Verified: `sync_status=manifest`, 30/30 matched. |
| C4 video SfM unvalidated | **Fixed** | `make_synthetic_video.py` renders a persistent 3D landmark cloud (plus manifest and evaluation-only ground truth); the full video chain now runs 30/30 registered, 1.159 px, ATE 0.5468 m. |
| C5 two timestamp rules for one frame | **Fixed** | One validated rule (`_timestamp_for`): container PTS accepted only within 1/fps+10 ms of the index/fps estimate; the source of each timestamp is recorded. |
| D1 missing `--config` silently ignored | **Fixed** | `_load_pipeline_config` raises `ConfigError` (exit 1) for an explicitly requested missing file; the repo baseline is used only when no `--config` is given. Test: `TestCliFailsLoudlyWithoutInputs`. |
| D2 10 dead config keys | **Fixed** | `ingest.*`, `sfm.min_num_matches`, `sharpness_threshold`/`target_count` aliases, KPI thresholds and `validation.*` are now consumed; unimplemented choices are `Literal`-restricted so they cannot be silently ignored. |
| D3 run pick by name order | **Fixed** | `_latest_run_dir` selects by mtime among runs that contain the required artifacts and logs the choice; `--run-dir` unchanged. |
| D4 two different focal guesses | **Fixed** | Single exported constant `DEFAULT_FOCAL_RATIO` (0.9) used by both the source calibration and the SfM auto-config. |
| E1 synthetic numbers presented as real | **Fixed** | `phase_state.md` results relabelled "Synthetic-sample Results" and refreshed from a post-fix run; provenance note added with reproduction commands; `README.md` documents dataset placement. |
| E2 report hardcodes outcomes | **Fixed** | `SP1_REPORT.md` derives every KPI value, target, PASS/FAIL, dropped-frame count, camera model/params, stride/blur gate, sync status and CRS from the run artifacts + config. Verified by re-running validate with `ate_threshold_meters: 1.0` → report shows `FAILED`. |
| E3 silent folder choice / asymmetric duplicate handling | **Fixed** | Image folders are chosen in name order with a warning; duplicate ids raise; `sources.base` no longer exposes private `_gps_records` to the CLI. |

### Left to the team (needs a human decision, not a bug)

- **`LICENSE`** — the repository has none; choosing a license is a policy decision.
- **`Makefile`/CI workflow** — `ruff check .`, `pytest`, `uv lock --check` are now all green and CI-ready,
  but wiring up a provider (GitHub Actions) and branch protection is outside this pass.
- **`min_blur` re-calibration on real imagery** — the gate is now observable and switchable to
  `adaptive`; picking the final absolute value requires the real AGZ frames (not available in the sandbox).
