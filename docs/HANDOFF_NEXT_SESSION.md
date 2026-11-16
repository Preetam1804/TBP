# Handoff: uav3d single-pass city accuracy + appearance work

Date: 2026-11-15. Repo: C:\Users\Preetam\Desktop\TBP (branch `main`, commit 5b8b218 + uncommitted changes).

Read this before touching anything. It records what is DONE, what is VERIFIED, what is
UNFINISHED, and the exact next steps. Do not trust older chat summaries over this file.

## 0. Current verified state

- Every Python file under `src/`, `scripts/`, `tests/` compiles (`py_compile`).
- Full suite: `158 passed` (`python -m pytest -q`), last run 2026-11-16 (146 + 12 shape-eval).
- `tests/test_prior.py` was RESTORED on 2026-11-15 with the 5 documented tests (affine fit,
  depth-ordered fit `raw = 1/(3/depth + 1.5)` -> inverted, uncorrelated -> None, tiny sample
  -> None, gated fill). The prior module is covered by the suite.
- The monocular prior is now WIRED IN (opt-in `--prior`, default off) and validated on the
  cached `run_20261113_055317` run — see section 6.
- PHASE 0 DONE (2026-11-16): shape evaluation rebuilt and the prior promotion gate run.
  **The prior FAILED the gate** — it stays opt-in. Full numbers in
  `docs/CITY_MODEL_EVAL.md`; machine-readable results in `docs/eval_dense*.json`.
- `uv.lock` shows as modified (M). It was touched by `uv run --with pypdf` during PDF inspection.
  Revert it if the change is not intentional: `git checkout -- uv.lock`.
- No commit was made. All work is in the working tree.

## 1. The original problem

- SIH-style evaluation uses a SINGLE-PASS drone video.
- Output 3D city models looked wrong / inaccurate.
- User asked for a swarm audit of the whole codebase, root-cause research ("is the video the
  problem?"), a safe plan, and implementation.

## 2. Confirmed root causes (with evidence)

### 2.1 Keyframe cap truncated the single pass
`src/uav3d/keyframes/selector.py` capped selection with `selected_records[:max_frames]`.
`max_frames=250`, clip has 600 frames, so only the first 8.3 s of a 20 s pass was ever used.
Fixed to spread picks uniformly across the whole clip with
`np.unique(np.linspace(0, total-1, max_frames).astype(int))`; first and last frames are kept.
Test: `tests/test_keyframes.py::test_cap_spreads_across_entire_clip`.

### 2.2 Dense depth range was destroyed by near outliers
`src/uav3d/dense/geometry.py::sparse_depth_range` weighted sparse depths by `1/z^2` (correct for
pixel coverage), but a few ill-triangulated near-camera tracks dominated the low percentile.
Measured on `run_20261113_055317`: near plane ~0.29 vs median depth 9.3; worst near 0.038.
At 0.7 px target and `max_planes=160` the required plane count was median ~2000, worst ~16000,
so the actual disparity step was tens of pixels and depth resolution collapsed.

Fix:
- reject sparse depths outside `[0.05, 20] x median` before weighting,
- floor the near plane at `depth_near_min_ratio * median` (default 0.28),
- `clamp_near_for_plane_budget` raises the near plane only when the requested step cannot fit
  `max_planes`, guaranteeing step <= `plane_step_px * depth_near_plane_relax` (1.5x),
- new `DenseConfig` fields in `src/uav3d/dense/config.py`; wired in `src/uav3d/dense/pipeline.py`
  (warning log names the view whenever the near plane is raised).
Tests: `tests/test_dense_geometry.py::test_sparse_depth_range_rejects_near_outliers`,
`test_clamp_near_for_plane_budget_keeps_step`.

## 3. Measured results

Setup A, old vs fixed on `run_20261113_055317` (first 250 keyframes), working width 960:

| metric | old `dense/` | fixed `dense_v2/` (ref_stride 4) |
|---|---|---|
| valid depth fraction | 26.95% | 42.30% |
| fused points | 448,789 | 649,706 |
| median match cost | 0.1776 | 0.1521 |
| depth range (SfM units) | 0.0575 - 17.67 | 2.206 - 17.67 |
| mesh faces | 1.88 M | 2.04 M |

Setup B, full 600-frame clip after the uniform keyframe cap
(`data/outputs/run_20261115_044902`, 125 refs, working 960):

- valid depth fraction 43.0%, fused points 2,987,990, mesh faces 7,669,202,
  median baseline/depth 0.090, stage time MVS ~28 min.
- BUT the fused model is a long flat ribbon: PCA extents 15.8 x 6.6 x 2.5 units.
- Cause is the video content, not the codec: frame 0 is an aerial city block, frame 300 is a
  low street canyon, frame 500 is grass + road, frame 599 is grass/road with only a building
  edge. The pass leaves the built-up area. The earlier 250-frame model covers the city part.

Conclusion for "is the video the problem?": the single-pass forward trajectory and its content
change (descent, leaving the city, no loop closure) are the limiting factor. Resolution/bitrate
are not the issue. Hidden rear facades are physically unobservable in one pass; no pipeline can
recover them truthfully. The pipeline must (a) select the informative city window, (b) keep the
visible geometry metric-consistent, (c) optionally use a monocular prior for textureless visible
surfaces, (d) label inferred geometry instead of presenting it as measured.

## 4. Also implemented and verified

### 4.1 Frame-range ingestion
`src/uav3d/sources/video.py`: `VideoFileSource(video_path, start_frame=0, end_frame=None)`.
- metadata reflects the selected count/first/last id/timestamps,
- iteration seeks to `start_frame` and stops at `end_frame`,
- `get_frame` rejects ids outside the selected range.
`src/uav3d/cli.py`: `keyframes --start-frame --end-frame` passed through `_build_source`.
`scripts/run_video_pipeline.py`: same two flags, forwarded to the keyframes stage.
Tests: `tests/test_sources.py::TestVideoFrameRange` (14 pass in that module).

### 4.2 Per-vertex best-view texturing (the "looks" win)
- `src/uav3d/modeling/texture_bake.py` (clean, compiles):
  - `vertex_normals`
  - `bilinear_sample_rgb`
  - `choose_vertex_views`: for each vertex picks the most fronto-parallel camera, rejects
    cameras whose filtered MVS depth map shows a surface in front of it (occlusion test),
  - `sample_vertex_colors`: samples the chosen rectified keyframe, falls back to original
    vertex colours where no camera sees the vertex,
  - `gravity_transform`: PCA up (smallest-variance axis) for a city cloud.
  Tests: `tests/test_texture_bake.py`, 6 pass.
- `scripts/colorize_city_model.py`: loads a dense run, simplifies to `--max-faces`,
  removes degenerate/duplicate geometry, merges close vertices (0.004), drops connected
  components below `--min-component-faces`, Taubin smooths 5 iterations, bakes colours,
  rotates to gravity/centres, writes Y-up GLB + PLY + JSON report + 2-panel preview.
- `scripts/build_city_model.py`: older measured exporter (cloud RANSAC/PCA gravity,
  vertex colours); kept as fallback, not the current pipeline for `city_textured.glb`.
- `scripts/eval_city_model.py`: Chamfer/F-score vs the GT GLB. CAVEAT: the GT spans the whole
  ~300 m city while current recon covers a 15-20 unit corridor, so absolute numbers are not
  comparable yet. Do not quote them as accuracy. `docs/city_eval_report.json` is provisional.
- `scripts/render_preview.py`: quick top/iso preview of any cloud.

## 5. Current output artifacts (do not overwrite blindly)

- `city_textured.glb` (root, 9.31 MB, 204,825 verts / 298,119 tris):
  built from `data/outputs/run_20261113_055317/dense_v2/` (first 250 keyframes, fixed depth
  range), 94.9% of vertices coloured from 250 frames. Bbox X 13.46, Y 20.49, Z 6.65.
  This is the best-looking / most block-like measured city model currently.
- `city_textured_prior.glb` (8.53 MB, 183,354 verts / 282,659 tris): same source as
  `city_textured.glb` but from `dense_prior/` (MVS + monocular prior fill, 84.4% final depth
  coverage vs 14.7% measured-only), 93.3% of vertices coloured from 250 views.
  Bbox X 15.49, Y 23.36, Z 6.74. Clearly more complete surfaces (solid roofs/walls/ground),
  same footprint, no sky sheet. NOT promoted over `city_textured.glb` yet: Chamfer/F-score
  vs the GT region is still unregistered, so keep both.
- `city_textured_original.glb` (7.05 MB): the user's old file, preserved. It had the wrong
  axes, mirrored facades, invented minarets and a single facade photo stretched over everything.
- `city_textured_citypass.glb` (9.51 MB, 194,871 verts / 337,686 tris):
  built from `data/outputs/run_20261115_103633/dense/` (frames 0..459, 250 keyframes, 125 refs).
  Bbox X 6.76, Y 15.39, Z 3.35 - a long narrow street corridor. Valid depth 40.38%, 2.59M
  fused points.
- `docs/preview_textured.png`, `docs/preview_citypass.png`: top/iso previews of the above.
They are point-cloud previews, not shaded renders; open the GLB for the real look.
- Reports: `city_textured_report.json`, `city_textured_citypass_report.json`.

## 6. DONE: monocular depth prior (wired in and validated 2026-11-15)

`src/uav3d/dense/prior.py` (259 lines, compiles, no syntax corruption) provides:
- `fit_inverse_depth(dispersion, depth, min_samples, min_rank_corr, max_relative_error)`:
  robust affine `1/z = scale*x + shift` fit in inverse depth, MAD outlier rejection, returns
  `(scale, shift, corr, err, inverted)` or None,
- `depth_from_prior(raw, scale, shift, inverted)`,
- `MonocularPriorModel(model_path, infer_height=518, threads=2)` (onnxruntime CPU; dynamic
  H/W; ImageNet normalization; verified on `models/checkpoints/depth_anything_v2_vits.onnx`,
  ~0.4 s at 518^2 and ~1.1 s at 518x924),
- `build_prior(model, view, image_bgr, sparse_points, near, far, min_rank_corr, max_relative_error)`
  -> `(depth, scale, shift, corr, err, inverted)`,
- `edge_mask(image_bgr, radius=1)`,
- `fill_from_prior(estimate, prior, ref_view, neighbour_views, neighbour_priors, near, far,
  min_support=2, tolerance=0.06, avoid_edges=None)` -> `(DepthEstimate, filled_pixel_count)`.
  It never overwrites `estimate.valid` pixels and requires agreement with >= min_support
  neighbour PRIOR maps.

Integrated as specified in section 7 (all of it):
- `tests/test_prior.py` restored: 5 tests, all passing (the depth-ordered case uses
  `raw = 1/(3/depth + 1.5)`, expects `inverted=True`).
- `DenseConfig` prior fields (default OFF): `prior_enabled`, `prior_model_path`,
  `prior_infer_height`, `prior_min_rank_corr`, `prior_max_relative_error`, `prior_min_support`,
  `prior_tolerance`, `prior_edge_guard_px`.
- `run_dense`: prepass runs `build_prior` for every usable view BEFORE the MVS loop
  (any failure degrades to "no prior" with a warning; never fails the run); writes
  `prior_report.json` (accepted/total, median corr/error, per-view metrics); the fill runs
  AFTER the consistency-filter loop using the clamped per-view depth limits, so filled
  pixels are never rejected for having the same holes as their neighbours.
  `fill_from_prior` never overwrites measured (valid) pixels.
- `--prior` flag on `uav3d dense` and `scripts/run_video_pipeline.py`.
- Report: new `DensePriorSummary` in `dense_report.json` (views accepted, median corr/error,
  `pixels_filled`, `pixels_measured`, `final_valid_fraction`); markdown labels filled pixels
  as *inferred, not measured*. `depth.valid_fraction` still reports the MVS stage only.
  ACCOUNTING NOTE: the first validation run double-counted (MVS-valid + filled, because the
  consistency filter rejects some MVS pixels that the fill re-adds as prior values); fixed by
  counting coverage from the final depth maps. The shipped `dense_prior/dense_report.json`
  was patched with the corrected numbers.

Measured validation on `run_20261113_055317`, first 250 keyframes, 960 px, ref-stride 4
(63 reference views), `dense/` = `dense_v2` (no prior) vs `dense_prior/` (`--prior`):

| metric | `dense_v2` (MVS only) | `dense_prior` (--prior) |
|---|---|---|
| MVS valid pixels | 42.30% (13.81M) | 42.30% (13.81M, unchanged) |
| valid after consistency filter | 14.73% (4.81M) | 14.73% (4.81M) |
| prior fill | — | 22.76M px, 63/63 views accepted, median corr 0.9965, median fit error 0.56% |
| FINAL depth coverage | 14.73% | 84.42% |
| median MVS cost | 0.1521 | 0.1521 (unchanged) |
| fused points | 649,706 | 3,500,118 |
| mesh faces | 2.04M | 2.50M |
| colorized GLB | 204,825 verts, 94.9% coloured | 183,354 verts, 93.3% coloured |

Observations:
- Filled depths sit inside the measured range (view 0: filled p50 9.16 vs measured 9.62;
  max 13.4 vs far 17.7) — the per-view affine alignment is doing its job.
- No sky sheet appeared (camera looks down at the city; Z span unchanged 5.95 vs 5.99).
  On footage with a visible horizon/sky, verify the fill before trusting the output: the
  neighbour-prior agreement gate (6% tolerance) cannot by itself distinguish a smooth
  pseudo-depth (sky) from a smooth real surface. Consider a sky guard for such footage.
- The prior fill is opt-in and labelled inferred in the report, per the honesty guardrail.

## 7. How the prior was integrated (implemented exactly per this sketch, 2026-11-15)

1. Restore `tests/test_prior.py` with:
   - affine recovery: depth uniform 2..20, raw = 1/depth -> inverted False, corr>0.99,
     reconstructed allclose (rtol 1e-4).
   - depth-ordered recovery: raw = 1/(3/depth + 1.5) -> inverted True, reconstructed allclose
     (rtol 1e-3).
   - uncorrelated: raw uniform 0..1, min_rank_corr 0.9 -> None.
   - tiny sample -> None.
   - fill: fake 64x48 view, prior 6.0, estimate valid everywhere except pixel (24,32),
     one neighbour with same pose and prior 6.0, min_support=1 -> count 1, pixel filled,
     all measured depths unchanged. With neighbour prior 2.0 -> count 0.
   Run pytest before touching the pipeline.
2. Add config fields (default OFF so nothing changes for existing tests):
```python
prior_enabled: bool = False
prior_model_path: str = "models/checkpoints/depth_anything_v2_vits.onnx"
prior_infer_height: int = 518
prior_min_rank_corr: float = 0.35
prior_max_relative_error: float = 0.35
prior_min_support: int = 2
prior_tolerance: float = 0.06
prior_edge_guard_px: int = 1
```
3. In `run_dense`, after `usable` is built and before the MVS loop, if `prior_enabled` and
   the model exists, run `build_prior` for every usable view. Use `ranges[index]` for near/far
   and `images.get(index)` for the image. Store accepted depth maps in `prior_maps`; write a
   separate `out_dir/prior_report.json` (accepted/total views, median corr/error, model path,
   per-view metrics) and add `artifacts["prior_report"]`. Log a warning and continue if the
   model is missing or no view is accepted. Never fail the whole run because of the prior.
4. Apply fills AFTER the consistency-filter loop, not before. If filled before,
   `cross_view_agreement` checks them against neighbour MVS maps that have the same holes and
   rejects them. Concretely, after the loop that writes `depth_filtered/`:
```python
for index in usable:
    if index not in prior_maps:
        continue
    depth = filtered_in_memory[index]
    estimate = DepthEstimate(depth=depth, cost=np.zeros_like(depth),
                             num_views=(depth > 0).astype(np.uint8),
                             valid=depth > 0)
    neighbours = [j for j in _neighbour_views(views, usable, index, config.consistency_views)
                  if j in prior_maps]
    # IMPORTANT: use the same clamped nearest/farthest the MVS loop used, not ranges[index]
    filled_est, filled = fill_from_prior(
        estimate, prior_maps[index], views[index],
        [views[j] for j in neighbours], [prior_maps[j] for j in neighbours],
        depth_limits[index][0], depth_limits[index][1],
        min_support=config.prior_min_support,
        tolerance=config.prior_tolerance,
        avoid_edges=edge_mask(images.get(index), config.prior_edge_guard_px),
    )
    if filled:
        filtered_in_memory[index] = filled_est.depth
        _save_depth(_depth_path(filtered_dir, index), filled_est.depth)
        prior_filled += filled
```
   Store `depth_limits[index] = (nearest, farthest)` during the MVS loop; do not recompute with `ranges` because
   `clamp_near_for_plane_budget` may have raised near.
5. Add `--prior` to the dense CLI parser and set `dense_cfg.prior_enabled = True` in
   `_handle_dense_command`. Add `--prior` to `scripts/run_video_pipeline.py` and to
   `stage_dense`, forwarding it only when set. Default remains off.
6. Validate on the existing first-half run (no SFM rerun):
```
.\.venv\Scripts\python.exe -m uav3d.cli dense --run-dir data\outputs\run_20261113_055317 ^
  --output-dir data\outputs\run_20261113_055317\dense_prior --max-width 960 --ref-stride 4 --prior
```
   Compare `dense_prior/dense_report.json` to `dense_v2/dense_report.json`:
   valid fraction should rise a few points, fused points should rise, and median cost should not
   regress much. Then colorize:
```
.\.venv\Scripts\python.exe scripts\colorize_city_model.py ^
  --run-dir data\outputs\run_20261113_055317 ^
  --dense-dir data\outputs\run_20261113_055317\dense_prior ^
  --out city_textured_prior.glb --preview docs\preview_prior.png
```
   Keep the no-prior `city_textured.glb`; only promote the prior model if the preview and the
   Chamfer/F-score improve. Prior fill is not proven yet.

## 8. Guardrails and known risks

- The model is single-pass: rear facades / occluded interiors are NOT recoverable truth. Do not
  reintroduce `completion.py` bilateral symmetry or `lod2.py` minaret templates as "accuracy".
  The legacy completion code is still in the tree and is hallucination; the current exporter
  does not use it. Keep it that way unless there is a GT-validated reconstruction method.
- `sfm.refine_intrinsics=true` on forward flight is ill-conditioned. Solved f ~1159 px vs the
  1024 px prior; a free focal can absorb depth stretch. If shape is still wrong, test a run with
  fixed focal and compare reprojection error, or add a focal-drift gate. Do not silently clamp.
- `matcher_type` is sequential only (no loop closure). For a pass that returns over the same
  area this accumulates drift. The full-run ribbon may be partly drift and partly content.
  Options: vocab-tree/spatial matching, explicit loop pairs, or segment the clip and align.
- The texturing is per-vertex, not a UV atlas. 204k verts gives ~3 cm colour sampling on this
  scene and looks good in a viewer, but close-ups will be soft. A per-face atlas is a later
  improvement; do not call the current output "photorealistic".
- `city_textured.glb`/`city_textured_citypass.glb` and all `debug_*`, `*.ply`, `*.obj`, `models/`
  are untracked. They will not travel with a clone. Keep `scripts/` + `src/` + docs tracked.
- `.venv` is Python 3.12; the declared project range is `>=3.11,<3.13`. System Python 3.14 is
  outside the range and lacks cv2. Always use `.venv`.
- Stale `.pyc`-only modules exist under `src/uav3d/quality|depth|fusion`, `modeling/mvs`,
  `sfm/focal_search|vp_focal|native_log`, `validation/geometry|surface|sp2_report`,
  `calibration.py`, `visualization.py`. Nothing imports them today; do not resurrect them
  blindly. The audit is in `docs/SWARM_AUDIT.md`.
- The PDF `SIH26158.pdf` yields no extractable text with pypdf (likely scanned/vector). The
  official SIH metrics were NOT verified. If the new session can OCR/read it, do that first.

## 9. Useful commands

```powershell
# test / compile
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m py_compile src\uav3d\dense\prior.py

# inspect outputs
.\.venv\Scripts\python.exe scripts\render_preview.py `
  --cloud data\outputs\run_20261113_055317\dense_v2\cloud.ply `
  --out docs\preview_dense_v2.png

# replicate the current textured city from cached dense output (about 60 s)
.\.venv\Scripts\python.exe scripts\colorize_city_model.py `
  --run-dir data\outputs\run_20261113_055317 `
  --dense-dir data\outputs\run_20261113_055317\dense_v2 `
  --out city_textured.glb

# full city-window pass from the video (SFM + dense + texture, ~40-90 min)
.\.venv\Scripts\python.exe scripts\run_video_pipeline.py `
  --video data\city_flyover.mp4 --config configs\video_flyover.yaml `
  --start-frame 0 --end-frame 460 --max-width 960 --ref-stride 2 --mesh
```

## 10. What a new session should do first

1. Read `docs/SWARM_ANALYSIS_AND_PLAN.md`, `docs/SWARM_AUDIT.md`,
   `docs/SINGLE_PASS_RESEARCH.md`, then this file, then `docs/CITY_MODEL_EVAL.md`.
2. Run `pytest -q` and `git diff --stat`; confirm 158 pass and inspect the uncommitted diff.
3. USER REQUIREMENT (2026-11-15): the pipeline must produce a proper 3D model for ANY
   single-pass drone video — not a hand-picked window of one clip. Do not optimize for the
   block-vs-corridor choice on this specific video; make the full-pass run coherent instead.
4. Open `city_textured.glb`, `city_textured_prior.glb` and `city_textured_citypass.glb`;
   do not assume the wider one is more accurate. `docs/CITY_MODEL_EVAL.md` now settles the
   MVS-vs-prior question with numbers.
5. Evaluation is solved — do NOT rebuild it. Use
   `scripts/eval_city_model.py --run-dir <run> --gt-cameras data/raw/test_videos/city_flyover.GroundTruthAGL.csv --sfm-frame --gt-crop-from-recon`
   to score a run mesh in the SfM frame. Registration is exact (0.035 m camera residual).
   Two traps this replaced: the reference is up = **-y** (not Z-up), and a blind shape fit on
   a 300 m ground plane produces a meaningless scale (it reported 19.4 m/unit vs a true 6.76).
6. Prior status: FAILED the promotion gate (completeness @0.5 m -4.17 pp, precision -1.35 pp).
   Do not flip the default without new evidence. The prior does add coverage at 2 m but loses
   precision heavily there (-15.66 pp) — it invents surface rather than recovering it.
7. Next levers, in order: (a) Phase 1 sky guard for the prior fill (expected near-no-op on
   this downward-looking clip, but required for footage with a visible horizon), (b) Phase 2
   focal-drift gate (solved f ~1159 px vs 1024 px prior on forward flight), (c) Phase 3
   matcher wiring + loop closure for the full 600-frame pass (the ribbon is partly drift),
   (d) Phase 4 coverage reporting. Phases 1-4 are still open.

## 11. 2026-11-16 session: focal gate, matcher wiring, sky guard (uncommitted)

User complaint (2026-11-16): "not getting proper accurate 3D reconstruction…
go through the code, see what is breaking, fix it" for any single-pass drone
video. Audit found three accuracy-breaking gaps, all fixed in the working tree
(no commit, no push):

1. **Phase 2 — focal-drift gate (was: nothing flagged it).** With
   `refine_intrinsics=true` COLMAP's BA freely refined the focal and solved
   f ~1159 px vs a 1024 px prior (+13%) on forward flight, absorbing a depth
   stretch into the camera; `SfmRunner` only checked intrinsics when
   refinement was *off*, the CLI only warned on aspect ratio, and
   `run_video_pipeline.summarise()` never compared solved vs prior focal at
   all. New `src/uav3d/sfm/focal_gate.py` (`solved_focal_px`,
   `check_focal_drift`, `FocalDriftResult`); `SfmConfig.focal_drift_max_rel`
   (default 0.10) and `focal_drift_action` (`warn` default, `fail` aborts);
   `SfmRunResult` now carries `matcher_type`, `prior_focal_px`,
   `solved_focal_px`, `focal_drift_rel`, `focal_drift_exceeded`; the SfM CLI
   prints a Focal Drift row and warns when exceeded; `summarise()` prints the
   gate verdict and marks the run WITH WARNINGS when exceeded. The gate
   **flags, never clamps** the focal. Caveat: the video prior focal is a
   0.8*width guess and can itself be off by tens of percent, so a flagged
   drift can mean "the prior was wrong and BA fixed it" — check the
   reprojection error and the trajectory before concluding the model is
   stretched. Tests: `tests/test_focal_gate.py` (17 tests).
2. **Phase 3 — matcher wiring (was: dead config).** `SfmConfig.matcher_type`
   existed but `SfmRunner` never read it — every run was sequential, and an
   unknown value would have silently run sequential anyway. New
   `SfmRunner.match_features()` dispatches `sequential` (default) and
   `exhaustive` (every pair — the loop-closure mode for passes that revisit
   ground, at O(n^2) cost); unknown values raise `Uav3dError`. CLI gained
   `uav3d sfm --matcher`. pycolmap 3.12.6 exposes no pair-list matcher, so
   explicit start/end loop pairs are not available in this binding.
3. **Phase 1 — sky guard for the prior fill (was: none).** `fill_from_prior`
   only required neighbour-prior agreement, which smooth sky pseudo-depth
   passes, so a visible horizon would be filled as a solid far-plane sheet.
   New `dense/prior.py::sky_mask` (top 35% of frame, bright + desaturated +
   low gradient) and `DenseConfig.prior_sky_guard` (default True);
   `run_dense` ORs the sky mask into the fill's avoid-mask. Tests in
   `tests/test_prior.py` (sky band masked, textured ground/roof not masked,
   fill respects the mask).
4. **Bugfix:** `scripts/run_video_pipeline.py::summarise()` referenced an
   undefined `calibration` in its `poses is None` branch — the trustworthiness
   report itself crashed with `NameError` on exactly the path where SfM
   produced no poses. Fixed to use the stored calibration; verified by
   calling `summarise` with `poses=None` (no crash) and with a +13% drifted
   focal (WARNINGS) vs a +4% focal (usable).

Tests after the fix (this Linux container; the user's Windows `.venv`
previously reported 158 passed): `pytest -q` excluding the two open3d-only
modules (`test_completion.py`, `test_lod2.py` — open3d cannot import here,
`libEGL.so.1` is missing from the container) gives **162 passed, 6 failed**;
the 6 failures are pre-existing and data-dependent (they need the untracked
`data/outputs/run_20260914_112220` and Zurich files, absent from the clone).
New tests: `test_focal_gate.py`, 4 sky tests in `test_prior.py`, 1 config
test in `test_dense_pipeline.py`.

Still open: Phase 4 coverage reporting; a real-drone-video end-to-end run
(the synthetic street-canyon video run was started for verification — see
`data/benchmark/flyover.mp4` and `~/workspace/tmp/check_traj.py`, which scores
the SfM trajectory against the analytic flight with Umeyama Sim(3)); whether
`refine_intrinsics=false` (locked prior focal) beats free focal on forward
flight — compare reprojection error and trajectory error before changing the
video config default.

## 12. 2026-11-16 (continued): two stale-cache bugs + focal experiment

The synthetic verification above produced the two most important findings
of the session — both are fixed in the working tree, both were silent
accuracy killers, and neither is visible in reprojection error.

**Synthetic ground truth.** `scripts/render_street_video.py` renders a
60-frame 480x270 street-canyon flyover with known cameras
(`scripts/synthetic_street.py::flight`, step 1.36 m, height 12 m,
pitch -0.12) and true focal 265 px; `data/benchmark/flyover.mp4` +
`flyover.depth.npy` (GT depth per frame) are the artefacts. Scoring uses
an orientation-aware Sim(3) alignment (rotation from the frame-0
quaternions, scale from centre spread) — a plain Umeyama fit on the
camera centres is degenerate because the flight is a straight line and
its roll around the flight axis is unconstrained. Scores below are raw
COLMAP sparse points / fused cloud vs 100k GT surface samples; the GT
sampling floor is ~0.3 m, so ~0.5 m median is "correct" here.

**Focal experiment (all 60/60 registered, reproj 0.28–0.38 px):**

| SfM setting | solved f | centre err (med) | sparse median acc | prec@0.5 m |
|---|---|---|---|---|
| free focal, default config (run 1) | 447 px | 0.035 m | — (not scored cleanly) | — |
| free focal (run 2, same data) | 492 px | 0.039 m | 4.32 m | 0.000 |
| locked prior focal 384 px | 384 px | 1.19 m | 3.84 m | 0.001 |
| locked TRUE focal 265 px | 265 px | 0.040 m | 0.47 m | 0.54 |

Conclusions: (1) the 0.8*width prior (384 px) is 45% off the true focal
on this clip, and **neither** free refinement **nor** locking the wrong
value fixes the geometry — reproj error stays ~0.28 px and the
trajectory looks fine while the points are metres off (bas-relief:
focal error is absorbed into depth, invisible to reproj error).
(2) Free refinement is unstable: identical data solved 447 px once and
492 px the next run. (3) Only the true focal gives a sub-metre model.
**The pipeline's accuracy on forward flight is focal-limited, not
matcher- or dense-limited.** For real footage the fix is a real focal
(EXIF / sensor spec / a calibration capture), not more BA freedom.
The focal gate (Phase 2, §11) is what makes this visible instead of
silent. Caveat: this synthetic is a hard case (narrow FOV, pure forward
flight); wider-FOV oblique drone footage constrains focal better.

**Bug A — stale rectified geometry cache (the bigger one).**
`dense/pipeline.py::_load_or_build_rectified` reused
`dense/rectified/geometry.json` + PNGs whenever they existed, without
checking them against the current run's intrinsics. After an SfM re-run
with a new focal, dense silently ran the plane sweep and fusion with
the OLD focal while the summary reported the new one. Measured on the
synthetic: geometry.json held f=446.98 from the first run while
poses.json said f=265; the fused cloud was laterally compressed by
265/447 = 0.59 (walls at y=±5.4 instead of ±9) and scored median
accuracy **3.73 m, precision@0.5 m = 0**. Fix: the loader now compares
the cached geometry with the current views' intrinsics and rebuilds
(`overwrite=True`) on mismatch, logging a warning. After the fix the
same true-focal run scores median accuracy **0.46 m, precision@0.5 m
0.55, precision@1 m 0.96**, valid depth 42.5% → 52.3%, fused points
97,582 → 191,800. Test:
`test_rectified_cache_rebuilds_when_focal_changes`.

**Bug B — stale sparse-points cache.** `dense/geometry.py::
load_sparse_points` preferred `sparse_points.npz` over the COLMAP
model, but the npz is only written by the dense stage, so after an
SfM re-run dense derived its depth ranges from the PREVIOUS
reconstruction. (This also corrupted the first scoring attempts in
this session — the "3.5 m" numbers for the locked variants were stale
npz scored against new poses.) Fix: prefer the COLMAP model, npz only
as fallback. Test: `test_load_sparse_points_prefers_fresh_colmap_over_stale_npz`
(fake pycolmap module; building a real COLMAP model via the 3.12 API
in a test was not worth the rig/frame ceremony).

**Scoring traps found while verifying (do not repeat them):**
- Score RAW COLMAP points (`colmap/sparse/0`), never `sparse_points.npz`
  after any re-run.
- Align with orientations, not centres-only Umeyama, on straight flights.
- The dense `depth/` maps can be accurate (0.7% vs GT depth here) while
  the fused cloud is wrong — always score the cloud, not just the maps.
- `cloud_metrics` completeness is low by construction on this scene
  (GT samples include unobserved backs of buildings); use accuracy /
  precision on the observed surfaces as the primary signal.

## 13. 2026-11-16 (completion session): coverage reporting, loop closure, focal decision, full-pass run

All work uncommitted, nothing pushed/PR'd/merged. Baseline at session
start: full suite (excl. open3d-only files) **166 passed / 5 failed**
(pre-existing data-dependent failures); HEAD 43219b6 (PR #5 merged).

### 13.1 Phase 4 — coverage reporting (DONE)

New `src/uav3d/dense/coverage.py` (`compute_coverage`,
`compute_coverage_for_run_dir`, `per_view_fractions`, `pca_footprint`,
`grid_coverage`, `decide_verdict`, `coverage_gate_ok`,
`write_coverage_report`), wired into `run_dense()` (step 8b, failure
warns and never discards a finished run): writes
`dense/coverage_report.json`, a `coverage` key in `dense_report.json`,
and a `## Coverage` section in `DENSE_REPORT.md`. New `DenseConfig`
fields (`coverage_cell_size_m=1.0`, `coverage_thin_fraction=0.10`,
`coverage_hole_fraction_warn=0.05`, `coverage_hole_fraction_bad=0.10`,
`coverage_thin_fraction_bad=0.25`). `summarise()` prints per-view
valid range, fused density, hole/thin fractions and verdict, and
marks the run WITH WARNINGS when the verdict is not `good`
(`coverage_gate_ok`); old run dirs without coverage data print a
one-line note instead of crashing.

Design (thresholds stated in code and in every report):
- Footprint plane = PCA PC1-PC2 (report says `projection:
  "pca_pc1_pc2"`, never claims a gravity-aligned ground plane).
- Grid counts only cells whose centre is inside the convex hull of
  the projected cloud (bbox corners outside a diagonal footprint are
  not missed surface).
- Thin = cell holds <10% of mean per-cell count. Verdict `holes` if
  >10% of cells empty; `thin` if >5% empty or >25% thin; else `good`.
- Verdict is grid-driven, not per-view-driven: a strict consistency
  filter can leave per-view final fractions ~12-18% while the fused
  cloud has zero empty cells. Both raw and final per-view fractions
  are reported so the ~40 pp consistency-filter loss is visible.
- Measured vs inferred: per-view split only when `run_dense` supplies
  filled counts; re-analysis from disk labels rows
  `final_mixed_measured_and_inferred` with `None` splits — never guessed.
  Units are SfM units, labelled as such (scale unverified) — no metres
  invented.

Tests: `tests/test_coverage.py` (20 tests). Smoke numbers on
`run_20261116_083045` (read-only): per-view final 7.5-22.7% (median
11.9%) vs raw MVS 39.0-56.2% (mean 52.3%); 191,800 pts, footprint
62.5 SfM unit^2, 3,069 pts/unit^2; grid 0/68 empty, 4 thin (5.9%);
verdict good. Deterministic (byte-identical JSON on repeat).

### 13.2 Loop closure — exhaustive does NOT help on a continuous revisit (DONE)

New `scripts/render_loop_video.py` (104 frames, 480x270, true f=265):
45 forward + 14-frame U-turn + 45 return over the same street;
artefacts `data/benchmark/loop.mp4`, `loop.depth.npy`,
`loop.poses.npz`. Full method/numbers in
`docs/loop_focal_experiments.md`.

| variant | solved f | gate | reg | reproj | centre med | sparse med acc | prec@0.5/@1 |
|---|---|---|---|---|---|---|---|
| sequential (`loop_seq`) | 264.6 px | -31.1% EXCEEDED | 104/104 | 0.280 px | 0.042 m | 0.456 m | 0.565 / 0.974 |
| exhaustive (`loop_exh`) | 264.4 px | -31.1% EXCEEDED | 104/104 | 0.290 px | 0.032 m | 0.457 m | 0.565 / 0.973 |

Exhaustive cost 2.4x wall time (84 s -> 205 s) for +8% points and
identical accuracy: the continuous turn already closes the loop under
sequential matching. Bigger finding: viewpoint variation let free
refinement recover the true focal (264.6 vs true 265) from a 45%-wrong
prior — the -31.1% gate flag here is BA *fixing* a wrong prior, not a
stretch. Caveat: a discontinuous revisit (cut / two separate passes)
was not tested and could still benefit; pycolmap 3.12.6 has no
pair-list matcher, so exhaustive is the only loop-closure mode.
Renderer subtlety: `flight()`'s `yaw` does not change view direction,
so turn/return rotations were built as a rigid rig rotation about
world z (`R = R_fwd @ rot_z(-beta)`); return-leg images are
upside-down vs forward (SIFT is rotation-invariant; 104/104
registered under both matchers).

### 13.3 Focal locked vs free — decision: KEEP `refine_intrinsics: true` (DONE)

Re-run post-PR#5 on `data/benchmark/flyover.mp4` (true f=265), all
sequential, 60 frames. Full tables in
`docs/loop_focal_experiments.md`.

| variant | solved f | gate | reg | reproj | centre med | sparse med acc | prec@0.5/@1 |
|---|---|---|---|---|---|---|---|
| free run 1 (`fly_free1`) | 417.0 px | +8.6% within | 60/60 | 0.280 px | 0.038 m | 3.488 m | 0.000 / 0.000 |
| free run 2 (`fly_free2`) | 367.1 px | -4.4% within | **28/60, 9 models** | 0.273 px | 0.016 m* | 2.856 m | 0.000 / 0.000 |
| locked prior 384 (`fly_lock384`) | 384 px | 0% | 60/60 | 0.282 px | 0.046 m | 3.061 m | 0.000 / 0.000 |
| locked true 265 (`fly_lock265`) | 265 px | 0% | 60/60 | 0.281 px | 0.066 m | **0.486 m** | **0.525 / 0.966** |

\* Free-2 registered only frames 10-37; its centre error is over that
subset only — the headline is the registration collapse.

Confirmed §12 with corrections: locked-true matches the old numbers
(0.47 -> 0.49 m); free focal is unstable across runs (447/492 px old,
417/367 px now — four distinct values); **new failure mode: free
refinement can collapse registration to 28/60 with 9 models** (reproj
stayed 0.27 px — it detects none of this); the old locked-prior centre
error 1.19 m was a stale-cache-era scoring artefact, corrected to
0.046 m. Trajectory centre error is small (0.016-0.066 m) in every
variant, including ones whose points are 3+ m off — bas-relief again.

**Decision: do NOT change `configs/video_flyover.yaml`.** Neither
setting is accurate on forward flight without a true focal (free
2.9-3.5 m, locked-wrong 3.1 m), so flipping to `false` buys
determinism, not accuracy — and it would remove the only mechanism
that can ever be right (Series 1: free recovered the true focal) and
silence the focal gate (locked runs show 0% drift by construction).
What would change it: a real focal (EXIF / sensor spec / calibration
capture) — then lock it, the only sub-metre setting — or evidence on
real wide-FOV/oblique footage that locked-prior beats free on
accuracy.

### 13.4 Full-pass end-to-end on the full synthetic clip (DONE)

`scripts/run_video_pipeline.py --video data/benchmark/flyover.mp4
--config configs/video_flyover.yaml --mesh` (no frame window — the
whole 60-frame clip), run dir `data/outputs/run_20261116_093559`,
wall time 193.3 s. `data/city_flyover.mp4` does not exist in this
checkout, so the synthetic flyover is the only full single-pass video
available locally; it is the full clip, not a hand-picked window.

| stage | result |
|---|---|
| keyframes | 60/60 selected |
| SfM | 60/60 registered, reproj 0.281 px, 7,392 sparse points, solved focal **436.0 px (+13.5%, gate EXCEEDED)** — free focal again wrong (true 265) |
| dense | 60 views, 30 reference, working 480x270, valid depth 54.0%, median cost 0.206, fused **396,554 points**, voxel 0.0099 units |
| mesh | 983,840 vertices / 1,501,121 triangles, `dense/mesh.glb` (45 MB, vertex colours `COLOR_0`) |
| coverage | per-view final valid 12.6-24.2% (median 17.5%, 30 views), all measured; density 9,876 pts/SfM unit^2 over 40.2 SfM unit^2; grid 0/45 empty (0.0%), 2 thin (4.4%); **verdict good** |
| trustworthiness | RESULT completed WITH WARNINGS (focal drift) |

Scored vs GT (raw COLMAP points + fused cloud, orientation-aware
Sim(3), `~/workspace/tmp/score_generic.py` /
`~/workspace/tmp/score_fused.py`): sparse median accuracy **3.71 m**,
fused median accuracy **3.61 m**, precision@0.5 m **0.000** both,
centre error median 0.042 m. The dense stage faithfully densified a
focal-stretched reconstruction — coverage being `good` does not mean
the model is metrically right, and the run says so (gate exceeded).
For comparison, the same clip with the true focal locked scores
0.46-0.49 m fused/sparse (§12, §13.3). This is the focal limit, not a
dense/coverage failure.

Textured GLB: `dense/mesh.glb` is vertex-coloured. The best-view
colourised version was produced with `scripts/colorize_city_model.py`
(open3d — which imports in this container only after installing the
system `libegl1` package, `sudo apt-get install libegl1`, an ephemeral
container change, not in the repo):
`data/outputs/run_20261116_093559/flyover_textured.glb` (9.5 MB,
205,879 vertices / 309,362 faces, 91.3% of vertices coloured from all
60 views; preview `flyover_textured_preview.png`, report
`flyover_textured_report.json`, done in 39.2 s).
`city_textured.glb` is not present in this clone (untracked — it stays
on the user's machine per handoff §8) and nothing in this session
touched it; it remains the best measured city model. The synthetic
flyover GLBs live under `data/outputs/run_20261116_093559/`.

### 13.5 Reporting-integrity bugfix: SfM "Quality Gate Passed" lied (DONE)

`src/uav3d/cli.py`: the "SfM Quality Gate Passed" line sat in the
focal-gate `else` branch, so a collapsed run printed both "Warning:
Registration rate below 85% KPI" and "SfM Quality Gate Passed:
Registration 46.7% >= 85%". Fixed with a pure, tested helper
`sfm_gate_failures()` (registration >= 85%, reproj <= 1.5 px, focal
drift within gate); the summary now prints "SfM Quality Gate NOT
passed: ..." listing the failed gates. Exit code unchanged (0) so
callers are unaffected. Tests: 4 new tests in
`tests/test_sfm_export.py::TestSfmGateFailures`.

### 13.6 Test-suite data skips (DONE)

The 5 pre-existing data-dependent failures are now honest skips, not
deletions and not fake data: `tests/test_keyframes.py` was missing its
`pytest` import (its own `pytest.skip` raised `NameError`);
`tests/test_schemas.py::TestCliSmoke::test_subcommand_stub_execution`
skips when the Zurich dataset (`data/raw/zurich_mav`) or the
historical run dir (`data/outputs/run_20260914_112220`) is absent, and
the SfM case is now pinned to that run dir instead of the
order-dependent newest run dir (other tests create run dirs, which
made it fail spuriously in the full suite).

### 13.7 What is still open (user-dependent only)

- **Real drone video + true focal.** Every number above is synthetic
  (narrow FOV, pure forward flight — the hard case). Real-data
  validation needs a real single-pass drone video and its focal
  (EXIF / sensor spec / calibration capture). Until then, treat any
  forward-flight model as depth-suspect even at 100% registration
  and ~0.3 px reprojection error; prefer flights with viewpoint
  variation (turns, oblique views).
- Monocular prior stays opt-in (failed its promotion gate, §10.6).
  No change.
- Discontinuous-revisit loop closure (cut / two separate passes)
  untested — exhaustive is the only mode available (no pair-list
  matcher in pycolmap 3.12.6).
- `uv.lock` untouched this session. `city_textured.glb` /
  `city_textured_prior.glb` / `city_textured_citypass.glb` are not in
  this clone (untracked, on the user's machine) and were not touched;
  they remain the best measured city models, and the synthetic flyover
  GLBs live under `data/outputs/run_20261116_093559/`.



## 14. 2026-11-16 (city focal session): true focal for city_flyover.mp4, known-focal path, full locked vs free runs

### 14.1 What arrived

User pushed `data/city_flyover.mp4` (600 frames, 1280x720, 30 fps,
AI-generated) plus `data/raw/test_videos/{city_flyover.meta.json,
city_flyover.GroundTruthAGL.csv, city_flyover.gt_model.ply,
city_flyover.gt_scene.json}` to origin/main (commit 28fb68d). The data
files are staged from main in this checkout and are NOT committed on
this branch. `meta.json` claims `focal_px` 1152.0 (0.9 x width), flight
"single pass along +x, x=-100..+100 m, 44-56 m AGL, ~38 deg oblique
down, gentle weave", `straight: false`, seed 20260922.

### 14.2 Focal verification — 1152 px consistent, moderate-low confidence

| Method | Result |
|---|---|
| Free-focal SfM, fully registered (SfM-only, 60 keyframes, frames 0-177) | Solved **1112, 1138, 1163 px** across three runs (mean ~1138); 1152 is within +1.2% of the mean and inside the spread |
| Free-focal SfM, partial registration | Unstable: one run collapsed to 26/60 and solved 1914 px — evidence of instability, not a clean repeat |
| GT-baseline essential-matrix scan (f 700-1600, 9 pairs, GT positions only) | Inlier curve essentially flat; 1152 within 0-3 inliers of best in every pair, but so are 700 and 1024 — **no discrimination**. Two-view focal is unidentifiable at these baselines |
| GT-correspondence via boxes/PLY | **FAILED** — no orientation aligns boxes to frames (best edge distance 5.6 px, ~chance). See 14.4 |
| Vanishing point / horizon | Inconclusive — f^2 negative in 4 of 7 pairs; positive estimates scatter 489-1094 px |
| Scene consistency | SIFT matches persist in all segments incl. 0-599; no evidence of buildings morphing. Sim3 residuals vs GT positions are a smooth bow (focal/depth-tradeoff signature, not frame-to-frame jumps) |

Conclusion: use **1152 px locked**. Best point estimate from SfM is
~1138 +/- 25 px. The focal gate "failing" (+11.2%) on the verification
runs is vs the 384 px prior guess, not vs truth — expected when the
truth is ~1150.

### 14.3 Known-focal path (implemented this session)

Precedence, highest first: CLI `--focal-px` > config `source.focal_px`
> sidecar `<video-stem>.meta.json` next to the video > 0.8*width guess.
The focal source is recorded in the run's `calibration.json`
(`source` field) and printed next to the focal-drift gate verdict by
the SfM CLI summary and `run_video_pipeline.py summarise()` (via
`sfm/focal_gate.py::describe_focal_source()`). Files:
`src/uav3d/config.py` (`SourceConfig.focal_px`), `src/uav3d/sources/
video.py`, `src/uav3d/cli.py`, `scripts/run_video_pipeline.py`,
`src/uav3d/sfm/focal_gate.py`, `tests/test_known_focal.py` (40 tests).
No `1152` or video-specific focal is hardcoded anywhere; defaults are
unchanged (guess stays 0.8*width, `refine_intrinsics` stays true).

**Limitation:** the sidecar is looked for ONLY next to the video file.
The city meta lives at `data/raw/test_videos/city_flyover.meta.json`,
not next to `data/city_flyover.mp4`, so the sidecar did NOT supply the
prior for the city runs — explicit `--focal-px 1152` / config
`focal_px` was used instead. Do not "fix" this by hardcoding the
`test_videos` path.

Full suite after implementation: **241 passed, 6 skipped** (including
the two open3d-only modules, which import in this container after
installing system `libegl1`).

### 14.4 GT model does NOT match this video — verdict (b)

Do NOT use `gt_model.ply` / `gt_scene.json` for cloud-to-GT scoring
of reconstructions of `city_flyover.mp4`. Evidence (two independent
implementations):

- GT assets are mutually consistent (29 boxes agree with the PLY;
  CSV path matches meta: x -100..100, height 44-56 m, z weave +/-9.2 m)
  — the mismatch is video-vs-GT, not GT-vs-GT.
- Box-edge projection fails under every tested camera model (global
  grid over pitch/yaw/roll/focal/frame-offset, plus forward/reversed/
  z-mirror hypotheses): best fraction of visible GT edge points within
  3 px of a Canny edge is 0.22-0.30 vs 0.16-0.21 for random points;
  a true alignment would score ~0.7-0.9. Overlays show boxes landing
  on empty lots/roads.
- Direct PLY renders from GT camera positions: the GT city has a wide
  empty corridor down the flight line with buildings set back; the
  video has a road down the middle with large buildings tight on both
  sides and a big building dead-centre. No pitch matches.
- The video's own geometry is inconsistent with the claimed camera
  model: the along-track vanishing point implies pitch ~45-58 deg at
  f=1152, not the claimed 38 deg (or f ~1460-1935 px at 38 deg). The
  video is AI-generated with supplied, unverified meta — it need not
  be a perspective rendering of any consistent 3D scene.
- The 0.66 m median / 5.49 m p95 / 17.75% completeness figure in
  `docs/CITY_FLYOVER_RECONSTRUCTION_ANALYSIS.md` was scored with the
  pre-rebuild method (blind 24-axis shape alignment, reference upside
  down) that `docs/CITY_MODEL_EVAL.md` declares meaningless — it is
  not a valid measurement. The valid trajectory-aligned evals
  (`docs/eval_dense*.json`) give 1.6-2.1 m median, but the reference
  is 59% ground points and the reconstruction is a ground-hugging
  ribbon, so ground agreement alone can produce most of that score.
- Trajectory alignment is roll-degenerate for straight-line SfM
  paths; the claimed 0.035 m registration belongs to deleted run
  `run_20261113_055317` and is not reproducible (available runs give
  1.82 m / 0.24 m median residuals).

Consequence: score trajectory-vs-GT-positions only (full-pass runs,
and only if the SfM path demonstrably follows the GT weave) plus
internal-consistency metrics. To get a GT-scored benchmark, render a
new synthetic video FROM `gt_model.ply` with a known camera — then
the GT is valid by construction.

### 14.5 Full runs — locked vs free, whole 600-frame pass

Both runs: 250 keyframes selected from all 600 frames (frame range
0-599, 0 decode failures), keyframe images full-res 1280x720,
`--max-width 960 --ref-stride 2 --mesh`, no `--prior`. Locked config
was a `/tmp` copy of `configs/video_flyover.yaml` with only
`refine_intrinsics: false`; free config the same with
`refine_intrinsics: true` and `source.focal_px: 1152` (sidecar not
found next to the video, see 14.3). Reproduce with those `/tmp`
configs; the tracked config is unchanged.

| | Locked `run_20261116_125949` | Free `run_20261116_142027` |
|---|---|---|
| SfM | 250/250 (100%), reproj 0.369 px, 51,536 pts, track 9.06, solved f **1152.0**, drift **0.0%**, gate PASSED, 959 s | 250/250 (100%), reproj 0.369 px, 51,219 pts, track 9.10, solved f **1156.95 (+0.4%)**, gate PASSED, 1079 s |
| Dense | 250 views, 125 refs, 960x540 (f=864), valid **43.0%**, median cost 0.1618, fused **2,984,325 pts**, mesh 4,560,340 verts / 7,649,833 faces, 1602 s | 250 views, 125 refs, 960x540 (f=867.7), valid **43.0%**, median cost 0.1615, fused **2,976,681 pts**, mesh 4,287,470 verts / 7,177,652 faces, 1664 s |
| Coverage | per-view filtered valid median **14.9%** (min 10.2 / max 26.0), grid **14/90 empty (15.6%), 24 thin (26.7%)**, verdict **"holes"** | identical grid counts (14 empty, 24 thin), verdict **"holes"** |
| Textured GLB | `flyover_textured.glb` 9.3 MB, 187,490 verts / 341,057 faces, **95.6% coloured**, 250 views | `flyover_textured.glb` 9.6 MB, 192,327 verts / 348,416 faces, **95.3% coloured**, 250 views |
| Trajectory vs GT positions (Sim(3), eval only) | RMS **0.104 m**, median 0.081, p90 0.160, max 0.300 (frame 2); scale 16.297, rotation 103.62 deg; by third 0.125 / 0.095 / 0.090 m — no bend | RMS **0.0995 m**, median 0.068, p90 0.151, max 0.249; scale 16.291, rotation 101.95 deg |

Conclusions:

- **Locking vs freeing from a correct 1152 prior changes almost
  nothing**: focal +0.4%, reprojection identical, trajectory RMS
  differs by 5 mm, fused points by 0.26%, coverage identical. On THIS
  video (weave + oblique), the focal is well-constrained once the
  prior is right. The earlier instability was a wrong-prior problem,
  not a free-refinement problem.
- Trajectory RMS ~0.10 m is after Sim(3) scale fit and the full-pass
  SfM path follows the GT weave (a straight-line path could not align
  to the weaving GT path with 0.1 m residuals) — so the path SHAPE is
  validated to ~10 cm. Absolute metric scale is NOT validated
  (`scale: unverified_monocular_scale`, fitted scale 16.3).
- No cloud-to-GT accuracy exists for this video (14.4). The two
  "valid" definitions differ: raw plane-sweep depth valid is 43.0%,
  post-filter per-view valid median is 14.9%.

Incidents: `--stage sfm` / `--stage dense` without `--run-dir`
reuses `newest_run_dir()` — the first launch targeted polluted
`run_20261116_093559` and was killed ~15 s in (only a `colmap/`
subdir written, removed; that dir's poses/calibration untouched).
Always create the dir via `--stage keyframes` first, then pass
`--run-dir` explicitly. A runtime restart killed the first locked
dense run after fusion; dense was re-run from scratch. `libEGL` was
missing for `colorize_city_model.py`; installed `libegl1 libgl1`.
The first trajectory scorer had a transposed Umeyama rotation (caught
by a synthetic test, RMS 105 m -> 0.104 m after fix). Progress log:
`~/workspace/tbp_locked_free_progress.md`; stage logs + scorer in
`~/workspace/tbp_logs/`.

**Do not use `run_20261116_093559` as a baseline** — its poses.json
was overwritten by a later SfM-only collapsed run (26/60, focal 1914
px @1280); its dense/GLB outputs belong to the older synthetic-video
run.

### 14.6 What "accurate" means for this video now

The model is internally consistent: 250/250 registered, 0.369 px
reprojection, camera path follows the GT path shape to ~0.10 m RMS,
focal confirmed within ~+/-25 px and free refinement agrees with
locked to 0.4%. It is NOT metrically validated: no cloud-to-GT
accuracy exists (GT model does not match the video), absolute scale
is unverified, coverage verdict is "holes" (15.6% of footprint cells
empty), and the video's own geometry is inconsistent with a single
pinhole camera at the claimed 38 deg (VP says 45-58 deg). Treat
distances/heights as suspect; this is a good-looking, self-consistent
reconstruction, not a surveyed one. The remaining honest step for a
GT-scored claim is to render a benchmark video from `gt_model.ply`
with a known camera (14.4), or to validate on real drone footage with
a real focal.
