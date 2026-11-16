# Loop-closure and focal experiments (synthetic street canyon)

Date: 2026-11-16. Branch `main`, HEAD 43219b6 (post-PR#5 code). All changes
uncommitted; nothing pushed, no PR, no merge. Only code change in this
experiment: `scripts/render_loop_video.py` (new). This file is the results
record. Pre-existing uncommitted changes in the tree
(`scripts/run_video_pipeline.py`, `src/uav3d/dense/*`, `tests/test_keyframes.py`,
`tests/test_schemas.py`, `src/uav3d/dense/coverage.py`, `tests/test_coverage.py`)
are **not** from this experiment and were not touched.

## Method (shared)

- Scene: `scripts/synthetic_street.py` analytic street canyon (ground + two
  facades with balconies, length 160 m, half-width 9 m, height 24 m),
  480x270, true focal **265 px**. GT depth/poses are evaluation-only, never
  SfM input.
- SfM: `.venv/bin/python -m uav3d.cli sfm`, one job at a time (CPU/memory
  limited), each variant in its own fresh run dir under `data/outputs/`
  (copied keyframes + `calibration.json`, never a re-run over a stale dir).
- Scoring (per handoff §12, no exceptions): **RAW COLMAP points**
  (`colmap/sparse/0`), never `sparse_points.npz`; **orientation-aware Sim(3)**
  alignment (frame-0 orientation + centre-spread scale — centres-only
  Umeyama is degenerate on straight flights); GT surface samples
  (100k, `sample_surface`, seed 7). Completeness is low by construction on
  this scene (GT samples include unobserved backs of buildings), so
  accuracy / precision on observed surfaces is the primary signal.
  Scoring helpers: `~/workspace/tmp/score_raw.py` (flyover, 60-frame flight)
  and `~/workspace/tmp/score_generic.py` (same method, GT poses from a
  `.npz`, so it also handles partial registration and the loop clip).
  The two scorers agree exactly where both apply (e.g. locked-true:
  0.486 m / 0.5247 in both).
- No dense runs were done in either series (SfM-only experiments), so there
  is no fused cloud to score here. The handoff §12 fused-cloud numbers
  (true focal: median 0.46 m, precision@0.5 m 0.55) are from the earlier
  session and were not re-run.
- Focal-gate verdicts use the 10% gate (`SfmConfig.focal_drift_max_rel`,
  default 0.10) against the run's prior focal. The gate flags, never clamps.

## Series 1 — Loop closure: sequential vs exhaustive on a revisit flight

### The clip

New renderer `scripts/render_loop_video.py` (104 frames, 480x270, true
focal 265 px), artefacts `data/benchmark/loop.mp4`,
`data/benchmark/loop.depth.npy`, `data/benchmark/loop.poses.npz`:

- forward leg: 45 frames, x = 0..59.84 m, y = 0, heading +x (identical
  convention to `flight()` in the flyover);
- U-turn: 14 frames, semicircle radius 3.5 m centred at (59.84, 3.5),
  heading sweeping 0 -> pi through +y;
- return leg: 45 frames, x = 59.84..0 m, y = 7, heading -x.

One subtlety found while building it: `flight()`'s `yaw` parameter does
**not** change the view direction (the forward vector is invariant under
it), so the turn/return rotations were built as a rigid rotation of the
whole camera rig about the world z-axis (`R = R_fwd @ rot_z(-beta)`).
That is what a fixed-mounted camera on a yawing drone does — but it means
the return-leg images are upside-down relative to the forward leg (the
synthetic camera's image-up points laterally). SIFT is rotation-invariant,
so matching is unaffected in principle, and the results below bear that
out (104/104 registered under both matchers). GT poses verified
numerically before rendering: forward +x, mid-turn +y, return -x, centre
steps continuous (1.36 m legs, 0.73 m turn). Keyframes: 104/104 selected
(`data/outputs/run_20261116_092338`).

Both variants: free focal (`configs/video_flyover.yaml`,
`refine_intrinsics=true`), prior focal 384 px, identical except matcher.

| variant | solved f | gate (vs 384) | reg | reproj | centre err med (p95) | sparse med acc (p90) | prec@0.5 | prec@1 | pts | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| sequential (`data/outputs/loop_seq`) | 264.6 px | -31.1% **EXCEEDED** | 104/104 | 0.280 px | 0.042 m (0.060) | 0.456 m (0.822) | 0.5646 | 0.9740 | 13,994 | 84 s |
| exhaustive (`data/outputs/loop_exh`) | 264.4 px | -31.1% **EXCEEDED** | 104/104 | 0.290 px | 0.032 m (0.072) | 0.457 m (0.820) | 0.5654 | 0.9728 | 15,103 | 205 s |

Per-leg centre errors (sequential): forward 0.042 m, turn 0.030 m, return
0.042 m; (exhaustive): forward 0.035 m, turn 0.038 m, return 0.017 m —
no leg is degraded, i.e. the return pass is not drifting relative to the
forward pass under either matcher.

### Verdict: exhaustive loop closure did NOT help on this revisit flight

The numbers are indistinguishable: solved focal 264.6 vs 264.4 px, sparse
median accuracy 0.456 vs 0.457 m, precision@0.5 m 0.565 vs 0.565. The only
differences are +8% points (13,994 -> 15,103) and 2.4x wall time
(84 s -> 205 s) for exhaustive. The loop was already closed by the
**continuous turn**: sequential matching walks the overlap unbroken from
the forward leg, through the turn, onto the return leg, so no cross-leg
pair is ever needed.

The bigger finding is what the turn did to the focal: with viewpoint
variation in the flight, free refinement recovered the true focal
(264.6 px vs true 265 px) from a prior 45% off — the same free
refinement that fails catastrophically on pure forward flight (Series 2).
The gate flags -31.1% here, but this is the documented caveat: the drift
is BA *fixing* a wrong prior, not the camera absorbing a stretch. The
gate makes it visible; the trajectory/point scores say the result is
correct.

Caveats: this loop has unbroken overlap through the turn. A discontinuous
revisit (two separate passes, a cut, a teleport) has no sequential bridge
and might still benefit from exhaustive matching — that case was not
tested, and pycolmap 3.12.6 exposes no pair-list matcher, so exhaustive
is the only loop-closure mode available. Do not generalise "exhaustive is
useless" beyond continuous-revisit flights.

## Series 2 — Focal locked vs free on the forward flyover (re-run, post-PR#5)

Clip: existing `data/benchmark/flyover.mp4`, 60 frames, true f = 265 px.
All variants sequential matcher, identical keyframes (base
`data/outputs/run_20261116_083045`, 60 images). The focal is pinned via
the run's stored `calibration.json` (what the SfM CLI actually reads):
384 px = the 0.8*width video prior, 265 px = the true focal. Locked
variants use `~/workspace/tmp/config_locked.yaml`
(`refine_intrinsics=false`); free variants use
`configs/video_flyover.yaml` (`refine_intrinsics=true`).

| variant | solved f | gate | reg | reproj | centre err med | sparse med acc (p90) | prec@0.5 | prec@1 | pts / models |
|---|---|---|---|---|---|---|---|---|---|
| (a) free, run 1 (`data/outputs/fly_free1`) | 417.0 px | +8.6% within | 60/60 | 0.280 px | 0.038 m | 3.488 m (4.484) | 0.0000 | 0.0000 | 7,421 / 1 |
| (a) free, run 2 (`data/outputs/fly_free2`) | 367.1 px | -4.4% within | **28/60** | 0.273 px | 0.016 m * | 2.856 m (3.480) | 0.0000 | 0.0000 | 3,489 / **9** |
| (b) locked prior 384 (`data/outputs/fly_lock384`) | 384.0 px | 0.0% within | 60/60 | 0.282 px | 0.046 m | 3.061 m (3.821) | 0.0000 | 0.0000 | 6,997 / 1 |
| (c) locked true 265 (`data/outputs/fly_lock265`) | 265.0 px | 0.0% within | 60/60 | 0.281 px | 0.066 m | **0.486 m (0.840)** | **0.5247** | **0.9661** | 7,631 / 1 |

\* Free run 2 registered only frames 10-37, so its centre error is over
that subset only and is not comparable to the full-flight numbers; the
headline for that run is the registration collapse, not the 0.016 m.

### Comparison with handoff §12 (old numbers) — confirmed, with two corrections

- **Locked true focal: confirmed.** Old: sparse 0.47 m, prec@0.5 0.54,
  centre 0.040 m. Now: 0.486 m, 0.5247, 0.066 m. Same conclusion: only
  the true focal gives a sub-metre model.
- **Free focal instability: confirmed and worse than reported.** Old:
  447 px then 492 px on identical data. Now: 417 px then 367 px — a
  third and fourth distinct value. The instability is not a two-sample
  fluke; the solved focal wanders by tens of percent run to run.
- **New: free refinement can collapse registration, not just the focal.**
  Run 2 formed 9 separate models and registered only 28/60 frames
  (frames 10-37). The handoff's free runs both registered 60/60, so this
  failure mode was not previously recorded. Reprojection error stayed
  0.27 px throughout — it detects none of this.
- **Locked-prior centre error: corrected.** Old table says 1.19 m; this
  re-run measures 0.046 m with the orientation-aware Sim(3) alignment.
  The sparse accuracy agrees qualitatively (old 3.84 m, now 3.06 m;
  prec@0.5 0.001 vs 0.0000), so the geometry conclusion is unchanged —
  but the old 1.19 m centre figure was likely a scoring artefact of the
  stale-cache era (cf. §12 Bug B) and should not be quoted. Note the
  general pattern the corrected numbers make even clearer: **trajectory
  centre error is small (0.016-0.066 m) in every variant, including the
  ones whose points are 3+ m off.** Bas-relief again — a good-looking
  trajectory and a 0.28 px reprojection error say nothing about depth
  accuracy on forward flight.
- Locked prior 384: sparse 3.06 m / prec@0.5 0.0000 — locking the wrong
  focal is no better than free refinement on accuracy; it is only more
  deterministic (60/60, one model, same answer every run).

## Decision: `configs/video_flyover.yaml` — DO NOT change `refine_intrinsics`

**Keep `refine_intrinsics: true` (the current default). No config change.**

Reasoning, strictly from the evidence above:

1. On pure forward flight, **neither** setting gives an accurate model
   without a true focal: free gives 2.9-3.5 m median sparse error,
   locked-wrong-prior gives 3.1 m. Flipping the default to `false`
   would trade an unstable catastrophic outcome for a stable
   catastrophic one — no accuracy is gained.
2. Locking removes the one mechanism that can ever be right: on the
   loop clip, free refinement recovered the true focal (264.6 px) from
   a 45%-wrong prior, which a locked 384 px could never do. Real drone
   flights with turns / oblique views have the viewpoint variation that
   makes this possible; locking the 0.8*width guess would guarantee a
   wrong focal on all of them.
3. Locking also silences the focal gate (locked runs show 0.0% drift
   by construction), removing the only visibility into the
   focal/depth ambiguity.
4. The free setting's real defect on forward flight is instability
   (focal 367-492 px across runs; one registration collapse to 28/60),
   which locking does not fix — it only hides the focal half of it.

What would change this decision:

- A **real focal** for the footage (EXIF, sensor spec, or a calibration
  capture): then lock it (`refine_intrinsics=false` + the measured focal
  in the run calibration) — Series 2(c) shows that is the only setting
  that produces a sub-metre model on forward flight.
- Evidence on real (non-synthetic, wider-FOV, oblique) footage that
  locked-prior beats free on accuracy, not just determinism.
- A focal prior with a bounded error (e.g. per-camera-model defaults
  from a device database) instead of the 0.8*width guess.

Practical consequences for real footage: the pipeline's accuracy on
forward flight is focal-limited, not matcher- or dense-limited. Until a
real focal is available, treat any forward-flight model as
scale/depth-suspect even when registration is 100% and reprojection
error is ~0.3 px, and prefer flights with viewpoint variation (turns,
oblique views) — Series 1 shows that variation is what lets BA find the
true focal at all.

## Observations (not fixes — out of scope for this task)

- The SfM CLI prints contradictory messages on a low-registration run:
  `fly_free2` printed both "Warning: Registration rate below 85% KPI"
  and "SfM Quality Gate Passed: Registration 46.7% >= 85%". The
  "Passed" line sits in the focal-gate else-branch and is not actually
  gated on registration. Left as-is (no `src/` changes in this task);
  worth a one-line fix by whoever next touches `cli.py`.
- `~/workspace/tmp/score_raw.py` crashes on partial registration
  (assumes 60 registered centres). `score_generic.py` in the same
  directory handles it; the flyover GT poses it needs are now saved as
  `data/benchmark/flyover.poses.npz` (generated from `flight()`, same
  parameters as the renderer).

## Artefacts

- New: `scripts/render_loop_video.py`; `docs/loop_focal_experiments.md`
  (this file); `data/benchmark/loop.mp4`, `loop.depth.npy`,
  `loop.poses.npz`; `data/benchmark/flyover.poses.npz`.
- Run dirs (untracked, under `data/outputs/`): `run_20261116_092338`
  (loop keyframes base), `loop_seq`, `loop_exh`, `fly_free1`,
  `fly_free2`, `fly_lock384`, `fly_lock265`. Each holds `poses.json`,
  `colmap/sparse/0`, `sfm.log` (full CLI output incl. focal-gate
  verdict) and `calibration.json` (the pinned prior focal).
- Scoring helpers (untracked, outside the repo):
  `~/workspace/tmp/score_generic.py`, `~/workspace/tmp/score_raw.py`,
  `~/workspace/tmp/check_traj.py`, `~/workspace/tmp/config_locked.yaml`,
  `~/workspace/tmp/prep_run.sh`.
