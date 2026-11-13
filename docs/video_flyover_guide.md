# Video ingestion: what actually happens to `city_flyover`

Status: measured on a synthetic stand-in, because the real file is not in this
workspace. Every number below comes from a full run of this repository's own
`inspect → keyframes → sfm → dense` chain.

## Short answer

Yes — running the pipeline on `city_flyover` produces a **new** 3D model. It is
a genuinely different artifact from whatever produced the earlier screenshots:
a fused multi-view-stereo point cloud with per-pixel depth maps, not a sparse
SfM cloud.

"No, not automatically more accurate" is the honest half. On forward-flying
monocular video there is **no scale information at all**, so the model comes out
in SfM units with `scale.status = unverified_monocular_scale`. It cannot be
metrically accurate in the sense the Zurich run was, because nothing in the
footage fixes a metre.

What *is* verifiable is geometric accuracy — and that turned out to be good, but
only after fixing a real defect described below.

## The defect this run found and fixed

`_handle_sfm_command` in `src/uav3d/cli.py` overrode the calibration for every
video run:

```python
fx = fy = float(w) * 0.9          # ignored the source's own estimate
cfg.sfm.refine_intrinsics = True  # forced on, ignoring the YAML setting
```

Two separate problems:

1. **The config was ignored.** Setting `refine_intrinsics: false` had no effect
   on video runs; the CLI re-enabled it unconditionally.
2. **`PINHOLE` refines fx and fy independently.** On forward flight the observed
   image velocity is `f · d⁻¹`, so focal length and scene depth trade off almost
   freely along the flight axis and the solver is free to spend its leverage on
   the *aspect ratio* instead. It did, every time:

   | run | camera model | fx (px) | fy (px) | fx/fy |
   |---|---|---|---|---|
   | old CLI path | `PINHOLE`, refined | 599.9 | 354.5 | **1.692** |
   | old CLI path | `PINHOLE`, refined | 505.5 | 321.9 | **1.571** |
   | new path | `SIMPLE_PINHOLE` | 342.2 | 342.2 | 1.000 |
   | ground truth | — | 265.0 | 265.0 | 1.000 |

   A 1.69× aspect ratio is not a camera; it is the optimiser absorbing
   degenerate degrees of freedom. Every pixel is back-projected along the wrong
   ray, so the scene is stretched along one image axis.

### The fix

`src/uav3d/cli.py` now:

* takes calibration from the source (honouring `VideoFileSource.calibration()`),
  rescaling only if the frame size on disk disagrees with the config;
* forces `SIMPLE_PINHOLE` for video, which ties both axes to **one** focal — the
  correct model for square sensor pixels, and one that cannot express the
  aspect-ratio failure;
* **respects** `sfm.refine_intrinsics` from the config instead of forcing it.

`src/uav3d/cli.py` also refuses to continue when every frame is rejected by the
quality gate, naming the offending threshold instead of failing later with an
opaque error, and the keyframes stage now writes `calibration.json` into the run
directory. `sfm` and `dense` read that file, so they no longer need the original
video to be present — previously `sfm` on an archived run failed with
"Video file not found" because it rebuilt the source purely to ask for
intrinsics. (Verified by running the full chain with the video moved away.)

## Measured accuracy (synthetic street canyon, exact analytic depth)

60 frames, 480×270, true focal 265 px, 1.36 m between frames, 12 m altitude. The
scene is a procedurally textured street canyon, so every pixel has exact Z-depth
ground truth. Depth is scored per frame in each camera's own frame, with a
**single global scalar** for the monocular scale — no per-frame fitting, no
3-D alignment (a straight flight leaves roll unconstrained and any 3-D
similarity fit can silently rotate the cloud).

| metric | value |
|---|---|
| frames registered | 60 / 60 (100 %) |
| mean reprojection error | 0.281 px |
| path straightness | 3.65 mm off the best-fit line |
| step spacing | 1.3605 ± 0.0037 m (true 1.3600 m, **0.03 % error**) |
| trajectory scale vs. depth scale | agree to **0.2 %** |
| depth error, median | **0.147 m** |
| depth error, p90 | 0.612 m |
| relative depth error, median | **0.7 %** |
| within 0.5 m | 86.6 % |
| within 1.0 m | 95.2 % |

The two independent scale estimates — one from camera motion, one from depth —
agreeing to 0.2 % is the strongest evidence the reconstruction is internally
consistent. A wrong focal length would have separated them.

### What the focal length does and does not explain

This is worth stating plainly because it is counter-intuitive, and I got it
wrong at first:

| camera model | fx/fy | depth median | within 1 m |
|---|---|---|---|
| `SIMPLE_PINHOLE`, locked | 1.000 | 0.149 m | 95.1 % |
| `SIMPLE_PINHOLE`, refined | 1.000 | 0.152 m | 95.0 % |
| `PINHOLE`, refined | **1.442** | 0.148 m | 95.0 % |

**Per-view depth is essentially immune to the focal error.** Focal length and
scene scale are the same unknown in forward flight, so one scalar absorbs the
error and depth stays accurate. That is why the anisotropy did *not* wreck the
depth maps, and why an earlier claim in this project that it "warps every depth"
was wrong.

The anisotropy instead corrupts **3-D shape**: it stretches the cloud along one
image axis, which no single scalar can undo. Depth maps are the metric artifact
and they are good; a point cloud viewed at a glance shows that stretch. This is
exactly why the camera model still had to be fixed, and why the fix is justified
by the aspect ratio being physically impossible (1.69) rather than by a depth
number.

## On the actual `city_flyover` footage

Three things must be said before trusting any output from it.

1. **The file is not in this workspace.** `data/` does not exist here and there
   is no video anywhere in the tree. Give the path (or drop the file in
   `data/`) and the commands below run as-is.

2. **The focal length is a guess.** `VideoFileSource.calibration()` uses
   `fx = fy = 0.8 · width`, i.e. an assumed ~64° horizontal field of view. On
   this synthetic scene the true focal was 0.55 · width and the pipeline
   recovered 0.71 · width — a 29 % error that only a shared-focal refinement
   could reduce. On real footage, `SIMPLE_PINHOLE` with refinement on lets the
   focal settle on whatever is self-consistent with the motion; if the geometry
   is degenerate (pure forward flight over a flat scene) the focal will instead
   absorb the error and be meaningless. Check `fx/fy` stays 1.0, and treat the
   focal itself as an output, not a known.

3. **Metres are unavailable.** With no GNSS and no surveyed object in frame,
   `scale.status` will be `unverified_monocular_scale` and the units are
   arbitrary. If `city_flyover` carries telemetry (the repo's `uav3d sync`
   stage can ingest it), or if two points of known separation are visible, the
   scale can be anchored. Otherwise accuracy is verifiable as *shape* only.

## Running it

```bash
export PYTHONPATH=src
VIDEO=/path/to/city_flyover.mp4

python -m uav3d.cli inspect   --video "$VIDEO"
python -m uav3d.cli keyframes --video "$VIDEO" --config configs/video_flyover.yaml
RUN=$(ls -d data/outputs/run_* | tail -1)
python -m uav3d.cli sfm   --config configs/video_flyover.yaml --run-dir "$RUN"
python -m uav3d.cli dense --config configs/video_flyover.yaml --run-dir "$RUN" \
    --max-width 1600 --ref-stride 1
```

`--video` belongs on `keyframes` only; that stage is the one that reads the
video. `sfm` and `dense` take `--run-dir` and work from the files the keyframes
stage wrote. Passing `--run-dir` explicitly matters: without it `sfm` silently
picks the newest `run_*` directory, which may not be the one you just made.

`configs/video_flyover.yaml` is provided. Budget from the measurements above:
SfM is ~0.9 s per frame at 480×270 and the dense stage ~1.5 s per reference view,
both dominated by pixel work, so scaling to 150 frames at 1600×900 lands near the
project's ~1 h allowance on CPU. Start with `--max-width 960 --ref-stride 2` to
get a complete model quickly, then re-run at full width once the registration
rate and reprojection error look sane.

Never judge a first run by how the cloud looks. Judge it by registration rate
(≥ 85 %), mean reprojection error (≤ 1.5 px) and `fx/fy == 1.0`.

## Validation tooling added

| script | purpose |
|---|---|
| `scripts/render_street_video.py` | renders the analytic street canyon to MP4 + per-frame depth |
| `scripts/eval_video_depth.py` | gauge-free per-frame depth scoring with two independent scale estimates |
| `scripts/ab_camera_model.py` | A/B of `SIMPLE_PINHOLE` vs `PINHOLE`, locked vs refined |
| `scripts/eval_shape_anisotropy.py` | back-projection shape error per camera model |

Known limitation of `eval_shape_anisotropy.py`: its residual is measured across
frames with a shared similarity transform, so it also contains the per-frame
focal/scale ambiguity. Use `fx/fy` and the two scale estimates as the primary
evidence; the shape script is corroborating only.
