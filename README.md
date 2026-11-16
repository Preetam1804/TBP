# uav3d

Single-pass UAV video → 3D pipeline (SIH 26158 SP-1).

## Running it

```bash
# full pass on one video (keyframes -> SfM -> dense -> mesh), with the
# trustworthiness report at the end
.venv/bin/python scripts/run_video_pipeline.py --video data/city_flyover.mp4 \
    --config configs/video_flyover.yaml --mesh

# tests (the two open3d-only modules need system libEGL; skip them if
# open3d cannot import in your container)
.venv/bin/python -m pytest -q --ignore=tests/test_completion.py --ignore=tests/test_lod2.py
```

## Supplying a known focal length for a video

Without one, the prior focal is a guess — `0.8 * width` — and on forward
flight the SfM solver can spend the focal/depth ambiguity on the focal
instead of the geometry, stretching the model. Supply the real focal
(from the camera spec, a calibration capture, or EXIF) and the run is both
more accurate and auditable: the SfM summary and the pipeline report print
which source the prior focal came from next to the focal-drift gate verdict.

Precedence, highest first:

1. **Explicit override** — `--focal-px <pixels>` on `uav3d keyframes`,
   `uav3d sfm`, or `scripts/run_video_pipeline.py`, or `source.focal_px` in
   the config. Recorded as `video_config_override`. Must be positive and
   within `0.2*width..5*width`; anything else fails the run loudly.
2. **Sidecar** — a JSON file named `<video-stem>.meta.json` next to the
   video, containing a numeric `focal_px` field, e.g. for
   `city_flyover.mp4` the file `city_flyover.meta.json` with
   `{"focal_px": <focal length in pixels>}` (other fields such as
   frames/fps are ignored).
   Recorded as `video_meta_sidecar:<filename>`. A missing, malformed, or
   out-of-range sidecar value is logged as a warning and ignored — the run
   falls back to the guess, never crashes. Note: the sidecar is looked for
   only next to the video file — a meta file kept elsewhere (e.g. in a
   `test_videos/` folder) is not picked up; use `--focal-px` instead.
3. **Guess** — `0.8 * width`, recorded as `video_estimate_<filename>`.
   This is the unchanged default for videos with no known focal.

The focal is stored in the run's `calibration.json` (`source` field), so
re-running SfM on an archived run keeps the same prior. `--focal-px` on
`uav3d sfm` also overrides the stored prior focal for that run. Supplying
a known focal does not change `sfm.refine_intrinsics`: with the default
`true` in `configs/video_flyover.yaml` the solver may still refine the
focal, and the gate reports how far it moved from the known prior.

## Current status (2026-11-16)

SP-1 phases 0-5 are complete, plus the accuracy work: focal-drift gate,
stale-cache fixes, matcher wiring (`sequential` / `exhaustive`),
sky guard for the monocular-prior fill, Phase 4 coverage reporting
(`dense/coverage_report.json`), and a known-focal path for videos
(CLI `--focal-px` > config `source.focal_px` > sidecar `<stem>.meta.json`
> `0.8*width` guess). Full test suite: **241 passed, 6 skipped**
(the skips are the Zurich-dataset smoke tests, absent from a fresh clone;
the two open3d-only modules pass on a container with system libEGL).

City flyover (`data/city_flyover.mp4`, AI-generated, 600 frames): the
focal was verified at **1152 px** (free-focal SfM solved 1112–1163 px)
and the full pass was run locked and free — both 250/250 registered,
0.369 px reprojection, ~2.98M fused points, textured model built, and
the camera path follows the GT path shape to ~0.10 m RMS. But the
supplied GT model does **not** match this video (it shows a different
city layout), so there is no cloud-to-GT accuracy number for it, the
absolute scale is unverified, and coverage verdict is "holes" (14/90
footprint cells empty). Treat distances/heights as suspect. See
`docs/HANDOFF_NEXT_SESSION.md` §14 for the full numbers.

Honest limitation, measured on synthetic ground truth (true focal 265 px):
on pure forward flight the pipeline is **focal-limited** — with the free
focal default the full-pass model scores ~3.6 m median error even at
100% registration and 0.28 px reprojection error, while the same clip
with the true focal locked scores 0.46 m. The focal gate flags this
(+13.5% drift on the synthetic full pass). For real drone footage, supply
a real focal (EXIF / sensor spec / calibration capture) and lock it.
The monocular prior stays opt-in (it failed its promotion gate).
See `docs/HANDOFF_NEXT_SESSION.md` §13–14 for the full numbers.

## Attribution
Zurich Urban MAV Dataset:
Majdik, A. L., Till, C., & Scaramuzza, D. (2017). The Zurich Urban Micro Aerial Vehicle Dataset. The International Journal of Robotics Research (IJRR).
All Zurich dataset outputs are labeled "replayed public dataset".
