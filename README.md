# uav3d

Single-pass UAV video → 3D pipeline (SIH 26158 SP-1).

## Quick start: video → 3D model

One command, **video only** as input (no telemetry or ground-truth files required):

```bash
uv run uav3d reconstruct --video /path/to/video.mp4 --max-frames 60
```

It creates a single run directory (`data/outputs/run_<timestamp>/`, or `--data-dir`) with:

| Artifact | Contents |
| --- | --- |
| `model.ply` | Colored 3D point cloud (CloudCompare / MeshLab / Blender) |
| `model/` | COLMAP sparse model (`cameras.txt`, `images.txt`, `points3D.txt`, …) |
| `trajectory.ply` | Camera flight path (vertices + edges, ordered by frame id) |
| `poses.json` / `keyframes.json` | SP-1 machine-readable outputs |
| `reconstruction_summary.json` | Versioned manifest: registration stats, artifact paths, scale caveat |
| `colmap_native.log` | Raw COLMAP C++ log (captured off the console) |

Stages: keyframe selection (quality-scored; `--max-frames` is a hard budget on
evaluated *and* selected frames) → COLMAP incremental SfM (pycolmap, automatic
PINHOLE intrinsics unless `--calibration-file` is given) → 3D model export.
Telemetry sync and ATE validation are skipped with an INFO note on video-only
runs; the summary records `null` for them.

Notes:

- Monocular SfM scale is arbitrary until aligned to a metric reference
  (GPS/GT) — the manifest carries this caveat.
- COLMAP runs in single-model mode by default (`sfm.multiple_models: false`):
  the run stops after the main reconstruction instead of spending hours
  retrying secondary-model initialization over leftover frames (the
  `Discarding reconstruction due to insufficient size` tail).
  `sfm.init_num_trials` caps initialization attempts.
- Long videos: lower `--max-frames` to bound runtime. If the budget leaves
  keyframes too far apart for SfM initialization, the command fails fast with
  an actionable error.

## Other commands

| Command | Purpose |
| --- | --- |
| `uav3d inspect` | Validate the data source and show calibration |
| `uav3d keyframes` | Quality-based keyframe selection (`--video` or Zurich dataset; `--max-frames` budget) |
| `uav3d sfm` | pycolmap incremental SfM (produces `poses.json`) |
| `uav3d sync` | GPS/telemetry ↔ frame synchronization (Zurich dataset) |
| `uav3d validate` | ATE validation against ground-truth AGL (Zurich dataset) |

## Testing

```bash
uv run pytest
```

The suite renders a deterministic synthetic parallax video on the fly
(`scripts/make_parallax_video.py`), so no downloaded data is required.

### Test videos for judging 3D model quality

See [TEST_VIDEOS.md](TEST_VIDEOS.md). TL;DR: `data/raw/test_videos/` holds a
synthetic city flyover **with 3D ground truth** (`scripts/make_city_flyover_video.py`;
compare `model.ply` against `city_flyover.gt_model.ply`, and `--gt-file` gives a
true-scale ATE report) plus a known-good baseline video in the Zurich layout,
public dataset links, and a field-capture checklist for why a given
single-pass video may fail SfM (motion blur, overlap, texture, pure rotation).

## Attribution

Zurich Urban MAV Dataset:
Majdik, A. L., Till, C., & Scaramuzza, D. (2017). The Zurich Urban Micro Aerial Vehicle Dataset. The International Journal of Robotics Research (IJRR).
All Zurich dataset outputs are labeled "replayed public dataset".
