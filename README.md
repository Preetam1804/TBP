# uav3d

Single-pass UAV video → 3D pipeline (SIH 26158 SP-1).

Ingest a video or image sequence, select sharp keyframes, reconstruct camera poses with
visual SfM, attach onboard telemetry, and validate the trajectory against ground truth.

```
video / image sequence
        │
        ▼
   ImageSource ──► FrameRecord ──► KeyframeSelector ──► SfMRunner ──► TelemetrySynchronizer ──► AteValidator
   (Zurich MAV,    (µs timestamps)  (blur/clipping/      (pycolmap      (image-ID join or        (7-DoF Umeyama
    MP4 video)                       ORB + overlap        incremental    timestamp matching)      Sim(3) vs GT)
                                     guard)               SfM)
```

## Scope (SP-1)

**In:** video and image-sequence ingestion, Zurich MAV dataset adapter, keyframe selection,
telemetry mapping, pycolmap SfM, pose export, ATE validation. Local batch CLI, no server.

**Out:** depth estimation, fusion, meshing, texturing, frontend, database, auth, cloud (SP-2+).

## Install

Requires Python 3.11 or 3.12.

```bash
uv sync                       # or: python -m venv .venv && .venv/bin/pip install -e ".[dev]"
uv run pytest                 # 99 tests
uv run ruff check .
```

`opencv-python` needs system `libGL`/`libglib` on Linux (`apt-get install libgl1 libglib2.0-0`),
or install `opencv-python-headless` instead for batch-only machines.

## Quickstart on the sample dataset

The repository deliberately does not ship image data. To get a runnable end-to-end example,
generate the synthetic stand-in dataset (mimics the Zurich directory layout):

```bash
uv run python scripts/generate_sample_zurich.py          # writes data/raw/zurich_mav + provenance marker
uv run uav3d inspect   --config configs/zurich_baseline.yaml
uv run uav3d keyframes --config configs/zurich_baseline.yaml
uv run uav3d sfm       --run-dir data/outputs/run_<timestamp>
uv run uav3d sync      --run-dir data/outputs/run_<timestamp>
uv run uav3d validate  --run-dir data/outputs/run_<timestamp>
```

Each `keyframes` run creates `data/outputs/run_<UTC timestamp>/`; `sfm`, `sync` and `validate`
default to the newest run when `--run-dir` is omitted.

### Video input

```bash
uv run python scripts/make_synthetic_video.py --frames 50        # data/interim/synthetic.mp4 (+ manifest.json)
uv run uav3d inspect   --video data/interim/synthetic.mp4
uv run uav3d keyframes --video data/interim/synthetic.mp4
uv run uav3d sfm       --video data/interim/synthetic.mp4 --run-dir <run_dir>
uv run uav3d sync      --video data/interim/synthetic.mp4 --run-dir <run_dir>
```

## CLI reference

| Command | Purpose | Key options |
| --- | --- | --- |
| `uav3d inspect` | Report source metadata, resolution, timestamps, calibration | `--config`, `--data-dir`, `--video` |
| `uav3d keyframes` | Select sharp keyframes and write `keyframes.json` + `sfm/images/` | `--config`, `--data-dir`, `--video` |
| `uav3d sfm` | Run pycolmap incremental SfM and export `poses.json` | `--run-dir`, `--overlap`, `--focal-px` |
| `uav3d sync` | Join telemetry to poses → `telemetry.json`, `poses_enriched.json` | `--run-dir`, `--mode auto\|zurich\|video` |
| `uav3d validate` | Umeyama ATE vs ground truth → `ate_report.json`, `SP1_REPORT.md`, plots | `--run-dir`, `--gt-file` |

## Outputs per run directory

| File | Contents |
| --- | --- |
| `keyframes.json` | Selected frames, timestamps, quality metrics |
| `poses.json` | `cam_from_world` poses (quaternion `wxyz`), camera block with intrinsics provenance, metrics |
| `telemetry.json` | Synchronized GPS records and sync status |
| `poses_enriched.json` | Poses plus `telemetry_recorded` flags (`poses.json` is never mutated) |
| `ate_report.json`, `SP1_REPORT.md`, `*.png` | ATE metrics, ground-truth provenance, trajectory plots |
| `debug/contact_sheet.jpg` | Visual check of the selected keyframes |
| `colmap/` | COLMAP database and sparse reconstruction |

## Conventions

- Timestamps are integer microseconds everywhere.
- Poses use COLMAP `cam_from_world`; quaternions are `(qw, qx, qy, qz)`; camera center = `-Rᵀt`.
- SfM output has arbitrary scale/rotation until Sim(3) alignment against ground truth.
- Every JSON output carries `schema_version` and a `dataset_label`.
- Ground truth (`GroundTruthAGL.csv`) is **evaluation-only** and never an SfM input; a test
  asserts it is imported only inside `validation/`.

## Ground truth trust boundary

`uav3d validate` will not guess which ground truth applies to a run:

- dataset runs (`zurich_*` frame ids) discover `GroundTruthAGL.csv` from the dataset adapter;
- any other run (e.g. `video_*`) **refuses to validate** without an explicit `--gt-file`, because
  matching records by numeric frame suffix against an unrelated CSV would produce a passing
  report that means nothing;
- the report records the file used, how it was selected, and whether the ATE threshold was exceeded.

## Video intrinsics

A video file carries no trustworthy intrinsics, so the focal length is either:

1. supplied — `--focal-px` or `sfm.video_focal_px` → intrinsics locked; or
2. assumed — `sfm.video_fov_deg` (default 60°) → focal refinement is attempted, then **gated**:
   the refined camera is kept only if its implied horizontal FoV is inside
   `sfm.video_min_fov_deg`/`video_max_fov_deg`, focal lengths are near-isotropic, and the
   principal point stays near the image centre. If it fails, the reconstruction is redone with
   the assumed intrinsics locked and the rejection is reported.

The camera model is `SIMPLE_PINHOLE`, so bundle adjustment cannot drift `fx` and `fy` apart.

## Determinism

Keyframe selection is deterministic: `keyframes.json` is byte-identical across runs on the same
input (verified). pycolmap's multithreaded bundle adjustment can vary the last digit of the mean
reprojection error, so SfM metrics are reproducible to ~0.001 px rather than bit-exact.

## Dataset

**Zurich Urban MAV Dataset** — Majdik, A. L., Till, C., & Scaramuzza, D. (2017), *The Zurich Urban
Micro Aerial Vehicle Dataset*, The International Journal of Robotics Research (IJRR). License:
unrestricted; attribution required.

Layout expected by the adapter: `MAV Images/` (JPEGs), `Log Files/OnbordGPS.csv`,
`Log Files/GroundTruthAGL.csv`, `calibration_data.npz`. 81,169 images in total — dev runs use a
contiguous slice of 150–300 images.

### Provenance labelling

`scripts/generate_sample_zurich.py` writes `dataset_provenance.json` into the directory it
populates. When that marker is present, every downstream `dataset_label` is rewritten to
`synthetic sample dataset … NOT the ETH Zurich Urban MAV dataset`, so locally generated numbers
cannot be mistaken for published-dataset results. The generator also refuses to write into a
non-empty directory without `--force`.

## Development

```bash
uv run pytest -q                 # 99 tests, ~30 s
uv run ruff check .              # lint (config lives in pyproject.toml)
python scripts/check_pycolmap_api.py   # print the installed pycolmap API surface
```

CI runs ruff plus the full test suite on Python 3.11 and 3.12 (`.github/workflows/ci.yml`).

## Licence

No project licence has been selected yet — that is a decision for the repository owner. The
attribution requirement for the Zurich Urban MAV Dataset (IJRR 2017) applies to any use of that
dataset regardless of this project's licence.
