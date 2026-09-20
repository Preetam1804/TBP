# uav3d

Single-pass UAV video → 3D pipeline (SIH 26158 SP-1).

## Attribution
Zurich Urban MAV Dataset:
Majdik, A. L., Till, C., & Scaramuzza, D. (2017). The Zurich Urban Micro Aerial Vehicle Dataset. The International Journal of Robotics Research (IJRR).
All Zurich dataset outputs are labeled "replayed public dataset".

## Usage

```bash
uv sync                                  # or: uv sync --extra dev
python scripts/generate_sample_zurich.py # synthetic 50-frame dataset -> data/synthetic/zurich_mav

# Synthetic sample dataset (refuses to write into a non-empty/real dataset directory):
DS="--dataset-dir data/synthetic/zurich_mav --data-dir data"
uav3d inspect   --config configs/zurich_baseline.yaml $DS
uav3d keyframes --config configs/zurich_baseline.yaml $DS
uav3d sfm       --config configs/zurich_baseline.yaml $DS
uav3d sync      --config configs/zurich_baseline.yaml $DS
uav3d validate  --config configs/zurich_baseline.yaml $DS
```

For the real dataset, set `source.dataset_dir` in the config (or pass `--dataset-dir <path>`) and
run the same commands without `$DS` — `data/raw/zurich_mav` is the config default.

Each stage writes into the newest run directory under `<data_dir>/outputs/run_<timestamp>/`;
`sync` and `validate` pick the most recent run that actually contains its required artifacts,
and `--run-dir` overrides the choice.

### Dataset placement

The real [Zurich Urban MAV dataset](https://rpg.ifi.uzh.ch/zurichmavdataset.html) is evaluation-only
and is not stored in this repository. Place it in the AGZ layout and point the config at it:

```
<dataset_dir>/
  MAV Images/            # *.jpg frames
  Log Files/             # OnbordGPS.csv, GroundTruthAGL.csv
  calibration_data.npz
```

- `source.dataset_dir` (or `--dataset-dir`) selects the dataset; it is independent of `--data-dir`,
  which only relocates outputs.
- OnbordGPS.csv values are normalised defensively: 1E7-scaled lat/lon, 1E3-scaled altitude and
  millisecond/microsecond timestamps are all detected and recorded per row in `raw_fields`.
- Missing/invalid `calibration_data.npz` fails loudly unless `source.allow_default_intrinsics: true`.

### Video runs

```bash
uav3d keyframes --config configs/zurich_baseline.yaml --data-dir data --video clip.mp4
uav3d sfm       --config configs/zurich_baseline.yaml --data-dir data --video clip.mp4
uav3d sync      --config configs/zurich_baseline.yaml --data-dir data --video clip.mp4    # uses <video_dir>/manifest.json
uav3d validate  --config configs/zurich_baseline.yaml --data-dir data --video clip.mp4    # uses <video_dir>/ground_truth_agl.csv
```

`ingest.start_frame` / `ingest.max_frames` bound the decoded frame window.

### Notes

- `opencv-python-headless` is used deliberately: `opencv-python` requires `libGL.so.1`, which is
  absent on servers/CI images.
- `keyframes.min_blur` is imagery- and resolution-dependent. The metric is computed on the
  grayscale frame resized to 640 px width (`analysis_width_px` is recorded per frame); re-calibrate
  the threshold for real imagery or use `keyframes.min_blur_mode: adaptive`.
