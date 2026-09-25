# Test Videos for Validating 3D Reconstruction

How to check whether `uav3d reconstruct` is producing correct 3D models, with
videos that are *known* to be reconstructable (plus why a field video may fail).

## 1. Ready-made test videos in this repo (`data/raw/test_videos/`)

### A. `city_flyover.mp4` — synthetic city district with full 3D ground truth

Rendered by `scripts/make_city_flyover_video.py` (deterministic, seed 20260922).
A 600-frame, 1280x720 @ 30 fps single-pass flight over a textured city block:
~29 buildings (windowed facades, parapet roofs, AC units), roads with lane
markings, cars, plazas, grass. Camera: 44–56 m AGL, ~50° oblique-down, gentle
weave — the exact profile the pipeline targets. Rendered intrinsics match the
pipeline's auto SIMPLE_PINHOLE prior (f = 0.9 × width).

Ground truth shipped alongside:

| File | Use |
| --- | --- |
| `city_flyover.GroundTruthAGL.csv` | Per-frame camera centers. Pass as `--gt-file` → ATE report with true metric scale. |
| `city_flyover.gt_model.ply` | Dense colored cloud of the *true* surfaces. Open beside the pipeline's `model.ply` in CloudCompare: `Edit > Registration > Align (ICP)`, then `Tools > Distances > Cloud/Cloud dist`. |
| `city_flyover.gt_scene.json` | True building footprints + heights (LoD2 reference). |
| `city_flyover.meta.json` | Camera/scene parameters. |

Run it:

```bash
uv run uav3d reconstruct --video data/raw/test_videos/city_flyover.mp4 \
    --max-frames 100 --gt-file data/raw/test_videos/city_flyover.GroundTruthAGL.csv
```

What "working" looks like:

- `Frames Registered` ≥ 85% of selected keyframes; mean reprojection ≤ 1.5 px.
- `ate_report.json`: RMSE within a few meters over the 200 m pass
  (`fit_condition` may flag `weak_flightline_geometry` — expected for any
  straight single-pass flight, on real footage too).
- `model.ply` in CloudCompare: flat ground plane, vertical facades, rectangular
  building clusters — not a scattered "fuzzball". Building heights should be
  visually proportional; exact scale is arbitrary unless ATE alignment is used.

Measured reference (`run_20260922_074818`, `--max-frames 100`, CPU-only):
50/50 keyframes registered (100%), mean reprojection 0.480 px, 10,903 points,
ATE RMSE 2.51 m / median 2.03 m / p95 4.09 m (scale 16.51, fit_condition ok),
328 s wall time. If your run is dramatically worse, suspect the video, not the
code.

Regenerate or vary: `uv run python scripts/make_city_flyover_video.py --out ... --frames 600 --width 1280 --height 720 --fps 30`

### B. `zurich_real_50f.mp4` — known-good baseline in Zurich dataset layout

Wraps the 50 images in `data/raw/zurich_mav/MAV Images/` into an MP4. **Note:**
those local images are the repo's *generated* Zurich-layout sample (from
`scripts/generate_sample_zurich.py` — consistent 3D scene, true intrinsics, GPS
and GroundTruthAGL logs), not real GoPro footage. Its value: this exact data is
the validated baseline (100% registration, 0.59 px reprojection, ATE RMSE
1.26 m), and `--gt-file` / calibration all line up automatically. It separates
"pipeline broken" from "my video is bad": if this one reconstructs and your
field video doesn't, the problem is the capture, not the code.

```bash
uv run uav3d reconstruct --video data/raw/test_videos/zurich_real_50f.mp4 --max-frames 50
# (GroundTruthAGL.csv under data/raw/zurich_mav is auto-discovered for ATE)
```

For *real* footage, download the full Zurich Urban MAV dataset (link in §2)
into `data/raw/zurich_mav/` and regenerate with `scripts/make_synthetic_video.py`.

## 2. Public datasets worth downloading

| Dataset | Why | Link |
| --- | --- | --- |
| Zurich Urban MAV (ETH RPG) | 81k real GoPro frames + GPS + ground-truth AGL + calibration. Already supported natively (`source: zurich`). | https://rpg.ifi.uzh.ch/zurichmavdataset.html |
| TartanAir 2 (CMU) | Photo-realistic synthetic UAV video with exact depth + pose GT. Best for validating the SP-2 dense/depth stage. | https://tartanair.org/ |
| UrbanScene3D (VCC) | Synthetic urban scenes with drone trajectories **and ground-truth 3D city models** — ideal for LoD2 comparison. | https://vcc.tech/UrbanScene3D/ |
| MatrixCity | Large synthetic city, aerial trajectories, GT models. | https://city-super.github.io/matrixcity/ |
| Pix4D / DroneDeploy sample projects | Real photogrammetry image sets with reference outputs (image sets, not video). | https://support.pix4d.com/hc/en-us/sections/360000237026-Example-projects |

## 3. If you record your own single-pass video (capture checklist)

Most failed field videos break SfM for capture reasons, not code reasons:

1. **Overlap**: fly slow enough that consecutive frames overlap ≥ 85%. At 30 fps
   and 50 m AGL that's roughly ≤ 8 m/s. The pipeline budgets keyframes
   (`--max-frames`), so capture more frames than you need.
2. **Motion blur**: the #1 killer. Keyframe gate rejects frames with Laplacian
   variance < 60. Fly in good light, avoid fast yaw, keep shutter fast
   (avoid evening/indoor flights).
3. **Pure rotation / hover-in-place**: SfM needs *translational* parallax.
   A panning shot from a fixed point triangulates nothing. Keep moving forward.
4. **Textureless scenes**: water, uniform rooftops, bare fields, sky-heavy
   frames give no SIFT features. Include roads, facades, vegetation edges.
5. **Rolling shutter / gimbal wobble**: smooth constant-speed pass; avoid
   aggressive acceleration at the start of the clip.
6. **Resolution**: 1080p+ preferred; the pipeline downsamples to
   `sfm.max_image_size` (1600) internally.
7. **Length**: 15–60 s is the sweet spot for CPU COLMAP. Long flights: pass a
   smaller `--max-frames` (e.g. 80–120) rather than the whole video.
8. **Watch the diagnostics**: `debug/contact_sheet.jpg` in the run dir shows
   which frames passed quality gates; `colmap_native.log` shows registration;
   a `< 85%` registration warning means capture more overlap/texture.
