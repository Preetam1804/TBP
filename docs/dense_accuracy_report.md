# Why the Zurich reconstruction was off by metres, and the CPU pipeline that fixes it

*All results in this document are from a **replayed public dataset** run (Zurich Urban MAV,
IJRR 2017) and from an analytic synthetic benchmark used only for validation. Nothing here is
a claim of accuracy on survey-grade data.*

---

## 1. Executive summary

The previous reconstruction was not inaccurate — it was **not a dense reconstruction at all**,
and the one accuracy number attached to it (0.664 m) was measured after a calibration step that
turned out to be a regression (1.353 m).

Two structural facts decide the outcome:

1. **The point cloud was the sparse SfM model.** `points3D` from COLMAP holds a few thousand
   triangulated interest points (1 744 points for `run_20261111_010644`). They sit on strong
   corners, not on surfaces. No post-processing of a 1 744-point cloud can produce a metric
   surface: at 0.5 m spacing over a 200 m street that is 0.02 % coverage.
2. **Scale was never observed.** Monocular SfM has seven degrees of freedom; the similarity is
   unobservable. Every metre quoted from a monocular SfM model is `k × (SfM unit)` with an
   unknown, constant `k`. Re-scaling the cloud's *slope* (the `global_anchored` experiment) does
   not recover `k`, it just changes which arbitrary number is wrong.

The fix is a real multi-view-stereo dense stage, built for **CPU only**, that turns the posed
images into per-pixel depth, gates every pixel by its own predicted uncertainty and by
cross-view agreement, and fuses the survivors into a point cloud and mesh.

Measured on the analytic benchmark (40 views, 480×270, deliberately **harder** geometry than
the real flight) with simulated SfM noise (2 cm, 0.05°):

| metric | value |
|---|---|
| depth accuracy, median | **0.315 m** |
| depth accuracy, p90 | **0.526 m** |
| precision @ 0.5 m | 87.2 % |
| precision @ 1.0 m | 100.0 % |
| fused points | 196 272 (from 40 views) |
| mesh | 689 021 triangles |
| runtime, 2 CPU cores | 81 s |

The previous run's cloud held **1 744 sparse points**; this one holds **196 272** from 40 views,
and its median error against exact geometry is 0.315 m — versus **5.8 m** for the same stage
with the uncertainty and photometric gates disabled. Those two numbers bracket the whole
argument: the gates are what turn a plausible-looking cloud into a metric one.

---

## 2. The physics that bounds any stereo solution

Everything downstream is decided by one equation. For a stereo pair with focal length `f`,
baseline `b`, and disparity noise `σ_d` (≈0.3 px for sub-pixel ZNCC interpolation):

```
σ_z = z² · σ_d / (f · b)
```

For `run_20261111_010644` (f = 1 060 px, b ≈ 6.8 m, median depth ≈ 52 m, b/z = 0.13):

```
σ_z(52 m) = 52² × 0.3 / (1060 × 6.8) = 0.11 m          per view
σ_z(120 m) = 120² × 0.3 / (1060 × 6.8) = 0.60 m        per view
σ_z(250 m) = 250² × 0.3 / (1060 × 6.8) = 2.60 m        per view
```

Three consequences that shape the design:

* **Fusing N views helps by √N, not N.** Five well-chosen sources give ≈0.05 m from the noise
  term. Anything below ~0.2 m end-to-end has to beat pose error, rolling shutter and lighting,
  not just disparity noise. Claims of "centimetre" accuracy from a 6.8 m baseline at 50 m are
  arithmetic impossibilities.
* **Error grows as z².** A gate on `σ_z/z` is therefore a hard range limit. Pixels past it do
  not carry usable depth and including them is what produces the "close to the wall but metres
  off" clouds. The stage computes this gate per pixel and drops the excess — see §4.2.
* **Baseline is the only real lever.** You cannot fix a far wall with a smarter algorithm; you
  need a wider baseline or a longer focal length. The stage therefore *selects* the source views
  whose baseline is close to the informative range instead of using whatever is nearby
  (§3.2). Using all wide views was measured to be catastrophic (7.5–8.6 m error).

---

## 3. What was actually wrong

### 3.1 No dense stage (`points3D` ≠ surface)

`run_20261111_010644/sfm/colmap/sparse/0/points3D.bin` is COLMAP's sparse triangulation. It
exists to bootstrap bundle adjustment. Its density is set by feature detector thresholds, and
its 3D points are correct but isolated. The report that quoted 1 744 points as "the model" was
describing a *scaffold*.

### 3.2 No verification of scale

`configs/zurich_baseline.yaml` leaves `scale_source: none`. The report claimed metric metres by
multiplying with `1.0`. The subsequent `global_anchored` calibration — which regressed 0.664 m
to 1.353 m — tried to infer the scale from the trajectory's altitude against
`GroundTruthAGL.csv`. That is (a) ground truth used as an input, which violates the project's
own rule that GT is evaluation-only, and (b) not identifiable from a single scalar.

The dense stage keeps this honest: every report carries `scale.status`, and it reads
`unverified_monocular_scale` unless you supply GNSS or a fixed factor. **The pipeline never
silently invents metres.**

### 3.3 Three defects found *inside* the new stage during validation

These are worth recording because each one produced plausible-looking wrong output:

| defect | symptom | fix |
|---|---|---|
| `(near + far) / 2` used as "representative depth" in view selection | in a street canyon the midpoint was 104 m against a true median of 22 m, so sources were picked at a 0.09 b/z and the error model assumed a 104 m depth | propagate the **pixel-weighted median** depth from the sparse points into view ranking and the noise model |
| unweighted percentiles over sparse points | distant walls dominate the count (they span more surface), median depth 121 m vs the true pixel-weighted 37 m | weight each sparse point by `1/z²`, i.e. its image footprint |
| nearest-point (Voronoi) signed-distance field for meshing | level set discontinuous → 14.5 M triangles for 196 k points, an 888 MB OBJ, and surfaces triangulated in empty space | splat oriented points with a tent kernel (Curless–Levoy), then classify unobserved voxels by flood fill from the tile boundary; 3.5 triangles per point, 32 MB |

---

## 4. The solution

`src/uav3d/dense/` — eleven modules, CPU only, no CUDA path anywhere. Entry point
`uav3d dense` or `run_dense(run_dir, config)`.

### 4.1 Working resolution

Images are undistorted once into a shared pinhole camera at `max_width` (default 1 600) and
cached as PNG. One camera model for the whole flight removes the per-frame pose/intrinsic
inconsistency that otherwise leaks into the depth.

### 4.2 Plane-sweep stereo with uncertainty gating

For each reference view, depth planes are spaced uniformly in **inverse depth** (uniform
disparity), with the count derived from the actual disparity span so that consecutive planes
never move more than `plane_step_px` (0.7 px) in the image.

Each plane is scored by masked ZNCC against several sources, and the per-pixel cost is fused
across sources. Then:

* **cost threshold** discards pixels whose best match is not photometrically supported;
* **uncertainty gate** — `σ_z = z²·σ_d/(f·b)` with the *actual* baseline of each view's sources —
  discards pixels whose relative uncertainty `σ_z/z` exceeds `max_relative_sigma` (default 2 %).
  This is the structural guarantee that far, unmeasurable surfaces are never turned into
  confident-looking geometry;
* **cross-view geometric consistency**: each candidate point is re-projected into the
  neighbouring depth maps and must agree within `consistency_tol` in at least
  `consistency_views` of them. This is the test that catches the classic MVS failure — a
  *photometrically good but geometrically wrong* match (repeated facade textures, balconies).

Measured contribution of each gate on the benchmark (89 406 pixels with known-correct depth):

| gate | kept | median error |
|---|---|---|
| `σ_z/z ≤ 0.01` alone | 56 % | 0.251 m |
| `cost < 0.25` alone | 6.5 % | 0.172 m |
| `cost < 0.40` alone | 24 % | 0.186 m |
| `cost < 0.35` **and** `σ_z/z ≤ 0.02` | 17 % | **0.172 m** |
| no gate | 100 % | 5.8 m |

The two gates are complementary: photometric cost alone is too strict (6 % coverage), geometric
uncertainty alone is too loose (56 % coverage but 0.25 m). Together they give the accuracy of a
strict cost gate at 2.6× the coverage.

### 4.3 Matched-baseline source selection

For each reference view, candidate sources within `source_window` frames are scored on their
baseline ratio `b/z`, the angle between optical axes, and the predicted frustum overlap. The
best `num_sources` (default 5) near `b/z ≈ 0.1` are used — deliberately *not* the widest
available, because beyond `max_baseline_ratio` (0.30) the overlap collapses and the photometric
model (Lambertian, no occlusion) breaks down.

Measured on the benchmark: selecting near sources at b/z = 0.037–0.184 gave a 0.315 m median;
the same stage forced onto 8–20-frame baselines gave 7.5 m, and onto 1–20 gave 8.6 m.

### 4.4 Weighted voxel fusion

Points are accumulated into a voxel grid whose size comes from the noise model
(`voxel ≈ σ_z/√N`, clamped to `[0.3σ, σ]` and `≤ median_depth/300`). Each voxel keeps a
weighted sum of position, colour and normal; the output position is the weighted mean. This
averages independent depth estimates, so the fused cloud is less noisy than any single depth
map — and it is what makes the spacing uniform, which the mesher needs.

### 4.5 Meshing with a self-check

Oriented points are splatted into a truncated signed-distance field, unobserved voxels are
classified inside/outside by flood fill from the tile boundary, and marching tetrahedra
extracts the surface. Faces nobody could see (normal pointing away from every camera) are
removed — without that step each facade is meshed as a thin double-sided shell and the area
doubles.

The mesh is a **visualisation surface, not the measurement**: with normals estimated from depth
gradients it still carries a fringe of thin double-sided faces, so its area measures ~1.5× the
analytic surface (16.1k m² vs 10.5k m²). It is metrically in the right place — the vertices sit
on the surfaces — but if you need a watertight, orientable surface, run Screened Poisson
reconstruction on `cloud.ply` in MeshLab or CloudCompare. The metric artifact is the point
cloud.

---

## 5. Validation methodology

Because there is no laser scan of the Zurich flight, accuracy is measured against a scene whose
geometry is known in closed form: `scripts/synthetic_street.py` ray-traces boxes (street canyon,
facades, balconies, windows) analytically, so ground-truth depth is exact rather than estimated.
`scripts/dense_benchmark.py` renders that scene, writes a complete SP-1 run directory
(`poses.json`, `keyframes.json`, images, sparse points), runs the real `run_dense`, and scores
the result with the same metric code used for real data.

The benchmark is deliberately pessimistic:

| property | benchmark | your flight |
|---|---|---|
| disparity (f·b/z) | 29 px | ≈ 138 px |
| focal length | 265 px | 1 060 px |
| image | 480 × 270 | 1 920 × 1 080 |
| views | 40 | 150–300 |

A 29 px disparity is ~4.8× harder than your data for the same relative accuracy. The benchmark
therefore understates the stage, and 0.315 m here is not what your data should produce — but it
is the number I can defend, because the ground truth is exact.

Synthetic-only caveats are handled explicitly: an anti-aliased renderer (2× supersample, edge
pixels excluded from GT depth) is required, because point-sampled procedural texture aliases
differently in every view and destroys ZNCC for reasons unrelated to geometry.

---

## 6. Runtime on CPU

Measured on 2 cores, 3 GB RAM, 40 views at 480×270, 96 planes, 5 sources:

| stage | seconds |
|---|---|
| undistort + prepare | 0.4 |
| view selection | 0.2 |
| plane sweep MVS | 52.2 |
| consistency | 0.5 |
| fusion | 1.5 |
| mesh | 44.6 |
| **total** | **99.5** |

MVS cost scales linearly in pixels and views. For your 10-minute clip (150 keyframes at
1600×900, `ref_stride=1`):

```
52.2 s × (150/40 views) × (1600×900 / 480×270 pixels) ≈ 1 300 s ≈ 22 min
```

plus fusion and meshing at the corresponding point density — inside the one-hour budget with
room to spare. `--ref-stride 2` halves it, and `--quick` sets a 960 px preset for a fast first
look.

---

## 7. How to run it

```bash
# full quality, on the latest run in data/outputs/
uav3d dense --config configs/zurich_baseline.yaml

# explicit run, 2 cores, everything at native resolution
uav3d dense --run-dir data/outputs/run_20261111_010644 \
            --max-width 1600 --num-sources 5

# fast first look
uav3d dense --run-dir data/outputs/run_20261111_010644 --quick

# score against an external reference cloud (metres)
uav3d dense --run-dir data/outputs/run_20261111_010644 --gt-ply scan.ply
```

Outputs land in `<run-dir>/dense/`:

```
cloud.ply            fused coloured point cloud (the model)
mesh.obj             surface mesh
depth/               per-view depth maps (float32)
depth_filtered/      depth maps after cost + uncertainty + consistency gating
rectified/           undistorted working images
dense_report.json    full report, schema_version 1.0.0
DENSE_REPORT.md      human-readable summary
```

The report always states `scale.status`. To obtain metres rather than SfM units, either pass
`--scale-source fixed --scale-factor K` if you know K, or `--scale-source gnss` if the flight
has GNSS telemetry. **Ground truth is never an input.**

---

## 8. What this cannot fix, and why

* **Far walls.** Beyond `z_max = f·b·max_relative_sigma/σ_d` the gate rejects everything — about
  90 m for the real flight at a 6.8 m baseline. A 250 m wall cannot be measured from a 6.8 m
  baseline; the honest answer is no geometry, not a smooth guess.
* **Scale.** A monocular flight has no absolute scale. The pipeline reports SfM units and says
  so.
* **Reflective/transparent surfaces.** Plane-sweep assumes brightness constancy. Glass facades
  and water will be rejected by the consistency test (correctly — they are unmeasured, not
  wrong).
* **Thin structures** below the working resolution: railings, wires, antennae.

---

## 9. Reproducing the numbers in this document

```bash
# analytic-truth benchmark (40 views, 480x270, simulated SfM noise)
python scripts/dense_benchmark.py --frames 40 --width 480 --height 270 \
       --pose-noise-m 0.02 --pose-noise-deg 0.05 --tag final

cat data/benchmark/final/benchmark_metrics.json
```

Unit and integration tests: `pytest tests/test_dense_*.py` (30 tests: projective geometry,
view selection, level-set meshing, end-to-end pipeline, determinism).
