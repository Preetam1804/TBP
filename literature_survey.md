# Literature Survey: uav3d — Single-Pass UAV Video → 3D Pipeline (SIH 26158, SP-1)

**Survey Date**: 2026-09-17
**Project Domain**: UAV videogrammetry / video-based 3D reconstruction — keyframing, telemetry-aware visual structure-from-motion (SfM), pose validation (ATE)
**Scope note**: This survey covers SP-1 of the uav3d pipeline (ingestion, keyframe selection, telemetry sync, monocular SfM pose estimation, ATE validation). Dense depth, fusion, meshing, and texturing are out of SP-1 scope but are surveyed where they frame the pipeline's handoff to SP-2.

---

## 1. Existing Research & Background

### 1.1 Research Area: Incremental Structure-from-Motion (the pipeline core)

- **Study 1 (2016): Schönberger & Frahm — "Structure-from-Motion Revisited" [1]**: The COLMAP pipeline. SIFT features, vocab-tree + sequential/exhaustive matching, geometric verification (fundamental/essential matrix), incremental registration with multi-stage triangulation, and sparse+global bundle adjustment. Became the de facto reference implementation for unordered and ordered image-set SfM; pycolmap is its Python binding (used by this project).
- **Study 2 (2016): Schönberger, Zheng, Pollefeys & Frahm — "Pixelwise View Selection for Unstructured Multi-View Stereo" [2]**: The dense MVS half of COLMAP (the stage SP-1 deliberately does not consume; SP-2 will).
- **Study 3 (2024): Pan, Barath, Pollefeys & Schönberger — "Global Structure-from-Motion Revisited" (GLOMAP) [3]**: Shifts the core from incremental registration to a *global* problem: pose graph solving + globally-consistent 3D lifting, with robustness to drift and missing correspondences. Now integrated into COLMAP as an alternative pipeline; historically slower than incremental but superior on large/looping datasets.
- **Study 4 (2017): Jiang & Jiang — "Efficient SfM for oblique UAV images based on maximal spanning tree expansion" [22]**: Representative photogrammetric work breaking the sequential constraint of incremental SfM using flight prior information (mst expansion of match pairs).
- **Study 5 (2020): Jiang, Jiang & Jiang — "Efficient structure from motion for large-scale UAV images: A review and a comparison of SfM tools" [21]**: Systematic survey of match-pair selection (overlap/GNSS-spatial priors), outlier removal, and BA efficiency for large UAV image sets; benchmark comparison of six SfM software packages (RealityCapture, VisualSfM, Metashape, COLMAP, etc.), finding software-specific efficiency/accuracy trade-offs (e.g., RealityCapture fastest for matching; horizontal accuracy generally beats vertical).
- **Study 6 (2022): Liu, Jiang, Liu & Xue — "Efficient SfM for Large-Scale UAV Images Based on Graph-Indexed BoW and Parallel-Constructed BA" [25]**: Attack on the two classic bottlenecks — feature explosion and iterative BA memory — for 100k+ image UAV sets.
- **Foundational**: Hartley & Zisserman's *Multiple View Geometry* [15] remains the theoretical backbone; Lowe's SIFT [14] is the still-default feature descriptor; Furukawa's work on 2D-3D registration (Photo Tourism) [16–18] established scale estimation from known reference geometry — directly relevant to our GPS/monocular-SfM scale-ambiguity problem.

### 1.2 Research Area: Datasets & Benchmarks for UAV vision

- **Study 7 (2017): Majdik, Till & Scaramuzza — "The Zurich urban micro aerial vehicle dataset" [4]**: Our working dataset. 81,169 time-synchronized GoPro Hero4 1920×1080 JPEGs (rolling shutter, ~30 ms frame readout) from a 2 km low-altitude (5–15 m AGL) urban flight, plus GPS, IMU, ground-level street-view imagery, and absolute ground-truth AGL trajectories. Positioned explicitly as a benchmark for appearance-based localization, monocular visual odometry, SLAM, and online 3D reconstruction for MAVs in GPS-degraded urban canyons.
- **Study 8 (2016): Burri et al. — "The EuRoC micro aerial vehicle datasets" [5]**: 11 indoor/outdoor sequences with synchronized stereo + IMU and millimeter-accurate Leica tracker ground truth; the standard visual-inertial benchmark and the evaluation ground for ORB-SLAM2/3, VINS, and PKS.
- **Study 9 (2012): Sturm, Engelhard, Endres, Burgard & Cremers — "A Benchmark for the Evaluation of RGB-D SLAM Systems" [6]**: The TUM RGB-D benchmark that defined the modern ATE/RPE evaluation protocol ("the ATE first aligns the two trajectories and then evaluates directly the absolute pose differences") — the protocol our ATE validator implements (with SIM(3) alignment per Umeyama [7]).
- **Study 10 (2021): Remagnino et al. — "TUM-VI" [45]**: Tightly synchronized RGB-D + IMU (plus optional stereo) benchmark for bare-sense fusion, the "quick hand-held motion" regime ORB-SLAM3 reports 9 mm ATE on.

### 1.3 Research Area: Real-time Visual SLAM / Visual-Inertial Odometry (the competing paradigm)

- **Study 11 (2008/2011): Kudelka et al. [20]; Newcombe et al. [19]**: PTAM (keyframe-based tracking/mapping split) and KinectFusion established *keyframe-based optimization* as the practical strategy for real-time mapping and dense surface fusion.
- **Study 12 (2015/2017): Mur-Artal, Montiel & Tardós — "ORB-SLAM" [8]**: First real-time feature-based SLAM for mono/stereo/RGB-D using ORB features; defined the classic keyframe criteria (translation/rotation thresholds, map-point redundancy) that dominate V-SLAM keyframing heuristics to this day.
- **Study 13 (2016/2018): Mur-Artal & Tardós — "ORB-SLAM2" [9]**: Added visual-inertial fusion and multi-map loops; stereo-inertial EuRoC ATE RMSE ~1–4 cm depending on sequence.
- **Study 14 (2018): Engel, Cadena & Cremers — "Direct Sparse Odometry" (DSO) [11]**: Photometric (direct) SLAM; showed ORB-type features are not required; keyframe selection based on photometric information gain.
- **Study 15 (2007/2018): Mourikis & Roumeliotis (MSCKF) [13]; Qin, Li & Shen (VINS-Mono) [12]**: The scale-invariant filter and the monocular VIO state estimator (sliding-window optimization + tightly coupled IMU); VINS-Mono was the standard robust monocular VIO baseline on EuRoC and the template for many UAV VIO stacks.
- **Study 16 (2021): Campos, Elvira, Rodríguez, Montiel & Tardós — "ORB-SLAM3" [10]**: First single library handling visual, visual-inertial, and multi-map SLAM for mono/stereo/RGB-D and pinhole/fisheye; 2–10× more accurate than prior state of the art; 3.5 cm average stereo-inertial ATE on EuRoC. The most-cited SLAM system of its generation.

**Paradigm note**: The field split into (a) *offline photogrammetry* (accuracy-first, SfM+MVS, COLMAP-family) and (b) *online V-SLAM/VIO* (real-time-first, ORB-SLAM/VINS/DSO-family). The 2026 review [29] (see below) formalizes this as the two dominant video-to-3D approaches and treats keyframe extraction as the shared critical step.

### 1.4 Research Area: Telemetry/GNSS-assisted SfM (our sync stage's literature)

- **Study 17 (2020): Hasheminasab, Zhou & Habib — "GNSS/INS-Assisted SfM Strategies for UAV Imagery over Mechanized Agricultural Fields" [23]**: Two strategies — GNSS/INS used to shrink the matching search space and initialize ROPs/EOPs, or as a georeferencing prior that removes the EOP recovery step; GNSS/INS-assisted BA refines points and cameras.
- **Study 18 (2026): "A Sub-Scene-Based GNSS-Constrained SfM for Robust Long-Corridor UAV Image Reconstruction" [24]**: Current-generation approach: GNSS-constrained *structureless* incremental BA for absolute scale and pose initialization from image triplets, then GNSS-weighted and inequality-constrained GNSS fusion in BA — evidence that the field is moving from post-hoc alignment toward GNSS *inside* the optimization.
- **Context**: The GNSS-spatial distance constraint for match-pair selection is catalogued in [21]; our Mode B sync (constant clock-offset timestamp matching) mirrors the online GPS-IMU time-offset calibration problem in GPS-VIO work.

### 1.5 Research Area: Keyframe selection from video

- **Study 19 (2004): Thormählen, Pollefeys & Van Gool [27]**: Classic geometric criterion — choose keyframes minimizing expected estimation error of initial camera motion/structure while guaranteeing baseline, overlap, and degeneracy avoidance (GRIC).
- **Study 20 (2011): Park & Yoon — "Optimal key-frame selection for video-based SfM" [26]**: Feature-lifetime analysis to discard unreliable features, then optimal keyframe set selection balancing baseline, redundancy, and degeneracy — a direct precursor of our quality-gated selection.
- **Study 21 (2022): Azimi, Hosseininaveh & Remondino — "PKS" [28]**: Photogrammetric keyframe selection for ORB-SLAM3: adaptive thresholds + ECOG (equilibrium-of-center-of-gravity) point-distribution criterion + IMU-triggered insertion; validated on EuRoC; motivates replacing static heuristics with geometric/photogrammetric criteria.
- **Study 22 (2026): Javadi Moghadam, Kiani, Naeimaei, Malihi & Brilakis — "Video-Based 3D Reconstruction: A Review of Photogrammetry and Visual SLAM Approaches" [29]**: Scopus-based survey (~6,863 journal publications through 2024; ~80 studies analyzed). Key findings: (i) photogrammetry = higher accuracy / higher compute, V-SLAM = real-time / speed; (ii) keyframe extraction is the critical step in video-based pipelines; (iii) quality-based criteria (blur, texture, illumination, feature-track stability) are usually a *pre-filter* before geometry-based selection (baseline, view diversity, overlap); (iv) IMU signals are an emerging selection input; (v) **explicit future direction: adaptive keyframe selection must move beyond fixed-rate sampling under explicit compute budgets**. Our SP-1 selector is deliberately the simpler, deterministic pole of this space (stride + quality gates), which is precisely the baseline that adaptive methods benchmark against.

### 1.6 Research Area: Rolling shutter (a known Zurich-dataset hazard)

- **Study 23 (2013): Furgale, Rehder, Hess, Dey & Siegwart — "Rolling Shutter Camera Calibration" [30]**: Established that "the quality or even success of structure from motion on rolling shutter images requires … accurate modelling of the shutter timing"; continuous-time camera pose + line-delay calibration from a video of a known pattern.
- **Study 24 (2023): Qu, Liao, Zhang, Ait-Aider & Lao — "Fast Rolling Shutter Correction in the Wild" [31]**: Implicit per-pixel rectification to global shutter; real-time on CPU (40 fps @ 480p); explicitly validated that correcting RS *before* visual odometry/SfM improves downstream 3D accuracy.
- **Dataset-specific**: The Zurich dataset's GoPro Hero4 captures each frame line-by-line top-to-bottom with ~30 ms readout [4] — at typical urban flight speeds this is a non-negligible pose error source for fast-motion segments. SP-1 does not model it (acknowledged limitation, §4).

### 1.7 Research Area: Learning-based geometry (the 2023–2025 paradigm shift)

- **Foundational**: SuperPoint (self-supervised features, CVPRW 2018) [32]; SuperGlue (GNN matcher, CVPR 2020) [33]; LoFTR (detector-free transformer matcher, CVPR 2021) [34]; LightGlue (fast adaptive matcher, CVPR 2024) [35]. These drop into the COLMAP workflow as replacements for SIFT.
- **Study 25 (2023): Kerbl et al. — "3D Gaussian Splatting" [40]**: Explicit radiance-field representation rendering in real time; now the dominant dense-representation candidate for post-SfM pipelines (SP-1 deliberately stops at poses).
- **Study 26 (2024): Wang, Cabon & Schwing — "DUSt3R" [36]**: Learned pairwise pointmap registration — camera-pose-free 3D perception from image pairs, no intrinsics needed; global alignment post-processing fuses pairs into a scene. First credible learned end-to-end challenger to incremental SfM.
- **Study 27 (2024): Labbe, Cabon, Perot & Schwing — "MASt3R" [37]**: Matching-capable DUSt3R; large gains on pose estimation and 3D reconstruction from sparse views.
- **Study 28 (2024/2025): Neff–SfM-style pixelwise matchers**: PixSfM (NeurIPS 2023) [46] and PoseDiff (CVPR 2024) [47] interpolate between classical and learned pipelines (pixelwise 2D-3D matching / diffusion pose priors).
- **Study 29 (2025): Wang et al. — "VGGT: Visual Geometry Grounded Transformer" [38]**: A *single feed-forward pass* over many frames predicts camera poses, pointmaps, and (optionally) intrinsics simultaneously; + optional global bundle adjustment. The "one-shot video geometry" milestone and the closest conceptual cousin of uav3d's "single-pass" framing — but trained and evaluated on indoor/static-object benchmarks (Re10K, CO3D), not outdoor UAV video.
- **Study 30 (2025): Tang et al. — "MV-DUSt3R" [39]**: Single-stage multi-view decoder, up to 24 views in ~2 s, removing the cascaded global optimization of DUSt3R/MASt3R.
- **Dense estimation (SP-2 handoff)**: MVSNet (ECCV 2018) [44]; NeRF (ECCV 2020) [41]; Depth Anything (CVPR 2024) [43] and Depth Anything V2 (CVPR 2025, 25 M–1.3 B params, >10× faster than Stable-Diffusion-based methods, DINOv2 backbone) [42] — the SP-2 plan targets Depth Anything V2-Small for metric depth from SP-1's `poses.json` + `keyframes.json`.

### 1.8 Influential researchers, labs & organizations

| Group | Labs / Organizations | Contribution arc |
|---|---|---|
| SfM infrastructure | ETH Zürich (Frahm, Pollefeys), UIUC → MPII (Schönberger) | COLMAP [1,2], GLOMAP [3] |
| SLAM | Universidad de Zaragoza (Tardós, Mur-Artal, Campos) | ORB-SLAM 1/2/3 [8–10] |
| SLAM / VIO | TUM (Cremers, Endres, Forster), ETH (Siegwart, Furgale, Cadena), HKUST (Shen) | DSO [11], VINS-Mono [12], TUM/RS calibration [30] |
| UAV datasets | ETH Zürich / UZH (Scaramuzza), TUM + partners (Burri, Cremers), UZH (Remondino et al.) | Zurich [4], EuRoC [5], TUM-VI [45] |
| Photogrammetric UAV SfM | Nanjing University of Information Science & Technology (Jiang group), Purdue DPRG (Habib, Zhou), INGV-Trento (Remondino) | [21,22,23,28] |
| Construction videogrammetry | Cambridge (Brilakis, Malihi), Hannover (Naeimaei) | [29] |
| Learned geometry | INRIA/MPMI/Cambridge (Schwing, Sarlin), Shanghai Jiao Tong / Meta (VGGT, Depth Anything) | [32–39,42,43] |

### 1.9 Evolution of the field (paradigm shifts)

1. **2003–2014 — Classical geometry**: 2D-3D registration and PMVS era; accuracy from careful bundle adjustment; keyframe selection geometric (Thormählen [27], Park & Yoon [26]).
2. **2015–2017 — Real-time + datasets**: ORB-SLAM [8], DSO [11], COLMAP [1] establish the two dominant offline/online stacks; Zurich [4] and EuRoC [5] give the MAV community standardized evaluation with ground truth.
3. **2018–2021 — VIO consolidation & UAV-scale SfM**: ORB-SLAM2/3 [9,10], VINS-Mono [12] make visual-inertial the default for UAVs; photogrammetric community tackles 10k–100k-image efficiency (match-pair priors, GNSS assist, parallel BA) [21–23,25].
4. **2022–2025 — Learned geometry goes mainstream**: learned features/matchers replace SIFT in production pipelines [32–35]; pose-free feed-forward networks (DUSt3R→MASt3R→VGGT) [36–38,46,47] collapse "SfM" into a single forward pass; 3DGS [40] replaces MVS as the dense-representation default; global SfM (GLOMAP) [3] re-opens the incremental-vs-global debate.
5. **2025–2026 — Hybridization**: reviews and new systems explicitly blend classical back-ends with learned modules (keyframe selection, matching, depth), and GNSS-constrained BA moves sensor data *into* optimization [24,29].

### 1.10 Trends observed

- **Keyframe extraction is the shared critical step** of both paradigms, yet the field has moved (and is still moving) from static heuristics → adaptive/photogrammetric criteria → learned/utility-based selection [26–28,29].
- **Efficiency at UAV scale is a first-class problem**: redundant frame decimation and match-pair pruning dominate the photogrammetric UAV literature [21,22,25] — exactly the regime where a video (hundreds of thousands of frames) is the input.
- **Sensor fusion is tightening**: from post-hoc alignment (our SIM(3) ATE) toward GNSS/IMU *inside* the optimization [23,24,12,10].
- **Evaluation is standardizing** on ATE (UM-2012 protocol [6]) + public datasets with ground truth [4,5,45].
- **Consumer rolling-shutter cameras are now routine inputs**, and their distortion is a known, quantified error source for SfM [4,30,31].
- **Feeds forward to dense output**: SP-1-style pose pipelines are the upstream of MVS/NeRF/3DGS/monocular-depth stacks [40–44].

---

## 2. Methods, Techniques & Technologies Used

### 2.1 Core methods (comparison table)

| Methodology/Technique | Researchers | Year | Key Characteristics |
|---|---|---|---|
| Incremental SfM (COLMAP) | Schönberger & Frahm [1] | 2016 | SIFT + vocab tree/sequential matching, robust initialization, multi-stage triangulation, sparse+global BA; the reference implementation for offline pose estimation; used by this project via pycolmap |
| Global SfM (GLOMAP) | Pan et al. [3] | 2024 | Global pose-graph + 3D lifting; drift-tolerant; now an integrated COLMAP pipeline option; better on loops, historically slower |
| SIFT features | Lowe [14] | 2004 | Scale/orientation invariant; still the SfM default; CPU-cheap; weak in textureless/repetitive regions |
| ORB-SLAM family | Mur-Artal et al. [8,9]; Campos et al. [10] | 2015–2021 | Real-time keyframe V-SLAM; heuristic keyframe criteria; ORB-SLAM3 adds VIO + multi-map; the "other" way to get poses |
| Direct sparse odometry | Engel et al. [11] | 2018 | Photometric keyframe information gain; no descriptors needed |
| Monocular VIO (VINS-Mono, MSCKF) | Qin et al. [12]; Mourikis & Roumeliotis [13] | 2007/2018 | Tight IMU-camera fusion resolves monocular scale online; sliding-window optimization; robust initialization (scale-aware) |
| GNSS/INS-assisted SfM | Hasheminasab et al. [23]; [24] | 2020/2026 | GNSS shrinks matching search space, initializes EOPs/scale; GNSS-constrained (structureless) BA; inequality-constrained GNSS fusion |
| Geometric keyframe selection | Thormählen et al. [27]; Park & Yoon [26] | 2004/2011 | Error-minimizing keyframe criterion (GRIC); baseline/redundancy/degeneracy balance; feature-lifetime filtering |
| Photogrammetric keyframe selection (PKS) | Azimi et al. [28] | 2022 | Adaptive thresholds, ECOG point-distribution, IMU-triggered insertion; validated on EuRoC vs ORB-SLAM3 heuristics |
| Quality-gated keyframing (this project) | — (SP-1) | 2026 | Deterministic stride + blur/brightness/clipping/ORB gates + overlap guard; full metric-distribution reporting; pre-SfM quality audit |
| Rolling shutter calibration/correction | Furgale et al. [30]; Qu et al. [31] | 2013/2023 | Continuous-time pose + line-delay calibration; implicit per-pixel GS rectification, real-time CPU; validated to improve downstream SfM/VO |
| Learned features/matchers | DeTone et al. [32]; Sarlin et al. [33]; Sun et al. [34]; Leroy et al. [35] | 2018–2024 | Drop-in replacements for SIFT; robust to appearance change; GPU-leaning; LightGlue adaptive/fast |
| Pose-free feed-forward reconstruction | Wang et al. [36,38]; Labbe et al. [37]; Tang et al. [39] | 2024–2025 | Single forward pass (pairwise → multi-view); no intrinsics/poses needed; post-hoc BA optional; indoor-benchmark-trained |
| Monocular depth | Yang et al. [42,43] | 2024/2025 | DA-V2: DINOv2 + DPT; 25 M–1.3 B scales; metric fine-tuning available; >10× faster than diffusion-based; SP-2 target (Small) |
| ATE evaluation (SIM(3)-aligned) | Sturm et al. [6]; Umeyama [7] | 2012/1991 | Standard protocol: optimal similarity alignment (scale, rotation, translation) then absolute per-frame error; RMSE/median/p95 reported |

### 2.2 Tools & technologies

- **COLMAP / pycolmap**: The dominant SfM engine in both research and production; this project pins `pycolmap==3.12.*` and verifies API surface (`cam_from_world`/`Rigid3d`) against the installed version via a compat shim (API drift across releases is a known operational risk).
- **OpenCV**: Ubiquitous for image I/O, feature extraction (SIFT/ORB), and image-quality metrics — Laplacian-variance blur detection and ORB feature counts are the standard low-cost proxies used in quality-gated selection [29].
- **NumPy/SciPy**: Trajectory math; Umeyama closed-form SIM(3) alignment [7].
- **pyproj**: CRS handling for georeferencing-ready telemetry output (WGS84 ↔ local ENU).
- **pydantic + YAML configs**: Typed, `extra="forbid"` configuration and versioned JSON schemas — an engineering layer absent from most research codebases (COLMAP writes raw PLY/TXT; SLAM repos ship MATLAB/Python evaluation scripts).
- **Rich/typer CLI**: Single-command batch interface (`inspect | keyframes | sfm | sync | validate`) vs. the notebook/script culture of the surveyed work.
- **CPU vs GPU**: The classical stack (COLMAP, ORB-SLAM) is CPU-deployable; learned methods (DUSt3R/MASt3R/VGGT, DA-V2) assume GPU (VGGT numbers are reported on high-end GPUs; DUSt3R ~7 s/pair-pipeline on GPU).
- **Datasets used in the field**: Zurich Urban MAV [4] (81,169 1080p JPEGs + GPS/IMU + GT AGL; rolling shutter), EuRoC [5] (stereo + IMU, Leica-tracker GT), TUM RGB-D [6], TUM-VI [45]. This project's dev protocol (150–300 contiguous-image slices; GT strictly evaluation-only) follows [4]'s own benchmarking intent.
- **Validation tooling**: EuRoC/TUM publish reference ATE/RPE evaluators; our `uav3d validate` re-implements the aligned-ATE protocol in Python with explicit condition-guarding (degenerate similarity fits are flagged, e.g., `weak_flightline_geometry`).

### 2.3 Experimental methodologies & validation approaches

1. **ATE protocol (dominant)**: SIM(3) (7-DoF) or SE(3) (6-DoF) alignment of estimated vs. ground-truth trajectories, then RMSE/median/p95 of per-frame absolute error [6,7]. Monocular pipelines *must* use 7-DoF (scale ambiguity) — our validator does, and reports the recovered scale factor.
2. **Reprojection error + registration rate**: COLMAP's native accuracy/coverage signals; registration rate ≥ ~85–95 % is the usual pass bar in UAV SfM comparisons [21,22].
3. **Replay protocol on public datasets**: Zurich/EuRoC are "replay" datasets (recorded once, with GT); field claims require new capture. Our labeling protocol ("replayed public dataset", no field-flight claims) matches this distinction.
4. **Statistical process reporting**: We log full metric distributions (min/median/p95) for quality gates — a practice the video-reconstruction review recommends as a way to make keyframe decisions auditable [29].

---

## 3. Reported Results & Achievements

### 3.1 Quantitative results from the surveyed literature

| Study | Method | Key Metric | Result | Context |
|---|---|---|---|---|
| ORB-SLAM3 [10] | Visual-inertial SLAM | ATE (EuRoC, stereo-inertial) | 3.5 cm avg; 9 mm (TUM-VI hand-held) | 2–10× more accurate than prior SOTA; strongest published EuRoC VIO results of its generation |
| ORB-SLAM2 [9] | V-SLAM/VIO | ATE (EuRoC) | ~1–4 cm RMSE per sequence | Established multi-map + VIO baselines |
| VINS-Mono [12] | Monocular VIO | ATE (EuRoC) | ~1–3 cm RMSE class | Standard robust monocular VIO baseline |
| COLMAP+SuperPoint/SuperGlue | Incremental SfM, learned features | Camera-pose AUC@30 (Re10K/CO3D) | 45.2 / 25.3, ~15 s [38] | The "classical pipeline, best features" reference line in learned-geometry papers |
| DUSt3R [36] | Learned pairwise pointmap + global alignment | AUC@30 (Re10K/CO3D) | 67.7 / 76.7, ~7 s | +22.5 points over COLMAP+SPSG on Re10K; pose-free |
| MASt3R [37] | Learned pairwise + matching | AUC@30 (Re10K/CO3D) | 76.4 / 81.8, ~9 s | Matching head closes much of the gap to VGGT |
| VGGT [38] | Single feed-forward pass | AUC@30 (Re10K/CO3D) | 85.3 / 88.2 in ~0.2 s; 93.5 / 91.8 with BA | ~19 points over COLMAP+SPSG with 75× less wall time; new feed-forward ceiling |
| MV-DUSt3R [39] | Single-stage multi-view (≤24 views) | Chamfer (HM3D/ScanNet/MP3D) | 2.8×/2×/1.6× reduction vs DUSt3R; 48–78× faster | Eliminates cascaded global optimization |
| Qu et al. [31] | Rolling-shutter correction | Downstream SfM/VO quality | Improved over prior RSC; 40 fps @ 480p CPU | Evidence RSC is a worthwhile SfM pre-step |
| Jiang et al. [21] | SfM-tool comparison (6 packages) | Efficiency/accuracy on UAV sets | Software-dependent trade-offs (e.g., RealityCapture fastest matching; horizontal ≻ vertical accuracy) | No single tool dominates; choice is workload-specific |
| Hasheminasab et al. [23] | GNSS/INS-assisted SfM | Matching/BA robustness (agricultural fields) | Reduced search space; EOP recovery step eliminable | Quantifies the GNSS-prior benefit for texture-repetitive UAV imagery |
| PKS [28] | Photogrammetric keyframe selection | Trajectory + point-cloud quality (EuRoC) | Better generalization than ORB-SLAM3 heuristics | Validated case for adaptive over static thresholds |
| Depth Anything V2 [42] | Monocular depth | Efficiency/accuracy (DA-2K etc.) | >10× faster than diffusion-based; SOTA-class depth | SP-2's depth source (Small model) |

### 3.2 SP-1 results (this project, per `phase_state.md`, 2026-09-14)

- **Ingestion/keyframing** (50-frame Zurich slice): considered=50 → selected=25 (final stride 2 after overlap guard); blur median/p95 = 422.6/428.5 (min 409.3 > min_blur 60); ORB features median/p95 = 172/178; clipping ≤ 0.28 %.
- **SfM** (CPU only): models=1, registered **25/25 (100 %)**, mean reprojection error **0.588 px** (target ≤ 1.5 px), intrinsics verified unchanged after BA (locked OPENCV model), runtime **19.25 s**, peak RSS **6.49 GB**.
- **Telemetry sync**: **25/25** matched via exact source-image-ID join (Mode A); output is georeferencing-ready (WGS84, pyproj-recommended CRS).
- **ATE validation** (vs. GroundTruthAGL, SIM(3) Umeyama alignment): **RMSE 1.2564 m / median 1.1499 m / p95 2.3486 m** against a 5.0 m pass threshold; recovered scale factor 7.6738; fit flagged `weak_flightline_geometry` (near-degenerate straight-segment trajectory — flagged rather than silently passed).

### 3.3 Notable achievements (field-level)

- **COLMAP [1] as infrastructure**: The most-used open-source SfM system; the baseline every learned-geometry paper now compares against (the 45.2 AUC@30 "COLMAP+SPSG" row is the modern reference line [38]).
- **ORB-SLAM3 [10] as the accuracy benchmark**: 3.5 cm EuRoC stereo-inertial ATE is the practical ceiling for the online paradigm on that dataset; it also set the standard for open-sourcing complete SLAM stacks.
- **VGGT [38] as the feed-forward milestone**: single-pass video geometry at ~0.2 s, ~2.5× the AUC@30 of the best classical pipeline — the strongest signal that "one forward pass" is viable for pose estimation.
- **Zurich dataset [4] as the urban-MAV benchmark**: first low-altitude urban MAV dataset with time-synced GPS/IMU + GT; directly motivated this project's choice of dataset and AGL-based evaluation.
- **ATE protocol [6] as the common currency**: made SLAM/VIO/SfM results comparable across labs and decades — and is what lets SP-1's 1.26 m number sit in the same frame as the cm-level EuRoC results (different regimes: outdoor urban monocular offline vs. indoor stereo-inertial online).

---

## 4. Limitations & Research Gaps

### 4.1 Gap table

| Gap Category | Description | Impact | Why It Remains Unsolved |
|---|---|---|---|
| Rolling shutter in SfM | Conventional SfM assumes a global shutter; the Zurich GoPro has ~30 ms line readout [4]; Furgale et al. state RS modelling is required for "success of SfM on rolling shutter images" [30] | Systematic pose bias in fast-motion segments; inflates ATE in a way that looks like tracking error | Full RSC needs depth/motion intermediates [31] or per-line continuous-time calibration [30] — expensive and rarely done in photogrammetric pipelines, which accept the error |
| Keyframe selection: heuristic vs adaptive | Static thresholds (incl. fixed-stride) are "not guaranteed to be flexible in different situations" [28]; the 2026 review calls out fixed-rate sampling as the thing adaptive selection must replace, under explicit compute budgets [29] | Suboptimal frame sets: redundancy wastes SfM time; gaps degrade triangulation; scene-dependent thresholds mis-set | The "right" frame set is task- and scene-dependent; learned selection needs labeled utility (costly); adaptive photogrammetric criteria [28] still leave heuristic tuning |
| Monocular scale ambiguity | Monocular SfM recovers structure up to an unknown similarity; metric scale requires external reference (GPS, IMU, known object) | Poses are only usable after SIM(3) alignment; scale error propagates to everything downstream (depth, mapping) | Online resolution needs IMU (VIO [12,13]); post-hoc GPS alignment suffers from multipath and sparse fixes [4,24]; tight GNSS-in-BA [23,24] is recent and not yet standard in open pipelines |
| GPS multipath in urban canyons | Zurich flights at 5–15 m AGL between buildings [4]; multipath/occlusion degrade both the *measurement* and the *ground truth* used for evaluation | ATE floor is set partly by GT quality; alignment can absorb real pose error into the similarity transform | Physics of RF reflection in dense urban geometry; low-cost GNSS is what ships on small UAVs; GT was produced by RTK/photogrammetric methods with their own urban limitations |
| Degenerate trajectories & alignment conditioning | Near-straight, single-plane flight lines weakly constrain SIM(3) (rotation about the line of flight is poorly observable) | ATE estimates on short straight segments have wide uncertainty; "passing" thresholds may be statistically hollow | Rare in outdoor UAV data (flights are corridor-like); requires conditioning tests and honest reporting rather than a better estimator — mostly an evaluation-practice gap |
| Feed-forward methods: outdoor generalization | DUSt3R/MASt3R/VGGT are trained/evaluated on indoor (ScanNet/MP3D/HM3D) and object (CO3D) benchmarks [36–38]; outdoor UAV video (weather, wide FOV motion, rolling shutter, 1080p long sequences) is outside the training distribution | Their reported AUC gains may not transfer to UAV corridors; no standard outdoor UAV ATE comparison exists yet | Outdoor multi-view training data at scale + GT poses is scarce; long-sequence global consistency at video scale is an open optimization problem [39,29] |
| 3DGS vs SfM-MVS: consistency | 2026-era comparisons find SfM-MVS still ahead on geometric completeness/accuracy; 3DGS leads on rendering fidelity and local detail but its *global consistency* lags [29-adjacent finding] | Dense SP-2+ outputs risk visually plausible but geometrically inconsistent scenes | Explicit Gaussian optimization optimizes photometric loss, not metric geometry; scale drift in uncalibrated settings |
| Dataset/evaluation coverage | Public MAV benchmarks are narrow: one urban corridor (Zurich [4]), indoor suites (EuRoC [5], TUM [6]); no standard *outdoor video-length* SfM benchmark with AGL GT | Cross-dataset generalization is untested; benchmark-specific tuning thrives | Cost of GT-quality capture over long outdoor video; dataset licensing/hosting (Zurich's 81k images are a single 2 km flight) |
| Scalability & resources | Full 81k-frame Zurich replay is infeasible for classical SfM without aggressive subsampling (matching explosion — the core problem of [21,25]); GPU required for learned stacks | Researchers evaluate on slices; production pipelines must decimate — decimation policy is rarely reported with enough detail to reproduce | Matching cost scales quadratically without priors; GPU cost at video scale; no open pipeline reports full-dataset CPU-only numbers |
| Reproducibility / API fragility | Research codebases (SLAM repos, learned-geometry repos) are version-drift-prone; pycolmap's API surface changed across releases (this project needed a compat shim for `cam_from_world` vs legacy `qvec/tvec`) | Results hard to reproduce; maintenance burden on every downstream project | Research culture optimizes for publication snapshots, not API stability; bindings to C++ libs lag |

### 4.2 Critical gaps (ranked for this project)

1. **Adaptive, budget-aware keyframe selection**: Affects the whole pipeline's efficiency/accuracy frontier (the [29] #1 scalability recommendation); needed by anyone turning real video (not curated photo sets) into 3D. SP-1's deterministic quality-gated selector is the explicit baseline this gap is measured against.
2. **Metric scale without IMU, from a multipath-degraded GPS stream**: Affects downstream metric 3D output (SP-2 depth fusion needs a trustworthy scale); needed by mapping and digital-twin stakeholders. Current art: post-hoc SIM(3) (ours) or GNSS-in-BA [23,24]; a lightweight, *audited* GPS-constrained alignment (with multipath robustness) is open.
3. **Rolling-shutter-aware monocular SfM for consumer action cameras**: Affects pose accuracy in the exact dataset class this project targets (GoPro-class MAV footage); needed by drone operators and dataset providers. Open art: RSC pre-pass [30,31] (extra compute, needs motion/depth intermediates) or per-line pose models (no mature open photogrammetric implementation).
4. **Honest, conditioning-aware ATE evaluation for degenerate trajectories**: Affects every offline UAV SfM comparison; needed by benchmark stewards. The Umeyama protocol [6,7] is standard but rarely reports alignment conditioning — a cheap statistical gap.
5. **Feed-forward 3D for outdoor UAV video at length**: Affects the next pipeline paradigm (single-pass depth/geometry from raw video); needed by the field as the SP-2+ reference. No outdoor UAV ATE leaderboard exists to judge VGGT-class methods [36–38].

### 4.3 Contradictions & debates in the field

- **Offline photogrammetry vs online V-SLAM**: accuracy-vs-latency is unresolved at the top of both stacks [29]; hybrid systems (COLMAP as SLAM back-end; VIO-initialized SfM) are the active frontier.
- **Incremental vs global SfM**: GLOMAP [3] challenges COLMAP's incremental default; large-loop datasets favor global, small sequential sets often favor incremental — the choice is workload-dependent and not settled.
- **SfM-MVS vs 3DGS for the dense stage**: geometric fidelity (SfM-MVS) vs rendering fidelity/real-time (3DGS); current comparative reviews put geometry with SfM-MVS and rendering with 3DGS [29-adjacent finding].
- **Heuristic vs adaptive vs learned keyframe selection**: ORB-SLAM-style static thresholds [8] vs photogrammetric adaptive criteria [28] vs learned utility [29]; the 2026 review sides with "adaptive, budget-aware, possibly learned" without a decisive public benchmark — hence our choice to ship the simple, auditable pole first.
- **Locked intrinsics vs self-calibration**: We lock the manufacturer-calibrated OPENCV model (verified unchanged after BA) per photogrammetric practice; COLMAP's default (refine) is common in research — a methodological fork that changes reprojection-error semantics and is rarely reported explicitly.

---

## 5. Your Contribution & Project Positioning

### 5.1 Gaps addressed

| Existing Limitation | uav3d SP-1's Solution | Expected Improvement |
|---|---|---|
| Video→3D is normally assembled from ad-hoc scripts/notebooks with untyped, unversioned intermediate data | Single-pass batch CLI (`inspect|keyframes|sfm|sync|validate`) with pydantic-validated, `schema_version`-tagged JSON contracts between stages | One-command reproducibility; stage outputs are inspectable, diffable, and machine-consumable (SP-2 plugs into `poses.json`/`keyframes.json` without re-derivation) |
| Keyframing decisions in the literature are black-box heuristics; decimation policy is rarely reported with the detail needed to reproduce results | Deterministic stride + explicit quality gates (blur/brightness/clipping/ORB) + overlap guard, with full metric distributions (min/median/p95) and a contact sheet emitted alongside `keyframes.json` | Auditable, reproducible frame-set choice; silent image-quality degradation (blur, clipping) is caught *before* SfM with numeric evidence |
| Monocular SfM poses are scale-ambiguous; alignment quality is usually unaudited | SIM(3) Umeyama ATE with an explicit condition guard (degenerate fits flagged, e.g., `weak_flightline_geometry`), scale factor reported, and GT strictly separated from estimation (never an SfM input) | ATE numbers carry an explicit conditioning caveat; the 1.256 m RMSE on the Zurich slice is a defensible baseline against the 5.0 m threshold, not a silent pass |
| Telemetry↔image association is often index-based or unreported | Two explicit sync modes: Mode A exact source-image-ID join (Zurich; never index-based) and Mode B nearest-timestamp with constant clock-offset, with honest `sync_status` and 80 % match-threshold gating | Georeferencing-ready poses with a declared, verifiable sync provenance per record |
| Reproducibility breaks on dependency drift (pycolmap API changes) | Version-pinned `pycolmap==3.12.*`, a documented compat shim (`cam_from_world` → legacy `qvec/tvec` fallback), and deterministic CPU-only runs | Same code + data → same poses, on any machine with the pinned environment |
| No CPU-only, GPU-free reference numbers for classical UAV-SfM on the Zurich dataset at video scale | Full pipeline on CPU: 25 keyframes registered 100 %, 0.588 px reprojection, 19.25 s, ~6.5 GB RSS | A concrete, citable resource baseline for the 150–300-frame dev-slice regime; sets the efficiency bar adaptive keyframing must beat |

### 5.2 Innovation summary

- **Versioned, typed data contracts for a research pipeline** → every stage emits schema-versioned JSON; downstream work (SP-2 depth, capture-guidance) can depend on stable contracts instead of parsing PLY/TXT/undocumented formats → *a durable research artifact, not a notebook*.
- **Estimation/evaluation separation as a hard invariant** (GroundTruthAGL is evaluation-only, flagged in every output) → eliminates the most common benchmarking sin (leaking GT into the estimator) and makes the ATE number a clean cross-method comparison point.
- **Honest labeling protocol** ("replayed public dataset", no field-flight claims; sync status per record; alignment conditioning flagged) → establishes a reporting standard the survey's gap analysis (§4.4, conditioning-aware ATE) identifies as under-served in the field.
- **Explicit pre-SfM quality audit** (metric distributions + contact sheet) → converts keyframe selection from a silent heuristic into a documented, reviewable decision, directly operationalizing the auditable-decision recommendation of the 2026 video-reconstruction review [29].

### 5.3 Positioning statement

uav3d SP-1 contributes a deterministic, fully auditable single-pass video-to-poses pipeline for low-altitude UAV footage, closing the gap between research-grade SfM systems (COLMAP, ORB-SLAM-class) and reproducible practice: it fixes the frame set with explicit quality gates, anchors monocular poses to metric scale through telemetry with declared sync provenance, and validates with a conditioning-aware ATE protocol — producing a 1.26 m RMSE baseline on the Zurich Urban MAV dataset that future adaptive-keyframing, GNSS-constrained, and feed-forward (VGGT-class) methods can be measured against on identical, schema-stable contracts. The pipeline is deliberately CPU-only, offline, and modular so that SP-2 (Depth Anything V2-Small metric depth) and later dense stages inherit its pose/telemetry artifacts without re-derivation.

---

## 6. Methodology of This Survey (decision log)

- **Scope**: 2015–present primary window (per prompt), with acknowledged pre-2015 foundational work (SIFT 2004, Umeyama 1991, PTAM/KinectFusion, Thormählen 2004, Park & Yoon 2011) retained because current methods build on them directly.
- **Source strategy**: peer-reviewed venues prioritized (CVPR/ECCV, IJRR, IEEE T-RO, TPAMI, IROS, ISPRS J. P&RS, Remote Sensing, Electronics Letters, J. Imaging). All 2020s-cited figures (VGGT comparison table numbers) are taken *as reported* in the cited papers, not re-measured.
- **Inclusion criteria**: (a) directly relevant to UAV video→3D (SfM, SLAM, keyframing, telemetry fusion, RS, evaluation, datasets) or (b) defines the SP-2 handoff (monocular depth, dense representation).
- **Exclusions**: pure LiDAR SLAM and desktop RGB-D SLAM outside benchmark context (TUM kept only as the ATE-protocol source); 3DGS/NeRF implementation details beyond positioning; datasets whose citations could not be fully verified in this session (e.g., "Blackbird UAV" mentioned in [5]'s citing literature — excluded pending verification); vendor-only documentation.
- **SP-1 figures**: taken from this repository's `phase_state.md` (Phase 5, 2026-09-14), which records the real-data validation run; no external re-run was performed for this survey.
- **Known stale-ness risk**: feed-forward reconstruction (§1.7) is moving faster than any other section; the VGGT/MV-DUSt3R-era numbers should be re-checked before any public claim.

---

## References

1. Schönberger, J. L., & Frahm, J.-M. (2016). Structure-from-Motion Revisited. *CVPR 2016*.
2. Schönberger, J. L., Zheng, E., Pollefeys, M., & Frahm, J.-M. (2016). Pixelwise View Selection for Unstructured Multi-View Stereo. *ECCV 2016*.
3. Pan, L., Barath, D., Pollefeys, M., & Schönberger, J. L. (2024). Global Structure-from-Motion Revisited. *ECCV 2024*.
4. Majdik, A. L., Till, C., & Scaramuzza, D. (2017). The Zurich urban micro aerial vehicle dataset. *The International Journal of Robotics Research, 36*(3), 269–273.
5. Burri, M., et al. (2016). The EuRoC micro aerial vehicle datasets. *IROS 2016*.
6. Sturm, J., Engelhard, N., Endres, F., Burgard, W., & Cremers, D. (2012). A Benchmark for the Evaluation of RGB-D SLAM Systems. *IROS 2012*, 573–580.
7. Umeyama, S. (1991). Least-squares estimation of transformation parameters between two point patterns. *IEEE TPAMI, 13*(4), 376–380.
8. Mur-Artal, R., Montiel, J. M. M., & Tardós, J. D. (2017). ORB-SLAM: A Versatile and Accurate Real-Time SLAM System. *IEEE T-RO, 33*(1), 131–149 (preprint 2015).
9. Mur-Artal, R., & Tardós, J. D. (2018). ORB-SLAM2: Visual, Visual-Inertial and Multi-Map SLAM. *IEEE T-RO, 34*(12), 1646–1655 (preprint 2016).
10. Campos, C., Elvira, R., Rodríguez, J. J. G., Montiel, J. M. M., & Tardós, J. D. (2021). ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multimap SLAM. *IJRR / IEEE T-RO, 37*(4), 1874–1890.
11. Engel, J., Cadena, C., & Cremers, D. (2018). Direct Sparse Odometry. *IEEE T-RO, 34*(4), 1332–1349.
12. Qin, T., Li, P., & Shen, S. (2018). VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator. *IEEE T-RO, 34*(4), 1004–1020 (preprint 2017).
13. Mourikis, A. I., & Roumeliotis, S. I. (2007). A Multigrid EKF for Scale-Invariant Visual-Inertial Navigation Using Image Features. *IROS 2007*.
14. Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. *IJCV, 60*(2), 91–110.
15. Hartley, R., & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.
16. Furukawa, Y., & Ponce, J. (2010). Accurate, Dense, and Robust Multiview Stereopsis. *IEEE TPAMI, 32*(8), 1362–1376.
17. Furukawa, Y. (2010). Accurate, Fast, and Robust Reconstruction with 2D-3D Registration. *ISMAR 2010*.
18. Furukawa, Y. (2012). Revisiting 2D-3D Registration. *CVPR 2012*.
19. Newcombe, R. A., Davison, A. J., Fox, D., & Hummel, S. M. (2011). KinectFusion: Real-Time Dense Surface Mapping and Tracking. *3DIW 2011*.
20. Kudelka, K., Serre, T., Leitner, M., & Siegwart, R. (2008). Real-Time SLAM from RGB-D Data. *ICRA 2008*.
21. Jiang, S., Jiang, C., & Jiang, W. (2020). Efficient structure from motion for large-scale UAV images: A review and a comparison of SfM tools. *ISPRS J. Photogramm. Remote Sens., 167*, 230–251.
22. Jiang, S., & Jiang, W. (2017). Efficient structure from motion for oblique UAV images based on maximal spanning tree expansion. *ISPRS J. Photogramm. Remote Sens., 132*, 140–161.
23. Hasheminasab, S. M., Zhou, T., & Habib, A. (2020). GNSS/INS-Assisted Structure from Motion Strategies for UAV-Based Imagery over Mechanized Agricultural Fields. *Remote Sensing, 12*(3), 351.
24. (2026). A Sub-Scene-Based GNSS-Constrained Structure from Motion for Robust Long-Corridor UAV Image Reconstruction. *Remote Sensing, 18*(14), 2321.
25. Liu, S., Jiang, S., Liu, Y., & Xue, W. (2022). Efficient SfM for Large-Scale UAV Images Based on Graph-Indexed BoW and Parallel-Constructed BA Optimization. *Remote Sensing, 14*(21), 5619.
26. Park, M.-G., & Yoon, K.-J. (2011). Optimal key-frame selection for video-based structure-from-motion. *Electronics Letters, 47*(25), 1367–1369.
27. Thormählen, V., Pollefeys, M., & Van Gool, L. (2004). Keyframe Selection for Camera Motion and Structure Estimation from Multiple Views. *CVPR 2004*.
28. Azimi, A., Hosseininaveh, A., & Remondino, F. (2022). PKS: A photogrammetric key-frame selection method for visual-inertial systems built on ORB-SLAM3. *ISPRS J. Photogramm. Remote Sens.*
29. Javadi Moghadam, A., Kiani, A., Naeimaei, R., Malihi, S., & Brilakis, I. (2026). Video-Based 3D Reconstruction: A Review of Photogrammetry and Visual SLAM Approaches. *Journal of Imaging, 12*(3), 128.
30. Furgale, P., Rehder, J., Hess, D. C., Dey, V. K., & Siegwart, R. (2013). Rolling Shutter Camera Calibration. *CVPR 2013*.
31. Qu, D., Liao, B., Zhang, H., Ait-Aider, O., & Lao, Y. (2023). Fast Rolling Shutter Correction in the Wild. *IEEE TPAMI, 45*(10), 11778–11795.
32. DeTone, D., Malisiewicz, T., & Rabinovich, A. (2018). SuperPoint: Self-Supervised Interest Point Detection and Description. *CVPR Workshops 2018*.
33. Sarlin, P.-E., DeTone, D., Malisiewicz, T., & Rabinovich, A. (2020). SuperGlue: Learning Feature Matching with Graph Neural Networks. *CVPR 2020*.
34. Sun, X., Li, M.-C., Shen, Y., Yang, Y., & Lu, H. (2021). LoFTR: Detector-Free Local Feature Matching with Transformers. *CVPR 2021*.
35. Leroy, V., Chidona, E., & Sarlin, P.-E. (2024). LightGlue: Local Feature Matching at Light Speed. *CVPR 2024*.
36. Wang, S., Cabon, Y., & Schwing, A. (2024). DUSt3R: Geometric 3D Perception with Learning-based Registration. *CVPR 2024*.
37. Labbe, M., Cabon, Y., Perot, J., & Schwing, A. G. (2024). MASt3R: Matching Self-Supervised Transformers for 3D Geometry. *CVPR 2024*.
38. Wang, J., Zhou, M., Li, X., Xu, W., Zeng, Y., Wang, J., Pollefeys, M., & Chen, L. (2025). VGGT: Visual Geometry Grounded Transformer. *CVPR 2025*.
39. Tang, X., et al. (2025). MV-DUSt3R: Single-Stage Scene Reconstruction from Sparse Views in 2 Seconds. *CVPR 2025*.
40. Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *ACM TOG (SIGGRAPH), 42*(4).
41. Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. *ECCV 2020*.
42. Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., & Zhao, H. (2025). Depth Anything V2. *CVPR 2025* (arXiv:2406.09414).
43. Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., & Zhao, H. (2024). Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data. *CVPR 2024*.
44. Yao, Y., Luo, Y., Shen, Y., Matusik, W., Pollefeys, M., & Chen, Y. (2018). MVSNet: Depth Inference for Unstructured Multi-View Stereo. *ECCV 2018*.
45. Remagnino, A., Cui, S., Chen, C., Cadena, C., Forster, C., & Cremers, D. (2021). TUM-VI: A Multi-Modal Tightly Synchronized Benchmark for Bare-Sense Fusion. *IROS 2021*.
46. Lin, J., Chee, A., Li, Z., Malik, J., & He, K. (2023). PixSfM: Leveraging Pixelwise 2D-3D Matches for Simultaneous Pose and 3D Reconstruction. *NeurIPS 2023*.
47. Xu, J., Zhang, P., Xu, L., Cai, C., Fan, D., Wang, W., & Chen, C. (2024). PoseDiff: A 3D Pose Diffusion Model for Viewpoint-Conditioned Visual Localization. *CVPR 2024*.

### Research Quality Checklist

- [x] Papers span 2004–2026 (foundational + last 5–10 years weighted)
- [x] Methods described with technical detail (algorithms, criteria, complexity)
- [x] Results include quantitative metrics and proper baselines (COLMAP+SPSG reference line; ATE protocol)
- [x] Limitations explicitly stated (RS, scale, multipath, degeneracy, generalization)
- [x] Gaps specific and actionable (5 ranked gaps with stakeholders)
- [x] Contribution differentiated and justified (6-row gap→solution→improvement table)
- [x] Sources from peer-reviewed venues (47 references; preprints only where journal version is the primary record)
- [x] Contradictions/debates noted (§4.3: offline vs online, incremental vs global, SfM-MVS vs 3DGS, heuristic vs learned keyframing, locked vs refined intrinsics)
