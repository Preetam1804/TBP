# Project Context — SIH 26158 SP-1
## Goal
SP-1 of a single-pass UAV video→3D pipeline: ingest, keyframe selection,
telemetry sync, visual SfM pose estimation. Local batch CLI. No server.
## Scope
IN: video + image-sequence ingestion, Zurich MAV adapter, keyframing,
telemetry mapping, pycolmap SfM, pose export, ATE validation.
OUT: depth, fusion, mesh, texturing, frontend, database, auth, cloud.
## Architecture
ImageSource interface (VideoFileSource, ZurichSequenceSource) → FrameRecord →
KeyframeSelector → TelemetrySynchronizer → SfMRunner → poses/telemetry JSON → AteValidator.
## Conventions
- Timestamps: integer microseconds, always.
- Poses: COLMAP cam_from_world, quaternion order wxyz; center = -R.T @ t.
- SfM coords are arbitrary scale/rotation until similarity alignment.
- Every JSON output has schema_version. Schemas in src/uav3d/models/.
## Dataset (Zurich Urban MAV, ETH RPG)
81,169 GoPro Hero4 1920x1080 JPEGs; logs col 1 = PX4 timestamp in µs;
OnbordGPS.csv: timestamp,image_id,raw GPS. GroundTruthAGL.csv: image_id,XYZ,...
→ EVALUATION ONLY, never SfM input. calibration_data.npz intrinsics.
No EXIF timestamps. License: unrestricted; attribute IJRR 2017 in README.
Never process all 81k images; dev runs use 150–300 contiguous images.
## Data labeling
All outputs labeled "replayed public dataset" (report §11). No field-flight claims.
## Commands
uv sync; uv run pytest; uv run uav3d inspect|keyframes|sfm|sync|validate
## Constraints
CPU-only OK. Deterministic runs. No TODO/mock/placeholder code. pycolmap API varies
by version — verify attribute names (cam_from_world vs qvec/tvec) against installed version.
