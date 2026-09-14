"""Deterministic keyframe selection with quality gates and overlap guard."""

import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from uav3d.config import KeyframeConfig
from uav3d.exceptions import Uav3dError
from uav3d.keyframes.quality import FrameQualityMetrics, compute_frame_quality
from uav3d.models.frames import FrameRecord
from uav3d.models.keyframes import KeyframeRecord, KeyframesFile
from uav3d.sources.base import ImageSource

logger = logging.getLogger("uav3d.keyframes.selector")


@dataclass
class MetricDistribution:
    """Distribution summary statistics for a metric."""

    min_val: float
    median_val: float
    p95_val: float


@dataclass
class SelectionSummary:
    """Summary of keyframe selection results."""

    considered_count: int
    selected_count: int
    decode_failed_count: int
    initial_stride: int
    final_stride: int
    stride_reductions: int
    blur_dist: MetricDistribution
    brightness_dist: MetricDistribution
    clipping_dist: MetricDistribution
    features_dist: MetricDistribution
    keyframes_json_path: Path
    sfm_images_dir: Path
    contact_sheet_path: Path


class KeyframeSelector:
    """Deterministic keyframe extraction and quality filtering."""

    def __init__(self, config: KeyframeConfig) -> None:
        self.config = config
        self.initial_stride = config.stride
        self.max_frames = config.max_frames
        self.min_blur = config.min_blur
        self.max_clipping = config.max_clipping
        self.max_gap = config.max_gap
        self.orb_features = config.orb_features

    def _calc_distribution(self, values: list[float]) -> MetricDistribution:
        """Calculate min, median, and 95th percentile."""
        if not values:
            return MetricDistribution(0.0, 0.0, 0.0)
        arr = np.array(values)
        return MetricDistribution(
            min_val=float(np.min(arr)),
            median_val=float(np.median(arr)),
            p95_val=float(np.percentile(arr, 95)),
        )

    def _resolve_image_path(self, frame: FrameRecord, source: ImageSource) -> Path:
        """Resolve the absolute image file path on disk."""
        p = Path(frame.file_path)
        if p.exists():
            return p.resolve()

        images_dir = getattr(source, "images_dir", None)
        if images_dir and (images_dir / p.name).exists():
            return (images_dir / p.name).resolve()

        dataset_dir = getattr(source, "dataset_dir", None)
        if dataset_dir:
            cand = Path(dataset_dir) / frame.file_path
            if cand.exists():
                return cand.resolve()
            cand2 = Path(dataset_dir) / p.name
            if cand2.exists():
                return cand2.resolve()
            matches = list(Path(dataset_dir).rglob(p.name))
            if matches:
                return matches[0].resolve()

        return p.resolve()

    def select(
        self,
        source: ImageSource,
        output_dir: str | Path,
        dataset_label: str = "replayed public dataset",
    ) -> SelectionSummary:
        """Execute deterministic keyframe selection and write outputs.

        Args:
            source: Ingestion source yielding FrameRecords.
            output_dir: Run output directory (e.g. data/outputs/run_<timestamp>).
            dataset_label: Provenance label for keyframes.json.
        """
        run_dir = Path(output_dir).resolve()
        sfm_images_dir = run_dir / "sfm" / "images"
        debug_dir = run_dir / "debug"
        sfm_images_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        # 1. Collect all frames sequentially
        frames: list[FrameRecord] = list(source.iter_frames())
        total_frames = len(frames)
        if total_frames == 0:
            raise Uav3dError("Source yielded 0 frames; cannot extract keyframes.")

        logger.info(f"Keyframe selection: evaluating {total_frames} frames from source")

        # 2. Pre-evaluate frame metrics cache
        metrics_cache: dict[str, FrameQualityMetrics | None] = {}
        decode_failed_count = 0
        all_blur_scores: list[float] = []
        all_brightness: list[float] = []
        all_clipping: list[float] = []
        all_features: list[float] = []

        # Detect if source is a video file or discrete image sequence
        first_frame_path = Path(frames[0].file_path) if frames else Path()
        is_video_source = (
            first_frame_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
            or hasattr(source, "video_path")
            or "VideoFileSource" in source.__class__.__name__
        )

        if is_video_source:
            video_file = str(getattr(source, "video_path", first_frame_path))
            cap = cv2.VideoCapture(video_file)
            try:
                frame_idx = 0
                while cap.isOpened() and frame_idx < total_frames:
                    ret, bgr = cap.read()
                    if not ret or bgr is None:
                        decode_failed_count += 1
                        frame_idx += 1
                        continue
                    metrics = compute_frame_quality(bgr, orb_features=self.orb_features)
                    fid = frames[frame_idx].frame_id
                    metrics_cache[fid] = metrics
                    if metrics is None:
                        decode_failed_count += 1
                    else:
                        all_blur_scores.append(metrics.blur_score)
                        all_brightness.append(metrics.brightness_mean)
                        all_clipping.append(metrics.clipping_ratio)
                        all_features.append(float(metrics.feature_count))
                    frame_idx += 1
            finally:
                cap.release()
        else:
            for frame in frames:
                img_path = self._resolve_image_path(frame, source)
                metrics = compute_frame_quality(img_path, orb_features=self.orb_features)
                metrics_cache[frame.frame_id] = metrics
                if metrics is None:
                    decode_failed_count += 1
                else:
                    all_blur_scores.append(metrics.blur_score)
                    all_brightness.append(metrics.brightness_mean)
                    all_clipping.append(metrics.clipping_ratio)
                    all_features.append(float(metrics.feature_count))

        # 3. Iterative selection with Overlap Guard
        current_stride = self.initial_stride
        stride_reductions = 0
        selected_records: list[KeyframeRecord] = []
        selected_indices: list[int] = []

        while True:
            candidate_records: list[KeyframeRecord] = []
            candidate_indices: list[int] = []

            for idx, frame in enumerate(frames):
                # Stride subsample
                if idx % current_stride != 0:
                    continue

                metrics = metrics_cache.get(frame.frame_id)
                if metrics is None:
                    continue

                # Quality gates
                if metrics.blur_score < self.min_blur:
                    continue
                if metrics.clipping_ratio > self.max_clipping:
                    continue

                candidate_indices.append(idx)
                candidate_records.append(
                    KeyframeRecord(
                        frame_id=frame.frame_id,
                        timestamp_us=frame.timestamp_us,
                        file_path=f"sfm/images/{frame.frame_id}.jpg",
                        score=round(metrics.blur_score, 2),
                        selected=True,
                        selection_reason="passed_quality_and_stride",
                        metrics=metrics.to_dict(),
                    )
                )

            # Check overlap guard: gap between consecutive selected frames <= max_gap
            overlap_violated = False
            if len(candidate_indices) > 1:
                for k in range(len(candidate_indices) - 1):
                    gap = candidate_indices[k + 1] - candidate_indices[k]
                    if gap > self.max_gap:
                        overlap_violated = True
                        logger.warning(
                            f"Overlap guard triggered: gap between frame index {candidate_indices[k]} "
                            f"and {candidate_indices[k + 1]} is {gap} > max_gap ({self.max_gap})"
                        )
                        break

            # If overlap guard violated and stride can be halved, reduce stride and retry
            if overlap_violated and current_stride > 1:
                new_stride = max(1, current_stride // 2)
                logger.warning(f"Reducing stride from {current_stride} to {new_stride} to preserve visual overlap.")
                current_stride = new_stride
                stride_reductions += 1
                continue

            # Overlap check passed or stride already at 1
            selected_indices = candidate_indices
            selected_records = candidate_records
            break

        # 4. Cap at max_frames
        if len(selected_records) > self.max_frames:
            logger.info(f"Capping selected keyframes from {len(selected_records)} to max_frames={self.max_frames}")
            selected_records = selected_records[: self.max_frames]
            selected_indices = selected_indices[: self.max_frames]

        logger.info(
            f"Selection completed: {len(selected_records)}/{total_frames} frames selected "
            f"(final stride={current_stride}, {decode_failed_count} decode failures)"
        )

        # 5. Export keyframes.json
        keyframes_file = KeyframesFile(
            schema_version="1.0.0",
            dataset_label=dataset_label,
            keyframes=selected_records,
        )
        json_path = run_dir / "keyframes.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(keyframes_file.model_dump_json(indent=2))

        # 6. Save selected keyframes to sfm/images/{frame_id}.jpg
        if is_video_source:
            video_file = str(getattr(source, "video_path", first_frame_path))
            cap = cv2.VideoCapture(video_file)
            try:
                for rec in selected_records:
                    match = re.search(r"(\d+)$", rec.frame_id)
                    if not match:
                        continue
                    idx = int(match.group(1))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, bgr = cap.read()
                    dest_path = sfm_images_dir / f"{rec.frame_id}.jpg"
                    if ret and bgr is not None:
                        cv2.imwrite(str(dest_path), bgr)
                    else:
                        logger.warning(f"Failed to read frame index {idx} from video {video_file}")
            finally:
                cap.release()
        else:
            for rec in selected_records:
                # Find original frame path
                source_frame = source.get_frame(rec.frame_id)
                orig_path = self._resolve_image_path(source_frame, source)

                dest_path = sfm_images_dir / f"{rec.frame_id}.jpg"
                if dest_path.exists() or dest_path.is_symlink():
                    dest_path.unlink()

                # Attempt symlink; fall back to copy on Windows if permissions not held
                try:
                    os.symlink(orig_path.resolve(), dest_path)
                except (OSError, NotImplementedError):
                    shutil.copy2(orig_path, dest_path)

        # 7. Generate contact sheet
        contact_sheet_path = debug_dir / "contact_sheet.jpg"
        self._generate_contact_sheet(selected_records, sfm_images_dir, contact_sheet_path)

        # Distributions of selected frames
        sel_blurs = [rec.metrics["blur_score"] for rec in selected_records if rec.metrics]
        sel_bright = [rec.metrics["brightness_mean"] for rec in selected_records if rec.metrics]
        sel_clip = [rec.metrics["clipping_ratio"] for rec in selected_records if rec.metrics]
        sel_feats = [rec.metrics["feature_count"] for rec in selected_records if rec.metrics]

        return SelectionSummary(
            considered_count=total_frames,
            selected_count=len(selected_records),
            decode_failed_count=decode_failed_count,
            initial_stride=self.initial_stride,
            final_stride=current_stride,
            stride_reductions=stride_reductions,
            blur_dist=self._calc_distribution(sel_blurs or all_blur_scores),
            brightness_dist=self._calc_distribution(sel_bright or all_brightness),
            clipping_dist=self._calc_distribution(sel_clip or all_clipping),
            features_dist=self._calc_distribution(sel_feats or all_features),
            keyframes_json_path=json_path,
            sfm_images_dir=sfm_images_dir,
            contact_sheet_path=contact_sheet_path,
        )

    def _generate_contact_sheet(
        self,
        selected_records: list[KeyframeRecord],
        sfm_images_dir: Path,
        output_path: Path,
    ) -> None:
        """Create a visual montage of sampled keyframes for quick verification."""
        if not selected_records:
            return

        # Sample every 20th frame, or up to 16 frames if few selected
        step = max(1, len(selected_records) // 16) if len(selected_records) < 80 else 20
        sample_records = selected_records[::step]
        if len(sample_records) > 24:
            sample_records = sample_records[:24]

        thumb_w, thumb_h = 320, 180
        cols = 4
        rows = (len(sample_records) + cols - 1) // cols

        header_h = 50
        sheet = np.zeros((header_h + rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

        # Header banner
        cv2.rectangle(sheet, (0, 0), (cols * thumb_w, header_h), (35, 35, 35), -1)
        cv2.putText(
            sheet,
            f"UAV3D KEYFRAME CONTACT SHEET - {len(selected_records)} Selected Keyframes (Sampled every {step})",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for idx, rec in enumerate(sample_records):
            r = idx // cols
            c = idx % cols
            y0 = header_h + r * thumb_h
            x0 = c * thumb_w

            img_file = sfm_images_dir / f"{rec.frame_id}.jpg"
            img = cv2.imread(str(img_file))
            if img is not None:
                thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                # Frame ID badge
                blur_val = rec.score if rec.score is not None else 0.0
                label = f"{rec.frame_id} (b:{blur_val:.1f})"
                cv2.rectangle(thumb, (0, thumb_h - 26), (thumb_w, thumb_h), (0, 0, 0), -1)
                cv2.putText(
                    thumb,
                    label,
                    (8, thumb_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                sheet[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = thumb

        cv2.imwrite(str(output_path), sheet)
        logger.info(f"Generated contact sheet at {output_path}")
