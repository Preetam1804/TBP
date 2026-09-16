"""Tests for keyframe quality metrics, deterministic selection, overlap guard, and CLI."""

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from uav3d.cli import main
from uav3d.config import KeyframeConfig
from uav3d.keyframes.quality import compute_frame_quality
from uav3d.keyframes.selector import KeyframeSelector
from uav3d.models.keyframes import KeyframesFile
from uav3d.sources.zurich import ZurichSequenceSource


class TestQualityMetrics:
    """Test individual quality metric calculations."""

    def test_sharp_vs_blurred_image(self, tmp_path: Path) -> None:
        sharp_path = tmp_path / "sharp.jpg"
        blurred_path = tmp_path / "blurred.jpg"

        # Sharp image with sharp contrast edges
        sharp_img = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(0, 640, 20):
            sharp_img[:, i : i + 10] = 255
        cv2.imwrite(str(sharp_path), sharp_img)

        # Heavily blurred image
        blurred_img = cv2.GaussianBlur(sharp_img, (51, 51), 25.0)
        cv2.imwrite(str(blurred_path), blurred_img)

        sharp_metrics = compute_frame_quality(sharp_path)
        blurred_metrics = compute_frame_quality(blurred_path)

        assert sharp_metrics is not None
        assert blurred_metrics is not None
        assert sharp_metrics.blur_score > 500.0
        assert blurred_metrics.blur_score < 50.0
        assert sharp_metrics.blur_score > blurred_metrics.blur_score * 10

    def test_clipping_detection(self, tmp_path: Path) -> None:
        clipped_path = tmp_path / "clipped.jpg"
        # 80% blown-out white pixels
        img = np.full((480, 640, 3), 254, dtype=np.uint8)
        cv2.imwrite(str(clipped_path), img)

        metrics = compute_frame_quality(clipped_path)
        assert metrics is not None
        assert metrics.clipping_ratio > 0.75

    def test_corrupt_or_missing_image_returns_none(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "non_existent.jpg"
        assert compute_frame_quality(missing_path) is None

        corrupt_path = tmp_path / "corrupt.jpg"
        corrupt_path.write_bytes(b"NOT_A_JPEG_DATA")
        assert compute_frame_quality(corrupt_path) is None


class TestKeyframeSelector:
    """Test deterministic selection, overlap guard, and outputs."""

    def test_selection_determinism(self, fake_zurich_dataset: Path, tmp_path: Path) -> None:
        source = ZurichSequenceSource(dataset_dir=fake_zurich_dataset, start_id=0, count=5)
        config = KeyframeConfig(stride=1, min_blur=10.0, max_clipping=0.5, max_gap=15)
        selector = KeyframeSelector(config)

        run1 = tmp_path / "run1"
        run2 = tmp_path / "run2"

        res1 = selector.select(source, output_dir=run1)
        res2 = selector.select(source, output_dir=run2)

        assert res1.selected_count == res2.selected_count
        assert res1.selected_count == 5

        # Read generated keyframes.json and compare keyframe IDs
        data1 = json.loads((run1 / "keyframes.json").read_text(encoding="utf-8"))
        data2 = json.loads((run2 / "keyframes.json").read_text(encoding="utf-8"))

        ids1 = [k["frame_id"] for k in data1["keyframes"]]
        ids2 = [k["frame_id"] for k in data2["keyframes"]]
        assert ids1 == ids2

    def test_blur_quality_gate_rejects(self, fake_zurich_dataset: Path, tmp_path: Path) -> None:
        source = ZurichSequenceSource(dataset_dir=fake_zurich_dataset, start_id=0, count=5)
        # Set min_blur higher than fixture frames can reach
        config = KeyframeConfig(stride=1, min_blur=99999.0, max_clipping=0.5, max_gap=15)
        selector = KeyframeSelector(config)

        run_dir = tmp_path / "run_blur_reject"
        res = selector.select(source, output_dir=run_dir)
        assert res.selected_count == 0

    def test_overlap_guard_triggers_stride_reduction(self, tmp_path: Path) -> None:
        # Create sequence of 10 sharp frames
        img_dir = tmp_path / "dataset" / "MAV Images"
        img_dir.mkdir(parents=True)
        img = np.full((480, 640, 3), 120, dtype=np.uint8)
        for x in range(0, 640, 30):
            img[:, x : x + 15] = 180

        for i in range(10):
            cv2.imwrite(str(img_dir / f"frame_{i:05d}.jpg"), img)

        source = ZurichSequenceSource(dataset_dir=tmp_path / "dataset", start_id=0, count=10)

        # Initial stride=8, but max_gap=3 forces stride reduction
        config = KeyframeConfig(stride=8, min_blur=10.0, max_clipping=0.5, max_gap=3)
        selector = KeyframeSelector(config)

        run_dir = tmp_path / "run_overlap"
        res = selector.select(source, output_dir=run_dir)

        assert res.initial_stride == 8
        assert res.final_stride < 8
        assert res.stride_reductions > 0
        assert res.selected_count >= 3

    def test_decode_failure_handled_gracefully(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "corrupt_ds" / "MAV Images"
        img_dir.mkdir(parents=True)

        # 2 good frames, 1 corrupt frame
        good_img = np.full((480, 640, 3), 120, dtype=np.uint8)
        good_img[100:200, 100:200] = 180
        cv2.imwrite(str(img_dir / "frame_00000.jpg"), good_img)
        (img_dir / "frame_00001.jpg").write_bytes(b"CORRUPT")
        cv2.imwrite(str(img_dir / "frame_00002.jpg"), good_img)

        source = ZurichSequenceSource(dataset_dir=tmp_path / "corrupt_ds", start_id=0, count=3)
        config = KeyframeConfig(stride=1, min_blur=5.0, max_clipping=0.5, max_gap=15)
        selector = KeyframeSelector(config)

        run_dir = tmp_path / "run_corrupt"
        res = selector.select(source, output_dir=run_dir)

        assert res.decode_failed_count == 1
        assert res.selected_count == 2

    def test_schema_round_trip(self, fake_zurich_dataset: Path, tmp_path: Path) -> None:
        source = ZurichSequenceSource(dataset_dir=fake_zurich_dataset, start_id=0, count=5)
        config = KeyframeConfig(stride=1, min_blur=10.0, max_clipping=0.5, max_gap=15)
        selector = KeyframeSelector(config)

        run_dir = tmp_path / "run_schema"
        res = selector.select(source, output_dir=run_dir)

        # Validate against Pydantic schema
        content = res.keyframes_json_path.read_text(encoding="utf-8")
        parsed = KeyframesFile.model_validate_json(content)
        assert parsed.schema_version == "1.0.0"
        assert len(parsed.keyframes) == res.selected_count
        assert parsed.keyframes[0].file_path.startswith("sfm/images/")


class TestCliKeyframesCommand:
    """Test uav3d keyframes CLI subcommand."""

    def test_cli_keyframes_run(self, fake_zurich_dataset: Path, tmp_path: Path) -> None:
        config_path = tmp_path / "keyframes_cfg.yaml"
        cfg_dict = {
            "data_dir": str(tmp_path),
            "dataset_label": "replayed public dataset",
            "source": {
                "type": "zurich",
                "dataset_dir": str(fake_zurich_dataset),
                "start_id": 0,
                "count": 5,
                "stride": 1,
            },
            "keyframes": {
                "stride": 1,
                "min_blur": 10.0,
                "max_clipping": 0.5,
                "max_gap": 15,
            },
        }
        config_path.write_text(yaml.dump(cfg_dict), encoding="utf-8")

        exit_code = main(["keyframes", "--config", str(config_path)])
        assert exit_code == 0

    def test_keyframe_selection_from_video(self, tmp_path: Path) -> None:
        """Verify that KeyframeSelector extracts and saves best frames from an MP4 video.

        The video is generated inside the test so the suite carries no dependency
        on a data/interim/synthetic.mp4 left over from a manual run.
        """
        from uav3d.sources.video import VideoFileSource

        video_path = tmp_path / "mini.mp4"
        width, height, num_frames = 320, 240, 20
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
        assert writer.isOpened(), "OpenCV could not open an MP4 writer in this environment"
        rng = np.random.default_rng(11)
        for i in range(num_frames):
            frame = np.full((height, width, 3), 40, dtype=np.uint8)
            for _ in range(60):
                x, y = int(rng.integers(5, width - 25)), int(rng.integers(5, height - 25))
                frame[y : y + 8, x : x + 8] = rng.integers(0, 255, 3)
            cv2.putText(frame, f"{i:03d}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            writer.write(frame)
        writer.release()
        assert video_path.stat().st_size > 0

        source = VideoFileSource(video_path=video_path)
        selector = KeyframeSelector(
            KeyframeConfig(
                stride=4,
                min_blur=1.0,
                max_clipping=0.9,
                max_gap=20,
            )
        )
        res = selector.select(source, output_dir=tmp_path)
        assert res.selected_count > 0
        assert res.keyframes_json_path.exists()
        extracted_images = list((tmp_path / "sfm" / "images").glob("*.jpg"))
        assert len(extracted_images) == res.selected_count
        assert all(img.stat().st_size > 1000 for img in extracted_images)

