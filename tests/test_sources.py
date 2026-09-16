"""Tests for ZurichSequenceSource, VideoFileSource, and telemetry parsers."""

import json
from pathlib import Path

import pytest
import yaml

from uav3d.cli import main
from uav3d.exceptions import DataError
from uav3d.sources.video import VideoFileSource
from uav3d.sources.zurich import ZurichSequenceSource
from uav3d.telemetry.zurich_parser import parse_ground_truth_agl, parse_onboard_gps


class TestZurichParser:
    """Test defensive CSV parsing for Zurich MAV logs."""

    def test_gps_parsing_and_us_normalization(self, fake_zurich_dataset: Path) -> None:
        gps_csv = fake_zurich_dataset / "Log Files" / "OnbordGPS.csv"
        records, skipped = parse_onboard_gps(gps_csv)
        assert skipped == 0
        assert len(records) == 4
        assert 0 in records
        assert records[0].timestamp_us == 1459425600000000
        assert isinstance(records[0].timestamp_us, int)
        assert records[0].lat == 47.376900
        assert records[0].lon == 8.541700
        assert records[0].alt == 450.00
        assert "lat" in records[0].raw_fields

    def test_malformed_csv_rows_skipped_and_counted(self, tmp_path: Path) -> None:
        bad_csv = tmp_path / "bad_gps.csv"
        bad_csv.write_text(
            "# timestamp, image_id, lat, lon\n"
            "1459425600000000, 0, 47.1, 8.1\n"
            "malformed_timestamp, 1, 47.2, 8.2\n"  # bad timestamp
            "1459425600200000\n"  # too few cols
            "1459425600300000, 3, 47.3, 8.3\n",
            encoding="utf-8",
        )
        records, skipped = parse_onboard_gps(bad_csv)
        assert len(records) == 2
        assert skipped == 2
        assert 0 in records
        assert 3 in records

    def test_duplicate_image_id_fails_loudly(self, tmp_path: Path) -> None:
        dup_csv = tmp_path / "dup_gps.csv"
        dup_csv.write_text(
            "# timestamp, image_id, lat, lon\n"
            "1459425600000000, 0, 47.1, 8.1\n"
            "1459425600100000, 0, 47.2, 8.2\n",  # duplicate image_id 0
            encoding="utf-8",
        )
        with pytest.raises(DataError) as exc_info:
            parse_onboard_gps(dup_csv)
        assert "duplicate image_id" in str(exc_info.value).lower()

    def test_ground_truth_parsing_evaluation_only(self, fake_zurich_dataset: Path) -> None:
        gt_csv = fake_zurich_dataset / "Log Files" / "GroundTruthAGL.csv"
        gt_records, skipped = parse_ground_truth_agl(gt_csv)
        assert skipped == 0
        assert len(gt_records) == 5
        assert gt_records[0].x == 10.0
        assert gt_records[0].y == 20.0
        assert gt_records[0].z == 30.0
        assert gt_records[0].is_evaluation_only is True


class TestZurichSequenceSource:
    """Test ZurichSequenceSource behavior, zero-padding, ordering, and missing telemetry."""

    def test_numeric_ordering_and_zero_padding(self, fake_zurich_dataset: Path) -> None:
        source = ZurichSequenceSource(dataset_dir=fake_zurich_dataset, start_id=0, count=10)
        meta = source.metadata()
        assert meta.frame_count == 5
        assert meta.first_frame_id == "zurich_000000"
        assert meta.last_frame_id == "zurich_000004"

        frame_ids = [f.frame_id for f in source.iter_frames()]
        assert frame_ids == [
            "zurich_000000",
            "zurich_000001",
            "zurich_000002",
            "zurich_000003",
            "zurich_000004",
        ]

    def test_missing_telemetry_never_fabricates_timestamp(self, fake_zurich_dataset: Path) -> None:
        source = ZurichSequenceSource(dataset_dir=fake_zurich_dataset, start_id=0, count=10)
        frames = list(source.iter_frames())

        # Frames 0-3 have telemetry
        for f in frames[:4]:
            assert f.timestamp_us is not None
            assert f.timestamp_source == "dataset_log"
            assert isinstance(f.timestamp_us, int)

        # Frame 4 was omitted from GPS csv: must have timestamp_us=None and timestamp_source="missing"
        assert frames[4].timestamp_us is None
        assert frames[4].timestamp_source == "missing"

    def test_subset_selection_with_stride(self, fake_zurich_dataset: Path) -> None:
        source = ZurichSequenceSource(dataset_dir=fake_zurich_dataset, start_id=1, count=2, stride=2)
        frames = list(source.iter_frames())
        assert len(frames) == 2
        assert frames[0].frame_id == "zurich_000001"
        assert frames[1].frame_id == "zurich_000003"

    def test_get_frame(self, fake_zurich_dataset: Path) -> None:
        source = ZurichSequenceSource(dataset_dir=fake_zurich_dataset)
        frame = source.get_frame("zurich_000002")
        assert frame.frame_id == "zurich_000002"
        assert frame.timestamp_us == 1459425600200000

    def test_calibration_loading(self, fake_zurich_dataset: Path) -> None:
        source = ZurichSequenceSource(dataset_dir=fake_zurich_dataset)
        calib = source.calibration()
        assert calib.model in ("PINHOLE", "OPENCV")
        assert len(calib.params) >= 4
        assert calib.params[0] == 1060.0


class TestVideoFileSource:
    """Test VideoFileSource frame iteration and PTS microsecond timestamps."""

    def test_video_iteration_and_pts(self, fake_video_file: Path) -> None:
        source = VideoFileSource(video_path=fake_video_file)
        meta = source.metadata()
        assert meta.frame_count == 10
        assert meta.fps == 10.0
        assert meta.width == 320
        assert meta.height == 240

        frames = list(source.iter_frames())
        assert len(frames) == 10
        assert frames[0].frame_id == "video_000000"
        assert frames[0].timestamp_source == "video_pts"
        assert isinstance(frames[0].timestamp_us, int)
        assert frames[9].frame_id == "video_000009"

    def test_video_get_frame(self, fake_video_file: Path) -> None:
        source = VideoFileSource(video_path=fake_video_file)
        frame = source.get_frame("video_000005")
        assert frame.frame_id == "video_000005"
        assert frame.timestamp_us == 500000  # 5th frame at 10 fps = 0.5s = 500,000 µs


class TestInspectCliWithFixture:
    """Test uav3d inspect CLI invocation against synthetic Zurich fixture."""

    def test_inspect_runs_successfully(self, fake_zurich_dataset: Path, tmp_path: Path) -> None:
        config_path = tmp_path / "test_config.yaml"
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
        }
        config_path.write_text(yaml.dump(cfg_dict), encoding="utf-8")

        # Run inspect via CLI main
        exit_code = main(["inspect", "--config", str(config_path)])
        assert exit_code == 0


class TestDatasetProvenanceLabelling:
    """Regression: locally generated sample data must never be labelled as the real dataset."""

    def test_plain_dataset_keeps_configured_label(self, fake_zurich_dataset: Path) -> None:
        """A directory without a marker is treated as whatever the config says."""
        source = ZurichSequenceSource(dataset_dir=fake_zurich_dataset, count=2)
        assert source.provenance is None
        assert source.dataset_label("replayed public dataset (Zurich Urban MAV)") == (
            "replayed public dataset (Zurich Urban MAV)"
        )

    def test_synthetic_marker_overrides_label(self, fake_zurich_dataset: Path) -> None:
        """A directory carrying the generator marker gets an explicit synthetic label."""
        marker = fake_zurich_dataset / "dataset_provenance.json"
        marker.write_text(
            json.dumps({"kind": "synthetic_sample", "generator": "scripts/generate_sample_zurich.py"}),
            encoding="utf-8",
        )
        source = ZurichSequenceSource(dataset_dir=fake_zurich_dataset, count=2)
        label = source.dataset_label("replayed public dataset (Zurich Urban MAV)")

        assert "synthetic sample dataset" in label
        assert "NOT the ETH Zurich Urban MAV dataset" in label
        assert "scripts/generate_sample_zurich.py" in label

    def test_corrupt_marker_still_flags_local_data(self, fake_zurich_dataset: Path) -> None:
        """An unreadable marker must fail safe (flag as local data), not silently pass."""
        (fake_zurich_dataset / "dataset_provenance.json").write_text("{not json", encoding="utf-8")
        source = ZurichSequenceSource(dataset_dir=fake_zurich_dataset, count=2)
        assert source.provenance is not None
        assert "synthetic sample dataset" in source.dataset_label("replayed public dataset")


class TestCalibrationResolution:
    """Verify calibration follows the pixels on disk instead of a hardcoded 1920x1080."""

    def test_calibration_uses_native_image_resolution(self, mini_recon_dataset: Path) -> None:
        """A 640x480 dataset must report 640x480 intrinsics geometry."""
        source = ZurichSequenceSource(dataset_dir=mini_recon_dataset, count=2)
        calib = source.calibration()

        assert (calib.width, calib.height) == (640, 480)
        assert calib.params[2] == 320.0
        assert calib.params[3] == 240.0
