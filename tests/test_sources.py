"""Tests for ZurichSequenceSource, VideoFileSource, and telemetry parsers."""

from pathlib import Path
from typing import ClassVar

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
        # A grab-only scan cannot validate container PTS, so it uses the deterministic estimate.
        assert frames[0].timestamp_source == "video_index_fps_estimate"
        assert frames[0].timestamp_us == 0
        assert isinstance(frames[0].timestamp_us, int)
        assert frames[9].frame_id == "video_000009"
        assert frames[9].timestamp_us == 900000  # 10 fps

    def test_video_get_frame(self, fake_video_file: Path) -> None:
        source = VideoFileSource(video_path=fake_video_file)
        frame = source.get_frame("video_000005")
        assert frame.frame_id == "video_000005"
        assert frame.timestamp_us == 500000  # 5th frame at 10 fps = 0.5s = 500,000 µs
        assert frame.timestamp_source == "video_pts"  # validated against index/fps

    def test_video_slice_and_timestamp_consistency(self, fake_video_file: Path) -> None:
        """The ingest window bounds work and both accessors report identical timestamps."""
        source = VideoFileSource(video_path=fake_video_file, start_frame=2, max_frames=4, stride=2)
        meta = source.metadata()
        assert meta.frame_count == 2
        assert meta.first_frame_id == "video_000002"
        assert meta.last_frame_id == "video_000004"

        iterated = {f.frame_id: f.timestamp_us for f in source.iter_frames()}
        assert list(iterated) == ["video_000002", "video_000004"]
        for frame_id, timestamp_us in iterated.items():
            assert source.get_frame(frame_id).timestamp_us == timestamp_us

    def test_long_video_is_not_fully_decoded(self, fake_video_file: Path) -> None:
        """A bounded slice must decode only the requested frames, not the whole file."""
        source = VideoFileSource(video_path=fake_video_file, max_frames=3)
        assert source.metadata().frame_count == 3
        decoded = [record.frame_id for record, _frame in source.iter_frames_with_pixels()]
        assert decoded == ["video_000000", "video_000001", "video_000002"]


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


class TestRealZurichTelemetryFormat:
    """Regression tests for the real AGZ log layout (documented in the dataset readme.txt).

    OnbordGPS.csv: `1: timestamp, 2: imgid, 3: lat (1E7 deg), 4: lon (1E7 deg),
    5: alt (1E3 m, millimetres), 6..: accuracy/velocity`.
    GroundTruthAGL.csv: `1: imgid, 2..4: x_gt, y_gt, z_gt` in WGS 84 / UTM 32N metres.
    The files may or may not carry a header line.
    """

    GPS_ROWS: ClassVar[list[str]] = [
        "1459425600000,0,473769000,85417000,450000,0.1,0.05,3,1.2,2.4,5.0,0.0,0.0,12",
        "1459425600200,1,473769150,85417018,450100,0.1,0.05,3,1.2,2.4,5.0,0.0,0.0,12",
        "1459425600400,2,473769300,85417036,450200,0.1,0.05,3,1.2,2.4,5.0,0.0,0.0,12",
    ]

    def test_headerless_fixed_point_gps_is_normalised(self, tmp_path: Path) -> None:
        """Lat/lon 1E7 fixed point, altitude millimetres and millisecond stamps are un-scaled."""
        csv_path = tmp_path / "OnbordGPS.csv"
        csv_path.write_text("\n".join(self.GPS_ROWS) + "\n", encoding="utf-8")

        records, skipped = parse_onboard_gps(csv_path)

        assert skipped == 0
        assert sorted(records) == [0, 1, 2], "the first data row must not be consumed as a header"
        assert records[0].lat == pytest.approx(47.3769)
        assert records[0].lon == pytest.approx(8.5417)
        assert records[0].alt == pytest.approx(450.0)
        assert records[1].lat == pytest.approx(47.376915)
        assert records[2].alt == pytest.approx(450.2)
        # 1459425600000 ms since epoch == 2016-03-31T12:00:00Z
        assert records[0].timestamp_us == 1459425600000000
        assert records[1].timestamp_us - records[0].timestamp_us == 200000
        # Traceability of the applied normalisation
        assert records[0].raw_fields["_timestamp_unit_detected"] == "milliseconds"
        assert records[0].raw_fields["_lat_scale"] == "1/1e+07"
        assert records[0].raw_fields["_alt_scale"] == "1/1e+03"

    def test_headers_and_microsecond_stamps_still_work(self, tmp_path: Path) -> None:
        """A commented header with degree/metre values keeps the legacy behaviour."""
        csv_path = tmp_path / "OnbordGPS.csv"
        csv_path.write_text(
            "# timestamp, image_id, lat, lon, alt\n"
            "1459425600000000, 7, 47.376900, 8.541700, 450.00\n",
            encoding="utf-8",
        )
        records, _ = parse_onboard_gps(csv_path)
        assert records[7].timestamp_us == 1459425600000000
        assert records[7].lat == pytest.approx(47.3769)
        assert records[7].alt == pytest.approx(450.0)

    def test_impossible_coordinates_fail_loudly(self, tmp_path: Path) -> None:
        """A latitude that cannot be degrees or 1E7-scaled degrees must be rejected, not stored."""
        csv_path = tmp_path / "OnbordGPS.csv"
        csv_path.write_text("1459425600000, 0, 9999999999, 85417000, 450000\n", encoding="utf-8")
        records, skipped = parse_onboard_gps(csv_path)
        assert skipped == 1
        assert records == {}

    def test_headerless_ground_truth(self, tmp_path: Path) -> None:
        """GroundTruthAGL.csv without a header keeps every row."""
        csv_path = tmp_path / "GroundTruthAGL.csv"
        csv_path.write_text(
            "0, 683100.0, 5248200.0, 450.0\n"
            "1, 683110.0, 5248210.0, 451.0\n"
            "2, 683120.0, 5248220.0, 452.0\n",
            encoding="utf-8",
        )
        records, skipped = parse_ground_truth_agl(csv_path)
        assert skipped == 0
        assert sorted(records) == [0, 1, 2]
        assert records[0].x == pytest.approx(683100.0)
        assert records[0].is_evaluation_only is True
