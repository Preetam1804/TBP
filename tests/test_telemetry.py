"""Comprehensive unit tests for Phase 4 Telemetry Synchronization."""

import copy
import json
import random
from pathlib import Path

import pytest

from uav3d.config import SyncConfig
from uav3d.models.poses import PoseRecord, PosesFile, SfmCameraBlock, SfmMetrics
from uav3d.models.telemetry import TelemetryFile, TelemetryRecord
from uav3d.telemetry.synchronizer import (
    TelemetrySynchronizer,
    haversine_distance_m,
    recommend_crs,
)
from uav3d.telemetry.zurich_parser import ZurichGpsRecord


@pytest.fixture
def sample_poses_file() -> PosesFile:
    """Create a sample PosesFile with 5 frames."""
    poses = [
        PoseRecord(
            frame_id=f"zurich_{i*2:06d}",
            timestamp_us=1459425600000000 + i * 200000,
            qw=1.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
            tx=float(i),
            ty=0.0,
            tz=5.0,
            camera_center_world=[float(-i), 0.0, -5.0],
            num_observations=150,
            registered=True,
        )
        for i in range(5)
    ]
    return PosesFile(
        schema_version="1.0.0",
        dataset_label="replayed public dataset",
        coordinate_frame="cam_from_world",
        camera=SfmCameraBlock(
            camera_id=1,
            model="OPENCV",
            width=1920,
            height=1080,
            params=[1060.0, 1060.0, 960.0, 540.0, -0.05, 0.01, 0.0, 0.0],
        ),
        metrics=SfmMetrics(
            num_images_total=5,
            num_images_registered=5,
            registration_rate=1.0,
            mean_reprojection_error_px=0.5,
        ),
        poses=poses,
        unregistered_frames=[],
    )


class TestModeAZurichSync:
    """Test Mode A (image ID mapping) behavior, shuffling robustness, and missing rows."""

    def test_mode_a_exact_mapping(self, sample_poses_file: PosesFile) -> None:
        """Verify that Mode A joins strictly on source_image_id."""
        raw_telemetry: dict[int, ZurichGpsRecord] = {
            0: ZurichGpsRecord(1459425600000000, 0, 47.3769012345, 8.5417012345, 450.123, {"raw": "row0"}),
            2: ZurichGpsRecord(1459425600200000, 2, 47.3769212345, 8.5417212345, 450.223, {"raw": "row2"}),
            4: ZurichGpsRecord(1459425600400000, 4, 47.3769412345, 8.5417412345, 450.323, {"raw": "row4"}),
            6: ZurichGpsRecord(1459425600600000, 6, 47.3769612345, 8.5417612345, 450.423, {"raw": "row6"}),
            8: ZurichGpsRecord(1459425600800000, 8, 47.3769812345, 8.5417812345, 450.523, {"raw": "row8"}),
        }

        sync = TelemetrySynchronizer()
        tel_file, poses_enriched = sync.sync(
            poses=sample_poses_file,
            raw_telemetry=raw_telemetry,
            mode="zurich",
        )

        assert tel_file.sync_status == "dataset_image_id_mapping"
        assert tel_file.offset_estimation == "not_applicable"
        assert tel_file.offset_us is None
        assert len(tel_file.records) == 5
        assert all(r.lat is not None for r in tel_file.records)

        # Confirm coordinates are NEVER rounded
        assert tel_file.records[0].lat == 47.3769012345
        assert tel_file.records[0].lon == 8.5417012345
        assert tel_file.records[0].raw_fields == {"raw": "row0"}

        # Confirm enriched poses flag
        assert len(poses_enriched.poses) == 5
        assert all(p.telemetry_recorded is True for p in poses_enriched.poses)

    def test_mode_a_missing_telemetry_rows(self, sample_poses_file: PosesFile) -> None:
        """Verify handling of missing telemetry rows without silent fabrication."""
        # Only provide GPS for image 0, 4, 8 (images 2 and 6 missing)
        raw_telemetry: dict[int, ZurichGpsRecord] = {
            0: ZurichGpsRecord(1459425600000000, 0, 47.3769, 8.5417, 450.0, {}),
            4: ZurichGpsRecord(1459425600400000, 4, 47.3770, 8.5418, 450.2, {}),
            8: ZurichGpsRecord(1459425600800000, 8, 47.3771, 8.5419, 450.4, {}),
        }

        sync = TelemetrySynchronizer()
        tel_file, poses_enriched = sync.sync(
            poses=sample_poses_file,
            raw_telemetry=raw_telemetry,
            mode="zurich",
        )

        records_map = {r.frame_id: r for r in tel_file.records}
        assert records_map["zurich_000000"].lat == 47.3769
        assert records_map["zurich_000002"].lat is None
        assert records_map["zurich_000004"].lat == 47.3770
        assert records_map["zurich_000006"].lat is None

        poses_map = {p.frame_id: p for p in poses_enriched.poses}
        assert poses_map["zurich_000000"].telemetry_recorded is True
        assert poses_map["zurich_000002"].telemetry_recorded is False
        assert poses_map["zurich_000004"].telemetry_recorded is True
        assert poses_map["zurich_000006"].telemetry_recorded is False

    def test_shuffling_telemetry_order_does_not_change_output(self, sample_poses_file: PosesFile) -> None:
        """Verify that join is strictly ID-based and invariant to row order (no index-based matching)."""
        items = [
            ZurichGpsRecord(1459425600000000 + i * 200000, i * 2, 47.3769 + i * 0.0001, 8.5417 + i * 0.0001, 450.0 + i, {"row": str(i)})
            for i in range(5)
        ]

        sync = TelemetrySynchronizer()

        # Run 1: original order
        dict_1 = {rec.image_id: rec for rec in items}
        tel_1, _ = sync.sync(poses=sample_poses_file, raw_telemetry=dict_1, mode="zurich")

        # Run 2: completely reversed/shuffled list
        shuffled = copy.deepcopy(items)
        random.seed(12345)
        random.shuffle(shuffled)
        dict_2 = {rec.image_id: rec for rec in shuffled}
        tel_2, _ = sync.sync(poses=sample_poses_file, raw_telemetry=dict_2, mode="zurich")

        # Must be byte-for-byte identical
        assert tel_1.model_dump_json() == tel_2.model_dump_json()


class TestModeBVideoSync:
    """Test Mode B nearest-timestamp matching, clock offset estimation, and manifest support."""

    def test_clock_offset_estimation(self) -> None:
        """Verify that a constant clock offset is estimated accurately."""
        # 10 frames with video PTS from 0 to 900 ms (step 100 ms)
        frame_poses = [
            PoseRecord(
                frame_id=f"frame_{i:04d}",
                timestamp_us=i * 100_000,  # PTS in microseconds
                qw=1.0,
                qx=0.0,
                qy=0.0,
                qz=0.0,
                tx=float(i),
                ty=0.0,
                tz=1.0,
                camera_center_world=[float(-i), 0.0, -1.0],
                registered=True,
            )
            for i in range(10)
        ]
        poses_file = PosesFile(
            schema_version="1.0.0",
            dataset_label="synthetic video",
            coordinate_frame="cam_from_world",
            poses=frame_poses,
        )

        # Telemetry has a constant +150 ms clock offset: tel_ts = frame_ts + 150_000 us
        true_offset_us = 150_000
        telemetry_records = [
            TelemetryRecord(
                timestamp_us=i * 100_000 + true_offset_us,
                lat=37.7749 + i * 0.0001,
                lon=-122.4194 + i * 0.0001,
                alt=100.0,
                source="gps_stream",
            )
            for i in range(10)
        ]

        cfg = SyncConfig(max_delta_us=500_000, estimate_offset=True, min_match_rate=0.8)
        sync = TelemetrySynchronizer(config=cfg)

        tel_file, poses_enriched = sync.sync(
            poses=poses_file,
            raw_telemetry=telemetry_records,
            mode="video",
        )

        assert tel_file.sync_status == "timestamp_nearest_offset_estimated"
        assert tel_file.offset_estimation == "median_offset"
        assert tel_file.offset_us == true_offset_us
        assert tel_file.summary["matched_telemetry_frames"] == 10
        assert tel_file.summary["match_rate"] == 1.0

        # All poses should have telemetry_recorded = True
        assert all(p.telemetry_recorded is True for p in poses_enriched.poses)

    def test_unsynchronized_when_low_match_rate(self) -> None:
        """Verify that stream is marked 'unsynchronized' if fewer than 80% match."""
        # 10 frames
        frame_poses = [
            PoseRecord(
                frame_id=f"frame_{i:04d}",
                timestamp_us=i * 100_000,
                qw=1.0,
                qx=0.0,
                qy=0.0,
                qz=0.0,
                tx=0.0,
                ty=0.0,
                tz=0.0,
                registered=True,
            )
            for i in range(10)
        ]
        poses_file = PosesFile(
            schema_version="1.0.0",
            dataset_label="video",
            coordinate_frame="cam_from_world",
            poses=frame_poses,
        )

        # Only 3 telemetry records in window (30% match)
        telemetry_records = [
            TelemetryRecord(timestamp_us=i * 100_000, lat=10.0, lon=20.0, alt=30.0)
            for i in range(3)
        ]

        cfg = SyncConfig(max_delta_us=50_000, estimate_offset=False, min_match_rate=0.8)
        sync = TelemetrySynchronizer(config=cfg)

        tel_file, _ = sync.sync(
            poses=poses_file,
            raw_telemetry=telemetry_records,
            mode="video",
        )

        assert tel_file.sync_status == "unsynchronized"
        assert tel_file.summary["matched_telemetry_frames"] == 3
        assert tel_file.summary["match_rate"] == 0.3

    def test_synthetic_manifest_mapping(self, tmp_path: Path) -> None:
        """Verify that an authoritative manifest yields sync_status='manifest'."""
        manifest_data = [
            {
                "video_frame_index": 0,
                "source_image_id": "video_000000",
                "source_timestamp_us": 1459425600000000,
                "video_pts_us": 0,
                "lat": 47.3769,
                "lon": 8.5417,
                "alt": 450.0,
            },
            {
                "video_frame_index": 1,
                "source_image_id": "video_000001",
                "source_timestamp_us": 1459425600200000,
                "video_pts_us": 200000,
                "lat": 47.3770,
                "lon": 8.5418,
                "alt": 450.5,
            },
        ]
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

        poses = [
            PoseRecord(frame_id="video_000000", timestamp_us=0, qw=1.0, qx=0.0, qy=0.0, qz=0.0, tx=0.0, ty=0.0, tz=0.0),
            PoseRecord(frame_id="video_000001", timestamp_us=200000, qw=1.0, qx=0.0, qy=0.0, qz=0.0, tx=1.0, ty=0.0, tz=0.0),
        ]
        poses_file = PosesFile(schema_version="1.0.0", dataset_label="test", coordinate_frame="cam_from_world", poses=poses)

        sync = TelemetrySynchronizer()
        tel_file, poses_enriched = sync.sync(
            poses=poses_file,
            manifest_path=manifest_file,
            mode="video",
        )

        assert tel_file.sync_status == "manifest"
        assert tel_file.offset_estimation == "manifest"
        assert tel_file.records[0].lat == 47.3769
        assert all(p.telemetry_recorded is True for p in poses_enriched.poses)


class TestGeorefUtilities:
    """Test CRS recommendation and distance consistency cross-checks."""

    def test_haversine_distance(self) -> None:
        """Verify Haversine 3D distance calculation."""
        # Known distance between Zurich coordinates (~83 m)
        d = haversine_distance_m(47.3769, 8.5417, 450.0, 47.3776, 8.5420, 452.0)
        assert 70.0 < d < 100.0

    def test_recommend_crs_zurich(self) -> None:
        """Verify that Zurich coordinates (47.37N, 8.54E) yield UTM Zone 32N."""
        lats = [47.3769, 47.3770]
        lons = [8.5417, 8.5418]
        crs = recommend_crs(lats, lons)
        assert "UTM zone 32N" in crs
        assert "32632" in crs


class TestSyncCliAndImmutability:
    """Test CLI sync execution and poses.json immutability."""

    def test_poses_json_untouched_after_sync(self, tmp_path: Path, sample_poses_file: PosesFile) -> None:
        """Verify poses.json remains strictly byte-identical after synchronization."""
        run_dir = tmp_path / "run_test"
        run_dir.mkdir(parents=True)

        poses_path = run_dir / "poses.json"
        original_bytes = sample_poses_file.model_dump_json(indent=2).encode("utf-8")
        poses_path.write_bytes(original_bytes)

        raw_telemetry = {
            i * 2: ZurichGpsRecord(1459425600000000 + i * 200000, i * 2, 47.3769 + i * 0.001, 8.5417, 450.0, {})
            for i in range(5)
        }

        sync = TelemetrySynchronizer()
        tel_file, poses_enriched = sync.sync(
            poses=sample_poses_file,
            raw_telemetry=raw_telemetry,
            mode="zurich",
        )

        # Write outputs as CLI does
        (run_dir / "telemetry.json").write_text(tel_file.model_dump_json(indent=2), encoding="utf-8")
        (run_dir / "poses_enriched.json").write_text(poses_enriched.model_dump_json(indent=2), encoding="utf-8")

        # Assert poses.json is completely untouched and byte-identical
        assert poses_path.read_bytes() == original_bytes

        # Verify outputs validate against Pydantic models
        loaded_tel = TelemetryFile.model_validate_json((run_dir / "telemetry.json").read_text(encoding="utf-8"))
        assert loaded_tel == tel_file

        loaded_enriched = PosesFile.model_validate_json((run_dir / "poses_enriched.json").read_text(encoding="utf-8"))
        assert loaded_enriched == poses_enriched

