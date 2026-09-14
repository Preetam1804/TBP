"""Tests for uav3d schemas, config loading, and CLI commands."""

import pytest
from pydantic import ValidationError

from uav3d.cli import main
from uav3d.config import PipelineConfig, load_config
from uav3d.exceptions import ConfigError
from uav3d.models.frames import CameraCalibration, FrameRecord
from uav3d.models.keyframes import KeyframeRecord, KeyframesFile
from uav3d.models.poses import PoseRecord, PosesFile
from uav3d.models.telemetry import TelemetryFile, TelemetryRecord


class TestSchemaSerialization:
    """Validate serialization and deserialization of all Pydantic models."""

    def test_frame_record_round_trip(self) -> None:
        record = FrameRecord(
            frame_id="zurich_000000",
            timestamp_us=1459425600000000,
            file_path="data/raw/zurich/images/frame_00000.jpg",
            width=1920,
            height=1080,
        )
        serialized = record.model_dump_json()
        loaded = FrameRecord.model_validate_json(serialized)
        assert loaded == record
        assert isinstance(loaded.timestamp_us, int)

    def test_camera_calibration_round_trip(self) -> None:
        calib = CameraCalibration(
            model="PINHOLE",
            width=1920,
            height=1080,
            params=[1200.0, 1200.0, 960.0, 540.0],
            source="calibration_data.npz",
        )
        serialized = calib.model_dump_json()
        loaded = CameraCalibration.model_validate_json(serialized)
        assert loaded == calib

    def test_keyframes_file_round_trip(self) -> None:
        kf_file = KeyframesFile(
            schema_version="1.0.0",
            dataset_label="replayed public dataset",
            keyframes=[
                KeyframeRecord(
                    frame_id="zurich_000000",
                    timestamp_us=1459425600000000,
                    file_path="data/interim/keyframes/frame_00000.jpg",
                    score=142.5,
                ),
                KeyframeRecord(
                    frame_id="zurich_000005",
                    timestamp_us=1459425600166666,
                    file_path="data/interim/keyframes/frame_00005.jpg",
                    score=168.2,
                ),
            ],
        )
        serialized = kf_file.model_dump_json()
        loaded = KeyframesFile.model_validate_json(serialized)
        assert loaded == kf_file
        assert len(loaded.keyframes) == 2
        assert loaded.keyframes[0].timestamp_us == 1459425600000000

    def test_telemetry_file_round_trip(self) -> None:
        tel_file = TelemetryFile(
            schema_version="1.0.0",
            dataset_label="replayed public dataset",
            records=[
                TelemetryRecord(
                    timestamp_us=1459425600000000,
                    frame_id="zurich_000000",
                    lat=47.3769,
                    lon=8.5417,
                    alt=450.0,
                    easting=683100.0,
                    northing=5248200.0,
                    source="OnbordGPS.csv",
                ),
            ],
        )
        serialized = tel_file.model_dump_json()
        loaded = TelemetryFile.model_validate_json(serialized)
        assert loaded == tel_file
        assert loaded.records[0].easting == 683100.0

    def test_poses_file_round_trip(self) -> None:
        poses_file = PosesFile(
            schema_version="1.0.0",
            dataset_label="replayed public dataset",
            coordinate_frame="cam_from_world",
            poses=[
                PoseRecord(
                    frame_id="zurich_000000",
                    timestamp_us=1459425600000000,
                    qw=1.0,
                    qx=0.0,
                    qy=0.0,
                    qz=0.0,
                    tx=0.1,
                    ty=-0.2,
                    tz=1.5,
                    camera_center=[0.1, -0.2, 1.5],
                )
            ],
        )
        serialized = poses_file.model_dump_json()
        loaded = PosesFile.model_validate_json(serialized)
        assert loaded == poses_file
        assert loaded.poses[0].qvec == [1.0, 0.0, 0.0, 0.0]
        assert loaded.poses[0].tvec == [0.1, -0.2, 1.5]

    def test_extra_fields_forbidden(self) -> None:
        """Verify that extra='forbid' strictly rejects unknown fields."""
        with pytest.raises(ValidationError):
            FrameRecord(
                frame_id="zurich_000000",
                timestamp_us=1000,
                file_path="test.jpg",
                width=1920,
                height=1080,
                unknown_field="extra",  # type: ignore[call-arg]
            )

        with pytest.raises(ValidationError):
            KeyframesFile(
                schema_version="1.0.0",
                dataset_label="replayed public dataset",
                keyframes=[],
                invented_metadata="fail",  # type: ignore[call-arg]
            )


class TestConfigLoading:
    """Verify configuration parsing, validation, and fail-fast behavior."""

    def test_load_zurich_baseline(self) -> None:
        cfg = load_config("configs/zurich_baseline.yaml")
        assert isinstance(cfg, PipelineConfig)
        assert cfg.dataset_label == "replayed public dataset"
        assert cfg.sfm.camera_model == "OPENCV"
        assert cfg.sfm.single_camera is True
        assert cfg.sfm.seq_overlap == 10
        assert cfg.sfm.max_image_size == 1600
        assert cfg.sfm.refine_intrinsics is False
        assert cfg.keyframes.strategy == "laplacian_variance"

    def test_missing_config_raises(self) -> None:
        with pytest.raises(ConfigError) as exc_info:
            load_config("configs/non_existent_config.yaml")
        assert "not found" in str(exc_info.value).lower()

    def test_invalid_yaml_raises(self, tmp_path) -> None:
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("invalid: [unclosed list", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(bad_file)


class TestCliSmoke:
    """Test CLI commands and smoke help invocations."""

    def test_cli_help_exits_zero(self) -> None:
        assert main(["--help"]) == 0

    def test_cli_version_exits_zero(self) -> None:
        assert main(["--version"]) == 0

    @pytest.mark.parametrize("subcommand", ["inspect", "keyframes", "sfm", "sync", "validate"])
    def test_subcommand_help_exits_zero(self, subcommand: str) -> None:
        assert main([subcommand, "--help"]) == 0

    @pytest.mark.parametrize("subcommand", ["inspect", "keyframes", "sfm", "sync", "validate"])
    def test_subcommand_stub_execution(self, subcommand: str) -> None:
        assert main([subcommand, "--config", "configs/zurich_baseline.yaml"]) == 0
