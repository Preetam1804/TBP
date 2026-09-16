"""Tests for uav3d schemas, config loading, and CLI commands."""

import json
from pathlib import Path

import pytest
import yaml
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
        assert "replayed public dataset" in cfg.dataset_label
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

    def test_missing_dataset_fails_loudly(self, tmp_path: Path) -> None:
        """A missing dataset must produce a clear non-zero failure, not a silent pass."""
        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "data_dir": str(tmp_path),
                    "source": {"type": "zurich", "dataset_dir": str(tmp_path / "absent")},
                }
            ),
            encoding="utf-8",
        )
        assert main(["inspect", "--config", str(config_path)]) == 1


class TestEndToEndPipeline:
    """Run all five stages on a self-contained, reconstructable mini dataset.

    This replaces the previous smoke test that depended on a run directory from
    the author's machine (`data/outputs/run_20260914_112220`), which made the
    sync and validate coverage permanently fail on any other checkout.
    """

    @pytest.fixture
    def run_dir(self, mini_recon_dataset: Path, tmp_path: Path) -> Path:
        """Execute inspect -> keyframes -> sfm -> sync -> validate and return the run directory."""
        config_path = tmp_path / "mini_cfg.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "data_dir": str(tmp_path),
                    "dataset_label": "unit-test mini dataset",
                    "source": {
                        "type": "zurich",
                        "dataset_dir": str(mini_recon_dataset),
                        "start_id": 0,
                        "count": 10,
                        "stride": 1,
                    },
                    "keyframes": {"stride": 1, "min_blur": 10.0, "max_clipping": 0.5, "max_gap": 15},
                    "sfm": {"seq_overlap": 5, "max_image_size": 640},
                }
            ),
            encoding="utf-8",
        )

        assert main(["inspect", "--config", str(config_path)]) == 0
        assert main(["keyframes", "--config", str(config_path)]) == 0

        outputs_dir = tmp_path / "outputs"
        run_dirs = sorted(d for d in outputs_dir.iterdir() if d.is_dir())
        assert run_dirs, "keyframes stage produced no run directory"
        run_dir = run_dirs[-1]

        assert main(["sfm", "--config", str(config_path), "--run-dir", str(run_dir)]) == 0
        assert main(["sync", "--config", str(config_path), "--run-dir", str(run_dir)]) == 0
        assert main(["validate", "--config", str(config_path), "--run-dir", str(run_dir)]) == 0
        return run_dir

    def test_all_artifacts_are_written(self, run_dir: Path) -> None:
        """Every documented stage artifact exists and is non-empty."""
        for name in (
            "keyframes.json",
            "poses.json",
            "telemetry.json",
            "poses_enriched.json",
            "ate_report.json",
            "SP1_REPORT.md",
            "trajectory_3d.png",
            "ate_error_profile.png",
        ):
            artifact = run_dir / name
            assert artifact.is_file(), f"missing artifact {name}"
            assert artifact.stat().st_size > 0

    def test_poses_schema_and_registration(self, run_dir: Path) -> None:
        """poses.json validates against the schema and registers the mini scene."""
        poses_file = PosesFile.model_validate_json((run_dir / "poses.json").read_text(encoding="utf-8"))
        assert poses_file.metrics is not None
        assert poses_file.metrics.num_images_registered >= 8
        assert poses_file.metrics.registration_rate >= 0.8
        assert poses_file.metrics.mean_reprojection_error_px <= 1.5
        assert poses_file.camera is not None
        # Locked intrinsics must be reported as locked, with the real calibration.
        assert poses_file.camera.intrinsics_locked is True
        assert poses_file.camera.model == "OPENCV"
        assert poses_file.camera.intrinsics_source == "calibration_data.npz"
        assert poses_file.camera.horizontal_fov_deg is not None

    def test_telemetry_matches_every_frame(self, run_dir: Path) -> None:
        """Mode A joins the dataset image IDs exactly and never reports a false mapping."""
        telemetry = TelemetryFile.model_validate_json((run_dir / "telemetry.json").read_text(encoding="utf-8"))
        assert telemetry.sync_status == "dataset_image_id_mapping"
        assert telemetry.summary["matched_telemetry_frames"] == telemetry.summary["total_frames"]
        assert all(r.lat is not None for r in telemetry.records)

    def test_validation_report_records_ground_truth_provenance(self, run_dir: Path) -> None:
        """ate_report.json must state which ground truth produced the numbers."""
        report = json.loads((run_dir / "ate_report.json").read_text(encoding="utf-8"))
        provenance = report["ground_truth"]
        assert provenance["selection"] == "auto_discovered_from_source"
        assert provenance["file"].endswith("GroundTruthAGL.csv")
        assert provenance["ate_threshold_m"] == 5.0
        assert report["status"] == "completed"
        assert report["metrics"]["num_matched_frames"] >= 8

    def test_validate_refuses_run_without_ground_truth_definition(self, run_dir: Path) -> None:
        """A video-namespace run must not be graded against unrelated ground truth."""
        poses_path = run_dir / "poses.json"
        payload = json.loads(poses_path.read_text(encoding="utf-8"))
        for pose in payload["poses"]:
            pose["frame_id"] = pose["frame_id"].replace("zurich_", "video_")
        payload["unregistered_frames"] = []
        poses_path.write_text(json.dumps(payload), encoding="utf-8")

        # Without --gt-file the CLI must refuse rather than silently reuse the dataset GT.
        assert main(["validate", "--run-dir", str(run_dir)]) == 1
