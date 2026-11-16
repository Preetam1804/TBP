"""Tests for the known-focal path for video runs.

Precedence for the prior focal (highest first):

1. explicit override — CLI ``--focal-px`` or config ``source.focal_px``
   (``VideoFileSource(focal_px=...)``; source ``"video_config_override"``)
2. sidecar — ``focal_px`` in ``<video-stem>.meta.json`` next to the video
   (source ``"video_meta_sidecar:<filename>"``)
3. guess — ``0.8 * width`` (source ``"video_estimate_<filename>"``)

An invalid sidecar is warned about and ignored, never fatal; an invalid
explicit override fails loudly. All fixtures are the synthetic
``fake_video_file`` (320x240, so the guess is 256.0 px) — never real footage.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from uav3d.cli import _build_source, create_parser
from uav3d.config import PipelineConfig, SourceConfig, load_config
from uav3d.exceptions import ConfigError, DataError
from uav3d.models.calibration_io import write_calibration
from uav3d.models.frames import CameraCalibration
from uav3d.sfm.focal_gate import describe_focal_source
from uav3d.sources.video import VideoFileSource

REPO_ROOT = Path(__file__).resolve().parent.parent

# 320x240 fixture video: guess 0.8*320 = 256.0, sane range 64..1600 px.
GUESS_FOCAL = 256.0
SIDECAR_FOCAL = 288.0
OVERRIDE_FOCAL = 200.0


def _write_sidecar(video: Path, payload) -> Path:
    sidecar = video.with_suffix(".meta.json")
    if isinstance(payload, str):
        sidecar.write_text(payload, encoding="utf-8")
    else:
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
    return sidecar


def _load_run_video_pipeline():
    spec = importlib.util.spec_from_file_location(
        "run_video_pipeline", REPO_ROOT / "scripts" / "run_video_pipeline.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSidecarDiscovery:
    def test_sidecar_focal_is_used(self, fake_video_file: Path) -> None:
        _write_sidecar(fake_video_file, {"focal_px": SIDECAR_FOCAL})
        calib = VideoFileSource(video_path=fake_video_file).calibration()
        assert calib.params[0] == SIDECAR_FOCAL
        assert calib.params[1] == SIDECAR_FOCAL
        assert calib.source == "video_meta_sidecar:mini_test.meta.json"
        assert calib.params[2] == 160.0  # cx = width / 2, unchanged
        assert calib.params[3] == 120.0  # cy = height / 2, unchanged

    def test_sidecar_extra_fields_are_ignored(self, fake_video_file: Path) -> None:
        # The real sidecar convention carries frames/fps/flight metadata too.
        _write_sidecar(
            fake_video_file,
            {"frames": 10, "width": 320, "height": 240, "fps": 10.0,
             "focal_px": SIDECAR_FOCAL, "seed": 1},
        )
        assert VideoFileSource(video_path=fake_video_file).calibration().params[0] == SIDECAR_FOCAL

    def test_no_sidecar_falls_back_to_guess(self, fake_video_file: Path) -> None:
        calib = VideoFileSource(video_path=fake_video_file).calibration()
        assert calib.params[0] == GUESS_FOCAL
        assert calib.source == "video_estimate_mini_test.mp4"


class TestPrecedence:
    def test_override_beats_sidecar(self, fake_video_file: Path) -> None:
        _write_sidecar(fake_video_file, {"focal_px": SIDECAR_FOCAL})
        calib = VideoFileSource(
            video_path=fake_video_file, focal_px=OVERRIDE_FOCAL
        ).calibration()
        assert calib.params[0] == OVERRIDE_FOCAL
        assert calib.source == "video_config_override"

    def test_sidecar_beats_guess(self, fake_video_file: Path) -> None:
        _write_sidecar(fake_video_file, {"focal_px": SIDECAR_FOCAL})
        calib = VideoFileSource(video_path=fake_video_file).calibration()
        assert calib.params[0] == SIDECAR_FOCAL
        assert calib.source.startswith("video_meta_sidecar:")

    def test_guess_is_default(self, fake_video_file: Path) -> None:
        calib = VideoFileSource(video_path=fake_video_file).calibration()
        assert calib.params[0] == GUESS_FOCAL
        assert calib.source.startswith("video_estimate")


class TestInvalidSidecarIgnored:
    @pytest.mark.parametrize(
        "payload",
        [
            {"focal_px": -5.0},          # non-positive
            {"focal_px": 0},             # zero
            {"focal_px": "288"},         # wrong type (string, not numeric)
            {"focal_px": None},          # null
            {"focal_px": True},          # bool is not a focal length
            {"focal_px": 100000.0},      # out of range (> 5*width = 1600)
            {"focal_px": 10.0},          # out of range (< 0.2*width = 64)
            {"frames": 10},              # no focal_px field at all
        ],
    )
    def test_invalid_sidecar_falls_back_to_guess_with_warning(
        self, fake_video_file: Path, payload, caplog
    ) -> None:
        _write_sidecar(fake_video_file, payload)
        with caplog.at_level(logging.WARNING, logger="uav3d.sources.video"):
            calib = VideoFileSource(video_path=fake_video_file).calibration()
        assert calib.params[0] == GUESS_FOCAL
        assert calib.source.startswith("video_estimate")
        assert "sidecar" in caplog.text.lower()

    def test_malformed_sidecar_json_falls_back_to_guess_with_warning(
        self, fake_video_file: Path, caplog
    ) -> None:
        _write_sidecar(fake_video_file, "{not json")
        with caplog.at_level(logging.WARNING, logger="uav3d.sources.video"):
            calib = VideoFileSource(video_path=fake_video_file).calibration()
        assert calib.params[0] == GUESS_FOCAL
        assert "sidecar" in caplog.text.lower()

    def test_sidecar_with_wrong_top_level_type_falls_back(
        self, fake_video_file: Path, caplog
    ) -> None:
        _write_sidecar(fake_video_file, [288.0])
        with caplog.at_level(logging.WARNING, logger="uav3d.sources.video"):
            calib = VideoFileSource(video_path=fake_video_file).calibration()
        assert calib.params[0] == GUESS_FOCAL


class TestConfigValidation:
    def test_config_accepts_positive_focal(self) -> None:
        cfg = SourceConfig(type="video", video_path="x.mp4", focal_px=288.0)
        assert cfg.focal_px == 288.0

    def test_config_defaults_to_no_focal(self) -> None:
        assert SourceConfig().focal_px is None

    @pytest.mark.parametrize("bad", [0.0, -10.0])
    def test_config_rejects_non_positive_focal(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            SourceConfig(type="video", video_path="x.mp4", focal_px=bad)

    def test_load_config_rejects_non_positive_focal(self, tmp_path: Path) -> None:
        config_path = tmp_path / "bad.yaml"
        config_path.write_text(
            "source:\n  type: video\n  video_path: x.mp4\n  focal_px: -5\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError):
            load_config(config_path)

    @pytest.mark.parametrize("bad", [0.0, -1.0, 10.0, 100000.0])
    def test_explicit_override_out_of_range_fails_loudly(
        self, fake_video_file: Path, bad: float
    ) -> None:
        # 0 / negative are non-positive; 10 px is below 0.2*320 = 64 and
        # 100000 px is above 5*320 = 1600. An explicit override is the
        # caller's stated ground truth, so it must fail, not guess.
        with pytest.raises(DataError):
            VideoFileSource(video_path=fake_video_file, focal_px=bad)


class TestCliFlagPlumbing:
    def test_keyframes_parser_accepts_focal_px(self, fake_video_file: Path) -> None:
        args = create_parser().parse_args(
            ["keyframes", "--video", str(fake_video_file), "--focal-px", "200"]
        )
        assert args.focal_px == 200.0

    def test_sfm_parser_accepts_focal_px(self) -> None:
        args = create_parser().parse_args(["sfm", "--focal-px", "200"])
        assert args.focal_px == 200.0

    def test_inspect_parser_accepts_focal_px(self) -> None:
        args = create_parser().parse_args(["inspect", "--focal-px", "200"])
        assert args.focal_px == 200.0

    def test_cli_flag_reaches_video_source(self, fake_video_file: Path) -> None:
        cfg = PipelineConfig()
        args = argparse.Namespace(
            video=str(fake_video_file), focal_px=OVERRIDE_FOCAL,
            start_frame=0, end_frame=None, data_dir=None,
        )
        source = _build_source(cfg, ".", args)
        assert isinstance(source, VideoFileSource)
        calib = source.calibration()
        assert calib.params[0] == OVERRIDE_FOCAL
        assert calib.source == "video_config_override"

    def test_cli_flag_beats_config_focal(self, fake_video_file: Path) -> None:
        cfg = PipelineConfig(
            source=SourceConfig(type="video", video_path=str(fake_video_file),
                                focal_px=SIDECAR_FOCAL)
        )
        args = argparse.Namespace(
            video=str(fake_video_file), focal_px=OVERRIDE_FOCAL,
            start_frame=0, end_frame=None, data_dir=None,
        )
        source = _build_source(cfg, ".", args)
        assert source.calibration().params[0] == OVERRIDE_FOCAL

    def test_config_focal_reaches_video_source(self, fake_video_file: Path) -> None:
        cfg = PipelineConfig(
            source=SourceConfig(type="video", video_path=str(fake_video_file),
                                focal_px=SIDECAR_FOCAL)
        )
        args = argparse.Namespace(
            video=str(fake_video_file), focal_px=None,
            start_frame=0, end_frame=None, data_dir=None,
        )
        source = _build_source(cfg, ".", args)
        assert source.calibration().params[0] == SIDECAR_FOCAL
        assert source.calibration().source == "video_config_override"

    def test_config_video_path_uses_config_focal(self, fake_video_file: Path) -> None:
        # The --video flag is absent: the source comes from the config.
        cfg = PipelineConfig(
            source=SourceConfig(type="video", video_path=str(fake_video_file),
                                focal_px=OVERRIDE_FOCAL)
        )
        args = argparse.Namespace(
            video=None, focal_px=None, start_frame=0, end_frame=None,
            data_dir=None,
        )
        source = _build_source(cfg, ".", args)
        assert source.calibration().params[0] == OVERRIDE_FOCAL


class TestDescribeFocalSource:
    def test_override_label(self) -> None:
        assert describe_focal_source("video_config_override") == "config/CLI override"

    def test_sidecar_label_names_the_file(self) -> None:
        label = describe_focal_source("video_meta_sidecar:city_flyover.meta.json")
        assert "sidecar" in label
        assert "city_flyover.meta.json" in label

    def test_guess_label(self) -> None:
        assert describe_focal_source("video_estimate_mini_test.mp4") == "0.8*width guess"

    def test_other_sources_pass_through(self) -> None:
        assert describe_focal_source("calibration_data.npz") == "calibration_data.npz"


class TestRunVideoPipelinePlumbing:
    def test_stage_keyframes_passes_focal_px(self, tmp_path: Path, monkeypatch) -> None:
        module = _load_run_video_pipeline()
        captured: list[list[str]] = []
        monkeypatch.setattr(module, "main", lambda argv: captured.append(list(argv)) or 0)
        monkeypatch.setattr(module, "newest_run_dir", lambda: tmp_path)
        video = tmp_path / "clip.mp4"
        config = tmp_path / "cfg.yaml"
        run_dir = module.stage_keyframes(video, config, focal_px=OVERRIDE_FOCAL)
        assert run_dir == tmp_path
        assert captured[0][:2] == ["keyframes", "--video"]
        assert "--focal-px" in captured[0]
        assert captured[0][captured[0].index("--focal-px") + 1] == "200.0"

    def test_stage_sfm_passes_focal_px(self, tmp_path: Path, monkeypatch) -> None:
        module = _load_run_video_pipeline()
        captured: list[list[str]] = []
        monkeypatch.setattr(module, "main", lambda argv: captured.append(list(argv)) or 0)
        (tmp_path / "poses.json").write_text("{}", encoding="utf-8")
        module.stage_sfm(tmp_path, tmp_path / "cfg.yaml", focal_px=OVERRIDE_FOCAL)
        assert "--focal-px" in captured[0]
        assert captured[0][captured[0].index("--focal-px") + 1] == "200.0"

    def test_stage_keyframes_without_focal_px_passes_no_flag(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        module = _load_run_video_pipeline()
        captured: list[list[str]] = []
        monkeypatch.setattr(module, "main", lambda argv: captured.append(list(argv)) or 0)
        monkeypatch.setattr(module, "newest_run_dir", lambda: tmp_path)
        module.stage_keyframes(tmp_path / "clip.mp4", tmp_path / "cfg.yaml")
        assert "--focal-px" not in captured[0]


class TestSummariseShowsFocalSource:
    def test_summarise_prints_sidecar_source(self, tmp_path: Path, capsys) -> None:
        module = _load_run_video_pipeline()
        write_calibration(
            CameraCalibration(
                model="PINHOLE", width=320, height=240,
                params=[SIDECAR_FOCAL, SIDECAR_FOCAL, 160.0, 120.0],
                source="video_meta_sidecar:mini_test.meta.json",
            ),
            tmp_path,
        )
        poses = {
            "camera": {"model": "SIMPLE_PINHOLE", "params": [300.0, 160.0, 120.0]},
            "metrics": {
                "num_images_registered": 10, "num_images_total": 10,
                "mean_reprojection_error_px": 0.5, "num_points3D": 100,
            },
        }
        info = {"width": 320, "height": 240, "frames": 10, "fps": 10.0, "seconds": 1.0}
        assert module.summarise(Path("mini_test.mp4"), info, tmp_path, poses, None) is True
        out = capsys.readouterr().out
        assert "sidecar" in out
        assert "mini_test.meta.json" in out
        assert "focal drift" in out
