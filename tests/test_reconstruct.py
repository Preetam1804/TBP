"""Tests for the single-pass `reconstruct` command and its 3D model export."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from uav3d.cli import main
from uav3d.exceptions import Uav3dError
from uav3d.modeling.export import _write_trajectory_ply
from uav3d.sfm.native_log import NativeStderrCapture


def _latest_run_dir(data_dir: Path) -> Path:
    runs = sorted((Path(data_dir) / "outputs").glob("run_*"))
    assert runs, "no run directory created"
    return runs[-1]


def test_reconstruct_end_to_end(tmp_path: Path, parallax_video: Path):
    """Full pipeline: video -> keyframes -> SfM -> 3D model artifacts."""
    data_dir = tmp_path / "data"
    rc = main(
        [
            "reconstruct",
            "--video",
            str(parallax_video),
            "--data-dir",
            str(data_dir),
            "--max-frames",
            "20",
        ]
    )
    assert rc == 0

    run_dir = _latest_run_dir(data_dir)

    # SP-1 contracts
    assert (run_dir / "keyframes.json").exists()
    assert (run_dir / "poses.json").exists()
    assert (run_dir / "colmap_native.log").exists()

    # 3D model artifacts
    model_ply = run_dir / "model.ply"
    assert model_ply.exists()
    ply_text = model_ply.read_text(encoding="utf-8", errors="replace")
    assert ply_text.startswith("ply")
    assert model_ply.stat().st_size > 200

    model_dir = run_dir / "model"
    assert model_dir.is_dir()
    for required in ("cameras.txt", "images.txt", "points3D.txt"):
        assert (model_dir / required).exists(), f"missing {required} in COLMAP model dir"

    # Trajectory PLY is a well-formed polyline
    traj_text = (run_dir / "trajectory.ply").read_text(encoding="utf-8")
    lines = traj_text.splitlines()
    n_vertices = n_edges = None
    for line in lines:
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            n_vertices = int(parts[2])
        elif parts[:2] == ["element", "edge"]:
            n_edges = int(parts[2])
    assert n_vertices is not None and n_vertices >= 2
    assert n_edges == n_vertices - 1

    # Versioned summary manifest is consistent with the keyframe/pose files
    summary = json.loads((run_dir / "reconstruction_summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "1.0.0"
    assert summary["dataset_label"] == "user-provided video (field capture)"
    keyframes = json.loads((run_dir / "keyframes.json").read_text(encoding="utf-8"))
    assert summary["num_keyframes"] == len(keyframes["keyframes"])
    assert summary["num_registered_frames"] == summary["num_keyframes"] - summary["num_unregistered_frames"]
    assert summary["num_registered_frames"] >= 2
    assert 0.0 < summary["registration_rate"] <= 1.0
    assert summary["num_points3d"] >= 1
    # video-only run: sync/ATE skipped, and the manifest reflects that
    assert summary["telemetry_json"] is None
    assert summary["ate_report_json"] is None


def test_reconstruct_respects_max_frames_budget(
    tmp_path: Path, parallax_video: Path, caplog: pytest.LogCaptureFixture
):
    """--max-frames bounds evaluated frames (pre-stride) and selected keyframes."""
    import logging

    data_dir = tmp_path / "data"

    # keyframes stage: hard cap on both candidates evaluated and keyframes selected
    caplog.set_level(logging.INFO, logger="uav3d.keyframes.selector")
    rc = main(["keyframes", "--video", str(parallax_video), "--data-dir", str(data_dir), "--max-frames", "10"])
    assert rc == 0
    run_dir = _latest_run_dir(data_dir)
    sfm_images = list((run_dir / "sfm" / "images").iterdir())
    assert len(sfm_images) <= 10
    keyframes = json.loads((run_dir / "keyframes.json").read_text(encoding="utf-8"))
    assert len(keyframes["keyframes"]) <= 10
    assert any("pre-striding with stride=4" in r.getMessage() for r in caplog.records)

    # reconstruct stage: same budget, terminates cleanly (rc 0, or an informative
    # rc 1 if the budgeted keyframes end up too far apart for SfM initialization).
    rc2 = main(["reconstruct", "--video", str(parallax_video), "--data-dir", str(data_dir), "--max-frames", "10"])
    assert rc2 in (0, 1)
    run_dir2 = _latest_run_dir(data_dir)
    keyframes2 = json.loads((run_dir2 / "keyframes.json").read_text(encoding="utf-8"))
    assert len(keyframes2["keyframes"]) <= 10


def test_reconstruct_single_model_options_plumbed(tmp_path: Path):
    """SfmRunner forwards multiple_models/init_num_trials to pycolmap options."""
    import pycolmap as pc

    from uav3d.config import SfmConfig
    from uav3d.models.frames import CameraCalibration
    from uav3d.sfm.runner import SfmRunner

    captured: dict = {}

    def fake_incremental_mapping(database_path, image_path, output_path, options):
        captured["options"] = options
        recon = pc.Reconstruction()
        return {0: recon}

    class FakeDatabase:
        def open(self, path):
            pass

        def read_all_cameras(self):
            return []

        def write_camera(self, cam):
            pass

        def close(self):
            pass

    cfg = SfmConfig(
        multiple_models=False,
        init_num_trials=7,
        refine_intrinsics=True,
        camera_model="PINHOLE",
    )
    runner = SfmRunner(config=cfg)
    calibration = CameraCalibration(
        model="PINHOLE", width=320, height=240, params=[288.0, 288.0, 160.0, 120.0], source="test"
    )

    images_dir = tmp_path / "images_stub"
    images_dir.mkdir(exist_ok=True)

    img = np.full((240, 320, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(images_dir / "stub.jpg"), img)

    def _run(runner_, out_name):
        with patch("uav3d.sfm.runner.pycolmap.extract_features", lambda **kw: None), patch(
            "uav3d.sfm.runner.pycolmap.match_sequential", lambda *a, **kw: None
        ), patch("uav3d.sfm.runner.pycolmap.Database", FakeDatabase), patch(
            "uav3d.sfm.runner.pycolmap.incremental_mapping", fake_incremental_mapping
        ):
            runner_.run(
                images_dir=images_dir,
                output_dir=tmp_path / out_name,
                calibration=calibration,
            )

    with pytest.raises(Uav3dError):
        _run(runner, "sfm_stub_out")
    opts = captured["options"]
    assert opts.multiple_models is False
    assert opts.init_num_trials == 7

    # And the flag flips with multiple_models=True
    cfg2 = SfmConfig(multiple_models=True, init_num_trials=33, refine_intrinsics=True, camera_model="PINHOLE")
    runner2 = SfmRunner(config=cfg2)
    captured.clear()
    with pytest.raises(Uav3dError):
        _run(runner2, "sfm_stub_out2")
    assert captured["options"].multiple_models is True
    assert captured["options"].init_num_trials == 33


def test_write_trajectory_ply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Trajectory PLY writer emits a valid vertex/edge pair set, ordered by frame id."""
    import uav3d.modeling.export as export_mod

    class FakeImage:
        def __init__(self, name: str):
            self.name = name

    class FakeRecon:
        def reg_image_ids(self):
            return [3, 1, 2]  # deliberately unordered

        def image(self, image_id):
            return FakeImage(f"video_{image_id:06d}.jpg")

    def fake_get_pose_components(image):
        idx = int(image.name.split("_")[1].split(".")[0])
        return [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [float(idx), 0.0, 0.0]

    monkeypatch.setattr(export_mod, "get_pose_components", fake_get_pose_components)

    out = tmp_path / "traj.ply"
    _write_trajectory_ply(FakeRecon(), out)
    text = out.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "ply"
    assert "element vertex 3" in lines
    assert "element edge 2" in lines
    # vertices must be ordered by frame id (video_000001, video_000002, video_000003)
    body = [l for l in lines if l and not l.startswith(("ply", "format", "element", "property", "end_", "comment"))]
    verts = body[:3]
    assert verts[0].startswith("1.000000 ")
    assert verts[1].startswith("2.000000 ")
    assert verts[2].startswith("3.000000 ")
    edges = body[-2:]
    assert edges == ["0 1", "1 2"]


def test_native_stderr_capture(tmp_path: Path, capfd: pytest.CaptureFixture):
    """NativeStderrCapture redirects fd 2 to a file and restores it afterwards."""
    log_path = tmp_path / "native.log"
    with NativeStderrCapture(log_path):
        sys.stderr.buffer.write(b"python-level goes to console\n")
        sys.stderr.buffer.flush()
        # write directly to native fd 2 (what the C++ library does)
        os_write(2, b"native line goes to file\n")
    captured = capfd.readouterr()
    assert "python-level goes to console" in captured.err
    content = log_path.read_text(encoding="utf-8")
    assert "native line goes to file" in content
    # fd restored: further native writes go back to stderr
    os_write(2, b"after restore\n")
    assert "after restore" in capfd.readouterr().err


def os_write(fd: int, data: bytes) -> None:
    import os

    os.write(fd, data)


def test_reconstruct_missing_video_errors(tmp_path: Path):
    rc = main(["reconstruct", "--video", str(tmp_path / "nope.mp4"), "--data-dir", str(tmp_path)])
    assert rc == 1
