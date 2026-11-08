"""Tests for focal-length selection on forward-flight video (sfm/focal_search.py)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from uav3d.cli import _resolve_calibration
from uav3d.config import PipelineConfig, SfmConfig
from uav3d.exceptions import Uav3dError
from uav3d.sfm import focal_search as fs
from uav3d.sfm.focal_search import (
    FocalCandidate,
    FocalSearchResult,
    _subset_names,
    _within,
)


def test_subset_names_even_spacing_and_bounds():
    names = [f"f{i:03d}.jpg" for i in range(40)]
    sub = _subset_names(names, 14)
    assert len(sub) == 14
    assert sub[0] == names[0] and sub[-1] == names[-1]
    assert sub == sorted(sub)
    # short lists are returned whole
    assert _subset_names(names[:5], 14) == names[:5]


def test_within_requires_same_registration_and_relative_tolerance():
    a = FocalCandidate(900.0, 14, 0.90, 100)
    assert _within(FocalCandidate(700.0, 14, 0.93, 100), a, 0.05)
    assert not _within(FocalCandidate(700.0, 14, 1.25, 100), a, 0.05)  # bent-solution style gap
    assert not _within(FocalCandidate(700.0, 13, 0.90, 100), a, 0.05)  # fewer registered
    assert not _within(FocalCandidate(700.0, 0, None, 0), a, 0.05)


def test_result_note_and_dict_reflect_observability():
    cands = [FocalCandidate(972.0, 14, 1.379, 13282), FocalCandidate(729.0, 14, 0.894, 12767)]
    fixed = FocalSearchResult(972.0, 729.0, False, cands, {"x0.8": 0.0, "x1.25": 0.0})
    assert "fixed at 729 px" in fixed.note()
    assert "too little camera rotation" in fixed.note()
    d = fixed.as_dict()
    assert d["selected_focal_px"] == 729.0 and d["focal_observable"] is False
    assert d["candidates"][0]["mean_reproj_px"] == 1.379

    refined = FocalSearchResult(972.0, 972.0, True, cands)
    assert "refined" in refined.note()


def test_sfm_config_focal_search_defaults_and_validation():
    cfg = SfmConfig()
    assert cfg.focal_search is False  # opt-in; the CLI enables it for auto-intrinsics video
    assert cfg.focal_search_factors[0] == 1.0
    with pytest.raises(ValueError):
        SfmConfig(focal_search_max_images=2)


def _fake_trial(table: dict[int, tuple[int, float | None]], calls: list[float]):
    """Build a _trial_mapping stand-in returning (registered, reproj) per rounded focal."""

    def trial(base_db, image_dir, names, focal, cfg, workdir, native_log_path):
        calls.append(focal)
        reg, err = table[round(focal)]
        return FocalCandidate(focal, reg, err, 100 if err else 0), None

    return trial


def test_search_picks_good_regime_and_stops_after_tie(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # prior 1000: bent (1.4 px); 750 and 550 fit well and tie -> pick nearest-to-prior of the tie, stop before 400
    calls: list[float] = []
    table = {1000: (14, 1.40), 750: (14, 0.90), 550: (14, 0.89), 400: (14, 0.89)}
    monkeypatch.setattr(fs, "_trial_mapping", _fake_trial(table, calls))
    cfg = SfmConfig(focal_search=True)
    res = fs.search_focal(tmp_path / "db", tmp_path, [f"i{k}.jpg" for k in range(20)], 1000.0, cfg, tmp_path / "log")
    assert calls == [1000.0, 750.0, 550.0]
    assert res.selected_focal_px == 750.0
    assert res.prior_focal_px == 1000.0
    assert len(res.candidates) == 3
    # no model on disk in the fake -> observability test is skipped and defaults to "observable"
    assert res.focal_observable is True


def test_search_keeps_prior_when_it_already_fits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[float] = []
    table = {1000: (14, 0.50), 750: (14, 0.51), 550: (14, 0.52), 400: (14, 0.53), 300: (14, 0.54)}
    monkeypatch.setattr(fs, "_trial_mapping", _fake_trial(table, calls))
    res = fs.search_focal(
        tmp_path / "db", tmp_path, [f"i{k}.jpg" for k in range(20)], 1000.0, SfmConfig(focal_search=True), tmp_path / "log"
    )
    assert res.selected_focal_px == 1000.0


def test_search_prefers_more_registered_images(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[float] = []
    table = {1000: (9, 0.40), 750: (14, 0.90), 550: (14, 0.91), 400: (14, 0.91), 300: (14, 0.91)}
    monkeypatch.setattr(fs, "_trial_mapping", _fake_trial(table, calls))
    res = fs.search_focal(
        tmp_path / "db", tmp_path, [f"i{k}.jpg" for k in range(20)], 1000.0, SfmConfig(focal_search=True), tmp_path / "log"
    )
    assert res.selected_focal_px == 750.0


def test_search_falls_back_to_prior_when_all_trials_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[float] = []
    table = {1000: (0, None), 750: (0, None), 550: (0, None), 400: (0, None), 300: (0, None)}
    monkeypatch.setattr(fs, "_trial_mapping", _fake_trial(table, calls))
    res = fs.search_focal(
        tmp_path / "db", tmp_path, [f"i{k}.jpg" for k in range(20)], 1000.0, SfmConfig(focal_search=True), tmp_path / "log"
    )
    assert res.selected_focal_px == 1000.0 and res.focal_observable is True


def test_focal_length_setter_updates_pinhole_family():
    """cam.focal_length is how the module sets focal for SIMPLE_* and PINHOLE models."""
    simple = pycolmap.Camera(model="SIMPLE_PINHOLE", width=100, height=80, params=[10.0, 50.0, 40.0])
    simple.focal_length = 33.0
    assert np.allclose(simple.params, [33.0, 50.0, 40.0])
    pin = pycolmap.Camera(model="PINHOLE", width=100, height=80, params=[10.0, 11.0, 50.0, 40.0])
    pin.focal_length = 33.0
    assert np.allclose(pin.params, [33.0, 33.0, 50.0, 40.0])


def _video_args(video: Path, focal: float | None) -> argparse.Namespace:
    return argparse.Namespace(video=str(video), focal_px=focal)


def test_resolve_calibration_auto_enables_focal_search(tmp_path: Path):
    cfg = PipelineConfig()
    cal = _resolve_calibration(tmp_path, ["video_000000.jpg"], cfg, str(tmp_path), _video_args(tmp_path / "v.mp4", None))
    assert cal.model == "SIMPLE_PINHOLE"
    assert cfg.sfm.focal_search is True and cfg.sfm.refine_intrinsics is True


def test_resolve_calibration_user_focal_is_fixed(tmp_path: Path):
    cfg = PipelineConfig()
    cal = _resolve_calibration(tmp_path, ["video_000000.jpg"], cfg, str(tmp_path), _video_args(tmp_path / "v.mp4", 1234.5))
    assert cal.params[0] == 1234.5
    assert cfg.sfm.focal_search is False and cfg.sfm.refine_intrinsics is False


def test_resolve_calibration_rejects_nonpositive_focal(tmp_path: Path):
    with pytest.raises(Uav3dError):
        _resolve_calibration(
            tmp_path, ["video_000000.jpg"], PipelineConfig(), str(tmp_path), _video_args(tmp_path / "v.mp4", -5.0)
        )
