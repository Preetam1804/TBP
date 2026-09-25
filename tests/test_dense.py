"""Unit tests for dense TSDF reconstruction helpers (robust alignment math)."""

from __future__ import annotations

import numpy as np
import pytest

from uav3d.modeling.dense import (
    _align_disparity_to_depth,
    _sample_disparity,
    theil_sen_affine,
)


class TestTheilSenAffine:
    def test_recovers_exact_affine(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(0.2, 1.0, 200)
        y = 2.5 * x + 0.4
        fit = theil_sen_affine(x, y)
        assert fit is not None
        s, t = fit
        assert s == pytest.approx(2.5, rel=1e-6)
        assert t == pytest.approx(0.4, rel=1e-6)

    def test_robust_to_30_percent_outliers(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(0.2, 1.0, 300)
        y = 1.8 * x + 0.2 + rng.normal(0, 0.005, 300)
        outlier_idx = rng.choice(300, 90, replace=False)
        y[outlier_idx] += rng.uniform(1.0, 3.0, 90)  # gross outliers
        fit = theil_sen_affine(x, y)
        assert fit is not None
        s, t = fit
        assert s == pytest.approx(1.8, rel=0.05)
        assert t == pytest.approx(0.2, abs=0.05)

    def test_insufficient_data_returns_none(self):
        x = np.array([0.5, 0.6])
        y = np.array([1.0, 1.2])
        assert theil_sen_affine(x, y) is None

    def test_negative_slope_rejected(self):
        x = np.linspace(0.2, 1.0, 50)
        y = -2.0 * x  # inverse disparity can't have negative slope
        assert theil_sen_affine(x, y) is None


class TestAlignDisparityToDepth:
    def _make_obs(self, n=80, s=0.02, t=0.001, outlier_frac=0.0, seed=3):
        rng = np.random.default_rng(seed)
        z = rng.uniform(5.0, 60.0, n)
        disp = (1.0 / z - t) / s + rng.normal(0, 0.01, n)
        if outlier_frac > 0:
            k = int(n * outlier_frac)
            idx = rng.choice(n, k, replace=False)
            disp[idx] *= rng.uniform(2.0, 4.0, k)
        return disp, z

    def test_clean_fit_passes_gate(self):
        disp, z = self._make_obs()
        result = _align_disparity_to_depth(disp, z)
        assert result is not None
        s, _t, method, rmse = result
        assert method == "theil_sen_huber"
        assert s == pytest.approx(0.02, rel=0.1)
        assert rmse < 0.05

    def test_outliers_survive_via_robust_fit(self):
        disp, z = self._make_obs(outlier_frac=0.25)
        result = _align_disparity_to_depth(disp, z)
        assert result is not None
        s, _, _method, _ = result
        assert s == pytest.approx(0.02, rel=0.15)

    def test_garbage_fails_quality_gate(self):
        rng = np.random.default_rng(11)
        disp = rng.uniform(0.1, 2.0, 60)
        z = rng.uniform(5.0, 60.0, 60)  # unrelated
        assert _align_disparity_to_depth(disp, z) is None


class TestSampleDisparity:
    def test_bilinear_center_equals_value(self):
        disp = np.arange(100, dtype=np.float64).reshape(10, 10)
        uv = np.array([[3.0, 5.0]])
        assert _sample_disparity(disp, uv)[0] == pytest.approx(53.0)

    def test_bilinear_interpolates(self):
        disp = np.zeros((10, 10))
        disp[5, 3] = 4.0
        disp[5, 4] = 8.0
        val = _sample_disparity(disp, np.array([[3.5, 5.0]]))[0]
        assert val == pytest.approx(6.0)

    def test_clamps_out_of_bounds(self):
        disp = np.full((10, 10), 7.0)
        val = _sample_disparity(disp, np.array([[-5.0, 99.0]]))[0]
        assert val == pytest.approx(7.0)
