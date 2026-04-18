# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for topological data analysis on click patterns."""

import numpy as np
import pytest

# ripser is the [tda] extra and is not installed by default. Skip the
# whole module when it is missing so the default test run stays green.
pytest.importorskip('ripser')

from eris_ketos.tda_clicks import (
    compute_persistence,
    subsample_cloud,
    tda_feature_vector,
    time_delay_embedding,
)


class TestTimeDelayEmbedding:
    def test_output_shape(self) -> None:
        signal = np.random.randn(1000)
        cloud = time_delay_embedding(signal, delay=5, dim=3)
        expected_n = 1000 - 2 * 5
        assert cloud.shape == (expected_n, 3)

    def test_signal_too_short(self) -> None:
        signal = np.random.randn(10)
        cloud = time_delay_embedding(signal, delay=10, dim=3)
        assert cloud.shape == (1, 3)

    def test_delay_1_dim_1(self) -> None:
        signal = np.arange(10, dtype=float)
        cloud = time_delay_embedding(signal, delay=1, dim=1)
        assert cloud.shape == (10, 1)
        np.testing.assert_array_equal(cloud[:, 0], signal)

    def test_values_correct(self) -> None:
        signal = np.arange(20, dtype=float)
        cloud = time_delay_embedding(signal, delay=3, dim=2)
        # cloud[i] = [signal[i], signal[i+3]]
        assert cloud[0, 0] == 0.0
        assert cloud[0, 1] == 3.0
        assert cloud[5, 0] == 5.0
        assert cloud[5, 1] == 8.0


class TestSubsampleCloud:
    def test_no_subsample_when_small(self) -> None:
        cloud = np.random.randn(50, 3)
        result = subsample_cloud(cloud, max_points=100)
        assert result.shape == (50, 3)

    def test_subsample_reduces(self) -> None:
        cloud = np.random.randn(5000, 3)
        result = subsample_cloud(cloud, max_points=200, seed=42)
        assert result.shape == (200, 3)

    def test_reproducible_with_seed(self) -> None:
        cloud = np.random.randn(1000, 3)
        r1 = subsample_cloud(cloud, max_points=100, seed=42)
        r2 = subsample_cloud(cloud, max_points=100, seed=42)
        np.testing.assert_array_equal(r1, r2)


class TestPersistence:
    @pytest.fixture
    def sine_signal(self) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi * 5, 5000)
        return np.sin(t).astype(np.float32)

    def test_compute_persistence_returns_result(self, sine_signal: np.ndarray) -> None:
        result = compute_persistence(sine_signal, delay=10, dim=3, max_points=300)
        assert result.max_dim == 1
        assert len(result.diagrams) >= 2
        assert result.cloud.shape[1] == 3

    def test_feature_vector_shape(self, sine_signal: np.ndarray) -> None:
        result = compute_persistence(sine_signal, delay=10, dim=3, max_points=300)
        features = tda_feature_vector(result)
        # 8 features per homology dim, 2 dims (H0, H1)
        assert features.shape == (16,)
        assert features.dtype == np.float32

    def test_sine_has_h1_features(self, sine_signal: np.ndarray) -> None:
        """A sine wave's time-delay embedding is a circle -> should have H1 features."""
        result = compute_persistence(sine_signal, delay=50, dim=2, max_points=500)
        h1 = result.diagrams[1]
        finite_h1 = h1[np.isfinite(h1[:, 1])]
        # Should have at least one loop
        assert len(finite_h1) > 0

    def test_missing_ripser_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ImportError with helpful message if ripser not installed."""

        orig = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name, *args, **kwargs):
            if name == "ripser":
                raise ImportError("No module named 'ripser'")
            return orig(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)
        with pytest.raises(ImportError, match="ripser is required"):
            compute_persistence(np.random.randn(100), delay=5, dim=2)
