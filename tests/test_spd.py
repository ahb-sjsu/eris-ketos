# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for SPD manifold analysis."""

import numpy as np
import pytest
import torch

from eris_ketos.spd_spectral import (
    SPDManifold,
    compute_covariance,
    compute_spectral_trajectory,
    spd_features_from_spectrogram,
)


class TestSPDManifold:
    @pytest.fixture
    def spd_matrices(self) -> torch.Tensor:
        """Generate random SPD matrices."""
        n = 8
        batch = 5
        A = torch.randn(batch, n, n)
        # S = A @ A^T + eps*I is guaranteed SPD
        return A @ A.transpose(-2, -1) + 0.1 * torch.eye(n).unsqueeze(0)

    def test_log_map_returns_symmetric(self, spd_matrices: torch.Tensor) -> None:
        log_S = SPDManifold.log_map(spd_matrices)
        # log(S) should be symmetric
        diff = (log_S - log_S.transpose(-2, -1)).abs().max()
        assert diff < 1e-5

    def test_exp_log_roundtrip(self, spd_matrices: torch.Tensor) -> None:
        log_S = SPDManifold.log_map(spd_matrices)
        recovered = SPDManifold.exp_map(log_S)
        torch.testing.assert_close(recovered, spd_matrices, atol=1e-4, rtol=1e-4)

    def test_distance_symmetric(self, spd_matrices: torch.Tensor) -> None:
        S1 = spd_matrices[:3]
        S2 = spd_matrices[2:]
        d12 = SPDManifold.distance(S1, S2)
        d21 = SPDManifold.distance(S2, S1)
        torch.testing.assert_close(d12, d21, atol=1e-5, rtol=1e-5)

    def test_distance_to_self_zero(self, spd_matrices: torch.Tensor) -> None:
        d = SPDManifold.distance(spd_matrices, spd_matrices)
        torch.testing.assert_close(d, torch.zeros_like(d), atol=1e-5, rtol=1e-5)

    def test_distance_non_negative(self, spd_matrices: torch.Tensor) -> None:
        S1 = spd_matrices[:3]
        S2 = spd_matrices[2:]
        d = SPDManifold.distance(S1, S2)
        assert (d >= 0).all()

    def test_frechet_mean_shape(self, spd_matrices: torch.Tensor) -> None:
        mean = SPDManifold.frechet_mean(spd_matrices)
        n = spd_matrices.shape[-1]
        assert mean.shape == (n, n)

    def test_frechet_mean_is_spd(self, spd_matrices: torch.Tensor) -> None:
        mean = SPDManifold.frechet_mean(spd_matrices)
        eigvals = torch.linalg.eigvalsh(mean)
        assert (eigvals > 0).all()

    def test_frechet_mean_of_one(self, spd_matrices: torch.Tensor) -> None:
        single = spd_matrices[0:1]
        mean = SPDManifold.frechet_mean(single)
        torch.testing.assert_close(mean, single[0], atol=1e-5, rtol=1e-5)


class TestCovariance:
    def test_output_shape(self) -> None:
        spec = np.random.randn(128, 100)
        cov = compute_covariance(spec, n_bands=16)
        assert cov.shape == (16, 16)

    def test_output_is_spd(self) -> None:
        spec = np.random.randn(128, 100)
        cov = compute_covariance(spec, n_bands=16)
        eigvals = np.linalg.eigvalsh(cov)
        assert (eigvals > 0).all()

    def test_output_is_symmetric(self) -> None:
        spec = np.random.randn(128, 100)
        cov = compute_covariance(spec, n_bands=16)
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)


class TestSPDFeatures:
    def test_feature_length(self) -> None:
        spec = np.random.randn(128, 100)
        features = spd_features_from_spectrogram(spec, n_bands=16)
        expected = 16 * 17 // 2  # upper triangle
        assert features.shape == (expected,)

    def test_features_are_finite(self) -> None:
        spec = np.random.randn(128, 100)
        features = spd_features_from_spectrogram(spec, n_bands=8)
        assert np.all(np.isfinite(features))


class TestSpectralTrajectory:
    def test_trajectory_shape(self) -> None:
        spec = np.random.randn(128, 200)
        traj = compute_spectral_trajectory(
            spec, n_bands=8, window_frames=32, hop_frames=16
        )
        assert traj.matrices.shape[1] == 8
        assert traj.matrices.shape[2] == 8
        assert len(traj.timestamps) == len(traj.matrices)

    def test_geodesic_deviation_non_negative(self) -> None:
        spec = np.random.randn(128, 200)
        traj = compute_spectral_trajectory(spec, n_bands=8)
        assert traj.geodesic_deviation >= 0
