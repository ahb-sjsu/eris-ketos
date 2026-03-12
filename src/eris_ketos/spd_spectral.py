# Copyright (c) 2026 Andrew H. Bond
# Department of Computer Engineering, San Jose State University
# Licensed under the MIT License.

"""
SPD manifold analysis of spectral patterns in cetacean clicks.

Frequency-band covariance matrices are Symmetric Positive Definite (SPD).
These matrices live on a Riemannian manifold where the log-Euclidean metric
is more discriminative than the standard Frobenius norm. This captures
correlations *between* frequency bands — e.g., harmonics that move together —
which flat spectrograms don't encode.

Applied to sperm whale codas, SPD analysis can capture the "vowel-like"
spectral patterns discovered by Begus et al. (Open Mind, 2025) and track
how these patterns evolve across clicks within a coda ("diphthongs").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# =============================================================================
# SPD MANIFOLD OPERATIONS
# =============================================================================


class SPDManifold:
    """Operations on the manifold of Symmetric Positive Definite matrices.

    The SPD manifold SPD(n) = {S ∈ R^{n×n} : S = Sᵀ, S ≻ 0} has a natural
    Riemannian structure. We use the log-Euclidean metric:

        d_LE(S₁, S₂) = ||log(S₁) - log(S₂)||_F

    which is a proper metric, computationally efficient, and more
    discriminative than Euclidean/Frobenius distance.
    """

    @staticmethod
    def log_map(S: torch.Tensor) -> torch.Tensor:
        """Log-Euclidean map: SPD matrix -> symmetric matrix (tangent space).

        Computes log(S) via eigendecomposition: log(S) = U·diag(log(λ))·Uᵀ

        Args:
            S: SPD matrix or batch, shape [..., n, n].

        Returns:
            Matrix logarithm, shape [..., n, n].
        """
        eigvals, eigvecs = torch.linalg.eigh(S)
        eigvals = eigvals.clamp_min(1e-10)
        result: torch.Tensor = eigvecs @ torch.diag_embed(eigvals.log()) @ eigvecs.transpose(-2, -1)
        return result

    @staticmethod
    def exp_map(X: torch.Tensor) -> torch.Tensor:
        """Exp map: symmetric matrix (tangent space) -> SPD matrix.

        Args:
            X: Symmetric matrix or batch, shape [..., n, n].

        Returns:
            SPD matrix, shape [..., n, n].
        """
        eigvals, eigvecs = torch.linalg.eigh(X)
        result: torch.Tensor = eigvecs @ torch.diag_embed(eigvals.exp()) @ eigvecs.transpose(-2, -1)
        return result

    @staticmethod
    def distance(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
        """Log-Euclidean distance between SPD matrices.

        d(S1, S2) = ||log(S1) - log(S2)||_F

        Args:
            S1, S2: SPD matrices, shape [..., n, n].

        Returns:
            Distance scalar or batch, shape [...].
        """
        log_diff = SPDManifold.log_map(S1) - SPDManifold.log_map(S2)
        result: torch.Tensor = torch.norm(log_diff.flatten(-2), dim=-1)
        return result

    @staticmethod
    def frechet_mean(
        matrices: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Log-Euclidean Fréchet mean of SPD matrices.

        In the log-Euclidean framework, the mean is simply:
            mean = exp(weighted_mean(log(S_i)))

        Args:
            matrices: Batch of SPD matrices, shape [n, d, d].
            weights: Optional weights, shape [n]. Defaults to uniform.

        Returns:
            Mean SPD matrix, shape [d, d].
        """
        logs = SPDManifold.log_map(matrices)
        if weights is not None:
            w = weights / weights.sum()
            mean_log = (logs * w.view(-1, 1, 1)).sum(dim=0)
        else:
            mean_log = logs.mean(dim=0)
        return SPDManifold.exp_map(mean_log)


# =============================================================================
# COVARIANCE COMPUTATION
# =============================================================================


def compute_covariance(
    spectrogram: np.ndarray,
    n_bands: int = 16,
    regularize: float = 1e-4,
) -> np.ndarray:
    """Compute frequency-band covariance matrix from a spectrogram.

    Groups mel bins into n_bands equal-sized bands, then computes the
    covariance across time for each band pair.

    Args:
        spectrogram: Mel spectrogram, shape [n_mels, n_frames].
        n_bands: Number of frequency band groups.
        regularize: Regularization for positive definiteness guarantee.

    Returns:
        SPD covariance matrix, shape [n_bands, n_bands].
    """
    n_mels, n_frames = spectrogram.shape
    band_size = n_mels // n_bands
    usable = n_bands * band_size

    # Group mel bins into bands by averaging
    bands = spectrogram[:usable, :].reshape(n_bands, band_size, n_frames).mean(axis=1)

    # Center and compute covariance
    centered = bands - bands.mean(axis=1, keepdims=True)
    cov = centered @ centered.T / max(n_frames - 1, 1)

    # Regularize for PD guarantee
    cov += regularize * np.eye(n_bands)

    result: np.ndarray = cov
    return result


def spd_features_from_spectrogram(
    spectrogram: np.ndarray,
    n_bands: int = 16,
    regularize: float = 1e-4,
) -> np.ndarray:
    """Extract SPD manifold features from a spectrogram.

    Computes the covariance matrix, applies the log-Euclidean map, and
    extracts the upper triangle as a feature vector.

    Args:
        spectrogram: Mel spectrogram, shape [n_mels, n_frames].
        n_bands: Number of frequency band groups.
        regularize: Regularization for positive definiteness.

    Returns:
        Feature vector from upper triangle of log(cov), shape [n_bands*(n_bands+1)/2].
    """
    cov = compute_covariance(spectrogram, n_bands=n_bands, regularize=regularize)

    # Log-Euclidean map via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-10)
    log_cov = eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T

    # Upper triangle
    idx = np.triu_indices(n_bands)
    result: np.ndarray = log_cov[idx].astype(np.float32)
    return result


# =============================================================================
# SPECTRAL TRAJECTORY (DIPHTHONG ANALYSIS)
# =============================================================================


@dataclass
class SpectralTrajectory:
    """Trajectory of SPD covariance matrices across time windows.

    Tracks how spectral covariance evolves within a coda. If the trajectory
    traces a geodesic on the SPD manifold, this suggests a "diphthong-like"
    smooth transition between vowel states.

    Attributes:
        matrices: Sequence of SPD covariance matrices, shape [n_windows, n_bands, n_bands].
        timestamps: Center time of each window in seconds.
        geodesic_deviation: How far the trajectory deviates from a geodesic (0 = perfect).
    """

    matrices: np.ndarray
    timestamps: np.ndarray
    geodesic_deviation: float = 0.0


def compute_spectral_trajectory(
    spectrogram: np.ndarray,
    n_bands: int = 16,
    window_frames: int = 32,
    hop_frames: int = 16,
    sr: int = 32000,
    hop_length: int = 512,
    regularize: float = 1e-4,
) -> SpectralTrajectory:
    """Compute time-varying SPD covariance trajectory.

    Slides a window across the spectrogram, computing a covariance matrix
    at each position. The resulting trajectory on the SPD manifold captures
    how spectral structure evolves over time.

    Args:
        spectrogram: Mel spectrogram, shape [n_mels, n_frames].
        n_bands: Frequency band groups.
        window_frames: Window size in spectrogram frames.
        hop_frames: Hop size in frames.
        sr: Sample rate (for timestamp computation).
        hop_length: STFT hop length (for timestamp computation).
        regularize: SPD regularization.

    Returns:
        SpectralTrajectory with per-window covariance matrices.
    """
    n_mels, n_frames = spectrogram.shape
    matrices = []
    timestamps = []

    for start in range(0, n_frames - window_frames + 1, hop_frames):
        window = spectrogram[:, start : start + window_frames]
        cov = compute_covariance(window, n_bands=n_bands, regularize=regularize)
        matrices.append(cov)
        center_frame = start + window_frames // 2
        timestamps.append(center_frame * hop_length / sr)

    if len(matrices) < 2:
        return SpectralTrajectory(
            matrices=np.array(matrices),
            timestamps=np.array(timestamps),
            geodesic_deviation=0.0,
        )

    mat_array = np.array(matrices)
    ts_array = np.array(timestamps)

    # Compute geodesic deviation: compare actual trajectory to geodesic
    # between first and last points
    mat_torch = torch.tensor(mat_array, dtype=torch.float32)
    total_dist = 0.0
    geodesic_dist = float(SPDManifold.distance(mat_torch[0], mat_torch[-1]))
    for i in range(len(mat_torch) - 1):
        total_dist += float(SPDManifold.distance(mat_torch[i], mat_torch[i + 1]))

    deviation = (total_dist - geodesic_dist) / max(geodesic_dist, 1e-10)

    return SpectralTrajectory(
        matrices=mat_array,
        timestamps=ts_array,
        geodesic_deviation=deviation,
    )
