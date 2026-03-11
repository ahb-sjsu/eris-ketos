# Copyright (c) 2026 Andrew H. Bond
# Department of Computer Engineering, San Jose State University
# Licensed under the MIT License.

"""
Topological data analysis of cetacean click patterns.

Uses Takens' time-delay embedding to reconstruct the attractor of the vocal
apparatus from 1D click timing sequences, then computes persistent homology
to extract topological invariants:

    - H0 (connected components): number of distinct click clusters
    - H1 (loops): cyclic patterns in timing, e.g., trills and repeated motifs

Different species and social units produce topologically distinct attractors,
enabling classification based on the *shape* of vocalization dynamics rather
than spectral features alone.

Requires: ripser (pip install eris-ketos[tda])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# =============================================================================
# TIME-DELAY EMBEDDING
# =============================================================================


def time_delay_embedding(
    signal: np.ndarray,
    delay: int = 10,
    dim: int = 3,
) -> np.ndarray:
    """Takens' time-delay embedding: reconstruct attractor from 1D series.

    Given a signal x(t), constructs vectors:
        v(t) = [x(t), x(t + τ), x(t + 2τ), ..., x(t + (d-1)τ)]

    By Takens' theorem, for generic τ and d ≥ 2m+1 (m = attractor dimension),
    this reconstructs the topology of the original dynamical system.

    Args:
        signal: 1D time series (audio samples or inter-click intervals).
        delay: Time delay τ in samples.
        dim: Embedding dimension d.

    Returns:
        Point cloud in R^dim, shape [n_points, dim].
    """
    n = len(signal) - (dim - 1) * delay
    if n <= 0:
        return np.zeros((1, dim))

    embedded = np.empty((n, dim))
    for d in range(dim):
        embedded[:, d] = signal[d * delay : d * delay + n]

    return embedded


def subsample_cloud(
    cloud: np.ndarray,
    max_points: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    """Subsample a point cloud for computational tractability.

    Args:
        cloud: Point cloud, shape [n, d].
        max_points: Maximum number of points to keep.
        seed: Random seed for reproducibility.

    Returns:
        Subsampled cloud, shape [min(n, max_points), d].
    """
    if len(cloud) <= max_points:
        return cloud
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(cloud), max_points, replace=False)
    return cloud[idx]


# =============================================================================
# PERSISTENCE COMPUTATION
# =============================================================================


@dataclass
class PersistenceResult:
    """Result of a persistent homology computation.

    Attributes:
        diagrams: List of persistence diagrams, one per homology dimension.
                  Each diagram is an array of shape [n_features, 2] with
                  columns (birth, death).
        cloud: The point cloud used for computation.
        max_dim: Maximum homology dimension computed.
    """

    diagrams: list[np.ndarray]
    cloud: np.ndarray
    max_dim: int


def compute_persistence(
    signal: np.ndarray,
    delay: int = 10,
    dim: int = 3,
    max_points: int = 1000,
    max_homology_dim: int = 1,
    thresh: float = 2.0,
    seed: int | None = None,
) -> PersistenceResult:
    """Compute persistent homology from an audio signal.

    Pipeline:
        1. Time-delay embed the signal into R^dim
        2. Subsample for tractability
        3. Normalize to zero mean, unit variance
        4. Compute Vietoris-Rips persistence via ripser

    Args:
        signal: 1D audio signal or inter-click interval sequence.
        delay: Time delay for embedding.
        dim: Embedding dimension.
        max_points: Maximum points after subsampling.
        max_homology_dim: Maximum homology dimension (0=components, 1=loops).
        thresh: Maximum filtration value for Rips complex.
        seed: Random seed for subsampling.

    Returns:
        PersistenceResult with diagrams and metadata.

    Raises:
        ImportError: If ripser is not installed.
    """
    try:
        from ripser import ripser
    except ImportError as e:
        raise ImportError(
            "ripser is required for TDA. Install with: pip install eris-ketos[tda]"
        ) from e

    cloud = time_delay_embedding(signal, delay=delay, dim=dim)
    cloud = subsample_cloud(cloud, max_points=max_points, seed=seed)

    # Normalize
    std = cloud.std(axis=0)
    std[std < 1e-10] = 1.0
    cloud = (cloud - cloud.mean(axis=0)) / std

    result = ripser(cloud, maxdim=max_homology_dim, thresh=thresh)

    return PersistenceResult(
        diagrams=result["dgms"],
        cloud=cloud,
        max_dim=max_homology_dim,
    )


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================


def _diagram_features(dgm: np.ndarray) -> np.ndarray:
    """Extract summary statistics from a single persistence diagram.

    Returns 8 features:
        0: count — number of finite features
        1: mean_lifetime
        2: std_lifetime
        3: max_lifetime — most persistent feature
        4: p75_lifetime — 75th percentile
        5: mean_birth
        6: total_persistence — sum of squared lifetimes (L2 norm)
        7: normalized_persistence — total / count

    Args:
        dgm: Persistence diagram, shape [n, 2] with (birth, death).

    Returns:
        Feature vector, shape [8].
    """
    if len(dgm) == 0:
        return np.zeros(8, dtype=np.float32)

    finite_mask = np.isfinite(dgm[:, 1])
    finite = dgm[finite_mask]

    if len(finite) == 0:
        return np.zeros(8, dtype=np.float32)

    lifetimes = finite[:, 1] - finite[:, 0]
    n = len(lifetimes)
    total = float(np.sum(lifetimes**2))

    return np.array(
        [
            n,
            lifetimes.mean(),
            lifetimes.std() if n > 1 else 0.0,
            lifetimes.max(),
            np.percentile(lifetimes, 75),
            finite[:, 0].mean(),
            total,
            np.sqrt(total) / (n + 1e-10),
        ],
        dtype=np.float32,
    )


def tda_feature_vector(
    result: PersistenceResult,
) -> np.ndarray:
    """Extract a fixed-length feature vector from persistence diagrams.

    Concatenates 8 summary statistics per homology dimension, yielding
    8 * (max_dim + 1) features total.

    Args:
        result: Output of compute_persistence().

    Returns:
        Feature vector, shape [8 * (max_dim + 1)].
    """
    features = []
    for d in range(result.max_dim + 1):
        if d < len(result.diagrams):
            features.append(_diagram_features(result.diagrams[d]))
        else:
            features.append(np.zeros(8, dtype=np.float32))

    return np.concatenate(features)
