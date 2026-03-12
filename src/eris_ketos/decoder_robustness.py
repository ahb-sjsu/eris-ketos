# Copyright (c) 2026 Andrew H. Bond
# Department of Computer Engineering, San Jose State University
# Licensed under the MIT License.

"""
Decoder Robustness Index (DRI) — adversarial fuzzing for cetacean decoders.

Adapted from the ErisML Bond Index calibration framework. The Bond Index
measures how robust an ethical evaluator is to surface-level perturbations.
The DRI applies the same methodology to cetacean communication decoders:

    Bond Index (ethics)              DRI (bioacoustics)
    ─────────────────                ──────────────────
    Scenario transforms      →      Acoustic transforms
    Option semantic distance  →      Coda semantic distance
    Evaluator selection      →      Decoder classification
    Graduated omega          →      Graduated omega (same math)
    Adversarial threshold    →      Adversarial threshold (same algorithm)
    Compositional chains     →      Compositional chains (same structure)

Key innovation: graduated omega uses semantic distance in the coda
combinatorial space. Misclassifying rhythm (fundamental structure) is
penalized more than misclassifying ornamentation (fine detail).

Usage:
    transforms = make_acoustic_transform_suite()
    dri = DecoderRobustnessIndex(transforms)
    result = dri.measure(decoder, signals, labels, sr=32000)
    print(f"DRI: {result.dri:.4f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .acoustic_transforms import AcousticTransform, TransformChain

# =============================================================================
# DECODER PROTOCOL
# =============================================================================


class CodaDecoder(Protocol):
    """Protocol for any cetacean coda decoder to be tested.

    Implement this interface to plug any decoder into the DRI framework.
    """

    def classify(self, signal: np.ndarray, sr: int) -> str:
        """Classify a coda signal into a coda type label.

        Args:
            signal: 1D audio signal containing a coda.
            sr: Sample rate.

        Returns:
            Predicted coda type label (e.g., "1+1+3", "5R1").
        """
        ...


# =============================================================================
# SEMANTIC DISTANCE
# =============================================================================


# Coda feature hierarchy: rhythm is most fundamental, ornamentation is finest
CODA_FEATURE_WEIGHTS = {
    "rhythm": 1.0,  # Most important — fundamental timing pattern
    "tempo": 0.7,  # Overall speed
    "rubato": 0.4,  # Subtle timing variation
    "ornamentation": 0.2,  # Extra clicks, finest detail
}


class CodaSemanticDistance:
    """Compute semantic distance between coda type predictions.

    Analogous to ErisML's SemanticDistanceCalculator, but uses the
    combinatorial structure of coda types (rhythm × tempo × rubato ×
    ornamentation) to weight errors by importance.

    If both predictions are parseable coda types, distance reflects which
    features differ. If not, falls back to binary match/mismatch.
    """

    def __init__(
        self,
        feature_weights: dict[str, float] | None = None,
        coda_features: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """
        Args:
            feature_weights: Importance weight per coda feature (rhythm, tempo, etc.).
            coda_features: Optional lookup mapping coda type labels to their
                          feature decomposition (e.g., {"1+1+3": {"rhythm": "A", ...}}).
        """
        self.weights = feature_weights or CODA_FEATURE_WEIGHTS
        self.coda_features = coda_features or {}

    def distance(self, pred1: str, pred2: str) -> float:
        """Compute semantic distance between two coda type predictions.

        Returns:
            0.0 if identical, up to 1.0 for maximally different.
            Any mismatch gets at least 0.5 penalty (hybrid approach from Bond Index).
        """
        if pred1.strip().lower() == pred2.strip().lower():
            return 0.0

        # Try feature-level comparison if we have decompositions
        feat1 = self.coda_features.get(pred1.strip())
        feat2 = self.coda_features.get(pred2.strip())

        if feat1 is not None and feat2 is not None:
            total_weight = sum(self.weights.values())
            mismatch_weight = 0.0
            for feature, weight in self.weights.items():
                if feat1.get(feature) != feat2.get(feature):
                    mismatch_weight += weight
            semantic_dist = mismatch_weight / total_weight
            # Hybrid: minimum 0.5 penalty for any flip
            return max(0.5, 0.5 + 0.5 * semantic_dist)

        # Fallback: binary mismatch
        return 0.75


# =============================================================================
# DRI MEASUREMENT
# =============================================================================


@dataclass
class TransformResult:
    """Result of measuring a single transform at a single intensity."""

    transform_name: str
    intensity: float
    omegas: list[float]
    mean_omega: float
    is_invariant: bool


@dataclass
class DRIResult:
    """Full Decoder Robustness Index measurement result.

    Attributes:
        dri: The scalar Decoder Robustness Index (lower = more robust).
        dri_invariant: DRI computed only on invariant transforms.
        dri_stress: DRI computed only on stress transforms.
        per_transform: Per-transform sensitivity profile.
        chain_results: Results from compositional chain testing.
        adversarial_thresholds: Per-transform minimal flip intensities.
    """

    dri: float
    dri_invariant: float
    dri_stress: float
    per_transform: dict[str, float]
    chain_results: list[tuple[str, float]] = field(default_factory=list)
    adversarial_thresholds: dict[str, float] = field(default_factory=dict)


class DecoderRobustnessIndex:
    """Measures decoder robustness via adversarial acoustic fuzzing.

    Analogous to ErisML's Bond Index calibration suite. Applies parametric
    acoustic transforms at varying intensities, measures how often and how
    badly the decoder's output changes, and produces a single scalar (DRI)
    plus detailed diagnostics.

    DRI formula (same weighting as Bond Index):
        DRI = 0.5 * mean(omega) + 0.3 * p75(omega) + 0.2 * p95(omega)

    Lower DRI = more robust decoder.

    Args:
        transforms: List of parametric acoustic transforms.
        semantic_distance: CodaSemanticDistance calculator for graduated omega.
    """

    def __init__(
        self,
        transforms: list[AcousticTransform],
        semantic_distance: CodaSemanticDistance | None = None,
    ) -> None:
        self.transforms = transforms
        self.semantic_distance = semantic_distance or CodaSemanticDistance()

    def _compute_omega(
        self,
        decoder: CodaDecoder,
        signal: np.ndarray,
        sr: int,
        transformed: np.ndarray,
    ) -> float:
        """Compute graduated omega between baseline and transformed predictions."""
        pred_baseline = decoder.classify(signal, sr)
        pred_transformed = decoder.classify(transformed, sr)
        return self.semantic_distance.distance(pred_baseline, pred_transformed)

    def measure_single_transform(
        self,
        decoder: CodaDecoder,
        signals: list[np.ndarray],
        sr: int,
        transform: AcousticTransform,
        intensity: float = 1.0,
    ) -> TransformResult:
        """Measure decoder robustness against a single transform at one intensity.

        Args:
            decoder: Decoder to test.
            signals: List of coda audio signals.
            sr: Sample rate.
            transform: Acoustic transform to apply.
            intensity: Transform intensity (0.0 to 1.0).

        Returns:
            TransformResult with per-signal omegas and mean.
        """
        omegas = []
        for signal in signals:
            transformed = transform(signal, sr, intensity)
            omega = self._compute_omega(decoder, signal, sr, transformed)
            omegas.append(omega)

        return TransformResult(
            transform_name=transform.name,
            intensity=intensity,
            omegas=omegas,
            mean_omega=float(np.mean(omegas)) if omegas else 0.0,
            is_invariant=transform.is_invariant,
        )

    def intensity_sweep(
        self,
        decoder: CodaDecoder,
        signals: list[np.ndarray],
        sr: int,
        transform: AcousticTransform,
        n_points: int = 10,
    ) -> list[tuple[float, float]]:
        """Sweep intensity from 0 to 1, recording mean omega at each level.

        Reveals the activation threshold where the decoder starts failing.

        Args:
            decoder: Decoder to test.
            signals: Coda signals.
            sr: Sample rate.
            transform: Transform to sweep.
            n_points: Number of intensity levels.

        Returns:
            List of (intensity, mean_omega) pairs.
        """
        curve = []
        for intensity in np.linspace(0, 1, n_points):
            result = self.measure_single_transform(
                decoder, signals, sr, transform, float(intensity)
            )
            curve.append((float(intensity), result.mean_omega))
        return curve

    def find_adversarial_threshold(
        self,
        decoder: CodaDecoder,
        signal: np.ndarray,
        sr: int,
        transform: AcousticTransform,
        tolerance: float = 0.01,
    ) -> float:
        """Binary search for minimal intensity that flips the decoder output.

        Identical algorithm to ErisML's find_adversarial_threshold().

        Args:
            decoder: Decoder to test.
            signal: Single coda signal.
            sr: Sample rate.
            transform: Transform to search.
            tolerance: Search precision.

        Returns:
            Minimal flip intensity (1.0 if no flip found at any intensity).
        """
        baseline = decoder.classify(signal, sr)

        def causes_flip(intensity: float) -> bool:
            transformed = transform(signal, sr, intensity)
            result = decoder.classify(transformed, sr)
            return result.strip().lower() != baseline.strip().lower()

        if not causes_flip(1.0):
            return 1.0

        low, high = 0.0, 1.0
        while high - low > tolerance:
            mid = (low + high) / 2
            if causes_flip(mid):
                high = mid
            else:
                low = mid
        return high

    def sensitivity_profile(
        self,
        decoder: CodaDecoder,
        signals: list[np.ndarray],
        sr: int,
    ) -> dict[str, float]:
        """Map which transforms cause the most decoder drift.

        Analogous to ErisML's sensitivity_profile().

        Returns:
            Dict mapping transform name to mean omega at full intensity.
        """
        profile = {}
        for transform in self.transforms:
            result = self.measure_single_transform(decoder, signals, sr, transform, intensity=1.0)
            profile[transform.name] = result.mean_omega
        return profile

    def measure(
        self,
        decoder: CodaDecoder,
        signals: list[np.ndarray],
        sr: int = 32000,
        intensities: list[float] | None = None,
        n_chains: int = 30,
        chain_max_length: int = 3,
    ) -> DRIResult:
        """Full DRI measurement: all transforms, chains, and adversarial search.

        Args:
            decoder: Decoder to test.
            signals: List of coda audio signals.
            sr: Sample rate.
            intensities: Intensity grid for sweep (default: [0.3, 0.6, 1.0]).
            n_chains: Number of compositional chains to test.
            chain_max_length: Maximum chain length.

        Returns:
            DRIResult with scalar DRI and full diagnostics.
        """
        if intensities is None:
            intensities = [0.3, 0.6, 1.0]

        all_omegas = []
        invariant_omegas = []
        stress_omegas = []
        per_transform: dict[str, float] = {}

        # Per-transform sweeps
        for transform in self.transforms:
            transform_omegas = []
            for intensity in intensities:
                result = self.measure_single_transform(decoder, signals, sr, transform, intensity)
                transform_omegas.extend(result.omegas)

            per_transform[transform.name] = float(np.mean(transform_omegas))
            all_omegas.extend(transform_omegas)

            if transform.is_invariant:
                invariant_omegas.extend(transform_omegas)
            else:
                stress_omegas.extend(transform_omegas)

        # Compositional chains
        chains = TransformChain.generate_chains(
            self.transforms,
            max_length=chain_max_length,
            intensities=intensities,
            n_chains=n_chains,
        )
        chain_results = []
        for chain in chains:
            chain_omegas = []
            for signal in signals:
                transformed = chain(signal, sr)
                omega = self._compute_omega(decoder, signal, sr, transformed)
                chain_omegas.append(omega)
                all_omegas.append(omega)
            chain_results.append((chain.name, float(np.mean(chain_omegas))))

        # Adversarial thresholds (on first signal for efficiency)
        adversarial = {}
        if signals:
            for transform in self.transforms:
                threshold = self.find_adversarial_threshold(decoder, signals[0], sr, transform)
                adversarial[transform.name] = threshold

        # Compute DRI (Bond Index formula)
        def _compute_dri(omegas: list[float]) -> float:
            if not omegas:
                return 0.0
            arr = np.array(omegas)
            return float(
                0.5 * np.mean(arr) + 0.3 * np.percentile(arr, 75) + 0.2 * np.percentile(arr, 95)
            )

        return DRIResult(
            dri=_compute_dri(all_omegas),
            dri_invariant=_compute_dri(invariant_omegas),
            dri_stress=_compute_dri(stress_omegas),
            per_transform=per_transform,
            chain_results=chain_results,
            adversarial_thresholds=adversarial,
        )
