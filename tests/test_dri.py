# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for Decoder Robustness Index (DRI) framework."""

import numpy as np
import pytest

from eris_ketos.acoustic_transforms import make_acoustic_transform_suite
from eris_ketos.decoder_robustness import (
    CodaSemanticDistance,
    DecoderRobustnessIndex,
    DRIResult,
)

# =============================================================================
# Mock decoders (analogous to ErisML's Evaluator test fixtures)
# =============================================================================


class PerfectDecoder:
    """Always returns the same label regardless of perturbation. DRI should be ~0."""

    def classify(self, signal: np.ndarray, sr: int) -> str:
        return "1+1+3"


class FragileDecoder:
    """Sensitive to even tiny signal changes. DRI should be high."""

    def classify(self, signal: np.ndarray, sr: int) -> str:
        # Hash first 100 samples — any perturbation changes this
        sig_hash = hash(signal[:100].tobytes()) % 4
        return ["5R1", "1+1+3", "2+3", "4R"][sig_hash]


class ModerateDecoder:
    """Somewhat robust — invariant to amplitude but sensitive to noise."""

    def classify(self, signal: np.ndarray, sr: int) -> str:
        # Normalize amplitude — invariant to scaling
        peak = np.abs(signal).max()
        if peak > 0:
            signal = signal / peak
        # But sensitive to spectral content
        spectrum = np.abs(np.fft.rfft(signal))
        peak_freq = np.argmax(spectrum[1:]) + 1
        if peak_freq < len(spectrum) // 4:
            return "1+1+3"
        elif peak_freq < len(spectrum) // 2:
            return "2+3"
        else:
            return "5R1"


# =============================================================================
# CodaSemanticDistance
# =============================================================================


class TestCodaSemanticDistance:
    def test_identical_is_zero(self) -> None:
        sd = CodaSemanticDistance()
        assert sd.distance("1+1+3", "1+1+3") == 0.0

    def test_case_insensitive(self) -> None:
        sd = CodaSemanticDistance()
        assert sd.distance("Type_A", "type_a") == 0.0

    def test_different_is_nonzero(self) -> None:
        sd = CodaSemanticDistance()
        d = sd.distance("1+1+3", "5R1")
        assert d > 0

    def test_minimum_penalty(self) -> None:
        """Any mismatch should get at least 0.5 penalty (hybrid approach)."""
        sd = CodaSemanticDistance()
        d = sd.distance("type_a", "type_b")
        assert d >= 0.5

    def test_feature_aware_distance(self) -> None:
        """When feature decomposition is available, rhythm errors > ornamentation."""
        features = {
            "coda_a": {"rhythm": "R1", "tempo": "fast", "rubato": "none", "ornamentation": "none"},
            "coda_b": {"rhythm": "R2", "tempo": "fast", "rubato": "none", "ornamentation": "none"},
            "coda_c": {"rhythm": "R1", "tempo": "fast", "rubato": "none", "ornamentation": "click"},
        }
        sd = CodaSemanticDistance(coda_features=features)
        d_rhythm = sd.distance("coda_a", "coda_b")  # rhythm differs
        d_ornament = sd.distance("coda_a", "coda_c")  # only ornamentation differs
        assert d_rhythm > d_ornament


# =============================================================================
# DecoderRobustnessIndex
# =============================================================================


@pytest.fixture
def signals() -> list[np.ndarray]:
    """Generate 5 simple test signals."""
    rng = np.random.RandomState(42)
    return [
        (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 32000))).astype(np.float32)
        + rng.randn(32000).astype(np.float32) * 0.01
        for _ in range(5)
    ]


class TestDRI:
    def test_perfect_decoder_low_dri(self, signals: list[np.ndarray]) -> None:
        transforms = make_acoustic_transform_suite()
        dri = DecoderRobustnessIndex(transforms)
        result = dri.measure(PerfectDecoder(), signals, sr=32000, n_chains=5)
        assert isinstance(result, DRIResult)
        assert result.dri == 0.0  # perfect decoder never flips

    def test_fragile_decoder_high_dri(self, signals: list[np.ndarray]) -> None:
        transforms = make_acoustic_transform_suite()
        dri = DecoderRobustnessIndex(transforms)
        result = dri.measure(FragileDecoder(), signals, sr=32000, n_chains=5)
        assert result.dri > 0.0  # fragile decoder should flip

    def test_dri_monotonicity(self, signals: list[np.ndarray]) -> None:
        """Perfect < Moderate < Fragile in DRI score."""
        transforms = make_acoustic_transform_suite()
        dri = DecoderRobustnessIndex(transforms)
        r_perfect = dri.measure(PerfectDecoder(), signals, sr=32000, n_chains=5)
        r_fragile = dri.measure(FragileDecoder(), signals, sr=32000, n_chains=5)
        assert r_perfect.dri <= r_fragile.dri

    def test_sensitivity_profile(self, signals: list[np.ndarray]) -> None:
        transforms = make_acoustic_transform_suite()
        dri = DecoderRobustnessIndex(transforms)
        profile = dri.sensitivity_profile(FragileDecoder(), signals, sr=32000)
        assert len(profile) == len(transforms)
        assert all(isinstance(v, float) for v in profile.values())

    def test_intensity_sweep(self, signals: list[np.ndarray]) -> None:
        transforms = make_acoustic_transform_suite()
        dri = DecoderRobustnessIndex(transforms)
        curve = dri.intensity_sweep(
            FragileDecoder(), signals, sr=32000, transform=transforms[0], n_points=5
        )
        assert len(curve) == 5
        # First point (intensity=0) should have omega=0
        assert curve[0][0] == 0.0
        assert curve[0][1] == 0.0  # no perturbation -> no change

    def test_adversarial_threshold(self, signals: list[np.ndarray]) -> None:
        transforms = make_acoustic_transform_suite()
        dri = DecoderRobustnessIndex(transforms)
        threshold = dri.find_adversarial_threshold(
            PerfectDecoder(), signals[0], sr=32000, transform=transforms[0]
        )
        assert threshold == 1.0  # perfect decoder never flips

    def test_result_has_all_fields(self, signals: list[np.ndarray]) -> None:
        transforms = make_acoustic_transform_suite()
        dri = DecoderRobustnessIndex(transforms)
        result = dri.measure(PerfectDecoder(), signals, sr=32000, n_chains=3)
        assert hasattr(result, "dri")
        assert hasattr(result, "dri_invariant")
        assert hasattr(result, "dri_stress")
        assert hasattr(result, "per_transform")
        assert hasattr(result, "chain_results")
        assert hasattr(result, "adversarial_thresholds")
