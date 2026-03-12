# Copyright (c) 2026 Andrew H. Bond
# Department of Computer Engineering, San Jose State University
# Licensed under the MIT License.

"""
Parametric acoustic transforms for robustness testing.

Each transform has a controllable intensity parameter (0.0 to 1.0) that
scales the perturbation strength. This enables:
    - Intensity sweeps to find activation thresholds
    - Adversarial threshold search (binary search for decision-flip intensity)
    - Compositional chains (compound distortions)

Adapted from the ErisML Bond Index fuzzing framework, where parametric
intensity transforms test ethical evaluator robustness. Here they test
cetacean communication decoder robustness.

Design principle: at intensity=0.0 the signal is unchanged; at intensity=1.0
the perturbation is at maximum realistic strength.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# =============================================================================
# TRANSFORM BASE
# =============================================================================


@dataclass
class AcousticTransform:
    """Parametric acoustic transform with controllable intensity.

    Analogous to ErisML's ParametricTransform, but operates on audio signals
    instead of ethical scenarios.

    Attributes:
        name: Transform identifier.
        fn: Transform function (signal, sr, intensity) -> signal.
        intensity_range: Valid intensity range (usually [0.0, 1.0]).
        is_invariant: If True, a correct decoder should be invariant to this
                      transform (e.g., amplitude scaling). If False, it's a
                      stress test that may legitimately change output.
    """

    name: str
    fn: Callable[[np.ndarray, int, float], np.ndarray]
    intensity_range: tuple[float, float] = (0.0, 1.0)
    is_invariant: bool = True

    def __call__(
        self,
        signal: np.ndarray,
        sr: int = 32000,
        intensity: float = 1.0,
    ) -> np.ndarray:
        clamped = max(self.intensity_range[0], min(self.intensity_range[1], intensity))
        return self.fn(signal, sr, clamped)

    def at_intensity(self, intensity: float) -> Callable[[np.ndarray, int], np.ndarray]:
        """Return a fixed-intensity version of this transform."""
        return lambda signal, sr: self(signal, sr, intensity)


# =============================================================================
# TRANSFORM CHAIN
# =============================================================================


class TransformChain:
    """Compose multiple transforms into a chain for compound distortion testing.

    Analogous to ErisML's TransformChain for compositional fuzzing.
    Tests whether decoder degrades gracefully or catastrophically under
    compound perturbations (e.g., noise + echo + Doppler simultaneously).

    Attributes:
        transforms: List of (transform, intensity) pairs.
        name: Human-readable chain description.
    """

    def __init__(self, transforms: list[tuple[AcousticTransform, float]]) -> None:
        self.transforms = transforms
        self.name = " -> ".join(f"{t.name}@{i:.1f}" for t, i in transforms)

    def __call__(self, signal: np.ndarray, sr: int = 32000) -> np.ndarray:
        result: np.ndarray = signal.copy()
        for transform, intensity in self.transforms:
            result = transform(result, sr, intensity)
        return result

    @staticmethod
    def generate_chains(
        transforms: list[AcousticTransform],
        max_length: int = 3,
        intensities: list[float] | None = None,
        n_chains: int = 50,
        seed: int = 42,
    ) -> list[TransformChain]:
        """Generate diverse transform chains for compound testing.

        Args:
            transforms: Available transforms.
            max_length: Maximum chain length.
            intensities: Intensity values to sample from.
            n_chains: Number of chains to generate.
            seed: Random seed for reproducibility.

        Returns:
            List of TransformChain instances.
        """
        if intensities is None:
            intensities = [0.3, 0.6, 1.0]

        rng = random.Random(seed)
        chains = []

        for _ in range(n_chains):
            length = rng.randint(1, max_length)
            selected = rng.sample(transforms, min(length, len(transforms)))
            chain_spec = [(t, rng.choice(intensities)) for t in selected]
            chains.append(TransformChain(chain_spec))

        return chains


# =============================================================================
# ACOUSTIC TRANSFORMS
# =============================================================================


def _additive_noise(signal: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Add Gaussian white noise at intensity-controlled SNR.

    intensity=0: no noise. intensity=1: SNR ~ 0 dB.
    """
    if intensity <= 0:
        return signal
    rms = np.sqrt(np.mean(signal**2)) + 1e-10
    # SNR ranges from ~40 dB (intensity=0) to ~0 dB (intensity=1)
    snr_db = 40 * (1 - intensity)
    noise_rms = rms / (10 ** (snr_db / 20))
    rng = np.random.RandomState(hash(("noise", signal.shape[0], intensity)) % 2**31)
    noise = rng.randn(len(signal)).astype(signal.dtype) * noise_rms
    result: np.ndarray = signal + noise
    return result


def _pink_noise(signal: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Add 1/f pink noise (more realistic than white for ocean environments)."""
    if intensity <= 0:
        return signal
    rms = np.sqrt(np.mean(signal**2)) + 1e-10
    snr_db = 40 * (1 - intensity)
    noise_rms = rms / (10 ** (snr_db / 20))

    rng = np.random.RandomState(hash(("pink", signal.shape[0], intensity)) % 2**31)
    white = rng.randn(len(signal))
    # Simple 1/f approximation via cumulative sum + highpass
    pink = np.cumsum(white)
    pink = pink - np.convolve(pink, np.ones(64) / 64, mode="same")
    pink_rms = np.sqrt(np.mean(pink**2)) + 1e-10
    pink = (pink / pink_rms * noise_rms).astype(signal.dtype)
    result: np.ndarray = signal + pink
    return result


def _doppler_shift(signal: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Simulate Doppler shift from relative whale/recorder motion.

    intensity=0: no shift. intensity=1: ~5% frequency shift.
    """
    if intensity <= 0:
        return signal
    # Shift factor: 1 ± intensity * 0.05
    rng = np.random.RandomState(hash(("doppler", signal.shape[0], intensity)) % 2**31)
    direction = rng.choice([-1, 1])
    factor = 1.0 + direction * intensity * 0.05

    # Resample
    n_out = int(len(signal) / factor)
    if n_out <= 0:
        return signal
    indices = np.linspace(0, len(signal) - 1, n_out)
    resampled = np.interp(indices, np.arange(len(signal)), signal)

    # Pad or trim to original length
    if len(resampled) < len(signal):
        resampled = np.pad(resampled, (0, len(signal) - len(resampled)))
    else:
        resampled = resampled[: len(signal)]

    result: np.ndarray = resampled.astype(signal.dtype)
    return result


def _multipath_echo(signal: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Simulate underwater multipath propagation (delayed attenuated echoes).

    intensity=0: no echo. intensity=1: strong echo at ~50ms delay.
    """
    if intensity <= 0:
        return signal
    delay_samples = int(0.05 * sr * intensity)  # up to 50ms
    attenuation = 0.3 * intensity

    if delay_samples <= 0 or delay_samples >= len(signal):
        return signal

    echo = np.zeros_like(signal)
    echo[delay_samples:] = signal[:-delay_samples] * attenuation
    result: np.ndarray = signal + echo
    return result


def _amplitude_scale(signal: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Scale amplitude (a correct decoder should be invariant to this).

    intensity=0: no change. intensity=1: scale by 0.1x to 10x (log-uniform).
    """
    if intensity <= 0:
        return signal
    rng = np.random.RandomState(hash(("amp", signal.shape[0], intensity)) % 2**31)
    log_scale = rng.uniform(-intensity * 1.0, intensity * 1.0)
    factor = 10**log_scale
    result: np.ndarray = (signal * factor).astype(signal.dtype)
    return result


def _time_stretch(signal: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Stretch or compress time (changes duration and timing).

    intensity=0: no change. intensity=1: up to ±20% speed change.
    """
    if intensity <= 0:
        return signal
    rng = np.random.RandomState(hash(("stretch", signal.shape[0], intensity)) % 2**31)
    factor = 1.0 + rng.uniform(-0.2, 0.2) * intensity
    n_out = int(len(signal) / factor)
    if n_out <= 0:
        return signal

    indices = np.linspace(0, len(signal) - 1, n_out)
    stretched = np.interp(indices, np.arange(len(signal)), signal)

    if len(stretched) < len(signal):
        stretched = np.pad(stretched, (0, len(signal) - len(stretched)))
    else:
        stretched = stretched[: len(signal)]

    result: np.ndarray = stretched.astype(signal.dtype)
    return result


def _spectral_mask(signal: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Zero out a frequency band (simulates recorder bandwidth limitations).

    intensity=0: no masking. intensity=1: mask up to 40% of spectrum.
    """
    if intensity <= 0:
        return signal
    spectrum = np.fft.rfft(signal)
    n_freq = len(spectrum)

    rng = np.random.RandomState(hash(("mask", signal.shape[0], intensity)) % 2**31)
    mask_width = int(n_freq * 0.4 * intensity)
    if mask_width <= 0:
        return signal
    start = rng.randint(0, max(1, n_freq - mask_width))
    spectrum[start : start + mask_width] = 0

    result: np.ndarray = np.fft.irfft(spectrum, n=len(signal)).astype(signal.dtype)
    return result


def _click_dropout(signal: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Zero out random segments (simulates missed click detections).

    intensity=0: no dropout. intensity=1: drop up to 30% of signal.
    """
    if intensity <= 0:
        return signal
    result: np.ndarray = signal.copy()
    rng = np.random.RandomState(hash(("dropout", signal.shape[0], intensity)) % 2**31)
    n_drops = max(1, int(10 * intensity))
    drop_len = int(len(signal) * 0.03 * intensity)

    for _ in range(n_drops):
        start = rng.randint(0, max(1, len(signal) - drop_len))
        result[start : start + drop_len] = 0

    return result


def _time_shift(signal: np.ndarray, sr: int, intensity: float) -> np.ndarray:
    """Circular shift in time (decoder should be shift-invariant for codas).

    intensity=0: no shift. intensity=1: shift up to 25% of signal length.
    """
    if intensity <= 0:
        return signal
    rng = np.random.RandomState(hash(("shift", signal.shape[0], intensity)) % 2**31)
    max_shift = int(len(signal) * 0.25 * intensity)
    shift = rng.randint(-max_shift, max_shift + 1) if max_shift > 0 else 0
    result: np.ndarray = np.roll(signal, shift).astype(signal.dtype)
    return result


# =============================================================================
# TRANSFORM SUITE
# =============================================================================


def make_acoustic_transform_suite() -> list[AcousticTransform]:
    """Create the full suite of acoustic transforms for robustness testing.

    Returns transforms categorized by invariance expectation:
        Invariant (is_invariant=True): a correct decoder should handle these
        Stress (is_invariant=False): may legitimately change decoder output

    Returns:
        List of AcousticTransform instances.
    """
    return [
        # --- Invariant transforms (decoder SHOULD handle) ---
        AcousticTransform("amplitude_scale", _amplitude_scale, is_invariant=True),
        AcousticTransform("time_shift", _time_shift, is_invariant=True),
        AcousticTransform("additive_noise", _additive_noise, is_invariant=True),
        AcousticTransform("pink_noise", _pink_noise, is_invariant=True),
        # --- Stress transforms (may legitimately affect output) ---
        AcousticTransform("doppler_shift", _doppler_shift, is_invariant=False),
        AcousticTransform("multipath_echo", _multipath_echo, is_invariant=False),
        AcousticTransform("time_stretch", _time_stretch, is_invariant=False),
        AcousticTransform("spectral_mask", _spectral_mask, is_invariant=False),
        AcousticTransform("click_dropout", _click_dropout, is_invariant=False),
    ]
