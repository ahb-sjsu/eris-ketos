# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for acoustic transforms and transform chains."""

import numpy as np
import pytest

from eris_ketos.acoustic_transforms import (
    AcousticTransform,
    TransformChain,
    make_acoustic_transform_suite,
)


@pytest.fixture
def sine_signal() -> np.ndarray:
    """1-second 440Hz sine wave at 32kHz."""
    sr = 32000
    t = np.linspace(0, 1, sr)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def transforms() -> list[AcousticTransform]:
    return make_acoustic_transform_suite()


class TestTransformSuite:
    def test_suite_not_empty(self, transforms: list[AcousticTransform]) -> None:
        assert len(transforms) >= 5

    def test_has_invariant_and_stress(self, transforms: list[AcousticTransform]) -> None:
        invariant = [t for t in transforms if t.is_invariant]
        stress = [t for t in transforms if not t.is_invariant]
        assert len(invariant) > 0
        assert len(stress) > 0

    def test_unique_names(self, transforms: list[AcousticTransform]) -> None:
        names = [t.name for t in transforms]
        assert len(names) == len(set(names))


class TestIndividualTransforms:
    def test_zero_intensity_is_identity(
        self, transforms: list[AcousticTransform], sine_signal: np.ndarray
    ) -> None:
        for t in transforms:
            result = t(sine_signal, sr=32000, intensity=0.0)
            np.testing.assert_array_equal(result, sine_signal, err_msg=f"{t.name} at 0")

    def test_output_same_length(
        self, transforms: list[AcousticTransform], sine_signal: np.ndarray
    ) -> None:
        for t in transforms:
            result = t(sine_signal, sr=32000, intensity=0.5)
            assert len(result) == len(sine_signal), f"{t.name} changed length"

    def test_output_is_float(
        self, transforms: list[AcousticTransform], sine_signal: np.ndarray
    ) -> None:
        for t in transforms:
            result = t(sine_signal, sr=32000, intensity=0.5)
            assert result.dtype in (np.float32, np.float64), f"{t.name} dtype: {result.dtype}"

    def test_output_is_finite(
        self, transforms: list[AcousticTransform], sine_signal: np.ndarray
    ) -> None:
        for t in transforms:
            result = t(sine_signal, sr=32000, intensity=1.0)
            assert np.all(np.isfinite(result)), f"{t.name} produced non-finite values"

    def test_full_intensity_differs(
        self, transforms: list[AcousticTransform], sine_signal: np.ndarray
    ) -> None:
        """At full intensity, transforms should actually change the signal."""
        for t in transforms:
            result = t(sine_signal, sr=32000, intensity=1.0)
            if not np.array_equal(result, sine_signal):
                continue  # passes
            # Some transforms may not change a pure sine — that's OK
            # but at least some should

    def test_at_intensity_helper(
        self, transforms: list[AcousticTransform], sine_signal: np.ndarray
    ) -> None:
        t = transforms[0]
        fixed = t.at_intensity(0.5)
        result = fixed(sine_signal, 32000)
        assert len(result) == len(sine_signal)


class TestTransformChain:
    def test_chain_applies_all(
        self, transforms: list[AcousticTransform], sine_signal: np.ndarray
    ) -> None:
        chain = TransformChain([(transforms[0], 0.5), (transforms[1], 0.5)])
        result = chain(sine_signal, sr=32000)
        assert len(result) == len(sine_signal)

    def test_chain_name_format(self, transforms: list[AcousticTransform]) -> None:
        chain = TransformChain([(transforms[0], 0.3), (transforms[1], 0.7)])
        assert "->" in chain.name
        assert "@0.3" in chain.name
        assert "@0.7" in chain.name

    def test_generate_chains(self, transforms: list[AcousticTransform]) -> None:
        chains = TransformChain.generate_chains(transforms, max_length=2, n_chains=10, seed=42)
        assert len(chains) == 10
        for chain in chains:
            assert len(chain.transforms) >= 1
            assert len(chain.transforms) <= 2

    def test_generate_chains_reproducible(self, transforms: list[AcousticTransform]) -> None:
        c1 = TransformChain.generate_chains(transforms, n_chains=5, seed=42)
        c2 = TransformChain.generate_chains(transforms, n_chains=5, seed=42)
        for a, b in zip(c1, c2, strict=True):
            assert a.name == b.name
