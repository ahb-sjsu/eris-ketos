# Copyright (c) 2026 Andrew H. Bond
# Department of Computer Engineering, San Jose State University
# Licensed under the MIT License.

"""
eris-ketos: Geometric analysis of cetacean communication.

Hyperbolic embeddings, topological data analysis, and adversarial
robustness testing for whale vocalization decoding.
"""

__version__ = "0.1.0"

from .acoustic_transforms import AcousticTransform, make_acoustic_transform_suite
from .decoder_robustness import DecoderRobustnessIndex
from .poincare_coda import HyperbolicMLR, PoincareBall, embed_taxonomy_hyperbolic
from .spd_spectral import SPDManifold, spd_features_from_spectrogram
from .tda_clicks import compute_persistence, tda_feature_vector, time_delay_embedding

__all__ = [
    "PoincareBall",
    "HyperbolicMLR",
    "embed_taxonomy_hyperbolic",
    "time_delay_embedding",
    "compute_persistence",
    "tda_feature_vector",
    "SPDManifold",
    "spd_features_from_spectrogram",
    "AcousticTransform",
    "make_acoustic_transform_suite",
    "DecoderRobustnessIndex",
]
