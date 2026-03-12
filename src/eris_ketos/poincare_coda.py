# Copyright (c) 2026 Andrew H. Bond
# Department of Computer Engineering, San Jose State University
# Licensed under the MIT License.

"""
Poincaré ball embeddings for hierarchical cetacean coda structure.

The Poincaré ball B^d_c = {x ∈ R^d : c||x||² < 1} is a model of hyperbolic
space where distance grows exponentially toward the boundary. Trees embed
with O(log n) distortion vs O(n) in Euclidean space, making it ideal for
taxonomic and combinatorial hierarchies in whale communication.

Key components:
    PoincareBall    — Core Möbius gyrovector operations
    HyperbolicMLR   — Classification via distance to prototypes on the ball
    embed_taxonomy_hyperbolic — Embed a taxonomic distance matrix into the ball
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# =============================================================================
# POINCARÉ BALL
# =============================================================================


class PoincareBall:
    """Poincaré ball model of hyperbolic space.

    All operations use the Möbius gyrovector formalism with curvature c.
    Points must satisfy c * ||x||² < 1.

    Args:
        c: Negative curvature parameter. Higher c = more hyperbolic spread.
    """

    def __init__(self, c: float = 1.0) -> None:
        if c <= 0:
            raise ValueError(f"Curvature c must be positive, got {c}")
        self.c = c
        self.sqrt_c = c**0.5

    def project(self, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Project points onto the open ball (clamp norm < 1/sqrt(c) - eps)."""
        max_norm = (1.0 / self.sqrt_c) - eps
        norm = x.norm(dim=-1, keepdim=True)
        return torch.where(norm > max_norm, x / norm * max_norm, x)

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map from the origin: tangent vector -> ball point.

        exp_0(v) = tanh(√c ||v||) · v / (√c ||v||)
        """
        v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        return self.project(torch.tanh(self.sqrt_c * v_norm) * v / (self.sqrt_c * v_norm))

    def logmap0(self, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map to the origin: ball point -> tangent vector.

        log_0(y) = arctanh(√c ||y||) · y / (√c ||y||)
        """
        y_norm = y.norm(dim=-1, keepdim=True).clamp(1e-15, (1 / self.sqrt_c) - 1e-5)
        result: torch.Tensor = torch.arctanh(self.sqrt_c * y_norm) * y / (self.sqrt_c * y_norm)
        return result

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Möbius addition: the hyperbolic analogue of vector addition.

        x ⊕ y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y)
                 / (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
        """
        x2 = (x * x).sum(-1, keepdim=True)
        y2 = (y * y).sum(-1, keepdim=True)
        xy = (x * y).sum(-1, keepdim=True)
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c**2 * x2 * y2
        return self.project(num / denom.clamp_min(1e-15))

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Geodesic distance on the Poincaré ball.

        d(x, y) = (2/√c) · arctanh(√c · ||(-x) ⊕ y||)
        """
        neg_x = -x
        add_result = self.mobius_add(neg_x, y)
        norm = add_result.norm(dim=-1).clamp(max=(1 / self.sqrt_c) - 1e-5)
        result: torch.Tensor = (2.0 / self.sqrt_c) * torch.arctanh(self.sqrt_c * norm)
        return result

    def dist_to_origin(self, x: torch.Tensor) -> torch.Tensor:
        """Distance from x to the origin (cheaper than general dist)."""
        x_norm = x.norm(dim=-1).clamp(1e-15, (1 / self.sqrt_c) - 1e-5)
        result: torch.Tensor = (2.0 / self.sqrt_c) * torch.arctanh(self.sqrt_c * x_norm)
        return result

    def midpoint(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Einstein midpoint of two points on the ball."""
        # Conformal factors
        gamma_x = 1.0 / (1.0 - self.c * (x * x).sum(-1, keepdim=True)).clamp_min(1e-15)
        gamma_y = 1.0 / (1.0 - self.c * (y * y).sum(-1, keepdim=True)).clamp_min(1e-15)
        num = gamma_x * x + gamma_y * y
        denom = gamma_x + gamma_y
        mid_euclidean = num / denom.clamp_min(1e-15)
        return self.project(mid_euclidean)


# =============================================================================
# HYPERBOLIC MULTINOMIAL LOGISTIC REGRESSION
# =============================================================================


class HyperbolicMLR(nn.Module):
    """Hyperbolic Multinomial Logistic Regression.

    Classification via geodesic distance to learned prototypes on the Poincaré
    ball. Prototypes can be initialized from a taxonomic tree so that species
    in the same genus start close and different orders start far apart.

    Args:
        embed_dim: Dimensionality of embeddings.
        num_classes: Number of output classes.
        c: Poincaré ball curvature.
    """

    def __init__(self, embed_dim: int, num_classes: int, c: float = 1.0) -> None:
        super().__init__()
        self.ball = PoincareBall(c)
        # Prototypes live in tangent space at origin; mapped to ball via expmap0
        self.proto_tangent = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.01)
        # Per-class learnable temperature (helps with rare species)
        self.log_scale = nn.Parameter(torch.zeros(num_classes))

    @property
    def prototypes(self) -> torch.Tensor:
        """Map tangent vectors to Poincaré ball points."""
        return self.ball.expmap0(self.proto_tangent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits as negative scaled hyperbolic distances.

        Args:
            x: Points on the Poincaré ball, shape [batch, embed_dim].

        Returns:
            Logits, shape [batch, num_classes].
        """
        protos = self.prototypes  # [C, D]
        dists = self.ball.dist(x.unsqueeze(1), protos.unsqueeze(0))  # [B, C]
        return -dists * self.log_scale.exp()

    def init_from_taxonomy(self, embeddings: torch.Tensor) -> None:
        """Initialize prototypes from pre-computed taxonomic embeddings.

        Args:
            embeddings: Taxonomy embeddings, shape [num_classes, embed_dim].
                        Will be scaled to fit inside the ball.
        """
        with torch.no_grad():
            on_ball = self.ball.project(embeddings * 0.5)
            tangent = self.ball.logmap0(on_ball)
            self.proto_tangent.copy_(tangent)

    @property
    def num_classes(self) -> int:
        return self.proto_tangent.shape[0]

    @property
    def embed_dim(self) -> int:
        return self.proto_tangent.shape[1]


# =============================================================================
# TAXONOMY EMBEDDING
# =============================================================================


def build_distance_matrix(
    species_info: dict[str, dict[str, str]],
    levels: tuple[str, ...] = ("family", "order"),
) -> np.ndarray:
    """Build a taxonomic distance matrix from species info.

    Distance encoding:
        0 = same species
        1 = same finest-level group (e.g., family)
        2 = same next-level group (e.g., order)
        ...
        len(levels) + 1 = different at all levels

    Args:
        species_info: Dict mapping species name to dict with taxonomic level keys.
        levels: Taxonomic levels from finest to coarsest.

    Returns:
        Square distance matrix, shape [n_species, n_species].
    """
    names = list(species_info.keys())
    n = len(names)
    dist = np.zeros((n, n))

    for i, si in enumerate(names):
        for j, sj in enumerate(names):
            if i == j:
                continue
            d = len(levels) + 1  # max distance
            for k, level in enumerate(levels):
                if species_info[si].get(level) == species_info[sj].get(level):
                    d = k + 1
                    break
            dist[i, j] = d

    return dist


def embed_taxonomy_hyperbolic(
    dist_matrix: np.ndarray,
    embed_dim: int,
    c: float = 1.0,
    scale: float = 0.7,
) -> torch.Tensor:
    """Embed a taxonomic distance matrix into the Poincaré ball.

    Uses spectral decomposition of a Gaussian kernel over the distance matrix.
    Species that are taxonomically close end up near each other on the ball.

    Args:
        dist_matrix: Square symmetric distance matrix, shape [n, n].
        embed_dim: Target embedding dimensionality.
        c: Poincaré ball curvature.
        scale: Maximum norm relative to ball radius (0 < scale < 1).

    Returns:
        Embeddings on the ball, shape [n, embed_dim].
    """
    K = np.exp(-(dist_matrix**2) / 2.0)
    eigvals, eigvecs = np.linalg.eigh(K)

    # Take top embed_dim components
    idx = np.argsort(eigvals)[::-1][:embed_dim]
    coords = eigvecs[:, idx] * np.sqrt(np.abs(eigvals[idx]))

    # Pad if embed_dim > n
    if coords.shape[1] < embed_dim:
        pad = np.zeros((coords.shape[0], embed_dim - coords.shape[1]))
        coords = np.concatenate([coords, pad], axis=1)

    # Scale to fit inside ball
    max_norm = np.max(np.linalg.norm(coords, axis=1))
    if max_norm > 0:
        coords = coords / max_norm * scale / (c**0.5)

    return torch.tensor(coords, dtype=torch.float32)
