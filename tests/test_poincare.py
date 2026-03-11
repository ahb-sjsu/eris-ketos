# Copyright (c) 2026 Andrew H. Bond
# Licensed under the MIT License.

"""Tests for Poincaré ball geometry and HyperbolicMLR."""

import numpy as np
import pytest
import torch

from eris_ketos.poincare_coda import (
    HyperbolicMLR,
    PoincareBall,
    build_distance_matrix,
    embed_taxonomy_hyperbolic,
)

# =============================================================================
# PoincareBall
# =============================================================================


class TestPoincareBall:
    def test_project_inside_ball(self) -> None:
        ball = PoincareBall(c=1.0)
        x = torch.randn(10, 8) * 0.3
        projected = ball.project(x)
        norms = projected.norm(dim=-1)
        assert (norms < 1.0).all()

    def test_project_clamps_large_vectors(self) -> None:
        ball = PoincareBall(c=1.0)
        x = torch.randn(5, 4) * 10.0  # way outside the ball
        projected = ball.project(x)
        norms = projected.norm(dim=-1)
        assert (norms < 1.0).all()

    def test_expmap0_stays_in_ball(self) -> None:
        ball = PoincareBall(c=1.0)
        v = torch.randn(20, 16)
        p = ball.expmap0(v)
        assert (p.norm(dim=-1) < 1.0).all()

    def test_expmap0_logmap0_roundtrip(self) -> None:
        ball = PoincareBall(c=1.0)
        v = torch.randn(10, 8) * 0.5
        p = ball.expmap0(v)
        v_recovered = ball.logmap0(p)
        torch.testing.assert_close(v, v_recovered, atol=1e-4, rtol=1e-4)

    def test_dist_symmetric(self) -> None:
        ball = PoincareBall(c=1.0)
        x = ball.expmap0(torch.randn(5, 4) * 0.3)
        y = ball.expmap0(torch.randn(5, 4) * 0.3)
        d_xy = ball.dist(x, y)
        d_yx = ball.dist(y, x)
        torch.testing.assert_close(d_xy, d_yx, atol=1e-5, rtol=1e-5)

    def test_dist_non_negative(self) -> None:
        ball = PoincareBall(c=1.0)
        x = ball.expmap0(torch.randn(10, 8) * 0.3)
        y = ball.expmap0(torch.randn(10, 8) * 0.3)
        d = ball.dist(x, y)
        assert (d >= 0).all()

    def test_dist_to_self_is_zero(self) -> None:
        ball = PoincareBall(c=1.0)
        x = ball.expmap0(torch.randn(5, 4) * 0.3)
        d = ball.dist(x, x)
        torch.testing.assert_close(d, torch.zeros_like(d), atol=1e-4, rtol=1e-4)

    def test_dist_triangle_inequality(self) -> None:
        ball = PoincareBall(c=1.0)
        x = ball.expmap0(torch.randn(5, 4) * 0.2)
        y = ball.expmap0(torch.randn(5, 4) * 0.2)
        z = ball.expmap0(torch.randn(5, 4) * 0.2)
        d_xz = ball.dist(x, z)
        d_xy = ball.dist(x, y)
        d_yz = ball.dist(y, z)
        # d(x,z) <= d(x,y) + d(y,z) + epsilon
        assert (d_xz <= d_xy + d_yz + 1e-4).all()

    def test_mobius_add_identity(self) -> None:
        """Adding zero should be identity."""
        ball = PoincareBall(c=1.0)
        x = ball.expmap0(torch.randn(5, 4) * 0.3)
        zero = torch.zeros_like(x)
        result = ball.mobius_add(x, zero)
        torch.testing.assert_close(result, x, atol=1e-5, rtol=1e-5)

    def test_curvature_affects_distances(self) -> None:
        """Higher curvature should spread distances."""
        v = torch.randn(5, 4) * 0.3
        ball_low = PoincareBall(c=0.5)
        ball_high = PoincareBall(c=2.0)
        x_low = ball_low.expmap0(v)
        x_high = ball_high.expmap0(v)
        # Points at same tangent vector position have different norms
        assert not torch.allclose(x_low.norm(dim=-1), x_high.norm(dim=-1))

    def test_invalid_curvature(self) -> None:
        with pytest.raises(ValueError):
            PoincareBall(c=0)
        with pytest.raises(ValueError):
            PoincareBall(c=-1.0)

    def test_midpoint_symmetric(self) -> None:
        ball = PoincareBall(c=1.0)
        x = ball.expmap0(torch.randn(3, 4) * 0.3)
        y = ball.expmap0(torch.randn(3, 4) * 0.3)
        mid_xy = ball.midpoint(x, y)
        mid_yx = ball.midpoint(y, x)
        torch.testing.assert_close(mid_xy, mid_yx, atol=1e-5, rtol=1e-5)


# =============================================================================
# HyperbolicMLR
# =============================================================================


class TestHyperbolicMLR:
    def test_output_shape(self) -> None:
        mlr = HyperbolicMLR(embed_dim=16, num_classes=5, c=1.0)
        ball = PoincareBall(c=1.0)
        x = ball.expmap0(torch.randn(8, 16) * 0.1)
        logits = mlr(x)
        assert logits.shape == (8, 5)

    def test_gradients_flow(self) -> None:
        mlr = HyperbolicMLR(embed_dim=8, num_classes=3, c=1.0)
        ball = PoincareBall(c=1.0)
        x = ball.expmap0(torch.randn(4, 8, requires_grad=True) * 0.1)
        logits = mlr(x)
        loss = logits.sum()
        loss.backward()
        assert mlr.proto_tangent.grad is not None
        assert mlr.log_scale.grad is not None

    def test_init_from_taxonomy(self) -> None:
        mlr = HyperbolicMLR(embed_dim=16, num_classes=5, c=1.0)
        tax = torch.randn(5, 16) * 0.3
        mlr.init_from_taxonomy(tax)
        # Prototypes should be inside the ball
        protos = mlr.prototypes
        assert (protos.norm(dim=-1) < 1.0).all()

    def test_properties(self) -> None:
        mlr = HyperbolicMLR(embed_dim=32, num_classes=10, c=1.0)
        assert mlr.num_classes == 10
        assert mlr.embed_dim == 32


# =============================================================================
# Taxonomy embedding
# =============================================================================


class TestTaxonomyEmbedding:
    def test_build_distance_matrix(self) -> None:
        species = {
            "sp_a": {"family": "F1", "order": "O1"},
            "sp_b": {"family": "F1", "order": "O1"},
            "sp_c": {"family": "F2", "order": "O1"},
            "sp_d": {"family": "F3", "order": "O2"},
        }
        dist = build_distance_matrix(species)
        assert dist.shape == (4, 4)
        assert dist[0, 0] == 0  # self
        assert dist[0, 1] == 1  # same family
        assert dist[0, 2] == 2  # same order, diff family
        assert dist[0, 3] == 3  # diff order

    def test_embed_taxonomy_shape(self) -> None:
        dist = np.array([[0, 1, 2], [1, 0, 2], [2, 2, 0]], dtype=float)
        emb = embed_taxonomy_hyperbolic(dist, embed_dim=8, c=1.0)
        assert emb.shape == (3, 8)

    def test_embed_taxonomy_inside_ball(self) -> None:
        dist = np.array([[0, 1, 2], [1, 0, 2], [2, 2, 0]], dtype=float)
        emb = embed_taxonomy_hyperbolic(dist, embed_dim=8, c=1.0)
        norms = emb.norm(dim=-1)
        assert (norms < 1.0).all()

    def test_closer_species_closer_embeddings(self) -> None:
        """Species in same family should embed closer than different orders."""
        dist = np.array(
            [[0, 1, 3], [1, 0, 3], [3, 3, 0]],
            dtype=float,
        )
        emb = embed_taxonomy_hyperbolic(dist, embed_dim=16, c=1.0)
        ball = PoincareBall(c=1.0)
        d_01 = ball.dist(emb[0:1], emb[1:2]).item()
        d_02 = ball.dist(emb[0:1], emb[2:3]).item()
        assert d_01 < d_02
