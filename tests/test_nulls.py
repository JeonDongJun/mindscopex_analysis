from __future__ import annotations

import unittest

import torch

from mindscopex_analysis.nulls import (
    empirical_percentile,
    gaussian_null_directions,
    peer_null_directions,
    selection_adjusted_percentile,
)
from mindscopex_analysis.qwen_scope import QwenScopeSAE

D_MODEL = 4
D_SAE = 6


class EmpiricalPercentileTests(unittest.TestCase):
    def test_fraction_of_null_below_observed(self) -> None:
        self.assertEqual(empirical_percentile(0.5, [0.0, 0.1, 0.9, 1.0]), 0.5)

    def test_beating_every_draw_is_one(self) -> None:
        self.assertEqual(empirical_percentile(2.0, [0.0, 1.0]), 1.0)

    def test_no_draws_returns_none(self) -> None:
        self.assertIsNone(empirical_percentile(1.0, []))


class SelectionAdjustmentTests(unittest.TestCase):
    def test_max_of_k_is_stochastically_larger(self) -> None:
        # The whole point of the adjustment: searching harder raises the bar.
        null = [float(i) / 100 for i in range(100)]
        one = selection_adjusted_percentile(0.5, null, selection_k=1, bootstrap=4000, seed=0)
        many = selection_adjusted_percentile(0.5, null, selection_k=50, bootstrap=4000, seed=0)
        self.assertGreater(many["max_mean"], one["max_mean"])
        self.assertLess(many["percentile"], one["percentile"])

    def test_percentile_and_p_value_are_complements(self) -> None:
        null = [float(i) / 10 for i in range(10)]
        out = selection_adjusted_percentile(0.45, null, selection_k=4, bootstrap=2000, seed=1)
        self.assertAlmostEqual(out["percentile"] + out["p_value"], 1.0, places=6)

    def test_capped_by_the_largest_null_draw(self) -> None:
        # A bootstrap maximum can never exceed the sample maximum, so beating every
        # draw gives percentile 1.0 by construction -- documented, not a discovery.
        out = selection_adjusted_percentile(99.0, [0.0, 1.0, 2.0], selection_k=10, bootstrap=500)
        self.assertEqual(out["percentile"], 1.0)
        self.assertLessEqual(out["max_mean"], 2.0)

    def test_empty_null_is_reported_not_crashed(self) -> None:
        out = selection_adjusted_percentile(1.0, [], selection_k=5)
        self.assertIsNone(out["percentile"])


class NullDirectionTests(unittest.TestCase):
    def test_gaussian_directions_are_unit_and_deterministic(self) -> None:
        first = gaussian_null_directions(D_MODEL, 8, seed=3)
        second = gaussian_null_directions(D_MODEL, 8, seed=3)
        self.assertEqual(tuple(first.shape), (8, D_MODEL))
        self.assertTrue(torch.allclose(first, second))
        self.assertTrue(torch.allclose(first.norm(dim=1), torch.ones(8), atol=1e-6))
        self.assertFalse(torch.allclose(first, gaussian_null_directions(D_MODEL, 8, seed=4)))

    def test_peer_directions_are_unit_rows_of_the_decoder(self) -> None:
        w_dec = torch.arange(D_MODEL * D_SAE, dtype=torch.float32).reshape(D_MODEL, D_SAE)
        sae = QwenScopeSAE(
            repo_id="test/sae",
            layer=0,
            W_enc=torch.zeros(D_SAE, D_MODEL),
            W_dec=w_dec,
            b_enc=torch.zeros(D_SAE),
            b_dec=torch.zeros(D_MODEL),
            top_k=2,
        )
        directions = peer_null_directions(sae, [1, 3])
        self.assertEqual(tuple(directions.shape), (2, D_MODEL))
        self.assertTrue(torch.allclose(directions.norm(dim=1), torch.ones(2), atol=1e-6))

    def test_peer_directions_handle_empty_selection(self) -> None:
        sae = QwenScopeSAE(
            repo_id="test/sae",
            layer=0,
            W_enc=torch.zeros(D_SAE, D_MODEL),
            W_dec=torch.zeros(D_MODEL, D_SAE),
            b_enc=torch.zeros(D_SAE),
            b_dec=torch.zeros(D_MODEL),
            top_k=2,
        )
        self.assertEqual(tuple(peer_null_directions(sae, []).shape), (0, D_MODEL))


if __name__ == "__main__":
    unittest.main()
