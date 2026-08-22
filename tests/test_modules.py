from __future__ import annotations

import unittest

import torch

from mindscopex_analysis.modules import (
    coactivation_edges,
    module_ablation_direction,
    module_coherence,
    modules_from_edges,
    rescale_to_norm,
    sample_frequency_matched_modules,
)
from mindscopex_analysis.qwen_scope import QwenScopeSAE

D_MODEL = 3
D_SAE = 6


def _sae() -> QwenScopeSAE:
    # W_dec is (d_model, d_sae); feature f is the one-hot column scaled by (f + 1).
    w_dec = torch.zeros(D_MODEL, D_SAE)
    for f in range(D_SAE):
        w_dec[f % D_MODEL, f] = float(f + 1)
    return QwenScopeSAE(
        repo_id="test/sae",
        layer=0,
        W_enc=torch.zeros(D_SAE, D_MODEL),
        W_dec=w_dec,
        b_enc=torch.zeros(D_SAE),
        b_dec=torch.zeros(D_MODEL),
        top_k=3,
    )


class CoactivationTests(unittest.TestCase):
    def test_jaccard_and_correlation_are_reported_separately(self) -> None:
        # 0 and 1 fire on the same two items and move together; 0 and 2 never overlap.
        matrix = torch.tensor([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 3.0]])
        edges = coactivation_edges(matrix, [10, 11, 12])
        by_pair = {(e["feature_a"], e["feature_b"]): e for e in edges}

        self.assertAlmostEqual(by_pair[(10, 11)]["jaccard"], 1.0, places=6)
        self.assertGreater(by_pair[(10, 11)]["activation_corr"], 0.9)
        self.assertAlmostEqual(by_pair[(10, 12)]["jaccard"], 0.0, places=6)

    def test_min_jaccard_filters_edges(self) -> None:
        matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(coactivation_edges(matrix, [1, 2], min_jaccard=0.5), [])

    def test_no_edges_without_two_features(self) -> None:
        self.assertEqual(coactivation_edges(torch.ones(3, 1), [7]), [])


class ModuleBuildingTests(unittest.TestCase):
    EDGES = [
        {"feature_a": 1, "feature_b": 2, "jaccard": 0.9, "activation_corr": 0.1},
        {"feature_a": 2, "feature_b": 3, "jaccard": 0.8, "activation_corr": 0.1},
        {"feature_a": 4, "feature_b": 5, "jaccard": 0.7, "activation_corr": 0.95},
        {"feature_a": 1, "feature_b": 5, "jaccard": 0.1, "activation_corr": 0.05},
    ]

    def test_connected_components_above_the_threshold(self) -> None:
        modules = modules_from_edges(self.EDGES, edge_threshold=0.5)
        self.assertEqual(modules, [[1, 2, 3], [4, 5]])

    def test_weak_edge_does_not_merge_components(self) -> None:
        # The 1-5 edge is below threshold, so the two groups must stay separate.
        modules = modules_from_edges(self.EDGES, edge_threshold=0.5)
        self.assertTrue(all(not ({1, 5} <= set(module)) for module in modules))

    def test_switching_metric_changes_the_grouping(self) -> None:
        modules = modules_from_edges(self.EDGES, edge_threshold=0.5, metric="activation_corr")
        self.assertEqual(modules, [[4, 5]])

    def test_max_size_drops_a_component_that_swallowed_the_graph(self) -> None:
        self.assertEqual(modules_from_edges(self.EDGES, edge_threshold=0.5, max_size=2), [[4, 5]])

    def test_unknown_metric_rejected(self) -> None:
        with self.assertRaises(ValueError):
            modules_from_edges(self.EDGES, metric="nope")


class ModuleDirectionTests(unittest.TestCase):
    def test_joint_direction_equals_the_sum_of_the_members(self) -> None:
        # This equivalence is why a same-layer module needs no multi-site edit.
        sae = _sae()
        joint = module_ablation_direction(sae, [1, 4], [2.0, 3.0])
        separately = module_ablation_direction(sae, [1], [2.0]) + module_ablation_direction(
            sae, [4], [3.0]
        )
        self.assertTrue(torch.allclose(joint, separately, atol=1e-6))

    def test_length_mismatch_and_empty_module_rejected(self) -> None:
        sae = _sae()
        with self.assertRaises(ValueError):
            module_ablation_direction(sae, [1, 2], [1.0])
        with self.assertRaises(ValueError):
            module_ablation_direction(sae, [], [])

    def test_rescale_matches_the_target_norm(self) -> None:
        rescaled = rescale_to_norm(torch.tensor([3.0, 4.0, 0.0]), 10.0)
        self.assertAlmostEqual(float(rescaled.norm()), 10.0, places=5)


class RandomModuleTests(unittest.TestCase):
    FEATURES = [1, 2, 3, 4, 5, 6, 7, 8]
    COUNTS = [10, 10, 10, 10, 2, 2, 2, 2]

    def test_matches_the_frequency_profile_of_the_real_module(self) -> None:
        modules = sample_frequency_matched_modules(
            self.FEATURES, self.COUNTS, size=2, exclude=[1, 2], n_modules=5, seed=0
        )
        self.assertTrue(modules)
        for module in modules:
            self.assertEqual(len(module), 2)
            # The real module's members both fire 10 times, so the null must too --
            # a null of rare features would remove less norm on fewer items.
            self.assertTrue(all(self.COUNTS[self.FEATURES.index(f)] == 10 for f in module))

    def test_never_reuses_the_real_module_members(self) -> None:
        modules = sample_frequency_matched_modules(
            self.FEATURES, self.COUNTS, size=2, exclude=[1, 2], n_modules=5, seed=1
        )
        for module in modules:
            self.assertNotIn(1, module)
            self.assertNotIn(2, module)

    def test_deterministic_in_seed(self) -> None:
        kwargs = {"size": 2, "exclude": [1], "n_modules": 4}
        first = sample_frequency_matched_modules(self.FEATURES, self.COUNTS, seed=7, **kwargs)
        second = sample_frequency_matched_modules(self.FEATURES, self.COUNTS, seed=7, **kwargs)
        self.assertEqual(first, second)

    def test_returns_empty_when_the_pool_cannot_fill_a_module(self) -> None:
        modules = sample_frequency_matched_modules(
            [1, 2], [10, 10], size=2, exclude=[1, 2], n_modules=3, seed=0
        )
        self.assertEqual(modules, [])


class CoherenceTests(unittest.TestCase):
    def test_singleton_has_no_coherence(self) -> None:
        self.assertEqual(module_coherence(torch.ones(3, 2), [1, 2], [1]), 0.0)

    def test_perfectly_correlated_members_score_one(self) -> None:
        matrix = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        self.assertAlmostEqual(module_coherence(matrix, [1, 2], [1, 2]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
