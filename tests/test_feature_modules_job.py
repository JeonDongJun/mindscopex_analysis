"""Pure parts of the feature_modules job: graph diagnostics, module score, splits.

Everything here runs without a GPU. The job's expensive half is untestable offline,
so the rule is that any decision the job makes -- which threshold, which module,
which sign -- lives in a pure function that is pinned here.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from experiments.jobs import feature_modules
from experiments.jobs.feature_modules import (
    DEFAULT_SCORE_WEIGHTS,
    SCORE_TERMS,
    _cofire_rate,
    _load_splits,
    _module_sites,
    causal_term,
    coherence_term,
    frequency_term,
    generalization_term,
    graph_feature_selection,
    mean_active_member_delta,
    module_specificity,
    modules_document,
    negative_tail_ratio,
    pairwise_metric_matrix,
    permutation_null_row,
    rank_modules,
    rescaled_null_direction,
    resolve_score_weights,
    score_module,
    specificity_term,
    threshold_sweep,
)
from mindscopex_analysis import coactivation_edges, module_norm, rescale_to_norm

ROOT = Path(__file__).resolve().parents[1]

# The run that crashed with "no module survived jaccard >= 0.5". Its edge table is the
# only real measurement of this graph, so the config's threshold justification is
# pinned against it rather than against a synthetic distribution.
CRASHED_RUN_EDGES = (
    ROOT
    / "results"
    / "runs"
    / "20260822-154406_modules_affordance_27b"
    / "artifacts"
    / "coactivation_edges.csv"
)


def _load_edges(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as handle:
        return [
            {
                "feature_a": int(row["feature_a"]),
                "feature_b": int(row["feature_b"]),
                "co_fire": int(row["co_fire"]),
                "jaccard": float(row["jaccard"]),
                "activation_corr": float(row["activation_corr"]),
            }
            for row in csv.DictReader(handle)
        ]


def _terms(**overrides: float) -> dict[str, float]:
    base = dict.fromkeys(SCORE_TERMS, 0.5)
    base.update(overrides)
    return base


class GraphFeatureSelectionTests(unittest.TestCase):
    def test_default_keeps_every_feature_including_always_on(self) -> None:
        kept, dropped = graph_feature_selection([1, 2, 3], [10, 7, 3], 10)

        self.assertEqual(kept, [1, 2, 3])
        self.assertEqual(dropped, [])

    def test_the_opposite_arm_drops_the_always_on_features(self) -> None:
        kept, dropped = graph_feature_selection([1, 2, 3], [10, 7, 3], 10, max_active_frac=0.95)

        self.assertEqual(kept, [2, 3])
        self.assertEqual(dropped, [1])

    def test_the_boundary_feature_is_kept_not_dropped(self) -> None:
        # count == max_active_frac * n_cases exactly; float error must not drop it.
        kept, _ = graph_feature_selection([9], [8], 10, max_active_frac=0.8)

        self.assertEqual(kept, [9])

    def test_rejects_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            graph_feature_selection([1, 2], [1], 5)
        with self.assertRaises(ValueError):
            graph_feature_selection([1], [1], 5, max_active_frac=0.0)
        with self.assertRaises(ValueError):
            graph_feature_selection([1], [1], 0)


class NegativeTailRatioTests(unittest.TestCase):
    """It counts tails. It does not estimate a false-discovery rate, and never did."""

    EDGES = [
        {"feature_a": 1, "feature_b": 2, "jaccard": 1.0, "activation_corr": 0.9},
        {"feature_a": 1, "feature_b": 3, "jaccard": 1.0, "activation_corr": 0.6},
        {"feature_a": 2, "feature_b": 3, "jaccard": 1.0, "activation_corr": -0.6},
        {"feature_a": 3, "feature_b": 4, "jaccard": 1.0, "activation_corr": 0.1},
    ]

    def test_counts_the_two_tails(self) -> None:
        self.assertAlmostEqual(
            negative_tail_ratio(self.EDGES, metric="activation_corr", threshold=0.5), 0.5
        )

    def test_no_ratio_for_a_non_negative_metric(self) -> None:
        # Jaccard cannot go below 0, so it has no negative tail at all.
        self.assertIsNone(negative_tail_ratio(self.EDGES, metric="jaccard", threshold=0.5))

    def test_no_ratio_when_nothing_survives(self) -> None:
        self.assertIsNone(negative_tail_ratio(self.EDGES, metric="activation_corr", threshold=0.99))


class PermutationEdgeNullTests(unittest.TestCase):
    """The null that replaces the negative tail: shuffle each feature across items."""

    def test_the_vectorised_metric_matches_coactivation_edges(self) -> None:
        # The null must count edges by the SAME rule the real graph used, or its
        # counts are not comparable with the observed ones.
        generator = torch.Generator().manual_seed(7)
        matrix = torch.rand(20, 6, generator=generator)
        matrix[matrix < 0.3] = 0.0
        edges = coactivation_edges(matrix, list(range(6)))

        for metric in ("jaccard", "activation_corr"):
            pairwise = pairwise_metric_matrix(matrix, metric)
            for edge in edges:
                self.assertAlmostEqual(
                    float(pairwise[edge["feature_a"], edge["feature_b"]]),
                    float(edge[metric]),
                    places=6,
                )

    def test_a_constant_column_correlates_with_nothing(self) -> None:
        matrix = torch.tensor([[1.0, 2.0], [1.0, 5.0], [1.0, 9.0]])

        self.assertEqual(float(pairwise_metric_matrix(matrix, "activation_corr")[0, 1]), 0.0)

    def test_an_always_on_graph_is_forced_by_the_marginals(self) -> None:
        # The 27B run's failure, reproduced without a GPU: when every feature fires on
        # every item, a column permutation cannot change WHO fires WHERE, so the
        # Jaccard null is the complete graph too and permutation_p comes back at 1.0.
        # "every pair is an edge" was a property of the marginals, not a finding, and
        # this is the column that says so before the search is allowed to fail.
        generator = torch.Generator().manual_seed(11)
        dense = torch.rand(35, 8, generator=generator) + 1.0
        edges = coactivation_edges(dense, list(range(8)))

        row = threshold_sweep(
            edges,
            metric="jaccard",
            thresholds=[0.5],
            min_size=2,
            max_size=4,
            matrix=dense,
            null_draws=25,
            seed=0,
        )[0]

        self.assertEqual(row["n_edges_kept"], 28)
        self.assertEqual(row["permutation_null_mean_edges"], 28.0)
        self.assertEqual(row["permutation_p"], 1.0)

    def test_real_coactivation_beats_its_permutation_null(self) -> None:
        generator = torch.Generator().manual_seed(3)
        matrix = torch.randn(30, 5, generator=generator)
        matrix[:, 1] = matrix[:, 0] + 0.05 * torch.randn(30, generator=generator)
        edges = coactivation_edges(matrix, list(range(5)))

        row = threshold_sweep(
            edges,
            metric="activation_corr",
            thresholds=[0.55],
            min_size=2,
            max_size=4,
            matrix=matrix,
            null_draws=200,
            seed=0,
        )[0]

        self.assertEqual(row["n_edges_kept"], 1)
        self.assertLess(row["permutation_p"], 0.05)
        self.assertLess(row["permutation_null_mean_edges"], 0.5)

    def test_a_p_of_zero_is_not_reportable(self) -> None:
        # (b + 1) / (draws + 1), the same floor stats.sign_flip_p uses: sampled draws
        # cannot establish p = 0.
        self.assertAlmostEqual(permutation_null_row(5, [0] * 99)["permutation_p"], 0.01)

    def test_no_draws_reads_as_not_measured_not_as_zero(self) -> None:
        row = permutation_null_row(5, [])

        self.assertIsNone(row["permutation_p"])
        self.assertIsNone(row["permutation_null_mean_edges"])
        self.assertEqual(row["permutation_draws"], 0)

    def test_the_sweep_leaves_the_null_columns_null_when_not_asked(self) -> None:
        row = threshold_sweep(
            self.EDGES_FOR_SWEEP, metric="activation_corr", thresholds=[0.5], min_size=2, max_size=8
        )[0]

        self.assertIn("negative_tail_ratio", row)
        self.assertIsNone(row["permutation_p"])
        self.assertNotIn("symmetric_null_fdr", row)

    EDGES_FOR_SWEEP = [
        {"feature_a": 1, "feature_b": 2, "jaccard": 1.0, "activation_corr": 0.9},
        {"feature_a": 2, "feature_b": 3, "jaccard": 1.0, "activation_corr": 0.8},
    ]


class ThresholdSweepTests(unittest.TestCase):
    # 1-2-3 and 4-5 are the real groups; the weak 3-4 link is what fuses them into one
    # component at a low threshold, exactly the way the 27B graph fused into 40 nodes.
    EDGES = [
        {"feature_a": 1, "feature_b": 2, "jaccard": 1.0, "activation_corr": 0.9},
        {"feature_a": 2, "feature_b": 3, "jaccard": 1.0, "activation_corr": 0.8},
        {"feature_a": 3, "feature_b": 4, "jaccard": 1.0, "activation_corr": 0.15},
        {"feature_a": 4, "feature_b": 5, "jaccard": 1.0, "activation_corr": 0.6},
        {"feature_a": 5, "feature_b": 6, "jaccard": 1.0, "activation_corr": 0.2},
    ]

    def test_reports_the_swallowing_component_not_only_the_in_range_ones(self) -> None:
        # The component that swallowed the graph still has to appear in the sweep, or
        # the artifact hides the exact failure it exists to explain.
        row = threshold_sweep(
            self.EDGES, metric="activation_corr", thresholds=[0.1], min_size=2, max_size=3
        )[0]

        self.assertEqual(row["component_sizes"], [6])
        self.assertEqual(row["largest_component"], 6)
        self.assertEqual(row["n_modules_in_range"], 0)

    def test_raising_the_threshold_splits_the_graph(self) -> None:
        rows = threshold_sweep(
            self.EDGES, metric="activation_corr", thresholds=[0.1, 0.5], min_size=2, max_size=8
        )

        self.assertEqual(rows[0]["component_sizes"], [6])
        self.assertEqual(rows[1]["component_sizes"], [3, 2])
        self.assertEqual([row["n_edges_kept"] for row in rows], [5, 3])
        self.assertEqual([row["n_modules_in_range"] for row in rows], [1, 2])

    def test_empty_edges_sweep_without_raising(self) -> None:
        rows = threshold_sweep(
            [], metric="activation_corr", thresholds=[0.5], min_size=2, max_size=8
        )

        self.assertEqual(rows[0]["component_sizes"], [])
        self.assertEqual(rows[0]["largest_component"], 0)


@unittest.skipUnless(CRASHED_RUN_EDGES.exists(), "measured 27B edge table not in the tree")
class MeasuredGraphRegressionTests(unittest.TestCase):
    """Pins the numbers the config's threshold comment cites."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.edges = _load_edges(CRASHED_RUN_EDGES)

    def test_jaccard_could_not_have_been_fixed_by_raising_the_threshold(self) -> None:
        # 40 features, 780 pairs is the complete graph, and its weakest edge is 0.686,
        # so every threshold the config could plausibly carry keeps one component.
        self.assertEqual(len(self.edges), 780)
        self.assertGreater(min(edge["jaccard"] for edge in self.edges), 0.68)
        sizes = threshold_sweep(
            self.edges, metric="jaccard", thresholds=[0.5, 0.68], min_size=2, max_size=8
        )
        self.assertEqual([row["component_sizes"] for row in sizes], [[40], [40]])
        self.assertEqual([row["n_modules_in_range"] for row in sizes], [0, 0])

    def test_activation_corr_at_the_configured_threshold_finds_modules(self) -> None:
        row = threshold_sweep(
            self.edges, metric="activation_corr", thresholds=[0.55], min_size=2, max_size=8
        )[0]

        self.assertEqual(row["component_sizes"], [19, 4, 3, 2])
        self.assertEqual(row["n_modules_in_range"], 3)

    def test_the_negative_tail_is_far_too_heavy_to_be_a_noise_floor(self) -> None:
        # The claim the old `symmetric_null_fdr` name rested on: the negative tail
        # counts the false positives in the positive tail. At n = 35 an iid null puts
        # P(r >= 0.55) at 3.1e-4 (Monte Carlo, 400k draws), i.e. 0.24 of 780 pairs per
        # tail. 21 were observed. Whatever fills that tail, it is not sampling noise,
        # so the ratio is not a false-discovery rate and must not be named as one.
        negative = sum(1 for edge in self.edges if edge["activation_corr"] <= -0.55)
        iid_expected_per_tail = 3.1e-4 * len(self.edges)

        self.assertEqual(len(self.edges), 780)
        self.assertEqual(negative, 21)
        self.assertGreater(negative, 50 * iid_expected_per_tail)

    def test_the_negative_tail_curve_does_not_single_out_any_threshold(self) -> None:
        # This replaces a test that pinned the argmin of this curve at 0.55 as a
        # regression invariant, which made a noise minimum load-bearing. Two facts
        # kill that: the curve is non-monotone, and the candidate thresholds are
        # indistinguishable on it AND on the module count.
        sweep = threshold_sweep(
            self.edges,
            metric="activation_corr",
            thresholds=[0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
            min_size=2,
            max_size=8,
        )
        ratios = {row["threshold"]: row["negative_tail_ratio"] for row in sweep}
        in_range = {row["threshold"]: row["n_modules_in_range"] for row in sweep}

        # Non-monotone: it falls to 0.40, rises at 0.45, falls again, rises at 0.60.
        self.assertLess(ratios[0.40], ratios[0.30])
        self.assertGreater(ratios[0.45], ratios[0.40])
        self.assertGreater(ratios[0.60], ratios[0.55])
        # Four thresholds tie on the only criterion that has a stake in the answer.
        self.assertEqual([in_range[t] for t in (0.50, 0.55, 0.60, 0.65)], [3, 3, 3, 3])
        # And the differences between their ratios are inside the sampling error of
        # the counts behind them (sqrt(negative) / positive, ~0.07-0.11 here).
        candidates = [ratios[t] for t in (0.50, 0.55, 0.60, 0.65)]
        self.assertLess(max(candidates) - min(candidates), 0.11)


class ScoreTermTests(unittest.TestCase):
    def test_frequency_is_the_share_of_discovery_items(self) -> None:
        self.assertAlmostEqual(frequency_term(15.0, 20), 0.75)
        self.assertEqual(frequency_term(5.0, 0), 0.0)

    def test_anti_correlated_members_are_not_a_weak_module(self) -> None:
        self.assertEqual(coherence_term(-0.9), 0.0)
        self.assertAlmostEqual(coherence_term(0.4), 0.4)

    def test_generalization_needs_both_halves(self) -> None:
        # Firing together but no longer co-varying is the degeneracy that broke the
        # first 27B run, so it must not keep half its score.
        self.assertEqual(generalization_term(0.0, 1.0), 0.0)
        self.assertEqual(generalization_term(0.9, 0.0), 0.0)
        self.assertAlmostEqual(generalization_term(0.8, 0.5), 0.4)

    def test_causal_scale_is_the_half_way_delta(self) -> None:
        self.assertAlmostEqual(causal_term(0.2, scale=0.2), 0.5)
        self.assertGreater(causal_term(0.6, scale=0.2), causal_term(0.2, scale=0.2))
        self.assertLess(causal_term(10.0, scale=0.2), 1.0)

    def test_an_ablation_that_helped_the_lure_scores_zero(self) -> None:
        self.assertEqual(causal_term(-0.4, scale=0.2), 0.0)

    def test_causal_scale_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            causal_term(0.1, scale=0.0)

    def test_specificity_is_scale_free(self) -> None:
        # Same contrast, ten times the activation: the term must not change.
        self.assertAlmostEqual(specificity_term(3.0, 1.0), 0.5)
        self.assertAlmostEqual(specificity_term(30.0, 10.0), 0.5)

    def test_a_missing_control_would_read_as_perfect_selectivity(self) -> None:
        # This is why run() forces the term to 0 when no val item has a matched
        # control, instead of passing control_activation = 0.0 through: unmeasured
        # would otherwise score higher than any real measurement can.
        self.assertEqual(specificity_term(2.0, 0.0), 1.0)

    def test_a_feature_that_fires_equally_on_the_control_is_not_specific(self) -> None:
        self.assertEqual(specificity_term(2.0, 2.0), 0.0)
        self.assertEqual(specificity_term(1.0, 5.0), 0.0)
        self.assertEqual(specificity_term(0.0, 0.0), 0.0)


class ModuleSpecificityTests(unittest.TestCase):
    """Per-feature contrasts, averaged -- not the contrast of the pooled activations."""

    # A contains a perfectly selective feature and a passenger that tracks the surface
    # story; B contains no selective feature at all, only a magnitude difference.
    A_HOSTILE, A_CONTROL = [40.0, 2.0], [40.0, 0.0]
    B_HOSTILE, B_CONTROL = [40.0, 1.0], [20.0, 1.0]

    @staticmethod
    def _pooled(hostile: list[float], control: list[float]) -> float:
        """What the job used to do: mean the raw activations, then normalise once."""

        return specificity_term(sum(hostile) / len(hostile), sum(control) / len(control))

    def test_pooling_inverted_the_ranking_and_per_feature_restores_it(self) -> None:
        other = dict.fromkeys(("hostile_frequency", "generalization", "causal", "coherence"), 0.5)
        pooled_a = score_module(
            {**other, "specificity": self._pooled(self.A_HOSTILE, self.A_CONTROL)}
        )
        pooled_b = score_module(
            {**other, "specificity": self._pooled(self.B_HOSTILE, self.B_CONTROL)}
        )
        per_feature_a = score_module(
            {**other, "specificity": module_specificity(self.A_HOSTILE, self.A_CONTROL)}
        )
        per_feature_b = score_module(
            {**other, "specificity": module_specificity(self.B_HOSTILE, self.B_CONTROL)}
        )

        # The defect: the module with the selective member scored BELOW the one without.
        self.assertLess(pooled_a["score"], pooled_b["score"])
        self.assertGreater(per_feature_a["score"], per_feature_b["score"])

    def test_a_large_passenger_cannot_mask_a_selective_member(self) -> None:
        self.assertAlmostEqual(module_specificity(self.A_HOSTILE, self.A_CONTROL), 0.5)
        self.assertLess(self._pooled(self.A_HOSTILE, self.A_CONTROL), 0.03)

    def test_a_negative_member_cannot_cancel_a_positive_one(self) -> None:
        # qwen_scope_sparse_feature_values keeps negative TopK values on purpose. Pooled,
        # -38 eats the +40 and the denominator collapses; per feature the guard inside
        # specificity_term does its job on each member separately.
        self.assertEqual(self._pooled([40.0, -38.0], [1.0, 1.0]), 0.0)
        self.assertGreater(module_specificity([40.0, -38.0], [1.0, 1.0]), 0.4)

    def test_no_controls_scores_zero_not_perfect(self) -> None:
        self.assertEqual(module_specificity([], []), 0.0)

    def test_mismatched_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            module_specificity([1.0, 2.0], [1.0])


class MeanActiveMemberDeltaTests(unittest.TestCase):
    """A feature that did not fire contributes a structural zero, not a measurement."""

    PROBE = {
        # fires on all 4 probe items
        1: {"n_probe_items": 4, "n_active_probe_items": 4, "deltas_on_active_items": [0.4] * 4},
        # same effect where it fires, but only fires on 2 of 4
        2: {"n_probe_items": 4, "n_active_probe_items": 2, "deltas_on_active_items": [0.4, 0.4]},
        # never fired: module_ablation_direction(sae, [f], [0.0]) is the zero vector, so
        # every one of its margin_deltas is exactly 0.0 by construction
        3: {"n_probe_items": 4, "n_active_probe_items": 0, "deltas_on_active_items": []},
    }

    def test_a_sparser_member_is_not_penalised_for_being_sparse(self) -> None:
        dense, dense_measured = mean_active_member_delta(self.PROBE, [1])
        sparse, sparse_measured = mean_active_member_delta(self.PROBE, [2])

        self.assertTrue(dense_measured and sparse_measured)
        self.assertAlmostEqual(dense, sparse)
        # Averaging the two structural zeros in would have halved it to 0.2, and
        # causal_term(0.2) is 0.5 against causal_term(0.4) = 0.667 -- on the heaviest
        # weighted term, for a difference that is not about causality at all.
        self.assertAlmostEqual(causal_term(dense, scale=0.2), causal_term(sparse, scale=0.2))

    def test_a_member_that_never_fired_is_excluded_not_counted_as_zero(self) -> None:
        mixed, measured = mean_active_member_delta(self.PROBE, [2, 3])

        self.assertTrue(measured)
        self.assertAlmostEqual(mixed, 0.4)

    def test_nothing_measured_is_reported_as_nothing_measured(self) -> None:
        value, measured = mean_active_member_delta(self.PROBE, [3])

        self.assertFalse(measured)
        self.assertEqual(value, 0.0)

    def test_a_member_outside_the_probe_is_skipped(self) -> None:
        self.assertEqual(mean_active_member_delta(self.PROBE, [99]), (0.0, False))


class ScoreModuleTests(unittest.TestCase):
    def test_every_required_term_must_be_supplied(self) -> None:
        with self.assertRaises(ValueError):
            score_module({"hostile_frequency": 1.0})

    def test_all_equal_terms_score_that_value(self) -> None:
        # A geometric mean of a constant is the constant, whatever the weights are.
        self.assertAlmostEqual(score_module(_terms())["score"], 0.5, places=6)

    def test_a_dead_axis_drags_the_whole_score_down(self) -> None:
        balanced = score_module(_terms())["score"]
        lopsided = score_module(
            _terms(hostile_frequency=1.0, coherence=1.0, generalization=1.0, specificity=0.0)
        )["score"]

        # An additive score would rate the lopsided candidate ABOVE the balanced one
        # (mean 0.6 vs 0.5); the whole point of the geometric mean is that it does not.
        self.assertLess(lopsided, balanced)

    def test_the_floor_keeps_a_zero_from_annihilating_the_score(self) -> None:
        scored = score_module(_terms(specificity=0.0))

        self.assertGreater(scored["score"], 0.0)
        self.assertLess(scored["score"], score_module(_terms())["score"])

    def test_two_modules_that_both_hit_zero_are_still_ordered(self) -> None:
        weak = score_module(_terms(specificity=0.0, causal=0.1))["score"]
        strong = score_module(_terms(specificity=0.0, causal=0.9))["score"]

        self.assertLess(weak, strong)

    def test_weights_move_the_ranking(self) -> None:
        terms = _terms(causal=1.0, specificity=0.2)
        causal_heavy = score_module(terms, weights={**DEFAULT_SCORE_WEIGHTS, "causal": 8.0})
        specificity_heavy = score_module(
            terms, weights={**DEFAULT_SCORE_WEIGHTS, "specificity": 8.0}
        )

        self.assertGreater(causal_heavy["score"], specificity_heavy["score"])

    def test_terms_are_clamped_and_reported_back(self) -> None:
        scored = score_module(_terms(causal=1.4, specificity=-0.3))

        self.assertEqual(scored["terms"]["causal"], 1.0)
        self.assertEqual(scored["terms"]["specificity"], 0.0)

    def test_degenerate_weights_and_floor_rejected(self) -> None:
        with self.assertRaises(ValueError):
            score_module(_terms(), weights=dict.fromkeys(SCORE_TERMS, 0.0))
        with self.assertRaises(ValueError):
            score_module(_terms(), floor=0.0)
        with self.assertRaises(ValueError):
            score_module(_terms(), floor=1.0)

    def test_a_negative_weight_raises_instead_of_silently_dropping_a_term(self) -> None:
        # It used to be clamped to 0, which removed the term from the geometric mean
        # with nothing in the run saying so.
        with self.assertRaises(ValueError):
            score_module(_terms(), weights={**DEFAULT_SCORE_WEIGHTS, "coherence": -1.0})


class ResolveScoreWeightsTests(unittest.TestCase):
    def test_defaults_when_the_config_says_nothing(self) -> None:
        self.assertEqual(resolve_score_weights(None), DEFAULT_SCORE_WEIGHTS)
        self.assertEqual(resolve_score_weights({}), DEFAULT_SCORE_WEIGHTS)

    def test_a_transposed_term_name_is_rejected_not_ignored(self) -> None:
        # `casual` for `causal` used to run to completion on the default 1.5, so the
        # operator believed they had run the causal-heavy arm on a 27B session.
        with self.assertRaises(ValueError) as caught:
            resolve_score_weights({"casual": 3.0})

        self.assertIn("casual", str(caught.exception))

    def test_a_negative_weight_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_score_weights({"coherence": -1.0})

    def test_all_zero_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_score_weights(dict.fromkeys(SCORE_TERMS, 0.0))

    def test_a_known_term_overrides_its_default(self) -> None:
        resolved = resolve_score_weights({"causal": 3.0})

        self.assertEqual(resolved["causal"], 3.0)
        self.assertEqual(resolved["coherence"], DEFAULT_SCORE_WEIGHTS["coherence"])


class RankModulesTests(unittest.TestCase):
    def test_orders_by_score_not_by_size(self) -> None:
        ranked = rank_modules(
            [
                {"features": [1, 2, 3, 4, 5], "score": 0.2},
                {"features": [7, 8], "score": 0.8},
            ]
        )

        self.assertEqual([row["features"] for row in ranked], [[7, 8], [1, 2, 3, 4, 5]])
        self.assertEqual([row["rank"] for row in ranked], [1, 2])

    def test_ties_break_on_size_then_feature_ids(self) -> None:
        ranked = rank_modules(
            [
                {"features": [9, 10], "score": 0.5},
                {"features": [1, 2], "score": 0.5},
                {"features": [3, 4, 5], "score": 0.5},
            ]
        )

        self.assertEqual([row["features"] for row in ranked], [[3, 4, 5], [1, 2], [9, 10]])


class CofireRateTests(unittest.TestCase):
    MATRIX = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [3.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )

    def test_counts_only_items_where_every_member_fires(self) -> None:
        self.assertAlmostEqual(_cofire_rate(self.MATRIX, [10, 11, 12], [10, 11]), 0.5)
        self.assertAlmostEqual(_cofire_rate(self.MATRIX, [10, 11, 12], [10, 11, 12]), 0.25)

    def test_unknown_or_empty_module_scores_zero(self) -> None:
        self.assertEqual(_cofire_rate(self.MATRIX, [10, 11, 12], [99]), 0.0)
        self.assertEqual(_cofire_rate(torch.zeros(0, 3), [10, 11, 12], [10]), 0.0)


class ModulesDocumentTests(unittest.TestCase):
    def test_both_exit_paths_write_the_same_shape(self) -> None:
        # feature_modules.json used to be a bare [] when nothing was found and an
        # object when something was, so a reader doing
        # json.load(...)["ranked_candidates"] raised TypeError on exactly the runs the
        # recorded-outcome machinery exists to make machine-readable.
        found = modules_document(
            ranked=[{"features": [1, 2], "score": 0.4, "rank": 1}],
            all_component_sizes=[2, 5],
            n_candidates_scored=2,
            n_probe_items=4,
        )
        nothing = modules_document(
            ranked=[],
            all_component_sizes=[],
            n_candidates_scored=0,
            n_probe_items=0,
            no_module_reason="graph_too_small",
        )

        self.assertEqual(sorted(found), sorted(nothing))
        self.assertEqual(nothing["ranked_candidates"], [])
        self.assertIsNone(found["no_module_reason"])
        self.assertEqual(nothing["no_module_reason"], "graph_too_small")


class NullDirectionTests(unittest.TestCase):
    def test_rescale_to_norm_cannot_rescue_a_dead_direction(self) -> None:
        # The bug this guard exists for: rescale_to_norm divides by
        # clamp_min(norm, 1e-12), so a zero direction comes back with norm 0, NOT with
        # the target norm the caller asked for and the comment claimed.
        self.assertEqual(module_norm(rescale_to_norm(torch.zeros(4), 3.0)), 0.0)

    def test_a_dead_draw_is_refused_rather_than_recorded_as_a_zero_effect(self) -> None:
        self.assertIsNone(rescaled_null_direction(torch.zeros(4), 3.0))

    def test_a_silent_module_makes_every_matched_null_a_no_op(self) -> None:
        # target_norm = 0 means the real module fired on nothing here, so every
        # norm-matched null is the zero vector too; recording those as null
        # measurements would drag random_module_mean toward 0.
        self.assertIsNone(rescaled_null_direction(torch.ones(4), 0.0))

    def test_a_live_draw_carries_the_requested_norm(self) -> None:
        direction = rescaled_null_direction(torch.tensor([1.0, 2.0, 2.0, 0.0]), 3.0)

        self.assertIsNotNone(direction)
        self.assertAlmostEqual(module_norm(direction), 3.0, places=5)


class EditSiteTests(unittest.TestCase):
    def test_a_single_layer_module_is_one_add_vector_site(self) -> None:
        direction = torch.ones(4)
        sites = _module_sites(15, direction)

        self.assertEqual(len(sites), 1)
        self.assertEqual(sites[0].layer, 15)
        self.assertEqual(sites[0].intervention_mode, "add_vector")
        # add_vector ignores feature_value, so coefficient -1 subtracts the whole
        # module direction exactly once. Anything else changes the dose silently.
        self.assertEqual(sites[0].coefficient, -1.0)
        self.assertTrue(torch.equal(sites[0].direction, direction))


class RunExitPathTests(unittest.TestCase):
    """Every exit of run() with the model faked out -- the half no GPU-less test reached.

    The job's decisions live in pure functions, but its *exit paths* do not, and they
    are the ones that cost a Colab session when they go wrong: run 20260822-154406
    died after the 27B was resident and left a manifest reading `status: running`, no
    ARTIFACT_DIR line and no module_search.json. Stubbing the eight model-touching
    calls costs nothing and pins all five outcomes.
    """

    CONFIG = """
[run]
name = "fake_modules"
job = "feature_modules"

[model]
profile = "27b"

[data]
dataset = "goal_affordance_traps_v1"
conditions = ["hostile"]
train_frac = 0.6
split_seed = 0
graph_frac = 0.75
graph_seed = 101
control_condition = "counterfactual"
max_test_items = 3

[module]
layer = 5
edge_threshold = 0.55
sweep_thresholds = [0.5, 0.55]
permutation_null_draws = 20
min_size = 2
max_size = 8
max_modules = 1
random_modules = 2
score_probe_items = 2
seed = 0
"""

    D_MODEL = 8

    @staticmethod
    def _fake_residual(lm, prompts, layer, token_position="last", **kwargs):
        generator = torch.Generator().manual_seed(abs(hash(prompts[0])) % 9973)
        return torch.rand(1, RunExitPathTests.D_MODEL, generator=generator)

    @staticmethod
    def _fake_values(residual, sae, feature_ids):
        base = float(residual.sum())
        return torch.tensor([abs((base * (i + 1)) % 1.7) for i in range(len(feature_ids))])

    @staticmethod
    def _fake_margin(value):
        return SimpleNamespace(
            margin=value,
            correct=SimpleNamespace(logprob=value - 1.0),
            lure=SimpleNamespace(logprob=-1.0),
        )

    @staticmethod
    def _fake_baseline(lm, prompt, *, correct_answer, lure_answer):
        return RunExitPathTests._fake_margin(1.0)

    @staticmethod
    def _fake_edited(lm, prompt, *, correct_answer, lure_answer, sites):
        # The edit's effect is a function of the removed norm alone, which makes the
        # norm-matched null exactly right by construction -- so any gap this test sees
        # between module_joint and random_module would be a bug in the matching.
        removed = float(torch.linalg.norm(sites[0].direction.to(torch.float32)))
        return RunExitPathTests._fake_margin(1.0 - 0.3 * removed)

    @staticmethod
    def _fake_direction(sae, feature_ids, values):
        vector = torch.zeros(RunExitPathTests.D_MODEL)
        for feature_id, value in zip(feature_ids, values, strict=True):
            vector[int(feature_id) % RunExitPathTests.D_MODEL] += float(value)
        return vector

    @staticmethod
    def _graph_matrix(n_cases, n_features, *, correlated=True):
        generator = torch.Generator().manual_seed(5)
        matrix = torch.rand(n_cases, n_features, generator=generator) + 0.5
        if correlated and n_features >= 2:
            matrix[:, 1] = matrix[:, 0] + 0.01 * torch.rand(n_cases, generator=generator)
        return matrix

    def _run(self, sparse, extra_config=""):
        patched = mock.patch.multiple(
            feature_modules,
            load_qwen_language_model=mock.DEFAULT,
            load_qwen_scope_sae=mock.DEFAULT,
            dtype_from_name=mock.DEFAULT,
            sparse_activation_matrix=mock.Mock(side_effect=sparse),
            capture_layer_residuals=mock.Mock(side_effect=self._fake_residual),
            qwen_scope_sparse_feature_values=mock.Mock(side_effect=self._fake_values),
            answer_logprob_margin=mock.Mock(side_effect=self._fake_baseline),
            multi_site_answer_margin=mock.Mock(side_effect=self._fake_edited),
            module_ablation_direction=mock.Mock(side_effect=self._fake_direction),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.toml"
            config_path.write_text(self.CONFIG + extra_config, encoding="utf-8")
            with patched, contextlib.redirect_stdout(io.StringIO()) as captured:
                run_dir = feature_modules.run(config_path, root / "out")
            return {
                "stdout": captured.getvalue(),
                "manifest": json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")),
                "modules": json.loads(
                    (run_dir / "feature_modules.json").read_text(encoding="utf-8")
                ),
                "search": json.loads((run_dir / "module_search.json").read_text(encoding="utf-8")),
                "summary": json.loads(
                    (run_dir / "module_summary.json").read_text(encoding="utf-8")
                ),
            }

    def _sparse(self, feature_ids, **kwargs):
        def factory(lm, cases, **_):
            if not feature_ids:
                return [], torch.zeros(len(cases), 0)
            return list(feature_ids), self._graph_matrix(len(cases), len(feature_ids), **kwargs)

        return factory

    # ------------------------------------------------------------------ outcomes

    def test_a_completed_run_writes_the_module_document_and_the_null_caveat(self) -> None:
        out = self._run(self._sparse([10, 11, 12, 13]))

        self.assertEqual(out["manifest"]["status"], "ok")
        self.assertIsNone(out["modules"]["no_module_reason"])
        self.assertEqual(out["modules"]["ranked_candidates"][0]["features"], [10, 11])
        null = out["summary"][0]["random_module_null"]
        # The artifact, not just a code comment, has to carry what the null controls
        # for -- this comparison is norm-matched, not selection-matched.
        self.assertIn("removed norm", null["matched_on"])
        self.assertEqual(null["not_matched_on"], ["the module score that selected this module"])
        # NOT `null["caveat"] + "not selection-matched"` -- concatenating the needle
        # onto the haystack made this assertion unfailable, and it was hiding a real
        # mismatch: that phrase never appears in the caveat. Assert the property the
        # caveat has to carry instead of a magic substring, so a rewording does not
        # break the test and a REMOVED disclaimer does.
        self.assertIn("not a test", null["caveat"])
        self.assertIn("matched on", null["caveat"])
        self.assertIn("selected_causal_term", null)

    def test_the_norm_matched_null_lands_on_the_module_when_only_norm_matters(self) -> None:
        # Sanity check on the matching itself: the fake model's effect depends only on
        # the removed norm, so a correctly rescaled null must reproduce the module's
        # delta exactly. A gap here would mean rescale_to_norm was not applied.
        out = self._run(self._sparse([10, 11, 12, 13]))
        summary = out["summary"][0]

        self.assertAlmostEqual(summary["joint"]["mean"], summary["random_module_mean"], places=6)
        self.assertEqual(summary["random_module_null"]["n_skipped_null_draws"], 0)

    def test_no_feature_cleared_min_active_cases_is_recorded_not_raised(self) -> None:
        out = self._run(self._sparse([]))

        self.assertEqual(out["manifest"]["status"], "no_module_found")
        self.assertEqual(out["manifest"]["no_module_reason"], "no_features_in_graph")
        self.assertIn("NO_MODULE_FOUND", out["stdout"])
        self.assertIn("ARTIFACT_DIR=", out["stdout"])
        # module_search.json exists even here, so the next run is not flying blind.
        self.assertEqual(out["search"]["no_module_reason"], "no_features_in_graph")

    def test_a_graph_too_small_to_have_edges_is_recorded_not_raised(self) -> None:
        # The documented opposite arm, max_active_frac = 0.95, can land exactly here.
        out = self._run(self._sparse([10]))

        self.assertEqual(out["manifest"]["status"], "no_module_found")
        self.assertEqual(out["manifest"]["no_module_reason"], "graph_too_small")
        self.assertIn("NO_MODULE_FOUND", out["stdout"])

    def test_nothing_grouping_at_the_threshold_is_recorded_not_raised(self) -> None:
        out = self._run(self._sparse([10, 11, 12], correlated=False))

        self.assertEqual(out["manifest"]["status"], "no_module_found")
        self.assertEqual(out["manifest"]["no_module_reason"], "no_component_in_size_range")
        self.assertTrue(out["search"]["sweep"])

    def test_every_no_module_path_writes_the_success_shape(self) -> None:
        # The break this replaces: `[]` on one path and an object on the other, so
        # json.load(...)["ranked_candidates"] raised TypeError on exactly these runs.
        shapes = [
            sorted(self._run(sparse)["modules"])
            for sparse in (
                self._sparse([]),
                self._sparse([10]),
                self._sparse([10, 11, 12], correlated=False),
                self._sparse([10, 11, 12, 13]),
            )
        ]

        self.assertEqual(len(set(map(tuple, shapes))), 1)
        self.assertIn("ranked_candidates", shapes[0])

    def test_an_unexpected_exception_stamps_failed_instead_of_running(self) -> None:
        # The point of the requirement is what a reader finds in the artifact after a
        # crash, so asserting only that the exception propagates tests nothing: a job
        # that never wrote a manifest at all would pass. _run tears its temp dir down
        # on the way out, so the manifest has to be read from a directory that outlives
        # the failure.
        def explode(lm, cases, **_):
            raise ValueError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.toml"
            config_path.write_text(self.CONFIG, encoding="utf-8")
            patched = mock.patch.multiple(
                feature_modules,
                load_qwen_language_model=mock.DEFAULT,
                load_qwen_scope_sae=mock.DEFAULT,
                dtype_from_name=mock.DEFAULT,
                sparse_activation_matrix=mock.Mock(side_effect=explode),
                capture_layer_residuals=mock.Mock(side_effect=self._fake_residual),
                qwen_scope_sparse_feature_values=mock.Mock(side_effect=self._fake_values),
                answer_logprob_margin=mock.Mock(side_effect=self._fake_baseline),
                multi_site_answer_margin=mock.Mock(side_effect=self._fake_edited),
                module_ablation_direction=mock.Mock(side_effect=self._fake_direction),
            )
            with self.assertRaises(ValueError), patched, contextlib.redirect_stdout(io.StringIO()):
                feature_modules.run(config_path, root / "out")

            manifest_paths = sorted((root / "out").rglob("manifest.json"))
            self.assertTrue(manifest_paths, "the crash left no manifest to read")
            manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertNotEqual(manifest["status"], "running")

    def test_fail_on_empty_raises_only_after_the_artifacts_are_written(self) -> None:
        with self.assertRaises(RuntimeError):
            self._run(self._sparse([10]), extra_config="\nfail_on_empty = true\n")


# _margin_row's sign convention is deliberately NOT re-tested here.
# tests/test_delta_conventions.py pins it across every job that writes those columns,
# on a fixture where all three deltas differ in sign; a second copy of that assertion
# here would only make it possible for the two to disagree.


class SplitTests(unittest.TestCase):
    CONFIG = {
        "data": {
            "dataset": "goal_affordance_traps_v1",
            "conditions": ["hostile"],
            "train_frac": 0.6,
            "split_seed": 0,
            "graph_frac": 0.75,
            "graph_seed": 101,
            "control_condition": "counterfactual",
            "instruction": True,
            "max_test_items": 12,
        }
    }

    def test_graph_val_and_test_are_disjoint(self) -> None:
        splits = _load_splits(self.CONFIG)
        graph = {case.case_id for case in splits["graph"]}
        val = {case.case_id for case in splits["val"]}
        test = {case.case_id for case in splits["test"]}

        self.assertTrue(graph and val and test)
        self.assertEqual(graph & val, set())
        self.assertEqual(val & test, set())
        self.assertEqual(graph & test, set())

    def test_every_val_item_has_a_matched_control(self) -> None:
        splits = _load_splits(self.CONFIG)
        unmatched = [
            case.case_id for case in splits["val"] if case.pair_id not in splits["controls"]
        ]

        self.assertEqual(unmatched, [])

    def test_controls_come_from_the_control_condition_only(self) -> None:
        splits = _load_splits(self.CONFIG)

        self.assertTrue(
            all(case.condition == "counterfactual" for case in splits["controls"].values())
        )

    def test_reusing_the_split_seed_is_rejected_before_the_model_loads(self) -> None:
        # split_lure_cases buckets on hash(seed, case_id), so re-splitting the train
        # side under the same seed returns all of it and leaves val empty. Silently
        # scoring modules on their own graph items is the failure this guard prevents.
        config = {"data": {**self.CONFIG["data"], "graph_seed": 0}}

        with self.assertRaises(ValueError):
            _load_splits(config)

    def test_no_control_condition_leaves_the_controls_empty(self) -> None:
        config = {"data": {**self.CONFIG["data"], "control_condition": ""}}

        self.assertEqual(_load_splits(config)["controls"], {})


if __name__ == "__main__":
    unittest.main()
