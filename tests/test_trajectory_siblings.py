from __future__ import annotations

import unittest

from mindscopex_analysis.siblings import (
    difference_in_differences,
    pearson,
    rank_siblings,
    sibling_score,
)
from mindscopex_analysis.trajectory import (
    cue_span,
    find_subsequence,
    quantile_indices,
    reasoning_phases,
)


class QuantileTests(unittest.TestCase):
    def test_covers_both_ends_of_the_span(self) -> None:
        self.assertEqual(quantile_indices(10, 20, 3), [10, 14, 19])

    def test_single_phase_takes_the_start(self) -> None:
        self.assertEqual(quantile_indices(4, 9, 1), [4])

    def test_empty_span_yields_nothing(self) -> None:
        self.assertEqual(quantile_indices(5, 5, 3), [])

    def test_phases_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            quantile_indices(0, 5, 0)


class ReasoningPhaseTests(unittest.TestCase):
    def test_thinking_run_splits_deliberation_from_the_answer(self) -> None:
        # prompt 0-9, thinking 10-29, answer 30-39
        phases = reasoning_phases(10, 40, phases=3, think_end=30)
        labels = [p.phase for p in phases]

        self.assertEqual(labels[0], "prompt_last")
        self.assertEqual(phases[0].token_index, 9)
        self.assertIn("pre_answer", labels)
        self.assertEqual(phases[-1].token_index, 29)
        # every sampled reasoning index sits inside the thinking block
        for phase in phases[1:-1]:
            self.assertTrue(10 <= phase.token_index < 30)

    def test_no_thinking_run_has_no_pre_answer_phase(self) -> None:
        phases = reasoning_phases(10, 40, phases=3)
        self.assertNotIn("pre_answer", [p.phase for p in phases])

    def test_fractions_are_item_relative(self) -> None:
        # Different trace lengths must produce comparable labels, which is the whole
        # reason for sampling quantiles instead of fixed offsets.
        short = [p.phase for p in reasoning_phases(5, 25, phases=3, think_end=20)]
        long = [p.phase for p in reasoning_phases(5, 205, phases=3, think_end=200)]
        self.assertEqual(short, long)

    def test_rejects_impossible_lengths(self) -> None:
        with self.assertRaises(ValueError):
            reasoning_phases(0, 10)
        with self.assertRaises(ValueError):
            reasoning_phases(10, 5)


class SubsequenceAndCueTests(unittest.TestCase):
    def test_finds_and_misses_correctly(self) -> None:
        self.assertEqual(find_subsequence([1, 2, 3, 4], [3, 4]), 2)
        self.assertIsNone(find_subsequence([1, 2], [9]))
        self.assertIsNone(find_subsequence([1], [1, 2]))

    def test_cue_span_matches_across_token_boundaries(self) -> None:
        # The tokeniser splits "walking" into two pieces; matching on concatenated
        # text is what makes the span findable at all.
        tokens = ["The", " car", " wash", " is", " near", " so", " walk", "ing", " is", " quicker"]
        span = cue_span(tokens, "walking is quicker")
        self.assertIsNotNone(span)
        start, end = span
        self.assertLessEqual(start, 6)
        self.assertEqual(end, len(tokens))

    def test_absent_cue_returns_none(self) -> None:
        self.assertIsNone(cue_span(["a", "b"], "completely different text"))


class SiblingScoreTests(unittest.TestCase):
    def test_one_strong_signal_cannot_carry_two_weak_ones(self) -> None:
        # This is the failure mode of ranking on decoder cosine alone.
        cosine_only = sibling_score(0.95, 0.02, 0.02)
        balanced = sibling_score(0.4, 0.4, 0.4)
        self.assertGreater(balanced, cosine_only)

    def test_any_non_positive_signal_zeroes_the_score(self) -> None:
        # A negative correlation is evidence against the pair, not a small positive.
        self.assertEqual(sibling_score(0.9, -0.5, 0.9), 0.0)
        self.assertEqual(sibling_score(0.9, 0.0, 0.9), 0.0)

    def test_all_equal_signals_return_that_value(self) -> None:
        self.assertAlmostEqual(sibling_score(0.6, 0.6, 0.6), 0.6, places=6)

    def test_weights_shift_the_ranking(self) -> None:
        cosine_heavy = sibling_score(0.9, 0.1, 0.1, weights=(8.0, 1.0, 1.0))
        even = sibling_score(0.9, 0.1, 0.1)
        self.assertGreater(cosine_heavy, even)

    def test_zero_weights_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sibling_score(0.5, 0.5, 0.5, weights=(0.0, 0.0, 0.0))


class RankSiblingsTests(unittest.TestCase):
    CANDIDATES = [
        {"feature_id": 1, "decoder_cosine": 0.9, "activation_corr": 0.05, "effect_corr": 0.05},
        {"feature_id": 2, "decoder_cosine": 0.5, "activation_corr": 0.5, "effect_corr": 0.5},
        {"feature_id": 3, "decoder_cosine": 0.8, "activation_corr": -0.4, "effect_corr": 0.7},
    ]

    def test_balanced_candidate_wins_over_high_cosine(self) -> None:
        ranked = rank_siblings(self.CANDIDATES)
        self.assertEqual(ranked[0]["feature_id"], 2)

    def test_anticorrelated_candidate_is_dropped_not_ranked(self) -> None:
        ranked = rank_siblings(self.CANDIDATES)
        self.assertNotIn(3, [row["feature_id"] for row in ranked])

    def test_min_score_distinguishes_absent_from_poor(self) -> None:
        self.assertEqual(rank_siblings(self.CANDIDATES, min_score=0.9), [])


class DifferenceInDifferencesTests(unittest.TestCase):
    def test_subtracts_the_null_interaction(self) -> None:
        joint = [1.0, 1.0]
        parts = [[0.3, 0.3], [0.3, 0.3]]  # real interaction = +0.4 each
        null_joint = [0.5, 0.5]
        null_parts = [[0.2, 0.2], [0.2, 0.2]]  # null interaction = +0.1 each
        result = difference_in_differences(joint, parts, null_joint, null_parts)
        for value in result:
            self.assertAlmostEqual(value, 0.3, places=9)

    def test_pure_nonlinearity_cancels_to_zero(self) -> None:
        # Identical interaction in both arms means the real pair added nothing the
        # null did not; the statistic must report exactly zero.
        joint = [0.8, 0.6]
        parts = [[0.2, 0.1], [0.2, 0.1]]
        self.assertEqual(difference_in_differences(joint, parts, joint, parts), [0.0, 0.0])

    def test_misaligned_lengths_rejected(self) -> None:
        with self.assertRaises(ValueError):
            difference_in_differences([1.0], [[0.5, 0.5]], [1.0], [[0.5]])


class PearsonTests(unittest.TestCase):
    def test_constant_input_is_zero_not_one(self) -> None:
        self.assertEqual(pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]), 0.0)

    def test_perfect_and_inverse(self) -> None:
        self.assertAlmostEqual(pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0, places=6)
        self.assertAlmostEqual(pearson([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]), -1.0, places=6)

    def test_length_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pearson([1.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
