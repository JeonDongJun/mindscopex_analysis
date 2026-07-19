from __future__ import annotations

import unittest

import torch
from torch import nn

from mindscopex_analysis import (
    find_decoder_block,
    null_summary,
    split_lure_cases,
    summarize_answer_labels,
)
from mindscopex_analysis.cases import LureCase


def _case(case_id: str, family: str) -> LureCase:
    return LureCase(
        case_id=case_id,
        family=family,
        prompt=f"{case_id}?\nAnswer:",
        correct_answer=" a",
        lure_answer=" b",
    )


class SplitLureCasesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = [
            _case(f"{family}_{index:03d}", family)
            for family in ("difference", "rate", "growth")
            for index in range(20)
        ]

    def test_split_is_deterministic(self) -> None:
        a_train, a_test = split_lure_cases(self.cases, train_frac=0.6, seed=0)
        b_train, b_test = split_lure_cases(self.cases, train_frac=0.6, seed=0)
        self.assertEqual([c.case_id for c in a_train], [c.case_id for c in b_train])
        self.assertEqual([c.case_id for c in a_test], [c.case_id for c in b_test])

    def test_split_is_disjoint_and_complete(self) -> None:
        train, test = split_lure_cases(self.cases, train_frac=0.6, seed=0)
        train_ids = {c.case_id for c in train}
        test_ids = {c.case_id for c in test}
        self.assertEqual(train_ids & test_ids, set())
        self.assertEqual(train_ids | test_ids, {c.case_id for c in self.cases})
        self.assertTrue(train and test)

    def test_seed_changes_the_split(self) -> None:
        train0, _ = split_lure_cases(self.cases, train_frac=0.6, seed=0)
        train1, _ = split_lure_cases(self.cases, train_frac=0.6, seed=1)
        self.assertNotEqual([c.case_id for c in train0], [c.case_id for c in train1])

    def test_stratifies_within_each_family(self) -> None:
        train, test = split_lure_cases(self.cases, train_frac=0.6, seed=0)
        for family in ("difference", "rate", "growth"):
            in_family = {c.case_id for c in self.cases if c.family == family}
            covered = {c.case_id for c in train if c.family == family}
            covered |= {c.case_id for c in test if c.family == family}
            self.assertEqual(covered, in_family)
            # roughly train_frac of each family lands in train (loose bound for hashing)
            n_train = sum(1 for c in train if c.family == family)
            self.assertTrue(6 <= n_train <= 18, n_train)

    def test_rejects_degenerate_fraction(self) -> None:
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                split_lure_cases(self.cases, train_frac=bad)


class NullSummaryTests(unittest.TestCase):
    def test_zero_variance_null(self) -> None:
        summary = null_summary(1.0, [0.0, 0.0, 0.0, 0.0])
        self.assertIsNone(summary["z"])
        self.assertEqual(summary["percentile"], 1.0)
        self.assertEqual(summary["null_mean"], 0.0)

    def test_zscore_and_percentile(self) -> None:
        summary = null_summary(2.0, [-1.0, 0.0, 1.0, 2.0])
        self.assertAlmostEqual(summary["null_mean"], 0.5)
        self.assertAlmostEqual(summary["z"], 1.5 / summary["null_std"], places=6)
        self.assertAlmostEqual(summary["percentile"], 0.75)

    def test_empty_null(self) -> None:
        summary = null_summary(1.0, [])
        self.assertEqual(summary["null_n"], 0)
        self.assertIsNone(summary["z"])


class SummarizeAnswerLabelsTests(unittest.TestCase):
    def test_counts_and_rates(self) -> None:
        summary = summarize_answer_labels(["correct", "correct", "lure", "other"])
        self.assertEqual(summary["n"], 4)
        self.assertEqual(summary["correct"], 2)
        self.assertAlmostEqual(summary["accuracy"], 0.5)
        self.assertAlmostEqual(summary["lure_rate"], 0.25)

    def test_empty(self) -> None:
        summary = summarize_answer_labels([])
        self.assertEqual(summary["accuracy"], 0.0)
        self.assertEqual(summary["lure_rate"], 0.0)


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])


class _LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _Inner()


class _Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _LanguageModel()


class FindDecoderBlockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _Wrapper()

    def test_resolves_via_template_path(self) -> None:
        block = find_decoder_block(
            self.model, 2, block_path_template="model.language_model.layers.{layer}"
        )
        self.assertIs(block, self.model.model.language_model.layers[2])

    def test_resolves_via_named_modules_fallback(self) -> None:
        block = find_decoder_block(self.model, 3, block_path_template="does.not.match.{layer}")
        self.assertIs(block, self.model.model.language_model.layers[3])

    def test_missing_layer_raises(self) -> None:
        with self.assertRaises(ValueError):
            find_decoder_block(self.model, 99, block_path_template="does.not.match.{layer}")

    def test_hookable(self) -> None:
        block = find_decoder_block(
            self.model, 0, block_path_template="model.language_model.layers.{layer}"
        )
        seen: list[int] = []
        handle = block.register_forward_hook(lambda m, i, o: seen.append(1))
        try:
            block(torch.zeros(1, 2))
        finally:
            handle.remove()
        self.assertEqual(seen, [1])


if __name__ == "__main__":
    unittest.main()
