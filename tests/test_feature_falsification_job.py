from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.jobs import feature_falsification as job
from experiments.jobs.feature_falsification import (
    NOT_TESTED,
    PROXY,
    TESTED,
    _acceptance,
    _contrast,
    _corr,
    _error_audit_acceptance,
    _family_round_robin,
    _jaccard,
    _length_matched_pairs,
    _paraphrase_groups,
    _partial_correlation,
    _split_map,
    _strict_failures,
    _template_family_confound,
)
from mindscopex_analysis import LureCase, lure_dataset_cases, pearson

CONFIG = Path("experiments/configs/falsify_affordance_27b.toml").resolve()


def _record(
    case_id: str,
    *,
    pair_id: str,
    template_id: str,
    condition: str = "hostile",
    prompt: str = "",
    family: str = "f",
) -> dict[str, str]:
    return {
        "case_id": case_id,
        "pair_id": pair_id,
        "template_id": template_id,
        "condition": condition,
        "prompt": prompt or case_id,
        "family": family,
    }


class ParaphraseGroupTests(unittest.TestCase):
    """F3 is only testable when one scenario exists in two wordings."""

    def test_one_wording_per_scenario_is_not_a_paraphrase_group(self) -> None:
        records = [
            _record("a_hostile", pair_id="a", template_id="t1"),
            _record("b_hostile", pair_id="b", template_id="t2"),
        ]
        self.assertEqual(_paraphrase_groups(records, condition="hostile"), {})

    def test_same_pair_two_templates_and_two_wordings_qualifies(self) -> None:
        records = [
            _record("a_v1_hostile", pair_id="a", template_id="t1", prompt="walk or drive?"),
            _record("a_v2_hostile", pair_id="a", template_id="t2", prompt="drive or walk?"),
        ]
        groups = _paraphrase_groups(records, condition="hostile")
        self.assertEqual(list(groups), ["a"])
        self.assertEqual(len(groups["a"]), 2)

    def test_same_wording_under_two_template_ids_is_not_a_paraphrase(self) -> None:
        # Relabelling an item does not make it a rewrite of itself.
        records = [
            _record("a_1_hostile", pair_id="a", template_id="t1", prompt="same text"),
            _record("a_2_hostile", pair_id="a", template_id="t2", prompt="same text"),
        ]
        self.assertEqual(_paraphrase_groups(records, condition="hostile"), {})

    def test_other_conditions_do_not_leak_into_the_group(self) -> None:
        # A hostile item and its own neutral twin differ in wording *because* the
        # structure differs -- pairing them would measure the trap, not the phrasing.
        records = [
            _record("a_v1_hostile", pair_id="a", template_id="t1", prompt="one"),
            _record(
                "a_v1_neutral", pair_id="a", template_id="t2", condition="neutral", prompt="two"
            ),
        ]
        self.assertEqual(_paraphrase_groups(records, condition="hostile"), {})

    def test_the_shipped_datasets_have_no_paraphrase_arm(self) -> None:
        # The regression that matters: the job used to report across-template spread as
        # paraphrase invariance. If this ever starts failing, a paraphrase arm was built
        # and the F3 slot should switch to `tested` on its own.
        for dataset in ("goal_affordance_traps_v1", "goal_affordance_traps_v2"):
            records = [
                {
                    "case_id": case.case_id,
                    "pair_id": case.pair_id,
                    "template_id": case.template_id,
                    "condition": case.condition,
                    "prompt": case.prompt,
                    "family": case.family,
                }
                for case in lure_dataset_cases(dataset)
            ]
            self.assertEqual(_paraphrase_groups(records, condition="hostile"), {}, msg=dataset)


class TemplateFamilyConfoundTests(unittest.TestCase):
    def test_template_id_that_never_crosses_a_family_is_a_family_label(self) -> None:
        records = [
            _record("a", pair_id="a", template_id="t1", family="one"),
            _record("b", pair_id="b", template_id="t1", family="one"),
            _record("c", pair_id="c", template_id="t2", family="two"),
        ]
        confound = _template_family_confound(records)
        self.assertTrue(confound["template_id_is_a_family_label"])
        self.assertEqual(confound["n_template_ids"], 2)
        self.assertEqual(confound["n_families"], 2)

    def test_two_templates_inside_one_family_carry_wording_information(self) -> None:
        records = [
            _record("a", pair_id="a", template_id="t1", family="one"),
            _record("b", pair_id="b", template_id="t2", family="one"),
        ]
        self.assertFalse(_template_family_confound(records)["template_id_is_a_family_label"])

    def test_shipped_v1_template_ids_are_family_labels(self) -> None:
        records = [
            {
                "case_id": case.case_id,
                "pair_id": case.pair_id,
                "template_id": case.template_id,
                "condition": case.condition,
                "prompt": case.prompt,
                "family": case.family,
            }
            for case in lure_dataset_cases("goal_affordance_traps_v1")
        ]
        self.assertTrue(_template_family_confound(records)["template_id_is_a_family_label"])


class LengthMatchingTests(unittest.TestCase):
    def test_pairs_only_inside_the_caliper(self) -> None:
        positive = [{"prompt_tokens": 50}, {"prompt_tokens": 80}]
        negative = [{"prompt_tokens": 48}, {"prompt_tokens": 49}]
        pairs = _length_matched_pairs(positive, negative, caliper=4)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0]["prompt_tokens"], 50)
        self.assertEqual(pairs[0][1]["prompt_tokens"], 49)

    def test_matching_is_without_replacement(self) -> None:
        positive = [{"prompt_tokens": 40}, {"prompt_tokens": 41}]
        negative = [{"prompt_tokens": 40}]
        pairs = _length_matched_pairs(positive, negative, caliper=4)
        self.assertEqual(len(pairs), 1)

    def test_no_overlap_yields_no_pairs_rather_than_a_stretched_match(self) -> None:
        # The honest outcome when the arms do not overlap in length is "nothing to
        # compare", not a comparison across a 30-token gap.
        positive = [{"prompt_tokens": 90}]
        negative = [{"prompt_tokens": 40}]
        self.assertEqual(_length_matched_pairs(positive, negative, caliper=4), [])

    def test_order_of_the_inputs_does_not_change_the_result(self) -> None:
        positive = [{"prompt_tokens": 52}, {"prompt_tokens": 48}, {"prompt_tokens": 50}]
        negative = [{"prompt_tokens": 49}, {"prompt_tokens": 51}]
        first = _length_matched_pairs(positive, negative, caliper=4)
        second = _length_matched_pairs(list(reversed(positive)), negative, caliper=4)
        self.assertEqual(
            [(p["prompt_tokens"], n["prompt_tokens"]) for p, n in first],
            [(p["prompt_tokens"], n["prompt_tokens"]) for p, n in second],
        )

    def test_negative_caliper_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _length_matched_pairs([], [], caliper=-1)


class PartialCorrelationTests(unittest.TestCase):
    def test_an_association_that_is_entirely_length_collapses_to_zero(self) -> None:
        # The exact failure mode this job has to be able to see: the hostile arm is
        # longer, activation tracks length, and nothing else. The raw correlation reads
        # as a strong condition effect (0.86); the partial has to report ~0.
        length = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
        condition = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        jitter = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]  # orthogonal to both
        activation = [3.0 * value + noise for value, noise in zip(length, jitter, strict=True)]

        self.assertGreater(pearson(activation, condition), 0.8)
        result = _partial_correlation(activation, condition, length)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_perfect_collinearity_is_undefined_rather_than_zero(self) -> None:
        # activation exactly determined by length leaves no residual to correlate. 0.0
        # would read as "the condition effect is gone", which is a claim this data
        # cannot make either way.
        length = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        condition = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        self.assertIsNone(
            _partial_correlation([2.0 * value for value in length], condition, length)
        )

    def test_an_association_orthogonal_to_length_survives(self) -> None:
        length = [10.0, 10.0, 10.0, 10.0]
        condition = [0.0, 0.0, 1.0, 1.0]
        activation = [1.0, 1.1, 3.0, 3.1]
        result = _partial_correlation(activation, condition, length)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0.9)

    def test_a_constant_condition_indicator_is_undefined_not_zero(self) -> None:
        # An empty control arm makes every indicator 1.0. `pearson` maps that to 0.0,
        # so the job published "the condition effect is exactly zero once length is
        # controlled" -- its single most damaging conclusion -- when there was no
        # control arm at all.
        activation = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        length = [10.0, 12.0, 11.0, 15.0, 13.0, 18.0]
        self.assertEqual(pearson(activation, [1.0] * 6), 0.0)
        self.assertIsNone(_partial_correlation(activation, [1.0] * 6, length))

    def test_a_silent_feature_is_undefined_not_zero(self) -> None:
        # A feature that never fires has no residual either; 0.0 would read as a
        # measured "the condition does not explain it".
        condition = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        length = [10.0, 12.0, 11.0, 15.0, 13.0, 18.0]
        self.assertIsNone(_partial_correlation([0.0] * 6, condition, length))

    def test_a_constant_z_is_not_degenerate(self) -> None:
        # Removing a constant regressor removes nothing, so the answer is corr(x, y).
        activation = [1.0, 2.0, 5.0, 6.0]
        condition = [0.0, 0.0, 1.0, 1.0]
        result = _partial_correlation(activation, condition, [7.0] * 4)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, pearson(activation, condition), places=9)

    def test_too_few_points_returns_none_not_a_number(self) -> None:
        self.assertIsNone(_partial_correlation([1.0, 2.0], [0.0, 1.0], [3.0, 4.0]))

    def test_mismatched_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _partial_correlation([1.0, 2.0, 3.0], [0.0, 1.0], [1.0, 2.0, 3.0])


class CorrelationGuardTests(unittest.TestCase):
    """`pearson` maps undefined to 0.0; an artifact reads 0.0 as a measurement."""

    def test_a_constant_side_is_none_not_zero(self) -> None:
        self.assertEqual(pearson([1.0] * 4, [1.0, 2.0, 3.0, 4.0]), 0.0)
        self.assertIsNone(_corr([1.0] * 4, [1.0, 2.0, 3.0, 4.0]))
        self.assertIsNone(_corr([1.0, 2.0, 3.0, 4.0], [7.0] * 4))

    def test_an_empty_or_single_point_input_is_none(self) -> None:
        self.assertIsNone(_corr([], []))
        self.assertIsNone(_corr([1.0], [2.0]))

    def test_a_real_correlation_matches_pearson(self) -> None:
        a, b = [1.0, 2.0, 3.0, 4.0], [2.0, 4.1, 5.9, 8.0]
        self.assertAlmostEqual(_corr(a, b), pearson(a, b), places=12)


class LexicalOverlapTests(unittest.TestCase):
    def test_identical_text_is_one(self) -> None:
        self.assertEqual(_jaccard("drive to the car wash", "drive to the car wash"), 1.0)

    def test_deleting_a_sentence_costs_overlap(self) -> None:
        hostile = "The car is beside me. Walking would be quicker for such a short distance."
        neutral = "The car is beside me."
        self.assertLess(_jaccard(hostile, neutral), 1.0)
        self.assertGreater(_jaccard(hostile, neutral), 0.0)

    def test_disjoint_text_is_zero(self) -> None:
        self.assertEqual(_jaccard("alpha beta", "gamma delta"), 0.0)

    def test_empty_pair_is_zero_not_a_division_error(self) -> None:
        self.assertEqual(_jaccard("", ""), 0.0)


class ContrastTests(unittest.TestCase):
    """Every published contrast carries its own length adjustment, or says NOT TESTED."""

    @staticmethod
    def _rows(pairs: list[tuple[float, float]]) -> list[dict[str, float]]:
        return [{"activation": activation, "prompt_tokens": tokens} for activation, tokens in pairs]

    def test_a_missing_arm_publishes_nulls_and_a_not_tested_caveat(self) -> None:
        block = _contrast(
            self._rows([(1.0, 50.0), (2.0, 52.0)]),
            [],
            positive_label="hostile",
            other_label="neutral",
        )
        self.assertIsNone(block["auc_positive_vs_other"])
        self.assertIsNone(block["gap_positive_minus_other"])
        self.assertIsNone(block["token_gap_positive_minus_other"])
        self.assertIsNone(block["partial_corr_condition_given_length"])
        self.assertTrue(block["length_caveat"].startswith("NOT TESTED"))

    def test_a_large_gap_is_flagged_and_the_matched_subset_is_published(self) -> None:
        # The reference-control shape: separated arms that still overlap a little.
        positive = self._rows([(3.0, 100.0), (3.0, 78.0), (3.0, 102.0)])
        other = self._rows([(1.0, 76.0), (1.0, 74.0), (1.0, 75.0)])
        block = _contrast(
            positive, other, positive_label="hostile", other_label="reference_control", caliper=4.0
        )
        self.assertGreater(block["token_gap_positive_minus_other"], 2.0)
        self.assertTrue(block["length_caveat"].startswith("NOT length-controlled"))
        self.assertEqual(block["length_matched"]["n_pairs"], 1)
        self.assertIsNotNone(block["length_matched"]["auc"])

    def test_arms_that_never_overlap_report_the_matched_auc_as_none(self) -> None:
        # Not a number: there is no length-controlled version of this contrast at all.
        block = _contrast(
            self._rows([(3.0, 120.0), (3.0, 125.0)]),
            self._rows([(1.0, 60.0), (1.0, 62.0)]),
            positive_label="hostile",
            other_label="reference_control",
            caliper=4.0,
        )
        self.assertEqual(block["length_matched"]["n_pairs"], 0)
        self.assertIsNone(block["length_matched"]["auc"])
        self.assertIn("NOT TESTED", block["length_caveat"])

    def test_a_matched_arm_is_not_flagged(self) -> None:
        block = _contrast(
            self._rows([(3.0, 50.0), (4.0, 51.0)]),
            self._rows([(1.0, 50.0), (1.0, 51.0)]),
            positive_label="hostile",
            other_label="neutral",
            max_gap_tokens=2.0,
        )
        self.assertFalse(block["length_caveat"].startswith("NOT"))
        self.assertEqual(block["auc_positive_vs_other"], 1.0)

    def test_the_arm_token_means_are_published_next_to_the_auc(self) -> None:
        block = _contrast(
            self._rows([(1.0, 100.0), (1.0, 104.0)]),
            self._rows([(1.0, 80.0), (1.0, 76.0)]),
            positive_label="hostile",
            other_label="counterfactual",
        )
        self.assertEqual(block["mean_prompt_tokens_positive"], 102.0)
        self.assertEqual(block["mean_prompt_tokens_other"], 78.0)
        self.assertEqual(block["token_gap_positive_minus_other"], 24.0)


class FamilyRoundRobinTests(unittest.TestCase):
    """A truncated behavioural readout must not be one family."""

    @staticmethod
    def _rows() -> list[dict[str, str]]:
        return [
            {"case_id": f"{family}_{index}", "family": family}
            for family in ("alpha", "beta", "gamma")
            for index in range(5)
        ]

    def test_a_truncated_prefix_covers_every_family(self) -> None:
        ordered = _family_round_robin(self._rows())
        families = {row["family"] for row in ordered[:6]}
        self.assertEqual(families, {"alpha", "beta", "gamma"})
        # Alphabetical case_id order, which this replaced, takes whole families.
        alphabetical = sorted(self._rows(), key=lambda row: row["case_id"])[:6]
        self.assertEqual({row["family"] for row in alphabetical}, {"alpha", "beta"})

    def test_nothing_is_dropped_or_duplicated(self) -> None:
        ordered = _family_round_robin(self._rows())
        self.assertEqual(len(ordered), 15)
        self.assertEqual({row["case_id"] for row in ordered}, {r["case_id"] for r in self._rows()})

    def test_the_order_is_deterministic_regardless_of_input_order(self) -> None:
        rows = self._rows()
        first = [row["case_id"] for row in _family_round_robin(rows)]
        second = [row["case_id"] for row in _family_round_robin(list(reversed(rows)))]
        self.assertEqual(first, second)

    def test_uneven_families_do_not_lose_their_tail(self) -> None:
        rows = [
            {"case_id": "a_1", "family": "alpha"},
            {"case_id": "a_2", "family": "alpha"},
            {"case_id": "a_3", "family": "alpha"},
            {"case_id": "b_1", "family": "beta"},
        ]
        ordered = [row["case_id"] for row in _family_round_robin(rows)]
        self.assertEqual(ordered, ["a_1", "b_1", "a_2", "a_3"])

    def test_an_empty_input_is_empty(self) -> None:
        self.assertEqual(_family_round_robin([]), [])


class ErrorAuditAcceptanceTests(unittest.TestCase):
    """F5 used to be the one criterion no measurement could move."""

    def test_a_positive_threshold_is_tested(self) -> None:
        entry = _error_audit_acceptance(n_discovery_positive=40, threshold=1.94)
        self.assertEqual(entry["status"], TESTED)
        self.assertIn("BY CONSTRUCTION", entry["reason"])

    def test_a_zero_threshold_is_a_proxy_not_a_pass(self) -> None:
        # A TopK SAE writes exactly 0.0 off its support, so the 25th percentile of the
        # discovery positives IS 0.0 once a quarter of them are silent. The audit that
        # runs then is a fire / no-fire count, not a calibrated threshold audit.
        entry = _error_audit_acceptance(n_discovery_positive=42, threshold=0.0)
        self.assertEqual(entry["status"], PROXY)
        self.assertIn("DEGENERATE", entry["reason"])

    def test_no_discovery_positives_is_not_tested(self) -> None:
        entry = _error_audit_acceptance(n_discovery_positive=0, threshold=0.0)
        self.assertEqual(entry["status"], NOT_TESTED)
        self.assertIn("NOT tested", entry["reason"])

    def test_the_preflight_call_says_it_is_a_plan(self) -> None:
        entry = _error_audit_acceptance(n_discovery_positive=40, threshold=None)
        self.assertEqual(entry["status"], TESTED)
        self.assertIn("planned", entry["reason"])

    def test_the_detection_rule_is_stated_in_the_artifact(self) -> None:
        entry = _error_audit_acceptance(n_discovery_positive=40, threshold=1.0)
        self.assertIn("activation > 0", entry["detection_rule"])

    def test_strict_arms_can_now_abort_on_a_degenerate_audit(self) -> None:
        entry = _error_audit_acceptance(n_discovery_positive=8, threshold=0.0)
        self.assertEqual(len(_strict_failures({"F5_error_audit": entry}, ["F5_error_audit"])), 1)


class AcceptanceTests(unittest.TestCase):
    BASE = {
        "positive_condition": "hostile",
        "negative_condition": "neutral",
        "lexical_condition": "counterfactual",
        "lexical_overlap": 0.96,
        "min_lexical_overlap": 0.8,
        "n_paraphrase_groups": 0,
        "n_reference_control": 30,
        "n_reference_lure": 30,
        "reference_control_overlap": 0.11,
        "reference_control_length_gap_words": 26.4,
        "length_gap_words": 15.3,
        "max_length_gap_words": 2.0,
        "n_discovery_positive": 40,
    }

    def test_missing_paraphrase_arm_is_marked_not_tested(self) -> None:
        acceptance = _acceptance(**self.BASE)
        entry = acceptance["F3_paraphrase_invariance"]
        self.assertEqual(entry["status"], NOT_TESTED)
        self.assertIn("NOT tested", entry["reason"])

    def test_length_gap_downgrades_the_matched_control_to_a_proxy(self) -> None:
        acceptance = _acceptance(**self.BASE)
        entry = acceptance["F4_matched_control"]
        self.assertEqual(entry["status"], PROXY)
        self.assertEqual(entry["length_gap_words"], 15.3)
        self.assertIn("sentence count", entry["reason"])

    def test_a_missing_arm_is_not_tested_rather_than_a_proxy(self) -> None:
        acceptance = _acceptance(**{**self.BASE, "length_gap_words": None})
        self.assertEqual(acceptance["F4_matched_control"]["status"], NOT_TESTED)

    def test_a_genuinely_matched_control_is_tested(self) -> None:
        acceptance = _acceptance(**{**self.BASE, "length_gap_words": 0.4})
        self.assertEqual(acceptance["F4_matched_control"]["status"], TESTED)

    def test_lexical_injection_is_never_more_than_a_proxy(self) -> None:
        acceptance = _acceptance(**{**self.BASE, "lexical_overlap": 1.0})
        self.assertEqual(acceptance["F1_lexical_injection"]["status"], PROXY)

    def test_a_low_overlap_arm_is_not_even_a_proxy(self) -> None:
        acceptance = _acceptance(**{**self.BASE, "lexical_overlap": 0.56})
        self.assertEqual(acceptance["F1_lexical_injection"]["status"], NOT_TESTED)

    def test_missing_reference_controls_are_not_tested(self) -> None:
        acceptance = _acceptance(**{**self.BASE, "n_reference_control": 0})
        self.assertEqual(acceptance["F2_template_control"]["status"], NOT_TESTED)

    def test_the_template_control_is_capped_at_proxy_by_row_count_alone(self) -> None:
        # The status used to be `TESTED if n_reference_control else PROXY` -- a row
        # count. Any reference set shipping any control prompt earned the strongest
        # label this job has, whatever its task, length or vocabulary.
        acceptance = _acceptance(**{**self.BASE, "n_reference_control": 5000})
        self.assertEqual(acceptance["F2_template_control"]["status"], PROXY)

    def test_the_template_control_publishes_the_numbers_that_grade_it(self) -> None:
        entry = _acceptance(**self.BASE)["F2_template_control"]
        self.assertEqual(entry["lexical_overlap_with_positive"], 0.11)
        self.assertEqual(entry["length_gap_words"], 26.4)
        self.assertIn("DIFFERENT TASK", entry["reason"])
        # Its length gap is larger than the one that demotes F4, and the artifact says so.
        self.assertGreater(abs(entry["length_gap_words"]), abs(self.BASE["length_gap_words"]))

    def test_the_error_audit_is_derived_not_asserted(self) -> None:
        degenerate = _acceptance(**{**self.BASE, "error_threshold": 0.0})
        self.assertEqual(degenerate["F5_error_audit"]["status"], PROXY)
        healthy = _acceptance(**{**self.BASE, "error_threshold": 2.1})
        self.assertEqual(healthy["F5_error_audit"]["status"], TESTED)

    def test_strict_arms_reports_every_untested_requirement(self) -> None:
        acceptance = _acceptance(**self.BASE)
        failures = _strict_failures(
            acceptance, ["F3_paraphrase_invariance", "F5_error_audit", "F9_nonsense"]
        )
        self.assertEqual(len(failures), 2)
        self.assertTrue(failures[0].startswith("F3_paraphrase_invariance"))
        self.assertIn("unknown criterion", failures[1])

    def test_no_strict_arms_means_no_failures(self) -> None:
        self.assertEqual(_strict_failures(_acceptance(**self.BASE), []), [])


class SplitMapTests(unittest.TestCase):
    """The audit is only non-circular if the threshold never sees the held-out scenario."""

    def _cases(self) -> list[LureCase]:
        cases: list[LureCase] = []
        for index in range(12):
            for condition in ("hostile", "neutral"):
                cases.append(
                    LureCase(
                        case_id=f"scenario_{index}_{condition}",
                        family="f",
                        prompt="q\nAnswer:",
                        correct_answer=" a",
                        lure_answer=" b",
                        pair_id=f"scenario_{index}",
                        condition=condition,
                    )
                )
        return cases

    def test_both_conditions_of_one_scenario_land_on_the_same_side(self) -> None:
        cases = self._cases()
        split = _split_map(cases, unit="pair_id", train_frac=0.6, seed=0)
        for index in range(12):
            self.assertEqual(
                split[f"scenario_{index}_hostile"],
                split[f"scenario_{index}_neutral"],
                msg=f"scenario_{index} straddles the split",
            )

    def test_both_sides_are_populated(self) -> None:
        split = _split_map(self._cases(), unit="pair_id", train_frac=0.6, seed=0)
        self.assertIn("discovery", split.values())
        self.assertIn("held_out", split.values())

    def test_the_split_is_deterministic(self) -> None:
        first = _split_map(self._cases(), unit="pair_id", train_frac=0.6, seed=0)
        second = _split_map(self._cases(), unit="pair_id", train_frac=0.6, seed=0)
        self.assertEqual(first, second)

    def test_missing_pair_ids_fall_back_to_the_case_id(self) -> None:
        cases = [
            LureCase(
                case_id=f"case_{index}",
                family="f",
                prompt="q\nAnswer:",
                correct_answer=" a",
                lure_answer=" b",
            )
            for index in range(8)
        ]
        split = _split_map(cases, unit="pair_id", train_frac=0.6, seed=0)
        self.assertEqual(len(split), 8)

    def test_an_unknown_split_unit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _split_map(self._cases(), unit="family", train_frac=0.6, seed=0)


# ------------------------------------------------------------ end-to-end artifact

FEATURE = 81663

_MARGIN = SimpleNamespace(
    as_row=lambda: {
        "correct_answer": " a",
        "lure_answer": " b",
        "correct_logprob": -3.0,
        "lure_logprob": -1.0,
        "margin_lure_minus_correct": 2.0,
        "correct_mean_logprob": -1.5,
        "lure_mean_logprob": -0.5,
        "mean_margin_lure_minus_correct": 1.0,
    }
)


def _length_activation(prompt: str) -> float:
    """Mostly prompt length plus a hostile bump: the confound, in feature form."""

    return 0.02 * len(prompt.split()) + (0.7 if "would normally" in prompt else 0.0)


def _topk_sparse_activation(prompt: str) -> float:
    """What a real TopK SAE feature looks like: exactly 0.0 off its support."""

    return 2.5 if ("would normally" in prompt and "car wash" in prompt) else 0.0


def _run_job(
    tmp: Path,
    *,
    activation: Callable[[str], float] = _length_activation,
    config_text: str | None = None,
) -> tuple[dict, dict, list[dict]]:
    config_path = CONFIG
    if config_text is not None:
        config_path = tmp / "config_under_test.toml"
        config_path.write_text(config_text, encoding="utf-8")

    def residual(prompts, *args, **kwargs):
        prompt = prompts[0] if not isinstance(prompts, str) else prompts
        return torch.tensor([[activation(prompt)]])

    with (
        patch.object(
            job,
            "load_qwen_language_model",
            lambda *a, **k: SimpleNamespace(
                tokenizer=SimpleNamespace(encode=lambda text, **kw: [0] * len(text.split()))
            ),
        ),
        patch.object(job, "load_qwen_scope_sae", lambda *a, **k: SimpleNamespace(top_k=50)),
        patch.object(job, "capture_layer_residuals", lambda lm, p, layer, **kw: residual(p)),
        patch.object(job, "qwen_scope_sparse_feature_values", lambda r, s, ids: r[:, :1]),
        patch.object(job, "qwen_scope_feature_preactivations", lambda r, s, ids: r[:, :1] + 0.5),
        patch.object(
            job,
            "active_prompt_features",
            # A feature that did not fire is not in the TopK -- the fake has to respect
            # that, or the degenerate-threshold test would be reading a fiction.
            lambda r, s, *, top_n=20: (
                ([(FEATURE, float(r.reshape(-1)[0]))] if float(r.reshape(-1)[0]) > 0 else [])
                + [(i, 0.1) for i in range(top_n - 1)]
            ),
        ),
        patch.object(job, "answer_logprob_margin", lambda *a, **k: _MARGIN),
    ):
        # The job narrates its progress on stdout; the assertions read the files.
        with contextlib.redirect_stdout(io.StringIO()):
            run_dir = job.run(config_path, tmp / "out")
    summary = json.loads((run_dir / "falsification_summary.json").read_text(encoding="utf-8"))
    errors = json.loads((run_dir / "falsification_errors.json").read_text(encoding="utf-8"))
    with (run_dir / "falsification_activations.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return summary, errors, rows


class _ArtifactCase(unittest.TestCase):
    """Runs the job once per class with the model faked, then reads the files."""

    ACTIVATION: Callable[[str], float] = staticmethod(_length_activation)
    CONFIG_EDITS: tuple[tuple[str, str], ...] = ()

    summary: dict
    errors: dict
    rows: list[dict]

    @classmethod
    def setUpClass(cls) -> None:
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        config_text = None
        if cls.CONFIG_EDITS:
            config_text = CONFIG.read_text(encoding="utf-8")
            for old, new in cls.CONFIG_EDITS:
                assert config_text.count(old) == 1, old
                config_text = config_text.replace(old, new)
        cls.summary, cls.errors, cls.rows = _run_job(
            Path(tmp.name), activation=cls.ACTIVATION, config_text=config_text
        )


class ArtifactTests(_ArtifactCase):
    """What the artifact is allowed to claim, exercised end to end with the model faked.

    The audit findings this guards are all "the file says something the data does not
    support", so they can only be caught by reading the file the job actually writes.
    """

    def test_the_artifact_says_paraphrase_invariance_is_not_tested(self) -> None:
        # The audit's most serious finding: this block used to report between-family
        # variance under the name "paraphrase".
        self.assertEqual(self.summary["paraphrase"]["status"], NOT_TESTED)
        self.assertEqual(self.summary["paraphrase"]["n_groups"], 0)
        self.assertNotIn("per_group_spread", self.summary["paraphrase"])
        self.assertTrue(any("paraphrase" in line for line in self.summary["caveats"]))

    def test_across_template_spread_is_labelled_as_between_family(self) -> None:
        variation = self.summary["template_variation"]
        self.assertTrue(variation["template_id_is_a_family_label"])
        self.assertIn("NOT paraphrase", variation["note"])

    def test_the_length_confound_is_visible_next_to_the_raw_gap(self) -> None:
        confound = self.summary["length_confound"]
        self.assertGreater(confound["token_gap_positive_minus_negative"], 5.0)
        for condition in ("hostile", "neutral"):
            self.assertGreater(confound["profile"][condition]["tokens"]["mean"], 0)
        # The fake feature is a length feature, so the raw AUC must look strong and the
        # length-matched one must not. If this ever inverts, the adjustment is broken.
        self.assertGreater(self.summary["structure_auc"], 0.8)
        self.assertLess(confound["length_matched"]["auc"], 0.7)
        self.assertGreater(confound["length_matched"]["n_pairs"], 0)
        self.assertLess(abs(confound["partial_corr_condition_given_length"]), 0.3)

    def test_every_published_contrast_carries_its_own_length_adjustment(self) -> None:
        # The length machinery used to be wired to hostile-vs-neutral alone, so these
        # three published raw AUCs with nothing attached -- and their gaps are larger
        # than the flagged one, one of them in the opposite direction.
        for name in ("lexical_proxy", "template_control", "crt_lure_transfer"):
            block = self.summary[name]
            with self.subTest(contrast=name):
                self.assertIsNotNone(block["mean_prompt_tokens_positive"])
                self.assertIsNotNone(block["mean_prompt_tokens_other"])
                self.assertIsNotNone(block["token_gap_positive_minus_other"])
                self.assertIn("length_matched", block)
                self.assertTrue(block["length_caveat"].startswith("NOT length-controlled"))
                # ...and the caveat is in the summary, not only inside the block.
                self.assertTrue(
                    any(line.startswith(f"{name}: NOT") for line in self.summary["caveats"])
                )

    def test_the_lexical_arm_does_not_claim_to_be_the_closest_by_wording(self) -> None:
        block = self.summary["lexical_proxy"]
        overlaps = block["per_condition_lexical_overlap"]
        self.assertLess(overlaps["counterfactual"], overlaps["explicit"])
        self.assertEqual(block["highest_overlap_arm"], "explicit")
        self.assertIn("STRUCTURE", block["arm_chosen_by"])

    def test_the_template_control_is_a_proxy_not_a_pass(self) -> None:
        self.assertEqual(self.summary["template_control"]["status"], PROXY)
        self.assertEqual(self.summary["acceptance"]["F2_template_control"]["status"], PROXY)

    def test_every_error_record_can_be_read_on_its_own(self) -> None:
        records = self.errors["false_negatives"] + self.errors["false_positives"]
        self.assertTrue(records)
        for record in records:
            self.assertTrue(record["prompt"].strip())  # F5.d
            self.assertEqual(record["topk_size"], 50)  # F5.g
            self.assertIsNotNone(record["topk_rank"])
            self.assertEqual(record["topk_status"], "measured")
            self.assertIn("margin_status", record)  # F5.h
            if record["margin_status"] == "measured":
                self.assertIn("margin_lure_minus_correct", record)

    def test_the_threshold_still_comes_from_the_discovery_split_only(self) -> None:
        # The one property the audit called good. Recomputed from the artifact itself.
        discovery = sorted(
            float(row["activation"])
            for row in self.rows
            if row["condition"] == "hostile" and row["split"] == "discovery"
        )
        held_out = [
            row for row in self.rows if row["condition"] == "hostile" and row["split"] == "held_out"
        ]
        self.assertTrue(discovery and held_out)
        self.assertAlmostEqual(self.errors["threshold"], discovery[len(discovery) // 4], places=9)
        self.assertIn("discovery", self.errors["threshold_source"])

    def test_the_error_counts_separate_the_finding_from_the_definition(self) -> None:
        # The threshold IS the 25th percentile of the discovery positives, so roughly a
        # quarter of those are false negatives however good the feature is. A single
        # pooled total quotes that definition as a result.
        summary = self.summary
        self.assertEqual(
            summary["n_false_negatives_all_splits"],
            summary["n_false_negatives_held_out"] + summary["n_false_negatives_discovery"],
        )
        self.assertGreater(summary["n_false_negatives_discovery"], 0)
        self.assertNotIn("n_false_negatives", summary)
        self.assertNotIn("n_false_positives", summary)
        self.assertIn("definition, not a finding", summary["error_count_note"])
        self.assertIn("BY CONSTRUCTION", self.errors["counts_note"])

    def test_the_behavioural_readout_is_spread_across_families(self) -> None:
        # Alphabetical case_id truncation took whole families, because case_ids here are
        # family-prefixed. Every family the errors touch has to be represented.
        error_families = {
            record["family"]
            for record in self.errors["false_negatives"] + self.errors["false_positives"]
            if record["family"]
        }
        self.assertGreater(len(error_families), 1)
        self.assertEqual(set(self.summary["margin_families_measured"]), error_families)
        self.assertIn("round-robin", self.summary["margin_selection"])

    def test_reference_lure_and_reference_control_are_separate_arms(self) -> None:
        conditions = {row["condition"] for row in self.rows}
        self.assertIn(job.REFERENCE_CONTROL, conditions)
        self.assertIn(job.REFERENCE_LURE, conditions)
        self.assertIn("NOT a template control", self.summary["crt_lure_transfer"]["note"])

    def test_the_topk_measurement_is_reported_as_measured(self) -> None:
        self.assertEqual(self.summary["condition_in_topk_status"], "measured")
        self.assertEqual(self.summary["condition_in_topk_rate"]["hostile"], 1.0)


class UnmeasuredTopKTests(_ArtifactCase):
    """Switching a measurement off must not publish it as a hard negative."""

    CONFIG_EDITS = (("record_topk_rank = true", "record_topk_rank = false"),)

    def test_the_in_topk_rate_is_null_rather_than_zero(self) -> None:
        # `in_topk` was `rank is not None`, so turning the probe off published
        # condition_in_topk_rate = 0.0 on every arm: "the feature was outside the SAE
        # TopK on 100% of items", the strongest falsification claim this job can make,
        # produced by a config flag rather than by data.
        self.assertIsNone(self.summary["condition_in_topk_rate"])
        self.assertIn("NOT TESTED", self.summary["condition_in_topk_status"])

    def test_the_csv_column_is_blank_not_false(self) -> None:
        self.assertEqual({row["in_topk"] for row in self.rows}, {""})
        self.assertEqual({row["topk_rank"] for row in self.rows}, {""})

    def test_the_feature_still_fires_on_every_arm(self) -> None:
        # The contradiction the old artifact carried: fires on 100%, in the TopK on 0%.
        self.assertEqual(self.summary["condition_fire_rate"]["hostile"], 1.0)

    def test_the_error_records_say_the_rank_was_not_measured(self) -> None:
        records = self.errors["false_negatives"] + self.errors["false_positives"]
        self.assertTrue(records)
        for record in records:
            self.assertIsNone(record["topk_rank"])
            self.assertIn("NOT MEASURED", record["topk_status"])

    def test_the_summary_caveats_carry_the_absence(self) -> None:
        self.assertTrue(any("record_topk_rank" in line for line in self.summary["caveats"]))


class DegenerateThresholdTests(_ArtifactCase):
    """A TopK feature that is mostly silent drives the discovery quantile to 0.0."""

    ACTIVATION = staticmethod(_topk_sparse_activation)

    def test_the_threshold_really_does_land_on_zero(self) -> None:
        self.assertEqual(self.errors["threshold"], 0.0)
        self.assertLess(self.summary["condition_fire_rate"]["hostile"], 0.1)

    def test_silent_controls_are_not_counted_as_false_positives(self) -> None:
        # With a bare `>= threshold` every non-firing control satisfied `>= 0.0`: the
        # run reported 90 false positives, all of them items where the feature never
        # fired at all, and then spent two 27B forwards on each of them.
        self.assertEqual(self.summary["n_false_positives_all_splits"], 0)
        for record in self.errors["false_positives"]:
            self.assertGreater(float(record["activation"]), 0.0)

    def test_silent_positives_are_counted_as_false_negatives(self) -> None:
        # And `< 0.0` was never true, so the run reported 0 false negatives while the
        # feature was silent on 59 of 60 hostile items.
        self.assertEqual(self.summary["n_false_negatives_all_splits"], 59)
        self.assertGreater(self.summary["n_false_negatives_held_out"], 0)

    def test_the_criterion_downgrades_itself(self) -> None:
        entry = self.summary["acceptance"]["F5_error_audit"]
        self.assertEqual(entry["status"], PROXY)
        self.assertIn("DEGENERATE", entry["reason"])
        self.assertEqual(self.errors["acceptance_status"], PROXY)
        self.assertTrue(any("DEGENERATE" in line for line in self.summary["caveats"]))

    def test_the_detection_rule_is_published(self) -> None:
        self.assertIn("activation > 0", self.errors["detection_rule"])


class MissingControlArmTests(_ArtifactCase):
    """Pointing the config at a condition the dataset does not have."""

    CONFIG_EDITS = (('negative_condition = "neutral"', 'negative_condition = "absent"'),)

    def test_no_contrast_number_is_published_for_an_arm_that_does_not_exist(self) -> None:
        # `_mean([])` is 0.0, so structure_gap used to publish the hostile mean as the
        # gap between two arms while structure_auc correctly reported null.
        self.assertIsNone(self.summary["structure_auc"])
        self.assertIsNone(self.summary["structure_gap"])

    def test_the_partial_correlation_is_null_rather_than_a_confident_zero(self) -> None:
        # "the condition effect is exactly zero once length is controlled" is the most
        # damaging conclusion this job can print, and it was printed with no control arm.
        confound = self.summary["length_confound"]
        self.assertIsNone(confound["partial_corr_condition_given_length"])
        self.assertIsNone(confound["corr_condition_vs_length"])
        self.assertIsNone(confound["corr_activation_vs_prompt_tokens_across_arms"])
        self.assertIsNone(confound["token_gap_positive_minus_negative"])

    def test_the_acceptance_marker_agrees_with_the_missing_numbers(self) -> None:
        self.assertEqual(self.summary["acceptance"]["F4_matched_control"]["status"], NOT_TESTED)


class StrictArmTests(unittest.TestCase):
    def test_strict_arms_aborts_before_the_model_is_loaded(self) -> None:
        # The whole point of the pre-flight: a missing arm costs seconds, not an hour of
        # 27B forwards followed by a number nobody can use.
        config = CONFIG.read_text(encoding="utf-8").replace(
            "strict_arms = []", 'strict_arms = ["F3_paraphrase_invariance"]'
        )
        loaded: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            strict = Path(tmp) / "strict.toml"
            strict.write_text(config, encoding="utf-8")
            with (
                patch.object(job, "load_qwen_language_model", lambda *a, **k: loaded.append("lm")),
                patch.object(job, "load_qwen_scope_sae", lambda *a, **k: loaded.append("sae")),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(RuntimeError) as caught,
            ):
                job.run(strict, Path(tmp) / "out")
        self.assertIn("F3_paraphrase_invariance", str(caught.exception))
        self.assertEqual(loaded, [])

    def test_requiring_the_cross_task_template_control_also_aborts(self) -> None:
        # F2 is capped at proxy, so a config that demands it as tested must not run.
        config = CONFIG.read_text(encoding="utf-8").replace(
            "strict_arms = []", 'strict_arms = ["F2_template_control"]'
        )
        with tempfile.TemporaryDirectory() as tmp:
            strict = Path(tmp) / "strict.toml"
            strict.write_text(config, encoding="utf-8")
            with (
                patch.object(job, "load_qwen_language_model", lambda *a, **k: None),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(RuntimeError) as caught,
            ):
                job.run(strict, Path(tmp) / "out")
        self.assertIn("F2_template_control", str(caught.exception))


class ConfigClaimTests(unittest.TestCase):
    """The config's prose is part of the artifact; it must not assert a measurement."""

    def test_the_lexical_arm_comment_does_not_call_itself_the_closest_match(self) -> None:
        text = CONFIG.read_text(encoding="utf-8")
        self.assertNotIn("Closest thing the dataset has", text)
        self.assertIn("STRUCTURE, not for lexical overlap", text)

    def test_the_reference_comment_does_not_call_itself_the_real_template_control(
        self,
    ) -> None:
        text = CONFIG.read_text(encoding="utf-8")
        self.assertNotIn("Those are the real template control", text)
        self.assertIn("capped at `proxy`", text)


if __name__ == "__main__":
    # LAST line in the file, on purpose. This block used to sit above ArtifactTests, so
    # running the module as __main__ collected only the unit tests and silently skipped
    # every end-to-end artifact test -- and still reported OK.
    unittest.main()
