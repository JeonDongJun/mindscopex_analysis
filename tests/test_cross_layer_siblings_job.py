"""Pure helpers of the cross-layer sibling job: the fourth signal and the clean row.

None of these needs a GPU. All of them are places where a silent mistake would not look
like a crash -- a dropped signal still produces a ranking, a rejected candidate still
produces a CSV, and a fallback still produces a co-ablation -- so they are pinned here.

The fourth signal gets the most attention because it is the newest and because it is the
one that can be wrong in a direction the artifact does not show: ``specificity_corr`` is
an AGREEMENT statistic, and agreement between two anti-specific features is just as high
as agreement between two cue-specific ones.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from experiments.jobs import cross_layer_siblings
from experiments.jobs.cross_layer_siblings import (
    COABLATION_COLUMNS,
    SELECTION_FOUR_SIGNAL,
    SELECTION_NONE,
    SELECTION_THREE_SIGNAL_FALLBACK,
    SELECTION_THREE_SIGNAL_SOURCE_NOT_SPECIFIC,
    SELECTION_THREE_SIGNAL_UNMEASURED,
    SIBLING_COLUMNS,
    SPECIFICITY_ALIGNED,
    SPECIFICITY_GAP_NON_POSITIVE,
    SPECIFICITY_HOSTILE_ARM_NON_POSITIVE,
    SPECIFICITY_UNMEASURED,
    _absolute_row,
    _condition_row,
    _load_splits,
    _margin_row,
    _write_csv,
    rank_siblings_with_specificity,
    score_siblings,
    select_sibling,
    sibling_score_with_specificity,
    specificity_level_verdict,
)
from mindscopex_analysis import stats
from mindscopex_analysis.siblings import sibling_score


@dataclass(frozen=True)
class _Logprob:
    logprob: float


@dataclass(frozen=True)
class _Margin:
    """Stand-in for effects.AnswerMargin, carrying only what the row builders read."""

    margin: float
    correct: _Logprob
    lure: _Logprob


@dataclass(frozen=True)
class _Case:
    case_id: str
    family: str


# Same fixture shape as tests/test_delta_conventions.py: a lure-promoting feature,
# where ablating raises the correct answer, lowers the lure, and so cuts the margin.
# All four numbers are distinct and the two logprob deltas have opposite signs, so no
# rearrangement of the terms can pass by accident.
BASELINE = _Margin(margin=2.5, correct=_Logprob(-3.0), lure=_Logprob(-0.5))
ABLATED = _Margin(margin=1.0, correct=_Logprob(-1.0), lure=_Logprob(-2.0))
CASE = _Case(case_id="target_transport_car_wash_hostile", family="target_transport")


def _candidate(feature: int, **overrides: object) -> dict:
    """A candidate whose levels pass the gate unless a test says otherwise."""

    row = {
        "target_feature": feature,
        "decoder_cosine": 0.6,
        "activation_corr": 0.6,
        "effect_corr": 0.6,
        "specificity_corr": 0.6,
        "mean_effect": 0.4,
        "mean_specificity": 0.2,
    }
    row.update(overrides)
    return row


class FourthSignalTests(unittest.TestCase):
    """The specificity signal has to enter the score, and enter it as a fourth term."""

    def test_equals_the_four_term_geometric_mean(self) -> None:
        cos, act, eff, spec = 0.8, 0.5, 0.4, 0.2
        expected = (cos * act * eff * spec) ** 0.25

        self.assertAlmostEqual(
            sibling_score_with_specificity(cos, act, eff, spec), expected, places=12
        )

    def test_weight_generalizes_the_geometric_mean(self) -> None:
        cos, act, eff, spec = 0.8, 0.5, 0.4, 0.2
        weight = 2.0
        expected = (cos * act * eff * spec**weight) ** (1.0 / (3.0 + weight))

        self.assertAlmostEqual(
            sibling_score_with_specificity(cos, act, eff, spec, specificity_weight=weight),
            expected,
            places=12,
        )

    def test_the_fourth_signal_actually_changes_the_ranking(self) -> None:
        # The whole point: a candidate that wins on geometry and firing but carries no
        # cue-specific effect must lose to a balanced one. Under the three-signal score
        # the geometric candidate wins; under four signals it must not.
        geometric = {"decoder_cosine": 0.95, "activation_corr": 0.9, "effect_corr": 0.6}
        balanced = {"decoder_cosine": 0.55, "activation_corr": 0.5, "effect_corr": 0.5}
        self.assertGreater(
            sibling_score(**geometric),
            sibling_score(**balanced),
            "fixture is wrong: the geometric candidate must win on three signals",
        )

        self.assertLess(
            sibling_score_with_specificity(**geometric, specificity_corr=0.02),
            sibling_score_with_specificity(**balanced, specificity_corr=0.5),
        )

    def test_negative_specificity_cannot_be_cancelled_by_a_strong_cosine(self) -> None:
        # Anti-correlated on the gap means the two features move it in opposite
        # directions -- evidence against the pair being the same feature. Averaging it
        # away would rank such a pair above an unrelated one.
        self.assertEqual(sibling_score_with_specificity(0.99, 0.9, 0.9, -0.8), 0.0)
        self.assertEqual(sibling_score_with_specificity(0.99, 0.9, 0.9, 0.0), 0.0)

    def test_a_dead_three_signal_score_stays_dead(self) -> None:
        self.assertEqual(sibling_score_with_specificity(0.9, -0.5, 0.9, 0.9), 0.0)

    def test_unmeasured_is_not_the_same_as_measured_and_zero(self) -> None:
        # No control condition configured: fall back to the three-signal score rather
        # than scoring 0, otherwise every candidate would be discarded in silence.
        self.assertAlmostEqual(
            sibling_score_with_specificity(0.8, 0.5, 0.4, None),
            sibling_score(0.8, 0.5, 0.4),
            places=12,
        )
        self.assertEqual(sibling_score_with_specificity(0.8, 0.5, 0.4, 0.0), 0.0)

    def test_zero_weight_is_the_three_signal_score(self) -> None:
        self.assertAlmostEqual(
            sibling_score_with_specificity(0.8, 0.5, 0.4, 0.2, specificity_weight=0.0),
            sibling_score(0.8, 0.5, 0.4),
            places=12,
        )

    def test_negative_weight_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sibling_score_with_specificity(0.8, 0.5, 0.4, 0.2, specificity_weight=-1.0)


class LevelGateTests(unittest.TestCase):
    """A correlation says "same contrast"; only the levels say "same direction"."""

    def test_both_levels_positive_is_the_only_aligned_verdict(self) -> None:
        self.assertEqual(specificity_level_verdict(0.4, 0.2), SPECIFICITY_ALIGNED)
        self.assertEqual(specificity_level_verdict(-0.1, 0.2), SPECIFICITY_HOSTILE_ARM_NON_POSITIVE)
        self.assertEqual(specificity_level_verdict(0.0, 0.2), SPECIFICITY_HOSTILE_ARM_NON_POSITIVE)
        self.assertEqual(specificity_level_verdict(0.4, -0.2), SPECIFICITY_GAP_NON_POSITIVE)
        self.assertEqual(specificity_level_verdict(0.4, 0.0), SPECIFICITY_GAP_NON_POSITIVE)

    def test_a_missing_level_is_unmeasured_not_failed(self) -> None:
        self.assertEqual(specificity_level_verdict(None, 0.2), SPECIFICITY_UNMEASURED)
        self.assertEqual(specificity_level_verdict(0.4, None), SPECIFICITY_UNMEASURED)
        # A row round-tripped through CSV carries "" where None was written.
        self.assertEqual(specificity_level_verdict("", ""), SPECIFICITY_UNMEASURED)

    def test_a_non_positive_hostile_arm_cannot_rank_on_its_correlation(self) -> None:
        # The exact candidate the reviewer found in a mocked full run: ablating it
        # INCREASED the lure lead (mean_effect < 0), so its +0.014 gap is a control
        # effect and not a trap effect -- docs/metrics_guide.md's rule. It used to rank
        # ABOVE a candidate whose arms are both positive.
        hostile_arm_negative = _candidate(
            280,
            decoder_cosine=0.42,
            activation_corr=0.31,
            effect_corr=0.28,
            specificity_corr=0.293,
            mean_effect=-0.329,
            mean_specificity=0.014,
        )
        aligned = _candidate(
            999,
            decoder_cosine=0.30,
            activation_corr=0.30,
            effect_corr=0.30,
            specificity_corr=0.30,
        )

        ranked = rank_siblings_with_specificity([hostile_arm_negative, aligned], min_score=0.05)

        self.assertEqual([row["target_feature"] for row in ranked], [999])

    def test_an_anti_specific_candidate_cannot_ride_a_high_correlation(self) -> None:
        # Both members anti-specific: the ablation moves the neutral twin more than the
        # hostile item, on both features, so their gap vectors correlate at +0.9 while
        # neither is cue-specific. Correlation alone would promote it to rank 1.
        anti = _candidate(
            7,
            decoder_cosine=0.9,
            activation_corr=0.8,
            effect_corr=0.8,
            specificity_corr=0.9,
            mean_specificity=-0.4,
        )

        self.assertEqual(rank_siblings_with_specificity([anti], min_score=0.05), [])

    def test_a_rejected_candidate_stays_in_the_artifact_with_its_reason(self) -> None:
        # Dropped from the RANKING, not from the record: an artifact that just gets
        # shorter tells the reader nothing about what was excluded or why.
        rejected = _candidate(280, mean_effect=-0.329)
        rows = score_siblings([rejected, _candidate(999)])

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["specificity_verdict"], SPECIFICITY_HOSTILE_ARM_NON_POSITIVE)
        self.assertIsNone(
            rows[0]["combined_score"],
            "a gated-out candidate must carry no score; 0.0 would read as 'scored badly'",
        )
        self.assertEqual(rows[1]["specificity_verdict"], SPECIFICITY_ALIGNED)
        self.assertGreater(rows[1]["combined_score"], 0.0)

    def test_a_failed_source_scores_everyone_on_three_rather_than_blanking_them(self) -> None:
        # When the source's own levels fail, the fourth signal is not in use at all, so
        # gating candidates on it would blank every combined_score in the CSV while
        # coablation_summary.json still quoted one. The two artifacts have to agree.
        rows = score_siblings(
            [_candidate(1), _candidate(2, mean_specificity=-0.3)],
            source_verdict=SPECIFICITY_GAP_NON_POSITIVE,
        )

        three = sibling_score(0.6, 0.6, 0.6)
        self.assertAlmostEqual(rows[0]["combined_score"], three, places=12)
        self.assertAlmostEqual(rows[1]["combined_score"], three, places=12)
        # Each candidate still carries its own verdict as a record.
        self.assertEqual(rows[1]["specificity_verdict"], SPECIFICITY_GAP_NON_POSITIVE)

    def test_the_gate_does_not_apply_when_the_signal_was_never_measured(self) -> None:
        # Three-signal runs keep their pre-existing behaviour exactly: the level gate
        # belongs to the fourth signal, so it must not start filtering runs that have no
        # control condition at all.
        unmeasured = _candidate(1, specificity_corr=None, mean_effect=-1.0, mean_specificity=None)
        rows = score_siblings([unmeasured])

        self.assertEqual(rows[0]["specificity_verdict"], SPECIFICITY_UNMEASURED)
        self.assertAlmostEqual(rows[0]["combined_score"], sibling_score(0.6, 0.6, 0.6), places=12)


class RankingTests(unittest.TestCase):
    def test_sorts_best_first_and_keeps_the_other_columns(self) -> None:
        ranked = rank_siblings_with_specificity(
            [
                _candidate(1, specificity_corr=0.1),
                _candidate(2, specificity_corr=0.9),
                _candidate(3, specificity_corr=0.5),
            ],
            min_score=-1.0,
        )

        self.assertEqual([row["target_feature"] for row in ranked], [2, 3, 1])
        self.assertEqual(ranked[0]["decoder_cosine"], 0.6)
        self.assertGreater(ranked[0]["combined_score"], ranked[-1]["combined_score"])

    def test_min_score_drops_rather_than_ranks(self) -> None:
        # "no sibling was found" has to stay distinguishable from "the best was poor".
        weak = _candidate(1, specificity_corr=0.000001)
        self.assertEqual(rank_siblings_with_specificity([weak], min_score=0.05), [])

    def test_blank_specificity_reads_as_unmeasured(self) -> None:
        # A row round-tripped through CSV carries "" where None was written.
        ranked = rank_siblings_with_specificity(
            [_candidate(1, specificity_corr=None)], min_score=-1.0
        )
        from_csv = rank_siblings_with_specificity(
            [_candidate(1, specificity_corr="")], min_score=-1.0
        )

        self.assertAlmostEqual(ranked[0]["combined_score"], from_csv[0]["combined_score"], 12)


class SelectionTests(unittest.TestCase):
    """A noisy fourth signal must degrade the claim, never cancel the experiment."""

    def _near_duplicate_layer(self, spec: float) -> list[dict]:
        # The 12 candidates are the cosine-nearest neighbours of ONE source direction,
        # so their per-item gap vectors are near-duplicates: effectively one draw. When
        # that one draw lands negative, all twelve score exactly 0.0 at once.
        return [_candidate(index, specificity_corr=spec) for index in range(12)]

    def test_a_negative_draw_falls_back_instead_of_ending_the_run(self) -> None:
        candidates = self._near_duplicate_layer(-0.05)
        self.assertEqual(
            rank_siblings_with_specificity(candidates, min_score=0.05),
            [],
            "fixture is wrong: the four-signal ranking must be empty here",
        )

        best, provenance = select_sibling(
            candidates, min_score=0.05, source_verdict=SPECIFICITY_ALIGNED
        )

        self.assertIsNotNone(best, "the co-ablation, DiD and repair must still be run")
        self.assertEqual(provenance["selection_rule"], SELECTION_THREE_SIGNAL_FALLBACK)
        self.assertFalse(provenance["specificity_signal_used"])
        self.assertEqual(provenance["n_cleared_min_score_four_signal"], 0)
        self.assertEqual(provenance["n_cleared_min_score_three_signal"], 12)
        self.assertIn("four signals", provenance["degraded_reason"])

    def test_a_real_absence_is_not_reported_as_a_fallback(self) -> None:
        # Nothing clears on three signals either, so this is genuinely no sibling --
        # the one case where "no sibling found" is the honest reading.
        dead = [_candidate(1, activation_corr=-0.6)]

        best, provenance = select_sibling(dead, min_score=0.05, source_verdict=SPECIFICITY_ALIGNED)

        self.assertIsNone(best)
        self.assertEqual(provenance["selection_rule"], SELECTION_NONE)
        self.assertEqual(provenance["n_cleared_min_score_three_signal"], 0)
        self.assertIn("absence of a sibling", provenance["degraded_reason"])

    def test_four_signals_are_used_and_recorded_when_they_work(self) -> None:
        best, provenance = select_sibling(
            [_candidate(1, specificity_corr=0.2), _candidate(2, specificity_corr=0.9)],
            min_score=0.05,
            source_verdict=SPECIFICITY_ALIGNED,
        )

        self.assertEqual(best["target_feature"], 2)
        self.assertEqual(provenance["selection_rule"], SELECTION_FOUR_SIGNAL)
        self.assertTrue(provenance["specificity_signal_used"])
        self.assertEqual(provenance["degraded_reason"], "")

    def test_a_source_that_is_not_cue_specific_disables_the_signal_globally(self) -> None:
        # Agreement with a source whose own gap is non-positive is agreement with a
        # contrast that is not there. Reporting specificity_corr = 0.9 for such a pair
        # would read as "the fourth signal confirms this pair is cue-specific".
        best, provenance = select_sibling(
            [_candidate(1, specificity_corr=0.9)],
            min_score=0.05,
            source_verdict=SPECIFICITY_GAP_NON_POSITIVE,
        )

        self.assertIsNotNone(best)
        self.assertEqual(provenance["selection_rule"], SELECTION_THREE_SIGNAL_SOURCE_NOT_SPECIFIC)
        self.assertFalse(provenance["specificity_signal_used"])
        self.assertIsNone(provenance["n_cleared_min_score_four_signal"])
        self.assertAlmostEqual(best["combined_score"], sibling_score(0.6, 0.6, 0.6), places=12)

    def test_an_unmeasured_run_says_unmeasured_not_fallback(self) -> None:
        best, provenance = select_sibling(
            [_candidate(1, specificity_corr=None)],
            min_score=0.05,
            source_verdict=SPECIFICITY_UNMEASURED,
        )

        self.assertIsNotNone(best)
        self.assertEqual(provenance["selection_rule"], SELECTION_THREE_SIGNAL_UNMEASURED)
        self.assertFalse(provenance["specificity_signal_used"])

    def test_the_provenance_counts_what_the_gate_removed(self) -> None:
        best, provenance = select_sibling(
            [_candidate(1), _candidate(2, mean_effect=-0.3), _candidate(3, mean_specificity=-0.3)],
            min_score=0.05,
            source_verdict=SPECIFICITY_ALIGNED,
        )

        self.assertEqual(best["target_feature"], 1)
        self.assertEqual(provenance["n_candidates"], 3)
        self.assertEqual(provenance["n_specificity_level_rejected"], 2)
        self.assertEqual(provenance["source_specificity_verdict"], SPECIFICITY_ALIGNED)


class PairedStatisticTests(unittest.TestCase):
    """One definition of a p-value for the whole study, not a private copy per job."""

    def test_the_job_has_no_private_sign_flip_copy(self) -> None:
        self.assertFalse(
            hasattr(cross_layer_siblings, "_sign_flip_p"),
            "the private b/draws copy is superseded by stats.paired_summary",
        )
        self.assertIs(cross_layer_siblings.paired_summary, stats.paired_summary)

    def test_a_unanimous_effect_never_reports_p_exactly_zero(self) -> None:
        # b/draws returns 0.0 here, which claims more resolution than 20k draws have.
        # At n = 30 only 2 of 2**30 sign assignments reach the observation, so no draw
        # does: b = 0, where b/draws would print 0.0 and claim a resolution 20k draws
        # cannot have. (b+1)/(draws+1) reports the floor instead.
        summary = cross_layer_siblings.paired_summary([0.5] * 30, seed=0)

        self.assertGreater(summary["p"], 0.0)
        self.assertAlmostEqual(summary["p"], 1.0 / 20001.0, places=12)
        # And it never travels without the interval and the sign count.
        self.assertEqual(summary["n_positive"], 30)
        self.assertLessEqual(summary["ci_low"], summary["mean"])
        self.assertGreaterEqual(summary["ci_high"], summary["mean"])

    def test_one_dominating_item_is_visible_in_the_summary(self) -> None:
        # The exact failure metrics_guide.md warns about: a mean carried by a single
        # outlier. Undetectable from n/mean/p alone; obvious from n_positive and the CI.
        summary = cross_layer_siblings.paired_summary([9.0] + [-0.1] * 11, seed=0)

        self.assertEqual(summary["n_positive"], 1)
        self.assertLess(summary["ci_low"], 0.0)


class SplitTests(unittest.TestCase):
    """The fourth signal is opt-in, because most datasets have no no-cue twin."""

    def test_a_dataset_without_neutral_twins_still_runs_on_three_signals(self) -> None:
        # goal_affordance_traps_v21 has conditions absent/offered/immediate/explicit/
        # counterfactual and no `_neutral` twin at all. With control_condition defaulting
        # to "neutral" this raised ValueError before the model was even loaded.
        splits = _load_splits(
            {
                "data": {
                    "dataset": "goal_affordance_traps_v21",
                    "conditions": ["immediate"],
                    "max_match_items": 4,
                    "max_test_items": 4,
                }
            }
        )

        self.assertEqual(splits["control_condition"], "")
        self.assertEqual(splits["match_controls"], [])
        self.assertTrue(splits["match"])

    def test_opting_in_pairs_every_matching_item_with_its_twin(self) -> None:
        splits = _load_splits(
            {
                "data": {
                    "dataset": "goal_affordance_traps_v1",
                    "conditions": ["hostile"],
                    "control_condition": "neutral",
                    "max_match_items": 4,
                    "max_test_items": 4,
                }
            }
        )

        self.assertEqual(splits["control_condition"], "neutral")
        self.assertEqual(len(splits["match_controls"]), len(splits["match"]))
        self.assertTrue(splits["match_controls"])

    def test_opting_in_without_conditions_says_what_to_do_about_it(self) -> None:
        with self.assertRaises(ValueError) as raised:
            _load_splits(
                {"data": {"dataset": "goal_affordance_traps_v1", "control_condition": "neutral"}}
            )

        self.assertIn("[data].conditions", str(raised.exception))


class CleanRowTests(unittest.TestCase):
    """`clean` is one of the seven conditions, and it is the anchor for the other six."""

    def test_clean_row_has_zero_deltas_and_the_absolute_margin(self) -> None:
        row = _condition_row(
            CASE,
            target_layer=31,
            target_feature=4242,
            condition="clean",
            draw=-1,
            margin=BASELINE,
            baseline=BASELINE,
        )

        self.assertEqual(row["condition"], "clean")
        self.assertEqual(row["margin_delta"], 0.0)
        self.assertEqual(row["correct_logprob_delta"], 0.0)
        self.assertEqual(row["lure_logprob_delta"], 0.0)
        # The part that no delta can recover.
        self.assertEqual(row["margin"], 2.5)
        self.assertEqual(row["correct_logprob"], -3.0)
        self.assertEqual(row["lure_logprob"], -0.5)

    def test_edited_row_follows_the_repo_sign_convention(self) -> None:
        row = _condition_row(
            CASE,
            target_layer=31,
            target_feature=4242,
            condition="joint",
            draw=-1,
            margin=ABLATED,
            baseline=BASELINE,
        )

        self.assertAlmostEqual(row["margin_delta"], 1.5, places=9)  # baseline - ablated
        self.assertAlmostEqual(row["correct_logprob_delta"], 2.0, places=9)  # ablated - baseline
        self.assertAlmostEqual(row["lure_logprob_delta"], -1.5, places=9)  # ablated - baseline

    def test_absolute_and_delta_reconstruct_each_other(self) -> None:
        # With both halves on the row, any delta can be re-derived if the convention is
        # ever questioned again -- which is why the absolutes are recorded at all.
        deltas = _margin_row(ABLATED, BASELINE)
        edited = _absolute_row(ABLATED)
        clean = _absolute_row(BASELINE)

        self.assertAlmostEqual(clean["margin"] - edited["margin"], deltas["margin_delta"], 9)
        self.assertAlmostEqual(
            edited["correct_logprob"] - clean["correct_logprob"],
            deltas["correct_logprob_delta"],
            9,
        )

    def test_every_row_key_has_a_csv_column(self) -> None:
        # DictWriter is built with extrasaction="ignore", so a key added to the row and
        # not to COABLATION_COLUMNS would vanish from the artifact in silence.
        row = _condition_row(
            CASE,
            target_layer=31,
            target_feature=4242,
            condition="rand_joint",
            draw=3,
            margin=ABLATED,
            baseline=BASELINE,
        )

        self.assertEqual(set(row) - set(COABLATION_COLUMNS), set())


class CsvTests(unittest.TestCase):
    def test_unmeasured_signal_writes_an_empty_cell_not_the_word_none(self) -> None:
        # pandas reads "" as NaN and "None" as a category; the latter would turn a
        # signal that was never measured into a value that looks legitimate.
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(
                Path(tmp) / "siblings.csv",
                [{"target_feature": 7, "specificity_corr": None}],
                ["target_feature", "specificity_corr"],
            )
            rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))

        self.assertEqual(rows[0]["specificity_corr"], "")

    def test_every_sibling_row_key_has_a_column(self) -> None:
        # Same guard as the co-ablation rows: the levels and the verdict are the numbers
        # that expose an anti-specific candidate, so they must not be droppable by an
        # out-of-date column list.
        row = score_siblings(
            [
                {
                    **_candidate(4242),
                    "source_layer": 15,
                    "source_feature": 81663,
                    "target_layer": 31,
                    "mean_activation": 1.25,
                    "source_mean_effect": 0.5,
                    "source_mean_specificity": 0.3,
                }
            ]
        )[0]

        self.assertEqual(set(row) - set(SIBLING_COLUMNS), set())

    def test_a_gated_out_candidate_writes_an_empty_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_csv(
                Path(tmp) / "siblings.csv",
                score_siblings([_candidate(280, mean_effect=-0.329)]),
                SIBLING_COLUMNS,
            )
            rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))

        self.assertEqual(rows[0]["combined_score"], "")
        self.assertEqual(rows[0]["specificity_verdict"], SPECIFICITY_HOSTILE_ARM_NON_POSITIVE)
        self.assertEqual(rows[0]["mean_effect"], "-0.329")


if __name__ == "__main__":
    unittest.main()
