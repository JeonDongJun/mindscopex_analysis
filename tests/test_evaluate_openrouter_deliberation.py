from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_openrouter_deliberation import (  # noqa: E402
    Case,
    extract_final,
    score_crt_visible_final,
    select_balanced_cases,
    strict_premise_correction,
)


class DeliberationDesignTests(unittest.TestCase):
    def test_balanced_suite_excludes_ambiguous_yolk_item(self) -> None:
        crt, semantic = select_balanced_cases()
        self.assertEqual(len(crt), 50)
        self.assertEqual(len(semantic), 50)
        self.assertNotIn("verbal_crt_010", {case.case_id for case in crt})
        self.assertEqual(len({case.question.casefold() for case in crt}), 50)

    def test_final_tag_is_separated_from_verification(self) -> None:
        final, found = extract_final(
            "<verification>The tempting answer is 10.</verification>"
            "<final>The answer is 5.</final>"
        )
        self.assertTrue(found)
        self.assertEqual(final, "The answer is 5.")

    def test_crt_scorer_does_not_penalize_rejected_lure_in_explanation(self) -> None:
        case = Case(
            dataset_id="test",
            case_id="test_rate",
            family="crt_rate",
            question="",
            scoring="logprob_margin",
            correct_answer="7 hours",
            lure_answer="4 hours",
            reference_answer="",
        )
        answer = (
            "7 hours. Four workers each finish one item in 7 hours; "
            "the tempting 4 hours is incorrect."
        )
        self.assertEqual(score_crt_visible_final(answer, case), "correct")

    def test_crt_scorer_respects_units_when_other_numbers_appear_first(self) -> None:
        case = Case(
            dataset_id="test",
            case_id="test_growth",
            family="crt_growth",
            question="",
            scoring="logprob_margin",
            correct_answer="3 weeks",
            lure_answer="2 weeks",
            reference_answer="",
        )
        answer = (
            "The pile is 2 meters high at the end of the third week, "
            "so the answer is 3 weeks."
        )
        self.assertEqual(score_crt_visible_final(answer, case), "correct")

    def test_strict_correction_requires_an_explicit_cue(self) -> None:
        self.assertFalse(strict_premise_correction("Argentina."))
        self.assertTrue(
            strict_premise_correction(
                "He represented Argentina, but his nickname was the Golden Boy."
            )
        )


if __name__ == "__main__":
    unittest.main()
