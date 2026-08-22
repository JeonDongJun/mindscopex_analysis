from __future__ import annotations

import unittest

from experiments.jobs.research_experiments import _pair_with_controls
from mindscopex_analysis.nulls import NullPanel


class PairWithControlsTests(unittest.TestCase):
    """The cue effect is undefined without a no-cue twin, so pairing must be exact."""

    CONFIG = {
        "data": {
            "dataset": "goal_affordance_traps_v1",
            "conditions": ["hostile"],
            "instruction": False,
        }
    }

    def _hostile(self, n: int = 4) -> list:
        from mindscopex_analysis import lure_dataset_cases

        cases = lure_dataset_cases("goal_affordance_traps_v1")
        return [case for case in cases if case.case_id.endswith("_hostile")][:n]

    def test_pairs_each_case_with_its_own_scenario_twin(self) -> None:
        cases = self._hostile()
        paired, controls = _pair_with_controls(cases, self.CONFIG, "neutral")

        self.assertEqual(len(paired), len(cases))
        self.assertEqual(len(controls), len(cases))
        for case, control in zip(paired, controls, strict=True):
            self.assertTrue(case.case_id.endswith("_hostile"))
            self.assertTrue(control.case_id.endswith("_neutral"))
            # Same scenario, different condition -- not merely the same family.
            self.assertEqual(
                case.case_id.removesuffix("_hostile"),
                control.case_id.removesuffix("_neutral"),
            )

    def test_neutral_twin_keeps_the_answer_mapping(self) -> None:
        case = self._hostile(1)[0]
        _, neutral = _pair_with_controls([case], self.CONFIG, "neutral")

        self.assertEqual(neutral[0].correct_answer, case.correct_answer)
        self.assertEqual(neutral[0].lure_answer, case.lure_answer)

    def test_counterfactual_is_rejected_as_a_discovery_control(self) -> None:
        # The counterfactual twin swaps correct and lure, so differencing against it
        # ADDS the two magnitudes instead of cancelling the shared baseline. It is the
        # specificity gate, not a control -- using it here would make the check
        # circular and inflate every candidate, so pairing must refuse it outright.
        case = self._hostile(1)[0]
        with self.assertRaises(ValueError) as caught:
            _pair_with_controls([case], self.CONFIG, "counterfactual")
        self.assertIn("answer mapping", str(caught.exception))

    def test_missing_twin_raises_rather_than_silently_scoring_nothing(self) -> None:
        with self.assertRaises(ValueError):
            _pair_with_controls(self._hostile(1), self.CONFIG, "no_such_condition")

    def test_empty_conditions_is_rejected(self) -> None:
        config = {"data": {**self.CONFIG["data"], "conditions": []}}
        with self.assertRaises(ValueError):
            _pair_with_controls(self._hostile(1), config, "neutral")


class NullPanelObjectiveTests(unittest.TestCase):
    def test_objective_reflects_whether_controls_are_present(self) -> None:
        bare = NullPanel(
            cases=(),
            baseline_margins=(),
            observed_deltas=(),
            target_norms=(),
            feature_values=(),
        )
        self.assertEqual(bare.objective, "margin_delta")

        paired = NullPanel(
            cases=(),
            baseline_margins=(),
            observed_deltas=(),
            target_norms=(),
            feature_values=(),
            controls=("placeholder",),
        )
        self.assertEqual(paired.objective, "cue_effect")


if __name__ == "__main__":
    unittest.main()
