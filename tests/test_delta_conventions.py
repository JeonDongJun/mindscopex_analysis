"""Pin the sign convention every job writes its logprob deltas with.

This exists because the convention silently diverged once. Three jobs
(feature_modules, multisite_ablation, cross_layer_siblings) wrote
``correct_logprob_delta`` and ``lure_logprob_delta`` as ``baseline - ablated`` while
effects.py and docs/metrics_guide.md define them as ``ablated - baseline``. Same
column name, opposite sign: anyone reading those artifacts against the documented
table would have drawn the exactly inverted conclusion about whether a feature was
suppressing the correct answer or promoting the lure.

The convention is deliberately mixed and that is the trap:

    margin_delta          = baseline - ablated    (positive = ablation cut the lure lead)
    correct_logprob_delta = ablated  - baseline   (positive = ablation raised the correct answer)
    lure_logprob_delta    = ablated  - baseline   (negative = ablation lowered the lure)

so a feature that promotes the lure shows margin_delta > 0, correct_delta > 0,
lure_delta < 0. These tests assert that on a scenario where every sign is distinct,
so no rearrangement of the terms can pass by accident.
"""

from __future__ import annotations

import importlib
import unittest
from dataclasses import dataclass


@dataclass(frozen=True)
class _Logprob:
    logprob: float


@dataclass(frozen=True)
class _Margin:
    """Stand-in for effects.AnswerMargin, carrying only what _margin_row reads."""

    margin: float
    correct: _Logprob
    lure: _Logprob


# A lure-promoting feature: ablating it raises the correct answer (-3.0 -> -1.0),
# lowers the lure (-0.5 -> -2.0) and so cuts the margin (2.5 -> 1.0). Every one of
# the four numbers is distinct, and the two deltas have opposite signs.
BASELINE = _Margin(margin=2.5, correct=_Logprob(-3.0), lure=_Logprob(-0.5))
ABLATED = _Margin(margin=1.0, correct=_Logprob(-1.0), lure=_Logprob(-2.0))

EXPECTED = {
    "margin_delta": 1.5,  # baseline - ablated
    "correct_logprob_delta": 2.0,  # ablated - baseline
    "lure_logprob_delta": -1.5,  # ablated - baseline
}

JOBS_WITH_MARGIN_ROW = (
    "experiments.jobs.feature_modules",
    "experiments.jobs.multisite_ablation",
)


class MarginRowConventionTests(unittest.TestCase):
    def test_every_job_writes_the_documented_signs(self) -> None:
        for module_name in JOBS_WITH_MARGIN_ROW:
            with self.subTest(module=module_name):
                row = importlib.import_module(module_name)._margin_row(ABLATED, BASELINE)
                for key, expected in EXPECTED.items():
                    self.assertAlmostEqual(
                        float(row[key]),
                        expected,
                        places=9,
                        msg=(
                            f"{module_name}.{key} = {row[key]}, expected {expected}. "
                            "See docs/metrics_guide.md; margin_delta is baseline-ablated "
                            "but the logprob deltas are ablated-baseline."
                        ),
                    )

    def test_the_deltas_disagree_in_sign_so_a_swap_cannot_pass_silently(self) -> None:
        # Guards the guard: if someone "simplifies" the fixture so all three deltas
        # share a sign, the test above stops being able to detect an inversion.
        self.assertGreater(EXPECTED["correct_logprob_delta"], 0)
        self.assertLess(EXPECTED["lure_logprob_delta"], 0)

    def test_effects_module_is_the_source_of_this_convention(self) -> None:
        from mindscopex_analysis.effects import FeatureAblationResult

        result = FeatureAblationResult(
            layer=0,
            feature_id=0,
            feature_value=1.0,
            baseline_margin=BASELINE.margin,
            ablated_margin=ABLATED.margin,
            margin_delta=BASELINE.margin - ABLATED.margin,
            baseline_mean_margin=0.0,
            ablated_mean_margin=0.0,
            mean_margin_delta=0.0,
            correct_logprob_delta=ABLATED.correct.logprob - BASELINE.correct.logprob,
            lure_logprob_delta=ABLATED.lure.logprob - BASELINE.lure.logprob,
        )
        row = result.as_row()
        for key, expected in EXPECTED.items():
            self.assertAlmostEqual(float(row[key]), expected, places=9, msg=key)


if __name__ == "__main__":
    unittest.main()
