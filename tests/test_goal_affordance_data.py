from __future__ import annotations

import json
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_goal_affordance_dataset import build_payload, canonical_scenarios  # noqa: E402


class GoalAffordanceDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = (
            ROOT
            / "src"
            / "mindscopex_analysis"
            / "data"
            / "goal_affordance_traps_v1.json"
        )
        cls.payload = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_canonical_dataset_rebuilds_exactly(self) -> None:
        scenarios, provenance = canonical_scenarios()
        self.assertEqual(build_payload(scenarios, provenance), self.payload)

    def test_sixty_balanced_scenarios_expand_to_four_conditions(self) -> None:
        self.assertEqual(self.payload["n_base_scenarios"], 60)
        self.assertEqual(self.payload["n_cases"], 240)
        self.assertEqual(set(self.payload["base_family_counts"].values()), {10})
        self.assertEqual(set(self.payload["family_counts"].values()), {40})
        self.assertEqual(set(self.payload["condition_counts"].values()), {60})

    def test_minimal_pair_answer_mappings_are_consistent(self) -> None:
        by_pair = defaultdict(dict)
        for row in self.payload["cases"]:
            by_pair[row["pair_id"]][row["condition"]] = row
        self.assertEqual(len(by_pair), 60)
        for pair_id, conditions in by_pair.items():
            self.assertEqual(
                set(conditions),
                {"counterfactual", "explicit", "hostile", "neutral"},
                pair_id,
            )
            hostile = conditions["hostile"]
            for name in ("explicit", "neutral"):
                self.assertEqual(
                    (
                        conditions[name]["correct_answer"],
                        conditions[name]["lure_answer"],
                    ),
                    (hostile["correct_answer"], hostile["lure_answer"]),
                    pair_id,
                )
            counterfactual = conditions["counterfactual"]
            self.assertEqual(
                (
                    counterfactual["correct_answer"],
                    counterfactual["lure_answer"],
                ),
                (hostile["lure_answer"], hostile["correct_answer"]),
                pair_id,
            )

    def test_option_positions_are_nearly_balanced(self) -> None:
        from evaluate_goal_affordance import correct_is_a, load_cases

        _, cases = load_cases(self.path, None)
        positions = Counter("A" if correct_is_a(case) else "B" for case in cases)
        self.assertLessEqual(abs(positions["A"] - positions["B"]), 24)


if __name__ == "__main__":
    unittest.main()
