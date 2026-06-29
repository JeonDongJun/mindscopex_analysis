from __future__ import annotations

import unittest

from mindscopex_analysis import crt_behavior_cases, crt_transfer_cases


class CrtBehaviorCasesTests(unittest.TestCase):
    def test_behavior_suite_extends_transfer_suite(self) -> None:
        transfer_ids = [case.case_id for case in crt_transfer_cases()]
        behavior = crt_behavior_cases()
        behavior_ids = [case.case_id for case in behavior]

        self.assertEqual(behavior_ids[: len(transfer_ids)], transfer_ids)
        self.assertEqual(len(behavior), 9)
        self.assertEqual(len(behavior_ids), len(set(behavior_ids)))
        self.assertIn("clock_strikes", behavior_ids)
        self.assertIn("discount_reversal", behavior_ids)

    def test_each_case_has_distinct_correct_and_lure_answers(self) -> None:
        for case in crt_behavior_cases():
            with self.subTest(case=case.case_id):
                self.assertNotEqual(case.correct_answer.strip(), case.lure_answer.strip())


if __name__ == "__main__":
    unittest.main()
