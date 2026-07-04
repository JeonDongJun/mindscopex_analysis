from __future__ import annotations

import unittest
from unittest.mock import patch

from mindscopex_analysis import (
    nature_crt150_cases,
    parse_nature_crt150_source,
)

FIXTURE_SOURCE = """
crt1=[{
    'task': 'A pen and a notebook cost $12. The pen costs $10 more. What is the notebook?',
    'correct': '$1',
    'intuitive': '$2',
    'number': 1,
}]
crt2=[{
    'task': 'How long for 4 workers to make 4 items if 7 workers make 7 in 7 hours?',
    'correct': '7 hours',
    'intuitive': '4 hours',
    'number': 1,
}]
crt3=[{
    'task': 'A patch doubles daily and fills a lake in 10 days. When is it half full?',
    'correct': '9 days',
    'intuitive': '5 days',
    'number': 1,
}]
si=[]
raise RuntimeError('the parser must never execute source code')
"""


class NatureCrt150DatasetTests(unittest.TestCase):
    def test_parses_literal_records_without_executing_source(self) -> None:
        items = parse_nature_crt150_source(FIXTURE_SOURCE, expected_per_type=1)

        self.assertEqual(len(items), 3)
        self.assertEqual(items[0].item_id, "nature_crt1_001")
        self.assertEqual(items[1].correct_answer, "7 hours")
        self.assertEqual(items[2].lure_answer, "5 days")

    def test_converts_item_to_lure_case(self) -> None:
        item = parse_nature_crt150_source(FIXTURE_SOURCE, expected_per_type=1)[0]

        task_only = item.as_lure_case()
        qa_case = item.as_lure_case(prompt_style="question_answer")

        self.assertEqual(task_only.case_id, "nature_crt1_001")
        self.assertEqual(task_only.family, "nature_crt_difference")
        self.assertEqual(task_only.correct_answer, " $1")
        self.assertEqual(task_only.lure_answer, " $2")
        self.assertTrue(qa_case.prompt.startswith("Question: "))
        self.assertTrue(qa_case.prompt.endswith("\nAnswer:"))

    def test_selects_balanced_types_and_limits(self) -> None:
        items = parse_nature_crt150_source(FIXTURE_SOURCE, expected_per_type=1)
        with patch("mindscopex_analysis.datasets.load_nature_crt150_items", return_value=items):
            cases = nature_crt150_cases(crt_types=("crt3", "crt1"), limit_per_type=1)

        self.assertEqual([case.case_id for case in cases], ["nature_crt3_001", "nature_crt1_001"])

    def test_rejects_unknown_type_and_nonpositive_limit(self) -> None:
        with self.assertRaises(ValueError):
            nature_crt150_cases(crt_types=("crt4",))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            nature_crt150_cases(limit_per_type=0)


if __name__ == "__main__":
    unittest.main()
