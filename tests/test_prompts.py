from __future__ import annotations

import unittest

from mindscopex_analysis import (
    BAT_BALL_CASE,
    CRT_FINAL_ANSWER_SYSTEM_PROMPT,
    instruct_lure_case,
    prepend_final_answer_instruction,
)


class FinalAnswerInstructionTests(unittest.TestCase):
    def test_instruction_is_answer_only_without_leaking_a_case_answer(self) -> None:
        self.assertIn("exactly one line", CRT_FINAL_ANSWER_SYSTEM_PROMPT)
        self.assertIn("Do not include calculations", CRT_FINAL_ANSWER_SYSTEM_PROMPT)
        self.assertNotIn("5 cents", CRT_FINAL_ANSWER_SYSTEM_PROMPT)

    def test_prefix_is_applied_only_once(self) -> None:
        prompt = "Question?\nAnswer:"
        instructed = prepend_final_answer_instruction(prompt)

        self.assertTrue(instructed.startswith(CRT_FINAL_ANSWER_SYSTEM_PROMPT))
        self.assertEqual(prepend_final_answer_instruction(instructed), instructed)

    def test_case_and_control_receive_the_same_instruction(self) -> None:
        instructed = instruct_lure_case(BAT_BALL_CASE)

        self.assertTrue(instructed.prompt.startswith(CRT_FINAL_ANSWER_SYSTEM_PROMPT))
        self.assertTrue(instructed.control_prompt.startswith(CRT_FINAL_ANSWER_SYSTEM_PROMPT))
        self.assertEqual(instructed.correct_answer, BAT_BALL_CASE.correct_answer)
        self.assertEqual(instructed.lure_answer, BAT_BALL_CASE.lure_answer)
        self.assertFalse(BAT_BALL_CASE.prompt.startswith(CRT_FINAL_ANSWER_SYSTEM_PROMPT))


if __name__ == "__main__":
    unittest.main()
