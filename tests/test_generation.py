from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from mindscopex_analysis import (
    BAT_BALL_CASE,
    classify_lure_answer,
    generate_qwen_text_response,
    summarize_crt_accuracy,
    text_contains_answer,
)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __init__(self, decoded_text=None) -> None:
        self.decoded_text = decoded_text or (
            "<think>check the algebra</think>The answer is 5 cents.<|im_end|>"
        )

    def apply_chat_template(self, _messages, **_kwargs):
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, _token_ids, **_kwargs):
        return self.decoded_text


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 4)

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, input_ids, attention_mask, **_kwargs):
        self.assert_attention_mask = attention_mask
        suffix = torch.tensor([[3, 9]], device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


class AnswerClassificationTests(unittest.TestCase):
    def test_recognizes_correct_cents_surface_forms(self) -> None:
        self.assertTrue(text_contains_answer("The answer is 5 cents.", "5 cents"))
        self.assertTrue(text_contains_answer("The ball costs $0.05.", "5 cents"))

    def test_does_not_confuse_five_with_ten(self) -> None:
        self.assertFalse(text_contains_answer("The answer is 10 cents.", "5 cents"))

    def test_recognizes_nature_currency_surface_forms(self) -> None:
        self.assertTrue(text_contains_answer("$20", "$20.0"))
        self.assertTrue(text_contains_answer("0.15 dollars", "$0.150"))
        self.assertFalse(text_contains_answer("$40", "$20.0"))

    def test_recognizes_extended_time_units_and_ordinal_days(self) -> None:
        self.assertTrue(text_contains_answer("7 hours", "7 hours"))
        self.assertTrue(text_contains_answer("50 weeks", "50 weeks"))
        self.assertTrue(text_contains_answer("day 5", "5th day"))

    def test_labels_final_answer(self) -> None:
        self.assertEqual(classify_lure_answer("5 cents", BAT_BALL_CASE), "correct")
        self.assertEqual(classify_lure_answer("10 cents", BAT_BALL_CASE), "lure")
        self.assertEqual(
            classify_lure_answer("It is 5 cents, not 10 cents.", BAT_BALL_CASE),
            "both",
        )

    def test_generation_splits_thinking_and_final_answer(self) -> None:
        response = generate_qwen_text_response(
            _FakeModel(),
            _FakeTokenizer(),
            BAT_BALL_CASE,
            model_id="Qwen/fake",
            enable_thinking=True,
            max_new_tokens=4,
            do_sample=False,
        )

        self.assertEqual(response.thinking, "check the algebra")
        self.assertEqual(response.answer, "The answer is 5 cents.")
        self.assertEqual(response.answer_label, "correct")
        self.assertEqual(response.output_tokens, 2)
        self.assertFalse(response.hit_max_tokens)
        self.assertTrue(response.has_thinking_block)
        self.assertTrue(response.reasoning_detected)
        self.assertTrue(response.thinking_protocol_ok)
        self.assertIsNone(response.thinking_protocol_issue)
        self.assertFalse(response.final_answer_format_ok)
        self.assertEqual(response.final_answer_format_issue, "label_or_explanation")

    def test_non_thinking_response_has_no_generated_thinking_block(self) -> None:
        response = generate_qwen_text_response(
            _FakeModel(),
            _FakeTokenizer("5 cents<|im_end|>"),
            BAT_BALL_CASE,
            model_id="Qwen/fake",
            enable_thinking=False,
            max_new_tokens=4,
            do_sample=False,
        )

        self.assertEqual(response.answer, "5 cents")
        self.assertFalse(response.has_thinking_block)
        self.assertFalse(response.reasoning_detected)
        self.assertTrue(response.thinking_protocol_ok)
        self.assertTrue(response.final_answer_format_ok)

    def test_thinking_protocol_reports_missing_block(self) -> None:
        response = generate_qwen_text_response(
            _FakeModel(),
            _FakeTokenizer("5 cents<|im_end|>"),
            BAT_BALL_CASE,
            model_id="Qwen/fake",
            enable_thinking=True,
            max_new_tokens=4,
            do_sample=False,
        )

        self.assertFalse(response.thinking_protocol_ok)
        self.assertEqual(response.thinking_protocol_issue, "missing_thinking_block")

    def test_thinking_protocol_distinguishes_truncated_block(self) -> None:
        response = generate_qwen_text_response(
            _FakeModel(),
            _FakeTokenizer("<think>unfinished reasoning"),
            BAT_BALL_CASE,
            model_id="Qwen/fake",
            enable_thinking=True,
            max_new_tokens=4,
            do_sample=False,
        )

        truncated = replace(response, hit_max_tokens=True)
        self.assertEqual(truncated.thinking_protocol_issue, "truncated_before_think_close")
        self.assertTrue(truncated.reasoning_detected)
        self.assertEqual(truncated.answer, "")
        self.assertEqual(truncated.answer_label, "other")

    def test_summarizes_accuracy_by_model_and_mode(self) -> None:
        correct = generate_qwen_text_response(
            _FakeModel(),
            _FakeTokenizer("5 cents<|im_end|>"),
            BAT_BALL_CASE,
            model_id="Qwen/fake",
            enable_thinking=False,
            max_new_tokens=4,
            do_sample=False,
        )
        lure = replace(correct, answer="10 cents", answer_label="lure")

        rows = summarize_crt_accuracy([correct, lure])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model"], "fake")
        self.assertEqual(rows[0]["mode"], "non_thinking")
        self.assertEqual(rows[0]["total"], 2)
        self.assertEqual(rows[0]["correct"], 1)
        self.assertEqual(rows[0]["incorrect"], 1)
        self.assertEqual(rows[0]["lure"], 1)
        self.assertEqual(rows[0]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
