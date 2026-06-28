from __future__ import annotations

import unittest

import torch

from mindscopex_analysis import (
    BAT_BALL_CASE,
    classify_lure_answer,
    generate_qwen_text_response,
    text_contains_answer,
)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def apply_chat_template(self, _messages, **_kwargs):
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, _token_ids, **_kwargs):
        return "<think>check the algebra</think>The answer is 5 cents.<|im_end|>"


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
            max_new_tokens=4,
            do_sample=False,
        )

        self.assertEqual(response.thinking, "check the algebra")
        self.assertEqual(response.answer, "The answer is 5 cents.")
        self.assertEqual(response.answer_label, "correct")
        self.assertEqual(response.output_tokens, 2)
        self.assertFalse(response.hit_max_tokens)


if __name__ == "__main__":
    unittest.main()
