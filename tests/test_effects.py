from __future__ import annotations

import math
import unittest

import torch

from mindscopex_analysis.effects import (
    _direction_edit,
    continuation_logprob_from_logits,
    continuation_token_span,
)


class _WordTokenizer:
    """Deterministic whitespace tokenizer with a growing vocabulary."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def _id(self, token: str) -> int:
        return self.vocab.setdefault(token, len(self.vocab) + 1)

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [self._id(tok) for tok in text.split()]
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids, **_kwargs):
        reverse = {value: key for key, value in self.vocab.items()}
        return " ".join(reverse.get(int(i), "?") for i in ids)


class ContinuationTokenSpanTests(unittest.TestCase):
    def test_returns_full_ids_and_answer_start(self) -> None:
        tokenizer = _WordTokenizer()
        full_ids, start = continuation_token_span(tokenizer, "a b", " c d")

        self.assertEqual(start, 2)
        self.assertEqual(full_ids.tolist(), [1, 2, 3, 4])

    def test_rejects_empty_continuation(self) -> None:
        tokenizer = _WordTokenizer()
        with self.assertRaises(ValueError):
            continuation_token_span(tokenizer, "a b", "")

    def test_rejects_prompt_not_a_prefix(self) -> None:
        class _RetokenizingTokenizer(_WordTokenizer):
            def __call__(self, text, return_tensors=None, add_special_tokens=True):
                # Force a different first token once the answer is appended so the
                # prompt ids are no longer a prefix of prompt+answer ids.
                ids = [self._id(tok) for tok in text.split()]
                if "merge" in text:
                    ids[0] = self._id("merged")
                return {"input_ids": torch.tensor([ids])}

        tokenizer = _RetokenizingTokenizer()
        with self.assertRaises(ValueError):
            continuation_token_span(tokenizer, "a b", " merge")


class ContinuationLogprobTests(unittest.TestCase):
    def test_uses_previous_position_logits_for_each_target(self) -> None:
        vocab = 4
        # Uniform logits -> each chosen token has logprob log(1/vocab).
        logits = torch.zeros(3, vocab)
        full_ids = torch.tensor([0, 2, 3])

        result = continuation_logprob_from_logits(logits, full_ids, target_start=1)

        expected_per_token = math.log(1.0 / vocab)
        self.assertEqual(result.token_ids, (2, 3))
        self.assertAlmostEqual(result.logprob, 2 * expected_per_token, places=5)
        self.assertAlmostEqual(result.mean_logprob, expected_per_token, places=5)
        self.assertEqual(len(result.token_logprobs), 2)

    def test_accepts_batched_logits(self) -> None:
        logits = torch.zeros(1, 3, 4)
        full_ids = torch.tensor([0, 1, 2])
        result = continuation_logprob_from_logits(logits, full_ids, target_start=1)
        self.assertEqual(len(result.token_logprobs), 2)

    def test_rejects_nonpositive_target_start(self) -> None:
        with self.assertRaises(ValueError):
            continuation_logprob_from_logits(torch.zeros(3, 4), torch.tensor([0, 1, 2]), 0)

    def test_rejects_logits_shorter_than_ids(self) -> None:
        with self.assertRaises(ValueError):
            continuation_logprob_from_logits(torch.zeros(2, 4), torch.tensor([0, 1, 2]), 1)


class DirectionEditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hidden = torch.tensor([[1.0, 1.0, 1.0]])
        # Norm 3 so the unit vector is exactly [1, 0, 0].
        self.direction = torch.tensor([3.0, 0.0, 0.0])

    def _edit(self, mode, *, feature_value=2.0, coefficient=1.0):
        return _direction_edit(
            self.hidden,
            self.direction,
            feature_value=feature_value,
            coefficient=coefficient,
            intervention_mode=mode,
        )

    def test_remove_activation_subtracts_scaled_direction(self) -> None:
        edited = self._edit("remove_activation", feature_value=2.0)
        self.assertTrue(torch.allclose(edited, torch.tensor([[-5.0, 1.0, 1.0]])))

    def test_add_activation_adds_scaled_direction(self) -> None:
        edited = self._edit("add_activation", feature_value=2.0)
        self.assertTrue(torch.allclose(edited, torch.tensor([[7.0, 1.0, 1.0]])))

    def test_subtract_unit_uses_normalized_direction(self) -> None:
        edited = self._edit("subtract_unit", coefficient=2.0)
        self.assertTrue(torch.allclose(edited, torch.tensor([[-1.0, 1.0, 1.0]])))

    def test_add_unit_uses_normalized_direction(self) -> None:
        edited = self._edit("add_unit", coefficient=2.0)
        self.assertTrue(torch.allclose(edited, torch.tensor([[3.0, 1.0, 1.0]])))

    def test_projection_remove_strips_component_along_direction(self) -> None:
        edited = self._edit("projection_remove", coefficient=1.0)
        self.assertTrue(torch.allclose(edited, torch.tensor([[0.0, 1.0, 1.0]])))

    def test_add_vector_ignores_feature_value(self) -> None:
        edited = self._edit("add_vector", feature_value=999.0, coefficient=2.0)
        self.assertTrue(torch.allclose(edited, torch.tensor([[7.0, 1.0, 1.0]])))

    def test_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            self._edit("nonsense")


if __name__ == "__main__":
    unittest.main()
