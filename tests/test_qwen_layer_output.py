from __future__ import annotations

import inspect
import unittest

import torch

from mindscopex_analysis import (
    answer_logprob_margin,
    capture_layer_residuals,
    capture_residual_stream,
    rank_lure_feature_effects,
    scan_qwen_scope_layers,
    score_answer_logprob,
)
from mindscopex_analysis.activations import select_token_positions


class QwenLayerOutputTests(unittest.TestCase):
    def test_qwen_tensor_output_keeps_batch_axis(self) -> None:
        hidden = torch.arange(24).reshape(1, 3, 8)

        selected = select_token_positions(hidden, "last")

        self.assertEqual(tuple(selected.shape), (1, 8))
        self.assertTrue(torch.equal(selected[0], hidden[0, -1]))

    def test_public_qwen_paths_do_not_index_block_output_by_default(self) -> None:
        functions = (
            capture_residual_stream,
            capture_layer_residuals,
            score_answer_logprob,
            answer_logprob_margin,
            rank_lure_feature_effects,
            scan_qwen_scope_layers,
        )

        for function in functions:
            with self.subTest(function=function.__name__):
                parameter = inspect.signature(function).parameters["output_index"]
                self.assertIsNone(parameter.default)


if __name__ == "__main__":
    unittest.main()
