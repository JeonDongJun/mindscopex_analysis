from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mindscopex_analysis.cases import LureCase
from mindscopex_analysis.workflows import intervention_mode_rows, layer_feature_search_rows


def _case() -> LureCase:
    return LureCase(
        case_id="case",
        family="test",
        prompt="Question\nAnswer:",
        correct_answer=" a",
        lure_answer=" b",
    )


def _margin(value: float) -> SimpleNamespace:
    return SimpleNamespace(
        margin=value,
        correct=SimpleNamespace(logprob=value),
        lure=SimpleNamespace(logprob=2 * value),
    )


class LayerSearchEfficiencyTests(unittest.TestCase):
    def test_captures_layers_and_baseline_once(self) -> None:
        baseline = _margin(1.0)
        residuals = {1: torch.ones(1, 2), 2: torch.ones(1, 2)}

        with (
            patch(
                "mindscopex_analysis.workflows.capture_residual_stream",
                return_value=residuals,
            ) as capture,
            patch(
                "mindscopex_analysis.workflows.answer_logprob_margin",
                return_value=baseline,
            ) as score,
            patch(
                "mindscopex_analysis.workflows.active_prompt_features",
                return_value=[(0, 1.0)],
            ),
            patch(
                "mindscopex_analysis.workflows.rank_lure_feature_effects",
                return_value=(baseline, []),
            ) as rank,
        ):
            rows = layer_feature_search_rows(
                object(),
                _case(),
                layers=[1, 2],
                sae_by_layer={1: object(), 2: object()},
                top_n=1,
            )

        self.assertEqual(rows, [])
        capture.assert_called_once()
        score.assert_called_once()
        self.assertEqual(rank.call_count, 2)
        self.assertTrue(all(call.kwargs["baseline"] is baseline for call in rank.call_args_list))


class InterventionModeEfficiencyTests(unittest.TestCase):
    def test_reuses_baseline_across_modes(self) -> None:
        margins = [_margin(3.0), _margin(2.0), _margin(1.0)]
        with (
            patch(
                "mindscopex_analysis.workflows.sae_decoder_direction",
                return_value=torch.ones(2),
            ),
            patch(
                "mindscopex_analysis.workflows.answer_logprob_margin",
                side_effect=margins,
            ) as score,
        ):
            rows = intervention_mode_rows(
                object(),
                _case(),
                layer=1,
                sae=object(),
                feature_id=2,
                feature_value=0.5,
                modes=["remove_activation", "add_activation"],
            )

        self.assertEqual(score.call_count, 3)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["baseline_margin"] == 3.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
