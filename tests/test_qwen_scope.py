from __future__ import annotations

import unittest

import torch

from mindscopex_analysis.qwen_scope import (
    QwenScopeSAE,
    encode_qwen_scope_topk,
    infer_top_k_from_repo,
    make_layer_feature_report,
    sae_decoder_direction,
    split_qwen_thinking,
    summarize_qwen_scope_features,
    top_qwen_scope_features,
)

D_MODEL = 3
D_SAE = 5


def _make_sae(*, w_dec: torch.Tensor | None = None, top_k: int = 2) -> QwenScopeSAE:
    """Build a tiny SAE where feature i has pre-activation = i * x[0].

    With input ``[1, 0, 0]`` the pre-activations are ``[0, 1, 2, 3, 4]``, so the
    top-k features are the highest-numbered ones. This keeps every assertion
    below hand-checkable.
    """

    w_enc = torch.zeros(D_SAE, D_MODEL)
    for i in range(D_SAE):
        w_enc[i, 0] = float(i)
    if w_dec is None:
        w_dec = torch.zeros(D_MODEL, D_SAE)
    return QwenScopeSAE(
        repo_id="test/sae",
        layer=0,
        W_enc=w_enc,
        W_dec=w_dec,
        b_enc=torch.zeros(D_SAE),
        b_dec=torch.zeros(D_MODEL),
        top_k=top_k,
    )


class InferTopKTests(unittest.TestCase):
    def test_parses_suffix(self) -> None:
        self.assertEqual(infer_top_k_from_repo("Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50"), 50)

    def test_falls_back_to_default(self) -> None:
        self.assertEqual(infer_top_k_from_repo("no-suffix-here", default=17), 17)


class EncodeTopKTests(unittest.TestCase):
    def test_returns_highest_pre_activations(self) -> None:
        sae = _make_sae(top_k=2)
        x = torch.tensor([[1.0, 0.0, 0.0]])
        vals, idx = encode_qwen_scope_topk(x, sae)
        self.assertEqual(idx[0].tolist(), [4, 3])
        self.assertTrue(torch.allclose(vals[0], torch.tensor([4.0, 3.0])))

    def test_rejects_dimension_mismatch(self) -> None:
        sae = _make_sae()
        with self.assertRaises(ValueError):
            encode_qwen_scope_topk(torch.zeros(1, D_MODEL + 1), sae)


class SummarizeFeaturesTests(unittest.TestCase):
    def test_aggregates_topk_activations_over_tokens(self) -> None:
        sae = _make_sae(top_k=2)
        residuals = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        summary = summarize_qwen_scope_features(residuals, sae)

        self.assertEqual(summary["n_tokens"], 2)
        # Features 3 and 4 fire on every token; the rest never fire.
        self.assertAlmostEqual(float(summary["mean"][4]), 4.0, places=5)
        self.assertAlmostEqual(float(summary["mean"][3]), 3.0, places=5)
        self.assertAlmostEqual(float(summary["mean"][0]), 0.0, places=5)
        self.assertAlmostEqual(float(summary["activation_rate"][4]), 1.0, places=5)
        self.assertAlmostEqual(float(summary["activation_rate"][0]), 0.0, places=5)
        self.assertAlmostEqual(float(summary["max"][4]), 4.0, places=5)
        # Never-fired feature collapses from -inf to 0.
        self.assertAlmostEqual(float(summary["max"][0]), 0.0, places=5)

    def test_batched_path_matches_single_batch(self) -> None:
        sae = _make_sae(top_k=2)
        residuals = torch.tensor([[1.0, 0.0, 0.0]] * 5)
        small = summarize_qwen_scope_features(residuals, sae, batch_size=2)
        whole = summarize_qwen_scope_features(residuals, sae, batch_size=64)
        self.assertTrue(torch.allclose(small["mean"], whole["mean"]))
        self.assertTrue(torch.allclose(small["activation_rate"], whole["activation_rate"]))


class TopFeaturesTests(unittest.TestCase):
    def _summary(self) -> dict[str, torch.Tensor | int]:
        return {
            "mean": torch.tensor([0.1, -5.0, 2.0]),
            "mean_abs": torch.tensor([0.1, 5.0, 2.0]),
            "max": torch.tensor([0.1, 0.0, 2.0]),
            "activation_rate": torch.tensor([1.0, 0.2, 0.5]),
            "n_tokens": 10,
        }

    def test_mean_abs_ranks_largest_magnitude_first(self) -> None:
        top = top_qwen_scope_features(self._summary(), top_n=2, metric="mean_abs")
        self.assertEqual([item.feature_id for item in top], [1, 2])

    def test_mean_metric_misses_strong_negative_feature(self) -> None:
        # Documents current behavior: signed `mean` ranking ignores the large
        # negative-mean feature (id 1), which `mean_abs` would surface.
        top = top_qwen_scope_features(self._summary(), top_n=1, metric="mean")
        self.assertEqual(top[0].feature_id, 2)

    def test_rejects_unknown_metric(self) -> None:
        with self.assertRaises(ValueError):
            top_qwen_scope_features(self._summary(), metric="bogus")


class LayerReportTests(unittest.TestCase):
    def test_score_combines_magnitude_and_rate(self) -> None:
        summary = {
            "mean": torch.tensor([2.0, 0.0]),
            "mean_abs": torch.tensor([2.0, 0.0]),
            "max": torch.tensor([2.0, 0.0]),
            "activation_rate": torch.tensor([0.5, 0.0]),
            "n_tokens": 4,
        }
        report = make_layer_feature_report(0, summary, top_n=1, metric="mean_abs")
        # mean_abs=2.0, rate=0.5 -> 2.0 * (1 + 0.5) = 3.0
        self.assertAlmostEqual(report.score, 3.0, places=5)
        self.assertEqual(report.n_tokens, 4)


class DecoderDirectionTests(unittest.TestCase):
    def test_column_oriented_decoder(self) -> None:
        w_dec = torch.zeros(D_MODEL, D_SAE)
        w_dec[:, 4] = torch.tensor([1.0, 2.0, 3.0])
        sae = _make_sae(w_dec=w_dec)
        direction = sae_decoder_direction(sae, [4])
        self.assertTrue(torch.allclose(direction, torch.tensor([1.0, 2.0, 3.0])))

    def test_row_oriented_decoder(self) -> None:
        w_dec = torch.zeros(D_SAE, D_MODEL)
        w_dec[4, :] = torch.tensor([1.0, 2.0, 3.0])
        sae = _make_sae(w_dec=w_dec)
        direction = sae_decoder_direction(sae, [4])
        self.assertTrue(torch.allclose(direction, torch.tensor([1.0, 2.0, 3.0])))

    def test_applies_coefficients(self) -> None:
        w_dec = torch.zeros(D_MODEL, D_SAE)
        w_dec[:, 3] = torch.tensor([1.0, 0.0, 0.0])
        w_dec[:, 4] = torch.tensor([0.0, 1.0, 0.0])
        sae = _make_sae(w_dec=w_dec)
        direction = sae_decoder_direction(sae, [3, 4], [2.0, 5.0])
        self.assertTrue(torch.allclose(direction, torch.tensor([2.0, 5.0, 0.0])))

    def test_rejects_mismatched_coefficients(self) -> None:
        sae = _make_sae(w_dec=torch.zeros(D_MODEL, D_SAE))
        with self.assertRaises(ValueError):
            sae_decoder_direction(sae, [0, 1], [1.0])


class SplitThinkingTests(unittest.TestCase):
    def test_splits_thinking_and_answer(self) -> None:
        thinking, answer = split_qwen_thinking("<think>reason here</think>5 cents<|im_end|>")
        self.assertEqual(thinking, "reason here")
        self.assertEqual(answer, "5 cents")

    def test_no_thinking_block_returns_cleaned_text(self) -> None:
        thinking, answer = split_qwen_thinking("just an answer<|endoftext|>")
        self.assertEqual(thinking, "")
        self.assertEqual(answer, "just an answer")

    def test_uses_last_closing_tag(self) -> None:
        thinking, answer = split_qwen_thinking("<think>a</think>mid</think>final")
        self.assertEqual(thinking, "a</think>mid")
        self.assertEqual(answer, "final")


if __name__ == "__main__":
    unittest.main()
