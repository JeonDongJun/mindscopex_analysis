from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "analyze_crt_responses",
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_crt_responses.py",
)
analyze = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyze)


class WilsonTests(unittest.TestCase):
    def test_bounds(self) -> None:
        low, high = analyze.wilson_interval(5, 10)
        self.assertAlmostEqual(low, 0.2366, places=3)
        self.assertAlmostEqual(high, 0.7634, places=3)

    def test_degenerate(self) -> None:
        self.assertEqual(analyze.wilson_interval(0, 0), (0.0, 0.0))
        _, high = analyze.wilson_interval(10, 10)
        self.assertLessEqual(high, 1.0)


class MergeRowsTests(unittest.TestCase):
    def test_sums_matching_keys_and_recomputes_rates(self) -> None:
        rows = [
            {"model": "X", "mode": "thinking", "total": 10, "correct": 6, "lure": 3},
            {"model": "X", "mode": "thinking", "total": 10, "correct": 4, "lure": 5},
            {"model": "X", "mode": "non_thinking", "total": 10, "correct": 2, "lure": 7},
        ]
        merged = {(r["model"], r["mode"]): r for r in analyze.merge_rows(rows, ("model", "mode"))}
        think = merged[("X", "thinking")]
        self.assertEqual(think["total"], 20)
        self.assertEqual(think["correct"], 10)
        self.assertAlmostEqual(think["accuracy"], 0.5)
        self.assertAlmostEqual(think["lure_rate"], 8 / 20)
        self.assertIn(("X", "non_thinking"), merged)

    def test_missing_count_fields_default_zero(self) -> None:
        rows = [{"model": "Y", "mode": "thinking", "total": 4}]
        merged = analyze.merge_rows(rows, ("model", "mode"))
        self.assertEqual(merged[0]["correct"], 0)
        self.assertEqual(merged[0]["accuracy"], 0.0)


class ThinkingEffectTests(unittest.TestCase):
    def test_delta(self) -> None:
        headline = [
            {"model": "X", "mode": "thinking", "accuracy": 0.5, "total": 10},
            {"model": "X", "mode": "non_thinking", "accuracy": 0.3, "total": 10},
        ]
        effect = analyze.thinking_effect(headline)[0]
        self.assertAlmostEqual(effect["acc_delta"], 0.2)

    def test_missing_mode_gives_none_delta(self) -> None:
        effect = analyze.thinking_effect(
            [{"model": "X", "mode": "thinking", "accuracy": 0.5, "total": 10}]
        )[0]
        self.assertIsNone(effect["acc_delta"])


class SizeKeyTests(unittest.TestCase):
    def test_orders_by_capability(self) -> None:
        models = ["Qwen3.5-27B", "Qwen3.5-2B", "Qwen3.5-35B-A3B", "Qwen3.5-9B"]
        ordered = sorted(models, key=analyze._size_key)
        self.assertEqual(ordered, ["Qwen3.5-2B", "Qwen3.5-9B", "Qwen3.5-27B", "Qwen3.5-35B-A3B"])


class ObservationsTests(unittest.TestCase):
    def test_flags_hardest_family_and_quality(self) -> None:
        headline = [
            {
                "model": "X",
                "mode": "thinking",
                "accuracy": 0.6,
                "total": 10,
                "format_failures": 2,
                "protocol_failures": 0,
            }
        ]
        family = [
            {"family": "crt_rate", "lure": 8, "total": 10},
            {"family": "crt_growth", "lure": 2, "total": 10},
        ]
        notes = " ".join(analyze.observations(headline, family))
        self.assertIn("crt_rate", notes)
        self.assertIn("Data-quality", notes)


if __name__ == "__main__":
    unittest.main()
