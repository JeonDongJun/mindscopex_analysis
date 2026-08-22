"""Pure-logic tests for the reasoning_trajectory job.

The job itself needs two model loads, so only the parts that decide *what* gets
measured are testable here: recovering the lure cue from the hostile/neutral pair,
and the phase list the CSV rows are built from.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mindscopex_analysis import LureCase, TokenPhase, cue_span, lure_dataset_cases, reasoning_phases

ROOT = Path(__file__).resolve().parents[1]


def _load_job():
    """Import the job by path: experiments/jobs is not an importable package."""

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    path = ROOT / "experiments" / "jobs" / "reasoning_trajectory.py"
    spec = importlib.util.spec_from_file_location("reasoning_trajectory_job", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


job = _load_job()


def _case(case_id: str, condition: str, question: str, pair_id: str = "p1") -> LureCase:
    return LureCase(
        case_id=case_id,
        family="target_transport",
        prompt=question,
        correct_answer=" drive there",
        lure_answer=" walk there",
        pair_id=pair_id,
        condition=condition,
    )


HOSTILE = (
    "I want my car washed. The car is beside me and the car wash is 50 meters away. "
    "For such a short distance, walking there would be quicker. What should I do?"
)
NEUTRAL = (
    "I want my car washed. The car is beside me and the car wash is 50 meters away. "
    "What should I do?"
)
EXPLICIT = (
    "I want my car washed. The car is beside me and the car wash is 50 meters away. "
    "For such a short distance, walking there would be quicker. "
    "One requirement matters here: the car must arrive. What should I do?"
)


class CueTextFromTwinTests(unittest.TestCase):
    def test_recovers_the_clause_the_hostile_arm_adds(self) -> None:
        self.assertEqual(
            job.cue_text_from_twin(HOSTILE, NEUTRAL),
            "For such a short distance, walking there would be quicker.",
        )

    def test_identical_prompts_yield_no_cue(self) -> None:
        self.assertEqual(job.cue_text_from_twin(NEUTRAL, NEUTRAL), "")

    def test_a_one_word_difference_is_not_trusted_as_a_cue(self) -> None:
        # A single changed token is far more likely to be wording drift than a lure
        # clause, and handing it to the matcher would put a "cue" row anywhere.
        self.assertEqual(job.cue_text_from_twin("a b c d", "a b x d"), "")

    def test_a_shared_instruction_prefix_is_not_part_of_the_cue(self) -> None:
        # Both arms are wrapped by instruct_lure_cases, so the instruction has to fall
        # out of the diff or every cue would start at token 0.
        prefix = "Solve the problem, answer in one line.\n\n"
        cue = job.cue_text_from_twin(prefix + HOSTILE, prefix + NEUTRAL)
        self.assertEqual(cue, "For such a short distance, walking there would be quicker.")

    def test_cue_is_word_aligned_so_the_matcher_can_find_it(self) -> None:
        # The reason the diff is word-level: the recovered text must still be findable
        # in a tokenised prompt, and a mid-word cut is not.
        cue = job.cue_text_from_twin(HOSTILE, NEUTRAL)
        pieces = [HOSTILE[i : i + 3] for i in range(0, len(HOSTILE), 3)]
        span = cue_span(pieces, cue)
        self.assertIsNotNone(span)
        assert span is not None
        start, end = span
        self.assertLess(start, end)
        self.assertIn("walking there would be quicker", "".join(pieces[start:end]))


class CueTextsByPairTests(unittest.TestCase):
    def test_one_cue_per_pair_from_the_hostile_neutral_contrast(self) -> None:
        texts = job.cue_texts_by_pair(
            [
                _case("c_hostile", "hostile", HOSTILE),
                _case("c_neutral", "neutral", NEUTRAL),
                _case("c_explicit", "explicit", EXPLICIT),
            ]
        )
        self.assertEqual(list(texts), ["p1"])
        self.assertEqual(texts["p1"], "For such a short distance, walking there would be quicker.")

    def test_the_pair_cue_is_verbatim_in_explicit_but_absent_from_neutral(self) -> None:
        # This is why the cue is defined once per pair and then searched for: the
        # explicit arm keeps the clause, the neutral arm genuinely has none.
        cue = job.cue_texts_by_pair(
            [_case("c_hostile", "hostile", HOSTILE), _case("c_neutral", "neutral", NEUTRAL)]
        )["p1"]
        self.assertIn(cue, EXPLICIT)
        self.assertNotIn(cue, NEUTRAL)

    def test_a_pair_without_a_neutral_twin_is_skipped(self) -> None:
        self.assertEqual(job.cue_texts_by_pair([_case("c_hostile", "hostile", HOSTILE)]), {})

    def test_cases_without_a_pair_id_are_ignored(self) -> None:
        loose = LureCase(
            case_id="x", family="f", prompt=HOSTILE, correct_answer=" a", lure_answer=" b"
        )
        self.assertEqual(job.cue_texts_by_pair([loose]), {})

    def test_every_v1_pair_yields_a_cue_the_hostile_prompt_contains(self) -> None:
        cases = lure_dataset_cases("goal_affordance_traps_v1")
        texts = job.cue_texts_by_pair(cases)
        pairs = {case.pair_id for case in cases if case.pair_id}
        self.assertEqual(set(texts), pairs)
        by_id = {case.case_id: case for case in cases}
        for pair_id, cue in texts.items():
            hostile = next(
                case
                for case in by_id.values()
                if case.pair_id == pair_id and case.condition == "hostile"
            )
            neutral = next(
                case
                for case in by_id.values()
                if case.pair_id == pair_id and case.condition == "neutral"
            )
            self.assertIn(cue, hostile.prompt)
            self.assertNotIn(cue, neutral.prompt)


class SampledPhaseTests(unittest.TestCase):
    """The phase list the job builds -- and what it deliberately does not contain."""

    @staticmethod
    def _sampled(prompt_tokens: int, total: int, think_end: int | None) -> list[TokenPhase]:
        # Exactly what the job samples now: reasoning_phases, with nothing invented
        # on top of it.
        return list(reasoning_phases(prompt_tokens, total, phases=5, think_end=think_end))

    def test_the_no_thinking_arm_gets_no_pre_answer_row(self) -> None:
        # Not an oversight. With no </think> the answer starts at the first generated
        # token, so the position it is emitted from is the last prompt token, which is
        # already emitted as prompt_last -- and is the same number in both arms, since
        # both read the same prompt under causal attention. A pre_answer row there is a
        # duplicate, not a measurement, and it made a cross-arm contrast at the answer
        # position look available when no such contrast exists.
        labels = [phase.phase for phase in self._sampled(10, 40, None)]
        self.assertNotIn("pre_answer", labels)
        self.assertIn("prompt_last", labels)

    def test_thinking_pre_answer_repeats_the_last_reasoning_quantile(self) -> None:
        sampled = self._sampled(10, 40, 30)
        by_phase = {phase.phase: phase.token_index for phase in sampled}
        self.assertEqual(by_phase["pre_answer"], 29)
        # </think> - 1 is where the last quantile lands too, so the two rows are one
        # measurement of one token and the repeat has to be labelled as such.
        self.assertEqual(by_phase["pre_answer"], by_phase["reasoning_100"])
        labels = job.label_duplicate_positions(sampled)
        self.assertEqual(labels[-1], "reasoning_100")
        self.assertEqual([label for label in labels[:-1] if label], [])

    def test_duplicate_labels_name_the_first_read_of_the_position(self) -> None:
        phases = [
            TokenPhase("cue", 4, -1.0),
            TokenPhase("prompt_last", 9, 0.0),
            TokenPhase("reasoning_0", 9, 0.0),
        ]
        self.assertEqual(job.label_duplicate_positions(phases), ["", "", "prompt_last"])

    def test_exactly_one_pre_answer_row_per_trace(self) -> None:
        for think_end in (None, 30):
            labels = [phase.phase for phase in self._sampled(10, 40, think_end)]
            self.assertLessEqual(labels.count("pre_answer"), 1)


REASONING_PHASES_5 = [
    "reasoning_0",
    "reasoning_25",
    "reasoning_50",
    "reasoning_75",
    "reasoning_100",
]


def _row(
    mode: str,
    phase: str,
    activation: float,
    token_index: int,
    *,
    case_id: str = "c1",
    margin: float | str = "",
) -> dict:
    return {
        "case_id": case_id,
        "mode": mode,
        "token_phase": phase,
        "token_index": token_index,
        "activation": activation,
        "is_firing": activation > 0.0,
        "is_topk": activation != 0.0,
        "margin_if_readout_available": margin,
    }


def _trace_rows(mode: str, activations: dict[str, float], *, case_id: str = "c1") -> list[dict]:
    """Rows for one trace, indexed the way reasoning_phases indexes a real one."""

    think_end = 400 if "pre_answer" in activations else None
    phases = list(reasoning_phases(50, 500, phases=5, think_end=think_end))
    return [
        _row(mode, phase.phase, activations[phase.phase], phase.token_index, case_id=case_id)
        for phase in phases
        if phase.phase in activations
    ]


def _summarise(rows: list[dict], *, margin_enabled: bool = True) -> dict:
    return job.summarise_trajectory(
        rows,
        feature_id=2144,
        layer=17,
        n_traces=len({(row["case_id"], row["mode"]) for row in rows}),
        cue_coverage={"cases_with_cue": 1, "cases_seen": 1},
        margin={"enabled": margin_enabled},
        trace_diagnostics={},
    )


class ReasoningDriftTests(unittest.TestCase):
    """phases=5 is the whole point of these tests.

    A phases=3 run labels its quantiles reasoning_0/50/100, which happen to sort the
    same lexicographically as numerically, so the bug pinned here is invisible there.
    Five phases is what both shipped configs use.
    """

    def test_a_five_phase_run_emits_labels_that_sort_wrong_as_strings(self) -> None:
        labels = [
            phase.phase
            for phase in reasoning_phases(50, 500, phases=5, think_end=400)
            if phase.phase.startswith("reasoning_")
        ]
        self.assertEqual(labels, REASONING_PHASES_5)
        # The trap: as strings, the *last* phase of the trace sorts second, so a
        # "last minus first" over a string-sorted mapping silently reads reasoning_75.
        self.assertEqual(sorted(labels)[-1], "reasoning_75")

    def test_reasoning_series_orders_by_percent_not_by_label(self) -> None:
        means = dict(zip(REASONING_PHASES_5, (1.0, 2.0, 3.0, 4.0, 5.0), strict=True))
        string_sorted = {key: means[key] for key in sorted(means)}
        series = job.reasoning_series(string_sorted)
        self.assertEqual([phase for _, phase, _ in series], REASONING_PHASES_5)
        self.assertEqual(series[-1][2], 5.0)

    def test_drift_differences_the_hundred_percent_phase(self) -> None:
        rows = _trace_rows("thinking", dict(zip(REASONING_PHASES_5, (1.0, 2.0, 3.0, 4.0, 5.0))))
        block = _summarise(rows)["per_mode"]["thinking"]
        # 5 - 1, not the 4 - 1 a lexicographic order produces.
        self.assertAlmostEqual(block["reasoning_drift"], 4.0)
        self.assertEqual(block["reasoning_drift_phases"], ["reasoning_0", "reasoning_100"])

    def test_a_late_collapse_is_not_reported_as_a_flat_trajectory(self) -> None:
        # The exact trajectory shape the hypothesis predicts: flat through 75% and
        # suppressed only at the end. Under the string ordering this reported 0.0 --
        # "positional feature, no suppression" -- on the strongest possible signal.
        flat_then_drop = dict(zip(REASONING_PHASES_5, (2.0, 2.0, 2.0, 2.0, -2.0)))
        block = _summarise(_trace_rows("thinking", flat_then_drop))["per_mode"]["thinking"]
        self.assertAlmostEqual(block["reasoning_drift"], -4.0)

    def test_drift_difference_is_thinking_minus_non_thinking(self) -> None:
        rows = _trace_rows("non_thinking", dict(zip(REASONING_PHASES_5, (1.0, 1.0, 1.0, 1.0, 1.0))))
        rows += _trace_rows("thinking", dict(zip(REASONING_PHASES_5, (1.0, 1.0, 1.0, 1.0, -1.0))))
        difference = _summarise(rows)["drift_difference"]
        self.assertEqual(difference["modes"], ["non_thinking", "thinking"])
        self.assertAlmostEqual(difference["value"], -2.0)
        self.assertEqual(difference["phases"], ["reasoning_0", "reasoning_100"])

    def test_arms_that_span_different_phases_are_not_differenced(self) -> None:
        # A 0->75 drift minus a 0->100 drift is not a like-for-like contrast, and the
        # number would be indistinguishable from a real one.
        rows = _trace_rows("thinking", dict(zip(REASONING_PHASES_5, (1.0, 1.0, 1.0, 1.0, -1.0))))
        rows += [
            _row("non_thinking", "reasoning_0", 1.0, 50, case_id="c2"),
            _row("non_thinking", "reasoning_75", 1.0, 312, case_id="c2"),
        ]
        summary = _summarise(rows)
        self.assertNotIn("drift_difference", summary)
        self.assertIn("different phases", summary["not_measured"]["drift_difference"])


class TrajectorySummaryTests(unittest.TestCase):
    def test_phase_means_are_ordered_along_the_sequence(self) -> None:
        rows = [
            _row("thinking", "pre_answer", 1.0, 399),
            _row("thinking", "reasoning_100", 1.0, 399),
            _row("thinking", "reasoning_25", 1.0, 195),
            _row("thinking", "prompt_last", 1.0, 49),
            _row("thinking", "cue", 1.0, 20),
        ]
        means = _summarise(rows)["per_mode"]["thinking"]["phase_means"]
        self.assertEqual(
            list(means), ["cue", "prompt_last", "reasoning_25", "reasoning_100", "pre_answer"]
        )

    def test_rates_count_each_token_position_once(self) -> None:
        # Both arms read the same four positions with the same values; the thinking arm
        # additionally emits pre_answer on the token reasoning_100 already sampled. Any
        # difference between the two fire rates here comes from the duplicated row.
        positions = [("cue", 1.0, 20), ("prompt_last", 1.0, 49), ("reasoning_0", 0.0, 50)]
        rows = [_row("non_thinking", *item, case_id="c1") for item in positions]
        rows.append(_row("non_thinking", "reasoning_100", 0.0, 499, case_id="c1"))
        rows += [_row("thinking", *item, case_id="c1") for item in positions]
        rows.append(_row("thinking", "reasoning_100", 0.0, 399, case_id="c1"))
        rows.append(_row("thinking", "pre_answer", 0.0, 399, case_id="c1"))

        per_mode = _summarise(rows)["per_mode"]
        self.assertEqual(per_mode["thinking"]["n_rows"], 5)
        self.assertEqual(per_mode["thinking"]["n_distinct_positions"], 4)
        self.assertAlmostEqual(
            per_mode["thinking"]["fire_rate_per_distinct_position"],
            per_mode["non_thinking"]["fire_rate_per_distinct_position"],
        )
        self.assertAlmostEqual(per_mode["thinking"]["fire_rate_per_distinct_position"], 0.5)
        self.assertAlmostEqual(per_mode["thinking"]["topk_rate_per_distinct_position"], 0.5)

    def test_the_rate_basis_caveat_is_in_the_artifact(self) -> None:
        # The caveat has to survive into trajectory_summary.json: a reader pooling this
        # run with an older one must be able to see the denominator changed.
        notes = _summarise(_trace_rows("thinking", {"prompt_last": 1.0}))["phase_notes"]
        self.assertIn("distinct token positions", notes["rates"])
        self.assertIn("Not comparable", notes["rates"])

    def test_pearson_is_none_when_no_margin_was_measured(self) -> None:
        rows = _trace_rows("thinking", dict(zip(REASONING_PHASES_5, (1.0, 2.0, 3.0, 4.0, 5.0))))
        summary = _summarise(rows, margin_enabled=False)
        block = summary["per_mode"]["thinking"]
        # 0.0 would read as "measured, no association" from a run that made zero
        # readout forwards.
        self.assertIsNone(block["activation_margin_pearson"])
        self.assertEqual(block["n_margin_pairs"], 0)
        self.assertEqual(block["margin_means"], {})
        self.assertIn("activation_margin_pearson", summary["not_measured"])

    def test_pearson_is_computed_from_distinct_positions_when_margins_exist(self) -> None:
        rows = [
            _row("thinking", "prompt_last", 1.0, 49, margin=1.0),
            _row("thinking", "reasoning_100", 3.0, 399, margin=3.0),
            _row("thinking", "pre_answer", 3.0, 399, margin=3.0),
        ]
        block = _summarise(rows)["per_mode"]["thinking"]
        self.assertEqual(block["n_margin_pairs"], 2)
        self.assertAlmostEqual(block["activation_margin_pearson"], 1.0)

    def test_no_cross_arm_pre_answer_difference_is_published(self) -> None:
        rows = _trace_rows("non_thinking", {"prompt_last": 1.0})
        rows += _trace_rows("thinking", {"prompt_last": 1.0, "pre_answer": 0.2})
        summary = _summarise(rows)
        self.assertNotIn("pre_answer_difference", summary)
        self.assertIn("pre_answer_difference", summary["not_measured"])
        # And the quantity such a difference would have reduced to is still there, in
        # the arm it actually belongs to.
        self.assertAlmostEqual(
            summary["per_mode"]["thinking"]["pre_answer_minus_prompt_last"], -0.8
        )
        self.assertIsNone(summary["per_mode"]["non_thinking"]["pre_answer_minus_prompt_last"])


class MarginPrefixTests(unittest.TestCase):
    """The prefix the margin is scored on must be the sequence the activation was read on."""

    PIECES = ("Drive ", "there", ".", "<|im_end|>")

    class _Tokenizer:
        def __init__(self, pieces: tuple[str, ...]) -> None:
            self.pieces = pieces

        def decode(self, ids, skip_special_tokens: bool = False) -> str:
            out = [self.pieces[int(i)] for i in ids]
            if skip_special_tokens:
                out = [piece for piece in out if not piece.startswith("<|")]
            return "".join(out)

        def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
            ids: list[int] = []
            rest = text
            while rest:
                for index, piece in enumerate(self.pieces):
                    if rest.startswith(piece):
                        ids.append(index)
                        rest = rest[len(piece) :]
                        break
                else:  # pragma: no cover - the stub vocabulary covers the fixtures
                    raise AssertionError(f"untokenisable remainder: {rest!r}")
            return {"input_ids": ids}

    def _run(self, position: int, result):
        tokenizer = self._Tokenizer(self.PIECES)
        seen: list[str] = []

        def fake_margin(lm, prompt, **kwargs):
            seen.append(prompt)
            if isinstance(result, Exception):
                raise result
            return result

        case = _case("c1", "hostile", "Drive there.")
        with mock.patch.object(job, "answer_logprob_margin", fake_margin):
            margin, prefix_tokens = job._margin_at_prefix(
                object(), tokenizer, [0, 1, 2, 3], case, position
            )
        return seen, margin, prefix_tokens

    def test_the_eos_token_stays_in_the_scored_prefix(self) -> None:
        # The no-thinking arm's last quantile sits exactly on <|im_end|>. Decoding with
        # skip_special_tokens=True dropped it, so the margin was scored on a prefix one
        # token shorter than the position the activation came from -- silently, because
        # the scorer still succeeds.
        seen, margin, prefix_tokens = self._run(3, SimpleNamespace(margin=0.5))
        self.assertEqual(seen, ["Drive there.<|im_end|>"])
        self.assertEqual(prefix_tokens, 4)
        self.assertEqual(margin, 0.5)

    def test_the_reported_prefix_length_is_the_one_the_scorer_saw(self) -> None:
        _, _, prefix_tokens = self._run(1, SimpleNamespace(margin=0.0))
        self.assertEqual(prefix_tokens, 2)

    def test_a_scorer_failure_leaves_the_margin_empty_but_keeps_the_length(self) -> None:
        _, margin, prefix_tokens = self._run(3, ValueError("answer no longer continues"))
        self.assertIsNone(margin)
        self.assertEqual(prefix_tokens, 4)


class CsvContractTests(unittest.TestCase):
    def test_the_work_order_columns_are_all_present(self) -> None:
        for column in (
            "case_id",
            "token_phase",
            "token_index",
            "layer",
            "feature_id",
            "activation",
            "is_topk",
            "margin_if_readout_available",
        ):
            self.assertIn(column, job.TRAJECTORY_COLUMNS)

    def test_the_span_mean_column_holds_one_quantity(self) -> None:
        # activation_span_mean carried a ~20-token clause mean on the cue row and a copy
        # of `activation` everywhere else, so a plot across phases compared different
        # window widths. The clause mean now has its own column and only the cue fills it.
        self.assertIn("cue_span_mean", job.TRAJECTORY_COLUMNS)
        self.assertNotIn("activation_span_mean", job.TRAJECTORY_COLUMNS)

    def test_repeated_positions_and_scored_length_are_visible_in_the_csv(self) -> None:
        self.assertIn("duplicate_of", job.TRAJECTORY_COLUMNS)
        self.assertIn("margin_prefix_tokens", job.TRAJECTORY_COLUMNS)

    def test_topk_membership_and_firing_are_separate_columns(self) -> None:
        # TopK is taken with no ReLU clamp, so a kept value can be negative: the two
        # flags are not the same question and collapsing them loses one of them.
        self.assertIn("is_topk", job.TRAJECTORY_COLUMNS)
        self.assertIn("is_firing", job.TRAJECTORY_COLUMNS)
        self.assertNotIn("is_active", job.TRAJECTORY_COLUMNS)


if __name__ == "__main__":
    unittest.main()
