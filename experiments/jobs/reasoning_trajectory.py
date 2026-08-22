"""Track a feature along the reasoning trace, not just at the answer position.

Every causal measurement in this study reads one static vector at the last prompt
token. That is the representation *just before answering*, which cannot say
anything about the reasoning itself -- and the behavioural results show reasoning
is exactly where the lure is resolved (2B: 55% lure without thinking, 21% with).

This job samples the feature along the sequence in both conditions:

    cue              the lure clause inside the prompt, where the trap is *read*
    prompt_last      the site every other experiment uses, kept as the anchor
    reasoning_0..100 item-relative quantiles through the generated trace, so
                     traces of wildly different length stay comparable
    pre_answer       the token the answer is emitted from -- the ``</think>``
                     position, so it exists only in the thinking arm

There is deliberately no cross-arm ``pre_answer_difference`` in the summary. Without a
``</think>`` the answer starts at the first generated token, so the position the answer
is emitted from *is* the last prompt token; both arms are read on the same prompt and
attention is causal, so that residual is the same number in both conditions. A
thinking-minus-non-thinking contrast there reduces exactly to the thinking arm's own
``pre_answer_minus_prompt_last`` and carries nothing from the no-thinking trace. The
summary records that under ``not_measured`` instead of publishing the degenerate value.

The hypothesis it can falsify: if the lure representation rises when the cue is
read and is *suppressed during deliberation*, the thinking condition should show a
falling trajectory that the no-thinking condition does not have. A feature that is
merely positional will look the same in both.

Two model loads, sequentially, as the study job does: the behaviour model to
generate the traces, then the analysis (Base) model to read activations over prompt
+ trace. Generation is greedy so the trace is reproducible and the two conditions
differ only by the thinking switch.

One approximation is deliberate and worth stating: the trace is generated *with* the
chat template but read back on the plain ``prompt + trace`` string, because the Base
analysis checkpoint the SAE was trained on has no chat template. The two conditions
stay apples-to-apples -- both are read the same way -- but the absolute token
positions differ from what the behaviour model saw.

Alongside the activation the job records the lure margin at each sampled position,
so the CSV can answer "did the readout move as the activation moved" instead of
leaving that to a separate run. That costs forwards (see ``_MARGIN_COST_NOTE``) and
is switched off with ``[trajectory] measure_margin = false``.
"""
# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch

from experiments.runners.config import load_toml, run_name, table
from mindscopex_analysis import (
    DEFAULT_ANALYSIS_PROFILE_KEY,
    LureCase,
    TokenPhase,
    answer_logprob_margin,
    capture_residual_stream,
    clear_device_cache,
    cue_span,
    default_sae_device,
    dtype_from_name,
    family_balanced_subset,
    find_subsequence,
    generate_qwen_text_response,
    get_qwen35_analysis_profile,
    instruct_lure_cases,
    load_qwen_language_model,
    load_qwen_scope_sae,
    load_qwen_text_generation_model,
    lure_dataset_cases,
    pearson,
    qwen_scope_sparse_feature_values,
    reasoning_phases,
    recommended_dtype_name,
    split_lure_cases,
)

# One sampled position costs TWO forwards, not one: the margin is a difference of
# two teacher-forced continuation logprobs and each answer string needs its own
# pass. The memo pays them once per *distinct* token position, so budget
# 2 x n_distinct_positions x n_traces forwards over prompt-length sequences on top of
# the single activation pass per trace.
_MARGIN_COST_NOTE = "2 forwards per distinct sampled position (one per answer string)"

TRAJECTORY_COLUMNS = (
    "case_id",
    "family",
    "case_condition",
    "mode",
    "thinking",
    "token_phase",
    "fraction",
    "token_index",
    "span_start",
    "span_end",
    "layer",
    "feature_id",
    "activation",
    # Only the cue row spans more than one token, so this column is filled only there:
    # putting a clause mean and a single-token read in one column made the cue look
    # systematically flatter than the reasoning phases purely from window width.
    "cue_span_mean",
    "is_topk",
    "is_firing",
    "margin_if_readout_available",
    # How many tokens the margin was actually conditioned on. Equals token_index + 1
    # unless a BPE merge moved the cut when the prefix was decoded and re-encoded.
    "margin_prefix_tokens",
    # The earlier phase that already sampled this exact token index ("" when this row
    # is the first read of the position). pre_answer is the </think> token, which the
    # last reasoning quantile also lands on, so the two rows are one measurement.
    "duplicate_of",
    "answer_label",
    "n_tokens",
    "has_think_end",
    "cue_located",
)

# The dataset arm the cue clause was subtracted from: having no cue there is the
# definition of the arm, not a matcher failure, so it is counted separately.
NEUTRAL_CONDITION = "neutral"

_PRE_ANSWER_NOT_MEASURED = (
    "no cross-arm contrast at the answer position. Without </think> the answer starts "
    "at the first generated token, so the no-thinking arm's answer position is its last "
    "prompt token; both arms are read on the same prompt under causal attention, so that "
    "residual is identical in the two conditions and the difference would reduce to "
    "per_mode.thinking.pre_answer_minus_prompt_last with no no-thinking-trace data in it"
)


def _log(message: str) -> None:
    print(f"[traj] {message}", flush=True)


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return path


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def phase_percent(phase: str) -> float | None:
    """The quantile percent inside a ``reasoning_<n>`` label, or None for other phases."""

    if not phase.startswith("reasoning_"):
        return None
    try:
        return float(phase.split("_", 1)[1])
    except ValueError:
        return None


def phase_sort_key(phase: str) -> tuple[int, float, str]:
    """Order phase labels along the sequence, numerically inside ``reasoning_*``.

    Sorting these labels as plain strings is the trap this exists to avoid:
    ``"reasoning_100"`` sorts between ``"reasoning_0"`` and ``"reasoning_25"``, so a
    lexicographic order puts the *last* phase of the trace second and any
    "last minus first" reads a mid-trace value instead.
    """

    percent = phase_percent(phase)
    if percent is not None:
        return (2, percent, phase)
    return ({"cue": 0, "prompt_last": 1, "pre_answer": 3}.get(phase, 4), 0.0, phase)


def reasoning_series(means: Mapping[str, float]) -> list[tuple[float, str, float]]:
    """``(percent, phase, value)`` for the reasoning quantiles, ordered by percent.

    Never trust the mapping's own order here: it comes from row iteration or from a
    string sort, and both put ``reasoning_100`` in the wrong place.
    """

    out: list[tuple[float, str, float]] = []
    for phase, value in means.items():
        percent = phase_percent(phase)
        if percent is not None:
            out.append((percent, phase, float(value)))
    return sorted(out)


def label_duplicate_positions(phases: Sequence[TokenPhase]) -> list[str]:
    """For each phase, the earlier phase that already sampled its token index.

    ``pre_answer`` is ``</think> - 1``, which is exactly where the last reasoning
    quantile lands, so those two rows are one measurement of one token. Labelling the
    repeat keeps per-position rates from weighting that token twice.
    """

    seen: dict[int, str] = {}
    labels: list[str] = []
    for phase in phases:
        labels.append(seen.get(phase.token_index, ""))
        seen.setdefault(phase.token_index, phase.phase)
    return labels


def cue_text_from_twin(hostile_prompt: str, neutral_prompt: str, *, min_words: int = 3) -> str:
    """The clause the hostile arm adds to its neutral twin, or "" if none is clear.

    ``goal_affordance_traps_v1`` carries no cue field -- ``note`` names only the lure
    *kind* (``intended_lure=local_efficiency``) and ``rationale`` explains the correct
    answer -- so the cue text has to be recovered from the pair. The neutral twin is
    the same scenario with the efficiency cue deleted, which makes the
    hostile-minus-neutral difference exactly the cue.

    The diff is word-level on purpose. A character-level common prefix/suffix cuts
    mid-word ("walk|ing"), and the matcher would then be handed a fragment starting
    inside a word, so a legitimate cue reads as a miss.
    """

    hostile_words = hostile_prompt.split()
    neutral_words = neutral_prompt.split()
    head = 0
    while (
        head < len(hostile_words)
        and head < len(neutral_words)
        and hostile_words[head] == neutral_words[head]
    ):
        head += 1
    tail = 0
    while (
        tail < len(hostile_words) - head
        and tail < len(neutral_words) - head
        and hostile_words[-1 - tail] == neutral_words[-1 - tail]
    ):
        tail += 1
    extra = hostile_words[head : len(hostile_words) - tail]
    return " ".join(extra) if len(extra) >= min_words else ""


def cue_texts_by_pair(cases: Sequence[LureCase]) -> dict[str, str]:
    """Map ``pair_id -> cue clause``, defined once per pair by hostile vs neutral.

    Defining the cue from the hostile/neutral contrast and then *searching* for that
    text in whichever arm is being read is what makes the other arms behave:
    ``explicit`` and ``counterfactual`` keep the clause verbatim and get a cue row,
    while ``neutral`` does not contain it at all and correctly gets none. Diffing each
    arm against neutral separately would instead hand the explicit arm its added
    requirement sentence and call that the cue.

    Must be fed the *unfiltered* dataset: a run restricted to ``conditions =
    ["hostile"]`` has already dropped the neutral twins the diff needs.
    """

    by_pair: dict[str, dict[str, str]] = {}
    for case in cases:
        if case.pair_id:
            by_pair.setdefault(case.pair_id, {})[case.condition] = case.prompt

    texts: dict[str, str] = {}
    for pair_id, arms in by_pair.items():
        hostile = arms.get("hostile", "")
        neutral = arms.get("neutral", "")
        if not hostile or not neutral:
            continue
        text = cue_text_from_twin(hostile, neutral)
        if text:
            texts[pair_id] = text
    return texts


def _margin_at_prefix(
    lm: Any,
    tokenizer: Any,
    full_ids: Sequence[int],
    case: LureCase,
    position: int,
) -> tuple[float | None, int]:
    """``(margin, prefix_tokens)`` for the prefix ending at ``position``.

    ``skip_special_tokens=False`` is load-bearing, not a default. ``raw_text`` is
    decoded with the specials kept, so ``full_ids`` ends in ``<|im_end|>`` and the
    no-thinking arm's last quantile sits exactly on it. Dropping specials on the way
    back out would score a prefix one token shorter than the one the activation was
    read on, silently, because the scorer still succeeds.

    The prefix is decoded and re-tokenised, so a BPE merge at the cut can still move
    the boundary by a token; both conditions are built the same way, so the comparison
    between them survives it, and the returned token count puts the actual conditioning
    length on the row instead of leaving it to be assumed. When the answer no longer
    continues the prefix cleanly the scorer raises and the margin is None, leaving the
    cell empty rather than filling it with a number measured on a different prompt.

    Prompt-internal positions (the cue) are scored the same way, but there the model
    is mid-sentence: read that value as a readout probe, not as a prediction of what
    the model would answer.
    """

    prefix = tokenizer.decode(full_ids[: position + 1], skip_special_tokens=False)
    prefix_tokens = len(tokenizer(prefix, add_special_tokens=True)["input_ids"])
    try:
        result = answer_logprob_margin(
            lm,
            prefix,
            correct_answer=case.correct_answer,
            lure_answer=case.lure_answer,
        )
    except ValueError:
        return None, prefix_tokens
    return result.margin, prefix_tokens


def _distinct_position_rows(rows: Sequence[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """One row per (trace, token index) in ``mode``, keeping the first read.

    Rates and correlations are per-position quantities, and the phase list samples one
    token twice: ``pre_answer`` repeats the ``</think>`` token the last reasoning
    quantile already read. Averaging over every emitted row would weight that token
    twice in the thinking arm and not at all in the no-thinking arm, which manufactures
    a cross-arm gap out of a duplicated row.
    """

    seen: set[tuple[str, int]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row["mode"]) != mode:
            continue
        key = (str(row["case_id"]), int(row["token_index"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def summarise_trajectory(
    rows: Sequence[dict[str, Any]],
    *,
    feature_id: int,
    layer: int,
    n_traces: int,
    cue_coverage: dict[str, Any],
    margin: dict[str, Any],
    trace_diagnostics: dict[str, int],
) -> dict[str, Any]:
    """The shape of the trajectory in each condition, from the emitted rows alone.

    Pure so the arithmetic that carries the study's headline claim can be exercised
    without a GPU: every number below is a function of the CSV rows.
    """

    def phase_values(mode: str, column: str) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for row in rows:
            if str(row["mode"]) == mode and row.get(column, "") != "":
                out.setdefault(str(row["token_phase"]), []).append(float(row[column]))
        # Ordered along the sequence, not lexicographically: see phase_sort_key.
        return {phase: out[phase] for phase in sorted(out, key=phase_sort_key)}

    modes = sorted({str(row["mode"]) for row in rows})
    summary: dict[str, Any] = {
        "feature_id": feature_id,
        "layer": layer,
        "n_traces": n_traces,
        "trace_diagnostics": dict(trace_diagnostics),
        "cue_coverage": cue_coverage,
        "margin": margin,
        # What this artifact does NOT contain, so a reader sees "not measured" rather
        # than an absence they have to notice.
        "not_measured": {"pre_answer_difference": _PRE_ANSWER_NOT_MEASURED},
        "phase_notes": {
            "cue": (
                "last token of the lure clause inside the prompt; fraction is -1 because "
                "the quantile scale covers only the generated span"
            ),
            "pre_answer": (
                "the token the answer is emitted from, i.e. </think> - 1. Thinking arm "
                "only, and there it is the same token the last reasoning quantile "
                "samples, so pre_answer_minus_prompt_last equals the last "
                "reasoning phase minus prompt_last by construction"
            ),
            "rates": (
                "fire_rate/topk_rate are averaged over distinct token positions per "
                "trace, so the repeated pre_answer row does not weight </think> twice. "
                "Not comparable with the fire_rate of runs made before this change, "
                "whose denominator was every emitted row"
            ),
        },
        "per_mode": {},
    }
    if not margin.get("enabled", False):
        summary["not_measured"]["margin_means"] = "measure_margin = false: no readout forward ran"
        summary["not_measured"]["activation_margin_pearson"] = (
            "measure_margin = false: no readout forward ran"
        )

    for mode in modes:
        means = {phase: _mean(values) for phase, values in phase_values(mode, "activation").items()}
        margin_means = {
            phase: _mean(values)
            for phase, values in phase_values(mode, "margin_if_readout_available").items()
        }
        series = reasoning_series(means)
        unique = _distinct_position_rows(rows, mode)
        paired = [
            (float(row["activation"]), float(row["margin_if_readout_available"]))
            for row in unique
            if row.get("margin_if_readout_available", "") != ""
        ]
        summary["per_mode"][mode] = {
            "phase_means": means,
            # A level, not a delta: AnswerMargin is lure - correct, so positive means
            # the lure answer is preferred. The repo's delta sign convention is about
            # baseline-vs-edited pairs and does not apply here.
            "margin_means": margin_means,
            # Negative means the feature fades as the model deliberates, which is the
            # suppression the thinking condition is supposed to show.
            "reasoning_drift": (series[-1][2] - series[0][2]) if len(series) > 1 else None,
            # Emitted so the artifact itself states which two phases were differenced.
            "reasoning_drift_phases": [series[0][1], series[-1][1]] if len(series) > 1 else [],
            "pre_answer_minus_prompt_last": (
                means["pre_answer"] - means["prompt_last"]
                if "pre_answer" in means and "prompt_last" in means
                else None
            ),
            "cue_minus_prompt_last": (
                means["cue"] - means["prompt_last"]
                if "cue" in means and "prompt_last" in means
                else None
            ),
            # Does the readout move with the activation along the trace? Positions
            # inside one trace are not independent, so this is a descriptive slope, not
            # a test -- and None, never 0.0, when there was nothing to correlate.
            "activation_margin_pearson": (
                pearson([a for a, _ in paired], [m for _, m in paired]) if len(paired) > 1 else None
            ),
            "n_margin_pairs": len(paired),
            "fire_rate_per_distinct_position": _mean(
                [1.0 if row["is_firing"] else 0.0 for row in unique]
            ),
            "topk_rate_per_distinct_position": _mean(
                [1.0 if row["is_topk"] else 0.0 for row in unique]
            ),
            "n_distinct_positions": len(unique),
            "n_rows": sum(1 for row in rows if str(row["mode"]) == mode),
        }

    if len(modes) == 2:
        drifts = [summary["per_mode"][mode]["reasoning_drift"] for mode in modes]
        endpoints = [summary["per_mode"][mode]["reasoning_drift_phases"] for mode in modes]
        if any(value is None for value in drifts):
            summary["not_measured"]["drift_difference"] = (
                "at least one arm has fewer than two reasoning phases to difference"
            )
        elif endpoints[0] != endpoints[1]:
            # Differencing a 0->75 drift against a 0->100 drift is not a like-for-like
            # contrast, and the number would look exactly like a real one.
            summary["not_measured"]["drift_difference"] = (
                f"the two arms span different phases ({endpoints[0]} vs {endpoints[1]}), "
                "so their drifts are not comparable"
            )
        else:
            summary["drift_difference"] = {
                "modes": modes,
                "value": drifts[1] - drifts[0],
                "phases": endpoints[1],
                "note": (f"{modes[1]} minus {modes[0]}; a positional feature drifts alike in both"),
            }
    return summary


def _resolve_env(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = table(config, "model")
    profile = get_qwen35_analysis_profile(
        str(model_cfg.get("profile", DEFAULT_ANALYSIS_PROFILE_KEY))
    )
    dtype = model_cfg.get("dtype", "auto")
    dtype = recommended_dtype_name() if dtype == "auto" else str(dtype)
    sae_device = model_cfg.get("sae_device", "auto")
    sae_device = default_sae_device() if sae_device == "auto" else str(sae_device)
    return {
        "profile": profile,
        # The analysis (Base) checkpoint has no chat template, so the trace is
        # generated by the matching behaviour model and read on the analysis one.
        "analysis_model_id": profile.analysis_model_id,
        "behavior_model_id": profile.behavior_model_id,
        "repo_id": profile.sae_repo_id,
        "dtype": dtype,
        "device_map": model_cfg.get("device_map", "auto"),
        "sae_device": sae_device,
        "sae_dtype": model_cfg.get("sae_dtype", dtype),
    }


def run(config_path: Path, output_root: Path) -> Path:
    config = load_toml(config_path)
    name = run_name(config)
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config.toml")

    fcfg = table(config, "feature")
    feature_id = int(fcfg["feature_id"])
    layer = int(fcfg["layer"])

    tcfg = table(config, "trajectory")
    phases = int(tcfg.get("phases", 5))
    max_new_tokens = int(tcfg.get("max_new_tokens", 512))
    thinking_modes = [bool(v) for v in (tcfg.get("thinking_modes") or [False, True])]
    seed = int(tcfg.get("seed", 42))
    measure_margin = bool(tcfg.get("measure_margin", True))

    data_cfg = table(config, "data")
    dataset = str(data_cfg.get("dataset", "goal_affordance_traps_v1"))
    cases = lure_dataset_cases(dataset)
    # Built before any filtering: the cue is defined by the hostile/neutral contrast
    # and a condition-restricted run no longer holds the neutral twin to diff against.
    cue_texts = cue_texts_by_pair(cases)
    conditions = data_cfg.get("conditions") or None
    if conditions:
        suffixes = tuple(f"_{c}" for c in conditions)
        cases = [case for case in cases if case.case_id.endswith(suffixes)]
    if bool(data_cfg.get("instruction", True)):
        cases = instruct_lure_cases(cases)
    _, test = split_lure_cases(
        cases,
        train_frac=float(data_cfg.get("train_frac", 0.6)),
        seed=int(data_cfg.get("split_seed", 0)),
    )
    max_items = int(data_cfg.get("max_items", 10))
    items = family_balanced_subset(test, max_cases=max_items) if max_items else test

    env = _resolve_env(config)
    manifest: dict[str, Any] = {
        "run_name": name,
        "job": "reasoning_trajectory",
        "started_at": _timestamp(),
        "feature_id": feature_id,
        "layer": layer,
        "phases": phases,
        "thinking_modes": thinking_modes,
        "measure_margin": measure_margin,
        "margin_cost": _MARGIN_COST_NOTE,
        "cue_source": "hostile_minus_neutral_twin",
        "n_cue_texts": len(cue_texts),
        "profile": env["profile"].key,
        "generation_model_id": env["behavior_model_id"],
        "analysis_model_id": env["analysis_model_id"],
        "sae_repo_id": env["repo_id"],
        "dataset": dataset,
        "n_items": len(items),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    _write_json(run_dir / "manifest.json", manifest)
    started = time.time()

    # --- phase 1: generate the traces --------------------------------------
    _log(f"generation phase: loading {env['behavior_model_id']}")
    model, tokenizer = load_qwen_text_generation_model(
        env["behavior_model_id"], device_map=env["device_map"], dtype=env["dtype"]
    )
    traces: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(items, start=1):
            for thinking in thinking_modes:
                response = generate_qwen_text_response(
                    model,
                    tokenizer,
                    case,
                    model_id=env["behavior_model_id"],
                    enable_thinking=thinking,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # greedy: the two conditions must differ only by
                    seed=seed,  # the thinking switch, not by sampling noise
                )
                traces.append(
                    {
                        "case_id": case.case_id,
                        "family": case.family,
                        "mode": response.mode,
                        "thinking": bool(thinking),
                        "raw_text": response.raw_text,
                        "answer": response.answer,
                        "answer_label": response.answer_label,
                        "output_tokens": response.output_tokens,
                        "has_thinking_block": response.has_thinking_block,
                    }
                )
            _log(f"generated {index}/{len(items)}")
    finally:
        del model, tokenizer
        clear_device_cache()
    _write_json(run_dir / "traces.json", traces)

    # --- phase 2: read the feature along each trace ------------------------
    _log(f"analysis phase: loading {env['analysis_model_id']}")
    lm = load_qwen_language_model(
        env["analysis_model_id"], device_map=env["device_map"], dtype=env["dtype"], dispatch=True
    )
    sae = load_qwen_scope_sae(
        env["repo_id"], layer, device=env["sae_device"], dtype=dtype_from_name(env["sae_dtype"])
    )
    rows: list[dict[str, Any]] = []
    by_case = {case.case_id: case for case in items}
    # Both traces of a case share the prompt, so the span is located once per case.
    cue_spans: dict[str, tuple[int, int] | None] = {}
    cue_misses = {"no_cue_text": 0, "expected_no_cue": 0, "not_found_in_prompt": 0}
    not_found_by_condition: dict[str, int] = {}
    skipped = {"empty_continuation": 0, "token_count_mismatch": 0}
    margin_failures = 0
    margin_prefix_mismatches = 0
    prompt_boundary_merges = 0

    try:
        tokenizer_close = getattr(lm.tokenizer, "tokenizer", lm.tokenizer)
        close_ids = tokenizer_close.encode("</think>", add_special_tokens=False)
        for index, trace in enumerate(traces, start=1):
            case = by_case[trace["case_id"]]
            full_text = case.prompt + trace["raw_text"]
            prompt_ids = tokenizer_close(case.prompt, add_special_tokens=True)["input_ids"]
            full_ids = tokenizer_close(full_text, add_special_tokens=True)["input_ids"]
            prompt_tokens = len(prompt_ids)
            if len(full_ids) <= prompt_tokens:
                _log(f"skip {trace['case_id']} ({trace['mode']}): empty continuation")
                skipped["empty_continuation"] += 1
                continue
            if list(full_ids[:prompt_tokens]) != list(prompt_ids):
                # A BPE merge across the prompt/continuation join moved the boundary, so
                # `prompt_last` is not quite the token the prompt alone ends on. Counted
                # rather than skipped: the shift is one token and both arms carry it.
                prompt_boundary_merges += 1

            if case.case_id not in cue_spans:
                cue_text = cue_texts.get(case.pair_id, "")
                if case.condition == NEUTRAL_CONDITION:
                    # The cue is *defined* as hostile-minus-neutral, so the neutral twin
                    # having none is the definition, not a matcher failure. Counting the
                    # two together would make the health check unreadable in exactly the
                    # run that needs it -- a twin-comparison run over both arms.
                    cue_misses["expected_no_cue"] += 1
                    cue_spans[case.case_id] = None
                elif not cue_text:
                    cue_misses["no_cue_text"] += 1
                    cue_spans[case.case_id] = None
                else:
                    # Decoded pieces, not convert_ids_to_tokens: Qwen's byte-level BPE
                    # writes a leading space as U+0120, which is not whitespace, so the
                    # raw token strings never match a plain-text cue.
                    pieces = tokenizer_close.batch_decode([[int(t)] for t in prompt_ids])
                    span = cue_span(pieces, cue_text)
                    if span is None:
                        # A real miss: this arm is supposed to contain the clause. The
                        # item gets no cue row rather than a guessed one, and the
                        # condition is recorded so a matcher regression is attributable.
                        cue_misses["not_found_in_prompt"] += 1
                        key = case.condition or "unknown"
                        not_found_by_condition[key] = not_found_by_condition.get(key, 0) + 1
                    cue_spans[case.case_id] = span
            span = cue_spans[case.case_id]

            think_end = None
            if trace["has_thinking_block"] and close_ids:
                found = find_subsequence(full_ids[prompt_tokens:], close_ids)
                if found is not None:
                    think_end = prompt_tokens + found + len(close_ids)

            residual = capture_residual_stream(lm, [full_text], [layer], token_position="all")[
                layer
            ]
            values = (
                qwen_scope_sparse_feature_values(residual, sae, [feature_id])
                .detach()
                .to(torch.float32)
                .cpu()
                .reshape(-1)
            )
            if values.numel() != len(full_ids):
                # The phase indices are computed from `tokenizer_close`; `values` comes
                # from the nnsight trace's own tokenisation. If those two ever disagree
                # (truncation at model_max_length, a path that adds a BOS) every index
                # is off, and clamping would stack the late phases onto the final token
                # -- which reads as the clean "positional feature, no suppression" null
                # rather than as a broken run. Refuse the trace instead.
                _log(
                    f"skip {trace['case_id']} ({trace['mode']}): tokenisation mismatch, "
                    f"{values.numel()} activations for {len(full_ids)} tokens"
                )
                skipped["token_count_mismatch"] += 1
                continue

            sampled: list[tuple[TokenPhase, int, int]] = []
            if span is not None:
                # Read at the last token of the cue clause: the first position that has
                # seen the whole trap. Fraction is -1 because the quantile scale only
                # covers the generated span and this position is inside the prompt.
                sampled.append((TokenPhase("cue", span[1] - 1, -1.0), span[0], span[1]))
            for phase in reasoning_phases(
                prompt_tokens, len(full_ids), phases=phases, think_end=think_end
            ):
                sampled.append((phase, phase.token_index, phase.token_index + 1))
            # No pre_answer row is invented for the no-thinking arm. With no </think>
            # the answer starts at the first generated token, so its "token the answer
            # is emitted from" is the last prompt token -- already emitted as
            # prompt_last, and identical to the thinking arm's prompt_last because both
            # arms read the same prompt under causal attention. A row there would add a
            # duplicate, not a measurement, and would make a cross-arm contrast look
            # available when it is not (see summary.not_measured).
            duplicates = label_duplicate_positions([phase for phase, _, _ in sampled])

            has_readout = bool(case.correct_answer.strip() and case.lure_answer.strip())
            scored = measure_margin and has_readout
            # pre_answer repeats the last reasoning quantile's token, so the per-position
            # memo keeps that pair of forwards from being paid twice.
            margins: dict[int, tuple[float | None, int]] = {}

            last = values.numel() - 1
            for (phase, span_start, span_end), duplicate_of in zip(
                sampled, duplicates, strict=True
            ):
                # In range by the invariant above; the clamp only guards the cue span,
                # which is located on a separately tokenised copy of the prompt.
                position = phase.token_index
                start = max(0, min(span_start, last))
                end = max(start + 1, min(span_end, last + 1))
                value = float(values[position])
                if scored and position not in margins:
                    margins[position] = _margin_at_prefix(
                        lm, tokenizer_close, full_ids, case, position
                    )
                    scored_margin, scored_tokens = margins[position]
                    if scored_margin is None:
                        margin_failures += 1
                    if scored_tokens != position + 1:
                        margin_prefix_mismatches += 1
                measured = margins.get(position)
                margin = measured[0] if measured is not None else None
                prefix_tokens = measured[1] if measured is not None else None
                rows.append(
                    {
                        "case_id": trace["case_id"],
                        "family": trace["family"],
                        # The dataset arm (hostile/neutral/...), not the thinking arm:
                        # `mode` carries thinking vs non-thinking. Both are on every row
                        # so pooled runs stay readable without the manifest.
                        "case_condition": case.condition,
                        "mode": trace["mode"],
                        "thinking": trace["thinking"],
                        "token_phase": phase.phase,
                        "fraction": phase.fraction,
                        "token_index": position,
                        "span_start": start,
                        "span_end": end,
                        # Repeated on every row: pooling several runs into one frame
                        # otherwise loses which feature and layer a row came from.
                        "layer": layer,
                        "feature_id": feature_id,
                        "activation": value,
                        # Only the multi-token cue clause gets a span mean. Filling it
                        # for single-token phases too (where it is just a copy of
                        # `activation`) put a ~20-token average and a single-token read
                        # in one column, so a plot across phases showed the cue flatter
                        # than the reasoning phases purely from window width.
                        "cue_span_mean": (
                            float(values[start:end].mean()) if end - start > 1 else ""
                        ),
                        # Two different questions, and they are not the same question.
                        # TopK is taken with no ReLU clamp, so a selected feature may
                        # carry a negative value: `!= 0` is membership in the TopK
                        # support (a selected value of exactly 0.0 would read as absent,
                        # a float-measure-zero case), `> 0` is the feature actually
                        # firing. The summary's fire_rate means the latter.
                        "is_topk": bool(value != 0.0),
                        "is_firing": bool(value > 0.0),
                        # Empty when the dataset carries no teacher-forced answer pair,
                        # or when measure_margin is off -- hence the column's name.
                        "margin_if_readout_available": "" if margin is None else margin,
                        "margin_prefix_tokens": "" if prefix_tokens is None else prefix_tokens,
                        "duplicate_of": duplicate_of,
                        "answer_label": trace["answer_label"],
                        "n_tokens": len(full_ids),
                        "has_think_end": think_end is not None,
                        "cue_located": span is not None,
                    }
                )
            if index % 4 == 0:
                _log(f"read {index}/{len(traces)} traces")
            _write_csv(run_dir / "reasoning_trajectory.csv", rows, TRAJECTORY_COLUMNS)
    finally:
        del lm, sae
        clear_device_cache()

    # --- summary: the shape of the trajectory in each condition ------------
    summary = summarise_trajectory(
        rows,
        feature_id=feature_id,
        layer=layer,
        n_traces=len(traces),
        cue_coverage={
            "cases_with_cue": sum(1 for value in cue_spans.values() if value is not None),
            "cases_seen": len(cue_spans),
            **cue_misses,
            # Which arm the matcher missed in, so a regression is attributable instead
            # of hiding inside one aggregate count.
            "not_found_by_condition": not_found_by_condition,
        },
        margin={
            "enabled": measure_margin,
            "cost": _MARGIN_COST_NOTE,
            "failed_positions": margin_failures,
            # Positions where decoding + re-encoding the prefix did not reproduce
            # token_index + 1 tokens, i.e. a BPE merge moved the scored cut.
            "prefix_length_mismatches": margin_prefix_mismatches,
        },
        trace_diagnostics={
            "skipped_empty_continuation": skipped["empty_continuation"],
            # Refused, not clamped: an index/activation length disagreement stacks the
            # late phases onto the final token and reads as a clean null result.
            "skipped_token_count_mismatch": skipped["token_count_mismatch"],
            # Not skipped -- traces where a BPE merge at the prompt/continuation join
            # moved `prompt_last` by a token. Both arms carry it alike.
            "prompt_boundary_merges": prompt_boundary_merges,
        },
    )
    _write_json(run_dir / "trajectory_summary.json", summary)
    manifest.update(
        {
            "finished_at": _timestamp(),
            "elapsed_seconds": round(time.time() - started, 1),
            "summary": summary,
        }
    )
    _write_json(run_dir / "manifest.json", manifest)
    _log(f"done: {json.dumps(summary['per_mode'], default=str)[:400]}")
    print(f"ARTIFACT_DIR={run_dir}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("outputs/experiments"), type=Path)
    args = parser.parse_args()
    run(args.config, args.output_root)


if __name__ == "__main__":
    main()
