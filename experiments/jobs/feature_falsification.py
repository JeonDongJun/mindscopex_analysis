"""Try to falsify a feature: does it track the lure structure, or the surface?

A feature can pass every causal gate and still be the wrong explanation, because the
trap items differ from their controls in more than the trap: same template, same answer
strings, overlapping vocabulary -- and, in this dataset, a different sentence count.
This job attacks the feature from the directions that would make it a surface artifact,
and reports the errors rather than a single verdict.

    condition profile   hostile / explicit / neutral / counterfactual share the scenario
                        and the answer strings and differ in structure. hostile-vs-
                        neutral is the headline, but neutral is built by DELETING the
                        cue sentence, so the arm is not length-matched; the length
                        block, the length-matched pairing and the partial correlation
                        all exist to keep that confound visible instead of hidden.
    lexical proxy       the condition that keeps the cue sentence verbatim while
                        dropping the lure relation. It is chosen for that STRUCTURAL
                        property, not because it is the most lexically matched arm: on
                        v1 it is the LEAST matched one (counterfactual 0.57 against
                        explicit 0.79), which the artifact prints beside it as
                        per_condition_lexical_overlap. A proxy, not the F1 arm: a real
                        lexical-injection item reuses the cue words in a scenario with
                        no goal conflict at all, and no dataset here has one.
    template control    the reference dataset's MATCHED CONTROL prompts -- same
                        "...\\nAnswer:" template, same numbers and entities, no trap --
                        but from a DIFFERENT TASK, which is why this criterion is capped
                        at proxy and can never reach `tested`, and why it carries the
                        largest uncontrolled length gap in the job (+26 tokens on v1).
                        The reference hostile prompts are scored too but reported
                        separately as transfer: they are themselves lure items, so
                        activation there is evidence for generalisation, not for
                        template-tracking, and scoring them as a "template control"
                        (which this job used to do) inverts the reading.
    paraphrase          same structure, different wording. Computed only from scenarios
                        that exist in two wordings, and NOT TESTED on any dataset in the
                        repo today, because none has a second wording of anything. The
                        block that used to carry this name averaged activation per
                        template_id, and template_id is a family label in every one of
                        these datasets -- so it reported across-family variance under the
                        paraphrase heading. That block still exists, named for what it
                        actually measures, and the F3 slot says the real answer.
    answer confound     correlation between activation and which answer strings the item
                        uses, plus answer length -- the two surface variables that have
                        already bitten this study.
    error audit         at a threshold picked on the discovery split, the hostile items
                        where the feature stays silent and the control items where it
                        fires, each with its prompt, its rank among the TopK actually
                        kept at that position, and its answer margin. Detection always
                        requires the feature to FIRE: a TopK SAE writes exactly 0.0
                        wherever the feature is off its support, so a discovery quantile
                        that lands on 0.0 would otherwise class every silent control as
                        an error and no silent positive as one -- both counts inverted.

Every contrast published here carries a length gap, and they do not all run the same way
(+15 tokens against neutral, -23 against counterfactual, +26 against the reference
control on v1), so every one of them goes through the same block: raw AUC, the two arm
token means, the length-matched subset, and a caveat naming the gap. A number whose name
asserts a test that was not run is deleted or renamed, never softened -- an unmeasured
quantity is emitted as null with a NOT TESTED reason, so that it cannot be read as a
measured zero.

Reported as separation statistics (AUC, mean gap) rather than pass/fail: a feature that
is 70% structural and 30% positional is the likely truth, and a boolean would hide it.

Every artifact carries ``acceptance``: one entry per work-order criterion (F1..F5) with
status tested / proxy / not_tested and the measurement that decided it. A criterion the
data cannot support gets that marker rather than a number under a name it has not
earned, and ``[audit].strict_arms`` turns a missing arm into an abort before the model
is even loaded.
"""
# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import re
import shutil
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
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
    mean_or_none,
    DEFAULT_ANALYSIS_PROFILE_KEY,
    LureCase,
    active_prompt_features,
    answer_logprob_margin,
    capture_layer_residuals,
    default_sae_device,
    dtype_from_name,
    get_qwen35_analysis_profile,
    instruct_lure_cases,
    load_qwen_language_model,
    load_qwen_scope_sae,
    lure_dataset_cases,
    pearson,
    qwen_scope_feature_preactivations,
    qwen_scope_sparse_feature_values,
    recommended_dtype_name,
    split_lure_cases,
)
from mindscopex_analysis.stats import paired_summary

REFERENCE_LURE = "reference_lure"
REFERENCE_CONTROL = "reference_control"


def _log(message: str) -> None:
    print(f"[falsify] {message}", flush=True)


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


def _constant(values: Sequence[float]) -> bool:
    """True when there is no variation to correlate against (empty counts as constant)."""

    return not values or (max(values) - min(values)) <= 1e-12


def _corr(a: Sequence[float], b: Sequence[float]) -> float | None:
    """``pearson``, but ``None`` rather than 0.0 when either side is constant.

    ``pearson`` maps an undefined correlation to 0.0, which inside an artifact reads as
    the measured claim "no relation". The two degenerate cases this job actually hits
    are a missing arm (the condition indicator is constant) and a silent feature (the
    activation is constant), and the honest output in both is "not measurable".
    """

    if len(a) != len(b) or len(a) < 2 or _constant(a) or _constant(b):
        return None
    return pearson(a, b)


def _auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    """P(a random positive scores above a random negative); 0.5 means no separation.

    Threshold-free, so it does not depend on where a cutoff happens to fall -- which
    matters because the cutoff here is itself estimated.
    """

    if not positive or not negative:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def _spread(values: Sequence[float]) -> dict[str, Any]:
    """Mean/median/min/max in one row; ``None`` fields when there is nothing to describe."""

    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    numbers = [float(value) for value in values]
    return {
        "n": len(numbers),
        "mean": _mean(numbers),
        "median": statistics.median(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


# ------------------------------------------------------- surface-form measures


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


def _jaccard(left: str, right: str) -> float:
    """Word-level overlap of two prompts, as |A n B| / |A u B|.

    The F1 question is "how much of the hostile surface does this control keep?", and
    that is a set question, not an edit-distance one: a control that reuses every cue
    word in a different order is still lexically matched.
    """

    a, b = set(_word_tokens(left)), set(_word_tokens(right))
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _partial_correlation(
    x: Sequence[float], y: Sequence[float], z: Sequence[float]
) -> float | None:
    """corr(x, y) with the linear part of z removed from both.

    Used for "does the activation gap survive prompt length?". It is only a linear
    adjustment -- it cannot rescue a design where the two arms barely overlap in length
    -- which is why the length-matched pairing is reported next to it rather than
    instead of it. Returns None when either residual is degenerate, because a partial
    correlation of a constant is undefined and 0.0 would read as "no relation".

    That degeneracy is checked on x and y DIRECTLY rather than inferred from the
    correlations: ``pearson`` maps a constant input to 0.0, so the all-ones condition
    indicator that an empty control arm produces would otherwise pass the denominator
    test and be published as a confident "the condition effect is exactly zero once
    length is controlled". A constant z is not degenerate -- removing a constant
    regressor removes nothing, and the answer is then just corr(x, y).
    """

    if not len(x) == len(y) == len(z):
        raise ValueError("partial correlation needs equal-length sequences")
    if len(x) < 3:
        return None
    if _constant(x) or _constant(y):
        return None
    r_xy, r_xz, r_yz = pearson(x, y), pearson(x, z), pearson(y, z)
    denominator = math.sqrt(max(0.0, (1.0 - r_xz**2) * (1.0 - r_yz**2)))
    if denominator < 1e-12:
        return None
    return max(-1.0, min(1.0, (r_xy - r_xz * r_yz) / denominator))


def _length_matched_pairs(
    positive: Sequence[Mapping[str, Any]],
    negative: Sequence[Mapping[str, Any]],
    *,
    key: str = "prompt_tokens",
    caliper: float = 4.0,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Greedy nearest-length 1:1 pairing without replacement, inside ``caliper``.

    The hostile arm is the neutral arm plus a cue sentence, so a hostile-vs-neutral gap
    is partly a length gap. Matching discards the items with no counterpart instead of
    adjusting them: that can only shrink the reported separation, never manufacture one,
    which is the right direction for an error to run in a falsification job.

    Deterministic given the input order: positives are consumed shortest-first and ties
    on the length gap go to the earliest remaining negative.
    """

    if caliper < 0:
        raise ValueError("caliper must be non-negative")
    remaining = list(negative)
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for row in sorted(positive, key=lambda item: float(item[key])):
        best_index: int | None = None
        best_gap = math.inf
        for index, candidate in enumerate(remaining):
            gap = abs(float(row[key]) - float(candidate[key]))
            if gap <= caliper and gap < best_gap:
                best_index, best_gap = index, gap
        if best_index is not None:
            pairs.append((row, remaining.pop(best_index)))
    return pairs


def _contrast(
    positive_rows: Sequence[Mapping[str, Any]],
    other_rows: Sequence[Mapping[str, Any]],
    *,
    positive_label: str,
    other_label: str,
    caliper: float = 4.0,
    max_gap_tokens: float = 2.0,
) -> dict[str, Any]:
    """One arm against the positive arm: raw separation AND the length adjustment.

    Every contrast this job publishes is confounded with prompt length, and they are not
    confounded in the same direction: on v1 the positive arm runs +15 tokens against
    ``neutral``, -23 against ``counterfactual`` and +26 against ``reference_control``.
    Publishing a length-matched AUC for one of them and a raw AUC for the others invites
    the reader to treat all four as adjusted, so they all get this block, and the caveat
    naming the gap goes in the ARTIFACT rather than in a comment.

    ``length_matched.auc`` is None when no pair of items falls inside the caliper. That
    is the honest answer -- the arms do not overlap in length, so this contrast cannot be
    length-controlled at all here -- and ``caveat`` says NOT TESTED in words.
    """

    positive = [float(row["activation"]) for row in positive_rows]
    other = [float(row["activation"]) for row in other_rows]
    positive_tokens = [float(row["prompt_tokens"]) for row in positive_rows]
    other_tokens = [float(row["prompt_tokens"]) for row in other_rows]
    both = list(positive_rows) + list(other_rows)
    token_gap = (
        _mean(positive_tokens) - _mean(other_tokens) if positive_rows and other_rows else None
    )
    matched = _length_matched_pairs(positive_rows, other_rows, key="prompt_tokens", caliper=caliper)
    deltas = [float(pos["activation"]) - float(neg["activation"]) for pos, neg in matched]

    if token_gap is None:
        caveat = (
            f"NOT TESTED: one of {positive_label!r} / {other_label!r} has no items in this run, "
            "so the contrast does not exist and every number here is null"
        )
    elif abs(token_gap) <= max_gap_tokens:
        caveat = (
            f"{other_label!r} sits within {max_gap_tokens:.1f} tokens of {positive_label!r} "
            f"({token_gap:+.1f} on average), so the raw separation is not mainly a length gap"
        )
    elif matched:
        caveat = (
            f"NOT length-controlled: {positive_label!r} runs {token_gap:+.1f} tokens against "
            f"{other_label!r}. Read auc_positive_vs_other next to length_matched.auc "
            f"({len(matched)} pairs); the raw number on its own cannot separate the structure "
            "from the length"
        )
    else:
        caveat = (
            f"NOT length-controlled, and NOT correctable in this run: {positive_label!r} runs "
            f"{token_gap:+.1f} tokens against {other_label!r} and no pair of items falls within "
            f"{caliper:.0f} tokens, so the length-matched AUC is NOT TESTED and the raw AUC "
            "below cannot be read as a structure effect"
        )
    # Named `length_caveat`, not `caveat`: the blocks that embed this one carry their own
    # acceptance caveat, and one key cannot honestly hold both.

    return {
        "arms": [positive_label, other_label],
        "auc_positive_vs_other": _auc(positive, other),
        "gap_positive_minus_other": (
            _mean(positive) - _mean(other) if positive and other else None
        ),
        "mean_prompt_tokens_positive": _mean(positive_tokens) if positive_tokens else None,
        "mean_prompt_tokens_other": _mean(other_tokens) if other_tokens else None,
        "token_gap_positive_minus_other": token_gap,
        "length_matched": {
            "caliper_tokens": caliper,
            "n_pairs": len(matched),
            "mean_abs_token_gap": _mean(
                [abs(float(p["prompt_tokens"]) - float(n["prompt_tokens"])) for p, n in matched]
            )
            if matched
            else None,
            "auc": _auc(
                [float(pos["activation"]) for pos, _ in matched],
                [float(neg["activation"]) for _, neg in matched],
            ),
            "paired_delta": paired_summary(deltas) if deltas else None,
        },
        "partial_corr_condition_given_length": (
            _partial_correlation(
                [float(row["activation"]) for row in both],
                [1.0 if index < len(positive_rows) else 0.0 for index in range(len(both))],
                [float(row["prompt_tokens"]) for row in both],
            )
            if positive_rows and other_rows
            else None
        ),
        "length_caveat": caveat,
    }


def _family_round_robin(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Order rows by cycling over ``family``, so a truncated audit is not one family.

    The behavioural readout is capped at ``audit.margin_limit``. case_ids in these
    datasets are family-prefixed (``target_transport_*``), so taking the first N in
    case_id order takes whole families: on v1 the first 20 hostile items are 10
    ``agent_capability`` and 10 ``means_end_conflict`` and nothing else. Round-robin
    keeps the cap spread across the family space, and stays deterministic because each
    family's bucket is sorted by case_id first.
    """

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: str(item.get("case_id", ""))):
        buckets[str(row.get("family") or "")].append(dict(row))
    ordered: list[dict[str, Any]] = []
    for index in range(max((len(bucket) for bucket in buckets.values()), default=0)):
        for family in sorted(buckets):
            if index < len(buckets[family]):
                ordered.append(buckets[family][index])
    return ordered


def _paraphrase_groups(
    records: Sequence[Mapping[str, Any]], *, condition: str
) -> dict[str, list[Mapping[str, Any]]]:
    """Same scenario, different wording -- the only honest basis for an F3 test.

    ``docs/datasets.md`` defines ``pair_id`` as the logical original that the
    hostile / control / paraphrase variants of one scenario share, and
    ``docs/study_design.md`` reads paraphrase as a ``template_id`` crossing inside that
    pair. So a paraphrase group is: one pair_id, one condition, two or more items whose
    template_id AND prompt text both differ.

    Grouping by template_id alone -- what this job used to do -- groups by FAMILY in
    every dataset in this repo (v1: 6 template_ids for 6 families; v2: 5 for 5), so its
    "spread" was between-family variance wearing a paraphrase label.
    """

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if str(record.get("condition", "")) != condition:
            continue
        pair_id = str(record.get("pair_id") or "")
        if pair_id:
            grouped[pair_id].append(record)
    return {
        pair_id: items
        for pair_id, items in sorted(grouped.items())
        if len({str(item.get("template_id") or "") for item in items}) > 1
        and len({str(item.get("prompt") or "") for item in items}) > 1
    }


def _template_family_confound(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Does template_id carry any wording information beyond the family label?

    Answered from the data rather than asserted, so the artifact stays right if a future
    dataset really does vary wording within a family.
    """

    templates = {str(record.get("template_id") or "") for record in records if record.get("family")}
    families = {str(record.get("family") or "") for record in records}
    per_template_families = defaultdict(set)
    for record in records:
        per_template_families[str(record.get("template_id") or "")].add(
            str(record.get("family") or "")
        )
    nested = all(len(values) == 1 for values in per_template_families.values())
    return {
        "n_template_ids": len(templates),
        "n_families": len(families),
        "template_id_is_a_family_label": bool(nested and len(templates) <= len(families)),
    }


# ------------------------------------------------------------- acceptance plan

TESTED = "tested"
PROXY = "proxy"
NOT_TESTED = "not_tested"


def _error_audit_acceptance(
    *, n_discovery_positive: int, threshold: float | None
) -> dict[str, Any]:
    """F5's status, DERIVED from the threshold the audit actually used.

    This criterion used to be hard-coded to ``tested``. It was the one status in the plan
    that no measurement could move, so ``strict_arms = ["F5_error_audit"]`` could never
    abort and the artifact asserted a calibrated threshold audit even when the threshold
    was 0.0.

    Called twice: once before the model loads, where only the discovery count is known,
    and again once the threshold has been measured. The second call is the one that can
    see the degenerate case. A discovery quantile of 0.0 means at least a quarter of the
    positive items never fire, so the split contributed nothing to the cutoff and what
    ran is a fire / no-fire count -- a real measurement, but not the one this criterion
    names, hence PROXY rather than TESTED.
    """

    if not n_discovery_positive:
        status = NOT_TESTED
        reason = (
            "the positive arm has no items on the discovery split, so no threshold can be "
            "estimated and the false-positive / false-negative audit is NOT tested by this run"
        )
    elif threshold is None:
        status = TESTED
        reason = (
            f"planned: the threshold will come from the {n_discovery_positive} positive discovery "
            "items only. This status is re-derived from the measured threshold after the run"
        )
    elif threshold > 0.0:
        status = TESTED
        reason = (
            f"threshold {threshold:.4f}, the 25th percentile of the {n_discovery_positive} "
            "positive discovery activations, applied unchanged to the held-out and control rows. "
            "The *_all_splits totals also cover the discovery rows the threshold was fitted on, "
            "where about the lower quartile are false negatives BY CONSTRUCTION -- the held-out "
            "counts are the finding"
        )
    else:
        status = PROXY
        reason = (
            f"DEGENERATE threshold: the 25th percentile of the {n_discovery_positive} positive "
            "discovery activations is 0.0, so at least a quarter of the positive items do not "
            "fire at all. The discovery split contributed nothing to the cutoff and the audit "
            "degrades to a fire / no-fire count (an item counts as detected only when the "
            "feature actually fires, which is what stops the two error counts from inverting). "
            "The counts are real, but this is NOT the calibrated threshold audit F5 asks for"
        )
    return {
        "criterion": "F5 false positive / false negative audit",
        "status": status,
        "n_discovery_positive": n_discovery_positive,
        "threshold": threshold,
        "detection_rule": "detected := activation > 0 and activation >= threshold",
        "reason": reason,
    }


def _acceptance(
    *,
    positive_condition: str,
    negative_condition: str,
    lexical_condition: str,
    lexical_overlap: float | None,
    min_lexical_overlap: float,
    n_paraphrase_groups: int,
    n_reference_control: int,
    n_reference_lure: int,
    reference_control_overlap: float | None,
    reference_control_length_gap_words: float | None,
    length_gap_words: float | None,
    max_length_gap_words: float,
    n_discovery_positive: int,
    error_threshold: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-criterion status for the work order's F1..F5, decided by measurement.

    Kept pure and separate from the run so it can be evaluated BEFORE the model is
    loaded: a required arm that does not exist should abort in seconds, not after an
    hour of 27B forwards, and the markers must survive even if the run dies later.

    F5 is the one criterion whose final status is not knowable pre-flight, because it
    depends on a threshold the model has to produce. It is filled in here from the
    discovery count alone (so ``strict_arms`` can still abort on an empty split) and
    REPLACED in ``run`` once the threshold exists. Nothing here is ever hard-coded to a
    status: that is what let F5 report `tested` on an audit whose error counts had
    inverted.
    """

    # Never "tested": every arm available here is another condition of the same scenario,
    # so it changes the goal as well as removing the conflict. A true lexical-only item
    # would inject the cue words into a scenario that poses no goal at all.
    lexical_status = (
        PROXY
        if lexical_overlap is not None and lexical_overlap >= min_lexical_overlap
        else NOT_TESTED
    )
    overlap_text = "n/a" if lexical_overlap is None else f"{lexical_overlap:.2f}"
    lexical: dict[str, Any] = {
        "criterion": "F1 lexical injection control",
        "arm": lexical_condition,
        "lexical_overlap_with_hostile": lexical_overlap,
        "min_lexical_overlap": min_lexical_overlap,
        "status": lexical_status,
        "reason": (
            f"no purpose-built lexical-only arm exists. {lexical_condition!r} drops the lure "
            f"relation and keeps {overlap_text} of the hostile vocabulary (Jaccard, instruction "
            "header excluded), "
            + (
                "so it stands in as a PROXY only -- it changes the goal as well as the conflict"
                if lexical_status == PROXY
                else f"which is under the {min_lexical_overlap:.2f} this job requires of a "
                "stand-in, so F1 is NOT tested by this run"
            )
        ),
    }

    # Capped at PROXY by construction, for the same reason as F1: the only trap-free
    # template arm available is a different TASK, so "the feature is quiet there" is not
    # the same claim as "the feature does not read this template". The status used to be
    # TESTED-if-any-control-row -- a row count, so any reference set shipping any control
    # prompt earned the strongest label this job has, whatever its length or vocabulary.
    reference_gap_text = (
        "n/a"
        if reference_control_length_gap_words is None
        else f"{reference_control_length_gap_words:+.1f}"
    )
    reference_overlap_text = (
        "n/a" if reference_control_overlap is None else f"{reference_control_overlap:.2f}"
    )
    template: dict[str, Any] = {
        "criterion": "F2 template control",
        "arms": [positive_condition, REFERENCE_CONTROL],
        "status": PROXY if n_reference_control else NOT_TESTED,
        "n_reference_control": n_reference_control,
        "n_reference_lure": n_reference_lure,
        "lexical_overlap_with_positive": reference_control_overlap,
        "length_gap_words": reference_control_length_gap_words,
        "max_length_gap_words": max_length_gap_words,
        "reason": (
            (
                f"{n_reference_control} matched-control prompts share the answer-delimiter "
                "template with no trap, but they come from a DIFFERENT TASK, so this is a "
                "cross-task template PROXY and is capped there: it can never read tested, "
                "however many control rows the reference set ships. Measured against "
                f"{positive_condition!r}: {reference_overlap_text} word-Jaccard overlap, and "
                f"{positive_condition!r} runs {reference_gap_text} words longer -- a LARGER "
                f"uncontrolled length gap than the {max_length_gap_words:.1f}-word bar that "
                "downgrades F4. Read summary.template_control through its length_matched block, "
                "never the raw AUC alone"
            )
            if n_reference_control
            else "the reference dataset ships no matched controls, so only its lure prompts are "
            "available and those cannot separate template-tracking from transfer: F2 is NOT "
            "tested by this run"
        ),
    }

    paraphrase: dict[str, Any] = {
        "criterion": "F3 semantic paraphrase invariance",
        "status": TESTED if n_paraphrase_groups else NOT_TESTED,
        "n_paraphrase_groups": n_paraphrase_groups,
        "reason": (
            "same pair_id, differing template_id and wording"
            if n_paraphrase_groups
            else "the dataset has no second wording of any scenario (no pair_id spans more "
            "than one template_id), so paraphrase invariance is NOT tested by this run"
        ),
    }

    if length_gap_words is None:
        matched_status, matched_reason = (
            NOT_TESTED,
            f"one of the arms {positive_condition!r} / {negative_condition!r} is empty, so the "
            "matched-control contrast does not exist in this run",
        )
    elif abs(length_gap_words) <= max_length_gap_words:
        matched_status, matched_reason = (
            TESTED,
            f"{negative_condition!r} matches {positive_condition!r} on numbers, entities, answer "
            f"strings, template and length ({length_gap_words:+.1f} words on average)",
        )
    else:
        # The confound F4 exists to eliminate, present in the data and now in the artifact.
        matched_status, matched_reason = (
            PROXY,
            f"{negative_condition!r} is built by deleting the cue sentence, so it is matched on "
            "numbers, entities, answer strings and template but NOT on sentence count: "
            f"{positive_condition!r} runs {length_gap_words:+.1f} words longer on average. Every "
            "raw gap must be read next to summary.length_confound (length-matched subset and "
            "partial correlation), never alone",
        )
    matched: dict[str, Any] = {
        "criterion": "F4 lure-free vocabulary-matched control",
        "arms": [positive_condition, negative_condition],
        "length_gap_words": length_gap_words,
        "max_length_gap_words": max_length_gap_words,
        "status": matched_status,
        "reason": matched_reason,
    }

    audit = _error_audit_acceptance(
        n_discovery_positive=n_discovery_positive, threshold=error_threshold
    )

    return {
        "F1_lexical_injection": lexical,
        "F2_template_control": template,
        "F3_paraphrase_invariance": paraphrase,
        "F4_matched_control": matched,
        "F5_error_audit": audit,
    }


def _strict_failures(
    acceptance: Mapping[str, Mapping[str, Any]], required: Sequence[str]
) -> list[str]:
    """Required criteria whose data does not exist, as human-readable lines."""

    failures: list[str] = []
    for key in required:
        entry = acceptance.get(str(key))
        if entry is None:
            failures.append(f"{key}: unknown criterion (expected one of {sorted(acceptance)})")
        elif entry.get("status") != TESTED:
            failures.append(f"{key}: {entry.get('status')} -- {entry.get('reason')}")
    return failures


# ------------------------------------------------------------------- setup


def _resolve_env(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = table(config, "model")
    profile = get_qwen35_analysis_profile(
        str(model_cfg.get("profile", DEFAULT_ANALYSIS_PROFILE_KEY))
    )
    dtype = model_cfg.get("dtype", "auto")
    dtype = recommended_dtype_name() if dtype == "auto" else str(dtype)
    sae_device = model_cfg.get("sae_device", "auto")
    sae_device = default_sae_device() if sae_device == "auto" else str(sae_device)
    sae_dtype = model_cfg.get("sae_dtype", "auto")
    sae_dtype = dtype if sae_dtype == "auto" else str(sae_dtype)
    return {
        "profile": profile,
        "model_id": profile.analysis_model_id,
        "repo_id": profile.sae_repo_id,
        "dtype": dtype,
        "device_map": model_cfg.get("device_map", "auto"),
        "sae_device": sae_device,
        "sae_dtype": sae_dtype,
    }


def _condition_of(case: LureCase, conditions: Sequence[str]) -> str:
    """The dataset's own condition label, with the case_id suffix as a fallback."""

    if case.condition in conditions:
        return case.condition
    for name in conditions:
        if case.case_id.endswith(f"_{name}"):
            return name
    return "unknown"


def _split_map(
    cases: Sequence[LureCase], *, unit: str, train_frac: float, seed: int
) -> dict[str, str]:
    """``case_id -> "discovery" | "held_out"``, split at scenario level by default.

    ``docs/datasets.md`` requires the condition variants of one scenario to stay on the
    same side of the split. Hashing case_id puts ``X_hostile`` in discovery while its own
    ``X_neutral`` lands in held-out, which leaks the scenario into the very audit the
    split exists to keep clean, so the hash is taken over pair_id instead whenever the
    dataset provides one. The threshold still comes from the positive condition's
    discovery items only -- that is what keeps the FP/FN audit non-circular.
    """

    if unit not in {"pair_id", "case_id"}:
        raise ValueError("split_unit must be 'pair_id' or 'case_id'")
    use_pairs = unit == "pair_id" and all(case.pair_id for case in cases)
    keys = {case.case_id: (case.pair_id if use_pairs else case.case_id) for case in cases}

    representatives: list[LureCase] = []
    seen: set[str] = set()
    for case in cases:
        key = keys[case.case_id]
        if key not in seen:
            seen.add(key)
            representatives.append(replace(case, case_id=key))
    train, _ = split_lure_cases(representatives, train_frac=train_frac, seed=seed)
    discovery = {case.case_id for case in train}
    return {
        case_id: ("discovery" if key in discovery else "held_out") for case_id, key in keys.items()
    }


def run(config_path: Path, output_root: Path) -> Path:
    config = load_toml(config_path)
    name = run_name(config)
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config.toml")

    feature_cfg = table(config, "feature")
    feature_id = int(feature_cfg["feature_id"])
    layer = int(feature_cfg["layer"])

    data_cfg = table(config, "data")
    dataset = str(data_cfg.get("dataset", "goal_affordance_traps_v1"))
    conditions = [
        str(c)
        for c in (
            data_cfg.get("conditions") or ["hostile", "explicit", "neutral", "counterfactual"]
        )
    ]
    positive_condition = str(data_cfg.get("positive_condition", "hostile"))
    negative_condition = str(data_cfg.get("negative_condition", "neutral"))
    lexical_condition = str(data_cfg.get("lexical_condition", "counterfactual"))
    instruction = bool(data_cfg.get("instruction", True))
    split_unit = str(data_cfg.get("split_unit", "pair_id"))
    train_frac = float(data_cfg.get("train_frac", 0.6))
    split_seed = int(data_cfg.get("split_seed", 0))

    ref_cfg = table(config, "reference")
    ref_dataset = str(ref_cfg.get("dataset", "hagendorff_crt"))
    ref_limit = int(ref_cfg.get("limit_per_family", 10))
    ref_use_control = bool(ref_cfg.get("use_matched_control", True))

    audit_cfg = table(config, "audit")
    strict_arms = [str(arm) for arm in (audit_cfg.get("strict_arms") or [])]
    min_lexical_overlap = float(audit_cfg.get("min_lexical_overlap", 0.80))
    max_length_gap_words = float(audit_cfg.get("max_length_gap_words", 2.0))
    length_caliper = float(audit_cfg.get("length_caliper_tokens", 4.0))
    record_rank = bool(audit_cfg.get("record_topk_rank", True))
    margin_for_errors = bool(audit_cfg.get("margin_for_errors", True))
    margin_for_all = bool(audit_cfg.get("margin_for_all_cases", False))
    margin_limit = int(audit_cfg.get("margin_limit", 80))

    env = _resolve_env(config)
    cases = [
        case for case in lure_dataset_cases(dataset) if _condition_of(case, conditions) != "unknown"
    ]
    if not cases:
        raise ValueError(f"{dataset!r} has no cases in conditions {conditions}")
    reference = lure_dataset_cases(ref_dataset, limit_per_family=ref_limit)
    # The item text without the shared instruction header. Overlap has to be measured on
    # this: the header is identical on every prompt, so scoring the instructed text drags
    # every pair's Jaccard toward 1 and would let a badly matched arm pass as an F1 proxy
    # (on v1 it turns counterfactual's real 0.56 into 0.76).
    item_text = {case.case_id: case.prompt for case in cases}
    reference_control_text = [
        case.control_prompt for case in reference if ref_use_control and case.control_prompt
    ]
    if instruction:
        cases = instruct_lure_cases(cases)
        reference = instruct_lure_cases(reference)

    effective_split_unit = (
        "pair_id" if split_unit == "pair_id" and all(case.pair_id for case in cases) else "case_id"
    )
    split_of = _split_map(cases, unit=split_unit, train_frac=train_frac, seed=split_seed)

    # ---- what the data can and cannot answer, decided before anything is loaded ----
    hostile_text_by_pair = {
        case.pair_id: item_text[case.case_id]
        for case in cases
        if case.pair_id and _condition_of(case, conditions) == positive_condition
    }
    records: list[dict[str, Any]] = []
    for case in cases:
        condition = _condition_of(case, conditions)
        twin = hostile_text_by_pair.get(case.pair_id, "")
        records.append(
            {
                "case_id": case.case_id,
                "pair_id": case.pair_id,
                "template_id": case.template_id,
                "family": case.family,
                "condition": condition,
                "prompt": case.prompt,
                "prompt_words": len(_word_tokens(case.prompt)),
                "prompt_chars": len(case.prompt),
                # 1.0 for the hostile item against itself; the number is only read for
                # the other arms, where it says how lexically matched the control is.
                "lexical_overlap_hostile": (
                    _jaccard(item_text[case.case_id], twin) if twin else None
                ),
            }
        )
    by_condition_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_condition_records[str(record["condition"])].append(record)

    paraphrase_groups = _paraphrase_groups(records, condition=positive_condition)
    template_confound = _template_family_confound(records)
    lexical_overlaps = [
        float(record["lexical_overlap_hostile"])
        for record in by_condition_records.get(lexical_condition, [])
        if record["lexical_overlap_hostile"] is not None
    ]
    lexical_overlap = _mean(lexical_overlaps) if lexical_overlaps else None
    positive_words = [
        float(r["prompt_words"]) for r in by_condition_records.get(positive_condition, [])
    ]
    negative_words = [
        float(r["prompt_words"]) for r in by_condition_records.get(negative_condition, [])
    ]
    length_gap_words = (
        _mean(positive_words) - _mean(negative_words) if positive_words and negative_words else None
    )
    n_reference_control = len(reference_control_text)

    # F2 is graded on the same kind of measurement as F1 and F4 instead of on a row
    # count. There is no pairing between a goal-affordance scenario and a CRT control, so
    # the overlap is the mean over every (control, positive) pair, and both sides are
    # measured on the un-instructed item text -- the shared header would drag every
    # Jaccard toward 1 exactly as it does for F1.
    positive_item_text = [
        item_text[str(record["case_id"])]
        for record in by_condition_records.get(positive_condition, [])
    ]
    reference_control_overlap = (
        _mean(
            [
                _jaccard(control, positive_text)
                for control in reference_control_text
                for positive_text in positive_item_text
            ]
        )
        if reference_control_text and positive_item_text
        else None
    )
    reference_control_words = [float(len(_word_tokens(body))) for body in reference_control_text]
    positive_item_words = [float(len(_word_tokens(body))) for body in positive_item_text]
    reference_control_length_gap_words = (
        _mean(positive_item_words) - _mean(reference_control_words)
        if positive_item_words and reference_control_words
        else None
    )
    n_discovery_positive_planned = sum(
        1
        for record in records
        if record["condition"] == positive_condition
        and split_of.get(str(record["case_id"])) == "discovery"
    )

    acceptance = _acceptance(
        positive_condition=positive_condition,
        negative_condition=negative_condition,
        lexical_condition=lexical_condition,
        lexical_overlap=lexical_overlap,
        min_lexical_overlap=min_lexical_overlap,
        n_paraphrase_groups=len(paraphrase_groups),
        n_reference_control=n_reference_control,
        n_reference_lure=len(reference),
        reference_control_overlap=reference_control_overlap,
        reference_control_length_gap_words=reference_control_length_gap_words,
        length_gap_words=length_gap_words,
        max_length_gap_words=max_length_gap_words,
        n_discovery_positive=n_discovery_positive_planned,
    )
    for key, entry in acceptance.items():
        if entry["status"] != TESTED:
            _log(f"{entry['status'].upper()}: {key} -- {entry['reason']}")
    failures = _strict_failures(acceptance, strict_arms)
    if failures:
        # Abort before the 27B is touched: the arm is missing from the dataset and no
        # amount of GPU time will produce it.
        raise RuntimeError(
            "audit.strict_arms requires arms this dataset does not have:\n  "
            + "\n  ".join(failures)
        )

    manifest: dict[str, Any] = {
        "run_name": name,
        "job": "feature_falsification",
        "started_at": _timestamp(),
        "feature_id": feature_id,
        "layer": layer,
        "profile": env["profile"].key,
        "model_id": env["model_id"],
        "sae_repo_id": env["repo_id"],
        "dataset": dataset,
        "conditions": conditions,
        "n_cases": len(cases),
        "reference_dataset": ref_dataset,
        "n_reference": len(reference),
        "n_reference_control": n_reference_control,
        "split_unit": effective_split_unit,
        "acceptance": acceptance,
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    _write_json(run_dir / "manifest.json", manifest)

    _log(f"loading {env['model_id']} (feature {feature_id} @ layer {layer})")
    lm = load_qwen_language_model(
        env["model_id"], device_map=env["device_map"], dtype=env["dtype"], dispatch=True
    )
    sae = load_qwen_scope_sae(
        env["repo_id"], layer, device=env["sae_device"], dtype=dtype_from_name(env["sae_dtype"])
    )
    tokenizer = getattr(lm.tokenizer, "tokenizer", lm.tokenizer)

    def _probe(prompt: str) -> dict[str, Any]:
        residual = capture_layer_residuals(lm, [prompt], layer, token_position="last")
        sparse = float(
            qwen_scope_sparse_feature_values(residual, sae, [feature_id])
            .detach()
            .to(torch.float32)
            .reshape(-1)[0]
        )
        preact = float(
            qwen_scope_feature_preactivations(residual, sae, [feature_id])
            .detach()
            .to(torch.float32)
            .reshape(-1)[0]
        )
        rank: int | None = None
        if record_rank:
            # An activation of 3.1 means one thing at rank 2 of the kept TopK and another
            # at rank 49, and the rank cannot be recovered from the artifact later without
            # re-running the model -- so it is taken here, at the cost of one extra
            # encoder matmul next to a full forward pass.
            topk = active_prompt_features(residual, sae, top_n=sae.top_k)
            rank = next(
                (position for position, (fid, _) in enumerate(topk, start=1) if fid == feature_id),
                None,
            )
        return {
            "activation": sparse,
            "preactivation": preact,
            "topk_rank": rank,
            # None, never False, when the rank was not taken. `rank is None` used to fold
            # "measured and absent" together with "not measured", so switching the probe
            # off published in_topk = False on every row and condition_in_topk_rate = 0.0
            # on every arm -- the strongest falsification claim this job can make, emitted
            # by a config flag instead of by data.
            "in_topk": (rank is not None) if record_rank else None,
            "prompt_tokens": len(tokenizer.encode(prompt)),
        }

    rows: list[dict[str, Any]] = []
    case_by_row_id: dict[str, LureCase] = {}
    for index, (case, record) in enumerate(zip(cases, records, strict=True), start=1):
        row = {
            "group": "trap",
            "case_id": case.case_id,
            "source_case_id": case.case_id,
            "pair_id": case.pair_id,
            "template_id": case.template_id,
            "family": case.family,
            "condition": record["condition"],
            "split": split_of.get(case.case_id, ""),
            "prompt": case.prompt,
            "prompt_words": record["prompt_words"],
            "prompt_chars": record["prompt_chars"],
            "lexical_overlap_hostile": record["lexical_overlap_hostile"],
            "answer_pair": f"{case.correct_answer.strip()}|{case.lure_answer.strip()}",
            "answer_len_delta": (len(case.lure_answer.split()) - len(case.correct_answer.split())),
            **_probe(case.prompt),
        }
        row["fires"] = row["activation"] > 0
        rows.append(row)
        case_by_row_id[case.case_id] = case
        if index % 40 == 0:
            _log(f"trap {index}/{len(cases)}")

    reference_jobs: list[tuple[LureCase, str, str, str]] = []
    for case in reference:
        reference_jobs.append((case, REFERENCE_LURE, case.case_id, case.prompt))
        if ref_use_control and case.control_prompt:
            reference_jobs.append(
                (case, REFERENCE_CONTROL, f"{case.case_id}__control", case.control_prompt)
            )
    for index, (case, condition, row_id, prompt) in enumerate(reference_jobs, start=1):
        row = {
            "group": "reference",
            "case_id": row_id,
            "source_case_id": case.case_id,
            "pair_id": case.pair_id,
            "template_id": case.template_id,
            "family": case.family,
            "condition": condition,
            "split": "reference",
            "prompt": prompt,
            "prompt_words": len(_word_tokens(prompt)),
            "prompt_chars": len(prompt),
            "lexical_overlap_hostile": None,
            "answer_pair": f"{case.correct_answer.strip()}|{case.lure_answer.strip()}",
            "answer_len_delta": (len(case.lure_answer.split()) - len(case.correct_answer.split())),
            **_probe(prompt),
        }
        row["fires"] = row["activation"] > 0
        rows.append(row)
        if index % 20 == 0:
            _log(f"reference {index}/{len(reference_jobs)}")

    _write_csv(
        run_dir / "falsification_activations.csv",
        rows,
        [
            "group",
            "case_id",
            "source_case_id",
            "pair_id",
            "template_id",
            "family",
            "condition",
            "split",
            "activation",
            "preactivation",
            "topk_rank",
            "in_topk",
            "fires",
            "prompt_tokens",
            "prompt_words",
            "prompt_chars",
            "lexical_overlap_hostile",
            "answer_pair",
            "answer_len_delta",
        ],
    )

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)

    def _values(condition: str) -> list[float]:
        return [float(row["activation"]) for row in by_condition.get(condition, [])]

    positive = _values(positive_condition)
    negative = _values(negative_condition)
    reference_control_values = _values(REFERENCE_CONTROL)
    reference_lure_values = _values(REFERENCE_LURE)

    # ---- F4: the length confound, measured instead of assumed -------------------
    length_profile = {
        condition: {
            "tokens": _spread([float(row["prompt_tokens"]) for row in items]),
            "words": _spread([float(row["prompt_words"]) for row in items]),
        }
        for condition, items in sorted(by_condition.items())
    }
    positive_rows = by_condition.get(positive_condition, [])
    negative_rows = by_condition.get(negative_condition, [])
    contrast_rows = positive_rows + negative_rows
    matched = _length_matched_pairs(
        positive_rows, negative_rows, key="prompt_tokens", caliper=length_caliper
    )
    matched_deltas = [float(pos["activation"]) - float(neg["activation"]) for pos, neg in matched]
    length_confound = {
        "note": (
            f"{negative_condition!r} deletes the cue sentence rather than rewriting it, so "
            f"{positive_condition!r} prompts are systematically longer. Any raw gap below is "
            "partly a length gap; these are the numbers that say how much."
        ),
        "profile": length_profile,
        "token_gap_positive_minus_negative": (
            _mean([float(row["prompt_tokens"]) for row in positive_rows])
            - _mean([float(row["prompt_tokens"]) for row in negative_rows])
            if positive_rows and negative_rows
            else None
        ),
        "corr_activation_vs_prompt_tokens_within_positive": _corr(
            [float(row["activation"]) for row in positive_rows],
            [float(row["prompt_tokens"]) for row in positive_rows],
        ),
        # Only "across arms" when both arms exist. With one arm this is the within-arm
        # correlation under a name that claims a contrast.
        "corr_activation_vs_prompt_tokens_across_arms": _corr(
            [float(row["activation"]) for row in contrast_rows],
            [float(row["prompt_tokens"]) for row in contrast_rows],
        )
        if positive_rows and negative_rows
        else None,
        # Does the condition still explain activation once length is taken out?
        "partial_corr_condition_given_length": _partial_correlation(
            [float(row["activation"]) for row in contrast_rows],
            [1.0 if row["condition"] == positive_condition else 0.0 for row in contrast_rows],
            [float(row["prompt_tokens"]) for row in contrast_rows],
        ),
        # `_corr`, not `pearson`: with one arm missing the indicator is constant and
        # `pearson` would return a confident 0.0 for an undefined correlation.
        "corr_condition_vs_length": _corr(
            [1.0 if row["condition"] == positive_condition else 0.0 for row in contrast_rows],
            [float(row["prompt_tokens"]) for row in contrast_rows],
        ),
        "length_matched": {
            "caliper_tokens": length_caliper,
            "n_pairs": len(matched),
            "n_positive_unmatched": len(positive_rows) - len(matched),
            "mean_abs_token_gap": _mean(
                [abs(float(p["prompt_tokens"]) - float(n["prompt_tokens"])) for p, n in matched]
            )
            if matched
            else None,
            "auc": _auc(
                [float(pos["activation"]) for pos, _ in matched],
                [float(neg["activation"]) for _, neg in matched],
            ),
            "paired_delta": paired_summary(matched_deltas) if matched_deltas else None,
        },
    }

    # Every arm gets the same block, because every arm has a length gap and they do not
    # all run the same way. The length machinery used to be wired to hostile-vs-neutral
    # alone, so the three contrasts below published raw AUCs with nothing attached while
    # the headline carried a length-matched twin -- which reads as though all four had
    # been adjusted.
    def _against_positive(other_rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
        return _contrast(
            positive_rows,
            other_rows,
            positive_label=positive_condition,
            other_label=label,
            caliper=length_caliper,
            max_gap_tokens=max_length_gap_words,
        )

    # ---- F1 proxy ----------------------------------------------------------------
    lexical_rows = by_condition.get(lexical_condition, [])
    per_condition_overlap = {
        condition: _mean(
            [
                float(row["lexical_overlap_hostile"])
                for row in items
                if row.get("lexical_overlap_hostile") is not None
            ]
        )
        for condition, items in sorted(by_condition_records.items())
        if any(row.get("lexical_overlap_hostile") is not None for row in items)
    }
    rival_overlap = {
        condition: value
        for condition, value in per_condition_overlap.items()
        if condition != positive_condition
    }
    lexical_proxy = {
        "status": acceptance["F1_lexical_injection"]["status"],
        "arm": lexical_condition,
        # The config used to call this arm the "closest thing the dataset has to a
        # lexical-injection control". On v1 it is the least matched arm of the three
        # (counterfactual 0.57 against explicit 0.79 and neutral 0.76) -- the choice is a
        # judgement about structure, and printing it as though it were a measurement is
        # exactly the kind of claim this job exists to catch.
        "arm_chosen_by": (
            "STRUCTURE, not lexical overlap: this is the arm that keeps the cue sentence "
            "while dropping the lure relation. It is not necessarily the most lexically "
            "matched arm available -- read per_condition_lexical_overlap and "
            "highest_overlap_arm before treating anything below as a lexical control."
        ),
        "mean_lexical_overlap_with_hostile": lexical_overlap,
        "per_condition_lexical_overlap": per_condition_overlap,
        "highest_overlap_arm": (
            max(rival_overlap, key=lambda condition: rival_overlap[condition])
            if rival_overlap
            else None
        ),
        **_against_positive(lexical_rows, lexical_condition),
        "caveat": acceptance["F1_lexical_injection"]["reason"],
    }

    # ---- F2 template control vs transfer ----------------------------------------
    template_control = {
        "status": acceptance["F2_template_control"]["status"],
        **_against_positive(by_condition.get(REFERENCE_CONTROL, []), REFERENCE_CONTROL),
        # Both arms must be non-empty. Guarding only the denominator let an EMPTY
        # reference arm publish 0.0, which reads as "measured, and the feature does
        # not fire on the template control" -- the strongest possible F2 pass, from
        # no data at all.
        "ratio_reference_control_over_positive": (
            (mean_or_none(reference_control_values) / mean_or_none(positive))
            if mean_or_none(reference_control_values) is not None
            and mean_or_none(positive) not in (None, 0.0)
            else None
        ),
        "note": (
            "reference_control prompts are the reference dataset's trap-free rewrites: same "
            "answer template, same numbers and entities, no conflict -- but a different TASK, "
            "and on v1 the largest length gap in this job. A ratio near 1 means the feature "
            "reads the template rather than the trap ONLY if the length gap is not doing the "
            "work; length_matched.auc is the version that controls for it, and it is null when "
            "the two arms do not overlap in length at all."
        ),
        "caveat": acceptance["F2_template_control"]["reason"],
    }
    crt_lure_transfer = {
        **_against_positive(by_condition.get(REFERENCE_LURE, []), REFERENCE_LURE),
        "ratio_reference_lure_over_positive": (
            _mean(reference_lure_values) / _mean(positive) if _mean(positive) else None
        ),
        "note": (
            "NOT a template control: the reference hostile prompts are themselves lure items, "
            "so activation there is evidence for generalisation across lure families. Reading "
            "it as template-tracking (as this job used to) inverts the conclusion."
        ),
    }

    # ---- F3 paraphrase: absent, and the block that used to stand in for it -------
    paraphrase = {
        "status": acceptance["F3_paraphrase_invariance"]["status"],
        "n_groups": len(paraphrase_groups),
        "reason": acceptance["F3_paraphrase_invariance"]["reason"],
    }
    if paraphrase_groups:
        activation_by_case = {str(row["case_id"]): float(row["activation"]) for row in rows}
        per_group = {
            pair_id: [activation_by_case[str(item["case_id"])] for item in items]
            for pair_id, items in paraphrase_groups.items()
        }
        paraphrase["per_group_spread"] = {
            pair_id: max(values) - min(values) for pair_id, values in per_group.items()
        }
        paraphrase["mean_within_group_spread"] = _mean(
            [max(values) - min(values) for values in per_group.values()]
        )
    template_variation = {
        "note": (
            "Variation of the positive arm ACROSS template_id. This is NOT paraphrase "
            "invariance: template_id is a family label in this dataset, so the spread below is "
            "between-family variance."
        ),
        **template_confound,
        "per_template_mean": {},
        "spread": None,
    }
    per_template: dict[str, list[float]] = defaultdict(list)
    for row in by_condition.get(positive_condition, []):
        if row["template_id"]:
            per_template[str(row["template_id"])].append(float(row["activation"]))
    if per_template:
        template_variation["per_template_mean"] = {
            key: _mean(values) for key, values in sorted(per_template.items())
        }
        template_variation["spread"] = (
            max(_mean(v) for v in per_template.values())
            - min(_mean(v) for v in per_template.values())
            if len(per_template) > 1
            else None
        )

    # ---- answer-string confounds -------------------------------------------------
    trap_rows = [row for row in rows if row["group"] == "trap"]
    confound = {
        "corr_activation_vs_answer_len_delta": _corr(
            [float(r["activation"]) for r in trap_rows],
            [float(r["answer_len_delta"]) for r in trap_rows],
        ),
        "n_distinct_answer_pairs": len({r["answer_pair"] for r in trap_rows}),
    }

    # ---- F5 error audit ----------------------------------------------------------
    discovery_positive = sorted(
        float(row["activation"])
        for row in rows
        if row["condition"] == positive_condition and row["split"] == "discovery"
    )
    threshold = discovery_positive[len(discovery_positive) // 4] if discovery_positive else 0.0
    # The status is DERIVED here, from the threshold that was actually measured, and
    # replaces the pre-flight entry. It used to be hard-coded to `tested`.
    acceptance["F5_error_audit"] = _error_audit_acceptance(
        n_discovery_positive=len(discovery_positive), threshold=threshold
    )

    def _detected(row: Mapping[str, Any]) -> bool:
        """Over the discovery cutoff AND actually firing.

        A TopK SAE writes exactly 0.0 wherever the feature is off its support, so the
        25th percentile of the discovery positives IS 0.0 as soon as a quarter of them
        are silent. At that threshold a bare ``>= threshold`` calls every silent control
        a false positive and ``< threshold`` calls no silent positive a false negative --
        both counts inverted, and the margin pass then spends two 27B forwards on each of
        the items that are not errors. Requiring the feature to fire is a no-op whenever
        the threshold is positive, so the property worth keeping (cutoff estimated on
        discovery, applied unchanged to held-out) is untouched.
        """

        activation = float(row["activation"])
        return activation > 0.0 and activation >= threshold

    false_negative_rows = [
        row for row in rows if row["condition"] == positive_condition and not _detected(row)
    ]
    false_positive_rows = [
        row
        for row in rows
        if row["condition"] in {negative_condition, REFERENCE_CONTROL} and _detected(row)
    ]

    # Margins default to the error cases only. That is where the audit needs them --
    # "the feature was silent AND the model still took the lure" is a different finding
    # from "the feature was silent on an item the model got right" -- and it keeps the
    # cost at two forwards per error instead of two per case.
    if margin_for_all:
        margin_candidates = list(trap_rows)
    elif margin_for_errors:
        margin_candidates = [
            row for row in false_negative_rows + false_positive_rows if row["group"] == "trap"
        ]
    else:
        margin_candidates = []
    # Round-robin over family rather than alphabetical case_id order: case_ids here are
    # family-prefixed, so `sorted(...)[:limit]` takes whole families (on v1 the first 20
    # hostile items are 10 agent_capability and 10 means_end_conflict and nothing else)
    # and the behavioural readout silently covers part of the family space.
    margin_rows = _family_round_robin(
        list({str(row["case_id"]): row for row in margin_candidates}.values())
    )
    measured_margin_rows = margin_rows[:margin_limit]
    n_margin_skipped = max(0, len(margin_rows) - margin_limit)
    margin_families: dict[str, int] = {}
    for row in measured_margin_rows:
        family_name = str(row.get("family") or "")
        margin_families[family_name] = margin_families.get(family_name, 0) + 1
    margins: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(measured_margin_rows, start=1):
        case = case_by_row_id[str(row["case_id"])]
        # Baseline margin only: `lure - correct`, positive means the lure answer wins.
        # No ablation happens here, so no delta and no sign convention to preserve.
        margin = answer_logprob_margin(
            lm,
            case.prompt,
            correct_answer=case.correct_answer,
            lure_answer=case.lure_answer,
        )
        margins[str(row["case_id"])] = margin.as_row()
        if index % 10 == 0:
            _log(f"margin {index}/{len(measured_margin_rows)}")

    def _error_record(row: Mapping[str, Any]) -> dict[str, Any]:
        record = {
            "case_id": row["case_id"],
            "condition": row["condition"],
            "family": row["family"],
            "pair_id": row["pair_id"],
            "split": row["split"],
            # F5.d: an error case has to be readable without re-joining the dataset.
            "prompt": row["prompt"],
            "activation": row["activation"],
            "preactivation": row["preactivation"],
            # F5.g: where the feature sat among the TopK actually kept at this position.
            "topk_rank": row["topk_rank"],
            "topk_size": int(sae.top_k),
            # Same rule as margin_status below: a measurement that did not run must not
            # read as a measured absence.
            "topk_status": (
                "measured"
                if record_rank
                else "NOT MEASURED: audit.record_topk_rank = false, so topk_rank and in_topk "
                "are null. Null here does NOT mean the feature was outside the TopK."
            ),
            "threshold": threshold,
            "activation_minus_threshold": float(row["activation"]) - threshold,
            "prompt_tokens": row["prompt_tokens"],
        }
        # F5.h: the behavioural side of the same item.
        margin_row = margins.get(str(row["case_id"]))
        if margin_row is not None:
            record.update(margin_row)
            record["margin_status"] = "measured"
        elif row["group"] != "trap":
            record["margin_status"] = (
                "skipped: the reference control's correct answer is the item's lure string, "
                "so a margin scored with the item's labels would be sign-flipped"
            )
        else:
            record["margin_status"] = "skipped: disabled or over audit.margin_limit"
        return record

    errors = {
        "threshold": threshold,
        "threshold_source": (
            f"25th percentile of {positive_condition} activation on the discovery split "
            f"(split unit: {effective_split_unit}, train_frac={train_frac}, seed={split_seed})"
        ),
        "detection_rule": (
            "detected := activation > 0 AND activation >= threshold. Firing is required "
            "because a TopK SAE writes exactly 0.0 wherever the feature is off its support: at "
            "a threshold of 0.0 a bare >= would make every silent control a false positive and "
            "no silent positive a false negative."
        ),
        "acceptance_status": acceptance["F5_error_audit"]["status"],
        "acceptance_reason": acceptance["F5_error_audit"]["reason"],
        "n_discovery_positive": len(discovery_positive),
        "counts_note": (
            "The lists below cover ALL splits, including the discovery rows the threshold was "
            "fitted on -- where roughly the lower quartile of the positive arm are false "
            "negatives BY CONSTRUCTION, because the threshold IS that quartile. Every record "
            "carries its own `split`; the held-out counts in the summary are the finding, the "
            "discovery counts are part definition."
        ),
        "false_positive_arms": [negative_condition, REFERENCE_CONTROL],
        "false_positive_note": (
            f"{REFERENCE_LURE} is deliberately NOT audited for false positives: those items are "
            "themselves lure items, so the feature firing on them is transfer, not an error."
        ),
        "margin_convention": "margin_lure_minus_correct > 0 means the lure answer is preferred",
        "false_negatives": [_error_record(row) for row in false_negative_rows],
        "false_positives": [_error_record(row) for row in false_positive_rows],
    }
    _write_json(run_dir / "falsification_errors.json", errors)

    def _in_split(items: Sequence[Mapping[str, Any]], split: str) -> int:
        return sum(1 for row in items if row["split"] == split)

    caveats = [entry["reason"] for entry in acceptance.values() if entry["status"] != TESTED]
    caveats.append(
        "reference_lure activation is transfer evidence, not a template control; only "
        "reference_control answers the template question."
    )
    # The per-contrast length caveats, in the summary rather than only inside their own
    # block, so a reader who only scans `caveats` still sees which AUCs are unadjusted.
    caveats.extend(
        f"{name}: {block['length_caveat']}"
        for name, block in (
            ("lexical_proxy", lexical_proxy),
            ("template_control", template_control),
            ("crt_lure_transfer", crt_lure_transfer),
        )
        if str(block.get("length_caveat", "")).startswith("NOT")
    )
    if not record_rank:
        caveats.append(
            "NOT TESTED: audit.record_topk_rank = false, so no TopK rank was measured. "
            "condition_in_topk_rate is null and the CSV in_topk column is blank -- blank means "
            "not measured, never 'the feature was outside the TopK'."
        )

    summary = {
        "feature_id": feature_id,
        "layer": layer,
        "acceptance": acceptance,
        "caveats": caveats,
        "condition_means": {
            condition: _mean([float(row["activation"]) for row in items])
            for condition, items in sorted(by_condition.items())
        },
        "condition_fire_rate": {
            condition: _mean([1.0 if float(row["activation"]) > 0 else 0.0 for row in items])
            for condition, items in sorted(by_condition.items())
        },
        # Null, not a dict of zeros, when the rank was never taken. `in_topk` is None on
        # every row in that case, and averaging `1.0 if row["in_topk"] else 0.0` over it
        # published "the feature was outside the SAE TopK on 100% of items in every arm"
        # -- the strongest falsification claim this job can make -- from a config flag.
        "condition_in_topk_rate": (
            {
                condition: _mean([1.0 if row["in_topk"] else 0.0 for row in items])
                for condition, items in sorted(by_condition.items())
            }
            if record_rank
            else None
        ),
        "condition_in_topk_status": (
            "measured"
            if record_rank
            else "NOT TESTED: audit.record_topk_rank = false, so no rank was taken"
        ),
        # The headline dissociation: same scenario and answer strings, different
        # structure -- and, unavoidably, a different length. Read it next to
        # length_confound, never alone.
        "structure_auc": _auc(positive, negative),
        # Guarded like structure_auc. `_mean([])` is 0.0, so an empty negative arm used to
        # publish the positive arm's own mean as the gap between two arms.
        "structure_gap": (_mean(positive) - _mean(negative)) if positive and negative else None,
        "length_confound": length_confound,
        "lexical_proxy": lexical_proxy,
        "template_control": template_control,
        "crt_lure_transfer": crt_lure_transfer,
        "paraphrase": paraphrase,
        "template_variation": template_variation,
        "answer_confound": confound,
        # Split out rather than pooled. The old `n_false_negatives` mixed a finding with a
        # definition: the threshold is the 25th percentile of the DISCOVERY positives, so
        # roughly a quarter of those are false negatives no matter how good the feature is.
        "n_false_negatives_held_out": _in_split(false_negative_rows, "held_out"),
        "n_false_negatives_discovery": _in_split(false_negative_rows, "discovery"),
        "n_false_negatives_all_splits": len(false_negative_rows),
        "n_false_positives_held_out": _in_split(false_positive_rows, "held_out"),
        "n_false_positives_discovery": _in_split(false_positive_rows, "discovery"),
        "n_false_positives_reference": _in_split(false_positive_rows, "reference"),
        "n_false_positives_all_splits": len(false_positive_rows),
        "error_count_note": (
            "Quote the held-out counts. The *_all_splits totals include the discovery rows the "
            "threshold was fitted on, where about a quarter of the positive arm are false "
            "negatives by construction -- that share is a definition, not a finding."
        ),
        "n_margins_measured": len(margins),
        "n_margins_skipped_over_limit": n_margin_skipped,
        "margin_families_measured": margin_families,
        "margin_selection": (
            "round-robin over family, case_id order within a family, truncated at "
            f"audit.margin_limit={margin_limit}. Alphabetical case_id order would take whole "
            "families, because case_ids here are family-prefixed."
        ),
    }
    _write_json(run_dir / "falsification_summary.json", summary)
    # acceptance is re-assigned, not just carried: F5's entry was replaced after the
    # threshold was measured, and the manifest must ship the derived status, not the plan.
    manifest.update({"finished_at": _timestamp(), "acceptance": acceptance, "summary": summary})
    _write_json(run_dir / "manifest.json", manifest)
    _log(
        f"structure AUC {summary['structure_auc']} (length-matched "
        f"{length_confound['length_matched']['auc']}, n={len(matched)}) | template-control "
        f"ratio {template_control['ratio_reference_control_over_positive']} (proxy) | held-out "
        f"FN {summary['n_false_negatives_held_out']} FP {summary['n_false_positives_held_out']}"
    )
    for key, entry in acceptance.items():
        if entry["status"] != TESTED:
            _log(f"{entry['status'].upper()}: {key}")
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
