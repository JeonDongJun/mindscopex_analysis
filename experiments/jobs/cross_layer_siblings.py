"""Find a feature's counterpart at other layers, then ablate the pair together.

``multisite_ablation`` transplants one layer's decoder direction to its neighbours.
That answers "does removing this direction at more places do more", but it does
not identify the feature at those layers -- each layer has its own SAE and its own
numbering, so L15 #81663 and L31 #81663 are unrelated by construction.

This job identifies the counterpart first, then intervenes on the identified pair:

    match     for every candidate at the target layer, four independent signals
              measured on the SAME discovery items -- decoder cosine, activation
              correlation, effect correlation (does ablating each move the margin
              the same way per item), and specificity correlation (does ablating
              each move the *cue-attributable* part of the margin the same way,
              hostile minus its matched no-cue twin). Ranked by their geometric
              mean, so a high cosine alone cannot win; cosine alone is exactly the
              failure mode this exists to avoid, the dictionary being overcomplete.
              Specificity is not a restatement of the effect signal: a direction
              that merely damages the model's judgement moves hostile and control
              together, so it can correlate on the raw effect while explaining
              nothing about the trap. Only the paired difference separates the two,
              which is why the work order lists it as a signal of its own.

              The fourth signal is a correlation, so it carries a direction only
              once its LEVELS are checked: ``specificity_level_verdict`` gates a
              candidate out unless its own hostile arm and its own gap are both
              positive, and gates the whole signal out unless the source's are too.
              Correlation says "same contrast"; the levels say "same direction",
              and the sibling claim needs both. When the fourth signal cannot be
              used the ranking degrades to three signals and says so in the
              artifact, because a noisy signal must weaken a claim, not delete an
              experiment.

    co-ablate on held-out items, the seven conditions the work order asks for:
              clean, A alone, B alone, A+B together, and the last three repeated
              with norm-matched random directions at the same two layers. The
              statistic is a difference-in-differences -- the real pair's
              interaction minus the random pair's -- because the joint condition
              removes strictly more norm and the network is non-linear, so a
              positive interaction shows up even for unrelated directions. ``clean``
              is a row of its own carrying the un-differenced margin, since a delta
              says how far the lure lead moved but not what it moved from.

    repair    B's own activation with and without A ablated. If removing A makes B
              fire harder, that is the compensation the Hydra-effect literature
              predicts, measured directly rather than inferred from a margin.
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
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch

# The pipeline's single definition of "control twin": same scenario, cue removed,
# answer mapping preserved -- and the counterfactual twin refused outright, because
# it swaps correct and lure so differencing against it ADDS the two magnitudes.
# Re-deriving the pairing here would let the two definitions drift, and then this
# job's specificity signal would stop meaning what the discovery job's specificity
# gate means. It belongs in src/mindscopex_analysis/research.py; that file is shared,
# so the move is requested in the run report rather than made here.
from experiments.jobs.research_experiments import _pair_with_controls
from experiments.runners.config import load_toml, run_name, table
from mindscopex_analysis import (
    DEFAULT_ANALYSIS_PROFILE_KEY,
    EditSite,
    answer_logprob_margin,
    capture_layer_residuals,
    clear_device_cache,
    default_sae_device,
    difference_in_differences,
    dtype_from_name,
    family_balanced_subset,
    gaussian_null_directions,
    get_qwen35_analysis_profile,
    instruct_lure_cases,
    load_qwen_language_model,
    load_qwen_scope_sae,
    lure_dataset_cases,
    multi_site_answer_margin,
    paired_summary,
    pearson,
    qwen_scope_sparse_feature_values,
    rank_siblings,
    recommended_dtype_name,
    sae_decoder_direction,
    sibling_score,
    split_lure_cases,
    trace_logits_multi_site,
)
from mindscopex_analysis.effects import continuation_token_span
from mindscopex_analysis.modules import rescale_to_norm


def _log(message: str) -> None:
    print(f"[siblings] {message}", flush=True)


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
            # A signal that was not measured writes an empty cell, never the string
            # "None": pandas reads the former as NaN and the latter as a category,
            # which would silently turn a missing signal into a valid-looking value.
            writer.writerow(
                {
                    column: ("" if row.get(column, "") is None else row.get(column, ""))
                    for column in columns
                }
            )
    return path


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _fmt(value: Any, spec: str = "+.2f") -> str:
    """Format an optional signal for a log line without inventing a number for it."""

    return "n/a" if value is None or value == "" else format(float(value), spec)


# -------------------------------------------------------- four-signal ranking

# What the fourth signal is, and what it is not.
#
# ``specificity_corr`` is a Pearson r between two per-item vectors of hostile-minus-
# control ``margin_delta`` gaps -- the same per-item quantity that
# ``research.control_specificity_rows`` calls ``specificity_gap``. The per-item
# quantity matches that function's; the SIGNAL does not. A correlation says the two
# features move the gap *together*. It says nothing whatever about which way either of
# them moves it, so on its own it is an AGREEMENT statistic and not a specificity test:
# a pair that is anti-specific in both members correlates exactly as strongly as a pair
# that is cue-specific in both.
#
# docs/metrics_guide.md's rule for every differenced cue quantity is that it may only
# be read together with its hostile arm -- a positive gap under a non-positive hostile
# arm is a control effect, not a trap effect -- and ``control_specificity_rows`` returns
# ``hostile_margin_delta`` and ``control_margin_delta`` next to ``specificity_gap`` for
# exactly that reason. So the level enters here as a GATE and not as another term to be
# averaged. Two features are the same feature at two layers when they point the same
# way, fire on the same items, act the same way when removed, and discriminate the same
# contrast *in the same direction*. The correlation is the "same contrast" half; these
# verdicts are the "same direction" half, and neither half alone is the claim.

SPECIFICITY_ALIGNED = "aligned"
SPECIFICITY_UNMEASURED = "unmeasured"
SPECIFICITY_HOSTILE_ARM_NON_POSITIVE = "hostile_arm_non_positive"
SPECIFICITY_GAP_NON_POSITIVE = "gap_non_positive"


def _missing(value: Any) -> bool:
    return value is None or value == ""


def specificity_level_verdict(mean_effect: Any, mean_specificity: Any) -> str:
    """Does this feature discriminate the cue contrast in the direction claimed?

    ``mean_effect`` is the hostile arm: mean ``margin_delta`` = baseline - ablated over
    the matching items, and the margin is ``lure - correct``, so a POSITIVE value means
    ablating cut the lure's lead. ``mean_specificity`` is the mean gap against the
    matched no-cue twin. Both must be positive for a correlation between gap vectors to
    be evidence of siblinghood; either being non-positive is evidence against it, no
    matter how high that correlation is.

    Returns ``SPECIFICITY_UNMEASURED`` when either level is absent -- which happens when
    no control condition was configured -- so "not measured" is never scored as
    "measured and failed", and never silently scored as "measured and passed" either.
    """

    if _missing(mean_effect) or _missing(mean_specificity):
        return SPECIFICITY_UNMEASURED
    if float(mean_effect) <= 0.0:
        return SPECIFICITY_HOSTILE_ARM_NON_POSITIVE
    if float(mean_specificity) <= 0.0:
        return SPECIFICITY_GAP_NON_POSITIVE
    return SPECIFICITY_ALIGNED


def sibling_score_with_specificity(
    decoder_cosine: float,
    activation_corr: float,
    effect_corr: float,
    specificity_corr: float | None,
    *,
    specificity_weight: float = 1.0,
) -> float:
    """``sibling_score`` plus the hostile-vs-control agreement, folded in from outside.

    ``siblings.sibling_score`` takes exactly three signals and a 3-tuple of weights and
    lives in shared ``src/`` this job does not own. Forking it would leave the project
    with two ranking rules that drift apart, so this uses the algebra of the weighted
    geometric mean it already computes. With equal weights it returns
    ``G3 = (c*a*e)**(1/3)``, and for any ``w >= 0``::

        G3 ** (3 / (3 + w))  *  s ** (w / (3 + w))  ==  (c * a * e * s**w) ** (1 / (3 + w))

    which is exactly the four-term weighted geometric mean. Nothing about the ranking is
    approximated. The only duplicated part is the clamp, reapplied here for the same
    reason ``sibling_score`` applies it: a negative agreement on the gap is evidence
    *against* the pair being the same feature, and letting a strong cosine cancel it
    would rank an anti-correlated pair above an unrelated one. The trade-off is that the
    rule now lives in two places -- the preferred fix is a fourth parameter on
    ``sibling_score`` itself, requested in the run report.

    This function scores AGREEMENT only. It knows nothing about the two levels, so it
    must never be handed a ``specificity_corr`` whose levels have not been checked by
    ``specificity_level_verdict``; ``score_siblings`` is what enforces that pairing.

    ``specificity_corr=None`` means the signal is not usable (no control condition, or
    levels unmeasured) and falls back to the three-signal score rather than scoring 0,
    so "unmeasured" never masquerades as "measured and bad".
    """

    three = sibling_score(decoder_cosine, activation_corr, effect_corr)
    if specificity_corr is None:
        return three
    weight = float(specificity_weight)
    if weight < 0.0:
        raise ValueError("specificity_weight must be non-negative")
    if weight == 0.0:
        return three
    specificity = max(0.0, float(specificity_corr))
    if three <= 0.0 or specificity <= 0.0:
        return 0.0
    return three ** (3.0 / (3.0 + weight)) * specificity ** (weight / (3.0 + weight))


def score_siblings(
    candidates: Sequence[dict[str, Any]],
    *,
    specificity_weight: float = 1.0,
    source_verdict: str = SPECIFICITY_ALIGNED,
) -> list[dict[str, Any]]:
    """Annotate every candidate with its level verdict and its score. Drops nothing.

    A candidate the level gate rejects gets ``combined_score: None`` rather than a
    number, because it was not scored at all -- writing ``0.0`` there would be
    indistinguishable from "scored, and scored badly". The full annotated list is what
    goes to cross_layer_siblings.csv, so a rejected candidate stays visible in the
    artifact next to the verdict that rejected it and next to the two levels that
    produced the verdict.

    ``source_verdict`` must be the same value ``select_sibling`` was given, so the CSV
    and the summary agree about which rule scored the run. When the source's own levels
    failed, the fourth signal is not in use at all, and gating candidates on a signal
    nobody is reading would blank every score in the CSV while the summary still quoted
    one -- so in that case every candidate is scored on three signals and keeps its own
    verdict as a record.
    """

    signal_in_use = source_verdict in (SPECIFICITY_ALIGNED, SPECIFICITY_UNMEASURED)
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        raw = candidate.get("specificity_corr")
        measured = not _missing(raw)
        verdict = (
            specificity_level_verdict(
                candidate.get("mean_effect"), candidate.get("mean_specificity")
            )
            if measured
            else SPECIFICITY_UNMEASURED
        )
        row: dict[str, Any] = {**candidate, "specificity_verdict": verdict}
        gated_out = signal_in_use and verdict in (
            SPECIFICITY_HOSTILE_ARM_NON_POSITIVE,
            SPECIFICITY_GAP_NON_POSITIVE,
        )
        if gated_out:
            # Gated out on the level. Not ranked low -- not ranked at all.
            row["combined_score"] = None
        else:
            use_fourth = signal_in_use and verdict == SPECIFICITY_ALIGNED
            row["combined_score"] = sibling_score_with_specificity(
                float(candidate["decoder_cosine"]),
                float(candidate["activation_corr"]),
                float(candidate["effect_corr"]),
                float(raw) if use_fourth else None,
                specificity_weight=specificity_weight,
            )
        scored.append(row)
    return scored


def rank_siblings_with_specificity(
    candidates: Sequence[dict[str, Any]],
    *,
    specificity_weight: float = 1.0,
    min_score: float = 0.0,
    source_verdict: str = SPECIFICITY_ALIGNED,
) -> list[dict[str, Any]]:
    """``rank_siblings`` over four signals, best first, with the level gate applied.

    Same contract as the three-signal version: candidates at or below ``min_score`` are
    dropped rather than ranked, so "no sibling was found" stays distinguishable from
    "the best one was poor". Candidates the level gate rejected are dropped as well;
    ``score_siblings`` is the call that keeps them, with their verdict, for the CSV.
    """

    scored = [
        row
        for row in score_siblings(
            candidates, specificity_weight=specificity_weight, source_verdict=source_verdict
        )
        if row["combined_score"] is not None and row["combined_score"] > min_score
    ]
    scored.sort(key=lambda row: -row["combined_score"])
    return scored


# -------------------------------------------------------------- which rule won

SELECTION_FOUR_SIGNAL = "four_signal"
SELECTION_THREE_SIGNAL_UNMEASURED = "three_signal_specificity_unmeasured"
SELECTION_THREE_SIGNAL_SOURCE_NOT_SPECIFIC = "three_signal_source_not_cue_specific"
SELECTION_THREE_SIGNAL_FALLBACK = "three_signal_fallback"
SELECTION_NONE = "none"


def select_sibling(
    candidates: Sequence[dict[str, Any]],
    *,
    specificity_weight: float = 1.0,
    min_score: float = 0.0,
    source_verdict: str = SPECIFICITY_UNMEASURED,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """This layer's sibling, plus a record of exactly which rule picked it.

    The fourth signal can empty a ranking that the three-signal rule would have filled.
    ``sibling_score_with_specificity`` returns 0.0 on any non-positive agreement, and the
    candidates are the cosine-nearest neighbours of ONE source direction, so their
    per-item gap vectors are near-duplicates of each other -- effectively one draw, not
    ``candidate_top_n`` independent ones. At a true correlation near zero that single
    draw lands negative about half the time, and then every candidate scores 0.0, no
    sibling is selected, and the co-ablation, the difference-in-differences and the
    repair measurement never run at all -- after the match phase has already been paid
    for. A noisy fourth signal has to degrade the claim, not cancel the experiment, so
    the ranking falls back to three signals and the artifact records that it did.

    ``source_verdict`` gates the signal globally. Correlating candidate gap vectors
    against a source whose own hostile arm or own gap is non-positive would rank
    candidates on agreement with a contrast that is not there -- the case where two
    anti-specific features correlate at +0.9 and would otherwise be published as a
    confirmed pair -- so the fourth signal is dropped outright rather than reported.

    The returned provenance is what makes "no sibling exists at this layer" readable in
    the artifact as something other than "the fourth signal emptied the ranking".
    """

    annotated = score_siblings(
        candidates, specificity_weight=specificity_weight, source_verdict=source_verdict
    )
    measured = any(row["specificity_verdict"] != SPECIFICITY_UNMEASURED for row in annotated)
    source_ok = source_verdict == SPECIFICITY_ALIGNED

    # The three-signal fallback must not re-admit a candidate the LEVEL gate rejected.
    # Being specific in the wrong direction is evidence AGAINST the pair, not absence
    # of evidence, and rank_siblings recomputes combined_score from three signals -- so
    # ranking over the full annotated list quietly hands the layer back to exactly the
    # candidate the gate threw out. The gate only applies when it actually ran: with the
    # fourth signal unmeasured, or the source itself not cue-specific, nothing was gated
    # and every candidate is still eligible.
    gate_ran = measured and source_ok
    three_pool = (
        [row for row in annotated if row["combined_score"] is not None] if gate_ran else annotated
    )
    three = rank_siblings(three_pool, min_score=min_score)
    n_gate_excluded_from_three = len(annotated) - len(three_pool)

    if not measured:
        four: list[dict[str, Any]] = []
        degraded_rule = SELECTION_THREE_SIGNAL_UNMEASURED
        reason = (
            "no control condition was configured, so the hostile-vs-control signal was "
            "never measured and the ranking used three signals"
        )
    elif not source_ok:
        four = []
        degraded_rule = SELECTION_THREE_SIGNAL_SOURCE_NOT_SPECIFIC
        reason = (
            f"the source feature's own levels are {source_verdict!r}, so agreement with "
            "its gap vector would be agreement with a contrast that is not there; the "
            "fourth signal was dropped and the ranking used three"
        )
    else:
        four = rank_siblings_with_specificity(
            candidates,
            specificity_weight=specificity_weight,
            min_score=min_score,
            source_verdict=source_verdict,
        )
        degraded_rule = SELECTION_THREE_SIGNAL_FALLBACK
        reason = (
            f"no candidate cleared min_score={min_score} on four signals; ranked on three "
            "instead, so this layer's sibling is not backed by the specificity signal"
        )

    provenance: dict[str, Any] = {
        "min_score": float(min_score),
        "specificity_weight": float(specificity_weight),
        "n_candidates": len(annotated),
        "source_specificity_verdict": source_verdict,
        "n_specificity_level_rejected": sum(
            1 for row in annotated if row["combined_score"] is None
        ),
        "n_cleared_min_score_four_signal": len(four) if gate_ran else None,
        "n_cleared_min_score_three_signal": len(three),
        "n_level_gate_excluded_from_three_signal": n_gate_excluded_from_three,
    }
    if four:
        provenance.update(
            {
                "selection_rule": SELECTION_FOUR_SIGNAL,
                "specificity_signal_used": True,
                "degraded_reason": "",
                # The CSV carries the four-signal score for this feature; a three-signal
                # fallback carries a DIFFERENT number under the same column name. Naming
                # the basis is what stops the summary and the CSV being compared as if
                # they were the same quantity.
                "combined_score_basis": "four_signal",
            }
        )
        return four[0], provenance
    if three:
        if gate_ran and n_gate_excluded_from_three:
            reason = (
                f"{reason}; {n_gate_excluded_from_three} candidate(s) stayed excluded "
                "because the level gate rejected them outright"
            )
        provenance.update(
            {
                "selection_rule": degraded_rule,
                "specificity_signal_used": False,
                "degraded_reason": reason,
                # NOT comparable with the four-signal score the CSV carries for this
                # same feature -- rank_siblings recomputes it from three signals.
                "combined_score_basis": "three_signal",
            }
        )
        return three[0], provenance
    provenance.update(
        {
            "selection_rule": SELECTION_NONE,
            "specificity_signal_used": False,
            "degraded_reason": (
                "nothing cleared min_score on three signals either, so this is an absence "
                "of a sibling at this layer and not an artefact of the fourth signal"
            ),
        }
    )
    return None, provenance


# ---------------------------------------------------------------- row assembly


def _margin_row(margin: Any, baseline: Any) -> dict[str, float]:
    return {
        "margin_delta": float(baseline.margin) - float(margin.margin),
        # Sign convention is effects.py's, NOT margin_delta's: the logprob deltas are
        # ablated - baseline, so `correct > 0` means the edit RAISED the correct answer
        # and `lure < 0` means it LOWERED the lure. margin_delta stays baseline - ablated.
        # docs/metrics_guide.md documents this pair; writing them the other way round
        # silently inverts every reader's conclusion.
        "correct_logprob_delta": float(margin.correct.logprob) - float(baseline.correct.logprob),
        "lure_logprob_delta": float(margin.lure.logprob) - float(baseline.lure.logprob),
    }


def _absolute_row(margin: Any) -> dict[str, float]:
    """The un-differenced margin the deltas on the same row are measured against.

    A delta alone under-determines the claim: +0.4 off a clean margin of 5.0 and
    +0.4 off a clean margin of 0.4 are different results, and the ``clean`` condition
    the work order asks for is nothing but these three numbers. Carrying them on every
    row also makes each delta re-derivable if the sign convention is ever questioned
    again -- which it has been.
    """

    return {
        "margin": float(margin.margin),
        "correct_logprob": float(margin.correct.logprob),
        "lure_logprob": float(margin.lure.logprob),
    }


SIBLING_COLUMNS = (
    "source_layer",
    "source_feature",
    "target_layer",
    "target_feature",
    "decoder_cosine",
    "activation_corr",
    "effect_corr",
    # The correlation and, immediately next to it, the three levels that say which way
    # it points and whether it was allowed to count. Reading the correlation without
    # them is the error this ordering is meant to make awkward.
    "specificity_corr",
    "specificity_verdict",
    "mean_effect",
    "mean_specificity",
    "source_mean_effect",
    "source_mean_specificity",
    "combined_score",
    "mean_activation",
)


COABLATION_COLUMNS = (
    "case_id",
    "family",
    "target_layer",
    "target_feature",
    "condition",
    "draw",
    "margin",
    "correct_logprob",
    "lure_logprob",
    "margin_delta",
    "correct_logprob_delta",
    "lure_logprob_delta",
)


def _condition_row(
    case: Any,
    *,
    target_layer: int,
    target_feature: int,
    condition: str,
    draw: int,
    margin: Any,
    baseline: Any,
) -> dict[str, Any]:
    """One co-ablation arm as a CSV row. Passing ``margin=baseline`` gives ``clean``."""

    return {
        "case_id": case.case_id,
        "family": case.family,
        "target_layer": int(target_layer),
        "target_feature": int(target_feature),
        "condition": condition,
        "draw": int(draw),
        **_absolute_row(margin),
        **_margin_row(margin, baseline),
    }


# --------------------------------------------------------------------- setup


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
        "model_id": profile.analysis_model_id,
        "repo_id": profile.sae_repo_id,
        "dtype": dtype,
        "device_map": model_cfg.get("device_map", "auto"),
        "sae_device": sae_device,
        "sae_dtype": model_cfg.get("sae_dtype", dtype),
    }


def _load_splits(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = table(config, "data")
    dataset = str(data_cfg.get("dataset", "goal_affordance_traps_v1"))
    cases = lure_dataset_cases(dataset)
    conditions = data_cfg.get("conditions") or None
    if conditions:
        suffixes = tuple(f"_{name}" for name in conditions)
        cases = [case for case in cases if case.case_id.endswith(suffixes)]
    if bool(data_cfg.get("instruction", True)):
        cases = instruct_lure_cases(cases)
    train, test = split_lure_cases(
        cases,
        train_frac=float(data_cfg.get("train_frac", 0.6)),
        seed=int(data_cfg.get("split_seed", 0)),
    )
    max_match = int(data_cfg.get("max_match_items", 12))
    max_test = int(data_cfg.get("max_test_items", 12))
    match = family_balanced_subset(train, max_cases=max_match)

    # The control twin is deliberately NOT added to `[data].conditions`: that would
    # make no-cue items first-class match/test items, and the co-ablation would then
    # be run on prompts that contain no trap at all. It is a paired *second
    # measurement of the same item*, looked up by scenario across the whole dataset,
    # so the discovery/held-out split stays a split of scenarios and a twin can never
    # smuggle a held-out scenario into the matching phase.
    #
    # Default "": the fourth signal is opt-in. `_neutral` case-id twins exist in only
    # three of the datasets in src/mindscopex_analysis/data/ (goal_affordance_traps_v1,
    # v2, v2_micro). Defaulting to "neutral" turned every other multi-condition config
    # into a hard failure raised from inside _pair_with_controls, naming a condition the
    # user never wrote -- goal_affordance_traps_v21 (conditions absent/offered/immediate/
    # explicit/counterfactual) is the live example. Unset therefore means "three signals",
    # which the job logs loudly, rather than "crash before the model loads".
    control_condition = str(data_cfg.get("control_condition", "")).strip()
    controls: list[Any] = []
    if control_condition:
        if not conditions:
            # _pair_with_controls would raise here anyway; this says what to do about
            # it. A single-condition dataset has no twin to difference against, so the
            # fourth signal is unavailable rather than merely unconfigured.
            raise ValueError(
                "[data].control_condition needs [data].conditions to find each item's "
                "condition suffix. On a dataset with no conditions, set "
                'control_condition = "" -- which drops the specificity signal and '
                "leaves the sibling ranking on three."
            )
        match, controls = _pair_with_controls(match, config, control_condition)
    return {
        "dataset": dataset,
        "control_condition": control_condition,
        "match": match,
        "match_controls": controls,
        "test": family_balanced_subset(test, max_cases=max_test) if max_test else test,
    }


def _ablate_margin(lm: Any, case: Any, sites: Sequence[EditSite]) -> Any:
    return multi_site_answer_margin(
        lm,
        case.prompt,
        correct_answer=case.correct_answer,
        lure_answer=case.lure_answer,
        sites=sites,
    )


def _feature_value(residual: Any, sae: Any, feature_id: int) -> float:
    """The feature's sparse (post-TopK) value, which is what the edit removes.

    Preactivations are a different number and would scale every intervention wrongly;
    ``qwen_scope_feature_preactivations`` stays a diagnostic.
    """

    return float(
        qwen_scope_sparse_feature_values(residual, sae, [int(feature_id)])
        .detach()
        .to(torch.float32)
        .reshape(-1)[0]
    )


def run(config_path: Path, output_root: Path) -> Path:
    config = load_toml(config_path)
    name = run_name(config)
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config.toml")

    fcfg = table(config, "feature")
    source_feature = int(fcfg["feature_id"])
    source_layer = int(fcfg["layer"])

    mcfg = table(config, "match")
    candidate_top_n = int(mcfg.get("candidate_top_n", 12))
    min_score = float(mcfg.get("min_score", 0.05))
    specificity_weight = float(mcfg.get("specificity_weight", 1.0))
    random_draws = int(mcfg.get("random_draws", 6))
    seed = int(mcfg.get("seed", 0))
    target_layers = [int(v) for v in (mcfg.get("layers") or [])]

    env = _resolve_env(config)
    profile = env["profile"]
    if not target_layers:
        target_layers = [int(v) for v in profile.scan_layers if int(v) != source_layer]
    splits = _load_splits(config)
    controls: list[Any] = splits["match_controls"]
    if not controls:
        _log(
            "WARNING: [data].control_condition is empty, so the hostile-vs-control "
            "specificity signal cannot be measured; ranking falls back to three signals"
        )

    manifest: dict[str, Any] = {
        "run_name": name,
        "job": "cross_layer_siblings",
        "started_at": _timestamp(),
        "source_feature": source_feature,
        "source_layer": source_layer,
        "target_layers": target_layers,
        "profile": profile.key,
        "model_id": env["model_id"],
        "sae_repo_id": env["repo_id"],
        "dataset": splits["dataset"],
        "control_condition": splits["control_condition"],
        "specificity_signal": bool(controls),
        "specificity_weight": specificity_weight,
        "candidate_top_n": candidate_top_n,
        "n_match_items": len(splits["match"]),
        "n_test_items": len(splits["test"]),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        # Caveats that belong with the numbers, not only in the source. A reader who
        # opens an artifact and compares it with an older one has no other way to learn
        # either of these.
        "conventions": {
            "margin": "lure - correct logprob; positive means the lure is ahead",
            "margin_delta": "baseline - ablated (positive = the edit cut the lure lead)",
            "correct_logprob_delta": "ablated - baseline",
            "lure_logprob_delta": "ablated - baseline",
            "schema_change": (
                "coablation.csv's correct_logprob_delta and lure_logprob_delta carry the "
                "OPPOSITE sign to every artifact this job wrote before the four-signal "
                "revision: they were baseline - ablated and are now ablated - baseline, "
                "per docs/metrics_guide.md. margin_delta is unchanged. Do not compare "
                "those two columns across the two artifact generations without flipping "
                "one of them. New columns in the same revision: margin, correct_logprob, "
                "lure_logprob, and the `clean` condition whose three deltas are 0 by "
                "construction."
            ),
        },
        "fourth_signal": {
            "specificity_corr": (
                "Pearson r between the source's per-item hostile-minus-control "
                "margin_delta gap and the candidate's. It is an AGREEMENT statistic: it "
                "says the two features move that gap together, not which way either of "
                "them moves it. It is not by itself a specificity test."
            ),
            "level_gate": (
                "specificity_corr is used only when mean_effect > 0 (the hostile arm "
                "moved) and mean_specificity > 0 (the gap is cue-attributable) for the "
                "candidate, and the same holds for the source. Both levels are published "
                "next to the correlation in cross_layer_siblings.csv and in "
                "coablation_summary.json; specificity_verdict records the outcome."
            ),
        },
    }
    _write_json(run_dir / "manifest.json", manifest)

    _log(f"loading {env['model_id']} (source L{source_layer} #{source_feature})")
    lm = load_qwen_language_model(
        env["model_id"], device_map=env["device_map"], dtype=env["dtype"], dispatch=True
    )
    sae_dtype = dtype_from_name(env["sae_dtype"])
    source_sae = load_qwen_scope_sae(
        env["repo_id"], source_layer, device=env["sae_device"], dtype=sae_dtype
    )
    source_direction = sae_decoder_direction(source_sae, [source_feature]).detach()
    source_unit = source_direction.to(torch.float32).cpu() / source_direction.to(
        torch.float32
    ).cpu().norm().clamp_min(1e-12)
    started = time.time()

    # --- source profile on the matching items ------------------------------
    # Both baselines are candidate- and layer-independent, so they are paid once here
    # and reused by every candidate at every target layer.
    _log(
        f"profiling the source feature on {len(splits['match'])} discovery items"
        + (f" (+{len(controls)} {splits['control_condition']} twins)" if controls else "")
    )
    source_values: list[float] = []
    source_effects: list[float] = []
    source_specificity: list[float] = []
    baselines: list[float] = []
    control_baselines: list[float] = []
    for index, case in enumerate(splits["match"]):
        residual = capture_layer_residuals(lm, [case.prompt], source_layer, token_position="last")
        value = _feature_value(residual, source_sae, source_feature)
        baseline = float(
            answer_logprob_margin(
                lm,
                case.prompt,
                correct_answer=case.correct_answer,
                lure_answer=case.lure_answer,
            ).margin
        )
        ablated = float(
            _ablate_margin(
                lm,
                case,
                [EditSite(source_layer, source_direction, value, 1.0, "remove_activation")],
            ).margin
        )
        source_values.append(value)
        baselines.append(baseline)
        source_effects.append(baseline - ablated)
        if not controls:
            continue
        # The control is ablated with the feature's value ON THE CONTROL PROMPT, not
        # with the hostile one: remove_activation subtracts what is actually there, so
        # reusing the hostile value would inject an activation into a prompt that never
        # had it -- that is a steering experiment, not a specificity measurement.
        control = controls[index]
        control_residual = capture_layer_residuals(
            lm, [control.prompt], source_layer, token_position="last"
        )
        control_value = _feature_value(control_residual, source_sae, source_feature)
        control_baseline = float(
            answer_logprob_margin(
                lm,
                control.prompt,
                correct_answer=control.correct_answer,
                lure_answer=control.lure_answer,
            ).margin
        )
        control_ablated = float(
            _ablate_margin(
                lm,
                control,
                [EditSite(source_layer, source_direction, control_value, 1.0, "remove_activation")],
            ).margin
        )
        control_baselines.append(control_baseline)
        # research.control_specificity_rows' `specificity_gap`, per item: how much of
        # the ablation's effect is attributable to the cue rather than to the scenario
        # the two prompts share.
        source_specificity.append((baseline - ablated) - (control_baseline - control_ablated))

    # The source's own arm levels. The fourth signal correlates candidate gap vectors
    # against THIS vector, so if the source itself does not carry the contrast -- its
    # hostile arm did not move, or its gap is not positive -- then agreement with it is
    # agreement with nothing, and the signal is dropped rather than reported.
    source_mean_effect = _mean(source_effects) if source_effects else None
    source_mean_specificity = _mean(source_specificity) if source_specificity else None
    source_verdict = (
        specificity_level_verdict(source_mean_effect, source_mean_specificity)
        if source_specificity
        else SPECIFICITY_UNMEASURED
    )
    if source_verdict not in (SPECIFICITY_ALIGNED, SPECIFICITY_UNMEASURED):
        _log(
            f"WARNING: the source feature's own levels are {source_verdict} "
            f"(mean_effect {_fmt(source_mean_effect, '+.4f')}, "
            f"mean_specificity {_fmt(source_mean_specificity, '+.4f')}); the "
            "hostile-vs-control signal is unusable and the ranking falls back to three"
        )

    # --- candidate matching at each target layer ---------------------------
    # Cost per target layer, in forward passes (one residual capture = 1 forward, one
    # margin = 2 because correct and lure are scored in separate passes):
    #
    #     captures    max_match_items * n_conditions
    #     ablations   max_match_items * candidate_top_n * n_conditions * 2
    #
    # with n_conditions = 2 once a control twin is configured. That factor of two is
    # the entire price of the specificity signal, and it is why candidate_top_n's
    # default dropped from 20 to 12; the full arithmetic is in the config.
    sibling_rows: list[dict[str, Any]] = []
    best_by_layer: dict[int, dict[str, Any]] = {}
    selection_by_layer: dict[int, dict[str, Any]] = {}
    for target_layer in target_layers:
        target_sae = load_qwen_scope_sae(
            env["repo_id"], target_layer, device=env["sae_device"], dtype=sae_dtype
        )
        decoder = (
            (
                target_sae.W_dec.T
                if target_sae.W_dec.shape[0] == target_sae.d_model
                else target_sae.W_dec
            )
            .detach()
            .to(torch.float32)
            .cpu()
        )
        units = decoder / decoder.norm(dim=1, keepdim=True).clamp_min(1e-12)
        cosines = units @ source_unit
        candidates = torch.topk(cosines, min(candidate_top_n, cosines.numel())).indices.tolist()
        _log(f"L{target_layer}: scoring {len(candidates)} cosine-nearest candidates")

        # Per-candidate activation, effect and specificity on the SAME items, so all
        # four signals are comparable with the source's. Residuals are captured once
        # per item and reused across candidates: the SAE encode is cheap, the forward
        # pass is not.
        target_residuals = [
            capture_layer_residuals(lm, [case.prompt], target_layer, token_position="last")
            for case in splits["match"]
        ]
        control_residuals = [
            capture_layer_residuals(lm, [control.prompt], target_layer, token_position="last")
            for control in controls
        ]
        for position, candidate in enumerate(candidates, start=1):
            direction = sae_decoder_direction(target_sae, [int(candidate)]).detach()
            values: list[float] = []
            effects: list[float] = []
            specificity: list[float] = []
            for index, case in enumerate(splits["match"]):
                value = _feature_value(target_residuals[index], target_sae, int(candidate))
                ablated = float(
                    _ablate_margin(
                        lm,
                        case,
                        [EditSite(target_layer, direction, value, 1.0, "remove_activation")],
                    ).margin
                )
                values.append(value)
                effects.append(baselines[index] - ablated)
                if not control_residuals:
                    continue
                control_value = _feature_value(control_residuals[index], target_sae, int(candidate))
                control_ablated = float(
                    _ablate_margin(
                        lm,
                        controls[index],
                        [
                            EditSite(
                                target_layer, direction, control_value, 1.0, "remove_activation"
                            )
                        ],
                    ).margin
                )
                specificity.append(effects[-1] - (control_baselines[index] - control_ablated))
            sibling_rows.append(
                {
                    "source_layer": source_layer,
                    "source_feature": source_feature,
                    "target_layer": target_layer,
                    "target_feature": int(candidate),
                    "decoder_cosine": float(cosines[int(candidate)]),
                    "activation_corr": pearson(source_values, values),
                    "effect_corr": pearson(source_effects, effects),
                    "specificity_corr": (
                        pearson(source_specificity, specificity) if specificity else None
                    ),
                    "mean_activation": _mean(values),
                    "mean_effect": _mean(effects),
                    "mean_specificity": _mean(specificity) if specificity else None,
                    # Carried on every row so the candidate's gap can be compared with
                    # the source's gap in the CSV without opening the manifest.
                    "source_mean_effect": source_mean_effect,
                    "source_mean_specificity": source_mean_specificity,
                }
            )
            if position % 4 == 0:
                _log(f"L{target_layer}: {position}/{len(candidates)} candidates scored")
        best, provenance = select_sibling(
            [row for row in sibling_rows if row["target_layer"] == target_layer],
            specificity_weight=specificity_weight,
            min_score=min_score,
            source_verdict=source_verdict,
        )
        selection_by_layer[target_layer] = provenance
        if best is not None:
            best_by_layer[target_layer] = best
            # Every differenced quantity is logged next to the arm it must be read
            # against: a gap is only a cue effect if the hostile arm moved.
            _log(
                f"L{target_layer}: best #{best['target_feature']} "
                f"score {best['combined_score']:.3f} "
                f"via {provenance['selection_rule']} "
                f"(cos {best['decoder_cosine']:.2f}, "
                f"act {best['activation_corr']:+.2f}, "
                f"eff {best['effect_corr']:+.2f} [arm {_fmt(best.get('mean_effect'), '+.4f')}], "
                f"spec {_fmt(best.get('specificity_corr'))} "
                f"[gap {_fmt(best.get('mean_specificity'), '+.4f')}, "
                f"src gap {_fmt(source_mean_specificity, '+.4f')}], "
                f"levels {best.get('specificity_verdict')})"
            )
            if provenance["degraded_reason"]:
                _log(f"L{target_layer}: WARNING {provenance['degraded_reason']}")
        else:
            _log(
                f"L{target_layer}: no sibling selected -- {provenance['degraded_reason']} "
                f"({provenance['n_specificity_level_rejected']} of "
                f"{provenance['n_candidates']} candidates were rejected on their levels)"
            )
        del target_sae, decoder, units, target_residuals, control_residuals
        clear_device_cache()

    # Every candidate, including the ones the level gate rejected: those carry an empty
    # combined_score and the verdict that rejected them, so the artifact shows what was
    # excluded and why instead of silently shortening the list.
    annotated = score_siblings(
        sibling_rows, specificity_weight=specificity_weight, source_verdict=source_verdict
    )
    annotated.sort(key=lambda row: (row["combined_score"] is None, -(row["combined_score"] or 0.0)))
    _write_csv(run_dir / "cross_layer_siblings.csv", annotated, SIBLING_COLUMNS)
    if not best_by_layer:
        _write_json(
            run_dir / "coablation_summary.json",
            {
                "note": (
                    "no sibling was selected at any target layer, so the co-ablation, the "
                    "difference-in-differences and the repair measurement were not run. "
                    "Read `selection` per layer before concluding anything: "
                    "selection_rule='none' means nothing cleared min_score on three "
                    "signals either, which is an absence of a sibling; any other value "
                    "means the fourth signal was involved and degraded_reason says how."
                ),
                "min_score": min_score,
                "source_mean_effect": source_mean_effect,
                "source_mean_specificity": source_mean_specificity,
                "source_specificity_verdict": source_verdict,
                "selection": [
                    {"target_layer": layer, **prov} for layer, prov in selection_by_layer.items()
                ],
            },
        )
        manifest.update(
            {
                "finished_at": _timestamp(),
                "siblings": [],
                "source_mean_effect": source_mean_effect,
                "source_mean_specificity": source_mean_specificity,
                "source_specificity_verdict": source_verdict,
                "selection": [
                    {"target_layer": layer, **prov} for layer, prov in selection_by_layer.items()
                ],
            }
        )
        _write_json(run_dir / "manifest.json", manifest)
        print(f"ARTIFACT_DIR={run_dir}", flush=True)
        return run_dir

    # --- conditional co-ablation on held-out items -------------------------
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    # Two INDEPENDENT draws per null pair, not one direction reused at both layers.
    # A real sibling pair is aligned by construction; baking that alignment into the
    # null would subtract away the very structure the DiD is meant to detect.
    generator_directions = gaussian_null_directions(
        int(source_sae.d_model), random_draws * 2, seed=seed
    )

    for target_layer, sibling in best_by_layer.items():
        target_sae = load_qwen_scope_sae(
            env["repo_id"], target_layer, device=env["sae_device"], dtype=sae_dtype
        )
        target_feature = int(sibling["target_feature"])
        target_direction = sae_decoder_direction(target_sae, [target_feature]).detach()
        _log(f"co-ablation L{source_layer}#{source_feature} + L{target_layer}#{target_feature}")

        clean_margins: list[float] = []
        clean_correct: list[float] = []
        clean_lure: list[float] = []
        a_only: list[float] = []
        b_only: list[float] = []
        joint: list[float] = []
        rand_a: list[float] = []
        rand_b: list[float] = []
        rand_joint: list[float] = []
        repair: list[float] = []

        for item_index, case in enumerate(splits["test"], start=1):
            _, start = continuation_token_span(lm.tokenizer, case.prompt, case.correct_answer)
            edit_index = start - 1
            source_residual = capture_layer_residuals(
                lm, [case.prompt], source_layer, token_position="last"
            )
            a_value = _feature_value(source_residual, source_sae, source_feature)
            target_residual = capture_layer_residuals(
                lm, [case.prompt], target_layer, token_position="last"
            )
            b_value = _feature_value(target_residual, target_sae, target_feature)
            baseline = _ablate_margin(lm, case, [])
            site_a = EditSite(source_layer, source_direction, a_value, 1.0, "remove_activation")
            site_b = EditSite(target_layer, target_direction, b_value, 1.0, "remove_activation")

            def _record(condition: str, sites: Sequence[EditSite], draw: int = -1) -> float:
                row = _condition_row(
                    case,
                    target_layer=target_layer,
                    target_feature=target_feature,
                    condition=condition,
                    draw=draw,
                    margin=_ablate_margin(lm, case, sites),
                    baseline=baseline,
                )
                rows.append(row)
                return float(row["margin_delta"])

            # The seventh condition, and it is free: `baseline` is already the un-edited
            # forward pass every delta below is taken against. Its own deltas are 0 by
            # construction; what the row carries is the absolute margin and its logprob
            # decomposition, without which "the joint edit cut the margin by 0.4" cannot
            # be read as large or small.
            rows.append(
                _condition_row(
                    case,
                    target_layer=target_layer,
                    target_feature=target_feature,
                    condition="clean",
                    draw=-1,
                    margin=baseline,
                    baseline=baseline,
                )
            )
            clean_margins.append(float(baseline.margin))
            clean_correct.append(float(baseline.correct.logprob))
            clean_lure.append(float(baseline.lure.logprob))

            a_only.append(_record("a_only", [site_a]))
            b_only.append(_record("b_only", [site_b]))
            joint.append(_record("joint", [site_a, site_b]))

            # Norm-matched random directions at the SAME two layers: the joint edit
            # removes more norm than either part, so an unmatched null would make any
            # pair look superadditive.
            a_norm = abs(a_value) * float(source_direction.to(torch.float32).norm())
            b_norm = abs(b_value) * float(target_direction.to(torch.float32).norm())
            draw_a: list[float] = []
            draw_b: list[float] = []
            draw_joint: list[float] = []
            for draw in range(random_draws):
                rsite_a = EditSite(
                    source_layer,
                    rescale_to_norm(generator_directions[draw], a_norm),
                    1.0,
                    -1.0,
                    "add_vector",
                )
                rsite_b = EditSite(
                    target_layer,
                    rescale_to_norm(generator_directions[random_draws + draw], b_norm),
                    1.0,
                    -1.0,
                    "add_vector",
                )
                draw_a.append(_record("rand_a", [rsite_a], draw))
                draw_b.append(_record("rand_b", [rsite_b], draw))
                draw_joint.append(_record("rand_joint", [rsite_a, rsite_b], draw))
            rand_a.append(_mean(draw_a))
            rand_b.append(_mean(draw_b))
            rand_joint.append(_mean(draw_joint))

            # Does B fire harder once A is gone? Measured on the activation itself,
            # not inferred from a margin.
            _, captures = trace_logits_multi_site(
                lm,
                case.prompt,
                sites=[site_a],
                token_index=edit_index,
                capture_layers=[target_layer],
            )
            key = (
                (target_layer, "post")
                if (target_layer, "post") in captures
                else (target_layer, "pre")
            )
            edited = captures[key].to(torch.float32)
            b_after = _feature_value(edited, target_sae, target_feature)
            repair.append(b_after - b_value)

            if item_index % 4 == 0:
                _log(f"L{target_layer}: {item_index}/{len(splits['test'])} items")
            _write_csv(run_dir / "coablation.csv", rows, COABLATION_COLUMNS)

        did = difference_in_differences(joint, [a_only, b_only], rand_joint, [rand_a, rand_b])
        selection = selection_by_layer[target_layer]
        summaries.append(
            {
                "target_layer": target_layer,
                "target_feature": target_feature,
                "sibling_score": sibling["combined_score"],
                "score_n_signals": 4 if selection["specificity_signal_used"] else 3,
                "decoder_cosine": sibling["decoder_cosine"],
                "activation_corr": sibling["activation_corr"],
                "effect_corr": sibling["effect_corr"],
                "specificity_corr": sibling["specificity_corr"],
                # The three levels the correlation above must be read against. Without
                # them `specificity_corr: 0.9` reads as "confirmed cue-specific" even
                # when neither member of the pair is; they used to live only in
                # cross_layer_siblings.csv while this is the file that gets read.
                "specificity_verdict": sibling.get("specificity_verdict"),
                "mean_effect": sibling.get("mean_effect"),
                "mean_specificity": sibling.get("mean_specificity"),
                "source_mean_effect": source_mean_effect,
                "source_mean_specificity": source_mean_specificity,
                "selection": selection,
                # No sign-flip test on `clean`: it is an absolute level, not a paired
                # difference, so permuting its signs would test nothing.
                "clean": {
                    "n": len(clean_margins),
                    "mean_margin": _mean(clean_margins),
                    "mean_correct_logprob": _mean(clean_correct),
                    "mean_lure_logprob": _mean(clean_lure),
                },
                # stats.paired_summary is the study's single definition of a paired
                # p-value: (b+1)/(draws+1), so a p can never be reported as exactly 0.0,
                # plus a bootstrap CI and n_positive. The private copy this job used
                # returned b/draws and no interval, which gave the same evidence two
                # different p-values across artifacts of one study and hid the case where
                # one dominating item carries the mean.
                "a_only": paired_summary(a_only, seed=seed),
                "b_only": paired_summary(b_only, seed=seed),
                "joint": paired_summary(joint, seed=seed),
                "difference_in_differences": paired_summary(did, seed=seed),
                "mean_rand_joint": _mean(rand_joint),
                # Positive means B fires HARDER once A is removed: compensation.
                "sibling_repair": paired_summary(repair, seed=seed),
            }
        )
        _write_json(run_dir / "coablation_summary.json", summaries)
        did_summary = summaries[-1]["difference_in_differences"]
        _log(
            f"L{target_layer}: clean {summaries[-1]['clean']['mean_margin']:+.4f} | "
            f"joint {summaries[-1]['joint']['mean']:+.4f} | "
            f"DiD {did_summary['mean']:+.4f} "
            f"(p={did_summary['p']:.4f}, "
            f"CI [{did_summary['ci_low']:+.4f}, {did_summary['ci_high']:+.4f}], "
            f"{did_summary['n_positive']}/{did_summary['n']} positive) | "
            f"repair {summaries[-1]['sibling_repair']['mean']:+.4f} | "
            f"spec {_fmt(sibling.get('specificity_corr'))} "
            f"[arm {_fmt(sibling.get('mean_effect'), '+.4f')}, "
            f"gap {_fmt(sibling.get('mean_specificity'), '+.4f')}] "
            f"via {selection['selection_rule']}"
        )
        del target_sae
        clear_device_cache()

    manifest.update(
        {
            "finished_at": _timestamp(),
            "elapsed_seconds": round(time.time() - started, 1),
            "source_mean_effect": source_mean_effect,
            "source_mean_specificity": source_mean_specificity,
            "source_specificity_verdict": source_verdict,
            "siblings": [
                {k: v for k, v in row.items() if k != "combined_score"}
                for row in best_by_layer.values()
            ],
            "selection": [
                {"target_layer": layer, **prov} for layer, prov in selection_by_layer.items()
            ],
            "summaries": summaries,
        }
    )
    _write_json(run_dir / "manifest.json", manifest)
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
