"""Rigor primitives for the CRT lure-feature study.

The exploratory helpers in :mod:`mindscopex_analysis.workflows` describe *one*
case: they find the feature whose ablation most reduces the bat-and-ball lure
margin and report that same margin drop. That is selection-on-the-outcome, so
the number cannot be read as an effect size. The helpers here add the controls a
skeptical reviewer needs:

* ``split_lure_cases`` — deterministic, family-stratified train/test split so
  feature discovery and coefficient choice happen only on a discovery split and
  are applied to held-out items (see ``docs/datasets.md`` principle 4).
* ``random_direction_margin_deltas`` / ``null_summary`` — a null distribution
  from removing *random* directions of matched norm, so a feature's margin drop
  can be z-scored instead of taken at face value.
* ``discover_generalizing_feature`` — rank features by their *mean* margin drop
  across many discovery items (not one), with an activation-frequency filter.
* ``control_specificity_rows`` — the same feature on hostile vs matched-control
  prompts; a lure feature should move the hostile margin far more than the
  control margin.
* ``steer_generation_labels`` — the behavioral readout: does suppressing the
  feature during constrained correct-vs-lure generation shift the selected
  answer, not just the teacher-forced logprob margin?

Every model-dependent function is a thin wrapper over the already-tested
:func:`~mindscopex_analysis.effects.answer_logprob_margin`,
:func:`~mindscopex_analysis.generation.generate_qwen_text_response`, and
:func:`~mindscopex_analysis.qwen_scope.make_feature_steering_hook`.
"""

from __future__ import annotations

import hashlib
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch

from mindscopex_analysis.activations import capture_layer_residuals, get_module
from mindscopex_analysis.cases import LureCase
from mindscopex_analysis.effects import (
    InterventionMode,
    active_prompt_features,
    answer_logprob_margin,
)
from mindscopex_analysis.generation import classify_lure_answer
from mindscopex_analysis.models import DEFAULT_BLOCK_PATH_TEMPLATE
from mindscopex_analysis.qwen_scope import (
    QwenScopeSAE,
    make_feature_steering_hook,
    qwen_scope_sparse_feature_values,
    sae_decoder_direction,
)

# ------------------------------------------------------------------- splitting


def _stable_bucket(case_id: str, seed: int) -> int:
    """Deterministic 0-99 bucket for a case id (stable across processes)."""

    digest = hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest()
    return int(digest[:8], 16) % 100


def split_lure_cases(
    cases: Sequence[LureCase],
    *,
    train_frac: float = 0.6,
    seed: int = 0,
    stratify_by_family: bool = True,
) -> tuple[list[LureCase], list[LureCase]]:
    """Split cases into ``(discovery, held_out)`` deterministically.

    Uses a stable hash of ``case_id`` (not RNG state), so the same split is
    reproduced on every machine and run. With ``stratify_by_family`` the fraction
    is applied within each family so the split keeps the family balance.
    """

    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    cutoff = round(train_frac * 100)

    def take(subset: Sequence[LureCase]) -> tuple[list[LureCase], list[LureCase]]:
        train: list[LureCase] = []
        test: list[LureCase] = []
        for case in subset:
            (train if _stable_bucket(case.case_id, seed) < cutoff else test).append(case)
        return train, test

    if not stratify_by_family:
        return take(cases)

    by_family: dict[str, list[LureCase]] = {}
    for case in cases:
        by_family.setdefault(case.family, []).append(case)
    train: list[LureCase] = []
    test: list[LureCase] = []
    for family in sorted(by_family):
        family_train, family_test = take(by_family[family])
        train.extend(family_train)
        test.extend(family_test)
    return train, test


def family_balanced_subset(
    cases: Sequence[LureCase],
    *,
    max_cases: int,
) -> list[LureCase]:
    """Take a deterministic round-robin subset balanced across case families."""

    if max_cases < 0:
        raise ValueError("max_cases must be non-negative")
    if max_cases == 0 or max_cases >= len(cases):
        return list(cases)

    by_family: dict[str, list[LureCase]] = {}
    for case in cases:
        by_family.setdefault(case.family, []).append(case)

    families = sorted(by_family)
    offsets = {family: 0 for family in families}
    subset: list[LureCase] = []
    while len(subset) < max_cases:
        added = False
        for family in families:
            offset = offsets[family]
            rows = by_family[family]
            if offset >= len(rows):
                continue
            subset.append(rows[offset])
            offsets[family] += 1
            added = True
            if len(subset) == max_cases:
                break
        if not added:
            break
    return subset


# ----------------------------------------------------------------- null model


def _perturbation_norm(direction: torch.Tensor, feature_value: float, coefficient: float) -> float:
    """L2 norm of the vector ``remove_activation`` subtracts for one feature."""

    return float(abs(coefficient) * abs(feature_value) * torch.linalg.norm(direction.float()))


def random_direction_margin_deltas(
    lm: Any,
    case: LureCase,
    *,
    layer: int,
    d_model: int,
    target_norm: float,
    n_samples: int = 32,
    seed: int = 0,
    block_path_template: str = DEFAULT_BLOCK_PATH_TEMPLATE,
    baseline_margin: float | None = None,
) -> list[float]:
    """Margin deltas from removing ``n_samples`` random directions of matched norm.

    This is the null for a feature's ``margin_delta``: it answers "does removing
    *this* direction reduce the lure margin more than removing a random vector of
    the same magnitude?" Returns one ``baseline - ablated`` delta per sample.
    """

    if baseline_margin is None:
        baseline_margin = answer_logprob_margin(
            lm,
            case.prompt,
            correct_answer=case.correct_answer,
            lure_answer=case.lure_answer,
            block_path_template=block_path_template,
        ).margin

    generator = torch.Generator().manual_seed(int(seed))
    deltas: list[float] = []
    for _ in range(int(n_samples)):
        vector = torch.randn(int(d_model), generator=generator)
        norm = torch.linalg.norm(vector).clamp_min(1e-12)
        vector = vector / norm * float(target_norm)
        ablated = answer_logprob_margin(
            lm,
            case.prompt,
            correct_answer=case.correct_answer,
            lure_answer=case.lure_answer,
            layer=int(layer),
            direction=vector,
            feature_value=1.0,
            coefficient=-1.0,  # add_vector with -1 subtracts the random vector
            intervention_mode="add_vector",
            block_path_template=block_path_template,
        ).margin
        deltas.append(float(baseline_margin) - ablated)
    return deltas


def random_direction_null_for_feature(
    lm: Any,
    case: LureCase,
    *,
    layer: int,
    sae: QwenScopeSAE,
    feature_id: int,
    feature_value: float,
    coefficient: float = 1.0,
    n_samples: int = 32,
    seed: int = 0,
    block_path_template: str = DEFAULT_BLOCK_PATH_TEMPLATE,
    baseline_margin: float | None = None,
) -> list[float]:
    """Null margin deltas matched to one feature's perturbation norm at ``layer``."""

    direction = sae_decoder_direction(sae, [int(feature_id)])
    target_norm = _perturbation_norm(direction, float(feature_value), float(coefficient))
    return random_direction_margin_deltas(
        lm,
        case,
        layer=int(layer),
        d_model=int(sae.d_model),
        target_norm=target_norm,
        n_samples=n_samples,
        seed=seed,
        block_path_template=block_path_template,
        baseline_margin=baseline_margin,
    )


def null_summary(observed_delta: float, null_deltas: Sequence[float]) -> dict[str, Any]:
    """Effect size of ``observed_delta`` against a null distribution."""

    values = [float(value) for value in null_deltas]
    if not values:
        return {"null_n": 0, "null_mean": None, "null_std": None, "z": None, "percentile": None}
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    z = (float(observed_delta) - mean) / std if std > 1e-12 else None
    percentile = sum(1 for value in values if value < float(observed_delta)) / len(values)
    return {
        "null_n": len(values),
        "null_mean": mean,
        "null_std": std,
        "z": z,
        "percentile": percentile,
    }


# ----------------------------------------------------- generalizing discovery


def _feature_activation(residual: torch.Tensor, sae: QwenScopeSAE, feature_id: int) -> float:
    vector = residual if residual.dim() > 1 else residual.unsqueeze(0)
    # The SAE's real activation, not the pre-activation: a feature outside the TopK
    # support contributes nothing to the reconstruction, so ablating it by its
    # pre-activation would subtract a contribution the model never made.
    values = qwen_scope_sparse_feature_values(vector, sae, [int(feature_id)])
    return float(values[0, 0])


def aggregate_feature_effect(
    lm: Any,
    cases: Sequence[LureCase],
    *,
    layer: int,
    sae: QwenScopeSAE,
    feature_id: int,
    coefficient: float = 1.0,
    intervention_mode: InterventionMode = "remove_activation",
    per_case_value: bool = True,
    fixed_feature_value: float = 0.0,
    block_path_template: str = DEFAULT_BLOCK_PATH_TEMPLATE,
) -> dict[str, Any]:
    """Mean margin delta of one feature applied across many cases.

    With ``per_case_value`` the feature's *actual* activation in each case is
    used, so ``remove_activation`` is a proper per-item ablation rather than a
    fixed-magnitude steer.
    """

    direction = sae_decoder_direction(sae, [int(feature_id)])
    deltas: list[float] = []
    per_case: list[dict[str, Any]] = []
    for case in cases:
        if per_case_value:
            residual = capture_layer_residuals(
                lm,
                [case.prompt],
                int(layer),
                token_position="last",
                block_path_template=block_path_template,
            )
            value = _feature_activation(residual, sae, int(feature_id))
        else:
            value = float(fixed_feature_value)
        baseline = answer_logprob_margin(
            lm,
            case.prompt,
            correct_answer=case.correct_answer,
            lure_answer=case.lure_answer,
            block_path_template=block_path_template,
        ).margin
        ablated = answer_logprob_margin(
            lm,
            case.prompt,
            correct_answer=case.correct_answer,
            lure_answer=case.lure_answer,
            layer=int(layer),
            direction=direction,
            feature_value=value,
            coefficient=coefficient,
            intervention_mode=intervention_mode,
            block_path_template=block_path_template,
        ).margin
        delta = baseline - ablated
        deltas.append(delta)
        per_case.append(
            {
                "case_id": case.case_id,
                "family": case.family,
                "feature_value": value,
                "baseline_margin": baseline,
                "edited_margin": ablated,
                "margin_delta": delta,
            }
        )
    n = len(deltas)
    return {
        "feature_id": int(feature_id),
        "layer": int(layer),
        "n_cases": n,
        "mean_margin_delta": statistics.fmean(deltas) if n else 0.0,
        "std_margin_delta": statistics.pstdev(deltas) if n > 1 else 0.0,
        "frac_positive": (sum(1 for d in deltas if d > 0) / n) if n else 0.0,
        "per_case": per_case,
    }


def discover_generalizing_feature(
    lm: Any,
    cases: Sequence[LureCase],
    *,
    layer: int,
    sae: QwenScopeSAE,
    candidate_top_n: int = 12,
    min_active_cases: int = 2,
    max_candidates: int = 40,
    coefficient: float = 1.0,
    intervention_mode: InterventionMode = "remove_activation",
    control_cases: Sequence[LureCase] | None = None,
    block_path_template: str = DEFAULT_BLOCK_PATH_TEMPLATE,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Rank features by mean margin delta across ``cases`` (a discovery split).

    Candidate features are those active (top-``candidate_top_n``) in at least
    ``min_active_cases`` discovery items, capped at ``max_candidates`` by
    activation frequency. Returns aggregate rows sorted by ``mean_margin_delta``.

    With ``control_cases`` -- one control item per case, in the same order -- the
    ranked quantity becomes the *cue effect*, ``delta(case) - delta(control)``,
    rather than the raw margin delta. A hostile item's margin is its control's
    margin plus whatever the salient cue added, so ranking on the raw delta also
    rewards features that merely weaken the model's baseline judgement; taking the
    paired difference cancels that shared component and scores only the part the
    cue carries. Candidate *selection* still uses the hostile items, because that
    is where the trap's representation has to be active in the first place.
    """

    if control_cases is not None and len(control_cases) != len(cases):
        raise ValueError("control_cases must be aligned one-to-one with cases")

    # Precompute each case's residual + baseline margin once; both are
    # candidate-independent, so recomputing them per candidate would ~double cost.
    contexts: list[tuple[LureCase, torch.Tensor, float]] = []
    frequency: Counter[int] = Counter()
    for case in cases:
        residual = capture_layer_residuals(
            lm,
            [case.prompt],
            int(layer),
            token_position="last",
            block_path_template=block_path_template,
        )
        baseline = answer_logprob_margin(
            lm,
            case.prompt,
            correct_answer=case.correct_answer,
            lure_answer=case.lure_answer,
            block_path_template=block_path_template,
        ).margin
        contexts.append((case, residual, baseline))
        for feature_id, _value in active_prompt_features(residual, sae, top_n=candidate_top_n):
            frequency[int(feature_id)] += 1

    # Control baselines and activations, when scoring the cue effect. The control
    # shares the answer pair, so its activation is read with its own residual.
    control_contexts: list[tuple[LureCase, torch.Tensor, float]] = []
    for control in control_cases or ():
        control_residual = capture_layer_residuals(
            lm,
            [control.prompt],
            int(layer),
            token_position="last",
            block_path_template=block_path_template,
        )
        control_baseline = answer_logprob_margin(
            lm,
            control.prompt,
            correct_answer=control.correct_answer,
            lure_answer=control.lure_answer,
            block_path_template=block_path_template,
        ).margin
        control_contexts.append((control, control_residual, control_baseline))

    candidates = [fid for fid, count in frequency.most_common() if count >= min_active_cases]
    candidates = candidates[: int(max_candidates)]
    if progress is not None:
        progress(f"layer {int(layer)}: {len(candidates)} candidates x {len(contexts)} cases")

    activation_rows = [
        qwen_scope_sparse_feature_values(residual, sae, candidates)[0].detach().float().cpu()
        for _case, residual, _baseline in contexts
    ]
    control_activation_rows = [
        qwen_scope_sparse_feature_values(residual, sae, candidates)[0].detach().float().cpu()
        for _control, residual, _baseline in control_contexts
    ]
    rows: list[dict[str, Any]] = []
    for candidate_index, feature_id in enumerate(candidates):
        direction = sae_decoder_direction(sae, [int(feature_id)])
        deltas: list[float] = []
        control_deltas: list[float] = []
        for case_index, (case, _residual, baseline_margin) in enumerate(contexts):
            value = float(activation_rows[case_index][candidate_index])
            ablated = answer_logprob_margin(
                lm,
                case.prompt,
                correct_answer=case.correct_answer,
                lure_answer=case.lure_answer,
                layer=int(layer),
                direction=direction,
                feature_value=value,
                coefficient=coefficient,
                intervention_mode=intervention_mode,
                block_path_template=block_path_template,
            ).margin
            delta = baseline_margin - ablated
            if control_contexts:
                control, _control_residual, control_baseline = control_contexts[case_index]
                control_value = float(control_activation_rows[case_index][candidate_index])
                control_ablated = answer_logprob_margin(
                    lm,
                    control.prompt,
                    correct_answer=control.correct_answer,
                    lure_answer=control.lure_answer,
                    layer=int(layer),
                    direction=direction,
                    feature_value=control_value,
                    coefficient=coefficient,
                    intervention_mode=intervention_mode,
                    block_path_template=block_path_template,
                ).margin
                control_deltas.append(control_baseline - control_ablated)
                # The cue effect: what the ablation removed from the hostile item
                # beyond what it removed from the same item without the cue.
                delta = delta - control_deltas[-1]
            deltas.append(delta)
        n = len(deltas)
        rows.append(
            {
                "feature_id": int(feature_id),
                "layer": int(layer),
                "n_cases": n,
                "objective": "cue_effect" if control_contexts else "margin_delta",
                "mean_control_delta": (
                    statistics.fmean(control_deltas) if control_deltas else None
                ),
                "mean_margin_delta": statistics.fmean(deltas) if n else 0.0,
                "std_margin_delta": statistics.pstdev(deltas) if n > 1 else 0.0,
                "frac_positive": (sum(1 for d in deltas if d > 0) / n) if n else 0.0,
                "active_in_cases": frequency[int(feature_id)],
            }
        )
        if progress is not None:
            progress(
                f"  [{candidate_index + 1}/{len(candidates)}] feature {int(feature_id)} "
                f"mean_delta={rows[-1]['mean_margin_delta']:+.4f}"
            )
    rows.sort(key=lambda row: row["mean_margin_delta"], reverse=True)
    return rows


# --------------------------------------------------------- control specificity


def control_specificity_rows(
    lm: Any,
    cases: Sequence[LureCase],
    *,
    layer: int,
    sae: QwenScopeSAE,
    feature_id: int,
    coefficient: float = 1.0,
    intervention_mode: InterventionMode = "remove_activation",
    per_case_value: bool = True,
    fixed_feature_value: float = 0.0,
    block_path_template: str = DEFAULT_BLOCK_PATH_TEMPLATE,
) -> list[dict[str, Any]]:
    """Per-case margin delta on the hostile prompt vs its matched control prompt.

    A lure feature should move the hostile margin (``hostile_margin_delta``) far
    more than the control margin (``control_margin_delta``). Cases without a
    ``control_prompt`` are skipped.
    """

    direction = sae_decoder_direction(sae, [int(feature_id)])

    def delta_for(prompt: str) -> tuple[float, float, float]:
        if per_case_value:
            residual = capture_layer_residuals(
                lm,
                [prompt],
                int(layer),
                token_position="last",
                block_path_template=block_path_template,
            )
            value = _feature_activation(residual, sae, int(feature_id))
        else:
            value = float(fixed_feature_value)
        base = answer_logprob_margin(
            lm,
            prompt,
            correct_answer=case.correct_answer,
            lure_answer=case.lure_answer,
            block_path_template=block_path_template,
        ).margin
        edited = answer_logprob_margin(
            lm,
            prompt,
            correct_answer=case.correct_answer,
            lure_answer=case.lure_answer,
            layer=int(layer),
            direction=direction,
            feature_value=value,
            coefficient=coefficient,
            intervention_mode=intervention_mode,
            block_path_template=block_path_template,
        ).margin
        return base, edited, base - edited

    rows: list[dict[str, Any]] = []
    for case in cases:
        if not case.control_prompt:
            continue
        hostile_base, hostile_edit, hostile_delta = delta_for(case.prompt)
        control_base, control_edit, control_delta = delta_for(case.control_prompt)
        rows.append(
            {
                "case_id": case.case_id,
                "family": case.family,
                "hostile_baseline_margin": hostile_base,
                "hostile_edited_margin": hostile_edit,
                "hostile_margin_delta": hostile_delta,
                "control_baseline_margin": control_base,
                "control_edited_margin": control_edit,
                "control_margin_delta": control_delta,
                "specificity_gap": hostile_delta - control_delta,
            }
        )
    return rows


# --------------------------------------------------------- behavioral readout

BehavioralOutputMode = Literal["binary_choice", "free"]


def find_decoder_block(
    model: Any,
    layer: int,
    *,
    block_path_template: str = DEFAULT_BLOCK_PATH_TEMPLATE,
) -> Any:
    """Locate a decoder block module for hooking, robust to wrapper prefixes.

    Tries the dotted template first, then searches ``named_modules()`` for the
    single module whose qualified name ends in ``layers.<layer>`` so it works
    whether the block lives at ``model.language_model.layers.N`` or under an
    extra wrapper.
    """

    path = block_path_template.format(layer=int(layer), i=int(layer))
    try:
        module = get_module(model, path)
        if hasattr(module, "register_forward_hook"):
            return module
    except (AttributeError, IndexError, KeyError, TypeError):
        pass

    target = ("layers", str(int(layer)))
    matches = [
        module for name, module in model.named_modules() if tuple(name.split(".")[-2:]) == target
    ]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Could not resolve a unique decoder block for layer {layer} "
        f"(template {path!r} failed; {len(matches)} suffix matches)."
    )


def summarize_answer_labels(labels: Sequence[str]) -> dict[str, Any]:
    """Accuracy / lure-rate summary from ``classify_lure_answer`` labels."""

    counts = Counter(labels)
    n = len(labels)
    return {
        "n": n,
        "correct": counts.get("correct", 0),
        "lure": counts.get("lure", 0),
        "both": counts.get("both", 0),
        "other": counts.get("other", 0),
        "accuracy": counts.get("correct", 0) / n if n else 0.0,
        "lure_rate": counts.get("lure", 0) / n if n else 0.0,
    }


def _greedy_completion(model: Any, tokenizer: Any, prompt: str, *, max_new_tokens: int) -> str:
    """Minimal deterministic base-model completion of ``prompt``.

    Unwraps an ``AutoProcessor`` to its inner text tokenizer so this works on
    plain base models without a chat template — the behavioral readout is a
    text-completion, not a chat, so it must not depend on the chat/processor path.
    """

    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    encoded = text_tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    gen_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
    }
    if "attention_mask" in encoded:
        gen_kwargs["attention_mask"] = encoded["attention_mask"].to(device)
    pad_id = text_tokenizer.pad_token_id
    if pad_id is None:
        eos = text_tokenizer.eos_token_id
        pad_id = eos[0] if isinstance(eos, (list, tuple)) else eos
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = int(pad_id)
    with torch.inference_mode():
        output = model.generate(**gen_kwargs)
    generated = output[0, input_ids.shape[-1] :]
    return text_tokenizer.decode(generated, skip_special_tokens=True)


def _binary_choice_completion(
    model: Any,
    tokenizer: Any,
    case: LureCase,
    *,
    max_new_tokens: int,
) -> str:
    """Generate exactly one of a case's correct/lure answer strings.

    Qwen's ``enable_thinking=False`` is a chat-template switch, not a
    ``model.generate`` argument.  The SAE study deliberately generates from the
    same Base checkpoint used for feature discovery, which has no applicable
    chat template.  Prefix-constrained decoding is therefore the hard,
    checkpoint-preserving way to prevent ``<think>`` and third-answer outputs.
    """

    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    candidates = [case.correct_answer, case.lure_answer]
    candidate_ids = [
        list(text_tokenizer.encode(answer, add_special_tokens=False)) for answer in candidates
    ]
    if any(not ids for ids in candidate_ids):
        raise ValueError(f"Case {case.case_id!r} has an answer with no tokenizer tokens")
    if candidate_ids[0] == candidate_ids[1]:
        raise ValueError(f"Case {case.case_id!r} has token-identical correct/lure answers")

    encoded = text_tokenizer(case.prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    prompt_length = int(input_ids.shape[-1])

    eos = text_tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Binary constrained generation requires an eos_token_id")
    eos_ids = [int(token_id) for token_id in eos] if isinstance(eos, (list, tuple)) else [int(eos)]

    def allowed_tokens(_batch_id: int, sequence: torch.Tensor) -> list[int]:
        generated = sequence[prompt_length:].tolist()
        allowed: set[int] = set()
        for ids in candidate_ids:
            if generated == ids[: len(generated)]:
                if len(generated) < len(ids):
                    allowed.add(int(ids[len(generated)]))
                else:
                    allowed.update(eos_ids)
        if not allowed:
            raise RuntimeError(
                f"Constrained generation left the answer trie for case {case.case_id!r}: "
                f"{generated!r}"
            )
        return sorted(allowed)

    gen_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "max_new_tokens": max(int(max_new_tokens), max(map(len, candidate_ids)) + 1),
        "do_sample": False,
        "prefix_allowed_tokens_fn": allowed_tokens,
        "renormalize_logits": True,
        "eos_token_id": eos_ids,
    }
    if "attention_mask" in encoded:
        gen_kwargs["attention_mask"] = encoded["attention_mask"].to(device)
    pad_id = text_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_ids[0]
    gen_kwargs["pad_token_id"] = int(pad_id)

    with torch.inference_mode():
        output = model.generate(**gen_kwargs)
    generated = output[0, prompt_length:].tolist()
    while generated and generated[-1] in eos_ids:
        generated.pop()

    for answer, ids in zip(candidates, candidate_ids, strict=True):
        if generated == ids:
            return answer.strip()
    raise RuntimeError(
        f"Constrained generation did not finish a valid answer for case {case.case_id!r}: "
        f"{generated!r}"
    )


def _generate_labels(
    model: Any,
    tokenizer: Any,
    cases: Sequence[LureCase],
    *,
    max_new_tokens: int,
    output_mode: BehavioralOutputMode = "binary_choice",
    progress: Callable[[str], None] | None = None,
    phase: str = "",
) -> list[dict[str, Any]]:
    if output_mode not in {"binary_choice", "free"}:
        raise ValueError(f"Unknown behavioral output_mode={output_mode!r}")

    rows: list[dict[str, Any]] = []
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        if output_mode == "binary_choice":
            text = _binary_choice_completion(
                model,
                tokenizer,
                case,
                max_new_tokens=max_new_tokens,
            )
        else:
            text = _greedy_completion(
                model,
                tokenizer,
                case.prompt,
                max_new_tokens=max_new_tokens,
            )
        label = classify_lure_answer(text, case)
        if output_mode == "binary_choice" and label not in {"correct", "lure"}:
            raise RuntimeError(
                f"Binary choice produced unexpected label {label!r} for case {case.case_id!r}"
            )
        rows.append(
            {
                "case_id": case.case_id,
                "family": case.family,
                "answer": text.strip(),
                "label": label,
            }
        )
        if progress is not None:
            progress(f"  {phase}gen [{index}/{total}] {case.case_id} -> {label}")
    return rows


def steer_generation_labels(
    model: Any,
    tokenizer: Any,
    cases: Sequence[LureCase],
    *,
    layer: int,
    sae: QwenScopeSAE,
    feature_id: int,
    coefficient: float,
    max_new_tokens: int = 16,
    token_position: str = "all",
    output_mode: BehavioralOutputMode = "binary_choice",
    block_path_template: str = DEFAULT_BLOCK_PATH_TEMPLATE,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Behavioral readout: correct-vs-lure generation with vs without steering.

    Registers :func:`make_feature_steering_hook` (use a negative ``coefficient``
    to suppress the lure feature) on the layer block and re-generates. Returns
    baseline and steered per-case labels plus accuracy/lure-rate summaries and
    their deltas, using the model the feature was found on (the analysis base
    model), so the intervention and the readout live in the same network.
    """

    baseline = _generate_labels(
        model,
        tokenizer,
        cases,
        max_new_tokens=max_new_tokens,
        output_mode=output_mode,
        progress=progress,
        phase="baseline ",
    )
    block = find_decoder_block(model, int(layer), block_path_template=block_path_template)
    hook = make_feature_steering_hook(
        sae, [int(feature_id)], coefficient=float(coefficient), token_position=token_position
    )
    handle = block.register_forward_hook(hook)
    try:
        steered = _generate_labels(
            model,
            tokenizer,
            cases,
            max_new_tokens=max_new_tokens,
            output_mode=output_mode,
            progress=progress,
            phase=f"steer(c={coefficient:g}) ",
        )
    finally:
        handle.remove()

    baseline_summary = summarize_answer_labels([row["label"] for row in baseline])
    steered_summary = summarize_answer_labels([row["label"] for row in steered])
    return {
        "coefficient": float(coefficient),
        "layer": int(layer),
        "feature_id": int(feature_id),
        "output_mode": output_mode,
        "baseline_summary": baseline_summary,
        "steered_summary": steered_summary,
        "accuracy_delta": steered_summary["accuracy"] - baseline_summary["accuracy"],
        "lure_rate_delta": steered_summary["lure_rate"] - baseline_summary["lure_rate"],
        "baseline_rows": baseline,
        "steered_rows": steered,
    }
