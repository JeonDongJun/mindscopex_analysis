"""Logprob scoring and Qwen-Scope feature ablation experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from mindscopex_analysis.activations import get_module
from mindscopex_analysis.models import DEFAULT_BLOCK_PATH_TEMPLATE
from mindscopex_analysis.qwen_scope import (
    QwenScopeSAE,
    encode_qwen_scope_topk,
    sae_decoder_direction,
)

InterventionMode = Literal[
    "remove_activation",
    "add_activation",
    "subtract_unit",
    "add_unit",
    "projection_remove",
    "add_vector",
]


@dataclass(frozen=True)
class AnswerLogprob:
    """Teacher-forced continuation logprob for one answer string."""

    answer: str
    logprob: float
    mean_logprob: float
    token_logprobs: tuple[float, ...]
    token_ids: tuple[int, ...]
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class AnswerMargin:
    """Lure-vs-correct logprob margin."""

    correct: AnswerLogprob
    lure: AnswerLogprob

    @property
    def margin(self) -> float:
        """Positive means the lure answer is preferred over the correct answer."""

        return self.lure.logprob - self.correct.logprob

    @property
    def mean_margin(self) -> float:
        """Length-normalized version of ``margin``."""

        return self.lure.mean_logprob - self.correct.mean_logprob

    def as_row(self) -> dict[str, Any]:
        return {
            "correct_answer": self.correct.answer,
            "lure_answer": self.lure.answer,
            "correct_logprob": self.correct.logprob,
            "lure_logprob": self.lure.logprob,
            "margin_lure_minus_correct": self.margin,
            "correct_mean_logprob": self.correct.mean_logprob,
            "lure_mean_logprob": self.lure.mean_logprob,
            "mean_margin_lure_minus_correct": self.mean_margin,
        }


@dataclass(frozen=True)
class FeatureAblationResult:
    """Effect of removing one SAE feature direction from a residual stream."""

    layer: int
    feature_id: int
    feature_value: float
    baseline_margin: float
    ablated_margin: float
    margin_delta: float
    baseline_mean_margin: float
    ablated_mean_margin: float
    mean_margin_delta: float
    correct_logprob_delta: float
    lure_logprob_delta: float
    intervention_mode: str = "remove_activation"
    coefficient: float = 1.0

    def as_row(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "feature_id": self.feature_id,
            "feature_value": self.feature_value,
            "baseline_margin": self.baseline_margin,
            "ablated_margin": self.ablated_margin,
            "margin_delta": self.margin_delta,
            "baseline_mean_margin": self.baseline_mean_margin,
            "ablated_mean_margin": self.ablated_mean_margin,
            "mean_margin_delta": self.mean_margin_delta,
            "correct_logprob_delta": self.correct_logprob_delta,
            "lure_logprob_delta": self.lure_logprob_delta,
            "intervention_mode": self.intervention_mode,
            "coefficient": self.coefficient,
        }


def _input_ids(tokenizer: Any, text: str) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    return encoded["input_ids"][0].detach().cpu()


def continuation_token_span(
    tokenizer: Any,
    prompt: str,
    answer: str,
) -> tuple[torch.Tensor, int]:
    """Tokenize ``prompt + answer`` and return full ids plus answer start index."""

    prompt_ids = _input_ids(tokenizer, prompt)
    full_ids = _input_ids(tokenizer, prompt + answer)
    start = int(prompt_ids.numel())

    if full_ids.numel() <= start:
        raise ValueError("answer produced no continuation tokens")
    if not torch.equal(full_ids[:start], prompt_ids):
        raise ValueError(
            "Prompt tokens are not a prefix of prompt+answer tokens. "
            "Use a delimiter such as '\\nAnswer:' and include a leading space in answer strings."
        )
    return full_ids, start


def continuation_logprob_from_logits(
    logits: torch.Tensor,
    full_input_ids: torch.Tensor,
    target_start: int,
    *,
    tokenizer: Any | None = None,
    answer: str = "",
) -> AnswerLogprob:
    """Compute continuation logprob from logits over ``prompt + answer``."""

    if logits.dim() == 3:
        logits = logits[0]
    if target_start <= 0:
        raise ValueError("target_start must be greater than 0")

    ids = full_input_ids.detach().cpu().long()
    if logits.shape[0] < ids.numel():
        raise ValueError(f"logits seq_len={logits.shape[0]} < input ids={ids.numel()}")

    log_probs = torch.log_softmax(logits.detach().cpu().float(), dim=-1)
    token_logprobs: list[float] = []
    target_ids: list[int] = []
    tokens: list[str] = []

    for pos in range(target_start, int(ids.numel())):
        token_id = int(ids[pos])
        token_logprob = float(log_probs[pos - 1, token_id])
        token_logprobs.append(token_logprob)
        target_ids.append(token_id)
        if tokenizer is not None:
            tokens.append(tokenizer.decode([token_id]))

    total = float(sum(token_logprobs))
    mean = total / max(len(token_logprobs), 1)
    return AnswerLogprob(
        answer=answer,
        logprob=total,
        mean_logprob=mean,
        token_logprobs=tuple(token_logprobs),
        token_ids=tuple(target_ids),
        tokens=tuple(tokens),
    )


def _direction_edit(
    hidden_vector: Any,
    direction: torch.Tensor,
    *,
    feature_value: float,
    coefficient: float,
    intervention_mode: InterventionMode,
) -> Any:
    direction = direction.to(device=hidden_vector.device, dtype=hidden_vector.dtype)
    unit = direction / torch.linalg.norm(direction.float()).clamp_min(1e-12).to(direction.dtype)
    if intervention_mode == "remove_activation":
        return hidden_vector - float(coefficient) * float(feature_value) * direction
    if intervention_mode == "add_activation":
        return hidden_vector + float(coefficient) * float(feature_value) * direction
    if intervention_mode == "subtract_unit":
        return hidden_vector - float(coefficient) * unit
    if intervention_mode == "add_unit":
        return hidden_vector + float(coefficient) * unit
    if intervention_mode == "projection_remove":
        projection = (hidden_vector @ unit).unsqueeze(-1) * unit
        return hidden_vector - float(coefficient) * projection
    if intervention_mode == "add_vector":
        return hidden_vector + float(coefficient) * direction
    raise ValueError(f"Unknown intervention_mode={intervention_mode!r}")


def _trace_logits_with_optional_intervention(
    lm: Any,
    text: str,
    *,
    layer: int | None = None,
    direction: torch.Tensor | None = None,
    feature_value: float = 0.0,
    coefficient: float = 1.0,
    intervention_mode: InterventionMode = "remove_activation",
    token_index: int = -1,
    block_path_template: str = DEFAULT_BLOCK_PATH_TEMPLATE,
    output_index: int | None = 0,
) -> torch.Tensor:
    with lm.trace(text):
        if layer is not None and direction is not None:
            block = get_module(lm, block_path_template.format(layer=int(layer), i=int(layer)))
            hidden = block.output if output_index is None else block.output[output_index]
            hidden[:, token_index, :] = _direction_edit(
                hidden[:, token_index, :],
                direction,
                feature_value=feature_value,
                coefficient=coefficient,
                intervention_mode=intervention_mode,
            )
        logits = lm.output.logits.save()

    return getattr(logits, "value", logits).detach().cpu()


def score_answer_logprob(
    lm: Any,
    prompt: str,
    answer: str,
    *,
    layer: int | None = None,
    direction: torch.Tensor | None = None,
    feature_value: float = 0.0,
    coefficient: float = 1.0,
    intervention_mode: InterventionMode = "remove_activation",
    token_index: int | None = None,
    block_path_template: str = DEFAULT_BLOCK_PATH_TEMPLATE,
    output_index: int | None = 0,
) -> AnswerLogprob:
    """Score one answer, optionally removing one feature direction during the trace.

    If ``token_index`` is ``None``, ablation is applied at the final prompt
    token, not the final token of ``prompt + answer``.
    """

    tokenizer = lm.tokenizer
    full_ids, start = continuation_token_span(tokenizer, prompt, answer)
    edit_token_index = start - 1 if token_index is None else token_index
    logits = _trace_logits_with_optional_intervention(
        lm,
        prompt + answer,
        layer=layer,
        direction=direction,
        feature_value=feature_value,
        coefficient=coefficient,
        intervention_mode=intervention_mode,
        token_index=edit_token_index,
        block_path_template=block_path_template,
        output_index=output_index,
    )
    return continuation_logprob_from_logits(
        logits,
        full_ids,
        start,
        tokenizer=tokenizer,
        answer=answer,
    )


def answer_logprob_margin(
    lm: Any,
    prompt: str,
    *,
    correct_answer: str,
    lure_answer: str,
    layer: int | None = None,
    direction: torch.Tensor | None = None,
    feature_value: float = 0.0,
    coefficient: float = 1.0,
    intervention_mode: InterventionMode = "remove_activation",
    token_index: int | None = None,
    block_path_template: str = DEFAULT_BLOCK_PATH_TEMPLATE,
    output_index: int | None = 0,
) -> AnswerMargin:
    """Return logprob margin ``lure - correct`` for two candidate answers."""

    correct = score_answer_logprob(
        lm,
        prompt,
        correct_answer,
        layer=layer,
        direction=direction,
        feature_value=feature_value,
        coefficient=coefficient,
        intervention_mode=intervention_mode,
        token_index=token_index,
        block_path_template=block_path_template,
        output_index=output_index,
    )
    lure = score_answer_logprob(
        lm,
        prompt,
        lure_answer,
        layer=layer,
        direction=direction,
        feature_value=feature_value,
        coefficient=coefficient,
        intervention_mode=intervention_mode,
        token_index=token_index,
        block_path_template=block_path_template,
        output_index=output_index,
    )
    return AnswerMargin(correct=correct, lure=lure)


def active_prompt_features(
    residual: torch.Tensor,
    sae: QwenScopeSAE,
    *,
    top_n: int = 20,
) -> list[tuple[int, float]]:
    """Return active feature ids and values for one residual vector."""

    if residual.dim() == 1:
        residual = residual.unsqueeze(0)
    if residual.shape[0] != 1:
        raise ValueError("active_prompt_features expects one residual vector")
    vals, idx = encode_qwen_scope_topk(residual, sae)
    return [
        (int(feature_id), float(value))
        for feature_id, value in zip(
            idx[0, :top_n].detach().cpu().tolist(),
            vals[0, :top_n].detach().cpu().float().tolist(),
            strict=True,
        )
    ]


def rank_lure_feature_effects(
    lm: Any,
    prompt: str,
    *,
    correct_answer: str,
    lure_answer: str,
    layer: int,
    sae: QwenScopeSAE,
    residual: torch.Tensor,
    candidate_features: Sequence[tuple[int, float]] | None = None,
    top_n_candidates: int = 20,
    coefficient: float = 1.0,
    intervention_mode: InterventionMode = "remove_activation",
    token_index: int | None = None,
    block_path_template: str = DEFAULT_BLOCK_PATH_TEMPLATE,
    output_index: int | None = 0,
) -> tuple[AnswerMargin, list[FeatureAblationResult]]:
    """Rank active SAE features by how much ablation reduces lure preference.

    ``margin_delta`` is ``baseline_margin - ablated_margin``. A positive value
    means removing the feature made the lure answer less preferred relative to
    the correct answer.
    """

    baseline = answer_logprob_margin(
        lm,
        prompt,
        correct_answer=correct_answer,
        lure_answer=lure_answer,
        token_index=token_index,
        block_path_template=block_path_template,
        output_index=output_index,
    )

    if candidate_features is None:
        candidate_features = active_prompt_features(
            residual,
            sae,
            top_n=top_n_candidates,
        )

    results: list[FeatureAblationResult] = []
    for feature_id, feature_value in list(candidate_features)[:top_n_candidates]:
        direction = sae_decoder_direction(sae, [int(feature_id)]).to(
            device=sae.W_dec.device,
            dtype=sae.W_dec.dtype,
        )
        ablated = answer_logprob_margin(
            lm,
            prompt,
            correct_answer=correct_answer,
            lure_answer=lure_answer,
            layer=layer,
            direction=direction,
            feature_value=float(feature_value),
            coefficient=coefficient,
            intervention_mode=intervention_mode,
            token_index=token_index,
            block_path_template=block_path_template,
            output_index=output_index,
        )
        results.append(
            FeatureAblationResult(
                layer=int(layer),
                feature_id=int(feature_id),
                feature_value=float(feature_value),
                baseline_margin=baseline.margin,
                ablated_margin=ablated.margin,
                margin_delta=baseline.margin - ablated.margin,
                baseline_mean_margin=baseline.mean_margin,
                ablated_mean_margin=ablated.mean_margin,
                mean_margin_delta=baseline.mean_margin - ablated.mean_margin,
                correct_logprob_delta=ablated.correct.logprob - baseline.correct.logprob,
                lure_logprob_delta=ablated.lure.logprob - baseline.lure.logprob,
                intervention_mode=intervention_mode,
                coefficient=float(coefficient),
            )
        )

    results.sort(key=lambda item: item.margin_delta, reverse=True)
    return baseline, results
