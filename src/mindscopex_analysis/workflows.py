"""Reusable experiment workflows for Qwen-Scope lure-feature notebooks."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from mindscopex_analysis.activations import capture_layer_residuals, capture_residual_stream
from mindscopex_analysis.cases import LureCase
from mindscopex_analysis.effects import (
    InterventionMode,
    active_prompt_features,
    answer_logprob_margin,
    rank_lure_feature_effects,
)
from mindscopex_analysis.qwen_scope import QwenScopeSAE, load_qwen_scope_sae, sae_decoder_direction


@dataclass(frozen=True)
class FeatureHandle:
    """Reusable reference to a discovered SAE feature."""

    case_id: str
    layer: int
    feature_id: int
    feature_value: float
    margin_delta: float
    baseline_margin: float
    ablated_margin: float
    intervention_mode: str = "remove_activation"
    coefficient: float = 1.0

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def candidate_feature_rows(
    lm: Any,
    case: LureCase,
    *,
    layer: int,
    sae: QwenScopeSAE,
    top_n: int = 20,
) -> tuple[torch.Tensor, list[dict[str, Any]], list[tuple[int, float]]]:
    residual = capture_layer_residuals(lm, [case.prompt], layer, token_position="last")
    candidates = active_prompt_features(residual, sae, top_n=top_n)
    rows = [
        {
            "case_id": case.case_id,
            "layer": layer,
            "rank": rank,
            "feature_id": feature_id,
            "feature_value": feature_value,
        }
        for rank, (feature_id, feature_value) in enumerate(candidates, start=1)
    ]
    return residual, rows, candidates


def layer_feature_search_rows(
    lm: Any,
    case: LureCase,
    *,
    layers: Sequence[int],
    sae_by_layer: dict[int, QwenScopeSAE],
    top_n: int = 12,
    coefficient: float = 1.0,
    intervention_mode: InterventionMode = "remove_activation",
) -> list[dict[str, Any]]:
    layer_ids = tuple(dict.fromkeys(int(layer) for layer in layers))
    if not layer_ids:
        raise ValueError("layers must not be empty")

    residuals = capture_residual_stream(
        lm,
        [case.prompt],
        layer_ids,
        token_position="last",
    )
    baseline = answer_logprob_margin(
        lm,
        case.prompt,
        correct_answer=case.correct_answer,
        lure_answer=case.lure_answer,
    )
    rows: list[dict[str, Any]] = []
    for layer in layer_ids:
        sae = sae_by_layer[layer]
        residual = residuals[layer]
        candidates = active_prompt_features(residual, sae, top_n=top_n)
        _baseline, results = rank_lure_feature_effects(
            lm,
            case.prompt,
            correct_answer=case.correct_answer,
            lure_answer=case.lure_answer,
            layer=layer,
            sae=sae,
            residual=residual,
            candidate_features=candidates,
            top_n_candidates=top_n,
            coefficient=coefficient,
            intervention_mode=intervention_mode,
            baseline=baseline,
        )
        for rank, result in enumerate(results, start=1):
            row = result.as_row()
            row.update({"case_id": case.case_id, "effect_rank": rank})
            rows.append(row)
    rows.sort(key=lambda row: row["margin_delta"], reverse=True)
    return rows


def feature_handle_from_result(case: LureCase, result: Any) -> FeatureHandle:
    return FeatureHandle(
        case_id=case.case_id,
        layer=int(result.layer),
        feature_id=int(result.feature_id),
        feature_value=float(result.feature_value),
        margin_delta=float(result.margin_delta),
        baseline_margin=float(result.baseline_margin),
        ablated_margin=float(result.ablated_margin),
        intervention_mode=str(result.intervention_mode),
        coefficient=float(result.coefficient),
    )


def save_feature_handle(handle: FeatureHandle, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(handle.as_row(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_feature_handle(path: str | Path) -> FeatureHandle:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return FeatureHandle(**data)


def discover_feature_handle(
    lm: Any,
    case: LureCase,
    *,
    layer: int,
    sae: QwenScopeSAE,
    top_n: int = 12,
    coefficient: float = 1.0,
    intervention_mode: InterventionMode = "remove_activation",
) -> tuple[FeatureHandle, list[dict[str, Any]]]:
    if top_n < 1:
        raise ValueError("top_n must be positive")
    residual, _candidate_rows, candidates = candidate_feature_rows(
        lm,
        case,
        layer=layer,
        sae=sae,
        top_n=top_n,
    )
    _baseline, ranked = rank_lure_feature_effects(
        lm,
        case.prompt,
        correct_answer=case.correct_answer,
        lure_answer=case.lure_answer,
        layer=layer,
        sae=sae,
        residual=residual,
        candidate_features=candidates,
        top_n_candidates=top_n,
        coefficient=coefficient,
        intervention_mode=intervention_mode,
    )
    if not ranked:
        raise ValueError("Feature discovery produced no candidates")
    rows = [result.as_row() for result in ranked]
    return feature_handle_from_result(case, ranked[0]), rows


def load_or_discover_feature_handle(
    lm: Any,
    case: LureCase,
    *,
    layer: int,
    sae: QwenScopeSAE,
    cache_path: str | Path,
    top_n: int = 12,
    coefficient: float = 1.0,
    intervention_mode: InterventionMode = "remove_activation",
    refresh: bool = False,
) -> tuple[FeatureHandle, list[dict[str, Any]], bool]:
    path = Path(cache_path)
    if path.is_file() and not refresh:
        return load_feature_handle(path), [], True

    handle, rows = discover_feature_handle(
        lm,
        case,
        layer=layer,
        sae=sae,
        top_n=top_n,
        coefficient=coefficient,
        intervention_mode=intervention_mode,
    )
    save_feature_handle(handle, path)
    return handle, rows, False


def load_or_discover_handle_and_sae(
    lm: Any,
    case: LureCase,
    *,
    repo_id: str,
    cache_path: str | Path,
    default_layer: int = 14,
    sae_device: str | torch.device = "cpu",
    sae_dtype: torch.dtype | None = None,
    top_n: int = 12,
    coefficient: float = 1.0,
    intervention_mode: InterventionMode = "remove_activation",
    refresh: bool = False,
) -> tuple[FeatureHandle, QwenScopeSAE, list[dict[str, Any]], bool]:
    path = Path(cache_path)
    if path.is_file() and not refresh:
        handle = load_feature_handle(path)
        sae = load_qwen_scope_sae(
            repo_id,
            handle.layer,
            device=sae_device,
            dtype=sae_dtype,
        )
        return handle, sae, [], True

    sae = load_qwen_scope_sae(
        repo_id,
        default_layer,
        device=sae_device,
        dtype=sae_dtype,
    )
    handle, rows = discover_feature_handle(
        lm,
        case,
        layer=default_layer,
        sae=sae,
        top_n=top_n,
        coefficient=coefficient,
        intervention_mode=intervention_mode,
    )
    save_feature_handle(handle, path)
    return handle, sae, rows, False


def coefficient_sweep_rows(
    lm: Any,
    case: LureCase,
    *,
    layer: int,
    sae: QwenScopeSAE,
    feature_id: int,
    feature_value: float,
    coefficients: Iterable[float],
    intervention_mode: InterventionMode = "remove_activation",
) -> list[dict[str, Any]]:
    direction = sae_decoder_direction(sae, [int(feature_id)])
    baseline = answer_logprob_margin(
        lm,
        case.prompt,
        correct_answer=case.correct_answer,
        lure_answer=case.lure_answer,
    )
    return _coefficient_sweep_rows(
        lm,
        case,
        layer=layer,
        feature_id=feature_id,
        feature_value=feature_value,
        coefficients=coefficients,
        intervention_mode=intervention_mode,
        direction=direction,
        baseline=baseline,
    )


def _coefficient_sweep_rows(
    lm: Any,
    case: LureCase,
    *,
    layer: int,
    feature_id: int,
    feature_value: float,
    coefficients: Iterable[float],
    intervention_mode: InterventionMode,
    direction: torch.Tensor,
    baseline: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for coefficient in coefficients:
        margin = answer_logprob_margin(
            lm,
            case.prompt,
            correct_answer=case.correct_answer,
            lure_answer=case.lure_answer,
            layer=layer,
            direction=direction,
            feature_value=float(feature_value),
            coefficient=float(coefficient),
            intervention_mode=intervention_mode,
        )
        rows.append(
            {
                "case_id": case.case_id,
                "layer": layer,
                "feature_id": feature_id,
                "feature_value": feature_value,
                "coefficient": float(coefficient),
                "intervention_mode": intervention_mode,
                "baseline_margin": baseline.margin,
                "margin": margin.margin,
                "margin_delta": baseline.margin - margin.margin,
                "correct_logprob_delta": margin.correct.logprob - baseline.correct.logprob,
                "lure_logprob_delta": margin.lure.logprob - baseline.lure.logprob,
            }
        )
    return rows


def coefficient_sweep_for_handle(
    lm: Any,
    case: LureCase,
    *,
    sae: QwenScopeSAE,
    handle: FeatureHandle,
    coefficients: Iterable[float],
    intervention_mode: InterventionMode = "remove_activation",
) -> list[dict[str, Any]]:
    return coefficient_sweep_rows(
        lm,
        case,
        layer=handle.layer,
        sae=sae,
        feature_id=handle.feature_id,
        feature_value=handle.feature_value,
        coefficients=coefficients,
        intervention_mode=intervention_mode,
    )


def intervention_mode_rows(
    lm: Any,
    case: LureCase,
    *,
    layer: int,
    sae: QwenScopeSAE,
    feature_id: int,
    feature_value: float,
    modes: Sequence[InterventionMode],
    coefficient: float = 1.0,
) -> list[dict[str, Any]]:
    direction = sae_decoder_direction(sae, [int(feature_id)])
    baseline = answer_logprob_margin(
        lm,
        case.prompt,
        correct_answer=case.correct_answer,
        lure_answer=case.lure_answer,
    )
    rows: list[dict[str, Any]] = []
    for mode in modes:
        rows.extend(
            _coefficient_sweep_rows(
                lm,
                case,
                layer=layer,
                feature_id=feature_id,
                feature_value=feature_value,
                coefficients=[coefficient],
                intervention_mode=mode,
                direction=direction,
                baseline=baseline,
            )
        )
    rows.sort(key=lambda row: row["margin_delta"], reverse=True)
    return rows


def case_transfer_rows(
    lm: Any,
    cases: Sequence[LureCase],
    *,
    layer: int,
    sae: QwenScopeSAE,
    feature_id: int,
    feature_value: float,
    coefficient: float = 1.0,
    intervention_mode: InterventionMode = "remove_activation",
) -> list[dict[str, Any]]:
    direction = sae_decoder_direction(sae, [int(feature_id)])
    rows: list[dict[str, Any]] = []
    for case in cases:
        baseline = answer_logprob_margin(
            lm,
            case.prompt,
            correct_answer=case.correct_answer,
            lure_answer=case.lure_answer,
        )
        edited = answer_logprob_margin(
            lm,
            case.prompt,
            correct_answer=case.correct_answer,
            lure_answer=case.lure_answer,
            layer=layer,
            direction=direction,
            feature_value=float(feature_value),
            coefficient=float(coefficient),
            intervention_mode=intervention_mode,
        )
        rows.append(
            {
                "case_id": case.case_id,
                "family": case.family,
                "layer": layer,
                "feature_id": feature_id,
                "intervention_mode": intervention_mode,
                "baseline_margin": baseline.margin,
                "edited_margin": edited.margin,
                "margin_delta": baseline.margin - edited.margin,
                "note": case.note,
            }
        )
    return rows


def answer_variant_rows(
    lm: Any,
    case: LureCase,
    *,
    answer_pairs: Sequence[tuple[str, str, str]],
    layer: int | None = None,
    direction: torch.Tensor | None = None,
    feature_value: float = 0.0,
    coefficient: float = 1.0,
    intervention_mode: InterventionMode = "remove_activation",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, correct, lure in answer_pairs:
        baseline = answer_logprob_margin(
            lm,
            case.prompt,
            correct_answer=correct,
            lure_answer=lure,
        )
        row = {
            "variant": label,
            "correct_answer": correct,
            "lure_answer": lure,
            "baseline_margin": baseline.margin,
            "edited_margin": None,
            "margin_delta": None,
        }
        if layer is not None and direction is not None:
            edited = answer_logprob_margin(
                lm,
                case.prompt,
                correct_answer=correct,
                lure_answer=lure,
                layer=layer,
                direction=direction,
                feature_value=feature_value,
                coefficient=coefficient,
                intervention_mode=intervention_mode,
            )
            row.update(
                {
                    "edited_margin": edited.margin,
                    "margin_delta": baseline.margin - edited.margin,
                }
            )
        rows.append(row)
    return rows


def prompt_token_window_rows(
    tokenizer: Any,
    prompt: str,
    *,
    window: int = 8,
) -> list[dict[str, Any]]:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
    answer_start = int(prompt_ids.numel())
    start = max(0, int(prompt_ids.numel()) - int(window))
    rows = []
    for idx in range(start, int(prompt_ids.numel())):
        token_id = int(prompt_ids[idx])
        rows.append(
            {
                "token_index": idx,
                "relative_to_prompt_end": idx - (answer_start - 1),
                "token": tokenizer.decode([token_id]),
            }
        )
    return rows


def token_position_sweep_rows(
    lm: Any,
    case: LureCase,
    *,
    layer: int,
    sae: QwenScopeSAE,
    feature_id: int,
    feature_value: float,
    token_indices: Sequence[int],
    coefficient: float = 1.0,
    intervention_mode: InterventionMode = "remove_activation",
) -> list[dict[str, Any]]:
    direction = sae_decoder_direction(sae, [int(feature_id)])
    baseline = answer_logprob_margin(
        lm,
        case.prompt,
        correct_answer=case.correct_answer,
        lure_answer=case.lure_answer,
    )
    rows: list[dict[str, Any]] = []
    for token_index in token_indices:
        edited = answer_logprob_margin(
            lm,
            case.prompt,
            correct_answer=case.correct_answer,
            lure_answer=case.lure_answer,
            layer=layer,
            direction=direction,
            feature_value=float(feature_value),
            coefficient=float(coefficient),
            intervention_mode=intervention_mode,
            token_index=int(token_index),
        )
        rows.append(
            {
                "case_id": case.case_id,
                "token_index": int(token_index),
                "baseline_margin": baseline.margin,
                "edited_margin": edited.margin,
                "margin_delta": baseline.margin - edited.margin,
            }
        )
    rows.sort(key=lambda row: row["margin_delta"], reverse=True)
    return rows


def control_delta_bypass_rows(
    lm: Any,
    case: LureCase,
    *,
    layer: int,
    coefficients: Iterable[float],
) -> list[dict[str, Any]]:
    if not case.control_prompt:
        raise ValueError(f"case {case.case_id!r} has no control_prompt")
    lure_residual = capture_layer_residuals(lm, [case.prompt], layer, token_position="last")[0]
    control_residual = capture_layer_residuals(
        lm, [case.control_prompt], layer, token_position="last"
    )[0]
    delta = control_residual - lure_residual
    baseline = answer_logprob_margin(
        lm,
        case.prompt,
        correct_answer=case.correct_answer,
        lure_answer=case.lure_answer,
    )

    rows: list[dict[str, Any]] = []
    for coefficient in coefficients:
        edited = answer_logprob_margin(
            lm,
            case.prompt,
            correct_answer=case.correct_answer,
            lure_answer=case.lure_answer,
            layer=layer,
            direction=delta,
            feature_value=1.0,
            coefficient=float(coefficient),
            intervention_mode="add_vector",
        )
        rows.append(
            {
                "case_id": case.case_id,
                "layer": layer,
                "coefficient": float(coefficient),
                "baseline_margin": baseline.margin,
                "edited_margin": edited.margin,
                "margin_delta": baseline.margin - edited.margin,
            }
        )
    return rows


def decoder_cosine_rows(
    sae: QwenScopeSAE,
    feature_ids: Sequence[int],
) -> list[dict[str, Any]]:
    ids = list(dict.fromkeys(int(feature_id) for feature_id in feature_ids))
    if not ids:
        return []
    directions = torch.stack(
        [sae_decoder_direction(sae, [feature_id]).detach().float().cpu() for feature_id in ids]
    )
    directions = torch.nn.functional.normalize(directions, dim=1, eps=1e-12)
    cosine = directions @ directions.T

    rows: list[dict[str, Any]] = []
    for left_index, left_id in enumerate(ids):
        for right_index, right_id in enumerate(ids):
            rows.append(
                {
                    "feature_i": left_id,
                    "feature_j": right_id,
                    "decoder_cosine": float(cosine[left_index, right_index]),
                }
            )
    return rows
