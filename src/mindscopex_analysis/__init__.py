"""Public API for the MindScopeX analysis helpers.

The package exposes a convenient flat API for notebooks, but resolves each
symbol lazily. Importing a lightweight module such as ``cases`` should not also
load Torch, Hugging Face Hub, and every experiment workflow.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

_MODULE_EXPORTS = {
    "activations": (
        "capture_layer_residuals",
        "capture_residual_stream",
        "count_layers",
    ),
    "cases": (
        "BAT_BALL_CASE",
        "PILOT_CRT_DATASET_ID",
        "LureCase",
        "bat_ball_answer_variants",
        "bat_ball_paraphrases",
        "crt_behavior_cases",
        "crt_transfer_cases",
        "load_pilot_crt_cases",
        "semantic_lure_cases",
    ),
    "datasets": (
        "HAGENDORFF_CRT150_DOI",
        "HAGENDORFF_CRT150_OSF_URL",
        "HAGENDORFF_CRT150_SOURCE_SHA256",
        "HAGENDORFF_CRT150_SOURCE_URL",
        "NATURE_CRT150_DOI",
        "NATURE_CRT150_OSF_URL",
        "NATURE_CRT150_SOURCE_SHA256",
        "NATURE_CRT150_SOURCE_URL",
        "HagendorffCRTItem",
        "NatureCRTItem",
        "download_hagendorff_crt150_source",
        "download_nature_crt150_source",
        "load_hagendorff_crt150_items",
        "load_nature_crt150_items",
        "nature_crt150_cases",
        "parse_hagendorff_crt150_source",
        "parse_nature_crt150_source",
    ),
    "effects": (
        "AnswerLogprob",
        "AnswerMargin",
        "EditSite",
        "FeatureAblationResult",
        "active_prompt_features",
        "answer_logprob_margin",
        "multi_site_answer_margin",
        "rank_lure_feature_effects",
        "score_answer_logprob",
        "trace_logits_multi_site",
    ),
    "generation": (
        "PremiseVerdict",
        "QwenTextResponse",
        "classify_lure_answer",
        "classify_premise_rejection",
        "generate_crt_response_suite",
        "generate_qwen_text_response",
        "generate_qwen_text_response_with_retries",
        "qwen_recommended_sampling_kwargs",
        "response_retry_reason",
        "save_crt_markdown_report",
        "save_qwen_text_responses",
        "summarize_crt_accuracy",
        "summarize_crt_accuracy_by_family",
        "text_contains_answer",
    ),
    "lure_datasets": (
        "LureDatasetInfo",
        "available_lure_datasets",
        "load_all_lure_cases",
        "load_lure_dataset",
        "lure_dataset_cases",
        "lure_dataset_catalog",
        "lure_dataset_info",
    ),
    "models": (
        "DEFAULT_ANALYSIS_PROFILE_KEY",
        "DEFAULT_BLOCK_PATH_TEMPLATE",
        "DEFAULT_MODEL_ID",
        "DEFAULT_QWEN_CHAT_MODEL_IDS",
        "DEFAULT_QWEN_SCOPE_REPO_ID",
        "DEFAULT_SCAN_LAYERS",
        "QWEN35_ANALYSIS_PROFILES",
        "QWEN35_BLOCK_PATH_TEMPLATE",
        "QWEN_FORMAT_STRESS_MODEL_IDS",
        "QWEN_LARGE_CHAT_MODEL_IDS",
        "RECOMMENDED_INTERPRETABILITY_MODEL_ID",
        "RECOMMENDED_INTERPRETABILITY_SAE_REPO_ID",
        "Qwen35AnalysisProfile",
        "clear_device_cache",
        "default_sae_device",
        "dtype_from_name",
        "get_qwen35_analysis_profile",
        "load_qwen_language_model",
        "load_qwen_text_generation_model",
        "recommended_dtype_name",
    ),
    "prompts": (
        "CRT_FINAL_ANSWER_SYSTEM_PROMPT",
        "instruct_lure_case",
        "instruct_lure_cases",
        "prepend_final_answer_instruction",
    ),
    "qwen_scope": (
        "LayerFeatureReport",
        "LayerScanResult",
        "QwenScopeSAE",
        "encode_qwen_scope_topk",
        "load_qwen_scope_sae",
        "qwen_scope_feature_preactivations",
        "qwen_scope_feature_values",
        "qwen_scope_sparse_feature_values",
        "sae_decoder_direction",
        "scan_qwen_scope_layers",
        "summarize_qwen_scope_features",
        "top_qwen_scope_features",
    ),
    "modules": (
        "coactivation_edges",
        "module_ablation_direction",
        "module_coherence",
        "module_norm",
        "modules_from_edges",
        "rescale_to_norm",
        "sample_frequency_matched_modules",
        "sparse_activation_matrix",
    ),
    "nulls": (
        "NullPanel",
        "build_null_panel",
        "empirical_percentile",
        "evaluate_feature_null",
        "gaussian_null_directions",
        "null_panel_means",
        "peer_null_directions",
        "selection_adjusted_percentile",
    ),
    "siblings": (
        "difference_in_differences",
        "pearson",
        "rank_siblings",
        "sibling_score",
    ),
    "trajectory": (
        "TokenPhase",
        "cue_span",
        "find_subsequence",
        "quantile_indices",
        "reasoning_phases",
    ),
    "research": (
        "aggregate_feature_effect",
        "control_specificity_rows",
        "discover_generalizing_feature",
        "family_balanced_subset",
        "find_decoder_block",
        "null_summary",
        "random_direction_margin_deltas",
        "random_direction_null_for_feature",
        "split_lure_cases",
        "steer_generation_labels",
        "summarize_answer_labels",
    ),
    "workflows": (
        "FeatureHandle",
        "answer_variant_rows",
        "candidate_feature_rows",
        "case_transfer_rows",
        "coefficient_sweep_for_handle",
        "coefficient_sweep_rows",
        "control_delta_bypass_rows",
        "decoder_cosine_rows",
        "discover_feature_handle",
        "feature_handle_from_result",
        "intervention_mode_rows",
        "layer_feature_search_rows",
        "load_feature_handle",
        "load_or_discover_feature_handle",
        "load_or_discover_handle_and_sae",
        "prompt_token_window_rows",
        "save_feature_handle",
        "token_position_sweep_rows",
    ),
}

_EXPORT_TO_MODULE = {
    name: module_name
    for module_name, exported_names in _MODULE_EXPORTS.items()
    for name in exported_names
}
__all__ = list(_EXPORT_TO_MODULE)


def __getattr__(name: str) -> Any:
    """Load a public symbol from its defining module on first access."""

    try:
        module_name = _EXPORT_TO_MODULE[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
