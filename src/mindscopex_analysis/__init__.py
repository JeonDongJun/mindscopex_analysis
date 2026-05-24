"""Focused Qwen + Qwen-Scope interpretability helpers."""

__version__ = "0.1.0"

from mindscopex_analysis.activations import (
    capture_layer_residuals,
    capture_residual_stream,
    count_layers,
)
from mindscopex_analysis.effects import (
    AnswerLogprob,
    AnswerMargin,
    FeatureAblationResult,
    active_prompt_features,
    answer_logprob_margin,
    rank_lure_feature_effects,
    score_answer_logprob,
)
from mindscopex_analysis.models import (
    DEFAULT_BLOCK_PATH_TEMPLATE,
    DEFAULT_MODEL_ID,
    DEFAULT_QWEN_SCOPE_REPO_ID,
    DEFAULT_SCAN_LAYERS,
    default_sae_device,
    dtype_from_name,
    load_qwen_language_model,
    recommended_dtype_name,
)
from mindscopex_analysis.qwen_scope import (
    LayerFeatureReport,
    LayerScanResult,
    QwenScopeSAE,
    encode_qwen_scope_topk,
    load_qwen_scope_sae,
    scan_qwen_scope_layers,
    summarize_qwen_scope_features,
    top_qwen_scope_features,
)

__all__ = [
    "DEFAULT_BLOCK_PATH_TEMPLATE",
    "DEFAULT_MODEL_ID",
    "DEFAULT_QWEN_SCOPE_REPO_ID",
    "DEFAULT_SCAN_LAYERS",
    "LayerFeatureReport",
    "LayerScanResult",
    "QwenScopeSAE",
    "AnswerLogprob",
    "AnswerMargin",
    "FeatureAblationResult",
    "active_prompt_features",
    "answer_logprob_margin",
    "capture_layer_residuals",
    "capture_residual_stream",
    "count_layers",
    "default_sae_device",
    "dtype_from_name",
    "encode_qwen_scope_topk",
    "load_qwen_language_model",
    "load_qwen_scope_sae",
    "recommended_dtype_name",
    "rank_lure_feature_effects",
    "scan_qwen_scope_layers",
    "score_answer_logprob",
    "summarize_qwen_scope_features",
    "top_qwen_scope_features",
]
