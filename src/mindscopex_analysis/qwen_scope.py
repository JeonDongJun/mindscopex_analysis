"""Qwen-Scope SAE loading and feature-analysis helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from huggingface_hub import hf_hub_download

from mindscopex_analysis.activations import TokenPosition, capture_residual_stream
from mindscopex_analysis.models import (
    DEFAULT_BLOCK_PATH_TEMPLATE,
    DEFAULT_QWEN_SCOPE_REPO_ID,
    DEFAULT_SCAN_LAYERS,
    default_sae_device,
    dtype_from_name,
)

FeatureMetric = Literal["mean", "mean_abs", "max", "activation_rate"]


@dataclass
class QwenScopeSAE:
    """A single Qwen-Scope TopK SAE checkpoint."""

    repo_id: str
    layer: int
    W_enc: torch.Tensor
    W_dec: torch.Tensor
    b_enc: torch.Tensor
    b_dec: torch.Tensor
    top_k: int

    @property
    def d_sae(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.W_enc.shape[1])

    def to(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> QwenScopeSAE:
        kwargs: dict[str, Any] = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return QwenScopeSAE(
            repo_id=self.repo_id,
            layer=self.layer,
            W_enc=self.W_enc.to(**kwargs),
            W_dec=self.W_dec.to(**kwargs),
            b_enc=self.b_enc.to(**kwargs),
            b_dec=self.b_dec.to(**kwargs),
            top_k=self.top_k,
        )


@dataclass(frozen=True)
class FeatureSummary:
    """Compact report for one active Qwen-Scope feature."""

    feature_id: int
    mean: float
    mean_abs: float
    max: float
    activation_rate: float


@dataclass(frozen=True)
class LayerFeatureReport:
    """Layer-level summary used to pick a first layer to inspect."""

    layer: int
    score: float
    n_tokens: int
    top_features: tuple[FeatureSummary, ...]

    def as_row(self) -> dict[str, Any]:
        best = self.top_features[0] if self.top_features else None
        return {
            "layer": self.layer,
            "score": self.score,
            "n_tokens": self.n_tokens,
            "best_feature_id": None if best is None else best.feature_id,
            "best_feature_mean_abs": None if best is None else best.mean_abs,
            "best_feature_rate": None if best is None else best.activation_rate,
        }

    def feature_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "rank": rank,
                "layer": self.layer,
                "feature_id": item.feature_id,
                "mean": item.mean,
                "mean_abs": item.mean_abs,
                "max": item.max,
                "activation_rate": item.activation_rate,
            }
            for rank, item in enumerate(self.top_features, start=1)
        ]


@dataclass
class LayerScanResult:
    """Result returned by ``scan_qwen_scope_layers``."""

    reports: list[LayerFeatureReport]
    residuals: dict[int, torch.Tensor]

    @property
    def best(self) -> LayerFeatureReport:
        if not self.reports:
            raise ValueError("No layer reports are available")
        return self.reports[0]

    def layer_rows(self) -> list[dict[str, Any]]:
        return [report.as_row() for report in self.reports]


def infer_top_k_from_repo(repo_id: str, default: int = 50) -> int:
    """Infer TopK from Qwen-Scope repository names such as ``...-L0_50``."""

    suffix = repo_id.rsplit("L0_", 1)
    if len(suffix) == 2:
        try:
            return int(suffix[-1].split("/", 1)[0])
        except ValueError:
            pass
    return default


def load_qwen_scope_sae(
    repo_id: str,
    layer: int,
    *,
    cache_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    top_k: int | None = None,
) -> QwenScopeSAE:
    """Download and load one Qwen-Scope SAE checkpoint from Hugging Face."""

    filename = f"layer{int(layer)}.sae.pt"
    path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")

    required = {"W_enc", "W_dec", "b_enc", "b_dec"}
    missing = required.difference(state)
    if missing:
        raise KeyError(f"{repo_id}/{filename} 에 필요한 키가 없습니다: {sorted(missing)}")

    sae = QwenScopeSAE(
        repo_id=repo_id,
        layer=int(layer),
        W_enc=state["W_enc"],
        W_dec=state["W_dec"],
        b_enc=state["b_enc"],
        b_dec=state["b_dec"],
        top_k=int(top_k or infer_top_k_from_repo(repo_id)),
    )
    return sae.to(device=device, dtype=dtype)


def encode_qwen_scope_topk(
    residual: torch.Tensor,
    sae: QwenScopeSAE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode residual vectors and return TopK feature values and indices.

    The implementation follows the Qwen-Scope model cards: ``residual @
    W_enc.T + b_enc`` followed by TopK, without adding a ReLU clamp.
    """

    if residual.shape[-1] != sae.d_model:
        raise ValueError(f"residual d_model={residual.shape[-1]} != SAE d_model={sae.d_model}")
    x = residual.to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
    pre_acts = x @ sae.W_enc.T + sae.b_enc
    return pre_acts.topk(sae.top_k, dim=-1)


def summarize_qwen_scope_features(
    residuals: torch.Tensor,
    sae: QwenScopeSAE,
    *,
    batch_size: int = 128,
) -> dict[str, torch.Tensor | int]:
    """Summarize SAE feature activations without storing a dense token x feature matrix."""

    if residuals.dim() != 2:
        residuals = residuals.reshape(-1, residuals.shape[-1])
    n_tokens = int(residuals.shape[0])
    device = sae.W_enc.device
    sums = torch.zeros(sae.d_sae, device=device, dtype=torch.float32)
    abs_sums = torch.zeros(sae.d_sae, device=device, dtype=torch.float32)
    counts = torch.zeros(sae.d_sae, device=device, dtype=torch.float32)
    maxima = torch.full((sae.d_sae,), float("-inf"), device=device, dtype=torch.float32)

    for start in range(0, n_tokens, batch_size):
        batch = residuals[start : start + batch_size]
        vals, idx = encode_qwen_scope_topk(batch, sae)
        vals_f = vals.float().reshape(-1)
        abs_vals_f = vals_f.abs()
        idx_f = idx.reshape(-1)
        sums.scatter_add_(0, idx_f, vals_f)
        abs_sums.scatter_add_(0, idx_f, abs_vals_f)
        counts.scatter_add_(0, idx_f, torch.ones_like(vals_f))
        maxima.scatter_reduce_(0, idx_f, vals_f, reduce="amax", include_self=True)

    maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
    denom = max(n_tokens, 1)
    return {
        "mean": (sums / denom).cpu(),
        "mean_abs": (abs_sums / denom).cpu(),
        "max": maxima.cpu(),
        "activation_rate": (counts / denom).cpu(),
        "n_tokens": n_tokens,
    }


def top_qwen_scope_features(
    summary: dict[str, torch.Tensor | int],
    *,
    top_n: int = 20,
    metric: FeatureMetric = "mean_abs",
) -> tuple[FeatureSummary, ...]:
    """Return top feature summaries sorted by a summary metric."""

    if metric not in {"mean", "mean_abs", "max", "activation_rate"}:
        raise ValueError(f"Unknown metric={metric!r}")
    scores = summary[metric]
    if not isinstance(scores, torch.Tensor):
        raise TypeError(f"summary[{metric!r}] must be a tensor")

    n = min(int(top_n), int(scores.numel()))
    _, indices = scores.topk(n)
    means = summary["mean"]
    mean_abs = summary["mean_abs"]
    maxima = summary["max"]
    rates = summary["activation_rate"]
    assert isinstance(means, torch.Tensor)
    assert isinstance(mean_abs, torch.Tensor)
    assert isinstance(maxima, torch.Tensor)
    assert isinstance(rates, torch.Tensor)

    items = []
    for idx in indices.tolist():
        items.append(
            FeatureSummary(
                feature_id=int(idx),
                mean=float(means[idx]),
                mean_abs=float(mean_abs[idx]),
                max=float(maxima[idx]),
                activation_rate=float(rates[idx]),
            )
        )
    return tuple(items)


def make_layer_feature_report(
    layer: int,
    summary: dict[str, torch.Tensor | int],
    *,
    top_n: int = 20,
    metric: FeatureMetric = "mean_abs",
) -> LayerFeatureReport:
    """Create a layer report with a simple first-pass interpretability score."""

    top_features = top_qwen_scope_features(summary, top_n=top_n, metric=metric)
    if top_features:
        mean_abs = sum(item.mean_abs for item in top_features) / len(top_features)
        rate = sum(item.activation_rate for item in top_features) / len(top_features)
        score = mean_abs * (1.0 + rate)
    else:
        score = 0.0
    return LayerFeatureReport(
        layer=int(layer),
        score=float(score),
        n_tokens=int(summary["n_tokens"]),
        top_features=top_features,
    )


def scan_qwen_scope_layers(
    lm: Any,
    prompts: Sequence[str],
    layers: Sequence[int] = DEFAULT_SCAN_LAYERS,
    *,
    repo_id: str = DEFAULT_QWEN_SCOPE_REPO_ID,
    block_path_template: str = DEFAULT_BLOCK_PATH_TEMPLATE,
    token_position: TokenPosition = "last",
    output_index: int | None = None,
    sae_device: str | torch.device | None = None,
    sae_dtype: str | torch.dtype | None = None,
    cache_dir: str | Path | None = None,
    batch_size: int = 128,
    top_n: int = 20,
    metric: FeatureMetric = "mean_abs",
) -> LayerScanResult:
    """Capture residuals and rank candidate layers by Qwen-Scope feature strength.

    The ranking is only a triage heuristic for choosing a layer to inspect
    first. Semantic interpretation still requires looking at prompts, tokens,
    and downstream effects.
    """

    residuals = capture_residual_stream(
        lm,
        prompts,
        layers,
        block_path_template=block_path_template,
        token_position=token_position,
        output_index=output_index,
    )

    device = sae_device or default_sae_device()
    dtype = dtype_from_name(sae_dtype)
    reports: list[LayerFeatureReport] = []

    for layer in layers:
        layer = int(layer)
        sae = load_qwen_scope_sae(
            repo_id,
            layer,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
        )
        summary = summarize_qwen_scope_features(
            residuals[layer],
            sae,
            batch_size=batch_size,
        )
        reports.append(
            make_layer_feature_report(
                layer,
                summary,
                top_n=top_n,
                metric=metric,
            )
        )
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    reports.sort(key=lambda report: report.score, reverse=True)
    return LayerScanResult(reports=reports, residuals=residuals)


def format_qwen_chat(
    tokenizer: Any,
    prompt: str,
    *,
    system_prompt: str = "",
    enable_thinking: bool | None = None,
) -> str:
    """Format a Qwen chat prompt with an optional Qwen3.5 thinking switch."""

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            text = tokenizer.apply_chat_template(messages, **kwargs)
            if enable_thinking is True:
                return text + "\n/think"
            if enable_thinking is False:
                return text + "\n/no_think"
            return text
    if system_prompt:
        return f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
    return f"User: {prompt}\nAssistant:"


def split_qwen_thinking(text: str) -> tuple[str, str]:
    """Split Qwen-style ``<think>...</think>`` text into thinking and final answer."""

    def clean(part: str) -> str:
        for tok in (
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|end_of_text|>",
        ):
            part = part.replace(tok, "")
        return part.strip()

    start = text.find("<think>")
    end = text.rfind("</think>")
    if start >= 0 and (end == -1 or end < start):
        return clean(text[start + len("<think>") :]), ""
    if start == -1 and end >= 0:
        return clean(text[:end]), clean(text[end + len("</think>") :])
    if start == -1:
        return "", clean(text)
    thinking = clean(text[start + len("<think>") : end])
    answer = clean(text[end + len("</think>") :])
    return thinking, answer


def sae_decoder_direction(
    sae: QwenScopeSAE,
    feature_ids: list[int],
    coefficients: list[float] | None = None,
) -> torch.Tensor:
    """Return a weighted residual-stream direction from SAE decoder columns."""

    if coefficients is None:
        coefficients = [1.0] * len(feature_ids)
    if len(feature_ids) != len(coefficients):
        raise ValueError("feature_ids 와 coefficients 길이가 다릅니다.")
    idx = torch.as_tensor(feature_ids, device=sae.W_dec.device, dtype=torch.long)
    coeff = torch.as_tensor(coefficients, device=sae.W_dec.device, dtype=sae.W_dec.dtype)
    if sae.W_dec.shape[0] == sae.d_model:
        return (sae.W_dec[:, idx] * coeff.unsqueeze(0)).sum(dim=1)
    if sae.W_dec.shape[1] == sae.d_model:
        return (sae.W_dec[idx, :] * coeff.unsqueeze(1)).sum(dim=0)
    raise ValueError(f"Unexpected W_dec shape={tuple(sae.W_dec.shape)} for d_model={sae.d_model}")


def make_feature_steering_hook(
    sae: QwenScopeSAE,
    feature_ids: list[int],
    *,
    coefficient: float = 1.0,
    token_position: Literal["all", "last"] = "last",
):
    """Create a forward hook that adds SAE decoder feature directions to a residual stream."""

    direction = sae_decoder_direction(
        sae,
        feature_ids,
        [float(coefficient)] * len(feature_ids),
    )

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        vec = direction.to(device=hidden.device, dtype=hidden.dtype)
        edited = hidden.clone()
        if token_position == "all":
            edited = edited + vec
        elif token_position == "last":
            edited[:, -1, :] = edited[:, -1, :] + vec
        else:
            raise ValueError(f"Unknown token_position={token_position!r}")
        if isinstance(output, tuple):
            return (edited, *output[1:])
        return edited

    return hook
