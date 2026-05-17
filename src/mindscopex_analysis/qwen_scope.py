"""Qwen-Scope SAE loading, feature extraction, and lightweight steering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from huggingface_hub import hf_hub_download

TokenPosition = Literal["all", "last", "mean"]


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


def get_transformer_layers(model: Any) -> Any:
    """Return the transformer block list for common causal LM architectures."""

    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise ValueError("지원하지 않는 모델 아키텍처: transformer 블록을 찾을 수 없습니다.")


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
    """Download and load one Qwen-Scope SAE checkpoint from Hugging Face.

    Qwen-Scope repositories store one ``layer{n}.sae.pt`` file per layer. The
    file is a dict with ``W_enc``, ``W_dec``, ``b_enc``, and ``b_dec`` tensors.
    """

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

    The implementation follows Qwen-Scope model cards: ``residual @ W_enc.T +
    b_enc`` followed by TopK, without adding a ReLU clamp.
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
    """Summarize SAE feature activations without storing the dense token x feature matrix."""

    if residuals.dim() != 2:
        residuals = residuals.reshape(-1, residuals.shape[-1])
    n_tokens = int(residuals.shape[0])
    device = sae.W_enc.device
    sums = torch.zeros(sae.d_sae, device=device, dtype=torch.float32)
    counts = torch.zeros(sae.d_sae, device=device, dtype=torch.float32)
    maxima = torch.full((sae.d_sae,), float("-inf"), device=device, dtype=torch.float32)

    for start in range(0, n_tokens, batch_size):
        batch = residuals[start : start + batch_size]
        vals, idx = encode_qwen_scope_topk(batch, sae)
        vals_f = vals.float().reshape(-1)
        idx_f = idx.reshape(-1)
        sums.scatter_add_(0, idx_f, vals_f)
        counts.scatter_add_(0, idx_f, torch.ones_like(vals_f))
        maxima.scatter_reduce_(0, idx_f, vals_f, reduce="amax", include_self=True)

    maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
    denom = max(n_tokens, 1)
    return {
        "mean": (sums / denom).cpu(),
        "max": maxima.cpu(),
        "activation_rate": (counts / denom).cpu(),
        "n_tokens": n_tokens,
    }


def capture_residuals(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    layers: list[int],
    *,
    device: str | torch.device,
    max_length: int = 1024,
    token_position: TokenPosition = "last",
) -> dict[int, torch.Tensor]:
    """Capture residual stream tensors from selected layers for a list of formatted texts."""

    blocks = get_transformer_layers(model)
    storage: dict[int, torch.Tensor] = {}
    handles = []

    def hook_for(layer_idx: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            storage[layer_idx] = hidden.detach().cpu()

        return hook

    for layer in layers:
        handles.append(blocks[layer].register_forward_hook(hook_for(layer)))

    collected: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    model.eval()
    try:
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model(**inputs)
                attention_mask = inputs.get("attention_mask")
                for layer in layers:
                    h = storage[layer][0]
                    if token_position == "last":
                        collected[layer].append(h[-1:].cpu())
                    elif token_position == "mean":
                        if attention_mask is None:
                            collected[layer].append(h.mean(dim=0, keepdim=True).cpu())
                        else:
                            mask = attention_mask[0].detach().cpu().float().unsqueeze(-1)
                            collected[layer].append(
                                (h * mask).sum(dim=0, keepdim=True) / mask.sum().clamp_min(1.0)
                            )
                    elif token_position == "all":
                        if attention_mask is None:
                            collected[layer].append(h.cpu())
                        else:
                            mask = attention_mask[0].detach().cpu().bool()
                            collected[layer].append(h[mask].cpu())
                    else:
                        raise ValueError(f"Unknown token_position={token_position!r}")
    finally:
        for handle in handles:
            handle.remove()

    return {layer: torch.cat(parts, dim=0) for layer, parts in collected.items()}


def format_qwen_chat(
    tokenizer: Any,
    prompt: str,
    *,
    system_prompt: str = "",
    enable_thinking: bool | None = None,
) -> str:
    """Format a Qwen chat prompt with optional Qwen3 thinking switch."""

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
    end = text.find("</think>")
    if start == -1 or end == -1 or end < start:
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
    return (sae.W_dec[:, idx] * coeff.unsqueeze(0)).sum(dim=1)


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
