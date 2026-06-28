"""Model loading helpers for Qwen interpretability experiments."""

from __future__ import annotations

from typing import Any

import torch

DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B-Base"
DEFAULT_QWEN_SCOPE_REPO_ID = "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50"
DEFAULT_BLOCK_PATH_TEMPLATE = "model.layers.{layer}"
DEFAULT_SCAN_LAYERS = (6, 14, 21, 27)
DEFAULT_QWEN_CHAT_MODEL_IDS = (
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
)


def dtype_from_name(dtype: str | torch.dtype | None) -> torch.dtype | None:
    """Resolve a user-facing dtype string into a torch dtype."""

    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype

    normalized = dtype.lower().replace("torch.", "")
    if normalized in {"auto", "none", ""}:
        return None

    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype!r}") from exc


def recommended_dtype_name() -> str:
    """Return a conservative default dtype for the current machine."""

    if not torch.cuda.is_available():
        return "float32"
    if torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float16"


def default_sae_device() -> str:
    """Use GPU for SAE matrix multiplies when available."""

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_qwen_language_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    device_map: str | dict[str, Any] = "auto",
    dtype: str | torch.dtype | None = None,
    trust_remote_code: bool = True,
    dispatch: bool = True,
    **kwargs: Any,
) -> Any:
    """Load a Qwen causal LM through NNsight's ``LanguageModel`` wrapper.

    NNsight's public examples currently use ``dispatch=True`` so weights are
    loaded immediately. The fallback attempts keep this helper usable across a
    small range of NNsight/Transformers signatures.
    """

    try:
        from nnsight import LanguageModel
    except ImportError as exc:
        raise ImportError(
            "nnsight is required. Install with `uv sync --extra dev` or `pip install -e .[dev]`."
        ) from exc

    resolved_dtype = dtype_from_name(dtype)
    base_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        **kwargs,
    }
    if resolved_dtype is not None:
        base_kwargs["torch_dtype"] = resolved_dtype

    attempts: list[dict[str, Any]] = []
    with_dispatch = dict(base_kwargs)
    with_dispatch["dispatch"] = dispatch
    attempts.append(with_dispatch)
    attempts.append(dict(base_kwargs))

    if "torch_dtype" in base_kwargs:
        dtype_kwargs = dict(base_kwargs)
        dtype_kwargs["dtype"] = dtype_kwargs.pop("torch_dtype")
        dtype_with_dispatch = dict(dtype_kwargs)
        dtype_with_dispatch["dispatch"] = dispatch
        attempts.extend([dtype_with_dispatch, dtype_kwargs])

    no_dtype = {k: v for k, v in base_kwargs.items() if k not in {"torch_dtype", "dtype"}}
    no_dtype_with_dispatch = dict(no_dtype)
    no_dtype_with_dispatch["dispatch"] = dispatch
    attempts.extend([no_dtype_with_dispatch, no_dtype])

    errors: list[str] = []
    for attempt in attempts:
        try:
            return LanguageModel(model_id, **attempt)
        except TypeError as exc:
            errors.append(f"{attempt}: {exc}")

    joined = "\n".join(errors[-3:])
    raise TypeError(f"Could not load {model_id!r} with NNsight. Last errors:\n{joined}")


def load_qwen_text_generation_model(
    model_id: str,
    *,
    device_map: str | dict[str, Any] = "auto",
    dtype: str | torch.dtype | None = None,
    trust_remote_code: bool = True,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Load a Qwen tokenizer and causal LM for ordinary text generation."""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for Qwen text generation.") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    model_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        **kwargs,
    }
    resolved_dtype = dtype_from_name(dtype)
    if resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    return model, tokenizer


def clear_device_cache() -> None:
    """Release unused Python and CUDA objects between sequential model runs."""

    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
