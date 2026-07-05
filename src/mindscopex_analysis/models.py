"""Model loading helpers for Qwen interpretability experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

QWEN35_BLOCK_PATH_TEMPLATE = "model.language_model.layers.{layer}"


@dataclass(frozen=True)
class Qwen35AnalysisProfile:
    """One Qwen3.5 behavior checkpoint and its official SAE-backed analysis pair."""

    key: str
    behavior_model_id: str
    analysis_model_id: str
    sae_repo_id: str
    num_layers: int
    hidden_size: int
    scan_layers: tuple[int, ...]
    architecture: str
    sae_matches_behavior_model: bool
    block_path_template: str = QWEN35_BLOCK_PATH_TEMPLATE


QWEN35_ANALYSIS_PROFILES = {
    "2b": Qwen35AnalysisProfile(
        key="2b",
        behavior_model_id="Qwen/Qwen3.5-2B",
        analysis_model_id="Qwen/Qwen3.5-2B-Base",
        sae_repo_id="Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50",
        num_layers=24,
        hidden_size=2048,
        scan_layers=(5, 11, 17, 23),
        architecture="dense",
        sae_matches_behavior_model=False,
    ),
    "9b": Qwen35AnalysisProfile(
        key="9b",
        behavior_model_id="Qwen/Qwen3.5-9B",
        analysis_model_id="Qwen/Qwen3.5-9B-Base",
        sae_repo_id="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50",
        num_layers=32,
        hidden_size=4096,
        scan_layers=(7, 15, 23, 31),
        architecture="dense",
        sae_matches_behavior_model=False,
    ),
    "27b": Qwen35AnalysisProfile(
        key="27b",
        behavior_model_id="Qwen/Qwen3.5-27B",
        analysis_model_id="Qwen/Qwen3.5-27B",
        sae_repo_id="Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50",
        num_layers=64,
        hidden_size=5120,
        scan_layers=(15, 31, 47, 63),
        architecture="dense",
        sae_matches_behavior_model=True,
    ),
    "35b-a3b": Qwen35AnalysisProfile(
        key="35b-a3b",
        behavior_model_id="Qwen/Qwen3.5-35B-A3B",
        analysis_model_id="Qwen/Qwen3.5-35B-A3B-Base",
        sae_repo_id="Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50",
        num_layers=40,
        hidden_size=2048,
        scan_layers=(9, 19, 29, 39),
        architecture="moe",
        sae_matches_behavior_model=False,
    ),
}

DEFAULT_ANALYSIS_PROFILE_KEY = "27b"
_DEFAULT_ANALYSIS_PROFILE = QWEN35_ANALYSIS_PROFILES[DEFAULT_ANALYSIS_PROFILE_KEY]

DEFAULT_MODEL_ID = _DEFAULT_ANALYSIS_PROFILE.analysis_model_id
DEFAULT_QWEN_SCOPE_REPO_ID = _DEFAULT_ANALYSIS_PROFILE.sae_repo_id
DEFAULT_BLOCK_PATH_TEMPLATE = _DEFAULT_ANALYSIS_PROFILE.block_path_template
DEFAULT_SCAN_LAYERS = _DEFAULT_ANALYSIS_PROFILE.scan_layers
DEFAULT_QWEN_CHAT_MODEL_IDS = tuple(
    profile.behavior_model_id for profile in QWEN35_ANALYSIS_PROFILES.values()
)
QWEN_LARGE_CHAT_MODEL_IDS = (
    QWEN35_ANALYSIS_PROFILES["27b"].behavior_model_id,
    QWEN35_ANALYSIS_PROFILES["35b-a3b"].behavior_model_id,
)
QWEN_FORMAT_STRESS_MODEL_IDS = ("Qwen/Qwen3.5-0.8B",)
RECOMMENDED_INTERPRETABILITY_MODEL_ID = _DEFAULT_ANALYSIS_PROFILE.analysis_model_id
RECOMMENDED_INTERPRETABILITY_SAE_REPO_ID = _DEFAULT_ANALYSIS_PROFILE.sae_repo_id


def get_qwen35_analysis_profile(key: str = DEFAULT_ANALYSIS_PROFILE_KEY) -> Qwen35AnalysisProfile:
    """Resolve a short model key to a validated Qwen3.5/SAE experiment profile."""

    normalized = key.strip().lower().replace("_", "-")
    normalized = {"35b": "35b-a3b", "35b-a3b-base": "35b-a3b"}.get(
        normalized,
        normalized,
    )
    try:
        return QWEN35_ANALYSIS_PROFILES[normalized]
    except KeyError as exc:
        valid = ", ".join(QWEN35_ANALYSIS_PROFILES)
        raise ValueError(
            f"Unknown Qwen3.5 analysis profile {key!r}; choose one of: {valid}"
        ) from exc


def _qwen35_automodel(model_id: str) -> Any | None:
    if "qwen3.5" not in model_id.lower():
        return None
    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError as exc:
        raise ImportError(
            "Qwen3.5 requires AutoModelForMultimodalLM from the Transformers main branch. "
            "Run the notebook bootstrap cell before importing mindscopex_analysis."
        ) from exc
    return AutoModelForMultimodalLM


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
    automodel = _qwen35_automodel(model_id)
    if automodel is not None:
        base_kwargs["automodel"] = automodel
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
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for Qwen text generation.") from exc

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except ValueError as exc:
        if "qwen3_5" in str(exc).lower() or "qwen3.5" in model_id.lower():
            raise RuntimeError(
                "Qwen3.5 requires a Transformers build with qwen3_5 support. "
                "Follow the official Qwen3.5 model card and install the latest Transformers."
            ) from exc
        raise

    model_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        **kwargs,
    }
    resolved_dtype = dtype_from_name(dtype)
    if resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype

    if config.model_type in {"qwen3_5", "qwen3_5_moe"}:
        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Qwen3.5 requires AutoModelForMultimodalLM from the latest Transformers build."
            ) from exc
        tokenizer = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForMultimodalLM.from_pretrained(model_id, **model_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    return model, tokenizer


def clear_device_cache() -> None:
    """Release unused Python and CUDA objects between sequential model runs."""

    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
