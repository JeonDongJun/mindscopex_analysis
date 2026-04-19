from __future__ import annotations

from typing import Any

import torch
from nnsight import LanguageModel
from omegaconf import DictConfig


def _dtype_from_cfg(s: str) -> torch.dtype:
    m = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return m.get(s, torch.float32)


def load_language_model(cfg: DictConfig) -> LanguageModel:
    device = cfg.model.device
    kwargs: dict[str, Any] = {
        "dtype": _dtype_from_cfg(str(cfg.model.dtype)),
        "trust_remote_code": bool(cfg.model.trust_remote_code),
    }
    if device and device != "auto":
        kwargs["device_map"] = device
    else:
        kwargs["device_map"] = "auto"
    return LanguageModel(str(cfg.model.name), **kwargs)


def _get_module(lm: LanguageModel, path: str) -> Any:
    cur: Any = lm
    for part in path.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def count_blocks(lm: LanguageModel, template: str) -> int:
    base = template.replace("{i}", "0")
    parts = base.split(".")
    if not parts[-1].isdigit():
        raise ValueError(f"template must end with .{{i}}, got {template!r}")
    parent_path = ".".join(parts[:-1])
    parent = _get_module(lm, parent_path)
    return len(parent)


def reduce_hidden(h: torch.Tensor, token_index: str, reduce: str) -> float:
    if h.dim() == 2:
        h = h.unsqueeze(0)
    if token_index == "last":
        vec = h[0, -1]
    elif token_index == "mean":
        vec = h[0].mean(dim=0)
    else:
        raise ValueError(token_index)

    if reduce == "l2_norm":
        return float(torch.linalg.norm(vec).item())
    if reduce == "l2_norm_mean":
        norms = torch.linalg.norm(h[0], dim=-1)
        return float(norms.mean().item())
    if reduce == "mean_abs":
        return float(h[0].abs().mean().item())
    raise ValueError(reduce)


def _tensor_from_saved(saved: Any) -> torch.Tensor:
    tensor = getattr(saved, "value", saved)
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    return tensor


def capture_layer_profile(
    lm: LanguageModel,
    prompt: str,
    cfg: DictConfig,
) -> tuple[list[int], list[float], list[torch.Tensor | None]]:
    template: str = str(cfg.capture.block_path_template)
    n_layers = count_blocks(lm, template)
    layer_indices = list(range(n_layers))
    saved_slots: list[Any] = []

    with lm.trace(prompt):
        for i in layer_indices:
            path = template.format(i=i)
            block = _get_module(lm, path)
            hidden_proxy = block.output[0]
            saved_slots.append(hidden_proxy.save())

    scalars: list[float] = []
    raw_tensors: list[torch.Tensor | None] = []
    for saved in saved_slots:
        tensor = _tensor_from_saved(saved)
        scalars.append(
            reduce_hidden(tensor, str(cfg.capture.token_index), str(cfg.capture.reduce))
        )
        raw_tensors.append(tensor if bool(cfg.capture.save_raw_tensors) else None)

    return layer_indices, scalars, raw_tensors
