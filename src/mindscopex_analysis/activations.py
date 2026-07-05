"""NNsight activation capture utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch

from mindscopex_analysis.models import DEFAULT_BLOCK_PATH_TEMPLATE

TokenPosition = Literal["all", "last", "mean"]


def get_module(root: Any, path: str) -> Any:
    """Resolve a dotted module path, accepting integer path components."""

    current = root
    for part in path.split("."):
        if not part:
            continue
        if part.lstrip("-").isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def count_layers(lm: Any, block_path_template: str = DEFAULT_BLOCK_PATH_TEMPLATE) -> int:
    """Count blocks addressed by a template such as ``model.language_model.layers.{layer}``."""

    if "{layer}" not in block_path_template:
        raise ValueError("block_path_template must contain `{layer}`")
    parent_path = block_path_template.split("{layer}", 1)[0].rstrip(".")
    parent = get_module(lm, parent_path)
    return len(parent)


def tensor_from_saved(saved: Any) -> torch.Tensor:
    """Convert an NNsight saved proxy into a detached CPU tensor."""

    value = getattr(saved, "value", saved)
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.detach().cpu()


def select_token_positions(hidden: torch.Tensor, token_position: TokenPosition) -> torch.Tensor:
    """Reduce a hidden state tensor to token rows for SAE analysis."""

    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(0)
    if hidden.dim() != 3:
        raise ValueError(f"Expected hidden shape (batch, seq, d_model); got {tuple(hidden.shape)}")
    if hidden.shape[0] != 1:
        raise ValueError("capture_residual_stream currently expects one prompt per trace")

    tokens = hidden[0]
    if token_position == "all":
        return tokens
    if token_position == "last":
        return tokens[-1:].contiguous()
    if token_position == "mean":
        return tokens.mean(dim=0, keepdim=True)
    raise ValueError(f"Unknown token_position={token_position!r}")


def capture_residual_stream(
    lm: Any,
    prompts: Sequence[str],
    layers: Sequence[int],
    *,
    block_path_template: str = DEFAULT_BLOCK_PATH_TEMPLATE,
    token_position: TokenPosition = "last",
    output_index: int | None = None,
) -> dict[int, torch.Tensor]:
    """Capture residual stream tensors from Qwen layers using NNsight.

    Qwen3.5 decoder blocks return the hidden-state tensor directly, so the
    default captures ``block.output`` without indexing away the batch axis.
    Pass an explicit ``output_index`` only for models whose block output is a
    tuple.

    Returns a dict mapping ``layer -> tensor``. With ``token_position="last"``
    the tensor shape is ``(n_prompts, d_model)``; with ``"all"`` it is
    ``(total_tokens, d_model)``.
    """

    if not prompts:
        raise ValueError("prompts must not be empty")
    if not layers:
        raise ValueError("layers must not be empty")

    collected: dict[int, list[torch.Tensor]] = {int(layer): [] for layer in layers}

    for prompt in prompts:
        saved_by_layer: dict[int, Any] = {}
        with lm.trace(prompt):
            for layer in layers:
                layer = int(layer)
                block = get_module(lm, block_path_template.format(layer=layer, i=layer))
                output = block.output if output_index is None else block.output[output_index]
                saved_by_layer[layer] = output.save()

        for layer, saved in saved_by_layer.items():
            hidden = tensor_from_saved(saved)
            collected[layer].append(select_token_positions(hidden, token_position))

    return {layer: torch.cat(parts, dim=0) for layer, parts in collected.items()}


def capture_layer_residuals(
    lm: Any,
    prompts: Sequence[str],
    layer: int,
    *,
    block_path_template: str = DEFAULT_BLOCK_PATH_TEMPLATE,
    token_position: TokenPosition = "last",
    output_index: int | None = None,
) -> torch.Tensor:
    """Convenience wrapper for one layer."""

    return capture_residual_stream(
        lm,
        prompts,
        [int(layer)],
        block_path_template=block_path_template,
        token_position=token_position,
        output_index=output_index,
    )[int(layer)]
