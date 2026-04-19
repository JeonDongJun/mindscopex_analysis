from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


def _hook_layer_output(storage: dict, name: str):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        storage[name] = h.detach().cpu()
    return hook


def collect_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    layer_indices: list[int] | None = None,
    token_position: str = "all",
    device: str = "cpu",
) -> dict[int, torch.Tensor]:
    """텍스트 목록을 순전파하고 지정 레이어의 hidden state를 모은다.

    Returns
    -------
    dict[int, Tensor]
        layer_idx → (total_tokens, d_model)
    """
    layers = _get_layers(model)
    n_layers = len(layers)
    if layer_indices is None:
        layer_indices = list(range(n_layers))

    handles = []
    storage: dict[str, torch.Tensor] = {}
    for idx in layer_indices:
        h = layers[idx].register_forward_hook(_hook_layer_output(storage, str(idx)))
        handles.append(h)

    collected: dict[int, list[torch.Tensor]] = {i: [] for i in layer_indices}
    model.eval()

    with torch.no_grad():
        for text in tqdm(texts, desc="collecting activations"):
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            model(**enc)

            for idx in layer_indices:
                h = storage[str(idx)]  # (1, seq, d)
                if token_position == "last":
                    collected[idx].append(h[0, -1:])
                elif token_position == "mean":
                    collected[idx].append(h[0].mean(dim=0, keepdim=True))
                else:
                    collected[idx].append(h[0])

    for h in handles:
        h.remove()

    return {idx: torch.cat(vecs, dim=0) for idx, vecs in collected.items()}


def _get_layers(model):
    """HuggingFace causal LM의 transformer 블록 리스트를 반환."""
    for path in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise ValueError("지원하지 않는 모델 아키텍처: transformer 블록을 찾을 수 없음")
