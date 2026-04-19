"""뉴런(hidden dimension) 단위 활성화 분석 유틸."""
from __future__ import annotations

import torch
import numpy as np

# §8 코사인 유사도: 서로 다른 d_model 은 그룹별 행렬로 계산 (구버전은 단일 stack 이라 오류)
COSINE_SIM_IMPL_VERSION = 2


def per_neuron_stats(hidden: torch.Tensor) -> dict[str, np.ndarray]:
    """hidden: (tokens, d_model) → 뉴런별 통계.

    Returns: mean, std, max_abs, activation_rate (> 0 비율)
    """
    h = hidden.float()
    return {
        "mean": h.mean(dim=0).numpy(),
        "std": h.std(dim=0).numpy(),
        "max_abs": h.abs().max(dim=0).values.numpy(),
        "activation_rate": (h > 0).float().mean(dim=0).numpy(),
    }


def top_k_neurons(hidden: torch.Tensor, k: int = 30, metric: str = "mean_abs") -> list[int]:
    """가장 활성화가 큰 뉴런 인덱스 k개."""
    h = hidden.float()
    if metric == "mean_abs":
        scores = h.abs().mean(dim=0)
    elif metric == "max_abs":
        scores = h.abs().max(dim=0).values
    elif metric == "std":
        scores = h.std(dim=0)
    else:
        raise ValueError(metric)
    return scores.topk(k).indices.tolist()


def differential_neurons(
    hidden_a: torch.Tensor,
    hidden_b: torch.Tensor,
    k: int = 30,
) -> tuple[list[int], np.ndarray]:
    """두 조건 간 평균 활성화 차이가 가장 큰 뉴런 인덱스."""
    mean_a = hidden_a.float().mean(dim=0)
    mean_b = hidden_b.float().mean(dim=0)
    diff = (mean_a - mean_b).abs().numpy()
    top_idx = np.argsort(diff)[::-1][:k].tolist()
    return top_idx, diff


def cosine_similarity_matrix(
    hiddens: dict[str, torch.Tensor],
) -> dict[int, tuple[list[str], np.ndarray]]:
    """조건별 평균 hidden 벡터 간 코사인 유사도 행렬.

    모델마다 ``d_model`` 이 다르면 같은 차원끼리만 비교할 수 있으므로,
    ``d_model`` 별로 (레이블 순서, N×N 유사도 행렬) 딕셔너리를 반환한다.
    """
    by_d: dict[int, list[str]] = {}
    for k, t in hiddens.items():
        d = int(t.shape[-1])
        by_d.setdefault(d, []).append(k)

    out: dict[int, tuple[list[str], np.ndarray]] = {}
    for d, keys in by_d.items():
        means = [hiddens[k].float().mean(dim=0) for k in keys]
        stacked = torch.stack(means)  # (n, d)
        norms = stacked.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        normed = stacked / norms
        sim = (normed @ normed.T).numpy()
        out[d] = (keys, sim)
    return out
