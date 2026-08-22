"""Coactivation modules: the distributed alternative to a single mediating feature.

Single-feature ablation kept returning effects indistinguishable from matched
peers, which is the expected outcome if the behaviour is carried by a *set* of
features rather than one. These primitives build the set:

    sparse activations over the discovery items
        -> coactivation graph (which features fire together, and co-vary)
        -> modules (connected components above an edge threshold)
        -> one combined ablation direction per module

The joint ablation needs no multi-site machinery. ``remove_activation`` subtracts
``a_f * W_dec[f]``, and that is linear in ``f``, so removing a whole module at one
layer is a single edit along ``sum_f a_f * W_dec[f]`` -- which is what
:func:`module_ablation_direction` builds.

The null a module needs is another *module*: same layer, same size, comparable
firing frequency, same removed norm. A module of k features removes strictly more
norm than one feature, so comparing a module against a single-feature null would
reward size alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from mindscopex_analysis.activations import capture_layer_residuals
from mindscopex_analysis.cases import LureCase
from mindscopex_analysis.models import DEFAULT_BLOCK_PATH_TEMPLATE
from mindscopex_analysis.qwen_scope import (
    QwenScopeSAE,
    encode_qwen_scope_topk,
    qwen_scope_sparse_feature_values,
    sae_decoder_direction,
)

# ------------------------------------------------------------------ collection


def sparse_activation_matrix(
    lm: Any,
    cases: Sequence[LureCase],
    *,
    layer: int,
    sae: QwenScopeSAE,
    min_active_cases: int = 3,
    max_features: int = 200,
    block_path_template: str = DEFAULT_BLOCK_PATH_TEMPLATE,
) -> tuple[list[int], torch.Tensor]:
    """Sparse activations at the last prompt token: ``(feature_ids, cases x features)``.

    Only features that fire (TopK support, value > 0) in at least
    ``min_active_cases`` items are kept, most frequent first, so the matrix stays
    small instead of materialising the full dictionary.
    """

    supports: list[set[int]] = []
    residuals: list[torch.Tensor] = []
    counts: dict[int, int] = {}
    for case in cases:
        residual = capture_layer_residuals(
            lm,
            [case.prompt],
            int(layer),
            token_position="last",
            block_path_template=block_path_template,
        )
        residuals.append(residual)
        values, indices = encode_qwen_scope_topk(residual, sae)
        live = indices.detach().cpu()[values.detach().cpu() > 0].reshape(-1).tolist()
        support = {int(index) for index in live}
        supports.append(support)
        for feature_id in support:
            counts[feature_id] = counts.get(feature_id, 0) + 1

    feature_ids = [
        feature_id
        for feature_id, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if count >= min_active_cases
    ][: int(max_features)]
    if not feature_ids:
        return [], torch.zeros(len(cases), 0)

    rows = [
        qwen_scope_sparse_feature_values(residual, sae, feature_ids)
        .detach()
        .to(torch.float32)
        .cpu()
        .reshape(-1)
        for residual in residuals
    ]
    return feature_ids, torch.stack(rows)


# ----------------------------------------------------------------- graph


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(a.norm() * b.norm())
    return float(a @ b) / denominator if denominator > 1e-12 else 0.0


def coactivation_edges(
    matrix: torch.Tensor,
    feature_ids: Sequence[int],
    *,
    min_jaccard: float = 0.0,
) -> list[dict[str, Any]]:
    """Pairwise co-firing (Jaccard) and activation correlation over the items.

    Jaccard asks "do they fire on the same items"; the correlation asks "when both
    fire, do they move together". A pair can score high on one and low on the other,
    so both are reported and the caller decides which to threshold on.
    """

    if matrix.numel() == 0 or len(feature_ids) < 2:
        return []
    active = matrix > 0
    edges: list[dict[str, Any]] = []
    for i in range(len(feature_ids)):
        for j in range(i + 1, len(feature_ids)):
            both = int((active[:, i] & active[:, j]).sum())
            either = int((active[:, i] | active[:, j]).sum())
            jaccard = both / either if either else 0.0
            if jaccard < min_jaccard:
                continue
            edges.append(
                {
                    "feature_a": int(feature_ids[i]),
                    "feature_b": int(feature_ids[j]),
                    "co_fire": both,
                    "jaccard": jaccard,
                    "activation_corr": _pearson(matrix[:, i], matrix[:, j]),
                }
            )
    edges.sort(key=lambda edge: -edge["jaccard"])
    return edges


def modules_from_edges(
    edges: Sequence[dict[str, Any]],
    *,
    edge_threshold: float = 0.3,
    metric: str = "jaccard",
    min_size: int = 2,
    max_size: int = 50,
) -> list[list[int]]:
    """Connected components of the graph kept above ``edge_threshold``.

    Deliberately the simplest clustering that can answer the question: if a module
    exists at all, it should survive a plain threshold. Components larger than
    ``max_size`` are dropped -- a component that swallows the graph is the
    threshold being too low, not a module.
    """

    if metric not in {"jaccard", "activation_corr"}:
        raise ValueError(f"Unknown coactivation metric {metric!r}")

    parent: dict[int, int] = {}

    def find(node: int) -> int:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for edge in edges:
        if float(edge[metric]) >= edge_threshold:
            union(int(edge["feature_a"]), int(edge["feature_b"]))

    groups: dict[int, list[int]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    modules = [sorted(group) for group in groups.values() if min_size <= len(group) <= max_size]
    modules.sort(key=lambda group: (-len(group), group))
    return modules


# ------------------------------------------------------------- intervention


def module_ablation_direction(
    sae: QwenScopeSAE,
    feature_ids: Sequence[int],
    values: Sequence[float],
) -> torch.Tensor:
    """``sum_f a_f * W_dec[f]`` -- one vector that removes the whole module.

    ``remove_activation`` is linear in the feature, so subtracting this single
    vector is exactly equivalent to ablating each member in turn at one layer.
    """

    if len(feature_ids) != len(values):
        raise ValueError("feature_ids and values must be the same length")
    if not feature_ids:
        raise ValueError("a module needs at least one feature")
    return sae_decoder_direction(sae, [int(f) for f in feature_ids], [float(v) for v in values])


def sample_frequency_matched_modules(
    feature_ids: Sequence[int],
    activation_counts: Sequence[int],
    *,
    size: int,
    exclude: Sequence[int] = (),
    n_modules: int = 20,
    tolerance: int = 1,
    seed: int = 0,
) -> list[list[int]]:
    """Random modules of ``size`` features whose firing counts match ``exclude``'s.

    Matching frequency matters because a module of features that fire everywhere
    removes more norm on more items than a module of rare ones, so an unmatched
    random module is not a null for the real one.
    """

    banned = {int(f) for f in exclude}
    target = sorted(
        count
        for feature_id, count in zip(feature_ids, activation_counts, strict=True)
        if int(feature_id) in banned
    )
    pool_by_count: dict[int, list[int]] = {}
    for feature_id, count in zip(feature_ids, activation_counts, strict=True):
        if int(feature_id) in banned:
            continue
        pool_by_count.setdefault(int(count), []).append(int(feature_id))

    generator = torch.Generator().manual_seed(int(seed))
    modules: list[list[int]] = []
    for _ in range(int(n_modules)):
        chosen: list[int] = []
        used: set[int] = set()
        for wanted in (target or [0] * size)[:size]:
            options = [
                feature_id
                for count, ids in pool_by_count.items()
                if abs(count - wanted) <= tolerance
                for feature_id in ids
                if feature_id not in used
            ]
            if not options:
                break
            index = int(torch.randint(0, len(options), (1,), generator=generator))
            chosen.append(options[index])
            used.add(options[index])
        if len(chosen) == size:
            modules.append(sorted(chosen))
    return modules


def module_coherence(
    matrix: torch.Tensor,
    feature_ids: Sequence[int],
    module: Sequence[int],
) -> float:
    """Mean pairwise activation correlation inside a module (0 for singletons)."""

    index = {int(f): i for i, f in enumerate(feature_ids)}
    columns = [index[int(f)] for f in module if int(f) in index]
    if len(columns) < 2:
        return 0.0
    values = [
        _pearson(matrix[:, a], matrix[:, b])
        for i, a in enumerate(columns)
        for b in columns[i + 1 :]
    ]
    return sum(values) / len(values) if values else 0.0


def module_norm(direction: torch.Tensor) -> float:
    return float(torch.linalg.norm(direction.to(torch.float32)))


def rescale_to_norm(direction: torch.Tensor, target: float) -> torch.Tensor:
    """Unit-ise then scale, so a null module removes the same norm as the real one."""

    current = torch.linalg.norm(direction.to(torch.float32)).clamp_min(1e-12)
    return direction.to(torch.float32) / current * float(target)
