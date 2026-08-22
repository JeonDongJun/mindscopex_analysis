"""Match a feature to its counterpart at another layer, by more than geometry.

``multisite_ablation`` transplants one layer's decoder direction to its
neighbours. That is a fair diagnostic -- the residual stream shares a basis -- but
it is not the same as "the feature at that layer", because each layer has its own
SAE and its own numbering. To make a cross-layer claim, the counterpart has to be
identified first.

Decoder cosine alone is not enough: the dictionary is overcomplete, so directions
are non-orthogonal and a high cosine can be geometric coincidence. These helpers
combine three independent signals, all measured on the same items:

    decoder cosine        do they point the same way?
    activation corr       do they fire on the same items, at the same strength?
    effect corr           does ablating each move the margin the same way per item?

Everything here is pure so the scoring rule can be tested without a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation, 0 when either side is constant (undefined, not 1)."""

    if len(a) != len(b):
        raise ValueError("pearson needs equal-length sequences")
    if len(a) < 2:
        return 0.0
    x = torch.tensor([float(v) for v in a], dtype=torch.float64)
    y = torch.tensor([float(v) for v in b], dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(x.norm() * y.norm())
    return float(x @ y) / denominator if denominator > 1e-12 else 0.0


def sibling_score(
    decoder_cosine: float,
    activation_corr: float,
    effect_corr: float,
    *,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Combine the three signals into one ranking score.

    Each term is clamped at 0 first: a *negative* cosine, correlation, or effect
    correlation is evidence against the pair being the same feature, and letting a
    negative term be cancelled by a strong positive one would rank an anti-correlated
    pair above an unrelated one. The product-style aggregate (a weighted geometric
    mean) then refuses to let one strong signal carry two weak ones -- which plain
    cosine ranking does, and which is exactly the failure this function exists to
    avoid.
    """

    terms = [
        max(0.0, float(decoder_cosine)),
        max(0.0, float(activation_corr)),
        max(0.0, float(effect_corr)),
    ]
    if any(term <= 0.0 for term in terms):
        return 0.0
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    product = 1.0
    for term, weight in zip(terms, weights, strict=True):
        product *= term ** (weight / total)
    return product


def rank_siblings(
    candidates: Sequence[dict[str, Any]],
    *,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Score and sort candidate siblings, best first.

    Each candidate needs ``decoder_cosine``, ``activation_corr`` and ``effect_corr``.
    Candidates scoring at or below ``min_score`` are dropped rather than ranked, so
    "no sibling was found" stays distinguishable from "the best one was poor".
    """

    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        score = sibling_score(
            float(candidate["decoder_cosine"]),
            float(candidate["activation_corr"]),
            float(candidate["effect_corr"]),
            weights=weights,
        )
        if score > min_score:
            scored.append({**candidate, "combined_score": score})
    scored.sort(key=lambda row: -row["combined_score"])
    return scored


def difference_in_differences(
    joint: Sequence[float],
    parts: Sequence[Sequence[float]],
    null_joint: Sequence[float],
    null_parts: Sequence[Sequence[float]],
) -> list[float]:
    """Per-item interaction of the real pair minus the matched null pair's.

    The joint condition removes strictly more norm than either part, and the network
    is non-linear, so a positive interaction appears even for unrelated directions.
    Testing ``joint - sum(parts)`` against zero would therefore find "superadditivity"
    everywhere; subtracting the null's own interaction is what makes the number mean
    something.
    """

    if len(joint) != len(null_joint):
        raise ValueError("joint and null_joint must be aligned per item")
    real = _interaction(joint, parts)
    null = _interaction(null_joint, null_parts)
    return [r - n for r, n in zip(real, null, strict=True)]


def _interaction(joint: Sequence[float], parts: Sequence[Sequence[float]]) -> list[float]:
    if not parts:
        return [float(value) for value in joint]
    for part in parts:
        if len(part) != len(joint):
            raise ValueError("every part must be aligned with joint per item")
    return [
        float(joint[index]) - sum(float(part[index]) for part in parts)
        for index in range(len(joint))
    ]
