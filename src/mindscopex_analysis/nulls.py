"""Null models for feature-ablation effects, shared by the study and the diagnostics.

An ablation's margin delta means nothing on its own -- the question is always
"compared with removing *what else*?". Three nulls, in increasing strictness:

* **Gaussian** matched-norm directions. Cheap, but a very low bar: an isotropic
  vector in d_model dimensions has expected |cosine| ~ sqrt(2/(pi*d)) with any
  meaningful direction, so it barely touches anything the model uses and almost
  any trained decoder row beats it.
* **Peer features** -- decoder directions of other SAE features that actually fire
  at the same site, removed at the same norm. This controls for "a real feature
  fires here", which is the comparison a reader cares about.
* **Selection adjustment.** A winner picked as the best of k candidates is a
  maximum, and a maximum must be compared against the distribution of maxima, not
  against a single draw. Without this, searching harder always looks like a
  stronger result.

Headline statistics are empirical percentiles, not Gaussian z-scores: these delta
distributions are heavy-tailed and small, so a z fitted to a handful of draws
extrapolates a tail that was never observed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from mindscopex_analysis.activations import capture_layer_residuals
from mindscopex_analysis.cases import LureCase
from mindscopex_analysis.effects import InterventionMode, answer_logprob_margin
from mindscopex_analysis.models import DEFAULT_BLOCK_PATH_TEMPLATE
from mindscopex_analysis.qwen_scope import (
    QwenScopeSAE,
    encode_qwen_scope_topk,
    qwen_scope_sparse_feature_values,
    sae_decoder_direction,
)

# --------------------------------------------------------------- pure statistics


def empirical_percentile(observed: float, null_values: Sequence[float]) -> float | None:
    """Fraction of null draws below ``observed``; None when there are no draws."""

    if not null_values:
        return None
    return sum(1 for value in null_values if value < float(observed)) / len(null_values)


def selection_adjusted_percentile(
    observed: float,
    null_values: Sequence[float],
    *,
    selection_k: int,
    bootstrap: int = 20000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare ``observed`` against the distribution of the best of ``selection_k`` nulls.

    The reported winner was the best of roughly ``selection_k`` candidates, so the
    honest reference is the maximum of that many null draws. Resampling the null
    with replacement gives that distribution.

    Note the ceiling: a bootstrap maximum can never exceed the largest observed
    null draw, so when no single null beats ``observed`` the percentile is 1.0 by
    construction. ``max_mean`` and ``max_p95`` carry the informative content --
    they say what score searching this hard buys you for free.
    """

    if not null_values or selection_k < 1:
        return {"selection_k": selection_k, "percentile": None}
    source = torch.tensor([float(value) for value in null_values], dtype=torch.float64)
    generator = torch.Generator().manual_seed(int(seed))
    picks = torch.randint(
        0, source.numel(), (int(bootstrap), int(selection_k)), generator=generator
    )
    maxima = source[picks].max(dim=1).values
    return {
        "selection_k": int(selection_k),
        "bootstrap": int(bootstrap),
        "max_mean": float(maxima.mean()),
        "max_p95": float(maxima.quantile(0.95)),
        "percentile": float((maxima < float(observed)).to(torch.float64).mean()),
        "p_value": float((maxima >= float(observed)).to(torch.float64).mean()),
    }


def gaussian_null_directions(d_model: int, n_draws: int, *, seed: int = 0) -> torch.Tensor:
    """``(n_draws, d_model)`` unit rows, deterministic in ``seed``."""

    generator = torch.Generator().manual_seed(int(seed))
    directions = torch.randn(int(n_draws), int(d_model), generator=generator)
    return directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)


def peer_null_directions(sae: QwenScopeSAE, feature_ids: Sequence[int]) -> torch.Tensor:
    """Unit decoder rows for ``feature_ids`` -- the strict null's directions."""

    rows = [
        sae_decoder_direction(sae, [int(feature_id)]).detach().to(torch.float32).cpu()
        for feature_id in feature_ids
    ]
    if not rows:
        return torch.empty(0, int(sae.d_model))
    stacked = torch.stack(rows)
    return stacked / stacked.norm(dim=1, keepdim=True).clamp_min(1e-12)


# ------------------------------------------------------------------- the panel


@dataclass(frozen=True)
class NullPanel:
    """Per-case baselines and the feature's own effect, on a fixed set of cases.

    Every null draw is scored on exactly these cases at exactly these perturbation
    norms, so the null and the observation differ only in direction.
    """

    cases: tuple[LureCase, ...]
    baseline_margins: tuple[float, ...]
    observed_deltas: tuple[float, ...]
    target_norms: tuple[float, ...]
    feature_values: tuple[float, ...]
    peer_pool: tuple[int, ...] = field(default=())

    @property
    def observed_mean(self) -> float:
        values = self.observed_deltas
        return sum(values) / len(values) if values else 0.0

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "case_id": case.case_id,
                "family": case.family,
                "feature_value": value,
                "target_norm": norm,
                "baseline_margin": baseline,
                "observed_delta": delta,
            }
            for case, value, norm, baseline, delta in zip(
                self.cases,
                self.feature_values,
                self.target_norms,
                self.baseline_margins,
                self.observed_deltas,
                strict=True,
            )
        ]


def build_null_panel(
    lm: Any,
    cases: Sequence[LureCase],
    *,
    layer: int,
    sae: QwenScopeSAE,
    feature_id: int,
    coefficient: float = 1.0,
    intervention_mode: InterventionMode = "remove_activation",
    collect_peers: bool = True,
    block_path_template: str = DEFAULT_BLOCK_PATH_TEMPLATE,
) -> NullPanel:
    """Measure the feature's own effect and the perturbation norm it removes."""

    direction = sae_decoder_direction(sae, [int(feature_id)]).detach()
    direction_norm = float(direction.to(torch.float32).norm())
    baselines: list[float] = []
    deltas: list[float] = []
    norms: list[float] = []
    values: list[float] = []
    peers: set[int] = set()

    for case in cases:
        residual = capture_layer_residuals(
            lm,
            [case.prompt],
            int(layer),
            token_position="last",
            block_path_template=block_path_template,
        )
        value = float(
            qwen_scope_sparse_feature_values(residual, sae, [int(feature_id)])
            .detach()
            .to(torch.float32)
            .reshape(-1)[0]
        )
        if collect_peers:
            top_values, top_indices = encode_qwen_scope_topk(residual, sae)
            live = top_indices.detach().cpu()[top_values.detach().cpu() > 0]
            peers.update(int(index) for index in live.reshape(-1).tolist())
        baseline = float(
            answer_logprob_margin(
                lm,
                case.prompt,
                correct_answer=case.correct_answer,
                lure_answer=case.lure_answer,
                block_path_template=block_path_template,
            ).margin
        )
        ablated = float(
            answer_logprob_margin(
                lm,
                case.prompt,
                correct_answer=case.correct_answer,
                lure_answer=case.lure_answer,
                layer=int(layer),
                direction=direction,
                feature_value=value,
                coefficient=coefficient,
                intervention_mode=intervention_mode,
                block_path_template=block_path_template,
            ).margin
        )
        baselines.append(baseline)
        deltas.append(baseline - ablated)
        values.append(value)
        norms.append(abs(value) * abs(coefficient) * direction_norm)

    peers.discard(int(feature_id))
    return NullPanel(
        cases=tuple(cases),
        baseline_margins=tuple(baselines),
        observed_deltas=tuple(deltas),
        target_norms=tuple(norms),
        feature_values=tuple(values),
        peer_pool=tuple(sorted(peers)),
    )


def null_panel_means(
    lm: Any,
    panel: NullPanel,
    directions: torch.Tensor,
    *,
    layer: int,
    block_path_template: str = DEFAULT_BLOCK_PATH_TEMPLATE,
    progress: Any | None = None,
) -> list[float]:
    """Panel-mean margin delta for each unit direction, norm-matched per case.

    Direction *i* is applied to every case (scaled to that case's own removal
    norm), so averaging down the column yields one draw of the same panel-mean
    statistic ``panel.observed_mean`` is.
    """

    if directions.numel() == 0:
        return []
    sums = [0.0] * int(directions.shape[0])
    for index, (case, baseline, norm) in enumerate(
        zip(panel.cases, panel.baseline_margins, panel.target_norms, strict=True), start=1
    ):
        for draw in range(int(directions.shape[0])):
            vector = directions[draw] * float(norm)
            ablated = float(
                answer_logprob_margin(
                    lm,
                    case.prompt,
                    correct_answer=case.correct_answer,
                    lure_answer=case.lure_answer,
                    layer=int(layer),
                    direction=vector,
                    feature_value=1.0,
                    coefficient=-1.0,  # add_vector with -1 subtracts the vector
                    intervention_mode="add_vector",
                    block_path_template=block_path_template,
                ).margin
            )
            sums[draw] += float(baseline) - ablated
        if progress is not None:
            progress(f"null: case {index}/{len(panel.cases)} ({directions.shape[0]} draws)")
    return [total / len(panel.cases) for total in sums]


def evaluate_feature_null(
    lm: Any,
    cases: Sequence[LureCase],
    *,
    layer: int,
    sae: QwenScopeSAE,
    feature_id: int,
    gaussian_draws: int = 200,
    peer_draws: int = 200,
    selection_k: int = 0,
    bootstrap: int = 20000,
    seed: int = 0,
    coefficient: float = 1.0,
    intervention_mode: InterventionMode = "remove_activation",
    block_path_template: str = DEFAULT_BLOCK_PATH_TEMPLATE,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Full null evaluation for one feature: Gaussian, peer, and selection-adjusted."""

    panel = build_null_panel(
        lm,
        cases,
        layer=layer,
        sae=sae,
        feature_id=feature_id,
        coefficient=coefficient,
        intervention_mode=intervention_mode,
        collect_peers=peer_draws > 0,
        block_path_template=block_path_template,
    )
    observed = panel.observed_mean

    gaussian: list[float] = []
    if gaussian_draws > 0:
        gaussian = null_panel_means(
            lm,
            panel,
            gaussian_null_directions(int(sae.d_model), gaussian_draws, seed=seed),
            layer=layer,
            block_path_template=block_path_template,
            progress=progress,
        )

    peer_ids: list[int] = []
    peer: list[float] = []
    if peer_draws > 0 and panel.peer_pool:
        generator = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(len(panel.peer_pool), generator=generator).tolist()
        peer_ids = [panel.peer_pool[i] for i in order[:peer_draws]]
        peer = null_panel_means(
            lm,
            panel,
            peer_null_directions(sae, peer_ids),
            layer=layer,
            block_path_template=block_path_template,
            progress=progress,
        )

    # The peer null is the one worth adjusting: it already contains real features,
    # so "best of k" over it is the closest thing to repeating our own search.
    reference = peer or gaussian
    adjusted = (
        selection_adjusted_percentile(
            observed, reference, selection_k=selection_k, bootstrap=bootstrap, seed=seed + 1
        )
        if selection_k > 0
        else {"selection_k": 0, "percentile": None}
    )
    beaten = [fid for fid, value in zip(peer_ids, peer, strict=True) if value >= observed]
    return {
        "feature_id": int(feature_id),
        "layer": int(layer),
        "panel_n": len(panel.cases),
        "observed_mean_delta": observed,
        "gaussian_draws": len(gaussian),
        "gaussian_mean": (sum(gaussian) / len(gaussian)) if gaussian else None,
        "gaussian_percentile": empirical_percentile(observed, gaussian),
        "peer_draws": len(peer),
        "peer_mean": (sum(peer) / len(peer)) if peer else None,
        "peer_feature_percentile": empirical_percentile(observed, peer),
        "peer_pool_size": len(panel.peer_pool),
        "n_peers_matching_or_beating": len(beaten),
        "top_peers_beating": beaten[:10],
        "selection_adjusted_percentile": adjusted.get("percentile"),
        "selection_adjusted_p": adjusted.get("p_value"),
        "selection_adjusted": adjusted,
        "panel": panel.rows(),
        "gaussian_values": gaussian,
        "peer_values": peer,
        "peer_feature_ids": peer_ids,
    }
