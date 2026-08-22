"""Does a coactivating SET of features mediate the lure where no single one does?

Every single-feature result so far came back indistinguishable from matched peers.
That is the expected outcome if the behaviour is carried by a distributed module,
so this job runs the module arm of the question and keeps the single-feature arm
alongside it as the comparison:

    graph items -> sparse activations -> coactivation graph -> modules
        -> module SCORE (frequency, generalization, causality, specificity,
           coherence -- a weighted geometric mean, see DEFAULT_SCORE_WEIGHTS)
        -> joint ablation of the best-scoring modules on HELD-OUT items

Three disjoint slices, because each of those steps would otherwise be scored on
the very items that produced it:

    graph   builds the coactivation graph        (discovery split, majority)
    val     scores the candidate modules         (discovery split, remainder)
    test    runs the interventions               (held-out split)

Four conditions per held-out item, which is what makes the answer readable:

    single_best      the strongest individual member, ablated alone
    members_apart    every member ablated alone (their deltas summed afterwards)
    module_joint     the whole module removed in one edit
    random_module    frequency-matched, size-matched, norm-matched module nulls

Two traps this design avoids, and one it does not. A module removes strictly more
norm than a single feature, so "the module beat the single feature" proves nothing
without a norm-matched module null -- hence random_module, rescaled to the real
module's norm. And a module can beat its null by damaging the model rather than the
lure, so every margin is split into its correct and lure logprob deltas.

The one it does NOT close: the intervened module is the argmax of the score over the
candidates, and the score's heaviest term is measured from real single-feature causal
deltas, while the random nulls are drawn on size and firing frequency alone. So
`joint_minus_random` is a norm-matched contrast, not a selection-corrected test.
`module_summary.json` carries that caveat in `random_module_null`; nothing in this
file should describe that comparison as though the selection had been controlled.
"""
# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch

from experiments.runners.config import load_toml, run_name, table
from mindscopex_analysis import (
    mean_or_none,
    DEFAULT_ANALYSIS_PROFILE_KEY,
    EditSite,
    answer_logprob_margin,
    coactivation_edges,
    default_sae_device,
    dtype_from_name,
    family_balanced_subset,
    get_qwen35_analysis_profile,
    instruct_lure_cases,
    load_qwen_language_model,
    load_qwen_scope_sae,
    lure_dataset_cases,
    module_ablation_direction,
    module_coherence,
    module_norm,
    modules_from_edges,
    multi_site_answer_margin,
    paired_summary,
    qwen_scope_sparse_feature_values,
    recommended_dtype_name,
    rescale_to_norm,
    sample_frequency_matched_modules,
    sparse_activation_matrix,
    split_lure_cases,
)
from mindscopex_analysis.activations import capture_layer_residuals


def _log(message: str) -> None:
    print(f"[modules] {message}", flush=True)


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return path


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class _NoModulesFound(RuntimeError):
    """A recorded outcome, not a crash: the search produced nothing to intervene on.

    Every one of these is only reachable once the 27B and its SAE are resident -- an
    entire Colab session already spent -- so none of them may die with the manifest
    still stamped ``running``. run() catches this, writes the same artifacts the
    success path writes (empty), stamps ``status: no_module_found`` with ``code``,
    prints NO_MODULE_FOUND, and exits 0 unless ``[module].fail_on_empty``.
    """

    def __init__(self, message: str, *, code: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.detail = dict(detail or {})


# ------------------------------------------------------- graph diagnostics (pure)


def graph_feature_selection(
    feature_ids: Sequence[int],
    counts: Sequence[int],
    n_cases: int,
    *,
    max_active_frac: float = 1.0,
) -> tuple[list[int], list[int]]:
    """Split ``feature_ids`` into (kept, dropped) by how often they fire.

    ``max_active_frac = 1.0`` keeps everything, and that is the deliberate default.
    The first 27B run found 25 of its 40 graph features firing on *all* 35 hostile
    discovery items, which is what made binary co-firing carry no information -- but
    "on for every hostile item" is exactly what a genuine affordance mediator should
    look like when the discovery split contains nothing but hostile items. Thinning
    those features out would delete the best candidates and keep the noise.

    What this default does NOT do is make the always-on features harmless. The score
    checks their hostile-vs-control selectivity in exactly one term (``specificity``,
    weight 1.5 of 5.0), and two of the remaining four are still frequency-flavoured:
    ``hostile_frequency`` is a firing count on the graph slice, and
    ``generalization`` multiplies by a co-firing RATE on the val slice. Frequency is
    also what put these features in the graph at all -- ``sparse_activation_matrix``
    keeps the top ``max_features`` sorted by firing count -- so a feature that fires
    on 60% of hostile items and 0% of controls, the most interesting kind, may never
    become eligible. That is a limit of the upstream selection rule, not something
    this knob can fix; ``hostile_frequency`` is down-weighted to 0.5 in response, and
    the honest summary is that this job scores frequent features against each other.

    The knob exists so the opposite arm can be run on purpose (a study that wants
    only graded features sets it to e.g. 0.95), never silently by default.
    """

    if len(feature_ids) != len(counts):
        raise ValueError("feature_ids and counts must be the same length")
    if not 0.0 < max_active_frac <= 1.0:
        raise ValueError("max_active_frac must be in (0, 1]")
    if n_cases <= 0:
        raise ValueError("n_cases must be positive")
    limit = max_active_frac * n_cases
    kept: list[int] = []
    dropped: list[int] = []
    for feature_id, count in zip(feature_ids, counts, strict=True):
        (kept if float(count) <= limit + 1e-9 else dropped).append(int(feature_id))
    return kept, dropped


NEGATIVE_TAIL_RATIO_CAVEAT = (
    "negative_tail_ratio is #(r <= -t) / #(r >= +t). It is NOT a false-discovery rate. "
    "It shipped as `symmetric_null_fdr` and was read as one, and the measured data "
    "refutes that reading: in the 27B edge table the negative tail at t=0.55 holds 21 of "
    "780 pairs, against 0.24 expected from an iid-noise floor at n=35 (Monte Carlo, 400k "
    "draws, P(r >= 0.55) = 3.1e-4) -- ~90x too heavy to be sampling noise, so it measures "
    "real anti-correlation and dividing by it estimates nothing. Use permutation_p, which "
    "is a null actually drawn from this run's own activation matrix."
)


def negative_tail_ratio(
    edges: Sequence[Mapping[str, Any]], *, metric: str, threshold: float
) -> float | None:
    """``#(metric <= -threshold) / #(metric >= +threshold)`` -- a tail-symmetry ratio.

    Renamed from ``symmetric_null_fdr``, which asserted a test that was never run. The
    old reading was "correlations under no-coactivation are symmetric about zero, so
    the negative tail counts the false positives in the positive tail". On the only
    real measurement of this graph that premise is false --
    results/runs/20260822-154406_modules_affordance_27b/artifacts/coactivation_edges.csv
    puts 21 of 780 pairs at ``r <= -0.55`` where iid noise at n=35 expects 0.24 -- so
    the negative tail is structure (anti-correlated features), not a noise floor, and
    the ratio is not a false-discovery rate of any kind.

    What survives the rename is a cheap descriptive statistic: how one-sided the
    signed metric is at this threshold. Near 1 means the two tails are the same size;
    near 0 means the positive tail stands alone. It is NOT a selection criterion --
    the curve it traces on the measured data is non-monotone and every interior point
    sits inside its own sampling error, so its argmin is a noise minimum.
    :func:`permutation_edge_null` is the null that can actually be tested against.
    ``None`` for a non-negative metric such as Jaccard, which has no negative tail.
    """

    if metric != "activation_corr":
        return None
    positive = sum(1 for edge in edges if float(edge[metric]) >= threshold)
    negative = sum(1 for edge in edges if float(edge[metric]) <= -threshold)
    return negative / positive if positive else None


def pairwise_metric_matrix(matrix: torch.Tensor, metric: str) -> torch.Tensor:
    """``(features x features)`` Jaccard or activation correlation, vectorised.

    Same definitions as ``coactivation_edges``, including its degenerate cases (a
    constant column correlates 0 with everything, an empty union scores 0), because
    the permutation null has to count edges by exactly the rule the real graph used.
    The loop form costs O(F^2) python-level torch calls per draw, too slow for a few
    hundred draws; this is two matmuls.
    """

    if metric not in {"jaccard", "activation_corr"}:
        raise ValueError(f"Unknown coactivation metric {metric!r}")
    values = matrix.to(torch.float64)
    if metric == "jaccard":
        active = (values > 0).to(torch.float64)
        both = active.T @ active
        counts = active.sum(dim=0)
        either = counts.unsqueeze(0) + counts.unsqueeze(1) - both
        return torch.where(either > 0, both / either.clamp_min(1.0), torch.zeros_like(both))
    centred = values - values.mean(dim=0, keepdim=True)
    norms = centred.norm(dim=0)
    scaled = centred / norms.clamp_min(1e-12)
    corr = scaled.T @ scaled
    # `_pearson` zeroes on ``|a| * |b| <= 1e-12``, so this must too, or a constant
    # column would contribute edges to the null that the real graph never counts.
    outer = norms.unsqueeze(0) * norms.unsqueeze(1)
    return torch.where(outer > 1e-12, corr, torch.zeros_like(corr))


def _surviving_edge_counts(pairwise: torch.Tensor, thresholds: Sequence[float]) -> list[int]:
    upper = torch.triu(torch.ones_like(pairwise, dtype=torch.bool), diagonal=1)
    kept = pairwise[upper]
    return [int((kept >= float(threshold)).sum()) for threshold in thresholds]


def permutation_edge_null(
    matrix: torch.Tensor,
    *,
    metric: str,
    thresholds: Sequence[float],
    draws: int = 200,
    seed: int = 0,
) -> dict[float, list[int]]:
    """Edges surviving each threshold when every feature is shuffled across items.

    This is the null the threshold actually needs, and the one the negative tail was
    standing in for. Permuting each COLUMN independently keeps every feature's own
    activation distribution -- its sparsity, its scale, its heavy tail -- and destroys
    only the item-by-item alignment between features, which is exactly the "these
    features do not coactivate" hypothesis. Nothing is assumed to be normal, iid or
    symmetric, which is where the negative-tail estimator went wrong.

    It also reads the first 27B run's failure straight off. When 25 of 40 features
    fire on every item, a column permutation cannot change who fires where, so the
    Jaccard null returns the complete graph too and ``permutation_p`` comes back at
    1.0: "every pair is an edge" was forced by the marginals, not a finding. That was
    visible before the search crashed, and nothing in the run reported it.

    Returns ``{threshold: [surviving edge count per draw]}``; the caller compares
    against the observed count.
    """

    if draws <= 0 or matrix.numel() == 0 or matrix.shape[1] < 2:
        return {float(threshold): [] for threshold in thresholds}
    values = matrix.to(torch.float64)
    n_items, n_features = values.shape
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    counts: dict[float, list[int]] = {float(threshold): [] for threshold in thresholds}
    for _ in range(int(draws)):
        order = torch.argsort(torch.rand(n_items, n_features, generator=generator), dim=0)
        permuted = torch.gather(values, 0, order)
        surviving = _surviving_edge_counts(pairwise_metric_matrix(permuted, metric), thresholds)
        for threshold, count in zip(thresholds, surviving, strict=True):
            counts[float(threshold)].append(count)
    return counts


def permutation_null_row(observed: int, null_counts: Sequence[int]) -> dict[str, Any]:
    """Summarise one threshold's permutation null, or say it was not measured.

    ``None`` everywhere rather than 0 when no draws were taken: "not measured" and
    "measured as zero" are different claims and the artifact has to keep them apart.
    """

    if not null_counts:
        return {
            "permutation_null_mean_edges": None,
            "permutation_null_p95_edges": None,
            "permutation_p": None,
            "permutation_draws": 0,
        }
    ordered = sorted(int(count) for count in null_counts)
    at_least = sum(1 for count in ordered if count >= int(observed))
    index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "permutation_null_mean_edges": sum(ordered) / len(ordered),
        "permutation_null_p95_edges": float(ordered[index]),
        # (b + 1) / (draws + 1), the same estimator as stats.sign_flip_p: the draws are
        # sampled, so a plain ratio would report an exact 0.0 it cannot support.
        "permutation_p": (at_least + 1) / (len(ordered) + 1),
        "permutation_draws": len(ordered),
    }


def threshold_sweep(
    edges: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    thresholds: Sequence[float],
    min_size: int,
    max_size: int,
    matrix: torch.Tensor | None = None,
    null_draws: int = 0,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """What the module search WOULD have found at each candidate threshold.

    Written out before the search is allowed to fail. The first 27B run died at the
    search and left the next run with no way to pick a better threshold except by
    spending another GPU session; recomputing this table costs a union-find over a
    few hundred edges.

    Pass ``matrix`` (the same ``cases x features`` activations the edges came from)
    with ``null_draws > 0`` to attach a permutation null to every row. Without it the
    ``permutation_*`` fields are ``None`` -- not measured, which is what a reader
    should see, rather than a placeholder that reads as a result.
    """

    ordered_thresholds = [float(threshold) for threshold in thresholds]
    null_counts = (
        permutation_edge_null(
            matrix, metric=metric, thresholds=ordered_thresholds, draws=null_draws, seed=seed
        )
        if matrix is not None and null_draws > 0
        else {threshold: [] for threshold in ordered_thresholds}
    )
    rows: list[dict[str, Any]] = []
    for threshold in ordered_thresholds:
        components = modules_from_edges(
            edges,
            edge_threshold=threshold,
            metric=metric,
            min_size=2,
            max_size=len(edges) + 2,
        )
        sizes = [len(component) for component in components]
        in_range = [size for size in sizes if min_size <= size <= max_size]
        kept = sum(1 for e in edges if float(e[metric]) >= threshold)
        rows.append(
            {
                "threshold": threshold,
                "n_edges_kept": kept,
                "component_sizes": sizes,
                "largest_component": max(sizes) if sizes else 0,
                "n_modules_in_range": len(in_range),
                # Renamed from `symmetric_null_fdr`, which named a test nobody ran.
                # module_search.json carries NEGATIVE_TAIL_RATIO_CAVEAT next to it.
                "negative_tail_ratio": negative_tail_ratio(
                    edges, metric=metric, threshold=threshold
                ),
                **permutation_null_row(kept, null_counts.get(threshold, [])),
            }
        )
    return rows


# ---------------------------------------------------------- module score (pure)

SCORE_TERMS = ("hostile_frequency", "generalization", "causal", "specificity", "coherence")

# Causality and control specificity carry the extra weight because they are the two
# axes this pipeline has never had, and the two that a "fires on everything" graph
# cannot fake.
#
# The two 0.5s are both discounted for the same reason -- they re-reward the
# selection that produced the candidate in the first place. `coherence` is the
# within-module correlation, and a module exists precisely because its pairwise
# correlations cleared the edge threshold on that slice. `hostile_frequency` is the
# firing count on the graph slice, and a feature is IN the graph only because
# `sparse_activation_matrix` keeps the top `max_features` sorted by firing count --
# so among graph features the term is compressed against its ceiling and discriminates
# little. It was 1.0 and is now 0.5; the change is a judgement call, and the
# per-candidate `terms` block in feature_modules.json lets a reader re-weight.
#
# `generalization` keeps 1.0 even though one of its two factors is a co-firing RATE:
# that rate is measured on the val slice, which the graph never saw, so it is a
# held-out check rather than a restatement of the selection rule.
DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "hostile_frequency": 0.5,
    "generalization": 1.0,
    "causal": 1.5,
    "specificity": 1.5,
    "coherence": 0.5,
}


def resolve_score_weights(overrides: Mapping[str, Any] | None) -> dict[str, float]:
    """Merge ``[module.score_weights]`` onto the defaults, rejecting bad names/values.

    ``{**DEFAULT_SCORE_WEIGHTS, **config}`` accepted anything: a config carrying
    ``casual = 3.0`` ran to completion on the default causal weight of 1.5 with no
    warning, and the operator believed they had run the causal-heavy arm on a session
    that cost a 27B load. A negative weight was equally silent -- ``score_module``
    clamped it to 0 and dropped the term out of the geometric mean. Both raise now,
    and this runs before the model is loaded so the session is not spent first.
    """

    resolved = dict(DEFAULT_SCORE_WEIGHTS)
    if not overrides:
        return resolved
    unknown = sorted(set(overrides) - set(SCORE_TERMS))
    if unknown:
        raise ValueError(
            f"unknown [module.score_weights] key(s) {unknown}; known terms are {list(SCORE_TERMS)}"
        )
    for term, value in overrides.items():
        weight = float(value)
        if weight < 0.0:
            raise ValueError(f"[module.score_weights].{term} must be >= 0, got {weight}")
        resolved[str(term)] = weight
    if sum(resolved.values()) <= 0.0:
        raise ValueError("at least one [module.score_weights] entry must be positive")
    return resolved


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else float(value))


def frequency_term(mean_active_cases: float, n_cases: int) -> float:
    """Share of hostile discovery items the module's members fire on."""

    if n_cases <= 0:
        return 0.0
    return _clamp01(float(mean_active_cases) / float(n_cases))


def coherence_term(mean_pairwise_r: float) -> float:
    """Mean within-module activation correlation, floored at zero.

    A negative mean correlation is not a weak module, it is evidence against the
    members belonging together at all, so it maps to 0 rather than to ``|r|``.
    """

    return _clamp01(mean_pairwise_r)


def generalization_term(val_coherence: float, val_cofire_rate: float) -> float:
    """Does the module survive on discovery items the graph never saw?

    A product, not an average, because the two halves are separate failure modes and
    either alone disqualifies: the members must still fire together
    (``val_cofire_rate``) *and* their magnitudes must still co-vary
    (``val_coherence``). An average would let a module that fires everywhere but no
    longer co-varies -- the exact degeneracy that broke the first 27B run -- keep
    half of its score.
    """

    return _clamp01(val_coherence) * _clamp01(val_cofire_rate)


def causal_term(mean_member_margin_delta: float, *, scale: float) -> float:
    """Saturating map of the MEAN member's single-feature margin delta.

    Mean, not max, and that is the judgement call in this score: ranking by the
    strongest member puts "one real feature plus passengers" on top, which is the
    single-feature hypothesis the module arm exists to move past. The mean rewards a
    set whose members each carry part of the effect.

    ``scale`` is the delta that scores 0.5. Its default (0.2 nats) is the mean
    held-out single-feature margin delta already measured for this model and layer
    in results/runs/20260816-203538_study_affordance_27b/artifacts/causal_heldout.csv
    (n=25, mean +0.195), so a typical single-feature effect lands mid-scale instead
    of at an arbitrary point. A negative delta (the ablation helped the lure) is not
    a weak effect in the wanted direction, so it floors at 0.
    """

    if scale <= 0.0:
        raise ValueError("causal scale must be positive")
    delta = max(0.0, float(mean_member_margin_delta))
    return delta / (delta + float(scale))


def specificity_term(hostile_activation: float, control_activation: float) -> float:
    """Normalised hostile-vs-control activation contrast, in [0, 1].

    ``(h - c) / (h + c)`` rather than a bare difference or a ratio: activation scale
    differs per feature, so an unnormalised difference would rank features by
    magnitude instead of by selectivity. Equal firing on hostile and control scores
    0, which is the right answer for a feature that tracks the surface scenario
    rather than the trap.
    """

    hostile = max(0.0, float(hostile_activation))
    control = max(0.0, float(control_activation))
    total = hostile + control
    if total <= 1e-12:
        return 0.0
    return _clamp01((hostile - control) / total)


def score_module(
    terms: Mapping[str, float],
    *,
    weights: Mapping[str, float] | None = None,
    floor: float = 1e-3,
) -> dict[str, Any]:
    """Weighted GEOMETRIC mean of the five term values, each already in [0, 1].

    Geometric, not additive, because the failure mode this job keeps hitting is a
    candidate that is enormous on one axis and dead on another: features that fire
    on 100% of hostile items *and* 100% of controls would win any weighted sum. A
    geometric mean lets a near-zero term drag the whole score down however good the
    rest is. ``floor`` stops a measured zero from annihilating the score outright,
    so candidates can still be ordered against each other below it.
    """

    if not 0.0 < floor < 1.0:
        raise ValueError("floor must be in (0, 1)")
    missing = [term for term in SCORE_TERMS if term not in terms]
    if missing:
        raise ValueError(f"module score is missing term(s): {missing}")
    resolved = dict(DEFAULT_SCORE_WEIGHTS if weights is None else weights)
    used: dict[str, float] = {}
    for term in SCORE_TERMS:
        weight = float(resolved.get(term, 0.0))
        # Raised, not clamped to 0: a negative weight used to remove a term from the
        # geometric mean without saying so anywhere in the run.
        if weight < 0.0:
            raise ValueError(f"score weight for {term!r} must be >= 0, got {weight}")
        used[term] = weight
    total_weight = sum(used.values())
    if total_weight <= 0.0:
        raise ValueError("module score weights must not all be zero")
    values = {term: _clamp01(float(terms[term])) for term in SCORE_TERMS}
    accumulated = sum(used[term] * math.log(max(values[term], floor)) for term in SCORE_TERMS)
    return {"score": math.exp(accumulated / total_weight), "terms": values, "weights": used}


def module_specificity(
    hostile_by_feature: Sequence[float], control_by_feature: Sequence[float]
) -> float:
    """Mean of the PER-FEATURE contrasts -- never the contrast of the pooled means.

    ``specificity_term`` is scale-free per feature, and pooling raw activations across
    a module's members before normalising throws that away: the pooled contrast is
    dominated by whichever member has the largest magnitude, not by which member is
    selective. Measured inversion, with the other four terms held equal at 0.5:

        module A = {40.0 hostile / 40.0 control (tracks the surface story),
                    2.0 hostile / 0.0 control (perfectly selective)}
        module B = {40.0 / 20.0 (magnitude scaling, no selectivity), 1.0 / 1.0}

        pooled       -> A 0.024, B 0.323  -> score A 0.219 < B 0.444, B wins
        per feature  -> A 0.500, B 0.167  -> score A 0.500 > B 0.371, A wins

    Pooling also lets a NEGATIVE member cancel a positive one inside the mean --
    ``qwen_scope_sparse_feature_values`` keeps negative TopK values on purpose -- which
    can push the denominator ``h + c`` toward 0 and manufacture a large clamped ratio.
    Per-feature normalisation makes the ``max(0.0, ...)`` guards inside
    ``specificity_term`` do the job they were written for.
    """

    if len(hostile_by_feature) != len(control_by_feature):
        raise ValueError("hostile_by_feature and control_by_feature must be the same length")
    if not hostile_by_feature:
        return 0.0
    return _mean(
        [
            specificity_term(hostile, control)
            for hostile, control in zip(hostile_by_feature, control_by_feature, strict=True)
        ]
    )


def mean_active_member_delta(
    member_probe: Mapping[int, Mapping[str, Any]], members: Sequence[int]
) -> tuple[float, bool]:
    """Mean single-feature margin delta over the probe items where the member FIRED.

    A feature that does not fire on a probe item contributes a *structural* zero:
    ``module_ablation_direction(sae, [f], [0.0])`` is the zero vector, so the edit is a
    no-op and ``margin_delta`` is exactly 0.0 by construction, not by measurement.
    Averaging those in ranked a sparser module below a denser one for being sparser --
    with four probe items a single absence halves the term, and ``causal`` is the
    heaviest weight in the score.

    Returns ``(mean, measured)``. ``measured`` is False when no member fired on any
    probe item, and the caller must then score the term 0 and record the absence,
    the same way an unmeasured specificity is handled -- an unmeasured axis must never
    outrank a measured one.
    """

    per_member = [
        _mean([float(delta) for delta in member_probe[int(feature)]["deltas_on_active_items"]])
        for feature in members
        if int(feature) in member_probe and member_probe[int(feature)]["deltas_on_active_items"]
    ]
    return (_mean(per_member), bool(per_member))


def modules_document(
    *,
    ranked: Sequence[Mapping[str, Any]],
    all_component_sizes: Sequence[int],
    n_candidates_scored: int,
    n_probe_items: int,
    no_module_reason: str | None = None,
) -> dict[str, Any]:
    """The ONE shape feature_modules.json is written in, on every exit path.

    It used to be a bare ``[]`` when no module was found and an object when one was,
    so a reader written against the success shape
    (``json.load(...)["ranked_candidates"]``) raised ``TypeError`` on exactly the runs
    the recorded-outcome machinery exists to make machine-readable.
    """

    return {
        "ranked_candidates": [dict(row) for row in ranked],
        "all_component_sizes": [int(size) for size in all_component_sizes],
        "n_candidates_scored": int(n_candidates_scored),
        "n_probe_items": int(n_probe_items),
        "no_module_reason": no_module_reason,
    }


def rescaled_null_direction(
    direction: torch.Tensor, target_norm: float, *, tolerance: float = 1e-9
) -> torch.Tensor | None:
    """``rescale_to_norm`` unless the result would not actually carry ``target_norm``.

    ``rescale_to_norm`` divides by ``clamp_min(norm, 1e-12)``, so a zero direction
    comes back as another ZERO vector rather than one of norm ``target``. That happens
    whenever none of a drawn random module's features land in the SAE's TopK support on
    an item: the null forward is then identical to baseline, ``margin_delta`` is
    exactly 0.0, and that fabricated 0 is averaged into ``random_module_mean`` and
    subtracted from every joint delta -- pulling the null toward zero and inflating the
    module-vs-null contrast in the direction of the hypothesis, with nothing in
    module_ablation.csv to mark it. Returns None so the caller records a skipped draw
    instead. A zero ``target_norm`` (the real module itself silent on this item) makes
    every matched null a no-op too, and is refused for the same reason.
    """

    if float(target_norm) <= tolerance or module_norm(direction) <= tolerance:
        return None
    rescaled = rescale_to_norm(direction, float(target_norm))
    return rescaled if module_norm(rescaled) > tolerance else None


def rank_modules(scored: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Order candidates by score, breaking ties by size then by member ids.

    Deterministic tie-breaks matter here: two modules that score identically have to
    be tried in the same order on every rerun, or the intervention arm stops being
    reproducible.
    """

    ordered = sorted(
        scored,
        key=lambda row: (-float(row["score"]), -len(row["features"]), list(row["features"])),
    )
    return [{**row, "rank": index} for index, row in enumerate(ordered, start=1)]


# ------------------------------------------------------------------- setup


def _resolve_env(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = table(config, "model")
    profile = get_qwen35_analysis_profile(
        str(model_cfg.get("profile", DEFAULT_ANALYSIS_PROFILE_KEY))
    )
    dtype = model_cfg.get("dtype", "auto")
    dtype = recommended_dtype_name() if dtype == "auto" else str(dtype)
    sae_device = model_cfg.get("sae_device", "auto")
    sae_device = default_sae_device() if sae_device == "auto" else str(sae_device)
    sae_dtype = model_cfg.get("sae_dtype", "auto")
    sae_dtype = dtype if sae_dtype == "auto" else str(sae_dtype)
    return {
        "profile": profile,
        "model_id": profile.analysis_model_id,
        "repo_id": profile.sae_repo_id,
        "dtype": dtype,
        "device_map": model_cfg.get("device_map", "auto"),
        "sae_device": sae_device,
        "sae_dtype": sae_dtype,
    }


def _load_splits(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = table(config, "data")
    dataset = str(data_cfg.get("dataset", "goal_affordance_traps_v1"))
    all_cases = lure_dataset_cases(dataset)
    conditions = data_cfg.get("conditions") or None
    cases = list(all_cases)
    if conditions:
        suffixes = tuple(f"_{name}" for name in conditions)
        cases = [case for case in cases if case.case_id.endswith(suffixes)]
    if bool(data_cfg.get("instruction", True)):
        cases = instruct_lure_cases(cases)
        all_cases = instruct_lure_cases(all_cases)
    split_seed = int(data_cfg.get("split_seed", 0))
    train, test = split_lure_cases(
        cases, train_frac=float(data_cfg.get("train_frac", 0.6)), seed=split_seed
    )
    # The discovery split is cut a second time so the module SCORE never reads the
    # items that built the graph. It must use a different seed: split_lure_cases
    # buckets on a stable hash of (seed, case_id), so re-splitting `train` under the
    # same seed puts every case back on the train side and leaves `val` empty.
    graph_seed = int(data_cfg.get("graph_seed", 101))
    if graph_seed == split_seed:
        raise ValueError("[data].graph_seed must differ from [data].split_seed (see comment)")
    graph, val = split_lure_cases(
        train, train_frac=float(data_cfg.get("graph_frac", 0.7)), seed=graph_seed
    )
    # Checked here, before the model is loaded, because a misconfigured split that
    # only surfaces after a 27B is resident costs a whole session.
    if not graph or not val:
        raise ValueError(
            f"graph/val split produced {len(graph)}/{len(val)} items; "
            "widen the discovery split or move graph_frac"
        )
    max_test = int(data_cfg.get("max_test_items", 0))
    if max_test:
        test = family_balanced_subset(test, max_cases=max_test)
    if not test:
        raise ValueError("the held-out split is empty; widen train_frac or max_test_items")

    # Matched control counterparts, looked up by pair_id. Only their ACTIVATIONS are
    # ever read: the counterfactual condition swaps which answer is correct, so its
    # margins are not on the same scale as the hostile ones and must not be pooled.
    control_condition = str(data_cfg.get("control_condition", "") or "")
    controls: dict[str, Any] = {}
    if control_condition:
        controls = {
            case.pair_id: case
            for case in all_cases
            if case.condition == control_condition and case.pair_id
        }
    return {
        "dataset": dataset,
        "graph": graph,
        "val": val,
        "test": test,
        "controls": controls,
        "control_condition": control_condition,
        "n_train": len(train),
    }


def _margin_row(margin: Any, baseline: Any) -> dict[str, float]:
    return {
        "margin_delta": float(baseline.margin) - float(margin.margin),
        # A direction that merely damages the model drags both logprobs down; a lure
        # effect moves mostly the lure. Nearly free to record, and it settles that
        # question without another run.
        # Sign convention is effects.py's, NOT margin_delta's: the logprob deltas are
        # ablated - baseline, so `correct > 0` means the edit RAISED the correct answer
        # and `lure < 0` means it LOWERED the lure. margin_delta stays baseline - ablated.
        # docs/metrics_guide.md documents this pair; writing them the other way round
        # silently inverts every reader's conclusion.
        "correct_logprob_delta": float(margin.correct.logprob) - float(baseline.correct.logprob),
        "lure_logprob_delta": float(margin.lure.logprob) - float(baseline.lure.logprob),
    }


def _module_sites(layer: int, direction: torch.Tensor) -> list[EditSite]:
    """Edit sites for one module ablation: one site per layer the module occupies.

    A same-layer module collapses to a single site, because ``remove_activation`` is
    linear in the feature and ``sum_f a_f * W_dec[f]`` is therefore one vector -- so
    this is numerically identical to the single-direction call it replaces (both end
    in the same ``_direction_edit`` at the same ``start - 1`` token index). Routing
    through EditSite / multi_site_answer_margin anyway buys the one thing the
    single-direction call cannot express: a module whose members live at DIFFERENT
    layers, which nothing in this study has ruled out. When that arrives only this
    function and the direction bookkeeping change, not the four conditions below.
    """

    return [EditSite(int(layer), direction, 1.0, -1.0, "add_vector")]


def _feature_values(residual: torch.Tensor, sae: Any, feature_ids: Sequence[int]) -> list[float]:
    return (
        qwen_scope_sparse_feature_values(residual, sae, list(feature_ids))
        .detach()
        .to(torch.float32)
        .reshape(-1)
        .tolist()
    )


def _feature_value_matrix(
    lm: Any,
    cases: Sequence[Any],
    *,
    layer: int,
    sae: Any,
    feature_ids: Sequence[int],
) -> torch.Tensor:
    """``(cases x feature_ids)`` activations for a FIXED feature set.

    ``sparse_activation_matrix`` picks its own features by firing frequency, which is
    right for building the graph and wrong here -- the score has to read the *same*
    features on the val items and on their controls, or the two matrices are not
    comparable column by column.
    """

    rows = [
        torch.tensor(
            _feature_values(
                capture_layer_residuals(lm, [case.prompt], int(layer), token_position="last"),
                sae,
                feature_ids,
            )
        )
        for case in cases
    ]
    return torch.stack(rows) if rows else torch.zeros(0, len(feature_ids))


def _cofire_rate(matrix: torch.Tensor, feature_ids: Sequence[int], module: Sequence[int]) -> float:
    """Share of items on which EVERY member of the module fires."""

    index = {int(f): i for i, f in enumerate(feature_ids)}
    columns = [index[int(f)] for f in module if int(f) in index]
    if not columns or matrix.numel() == 0:
        return 0.0
    return float((matrix[:, columns] > 0).all(dim=1).to(torch.float32).mean())


# ------------------------------------------------------------------- runner


def run(config_path: Path, output_root: Path) -> Path:
    config = load_toml(config_path)
    name = run_name(config)
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config.toml")

    mcfg = table(config, "module")
    layer = int(mcfg["layer"])
    min_active_cases = int(mcfg.get("min_active_cases", 3))
    max_features = int(mcfg.get("max_features", 50))
    edge_threshold = float(mcfg.get("edge_threshold", 0.55))
    metric = str(mcfg.get("metric", "activation_corr"))
    max_active_frac = float(mcfg.get("max_active_frac", 1.0))
    sweep_thresholds = [float(t) for t in mcfg.get("sweep_thresholds", [])]
    permutation_null_draws = int(mcfg.get("permutation_null_draws", 0))
    min_size = int(mcfg.get("min_size", 2))
    max_size = int(mcfg.get("max_size", 12))
    max_modules = int(mcfg.get("max_modules", 2))
    max_score_candidates = int(mcfg.get("max_score_candidates", 6))
    score_probe_items = int(mcfg.get("score_probe_items", 4))
    causal_scale = float(mcfg.get("causal_scale", 0.2))
    # Validated here, before the model is loaded: a mistyped weight name used to run
    # to completion on the default and cost a whole 27B session to discover.
    score_weights = resolve_score_weights(table(mcfg, "score_weights"))
    fail_on_empty = bool(mcfg.get("fail_on_empty", False))
    random_modules = int(mcfg.get("random_modules", 10))
    seed = int(mcfg.get("seed", 0))

    env = _resolve_env(config)
    splits = _load_splits(config)
    manifest: dict[str, Any] = {
        "run_name": name,
        "job": "feature_modules",
        "started_at": _timestamp(),
        "status": "running",
        "layer": layer,
        "profile": env["profile"].key,
        "model_id": env["model_id"],
        "sae_repo_id": env["repo_id"],
        "dataset": splits["dataset"],
        "n_train": splits["n_train"],
        "n_graph": len(splits["graph"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "control_condition": splits["control_condition"],
        "module_config": dict(mcfg),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    _write_json(run_dir / "manifest.json", manifest)

    _log(f"loading {env['model_id']} (layer {layer})")
    lm = load_qwen_language_model(
        env["model_id"], device_map=env["device_map"], dtype=env["dtype"], dispatch=True
    )
    sae = load_qwen_scope_sae(
        env["repo_id"], layer, device=env["sae_device"], dtype=dtype_from_name(env["sae_dtype"])
    )

    started = time.time()

    def _finish(status: str, extra: dict[str, Any]) -> None:
        manifest.update(
            {
                "status": status,
                "finished_at": _timestamp(),
                "elapsed_seconds": round(time.time() - started, 1),
                **extra,
            }
        )
        _write_json(run_dir / "manifest.json", manifest)
        print(f"ARTIFACT_DIR={run_dir}", flush=True)

    # Sections 1-3 all run with the 27B and its SAE resident. Nothing below may
    # leave the manifest stamped `running`: a crash there costs a whole Colab
    # session and used to leave no ARTIFACT_DIR line, no module_search.json and a
    # status field that could not tell a crash from a run still in flight.
    try:
        # --- 1. coactivation graph on the GRAPH slice ---------------------------
        _log(f"collecting sparse activations on {len(splits['graph'])} graph items")
        all_feature_ids, all_matrix = sparse_activation_matrix(
            lm,
            splits["graph"],
            layer=layer,
            sae=sae,
            min_active_cases=min_active_cases,
            max_features=max_features,
        )
        if not all_feature_ids:
            raise _NoModulesFound(
                f"no feature fired on {min_active_cases}+ of the {len(splits['graph'])} graph "
                "items, so there is no coactivation graph to search",
                code="no_features_in_graph",
                detail={"n_features_in_graph": 0, "n_edges": 0, "n_always_on_candidates": 0},
            )
        all_counts = [int((all_matrix[:, i] > 0).sum()) for i in range(len(all_feature_ids))]
        _, dropped = graph_feature_selection(
            all_feature_ids, all_counts, len(splits["graph"]), max_active_frac=max_active_frac
        )
        banned = set(dropped)
        keep_columns = [i for i, f in enumerate(all_feature_ids) if int(f) not in banned]
        feature_ids = [all_feature_ids[i] for i in keep_columns]
        matrix = all_matrix[:, keep_columns]
        counts = [all_counts[i] for i in keep_columns]
        n_always_on = sum(1 for count in all_counts if count >= len(splits["graph"]))
        if dropped:
            _log(f"dropped {len(dropped)} features firing above {max_active_frac:.2f} of the items")
        if len(feature_ids) < 2:
            raise _NoModulesFound(
                f"{len(feature_ids)} feature(s) survived max_active_frac={max_active_frac:.2f} "
                f"(of {len(all_feature_ids)} candidates); a graph needs at least two",
                code="graph_too_small",
                detail={
                    "n_features_in_graph": len(feature_ids),
                    "n_edges": 0,
                    "n_always_on_candidates": n_always_on,
                    "dropped_features": dropped,
                },
            )

        edges = coactivation_edges(matrix, feature_ids)
        _write_csv(
            run_dir / "coactivation_edges.csv",
            edges,
            ["feature_a", "feature_b", "co_fire", "jaccard", "activation_corr"],
        )
        _log(
            f"graph: {len(feature_ids)} features, {len(edges)} pairs, "
            f"{n_always_on}/{len(all_feature_ids)} candidates fire on every graph item"
        )

        # Written BEFORE the search is allowed to fail, so a run that finds nothing still
        # hands the next run the threshold that would have worked.
        sweep = threshold_sweep(
            edges,
            metric=metric,
            thresholds=sorted({*sweep_thresholds, edge_threshold}),
            min_size=min_size,
            max_size=max_size,
            matrix=matrix,
            null_draws=permutation_null_draws,
            seed=seed,
        )
        _write_json(
            run_dir / "module_search.json",
            {
                "metric": metric,
                "edge_threshold": edge_threshold,
                "min_size": min_size,
                "max_size": max_size,
                "n_features": len(feature_ids),
                "n_edges": len(edges),
                "n_always_on_candidates": n_always_on,
                "dropped_features": dropped,
                "permutation_null_draws": permutation_null_draws,
                # The two null-shaped columns in `sweep` are not interchangeable, and the
                # artifact says which is which rather than leaving it to the key names.
                "caveats": {
                    "negative_tail_ratio": NEGATIVE_TAIL_RATIO_CAVEAT,
                    "permutation_p": (
                        "P(a column-permuted graph keeps at least as many edges as this run "
                        "did) under (b+1)/(draws+1). Each feature's own activation "
                        "distribution is preserved and only the cross-feature alignment is "
                        "destroyed, so a p near 1 means the surviving edges are forced by "
                        "the marginals -- which is what a graph of always-on features "
                        "produces. null when permutation_null_draws = 0."
                    ),
                    "edge_threshold": (
                        "edge_threshold is NOT the argmin of any column in this table, and "
                        "was chosen as one when this key was called symmetric_null_fdr. On "
                        "the measured 27B edges 0.50, 0.55, 0.60 and 0.65 every one yields "
                        "exactly 3 components inside min_size..max_size, and their "
                        "negative_tail_ratios (0.372, 0.344, 0.419, 0.367) all sit inside "
                        "one another's sampling error (+/-0.066 to +/-0.111). The shipped "
                        "value is a preference inside a band the data does not resolve. "
                        "permutation_p is the column that can actually justify one; set "
                        "permutation_null_draws and pick a threshold at a stated level."
                    ),
                },
                "sweep": sweep,
            },
        )

        found = modules_from_edges(
            edges,
            edge_threshold=edge_threshold,
            metric=metric,
            min_size=min_size,
            max_size=max_size,
        )
        if not found:
            raise _NoModulesFound(
                f"no module of size {min_size}-{max_size} survived {metric} >= {edge_threshold}; "
                "the features that fire here do not group",
                code="no_component_in_size_range",
                detail={
                    "n_features_in_graph": len(feature_ids),
                    "n_edges": len(edges),
                    "n_always_on_candidates": n_always_on,
                },
            )

        # --- 2. score the candidates on the VAL slice ---------------------------
        candidates = found[:max_score_candidates]
        union = sorted({int(f) for module in candidates for f in module})
        _log(f"scoring {len(candidates)} candidate modules over {len(union)} distinct features")

        val_matrix = _feature_value_matrix(
            lm, splits["val"], layer=layer, sae=sae, feature_ids=union
        )
        # Paired, not pooled: the hostile side of the specificity contrast is restricted to
        # the val items that actually have a control counterpart, so the two means are over
        # the same scenarios and the contrast is not confounded by which items got matched.
        paired = [
            index
            for index, case in enumerate(splits["val"])
            if case.pair_id and case.pair_id in splits["controls"]
        ]
        control_cases = [splits["controls"][splits["val"][index].pair_id] for index in paired]
        control_matrix = _feature_value_matrix(
            lm, control_cases, layer=layer, sae=sae, feature_ids=union
        )
        if not paired:
            # No control items means the specificity term has no evidence behind it, and
            # specificity_term(h, 0.0) would read as PERFECT selectivity -- the exact
            # opposite of the truth. It is forced to 0 instead. Every module in the run
            # takes the same penalty, so the ranking within the run is unchanged, while the
            # absolute score stays honestly low against properly controlled runs.
            _log("no matched control items; specificity scores 0 for every module")

        # Single-feature causal probe on VAL items, never on the held-out items the
        # interventions use: choosing a module on the same margins that later judge it
        # would stop the held-out arm from being held out.
        probe_cases = (
            family_balanced_subset(splits["val"], max_cases=score_probe_items)
            if score_probe_items > 0
            else []
        )
        probe_rows: list[dict[str, Any]] = []
        member_probe: dict[int, dict[str, Any]] = {
            int(feature_id): {
                "feature": int(feature_id),
                "n_probe_items": 0,
                "n_active_probe_items": 0,
                "deltas_on_active_items": [],
            }
            for feature_id in union
        }
        for probe_index, case in enumerate(probe_cases, start=1):
            residual = capture_layer_residuals(lm, [case.prompt], layer, token_position="last")
            values = _feature_values(residual, sae, union)
            baseline = answer_logprob_margin(
                lm, case.prompt, correct_answer=case.correct_answer, lure_answer=case.lure_answer
            )
            for feature_id, value in zip(union, values, strict=True):
                margin = multi_site_answer_margin(
                    lm,
                    case.prompt,
                    correct_answer=case.correct_answer,
                    lure_answer=case.lure_answer,
                    sites=_module_sites(
                        layer, module_ablation_direction(sae, [feature_id], [value])
                    ),
                )
                # `fired` separates a measured zero from a structural one. Outside the
                # SAE's TopK support the value is 0, the ablation direction is the zero
                # vector, and margin_delta comes back as exactly 0.0 because the edit did
                # nothing -- the row is real, but it is not evidence about the feature.
                fired = float(value) > 0.0
                row = {
                    "case_id": case.case_id,
                    "family": case.family,
                    "feature": int(feature_id),
                    "feature_value": float(value),
                    "fired": int(fired),
                    **_margin_row(margin, baseline),
                }
                probe_rows.append(row)
                probe = member_probe[int(feature_id)]
                probe["n_probe_items"] += 1
                if fired:
                    probe["n_active_probe_items"] += 1
                    probe["deltas_on_active_items"].append(float(row["margin_delta"]))
            _log(f"causal probe: {probe_index}/{len(probe_cases)} val items")
        _write_csv(
            run_dir / "feature_causal_probe.csv",
            probe_rows,
            [
                "case_id",
                "family",
                "feature",
                "feature_value",
                "fired",
                "margin_delta",
                "correct_logprob_delta",
                "lure_logprob_delta",
            ],
        )

        silent_members = [
            feature_id
            for feature_id, probe in member_probe.items()
            if probe["n_probe_items"] and not probe["n_active_probe_items"]
        ]
        if silent_members:
            _log(
                f"causal probe: {len(silent_members)}/{len(union)} features never fired on a "
                "probe item; their ablations were structural no-ops and are excluded from "
                "the causal term (see raw.member_causal)"
            )

        scored: list[dict[str, Any]] = []
        for module in candidates:
            members = [int(f) for f in module]
            columns = [union.index(f) for f in members]
            mean_active = _mean(
                [float(counts[feature_ids.index(f)]) for f in members if f in feature_ids]
            )
            graph_coherence = module_coherence(matrix, feature_ids, members)
            val_coherence = module_coherence(val_matrix, union, members)
            val_cofire = _cofire_rate(val_matrix, union, members)
            mean_member_delta, causal_measured = mean_active_member_delta(member_probe, members)
            # Per feature, then averaged -- see module_specificity for the inversion that
            # pooling the raw activations across the module's columns produces.
            member_contrasts: list[dict[str, Any]] = []
            if paired and control_matrix.numel():
                for feature_id, column in zip(members, columns, strict=True):
                    hostile = float(val_matrix[paired][:, column].mean())
                    control = float(control_matrix[:, column].mean())
                    member_contrasts.append(
                        {
                            "feature": int(feature_id),
                            "hostile_activation": hostile,
                            "control_activation": control,
                            "specificity": specificity_term(hostile, control),
                        }
                    )
            result = score_module(
                {
                    "hostile_frequency": frequency_term(mean_active, len(splits["graph"])),
                    "generalization": generalization_term(val_coherence, val_cofire),
                    # An unmeasured causal axis scores 0 rather than defaulting to the
                    # structural zeros, the same rule specificity follows: never let "not
                    # measured" outrank a real measurement.
                    "causal": (
                        causal_term(mean_member_delta, scale=causal_scale)
                        if causal_measured
                        else 0.0
                    ),
                    "specificity": module_specificity(
                        [contrast["hostile_activation"] for contrast in member_contrasts],
                        [contrast["control_activation"] for contrast in member_contrasts],
                    ),
                    "coherence": coherence_term(graph_coherence),
                },
                weights=score_weights,
            )
            scored.append(
                {
                    "features": members,
                    "size": len(members),
                    "score": result["score"],
                    "terms": result["terms"],
                    "weights": result["weights"],
                    # The raw measurements behind every term, so a reader can rebuild the
                    # score without rerunning the model or trusting the mapping functions.
                    "raw": {
                        "mean_active_cases": mean_active,
                        "n_graph_items": len(splits["graph"]),
                        "graph_coherence": graph_coherence,
                        "val_coherence": val_coherence,
                        "val_cofire_rate": val_cofire,
                        "mean_member_margin_delta_on_active_items": mean_member_delta,
                        "causal_measured": causal_measured,
                        "causal_scale": causal_scale,
                        "member_causal": [
                            {
                                key: member_probe[f][key]
                                for key in ("feature", "n_probe_items", "n_active_probe_items")
                            }
                            for f in members
                            if f in member_probe
                        ],
                        # Per-feature, because the pooled hostile/control means this used to
                        # record were the defect: they let the largest-magnitude member
                        # decide the term. Kept in the artifact so the choice is auditable.
                        "member_specificity": member_contrasts,
                        "n_control_items": len(control_cases),
                    },
                }
            )

        ranked = rank_modules(scored)
        _write_json(
            run_dir / "feature_modules.json",
            modules_document(
                ranked=ranked,
                all_component_sizes=[len(module) for module in found],
                n_candidates_scored=len(candidates),
                n_probe_items=len(probe_cases),
            ),
        )
        modules = [row["features"] for row in ranked[:max_modules]]
        _log(
            "ranking: " + "; ".join(f"{row['features']} score={row['score']:.3f}" for row in ranked)
        )

        # --- 3. held-out interventions -----------------------------------------
        rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []

        for module_index, module in enumerate(modules, start=1):
            randoms = sample_frequency_matched_modules(
                feature_ids,
                counts,
                size=len(module),
                exclude=module,
                n_modules=random_modules,
                seed=seed,
            )
            _log(
                f"module {module_index}: {len(module)} features, "
                f"{len(randoms)} frequency-matched null modules"
            )
            per_condition: dict[str, list[float]] = {}
            joint_minus_sum: list[float] = []
            skipped_null_draws = 0
            silent_module_items = 0

            for item_index, case in enumerate(splits["test"], start=1):
                residual = capture_layer_residuals(lm, [case.prompt], layer, token_position="last")
                values = _feature_values(residual, sae, module)
                baseline = answer_logprob_margin(
                    lm,
                    case.prompt,
                    correct_answer=case.correct_answer,
                    lure_answer=case.lure_answer,
                )
                common = {"case_id": case.case_id, "family": case.family, "module": module_index}

                def _record(condition: str, direction: torch.Tensor, draw: int = -1) -> float:
                    margin = multi_site_answer_margin(
                        lm,
                        case.prompt,
                        correct_answer=case.correct_answer,
                        lure_answer=case.lure_answer,
                        sites=_module_sites(layer, direction),
                    )
                    row = {
                        **common,
                        "condition": condition,
                        "draw": draw,
                        **_margin_row(margin, baseline),
                    }
                    rows.append(row)
                    per_condition.setdefault(condition, []).append(row["margin_delta"])
                    return row["margin_delta"]

                joint_direction = module_ablation_direction(sae, module, values)
                target_norm = module_norm(joint_direction)
                if target_norm <= 1e-9:
                    # No member is in the SAE's TopK support here, so module_joint is a
                    # genuine measured zero -- and every norm-matched null is a no-op by
                    # construction, which is not. Counted, and the nulls are skipped below.
                    silent_module_items += 1
                joint = _record("module_joint", joint_direction)

                member_margin_deltas = []
                for feature_id, value in zip(module, values, strict=True):
                    member_margin_deltas.append(
                        _record(
                            f"member_{feature_id}",
                            module_ablation_direction(sae, [feature_id], [value]),
                        )
                    )
                # Strongest member by its own removed norm, ablated alone: the
                # single-feature arm the module has to beat.
                strongest = max(
                    range(len(module)),
                    key=lambda i: module_norm(
                        module_ablation_direction(sae, [module[i]], [values[i]])
                    ),
                )
                _record(
                    "single_best",
                    module_ablation_direction(sae, [module[strongest]], [values[strongest]]),
                )
                joint_minus_sum.append(joint - sum(member_margin_deltas))

                for draw, random_module in enumerate(randoms):
                    random_values = _feature_values(residual, sae, random_module)
                    random_direction = module_ablation_direction(sae, random_module, random_values)
                    # Same size, matched firing frequency, and the same removed norm, so
                    # only the identity of the features differs from the real module -- but
                    # only when the rescale can actually deliver that norm. A dead draw
                    # would record an exact 0.0 that never measured anything and drag the
                    # null toward the hypothesis, so it is skipped and counted instead.
                    rescaled = rescaled_null_direction(random_direction, target_norm)
                    if rescaled is None:
                        skipped_null_draws += 1
                        continue
                    _record("random_module", rescaled, draw)

                if item_index % 5 == 0:
                    _log(f"module {module_index}: {item_index}/{len(splits['test'])} items")
                _write_csv(
                    run_dir / "module_ablation.csv",
                    rows,
                    [
                        "case_id",
                        "family",
                        "module",
                        "condition",
                        "draw",
                        "margin_delta",
                        "correct_logprob_delta",
                        "lure_logprob_delta",
                    ],
                )

            random_deltas = per_condition.get("random_module", [])
            selected = ranked[module_index - 1]
            # With a single candidate there was no argmax to bias anything, and saying
            # otherwise would overstate the caveat exactly as the old comment
            # overstated the comparison.
            selection_caveat = (
                "joint_minus_random is norm-matched, not selection-matched. This module "
                f"is the argmax over {len(candidates)} candidates of a score whose "
                "heaviest term (causal, weight "
                f"{selected['weights']['causal']}) is measured from real single-feature "
                "margin deltas on the val slice, while the random modules are drawn on "
                "size and firing frequency alone. Any per-feature causal potency that is "
                "stable across items therefore transfers to the test items and biases "
                "this contrast upward even under a true null of 'no module mediates "
                "anything'. Read it as a descriptive gap between the selected module and "
                "norm-matched peers, not as a p-value for 'this module mediates the "
                "lure'."
                if len(candidates) > 1
                else "only one candidate module was scored, so no score-based selection "
                "took place among candidates and joint_minus_random carries no argmax "
                "bias from it. It is still not a test that a module mediates the lure: "
                "the candidate is whatever the edge threshold produced, and the nulls "
                "are matched on size, firing frequency and removed norm only."
            )
            summary = {
                "module": module_index,
                "features": module,
                "size": len(module),
                "score": selected["score"],
                "score_terms": selected["terms"],
                "score_weights": selected["weights"],
                "coherence": module_coherence(matrix, feature_ids, module),
                "n_random_modules": len(randoms),
                "joint": paired_summary(per_condition.get("module_joint", []), seed=seed),
                "single_best": paired_summary(per_condition.get("single_best", []), seed=seed),
                # None, not 0.0: when every drawn null failed the norm check there is
                # no null distribution, and a 0.0 here reads as "the null does nothing"
                # -- which would make the module look maximally causal on no evidence.
                "random_module_mean": mean_or_none(random_deltas),
                # NOT "the honest comparison" -- that comment was written when the
                # intervened module was the LARGEST component, a criterion uncorrelated with
                # causal effect. It is now the argmax of a score whose heaviest term is
                # measured from real single-feature margin deltas, so the selection and the
                # null no longer control for the same thing. `random_module_null` below
                # states what is and is not matched; do not restate this as a test that the
                # module beats its null.
                "joint_minus_random": paired_summary(
                    [
                        delta - _mean(random_deltas)
                        for delta in per_condition.get("module_joint", [])
                    ],
                    seed=seed,
                ),
                "random_module_null": {
                    "matched_on": [
                        "module size",
                        "graph firing count (tolerance 1)",
                        "removed norm",
                    ],
                    "not_matched_on": ["the module score that selected this module"],
                    "selection_rule": (
                        "argmax of the 5-term module score over the scored candidates"
                    ),
                    "n_candidates_scored": len(candidates),
                    "selected_score": selected["score"],
                    "selected_causal_term": selected["terms"]["causal"],
                    "causal_weight": selected["weights"]["causal"],
                    "selection_over_candidates": len(candidates) > 1,
                    "n_skipped_null_draws": skipped_null_draws,
                    "n_silent_module_items": silent_module_items,
                    "caveat": selection_caveat,
                    "n_skipped_null_draws_note": (
                        "draws whose direction could not carry the module's removed norm "
                        "(no member in the SAE TopK support on that item). Skipped rather "
                        "than recorded as a 0.0 the forward never measured."
                    ),
                },
                "mean_joint_minus_sum_of_members": _mean(joint_minus_sum),
                "decomposition": {
                    "joint_correct_delta": _mean(
                        [
                            float(row["correct_logprob_delta"])
                            for row in rows
                            if row["module"] == module_index and row["condition"] == "module_joint"
                        ]
                    ),
                    "joint_lure_delta": _mean(
                        [
                            float(row["lure_logprob_delta"])
                            for row in rows
                            if row["module"] == module_index and row["condition"] == "module_joint"
                        ]
                    ),
                },
            }
            summaries.append(summary)
            if skipped_null_draws or silent_module_items:
                _log(
                    f"module {module_index}: skipped {skipped_null_draws} null draw(s) that "
                    f"could not carry the module norm, on {silent_module_items} item(s) where "
                    "the module itself was silent"
                )
            _log(
                f"module {module_index}: joint {summary['joint']['mean']:+.4f} "
                f"(p={summary['joint']['p']}) vs random {summary['random_module_mean']:+.4f} "
                f"[norm-matched null only; see random_module_null.caveat]"
            )
            _write_json(run_dir / "module_summary.json", summaries)

        _finish(
            "ok",
            {
                "n_features_in_graph": len(feature_ids),
                "n_edges": len(edges),
                "n_always_on_candidates": n_always_on,
                "module_sizes": [len(m) for m in found],
                "ranked_candidates": ranked,
                "summaries": summaries,
            },
        )
    except _NoModulesFound as recorded:
        # A recorded outcome, not an error: 'nothing groups at this threshold' is a
        # result the study can publish. Every artifact the success path writes is
        # written here too, in the SAME shape, so a reader that already opened
        # feature_modules.json does not hit a TypeError on exactly the runs this
        # machinery exists to make machine-readable. A suite that genuinely wants a
        # non-zero exit sets [module].fail_on_empty, which raises after the writes.
        search_path = run_dir / "module_search.json"
        if not search_path.exists():
            _write_json(
                search_path,
                {
                    "metric": metric,
                    "edge_threshold": edge_threshold,
                    "min_size": min_size,
                    "max_size": max_size,
                    "sweep": [],
                    "no_module_reason": recorded.code,
                    "reason": str(recorded),
                    **recorded.detail,
                },
            )
        _write_json(
            run_dir / "feature_modules.json",
            modules_document(
                ranked=[],
                all_component_sizes=[],
                n_candidates_scored=0,
                n_probe_items=0,
                no_module_reason=recorded.code,
            ),
        )
        _write_json(run_dir / "module_summary.json", [])
        _log(str(recorded))
        _log("module_search.json lists the thresholds that would have produced modules")
        print("NO_MODULE_FOUND", flush=True)
        _finish(
            "no_module_found",
            {
                "reason": str(recorded),
                "no_module_reason": recorded.code,
                "module_sizes": [],
                "summaries": [],
                **recorded.detail,
            },
        )
        if fail_on_empty:
            raise RuntimeError(str(recorded)) from recorded
        return run_dir
    except Exception as exc:
        # Re-raised unchanged -- the point is only that the manifest stops lying.
        # `status` is a reliable tell of a crash now, and ARTIFACT_DIR still prints,
        # so the partial artifacts are collected instead of lost with the session.
        _finish(
            "failed",
            {"reason": f"{type(exc).__name__}: {exc}", "error_type": type(exc).__name__},
        )
        raise

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("outputs/experiments"), type=Path)
    args = parser.parse_args()
    run(args.config, args.output_root)


if __name__ == "__main__":
    main()
