"""Degeneracy screen for one pinned SAE feature: is its "causal" effect an artifact?

A feature can move the lure margin for reasons that have nothing to do with the
hypothesised computation. It may be positional (fires at the answer template token
whatever the content), dense (bias-like), token-identity (its decoder points at the
answer strings' unembeddings, so ablation moves the margin mechanically), or simply
the winner of a null that was too weak to reject anything. This job runs the checks
that separate those explanations from a genuine concept feature:

    logit_lens   decoder -> unembedding: which tokens it promotes, and how aligned it
                 is with each item's lure-minus-correct unembedding difference,
                 scored against random directions and against other features
    positions    activation across every token position on trap prompts AND on
                 content-unrelated reference prompts sharing the same answer template
                 (+ TopK firing density and the ablation's size relative to the residual)
    null         observed mean ablation delta on a fixed panel vs (a) matched-norm
                 Gaussian directions and (b) matched-norm decoder directions of other
                 features that fire at the same site, reported as empirical percentiles,
                 plus a selection-aware max-of-k correction for the winner's curse
    siblings     decoder-cosine neighbours at this layer and the best-matching partner
                 at the other scanned layers (cross-layer smearing)

Results are written as they are produced, so a killed run keeps what it finished.
"""
# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from collections.abc import Sequence
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
    DEFAULT_ANALYSIS_PROFILE_KEY,
    answer_logprob_margin,
    capture_layer_residuals,
    clear_device_cache,
    default_sae_device,
    dtype_from_name,
    encode_qwen_scope_topk,
    family_balanced_subset,
    get_qwen35_analysis_profile,
    instruct_lure_cases,
    load_qwen_language_model,
    load_qwen_scope_sae,
    lure_dataset_cases,
    qwen_scope_feature_preactivations,
    qwen_scope_sparse_feature_values,
    recommended_dtype_name,
    sae_decoder_direction,
    split_lure_cases,
)
from mindscopex_analysis.activations import get_module


def _log(message: str) -> None:
    print(f"[diag] {message}", flush=True)


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


def _load_sae(env: dict[str, Any], layer: int) -> Any:
    return load_qwen_scope_sae(
        env["repo_id"],
        int(layer),
        device=env["sae_device"],
        dtype=dtype_from_name(env["sae_dtype"]),
    )


def _load_data(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = table(config, "data")
    dataset = str(data_cfg.get("dataset", "goal_affordance_traps_v1"))
    instruction = bool(data_cfg.get("instruction", True))
    cases = lure_dataset_cases(dataset)
    conditions = data_cfg.get("conditions") or None
    if conditions:
        suffixes = tuple(f"_{name}" for name in conditions)
        cases = [case for case in cases if case.case_id.endswith(suffixes)]
    if not cases:
        raise ValueError(f"no {dataset!r} cases left after filtering conditions={conditions}")
    if instruction:
        cases = instruct_lure_cases(cases)
    train, test = split_lure_cases(
        cases,
        train_frac=float(data_cfg.get("train_frac", 0.6)),
        seed=int(data_cfg.get("split_seed", 0)),
    )

    ref_cfg = table(config, "reference")
    ref_dataset = str(ref_cfg.get("dataset", "hagendorff_crt"))
    ref_limit = ref_cfg.get("limit_per_family", 10)
    reference = lure_dataset_cases(
        ref_dataset, limit_per_family=int(ref_limit) if ref_limit else None
    )
    if instruction:
        reference = instruct_lure_cases(reference)
    return {
        "dataset": dataset,
        "cases": cases,
        "train": train,
        "test": test,
        "reference_dataset": ref_dataset,
        "reference": reference,
    }


def _decoder_matrix(sae: Any) -> torch.Tensor:
    """Return the decoder as (d_sae, d_model) whatever the checkpoint's orientation."""

    return sae.W_dec.T if sae.W_dec.shape[0] == sae.d_model else sae.W_dec


def _resolve(lm: Any, paths: Sequence[str]) -> Any | None:
    for path in paths:
        try:
            return get_module(lm, path)
        except (AttributeError, IndexError, TypeError):
            continue
    return None


def _first_token_id(tokenizer: Any, answer: str) -> int | None:
    ids = tokenizer.encode(answer, add_special_tokens=False)
    return int(ids[0]) if ids else None


def _answer_pairs(lm: Any, cases: Sequence[Any]) -> list[tuple[str, int, int]]:
    """(case_id, lure_first_token, correct_first_token) for usable answer pairs."""

    tokenizer = lm.tokenizer
    pairs: list[tuple[str, int, int]] = []
    for case in cases:
        lure = _first_token_id(tokenizer, case.lure_answer)
        correct = _first_token_id(tokenizer, case.correct_answer)
        if lure is not None and correct is not None and lure != correct:
            pairs.append((case.case_id, lure, correct))
    return pairs


# -------------------------------------------------------------- check: lens


def check_logit_lens(
    lm: Any,
    sae: Any,
    data: dict[str, Any],
    feature_id: int,
    run_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Does the decoder push the answer tokens directly, rather than any concept?"""

    cfg = table(config, "logit_lens")
    top_n = int(cfg.get("top_n", 30))
    n_reference = int(cfg.get("reference_directions", 500))

    head = _resolve(lm, ("lm_head", "model.lm_head", "model.language_model.lm_head"))
    if head is None or not hasattr(head, "weight"):
        _log("logit_lens: no unembedding found; skipping")
        return {"skipped": "unembedding not found"}
    w_u = head.weight.detach()

    norm_module = _resolve(
        lm, ("model.language_model.norm", "model.norm", "language_model.norm", "model.model.norm")
    )
    gamma = getattr(norm_module, "weight", None)

    direction = sae_decoder_direction(sae, [int(feature_id)]).detach()
    lens_vec = direction.to(w_u.device, dtype=torch.float32)
    if gamma is not None:
        lens_vec = lens_vec * gamma.detach().to(lens_vec.device, dtype=torch.float32)
    logits = (w_u.to(torch.float32) @ lens_vec).cpu()

    tokenizer = lm.tokenizer

    def _rows(indices: torch.Tensor) -> list[dict[str, Any]]:
        return [
            {
                "token_id": int(i),
                "token": tokenizer.decode([int(i)]),
                "logit": float(logits[int(i)]),
            }
            for i in indices
        ]

    boosted = _rows(torch.topk(logits, top_n).indices)
    suppressed = _rows(torch.topk(-logits, top_n).indices)

    # Alignment with the answer contrast the margin is computed from.
    pairs = _answer_pairs(lm, data["cases"])
    unit = (direction.to(torch.float32) / direction.to(torch.float32).norm().clamp_min(1e-12)).cpu()
    diffs = []
    per_case: list[dict[str, Any]] = []
    for case_id, lure_id, correct_id in pairs:
        diff = (w_u[lure_id] - w_u[correct_id]).to(torch.float32).cpu()
        diff = diff / diff.norm().clamp_min(1e-12)
        diffs.append(diff)
        per_case.append({"case_id": case_id, "cos_vs_answer_diff": float(unit @ diff)})
    if not diffs:
        return {"skipped": "no usable answer pairs"}
    diff_matrix = torch.stack(diffs)  # (n_pairs, d_model)
    observed = float(diff_matrix.mul(unit).sum(dim=1).abs().mean())

    generator = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
    d_model = unit.numel()
    random_dirs = torch.randn(n_reference, d_model, generator=generator)
    random_dirs = random_dirs / random_dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
    decoder = _decoder_matrix(sae).detach().to(torch.float32).cpu()
    picks = torch.randperm(decoder.shape[0], generator=generator)[:n_reference]
    other_dirs = decoder[picks]
    other_dirs = other_dirs / other_dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def _null(dirs: torch.Tensor) -> torch.Tensor:
        # mean over answer pairs of |cos(direction, lure - correct)|
        return (dirs @ diff_matrix.T).abs().mean(dim=1)

    random_null = _null(random_dirs)
    decoder_null = _null(other_dirs)
    summary = {
        "gamma_applied": gamma is not None,
        "mean_abs_cos_answer_diff": observed,
        "random_null_mean": float(random_null.mean()),
        "percentile_vs_random": float((random_null < observed).float().mean()),
        "decoder_null_mean": float(decoder_null.mean()),
        "percentile_vs_other_features": float((decoder_null < observed).float().mean()),
        "n_answer_pairs": len(pairs),
        "top_boosted": boosted[:10],
        "top_suppressed": suppressed[:10],
    }
    _write_json(
        run_dir / "logit_lens.json",
        {"summary": summary, "boosted": boosted, "suppressed": suppressed, "per_case": per_case},
    )
    _log(
        "logit_lens: |cos| vs answer diff "
        f"{observed:.4f} (random pct {summary['percentile_vs_random']:.3f}, "
        f"peer pct {summary['percentile_vs_other_features']:.3f})"
    )
    return summary


# --------------------------------------------------- check: positions etc.


def check_positions(
    lm: Any,
    sae: Any,
    data: dict[str, Any],
    feature_id: int,
    layer: int,
    run_dir: Path,
) -> dict[str, Any]:
    """Where does the feature fire, how often, and how big is the ablation?"""

    direction = sae_decoder_direction(sae, [int(feature_id)]).detach().to(torch.float32).cpu()
    direction_norm = float(direction.norm())

    rows: list[dict[str, Any]] = []
    perturbation_rows: list[dict[str, Any]] = []
    counts = torch.zeros(sae.d_sae, dtype=torch.long)
    total_tokens = 0
    dim_abs_sum: torch.Tensor | None = None

    for group, cases in (("trap", data["cases"]), ("reference", data["reference"])):
        for index, case in enumerate(cases, start=1):
            resid = capture_layer_residuals(lm, [case.prompt], layer, token_position="all")
            resid = resid.detach().to(torch.float32).cpu()
            # Sparse: "where does this feature actually fire", not "where is its
            # pre-activation large". The pre-activation is kept alongside so a
            # feature that is always near-threshold is still visible.
            values = qwen_scope_sparse_feature_values(resid, sae, [int(feature_id)])
            values = values.detach().to(torch.float32).cpu().reshape(-1)
            preacts = qwen_scope_feature_preactivations(resid, sae, [int(feature_id)])
            preacts = preacts.detach().to(torch.float32).cpu().reshape(-1)
            seq = int(values.numel())
            last = float(values[-1])
            argmax = int(values.argmax())
            rows.append(
                {
                    "group": group,
                    "case_id": case.case_id,
                    "n_tokens": seq,
                    "last_value": last,
                    "max_value": float(values.max()),
                    "argmax_position_frac": argmax / max(seq - 1, 1),
                    "mean_value": float(values.mean()),
                    "frac_positions_positive": float((values > 0).float().mean()),
                    "last_preactivation": float(preacts[-1]),
                    "mean_preactivation": float(preacts.mean()),
                }
            )

            top_values, top_indices = encode_qwen_scope_topk(resid, sae)
            live = top_indices.detach().cpu()[top_values.detach().cpu() > 0]
            counts += torch.bincount(live.reshape(-1), minlength=sae.d_sae)
            total_tokens += seq
            dim_abs = resid.abs().sum(dim=0)
            dim_abs_sum = dim_abs if dim_abs_sum is None else dim_abs_sum + dim_abs

            if group == "trap":
                resid_norm = float(resid[-1].norm())
                perturbation = abs(last) * direction_norm
                perturbation_rows.append(
                    {
                        "case_id": case.case_id,
                        "feature_value": last,
                        "perturbation_norm": perturbation,
                        "residual_norm": resid_norm,
                        "perturbation_ratio": perturbation / max(resid_norm, 1e-12),
                    }
                )
            if index % 20 == 0:
                _log(f"positions: {group} {index}/{len(cases)}")

    _write_csv(
        run_dir / "positions.csv",
        rows,
        [
            "group",
            "case_id",
            "n_tokens",
            "last_value",
            "max_value",
            "argmax_position_frac",
            "mean_value",
            "frac_positions_positive",
            "last_preactivation",
            "mean_preactivation",
        ],
    )
    _write_csv(
        run_dir / "perturbation.csv",
        perturbation_rows,
        ["case_id", "feature_value", "perturbation_norm", "residual_norm", "perturbation_ratio"],
    )

    def _group(name: str) -> dict[str, float]:
        subset = [row for row in rows if row["group"] == name]
        return {
            "n_prompts": len(subset),
            "mean_last_value": _mean([r["last_value"] for r in subset]),
            "frac_prompts_active_at_last": _mean(
                [1.0 if r["last_value"] > 0 else 0.0 for r in subset]
            ),
            "mean_frac_positions_positive": _mean([r["frac_positions_positive"] for r in subset]),
            "mean_argmax_position_frac": _mean([r["argmax_position_frac"] for r in subset]),
        }

    trap = _group("trap")
    reference = _group("reference")
    density = int(counts[int(feature_id)]) / max(total_tokens, 1)
    population = counts.to(torch.float32) / max(total_tokens, 1)

    assert dim_abs_sum is not None
    dim_mean_abs = dim_abs_sum / max(total_tokens, 1)
    outlier_dims = torch.topk(dim_mean_abs, k=min(20, dim_mean_abs.numel())).indices
    unit = direction / max(direction_norm, 1e-12)
    ratios = [row["perturbation_ratio"] for row in perturbation_rows]

    summary = {
        "positions": {
            "trap": trap,
            "reference": reference,
            "reference_over_trap_last_value": (
                reference["mean_last_value"] / trap["mean_last_value"]
                if trap["mean_last_value"]
                else None
            ),
            "note": (
                "reference prompts are a different task sharing the same answer template; "
                "similar activation there means the feature tracks the template, not the trap"
            ),
        },
        "density": {
            "corpus_tokens": total_tokens,
            "feature_topk_density": density,
            "population_percentile": float((population < density).float().mean()),
            "n_features_ever_active": int((counts > 0).sum()),
            "note": "corpus = trap + reference prompts at all positions, not generic web text",
        },
        "perturbation": {
            "decoder_norm": direction_norm,
            "mean_ratio_to_residual": _mean(ratios),
            "max_ratio_to_residual": max(ratios) if ratios else None,
            "decoder_top5_dim_mass": float(torch.topk(unit.abs(), k=5).values.pow(2).sum()),
            "decoder_mass_in_outlier_dims": float((unit[outlier_dims] ** 2).sum()),
            "outlier_dims": [int(i) for i in outlier_dims],
        },
    }
    _write_json(run_dir / "positions.json", summary)
    _log(
        f"positions: last-token activation trap {trap['mean_last_value']:.3f} vs "
        f"reference {reference['mean_last_value']:.3f}; density {density:.5f} "
        f"(pct {summary['density']['population_percentile']:.3f})"
    )
    return summary


# -------------------------------------------------------------- check: null


def _ablate_margin(
    lm: Any,
    case: Any,
    *,
    layer: int,
    direction: torch.Tensor,
    feature_value: float,
    coefficient: float,
    intervention_mode: str,
) -> float:
    return float(
        answer_logprob_margin(
            lm,
            case.prompt,
            correct_answer=case.correct_answer,
            lure_answer=case.lure_answer,
            layer=int(layer),
            direction=direction,
            feature_value=feature_value,
            coefficient=coefficient,
            intervention_mode=intervention_mode,
        ).margin
    )


def check_null(
    lm: Any,
    sae: Any,
    data: dict[str, Any],
    feature_id: int,
    layer: int,
    run_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Percentile of the observed effect against Gaussian and peer-feature nulls."""

    cfg = table(config, "null")
    gaussian_draws = int(cfg.get("gaussian_draws", 200))
    peer_draws = int(cfg.get("peer_draws", 200))
    panel_cases = int(cfg.get("panel_cases", 8))
    seed = int(cfg.get("seed", 0))
    selection_k = int(cfg.get("selection_k", 64))
    bootstrap = int(cfg.get("bootstrap", 20000))

    panel = family_balanced_subset(data["train"], max_cases=panel_cases)
    direction = sae_decoder_direction(sae, [int(feature_id)]).detach()
    decoder = _decoder_matrix(sae).detach()

    # Baseline, the feature's own effect, and the pool of peers that fire here.
    panel_rows: list[dict[str, Any]] = []
    pool: set[int] = set()
    for case in panel:
        resid = capture_layer_residuals(lm, [case.prompt], layer, token_position="last")
        value = float(
            qwen_scope_sparse_feature_values(resid, sae, [int(feature_id)])
            .detach()
            .to(torch.float32)
            .reshape(-1)[0]
        )
        top_values, top_indices = encode_qwen_scope_topk(resid, sae)
        live = top_indices.detach().cpu()[top_values.detach().cpu() > 0]
        pool.update(int(i) for i in live.reshape(-1).tolist())
        baseline = float(
            answer_logprob_margin(
                lm,
                case.prompt,
                correct_answer=case.correct_answer,
                lure_answer=case.lure_answer,
            ).margin
        )
        ablated = _ablate_margin(
            lm,
            case,
            layer=layer,
            direction=direction,
            feature_value=value,
            coefficient=1.0,
            intervention_mode="remove_activation",
        )
        target_norm = abs(value) * float(direction.to(torch.float32).norm())
        panel_rows.append(
            {
                "case_id": case.case_id,
                "feature_value": value,
                "target_norm": target_norm,
                "baseline_margin": baseline,
                "observed_delta": baseline - ablated,
            }
        )
        _log(f"null: {case.case_id} observed delta {baseline - ablated:+.4f}")

    observed = _mean([row["observed_delta"] for row in panel_rows])
    pool.discard(int(feature_id))
    peers = sorted(pool)
    generator = torch.Generator().manual_seed(seed)
    if peers:
        order = torch.randperm(len(peers), generator=generator).tolist()
        peers = [peers[i] for i in order[:peer_draws]]

    def _null_panel_means(kind: str, n_draws: int) -> list[float]:
        """Mean-over-panel delta for each null direction.

        Draw i uses the SAME direction for every case (per-case generator restarted
        from one seed), scaled to that case's matched norm, so averaging across cases
        gives a valid draw of the panel-mean statistic we compare against.
        """

        sums = [0.0] * n_draws
        for case_index, (case, row) in enumerate(zip(panel, panel_rows, strict=True), start=1):
            case_generator = torch.Generator().manual_seed(seed)
            for draw in range(n_draws):
                if kind == "gaussian":
                    vector = torch.randn(int(sae.d_model), generator=case_generator)
                else:
                    vector = decoder[peers[draw]].detach().to(torch.float32).cpu()
                vector = vector / vector.norm().clamp_min(1e-12) * float(row["target_norm"])
                ablated = _ablate_margin(
                    lm,
                    case,
                    layer=layer,
                    direction=vector,
                    feature_value=1.0,
                    coefficient=-1.0,  # add_vector with -1 subtracts it
                    intervention_mode="add_vector",
                )
                sums[draw] += float(row["baseline_margin"]) - ablated
            _log(f"null[{kind}]: case {case_index}/{len(panel)} done ({n_draws} draws)")
        return [total / len(panel) for total in sums]

    started = time.time()
    gaussian = _null_panel_means("gaussian", gaussian_draws)
    _write_json(run_dir / "null_gaussian.json", gaussian)
    peer = _null_panel_means("peer", len(peers)) if peers else []
    _write_json(run_dir / "null_peer.json", {"feature_ids": peers, "panel_mean_deltas": peer})

    def _percentile(values: Sequence[float]) -> float | None:
        if not values:
            return None
        return sum(1 for value in values if value < observed) / len(values)

    # Selection correction: we reported the best of ~selection_k candidates, so the
    # honest comparison is against the best of selection_k null draws, not one draw.
    max_null: dict[str, Any] = {"selection_k": selection_k}
    if peer:
        source = torch.tensor(peer, dtype=torch.float32)
        boot_generator = torch.Generator().manual_seed(seed + 1)
        picks = torch.randint(0, source.numel(), (bootstrap, selection_k), generator=boot_generator)
        maxima = source[picks].max(dim=1).values
        max_null.update(
            {
                "peer_max_mean": float(maxima.mean()),
                "peer_max_p95": float(maxima.quantile(0.95)),
                "observed_percentile_vs_peer_max": float((maxima < observed).float().mean()),
                "bootstrap": bootstrap,
            }
        )

    beaten = sorted(
        (
            {"feature_id": fid, "panel_mean_delta": value}
            for fid, value in zip(peers, peer, strict=True)
            if value >= observed
        ),
        key=lambda row: -row["panel_mean_delta"],
    )
    summary = {
        "panel_n": len(panel),
        "observed_mean_delta": observed,
        "gaussian_draws": len(gaussian),
        "gaussian_mean": _mean(gaussian),
        "percentile_vs_gaussian": _percentile(gaussian),
        "peer_draws": len(peer),
        "peer_mean": _mean(peer),
        "percentile_vs_peer": _percentile(peer),
        "peer_pool_size": len(pool),
        "n_peers_matching_or_beating": len(beaten),
        "top_peers": beaten[:10],
        "selection_aware": max_null,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    _write_json(run_dir / "null.json", {"summary": summary, "panel": panel_rows})
    _log(
        f"null: observed {observed:+.4f} | gaussian pct "
        f"{summary['percentile_vs_gaussian']} | peer pct {summary['percentile_vs_peer']} | "
        f"peers matching/beating {len(beaten)}/{len(peer)}"
    )
    return summary


# ---------------------------------------------------------- check: siblings


def check_siblings(
    env: dict[str, Any],
    sae: Any,
    feature_id: int,
    layer: int,
    run_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Near-duplicates at this layer, and the best partner at the other scanned layers."""

    cfg = table(config, "siblings")
    top_n = int(cfg.get("top_n", 20))
    layers = [
        int(other)
        for other in (cfg.get("layers") or env["profile"].scan_layers)
        if int(other) != int(layer)
    ]

    decoder = _decoder_matrix(sae).detach().to(torch.float32).cpu()
    target = decoder[int(feature_id)]
    target_unit = target / target.norm().clamp_min(1e-12)
    units = decoder / decoder.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cosines = units @ target_unit
    cosines[int(feature_id)] = float("-inf")
    top = torch.topk(cosines, min(top_n, cosines.numel()))
    same_layer = [
        {"feature_id": int(i), "cosine": float(v)}
        for v, i in zip(top.values, top.indices, strict=True)
    ]

    cross_layer: list[dict[str, Any]] = []
    for other in layers:
        try:
            other_sae = _load_sae(env, other)
        except Exception as exc:  # noqa: BLE001 - a missing SAE should not kill the run
            _log(f"siblings: layer {other} unavailable ({exc})")
            continue
        other_decoder = _decoder_matrix(other_sae).detach().to(torch.float32).cpu()
        other_units = other_decoder / other_decoder.norm(dim=1, keepdim=True).clamp_min(1e-12)
        best = torch.topk(other_units @ target_unit, 5)
        cross_layer.append(
            {
                "layer": other,
                "top_partners": [
                    {"feature_id": int(i), "cosine": float(v)}
                    for v, i in zip(best.values, best.indices, strict=True)
                ],
            }
        )
        best_cos = cross_layer[-1]["top_partners"][0]["cosine"]
        _log(f"siblings: layer {other} best cosine {best_cos:.3f}")
        del other_sae, other_decoder, other_units
        clear_device_cache()

    summary = {"same_layer_top": same_layer, "cross_layer": cross_layer}
    _write_json(run_dir / "siblings.json", summary)
    return summary


# ------------------------------------------------------------------ runner


CHECK_ORDER = ("logit_lens", "positions", "null", "siblings")


def run(config_path: Path, output_root: Path) -> Path:
    config = load_toml(config_path)
    name = run_name(config)
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config.toml")

    feature_cfg = table(config, "feature")
    if "feature_id" not in feature_cfg or "layer" not in feature_cfg:
        raise ValueError("[feature].feature_id and [feature].layer are required")
    feature_id = int(feature_cfg["feature_id"])
    layer = int(feature_cfg["layer"])
    checks_cfg = table(config, "checks")

    env = _resolve_env(config)
    data = _load_data(config)
    manifest: dict[str, Any] = {
        "run_name": name,
        "job": "feature_diagnostics",
        "started_at": _timestamp(),
        "feature_id": feature_id,
        "layer": layer,
        "profile": env["profile"].key,
        "model_id": env["model_id"],
        "sae_repo_id": env["repo_id"],
        "dataset": data["dataset"],
        "n_cases": len(data["cases"]),
        "n_train": len(data["train"]),
        "reference_dataset": data["reference_dataset"],
        "n_reference": len(data["reference"]),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    _write_json(run_dir / "manifest.json", manifest)

    _log(f"loading {env['model_id']} (feature {feature_id} @ layer {layer})")
    lm = load_qwen_language_model(
        env["model_id"], device_map=env["device_map"], dtype=env["dtype"], dispatch=True
    )
    sae = _load_sae(env, layer)

    results: dict[str, Any] = {}
    failures: dict[str, str] = {}
    try:
        for check in CHECK_ORDER:
            if not bool(checks_cfg.get(check, True)):
                _log(f"check {check}: disabled")
                continue
            _log(f"=== check: {check} ===")
            try:
                if check == "logit_lens":
                    results[check] = check_logit_lens(lm, sae, data, feature_id, run_dir, config)
                elif check == "positions":
                    results[check] = check_positions(lm, sae, data, feature_id, layer, run_dir)
                elif check == "null":
                    results[check] = check_null(lm, sae, data, feature_id, layer, run_dir, config)
                elif check == "siblings":
                    results[check] = check_siblings(env, sae, feature_id, layer, run_dir, config)
            except Exception as exc:  # noqa: BLE001 - keep the other checks' results
                _log(f"check {check} FAILED: {exc}")
                failures[check] = str(exc)
            manifest.update({"results": results, "failures": failures})
            _write_json(run_dir / "manifest.json", manifest)
    finally:
        manifest.update({"finished_at": _timestamp(), "results": results, "failures": failures})
        _write_json(run_dir / "manifest.json", manifest)

    print(f"ARTIFACT_DIR={run_dir}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("outputs/experiments"), type=Path)
    args = parser.parse_args()
    run(args.config, args.output_root)


if __name__ == "__main__":
    main()
