"""Does a coactivating SET of features mediate the lure where no single one does?

Every single-feature result so far came back indistinguishable from matched peers.
That is the expected outcome if the behaviour is carried by a distributed module,
so this job runs the module arm of the question and keeps the single-feature arm
alongside it as the comparison:

    discovery items -> sparse activations -> coactivation graph
        -> modules (connected components) -> joint ablation on HELD-OUT items

Four conditions per held-out item, which is what makes the answer readable:

    single_best      the strongest individual member, ablated alone
    members_apart    every member ablated alone (their deltas summed afterwards)
    module_joint     the whole module removed in one edit
    random_module    frequency-matched, size-matched, norm-matched module nulls

The joint edit needs no multi-site machinery: remove_activation is linear in the
feature, so a same-layer module is one vector, sum_f a_f * W_dec[f].

Two traps this design exists to avoid. A module removes strictly more norm than a
single feature, so "the module beat the single feature" proves nothing without a
norm-matched module null -- hence random_module, rescaled to the real module's
norm. And a module can beat its null by damaging the model rather than the lure,
so every margin is split into its correct and lure logprob deltas.
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


def _sign_flip_p(values: Sequence[float], draws: int = 20000, seed: int = 0) -> dict[str, Any]:
    """Paired sign-flip randomisation test -- these deltas are heavy-tailed at n~25."""

    if not values:
        return {"n": 0, "mean": None, "p": None}
    tensor = torch.tensor([float(v) for v in values], dtype=torch.float64)
    observed = float(tensor.mean())
    generator = torch.Generator().manual_seed(int(seed))
    signs = torch.randint(0, 2, (draws, tensor.numel()), generator=generator, dtype=torch.float64)
    means = ((signs * 2 - 1) * tensor).mean(dim=1)
    return {
        "n": len(values),
        "mean": observed,
        "p": float((means.abs() >= abs(observed)).to(torch.float64).mean()),
        "frac_positive": sum(1 for v in values if v > 0) / len(values),
    }


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
    cases = lure_dataset_cases(dataset)
    conditions = data_cfg.get("conditions") or None
    if conditions:
        suffixes = tuple(f"_{name}" for name in conditions)
        cases = [case for case in cases if case.case_id.endswith(suffixes)]
    if bool(data_cfg.get("instruction", True)):
        cases = instruct_lure_cases(cases)
    train, test = split_lure_cases(
        cases,
        train_frac=float(data_cfg.get("train_frac", 0.6)),
        seed=int(data_cfg.get("split_seed", 0)),
    )
    max_test = int(data_cfg.get("max_test_items", 0))
    if max_test:
        test = family_balanced_subset(test, max_cases=max_test)
    return {"dataset": dataset, "train": train, "test": test}


def _margin_row(margin: Any, baseline: Any) -> dict[str, float]:
    return {
        "margin_delta": float(baseline.margin) - float(margin.margin),
        # A direction that merely damages the model drags both logprobs down; a lure
        # effect moves mostly the lure. Nearly free to record, and it settles that
        # question without another run.
        "correct_logprob_delta": float(baseline.correct.logprob) - float(margin.correct.logprob),
        "lure_logprob_delta": float(baseline.lure.logprob) - float(margin.lure.logprob),
    }


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
    edge_threshold = float(mcfg.get("edge_threshold", 0.3))
    metric = str(mcfg.get("metric", "jaccard"))
    min_size = int(mcfg.get("min_size", 2))
    max_size = int(mcfg.get("max_size", 12))
    max_modules = int(mcfg.get("max_modules", 2))
    random_modules = int(mcfg.get("random_modules", 10))
    seed = int(mcfg.get("seed", 0))

    env = _resolve_env(config)
    splits = _load_splits(config)
    manifest: dict[str, Any] = {
        "run_name": name,
        "job": "feature_modules",
        "started_at": _timestamp(),
        "layer": layer,
        "profile": env["profile"].key,
        "model_id": env["model_id"],
        "sae_repo_id": env["repo_id"],
        "dataset": splits["dataset"],
        "n_train": len(splits["train"]),
        "n_test": len(splits["test"]),
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

    # --- 1. coactivation graph on the DISCOVERY split -----------------------
    _log(f"collecting sparse activations on {len(splits['train'])} discovery items")
    feature_ids, matrix = sparse_activation_matrix(
        lm,
        splits["train"],
        layer=layer,
        sae=sae,
        min_active_cases=min_active_cases,
        max_features=max_features,
    )
    if not feature_ids:
        raise RuntimeError("no feature fired often enough to build a coactivation graph")
    counts = [int((matrix[:, i] > 0).sum()) for i in range(len(feature_ids))]
    edges = coactivation_edges(matrix, feature_ids)
    _write_csv(
        run_dir / "coactivation_edges.csv",
        edges,
        ["feature_a", "feature_b", "co_fire", "jaccard", "activation_corr"],
    )
    _log(f"graph: {len(feature_ids)} features, {len(edges)} pairs")

    found = modules_from_edges(
        edges,
        edge_threshold=edge_threshold,
        metric=metric,
        min_size=min_size,
        max_size=max_size,
    )
    if not found:
        raise RuntimeError(
            f"no module of size {min_size}-{max_size} survived {metric} >= {edge_threshold}; "
            "the features that fire here do not group"
        )
    modules = found[:max_modules]
    module_rows = [
        {
            "rank": index,
            "features": module,
            "size": len(module),
            "coherence": module_coherence(matrix, feature_ids, module),
            "mean_active_cases": _mean(
                [float(counts[feature_ids.index(f)]) for f in module if f in feature_ids]
            ),
        }
        for index, module in enumerate(found, start=1)
    ]
    _write_json(run_dir / "feature_modules.json", module_rows)
    _log(f"modules: {[len(m) for m in found]} (testing top {len(modules)})")

    # --- 2. held-out interventions -----------------------------------------
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

        for item_index, case in enumerate(splits["test"], start=1):
            residual = capture_layer_residuals(lm, [case.prompt], layer, token_position="last")
            values = (
                qwen_scope_sparse_feature_values(residual, sae, module)
                .detach()
                .to(torch.float32)
                .reshape(-1)
                .tolist()
            )
            baseline = answer_logprob_margin(
                lm,
                case.prompt,
                correct_answer=case.correct_answer,
                lure_answer=case.lure_answer,
            )
            common = {"case_id": case.case_id, "family": case.family, "module": module_index}

            def _record(condition: str, direction: torch.Tensor, draw: int = -1) -> float:
                margin = answer_logprob_margin(
                    lm,
                    case.prompt,
                    correct_answer=case.correct_answer,
                    lure_answer=case.lure_answer,
                    layer=layer,
                    direction=direction,
                    feature_value=1.0,
                    coefficient=-1.0,  # add_vector with -1 subtracts the whole module
                    intervention_mode="add_vector",
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
            joint = _record("module_joint", joint_direction)

            member_deltas = []
            for feature_id, value in zip(module, values, strict=True):
                member_deltas.append(
                    _record(
                        f"member_{feature_id}",
                        module_ablation_direction(sae, [feature_id], [value]),
                    )
                )
            # Strongest member by its own removed norm, ablated alone: the
            # single-feature arm the module has to beat.
            strongest = max(
                range(len(module)),
                key=lambda i: module_norm(module_ablation_direction(sae, [module[i]], [values[i]])),
            )
            _record(
                "single_best",
                module_ablation_direction(sae, [module[strongest]], [values[strongest]]),
            )
            joint_minus_sum.append(joint - sum(member_deltas))

            for draw, random_module in enumerate(randoms):
                random_values = (
                    qwen_scope_sparse_feature_values(residual, sae, random_module)
                    .detach()
                    .to(torch.float32)
                    .reshape(-1)
                    .tolist()
                )
                random_direction = module_ablation_direction(sae, random_module, random_values)
                # Same size, matched firing frequency, and now the same removed norm,
                # so only the identity of the features differs from the real module.
                _record("random_module", rescale_to_norm(random_direction, target_norm), draw)

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
        summary = {
            "module": module_index,
            "features": module,
            "size": len(module),
            "coherence": module_coherence(matrix, feature_ids, module),
            "n_random_modules": len(randoms),
            "joint": _sign_flip_p(per_condition.get("module_joint", []), seed=seed),
            "single_best": _sign_flip_p(per_condition.get("single_best", []), seed=seed),
            "random_module_mean": _mean(random_deltas),
            # The honest comparison: the module against norm-matched modules, not
            # against zero and not against a single feature.
            "joint_minus_random": _sign_flip_p(
                [delta - _mean(random_deltas) for delta in per_condition.get("module_joint", [])],
                seed=seed,
            ),
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
        _log(
            f"module {module_index}: joint {summary['joint']['mean']:+.4f} "
            f"(p={summary['joint']['p']}) vs random {summary['random_module_mean']:+.4f}"
        )
        _write_json(run_dir / "module_summary.json", summaries)

    manifest.update(
        {
            "finished_at": _timestamp(),
            "elapsed_seconds": round(time.time() - started, 1),
            "n_features_in_graph": len(feature_ids),
            "n_edges": len(edges),
            "module_sizes": [len(m) for m in found],
            "summaries": summaries,
        }
    )
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
