"""Rigorous CRT lure-feature study as a Colab job.

Where the `notebooks/` explore one case at a time, this job runs the controlled
study: discover the lure feature on a *discovery split* of a real CRT dataset,
then apply it to *held-out* items, measured against a random-direction null and
read out both as teacher-forced margin and as actual free-generation accuracy.

Kinds:
    phenomenon          baseline lure margin over a dataset (establish the effect)
    discover            train-split localization + generalizing feature + null
    causal_heldout      apply the discovered feature to the held-out split (margin)
    control_specificity hostile vs matched-control margin delta (specificity)
    behavioral          free-generation accuracy with vs without feature steering
    study               all of the above (margin phase, then generation phase)

Datasets follow docs/datasets.md (default hagendorff_crt: 150 items, 3 families,
matched controls). See docs/study_design.md.
"""
# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.runners.config import load_toml, run_name, table
from mindscopex_analysis import (
    DEFAULT_ANALYSIS_PROFILE_KEY,
    aggregate_feature_effect,
    answer_logprob_margin,
    clear_device_cache,
    control_specificity_rows,
    default_sae_device,
    discover_generalizing_feature,
    dtype_from_name,
    get_qwen35_analysis_profile,
    instruct_lure_cases,
    load_qwen_language_model,
    load_qwen_scope_sae,
    load_qwen_text_generation_model,
    lure_dataset_cases,
    null_summary,
    random_direction_null_for_feature,
    recommended_dtype_name,
    split_lure_cases,
    steer_generation_labels,
)

DEFAULT_STEER_COEFFICIENTS = (0.0, -2.0, -4.0, -8.0)

# Which model each kind needs: nnsight LM for margins, HF model for generation.
MARGIN_KINDS = ("phenomenon", "discover", "causal_heldout", "control_specificity")
GEN_KINDS = ("behavioral",)
FEATURE_KINDS = ("causal_heldout", "control_specificity", "behavioral")

META_KINDS: dict[str, tuple[str, ...]] = {
    "study": (
        "phenomenon",
        "discover",
        "control_specificity",
        "causal_heldout",
        "behavioral",
    ),
    # Margin-only study (skips free-generation) for large models where loading a
    # second HF generation model and decoding is impractical.
    "study_margin": (
        "phenomenon",
        "discover",
        "control_specificity",
        "causal_heldout",
    ),
}


# --------------------------------------------------------------------------- io


def _log(message: str) -> None:
    print(f"[study] {message}", flush=True)


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
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


def _float_list(value: Any, default: Sequence[float]) -> list[float]:
    if value is None:
        return list(default)
    if not isinstance(value, list) or not all(isinstance(item, (int, float)) for item in value):
        raise TypeError(f"Expected a list of numbers, got {value!r}")
    return [float(item) for item in value]


def _int_list_or_none(value: Any) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise TypeError(f"Expected a list of ints, got {value!r}")
    return [int(item) for item in value]


def _safe_plot(fn: Callable[[], Any], path: Path) -> str | None:
    try:
        fn()
        return str(path)
    except Exception as exc:
        print(f"plot failed for {path.name}: {exc}", flush=True)
        return None


# ------------------------------------------------------------------- profile


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


# ------------------------------------------------------------------- cases


def _load_splits(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = table(config, "data")
    dataset = str(data_cfg.get("dataset", "hagendorff_crt"))
    families = data_cfg.get("families") or None
    limit = data_cfg.get("limit_per_family", 0)
    limit = None if not limit else int(limit)
    cases = lure_dataset_cases(
        dataset,
        families=tuple(families) if families else None,
        limit_per_family=limit,
    )
    if bool(data_cfg.get("instruction", True)):
        cases = instruct_lure_cases(cases)
    train_frac = float(data_cfg.get("train_frac", 0.6))
    split_seed = int(data_cfg.get("split_seed", 0))
    train, test = split_lure_cases(cases, train_frac=train_frac, seed=split_seed)
    return {
        "dataset": dataset,
        "all": cases,
        "train": train,
        "test": test,
        "train_frac": train_frac,
        "split_seed": split_seed,
    }


# ------------------------------------------------------------------- feature


def _discover_study_feature(
    lm: Any,
    train_cases: Sequence[Any],
    config: dict[str, Any],
    env: dict[str, Any],
) -> dict[str, Any]:
    profile = env["profile"]
    dcfg = table(config, "discover")
    layers = _int_list_or_none(dcfg.get("layers")) or list(profile.scan_layers)
    candidate_top_n = int(dcfg.get("candidate_top_n", 12))
    min_active_cases = int(dcfg.get("min_active_cases", 2))
    max_candidates = int(dcfg.get("max_candidates", 40))
    max_cases = int(dcfg.get("max_cases", 30))
    coefficient = float(dcfg.get("coefficient", 1.0))
    intervention_mode = str(dcfg.get("intervention_mode", "remove_activation"))
    null_samples = int(dcfg.get("null_samples", 32))
    null_seed = int(dcfg.get("null_seed", 0))

    subset = list(train_cases)[:max_cases]
    localization: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_sae = None
    best_feature_rows: list[dict[str, Any]] = []

    for layer in layers:
        _log(f"discover: layer {int(layer)} on {len(subset)} train items")
        sae = _load_sae(env, int(layer))
        rows = discover_generalizing_feature(
            lm,
            subset,
            layer=int(layer),
            sae=sae,
            candidate_top_n=candidate_top_n,
            min_active_cases=min_active_cases,
            max_candidates=max_candidates,
            coefficient=coefficient,
            intervention_mode=intervention_mode,
            progress=_log,
        )
        if not rows:
            del sae
            clear_device_cache()
            continue
        top = rows[0]
        # Null for the top feature on a representative discovery case.
        rep = subset[0]
        rep_effect = aggregate_feature_effect(
            lm,
            [rep],
            layer=int(layer),
            sae=sae,
            feature_id=int(top["feature_id"]),
            coefficient=coefficient,
            intervention_mode=intervention_mode,
        )
        rep_row = rep_effect["per_case"][0]
        null = random_direction_null_for_feature(
            lm,
            rep,
            layer=int(layer),
            sae=sae,
            feature_id=int(top["feature_id"]),
            feature_value=float(rep_row["feature_value"]),
            coefficient=coefficient,
            n_samples=null_samples,
            seed=null_seed,
            baseline_margin=float(rep_row["baseline_margin"]),
        )
        summary = null_summary(float(rep_row["margin_delta"]), null)
        localization.append(
            {
                "layer": int(layer),
                "feature_id": int(top["feature_id"]),
                "mean_margin_delta": top["mean_margin_delta"],
                "frac_positive": top["frac_positive"],
                "active_in_cases": top["active_in_cases"],
                "rep_margin_delta": rep_row["margin_delta"],
                "null_mean": summary["null_mean"],
                "null_z": summary["z"],
                "null_percentile": summary["percentile"],
            }
        )
        if best is None or top["mean_margin_delta"] > best["mean_margin_delta"]:
            best = {**top, "layer": int(layer)}
            best_sae = sae
            best_feature_rows = rows
        else:
            del sae
            clear_device_cache()

    if best is None:
        raise RuntimeError("discovery found no candidate features on the train split")
    return {
        "feature": best,
        "sae": best_sae,
        "localization": localization,
        "feature_rows": best_feature_rows,
        "coefficient": coefficient,
        "intervention_mode": intervention_mode,
    }


def _ensure_feature(
    lm: Any,
    splits: dict[str, Any],
    config: dict[str, Any],
    env: dict[str, Any],
    run_dir: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    if state.get("study_feature") is not None:
        return state["study_feature"]

    fcfg = table(config, "feature")
    feature_id = fcfg.get("feature_id")
    if feature_id is not None:
        layer = fcfg.get("layer")
        if layer is None:
            raise ValueError("[feature].feature_id requires [feature].layer")
        sae = _load_sae(env, int(layer))
        info = {
            "feature": {"feature_id": int(feature_id), "layer": int(layer)},
            "sae": sae,
            "coefficient": float(fcfg.get("coefficient", 1.0)),
            "intervention_mode": str(fcfg.get("intervention_mode", "remove_activation")),
            "localization": [],
            "feature_rows": [],
            "source": "pinned",
        }
        state["study_feature"] = info
        _write_json(
            run_dir / "study_feature.json", {"feature": info["feature"], "source": "pinned"}
        )
        return info

    info = _discover_study_feature(lm, splits["train"], config, env)
    info["source"] = "discovered"
    state["study_feature"] = info
    _write_json(
        run_dir / "study_feature.json",
        {"feature": info["feature"], "source": "discovered", "localization": info["localization"]},
    )
    return info


# --------------------------------------------------------------------- plots


def _plot_phenomenon(margins: Sequence[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.4), constrained_layout=True)
    ax.hist(list(margins), bins=20, color="#2a78d6", alpha=0.85)
    ax.axvline(0.0, color="#e34948", linewidth=1.2, label="lure = correct")
    ax.set_xlabel("baseline margin  logprob(lure) - logprob(correct)")
    ax.set_ylabel("items")
    ax.set_title("Baseline lure preference")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_localization(rows: Sequence[dict[str, Any]], path: Path) -> None:
    layers = [row["layer"] for row in rows]
    deltas = [row["mean_margin_delta"] for row in rows]
    nulls = [row["null_mean"] or 0.0 for row in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.4), constrained_layout=True)
    ax.plot(layers, deltas, color="#2a78d6", marker="o", label="feature mean margin_delta")
    ax.plot(
        layers, nulls, color="#898781", linestyle="--", marker=".", label="random-direction null"
    )
    ax.axhline(0.0, color="#c3c2b7", linewidth=1.0)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean margin_delta (train)")
    ax.set_title("Layer localization vs null")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_behavioral(rows: Sequence[dict[str, Any]], path: Path) -> None:
    coeffs = [row["coefficient"] for row in rows]
    acc = [row["steered_accuracy"] for row in rows]
    lure = [row["steered_lure_rate"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.4), constrained_layout=True)
    ax.plot(coeffs, acc, color="#008300", marker="o", label="accuracy")
    ax.plot(coeffs, lure, color="#e34948", marker="o", label="lure rate")
    ax.set_xlabel("steering coefficient (negative suppresses lure feature)")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.set_title("Generation under feature steering")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------- kinds


def run_phenomenon(
    lm: Any,
    splits: dict[str, Any],
    config: dict[str, Any],
    env: dict[str, Any],
    run_dir: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    cases = splits["all"]
    _log(f"phenomenon: baseline margins on {len(cases)} items")
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        margin = answer_logprob_margin(
            lm, case.prompt, correct_answer=case.correct_answer, lure_answer=case.lure_answer
        )
        if index % 25 == 0:
            _log(f"phenomenon: {index}/{len(cases)}")
        rows.append(
            {
                "case_id": case.case_id,
                "family": case.family,
                "baseline_margin": margin.margin,
                "baseline_mean_margin": margin.mean_margin,
                "lure_preferred": int(margin.margin > 0),
            }
        )
    n = len(rows)
    summary = {
        "dataset": splits["dataset"],
        "n_cases": n,
        "mean_margin": sum(r["baseline_margin"] for r in rows) / n if n else 0.0,
        "frac_lure_preferred": sum(r["lure_preferred"] for r in rows) / n if n else 0.0,
    }
    _write_csv(
        run_dir / "phenomenon.csv",
        rows,
        ["case_id", "family", "baseline_margin", "baseline_mean_margin", "lure_preferred"],
    )
    _write_json(run_dir / "phenomenon" / "summary.json", summary)
    png = _safe_plot(
        lambda: _plot_phenomenon([r["baseline_margin"] for r in rows], run_dir / "phenomenon.png"),
        run_dir / "phenomenon.png",
    )
    return {
        "kind": "phenomenon",
        "paper_csv": str(run_dir / "phenomenon.csv"),
        "png": png,
        **summary,
    }


def run_discover(
    lm: Any,
    splits: dict[str, Any],
    config: dict[str, Any],
    env: dict[str, Any],
    run_dir: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    info = _discover_study_feature(lm, splits["train"], config, env)
    info["source"] = "discovered"
    state["study_feature"] = info
    localization = info["localization"]
    _write_csv(
        run_dir / "discover_localization.csv",
        localization,
        [
            "layer",
            "feature_id",
            "mean_margin_delta",
            "frac_positive",
            "active_in_cases",
            "rep_margin_delta",
            "null_mean",
            "null_z",
            "null_percentile",
        ],
    )
    _write_csv(
        run_dir / "discover_features.csv",
        info["feature_rows"],
        [
            "feature_id",
            "layer",
            "mean_margin_delta",
            "std_margin_delta",
            "frac_positive",
            "active_in_cases",
            "n_cases",
        ],
    )
    _write_json(
        run_dir / "study_feature.json",
        {"feature": info["feature"], "source": "discovered", "localization": localization},
    )
    png = _safe_plot(
        lambda: _plot_localization(localization, run_dir / "discover_localization.png"),
        run_dir / "discover_localization.png",
    )
    return {
        "kind": "discover",
        "paper_csv": str(run_dir / "discover_localization.csv"),
        "png": png,
        "best_feature": info["feature"],
        "n_train": len(splits["train"]),
    }


def run_causal_heldout(
    lm: Any,
    splits: dict[str, Any],
    config: dict[str, Any],
    env: dict[str, Any],
    run_dir: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    info = _ensure_feature(lm, splits, config, env, run_dir, state)
    feature = info["feature"]
    _log(f"causal_heldout: applying feature to {len(splits['test'])} held-out items")
    effect = aggregate_feature_effect(
        lm,
        splits["test"],
        layer=int(feature["layer"]),
        sae=info["sae"],
        feature_id=int(feature["feature_id"]),
        coefficient=float(info.get("coefficient", 1.0)),
        intervention_mode=str(info.get("intervention_mode", "remove_activation")),
    )
    per_case = effect.pop("per_case")
    _write_csv(
        run_dir / "causal_heldout.csv",
        per_case,
        ["case_id", "family", "feature_value", "baseline_margin", "edited_margin", "margin_delta"],
    )
    _write_json(run_dir / "causal_heldout" / "summary.json", effect)
    return {
        "kind": "causal_heldout",
        "paper_csv": str(run_dir / "causal_heldout.csv"),
        "n_test": len(splits["test"]),
        "mean_margin_delta": effect["mean_margin_delta"],
        "frac_positive": effect["frac_positive"],
    }


def run_control_specificity(
    lm: Any,
    splits: dict[str, Any],
    config: dict[str, Any],
    env: dict[str, Any],
    run_dir: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    info = _ensure_feature(lm, splits, config, env, run_dir, state)
    feature = info["feature"]
    _log(f"control_specificity: hostile vs control on {len(splits['test'])} items")
    rows = control_specificity_rows(
        lm,
        splits["test"],
        layer=int(feature["layer"]),
        sae=info["sae"],
        feature_id=int(feature["feature_id"]),
        coefficient=float(info.get("coefficient", 1.0)),
        intervention_mode=str(info.get("intervention_mode", "remove_activation")),
    )
    _write_csv(
        run_dir / "control_specificity.csv",
        rows,
        ["case_id", "family", "hostile_margin_delta", "control_margin_delta", "specificity_gap"],
    )
    n = len(rows)
    summary = {
        "n_cases": n,
        "mean_hostile_delta": sum(r["hostile_margin_delta"] for r in rows) / n if n else 0.0,
        "mean_control_delta": sum(r["control_margin_delta"] for r in rows) / n if n else 0.0,
        "mean_specificity_gap": sum(r["specificity_gap"] for r in rows) / n if n else 0.0,
    }
    _write_json(run_dir / "control_specificity" / "summary.json", summary)
    return {
        "kind": "control_specificity",
        "paper_csv": str(run_dir / "control_specificity.csv"),
        **summary,
    }


def run_behavioral(
    model: Any,
    tokenizer: Any,
    splits: dict[str, Any],
    config: dict[str, Any],
    env: dict[str, Any],
    run_dir: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    info = state.get("study_feature")
    if info is None:
        raise RuntimeError("behavioral kind needs a discovered or pinned feature")
    feature = info["feature"]
    bcfg = table(config, "behavioral")
    coefficients = _float_list(bcfg.get("coefficients"), DEFAULT_STEER_COEFFICIENTS)
    max_new_tokens = int(bcfg.get("max_new_tokens", 16))
    max_cases = int(bcfg.get("max_cases", 40))
    token_position = str(bcfg.get("token_position", "all"))
    cases = list(splits["test"])[:max_cases]

    _log(f"behavioral: {len(coefficients)} coefficients x {len(cases)} items (baseline+steered)")
    rows: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    for coefficient in coefficients:
        _log(f"behavioral: coefficient {coefficient:g}")
        result = steer_generation_labels(
            model,
            tokenizer,
            cases,
            layer=int(feature["layer"]),
            sae=info["sae"],
            feature_id=int(feature["feature_id"]),
            coefficient=float(coefficient),
            max_new_tokens=max_new_tokens,
            token_position=token_position,
            progress=_log,
        )
        rows.append(
            {
                "coefficient": float(coefficient),
                "baseline_accuracy": result["baseline_summary"]["accuracy"],
                "steered_accuracy": result["steered_summary"]["accuracy"],
                "accuracy_delta": result["accuracy_delta"],
                "baseline_lure_rate": result["baseline_summary"]["lure_rate"],
                "steered_lure_rate": result["steered_summary"]["lure_rate"],
                "lure_rate_delta": result["lure_rate_delta"],
            }
        )
        detail.append({"coefficient": float(coefficient), "steered_rows": result["steered_rows"]})

    _write_csv(
        run_dir / "behavioral.csv",
        rows,
        [
            "coefficient",
            "baseline_accuracy",
            "steered_accuracy",
            "accuracy_delta",
            "baseline_lure_rate",
            "steered_lure_rate",
            "lure_rate_delta",
        ],
    )
    _write_json(run_dir / "behavioral" / "generations.json", detail)
    png = _safe_plot(
        lambda: _plot_behavioral(rows, run_dir / "behavioral.png"), run_dir / "behavioral.png"
    )
    return {
        "kind": "behavioral",
        "paper_csv": str(run_dir / "behavioral.csv"),
        "png": png,
        "n_cases": len(cases),
    }


MARGIN_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "phenomenon": run_phenomenon,
    "discover": run_discover,
    "causal_heldout": run_causal_heldout,
    "control_specificity": run_control_specificity,
}
GEN_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "behavioral": run_behavioral,
}


def _expand_kind(kind: str) -> list[str]:
    if kind in META_KINDS:
        return list(META_KINDS[kind])
    if kind in MARGIN_RUNNERS or kind in GEN_RUNNERS:
        return [kind]
    valid = ", ".join(sorted(set(MARGIN_RUNNERS) | set(GEN_RUNNERS) | set(META_KINDS)))
    raise ValueError(f"Unknown [experiment].kind={kind!r}. Choose one of: {valid}")


# ----------------------------------------------------------------------- main


def run(config_path: Path, output_root: Path) -> Path:
    import torch

    config = load_toml(config_path)
    name = run_name(config)
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config.toml")

    kind = str(table(config, "experiment").get("kind", "")).strip()
    if not kind:
        raise ValueError("[experiment].kind is required")
    kinds = _expand_kind(kind)

    env = _resolve_env(config)
    profile = env["profile"]
    splits = _load_splits(config)

    margin_kinds = [k for k in kinds if k in MARGIN_KINDS]
    gen_kinds = [k for k in kinds if k in GEN_KINDS]

    # A feature-dependent kind with no pin and no discovery needs discovery first.
    pinned = table(config, "feature").get("feature_id") is not None
    needs_feature = any(k in FEATURE_KINDS for k in kinds)
    if needs_feature and not pinned and "discover" not in margin_kinds:
        margin_kinds = ["discover", *margin_kinds]

    manifest: dict[str, Any] = {
        "run_name": name,
        "job": "research_experiments",
        "started_at": _timestamp(),
        "requested_kind": kind,
        "margin_kinds": margin_kinds,
        "gen_kinds": gen_kinds,
        "profile": profile.key,
        "model_id": env["model_id"],
        "sae_repo_id": env["repo_id"],
        "dataset": splits["dataset"],
        "n_all": len(splits["all"]),
        "n_train": len(splits["train"]),
        "n_test": len(splits["test"]),
        "train_frac": splits["train_frac"],
        "split_seed": splits["split_seed"],
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "git_commit": _git_value(["rev-parse", "HEAD"]),
    }
    _write_json(run_dir / "manifest.json", manifest)

    state: dict[str, Any] = {}
    fragments: list[dict[str, Any]] = []
    start = time.time()

    if margin_kinds:
        print(f"[margin phase] loading {env['model_id']}", flush=True)
        lm = load_qwen_language_model(
            env["model_id"], device_map=env["device_map"], dtype=env["dtype"], dispatch=True
        )
        try:
            for current in margin_kinds:
                print(f"=== kind: {current} (margin) ===", flush=True)
                fragments.append(MARGIN_RUNNERS[current](lm, splits, config, env, run_dir, state))
        finally:
            del lm
            clear_device_cache()

    if gen_kinds:
        print(f"[generation phase] loading {env['model_id']}", flush=True)
        model, tokenizer = load_qwen_text_generation_model(
            env["model_id"], device_map=env["device_map"], dtype=env["dtype"]
        )
        try:
            for current in gen_kinds:
                print(f"=== kind: {current} (generation) ===", flush=True)
                fragments.append(
                    GEN_RUNNERS[current](model, tokenizer, splits, config, env, run_dir, state)
                )
        finally:
            del model
            del tokenizer
            clear_device_cache()

    feature = state.get("study_feature")
    manifest.update(
        {
            "finished_at": _timestamp(),
            "elapsed_seconds": round(time.time() - start, 3),
            "study_feature": feature["feature"] if feature else None,
            "feature_source": feature.get("source") if feature else None,
            "results": fragments,
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
