"""Try to falsify a feature: does it track the lure structure, or the surface?

A feature can pass every causal gate and still be the wrong explanation, because
the trap items differ from their controls in more than the trap: same template,
same answer strings, overlapping vocabulary. This job attacks the feature from the
directions that would make it a surface artifact, and reports the errors rather
than a single verdict.

    condition profile   hostile / explicit / neutral / counterfactual carry the SAME
                        vocabulary and template and differ only in structure, so
                        this is the sharpest lexical-vs-structure dissociation the
                        data can give. A feature that reads the trap should separate
                        hostile from neutral; one that reads the words will not.
    template control    a different task sharing the "...\\nAnswer:" template. Equal
                        activation there means the feature tracks the position, not
                        the trap.
    paraphrase          same structure, different surface wording, via template_id.
                        Activation should survive a rewrite; if it does not, the
                        feature is tied to phrasing.
    answer confound     correlation between activation and which answer strings the
                        item uses, plus answer length -- the two surface variables
                        that have already bitten this study.
    error audit         at a threshold picked on the discovery split, list the
                        hostile items where the feature stays silent and the control
                        items where it fires. Those cases are the finding.

Reported as separation statistics (AUC, mean gap) rather than pass/fail: a feature
that is 70% structural and 30% positional is the likely truth, and a boolean would
hide it.
"""
# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
from collections import defaultdict
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
    capture_layer_residuals,
    default_sae_device,
    dtype_from_name,
    get_qwen35_analysis_profile,
    instruct_lure_cases,
    load_qwen_language_model,
    load_qwen_scope_sae,
    lure_dataset_cases,
    pearson,
    qwen_scope_feature_preactivations,
    qwen_scope_sparse_feature_values,
    recommended_dtype_name,
    split_lure_cases,
)


def _log(message: str) -> None:
    print(f"[falsify] {message}", flush=True)


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


def _auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    """P(a random positive scores above a random negative); 0.5 means no separation.

    Threshold-free, so it does not depend on where a cutoff happens to fall -- which
    matters because the cutoff here is itself estimated.
    """

    if not positive or not negative:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


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
    return {
        "profile": profile,
        "model_id": profile.analysis_model_id,
        "repo_id": profile.sae_repo_id,
        "dtype": dtype,
        "device_map": model_cfg.get("device_map", "auto"),
        "sae_device": sae_device,
        "sae_dtype": model_cfg.get("sae_dtype", dtype),
    }


def _condition_of(case_id: str, conditions: Sequence[str]) -> str:
    for name in conditions:
        if case_id.endswith(f"_{name}"):
            return name
    return "unknown"


def run(config_path: Path, output_root: Path) -> Path:
    config = load_toml(config_path)
    name = run_name(config)
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config.toml")

    feature_cfg = table(config, "feature")
    feature_id = int(feature_cfg["feature_id"])
    layer = int(feature_cfg["layer"])

    data_cfg = table(config, "data")
    dataset = str(data_cfg.get("dataset", "goal_affordance_traps_v1"))
    conditions = [
        str(c)
        for c in (
            data_cfg.get("conditions") or ["hostile", "explicit", "neutral", "counterfactual"]
        )
    ]
    positive_condition = str(data_cfg.get("positive_condition", "hostile"))
    negative_condition = str(data_cfg.get("negative_condition", "neutral"))
    instruction = bool(data_cfg.get("instruction", True))

    ref_cfg = table(config, "reference")
    ref_dataset = str(ref_cfg.get("dataset", "hagendorff_crt"))
    ref_limit = int(ref_cfg.get("limit_per_family", 10))

    env = _resolve_env(config)
    cases = [
        case
        for case in lure_dataset_cases(dataset)
        if _condition_of(case.case_id, conditions) != "unknown"
    ]
    reference = lure_dataset_cases(ref_dataset, limit_per_family=ref_limit)
    if instruction:
        cases = instruct_lure_cases(cases)
        reference = instruct_lure_cases(reference)
    # pair_id / template_id are dataset metadata the LureCase does not carry, and the
    # paraphrase check needs them.
    raw_payload = json.loads(
        (ROOT / "src" / "mindscopex_analysis" / "data" / f"{dataset}.json").read_text(
            encoding="utf-8"
        )
    )
    raw_by_id = {row["case_id"]: row for row in raw_payload["cases"]}

    manifest: dict[str, Any] = {
        "run_name": name,
        "job": "feature_falsification",
        "started_at": _timestamp(),
        "feature_id": feature_id,
        "layer": layer,
        "profile": env["profile"].key,
        "model_id": env["model_id"],
        "sae_repo_id": env["repo_id"],
        "dataset": dataset,
        "conditions": conditions,
        "n_cases": len(cases),
        "reference_dataset": ref_dataset,
        "n_reference": len(reference),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    _write_json(run_dir / "manifest.json", manifest)

    _log(f"loading {env['model_id']} (feature {feature_id} @ layer {layer})")
    lm = load_qwen_language_model(
        env["model_id"], device_map=env["device_map"], dtype=env["dtype"], dispatch=True
    )
    sae = load_qwen_scope_sae(
        env["repo_id"], layer, device=env["sae_device"], dtype=dtype_from_name(env["sae_dtype"])
    )

    def _activation(prompt: str) -> tuple[float, float]:
        residual = capture_layer_residuals(lm, [prompt], layer, token_position="last")
        sparse = float(
            qwen_scope_sparse_feature_values(residual, sae, [feature_id])
            .detach()
            .to(torch.float32)
            .reshape(-1)[0]
        )
        preact = float(
            qwen_scope_feature_preactivations(residual, sae, [feature_id])
            .detach()
            .to(torch.float32)
            .reshape(-1)[0]
        )
        return sparse, preact

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        raw = raw_by_id.get(case.case_id, {})
        sparse, preact = _activation(case.prompt)
        rows.append(
            {
                "group": "trap",
                "case_id": case.case_id,
                "pair_id": raw.get("pair_id", ""),
                "template_id": raw.get("template_id", ""),
                "family": case.family,
                "condition": _condition_of(case.case_id, conditions),
                "activation": sparse,
                "preactivation": preact,
                "fires": sparse > 0,
                "answer_pair": f"{case.correct_answer.strip()}|{case.lure_answer.strip()}",
                "answer_len_delta": (
                    len(case.lure_answer.split()) - len(case.correct_answer.split())
                ),
            }
        )
        if index % 40 == 0:
            _log(f"trap {index}/{len(cases)}")

    for index, case in enumerate(reference, start=1):
        sparse, preact = _activation(case.prompt)
        rows.append(
            {
                "group": "reference",
                "case_id": case.case_id,
                "pair_id": "",
                "template_id": "",
                "family": case.family,
                "condition": "reference",
                "activation": sparse,
                "preactivation": preact,
                "fires": sparse > 0,
                "answer_pair": f"{case.correct_answer.strip()}|{case.lure_answer.strip()}",
                "answer_len_delta": (
                    len(case.lure_answer.split()) - len(case.correct_answer.split())
                ),
            }
        )
        if index % 20 == 0:
            _log(f"reference {index}/{len(reference)}")

    _write_csv(
        run_dir / "falsification_activations.csv",
        rows,
        [
            "group",
            "case_id",
            "pair_id",
            "template_id",
            "family",
            "condition",
            "activation",
            "preactivation",
            "fires",
            "answer_pair",
            "answer_len_delta",
        ],
    )

    by_condition: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(float(row["activation"]))

    positive = by_condition.get(positive_condition, [])
    negative = by_condition.get(negative_condition, [])
    reference_values = by_condition.get("reference", [])

    # Paraphrase: same structure, different wording. Only meaningful when the set
    # actually varies template_id within a condition.
    templates: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["condition"] == positive_condition and row["template_id"]:
            templates[str(row["template_id"])].append(float(row["activation"]))
    paraphrase = (
        {
            "n_templates": len(templates),
            "per_template_mean": {k: _mean(v) for k, v in sorted(templates.items())},
            "spread": (
                max(_mean(v) for v in templates.values())
                - min(_mean(v) for v in templates.values())
                if len(templates) > 1
                else None
            ),
        }
        if len(templates) > 1
        else {"n_templates": len(templates), "note": "this set does not vary template_id"}
    )

    # Answer-string confounds: the two surface variables that already distorted the
    # margin on this dataset.
    trap_rows = [row for row in rows if row["group"] == "trap"]
    confound = {
        "corr_activation_vs_answer_len_delta": pearson(
            [float(r["activation"]) for r in trap_rows],
            [float(r["answer_len_delta"]) for r in trap_rows],
        ),
        "n_distinct_answer_pairs": len({r["answer_pair"] for r in trap_rows}),
    }

    # Threshold from the DISCOVERY split only, then audited on everything.
    train, _ = split_lure_cases(
        [case for case in cases if _condition_of(case.case_id, conditions) == positive_condition],
        train_frac=float(data_cfg.get("train_frac", 0.6)),
        seed=int(data_cfg.get("split_seed", 0)),
    )
    train_ids = {case.case_id for case in train}
    train_values = sorted(float(row["activation"]) for row in rows if row["case_id"] in train_ids)
    threshold = train_values[len(train_values) // 4] if train_values else 0.0

    errors = {
        "threshold": threshold,
        "threshold_source": f"25th percentile of {positive_condition} on the discovery split",
        "false_negatives": [
            {k: row[k] for k in ("case_id", "condition", "activation", "family")}
            for row in rows
            if row["condition"] == positive_condition and float(row["activation"]) < threshold
        ],
        "false_positives": [
            {k: row[k] for k in ("case_id", "condition", "activation", "family")}
            for row in rows
            if row["condition"] in {negative_condition, "reference"}
            and float(row["activation"]) >= threshold
        ],
    }
    _write_json(run_dir / "falsification_errors.json", errors)

    summary = {
        "feature_id": feature_id,
        "layer": layer,
        "condition_means": {k: _mean(v) for k, v in sorted(by_condition.items())},
        "condition_fire_rate": {
            condition: _mean(
                [
                    1.0 if float(r["activation"]) > 0 else 0.0
                    for r in rows
                    if r["condition"] == condition
                ]
            )
            for condition in sorted(by_condition)
        },
        # The headline dissociation: same words and template, different structure.
        "structure_auc": _auc(positive, negative),
        "structure_gap": _mean(positive) - _mean(negative),
        # Equal activation on an unrelated task means the feature tracks the template.
        "template_auc": _auc(positive, reference_values),
        "template_ratio": (_mean(reference_values) / _mean(positive) if _mean(positive) else None),
        "paraphrase": paraphrase,
        "answer_confound": confound,
        "n_false_negatives": len(errors["false_negatives"]),
        "n_false_positives": len(errors["false_positives"]),
    }
    _write_json(run_dir / "falsification_summary.json", summary)
    manifest.update({"finished_at": _timestamp(), "summary": summary})
    _write_json(run_dir / "manifest.json", manifest)
    _log(
        f"structure AUC {summary['structure_auc']} | template ratio "
        f"{summary['template_ratio']} | FN {summary['n_false_negatives']} "
        f"FP {summary['n_false_positives']}"
    )
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
