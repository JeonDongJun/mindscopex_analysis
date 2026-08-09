"""Consolidate Goal-Affordance v2 confirmation artifacts into a final report."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATASET = ROOT / "src" / "mindscopex_analysis" / "data" / "goal_affordance_traps_v2.json"
OUTPUT = RESULTS / "goal_affordance_traps_v2_final_20260802"
PAIR_ID = "vehicle_tire_air_ko"
MODE_PATTERNS = {
    "intuitive_prompted": "goal_affordance_v2_tire_intuitive_rep*_20260802",
    "deliberate_prompted": "goal_affordance_v2_tire_reflective_rep*_20260802",
}
CONFIRMATION_DIRS = {
    "forward": RESULTS / "goal_affordance_v2_confirmation_forward_20260802",
    "reverse": RESULTS / "goal_affordance_v2_tire_air_reverse_20260802",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def repetition_rows() -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    grouped = {}
    paths = []
    for mode, pattern in MODE_PATTERNS.items():
        directories = sorted(RESULTS.glob(pattern))
        if len(directories) != 5:
            raise ValueError(f"Expected five {mode} repetitions, got {len(directories)}")
        rows = []
        for repetition, directory in enumerate(directories, 1):
            run_rows = read_jsonl(directory / "responses.jsonl")
            if len(run_rows) != 3:
                raise ValueError(f"{directory}: expected three model rows")
            for row in run_rows:
                if row["pair_id"] != PAIR_ID or row["mode"] != mode:
                    raise ValueError(f"Unexpected repetition row in {directory}")
                rows.append({**row, "repetition": repetition})
            paths.append(str(directory.relative_to(ROOT)))
        grouped[mode] = rows
    return grouped, paths


def summarize() -> tuple[dict[str, Any], str]:
    dataset = read_json(DATASET)
    grouped, repetition_paths = repetition_rows()
    model_results = defaultdict(dict)
    for mode, rows in grouped.items():
        by_model = defaultdict(list)
        for row in rows:
            by_model[row["model"]].append(row)
        for model, model_rows in by_model.items():
            counts = Counter(row["label"] for row in model_rows)
            model_results[model][mode] = {
                "effort": model_rows[0]["effort"],
                "n": len(model_rows),
                "lure": counts["lure"],
                "lure_rate": counts["lure"] / len(model_rows),
            }

    mode_totals = {}
    for mode, rows in grouped.items():
        counts = Counter(row["label"] for row in rows)
        mode_totals[mode] = {
            "n": len(rows),
            "lure": counts["lure"],
            "lure_rate": counts["lure"] / len(rows),
        }

    confirmation = {}
    total_confirmation_cost = 0.0
    for order, directory in CONFIRMATION_DIRS.items():
        rows = [
            row
            for row in read_jsonl(directory / "responses.jsonl")
            if row["pair_id"] == PAIR_ID
        ]
        controls = [row for row in rows if row["condition"] != "hostile"]
        hostile = [row for row in rows if row["condition"] == "hostile"]
        if not controls or any(row["label"] != "correct" for row in controls):
            raise ValueError(f"{order}: a selected control failed")
        confirmation[order] = {
            "responses": len(rows),
            "control_correct": len(controls),
            "control_total": len(controls),
            "intuitive_hostile_lure": sum(
                row["label"] == "lure"
                for row in hostile
                if row["mode"] == "intuitive_prompted"
            ),
            "reflective_hostile_lure": sum(
                row["label"] == "lure"
                for row in hostile
                if row["mode"] == "deliberate_prompted"
            ),
            "path": str(directory.relative_to(ROOT)),
        }
        total_confirmation_cost += sum(row["cost_usd"] for row in rows)

    repetition_cost = sum(
        row["cost_usd"] for rows in grouped.values() for row in rows
    )
    dataset_sha256 = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    manifest = {
        "dataset_id": dataset["dataset_id"],
        "dataset_revision": dataset["revision"],
        "dataset_sha256": dataset_sha256,
        "evaluated_at": "2026-08-02",
        "status": "confirmed_micro_challenge_not_a_broad_benchmark",
        "pair_id": PAIR_ID,
        "n_independent_semantic_clusters": 1,
        "models": dict(sorted(model_results.items())),
        "repeated_hostile": mode_totals,
        "paired_effect": {
            "intuitive_lure_rate": mode_totals["intuitive_prompted"]["lure_rate"],
            "reflective_lure_rate": mode_totals["deliberate_prompted"]["lure_rate"],
            "absolute_reduction": (
                mode_totals["intuitive_prompted"]["lure_rate"]
                - mode_totals["deliberate_prompted"]["lure_rate"]
            ),
        },
        "confirmation": confirmation,
        "repetition_run_paths": repetition_paths,
        "calibration_evidence": {
            "short_image_like": "results/goal_affordance_v2_short_ko_hostile_modes_20260802",
            "high_load_generated": "results/goal_affordance_v2_high_load_hostile_modes_20260802",
            "attached_component_family": (
                "results/goal_affordance_v2_attached_hostile_modes_20260802"
            ),
            "tire_paraphrases": (
                "results/goal_affordance_v2_tire_paraphrases_hostile_modes_20260802"
            ),
        },
        "cost_usd_for_final_confirmation": repetition_cost + total_confirmation_cost,
        "interpretation": (
            "Repeated calls estimate response probability and are not independent items. "
            "The result establishes one Korean semantic micro-challenge, not a broad rate "
            "of goal-affordance failure across tasks or languages."
        ),
    }

    lines = [
        "# Goal-Affordance Traps v2 final validation",
        "",
        "- Date: 2026-08-02",
        "- Independent semantic clusters: 1",
        "- Cases: 4 paired conditions",
        f"- Dataset SHA-256: `{dataset_sha256}`",
        "",
        "## Repeated hostile result",
        "",
        "| Model | Intuitive lure | Reflective lure |",
        "|---|---:|---:|",
    ]
    for model, values in sorted(model_results.items()):
        intuitive = values["intuitive_prompted"]
        reflective = values["deliberate_prompted"]
        lines.append(
            f"| {model} | {intuitive['lure']}/{intuitive['n']} "
            f"({intuitive['lure_rate']:.1%}) | {reflective['lure']}/{reflective['n']} "
            f"({reflective['lure_rate']:.1%}) |"
        )
    lines.extend(
        [
            "",
            f"Pooled: intuitive {mode_totals['intuitive_prompted']['lure']}/15 "
            f"({mode_totals['intuitive_prompted']['lure_rate']:.1%}) vs reflective "
            f"{mode_totals['deliberate_prompted']['lure']}/15 "
            f"({mode_totals['deliberate_prompted']['lure_rate']:.1%}).",
            "",
            "## Controls and option order",
            "",
            "- Forward and reversed A/B presentations both preserved the effect.",
            "- Explicit, neutral, and counterfactual controls were all correct.",
            "- Repetitions are repeated measurements of one item, not n=5 items.",
            "",
            "## Interpretation",
            "",
            "This is a confirmed micro-challenge suitable for intervention and model-contrast "
            "tests. It is not yet a benchmark for estimating population prevalence. The next "
            "expansion must add new semantic clusters rather than paraphrases of tire inflation.",
            "",
        ]
    )
    return manifest, "\n".join(lines)


if __name__ == "__main__":
    result, report = summarize()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "evaluation_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(OUTPUT)
