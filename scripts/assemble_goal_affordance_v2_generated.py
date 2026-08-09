"""Assemble saved high-load v2 proposals into a development evaluation pool."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from goal_affordance_v2_data import validate_rows

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_DIR = ROOT / "results" / "goal_affordance_v2_development"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assemble(paths: list[Path]) -> dict[str, Any]:
    rows = []
    sources = []
    seen = set()
    for path in paths:
        payload = read_json(path)
        proposer = payload["generator_model"]
        for scenario in payload["scenarios"]:
            scenario_id = scenario["scenario_id"]
            if scenario_id in seen:
                raise ValueError(f"Duplicate scenario ID: {scenario_id}")
            seen.add(scenario_id)
            sources.append(
                {
                    **scenario,
                    "proposer_model": proposer,
                    "source_file": path.name,
                }
            )
            common = {
                "pair_id": scenario_id,
                "semantic_id": scenario_id,
                "template_id": f"v2_high_load_{scenario['family']}",
                "language": "ko",
                "family": scenario["family"],
                "heuristic": scenario["heuristic"],
                "rationale": scenario["rationale"],
                "critical_fact": scenario["critical_fact"],
                "ambiguity_check": scenario["ambiguity_check"],
                "proposer_model": proposer,
                "revision": "candidate_v2_high_load",
            }
            for condition, question, correct, lure in (
                (
                    "hostile",
                    scenario["hostile_question"],
                    scenario["correct_action"],
                    scenario["lure_action"],
                ),
                (
                    "explicit",
                    scenario["explicit_question"],
                    scenario["correct_action"],
                    scenario["lure_action"],
                ),
                (
                    "neutral",
                    scenario["neutral_question"],
                    scenario["correct_action"],
                    scenario["lure_action"],
                ),
                (
                    "counterfactual",
                    scenario["counterfactual_question"],
                    scenario["lure_action"],
                    scenario["correct_action"],
                ),
            ):
                rows.append(
                    {
                        **common,
                        "case_id": f"{scenario_id}_{condition}",
                        "condition": condition,
                        "question": question,
                        "correct_answer": correct,
                        "lure_answer": lure,
                        "note": "frontier_proposed_v2_candidate",
                    }
                )
    validate_rows(rows)
    return {
        "dataset_id": "goal_affordance_traps_v2_candidate_v2_high_load",
        "schema_version": 3,
        "title": "Goal-Affordance v2 high-load generated candidate pool",
        "description": "Uncurated Korean development candidates from frontier proposers.",
        "task_kind": "goal_affordance",
        "scoring": "binary_choice",
        "revision": "candidate_v2_high_load",
        "n_base_surfaces": len(sources),
        "n_cases": len(rows),
        "family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
        "condition_counts": dict(
            sorted(Counter(row["condition"] for row in rows).items())
        ),
        "proposer_counts": dict(
            sorted(Counter(row["proposer_model"] for row in sources).items())
        ),
        "surface_source": sources,
        "cases": rows,
    }


if __name__ == "__main__":
    paths = sorted(DEVELOPMENT_DIR.glob("generated_high_load_*.json"))
    if not paths:
        raise RuntimeError("No successful high-load candidate files found")
    output = DEVELOPMENT_DIR / "candidate_pool_v2_high_load.json"
    payload = assemble(paths)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{output} | surfaces={payload['n_base_surfaces']} | "
        f"cases={payload['n_cases']}"
    )
