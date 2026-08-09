"""Build the confirmed Goal-Affordance Traps v2 micro-challenge dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from goal_affordance_v2_data import validate_rows

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "results"
    / "goal_affordance_v2_development"
    / "candidate_pool_v4_tire_paraphrases_ko.json"
)
DESTINATION = (
    ROOT / "src" / "mindscopex_analysis" / "data" / "goal_affordance_traps_v2.json"
)
SELECTED_PAIR = "vehicle_tire_air_ko"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_payload() -> dict[str, Any]:
    source = read_json(SOURCE)
    rows = [row for row in source["cases"] if row["pair_id"] == SELECTED_PAIR]
    if len(rows) != 4:
        raise ValueError(f"Expected four selected rows, got {len(rows)}")
    rows = [
        {
            **row,
            "template_id": "v2_micro_challenge_attached_tire_ko",
            "revision": "v2.0",
            "note": (
                "confirmed_micro_challenge; one_semantic_cluster; "
                "repeated_calls_are_not_independent_items"
            ),
        }
        for row in rows
    ]
    validate_rows(rows)
    return {
        "dataset_id": "goal_affordance_traps_v2",
        "schema_version": 3,
        "title": "Goal-Affordance Traps v2 micro-challenge",
        "description": (
            "A Korean four-condition micro-challenge in which local travel ease "
            "conflicts with moving the vehicle-bound target of the goal."
        ),
        "task_kind": "goal_affordance",
        "scoring": "binary_choice",
        "source": {
            "authors": "MindScopeX project",
            "year": 2026,
            "title": "Goal-Affordance Traps v2 micro-challenge",
            "license": "Apache-2.0 (repository-generated and curated content)",
            "generation_note": (
                "Selected after bilingual short-form, high-load, attached-component, "
                "paraphrase, control, repetition, and option-reversal screening."
            ),
        },
        "generated_by": "scripts/build_goal_affordance_v2_dataset.py",
        "revision": "v2.0",
        "status": "confirmed_micro_challenge_not_a_broad_benchmark",
        "language": "ko",
        "n_independent_semantic_clusters": 1,
        "n_base_surfaces": 1,
        "n_cases": len(rows),
        "family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
        "condition_counts": dict(
            sorted(Counter(row["condition"] for row in rows).items())
        ),
        "selection": {
            "pair_id": SELECTED_PAIR,
            "intuitive_prompted_repetitions": 5,
            "reflective_prompted_repetitions": 5,
            "intuitive_hostile_lure": {
                "openai/gpt-5.6-sol": "0/5",
                "anthropic/claude-opus-5": "5/5",
                "google/gemini-3-flash-preview": "3/5",
                "pooled": "8/15",
            },
            "reflective_hostile_lure": {
                "openai/gpt-5.6-sol": "0/5",
                "anthropic/claude-opus-5": "0/5",
                "google/gemini-3-flash-preview": "0/5",
                "pooled": "0/15",
            },
            "controls": "all correct in confirmation and option-reversal runs",
            "option_reversal_confirmed": True,
        },
        "cases": rows,
    }


if __name__ == "__main__":
    payload = build_payload()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{DESTINATION} | clusters=1 | cases={payload['n_cases']}")
