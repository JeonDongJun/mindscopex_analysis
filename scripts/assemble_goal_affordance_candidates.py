"""Assemble generated Goal-Affordance proposals into a development pool."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from goal_affordance_data import Scenario, expand_scenarios

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_DIR = ROOT / "results" / "goal_affordance_development"
FAIRNESS_REVISIONS = {
    "target_transport_electronics": {
        "neutral_context": (
            "The safe is in the fourth-floor records room, while the laptop is still "
            "in the trunk of your car in the basement garage."
        ),
        "precondition": "The laptop itself has to be brought to the safe to be secured.",
    },
    "target_transport_donation": {
        "neutral_context": (
            "The shelter accepts donations until five, and the box is still on your "
            "porch at home."
        ),
        "precondition": "The box itself has to arrive at the shelter for the drop-off.",
    },
    "tool_transport_maintenance": {
        "neutral_context": (
            "The bookshelf is across the hall, but the spirit level needed to adjust "
            "it is in the attic toolbox."
        ),
        "precondition": "The bookshelf cannot be leveled accurately without the spirit level.",
    },
    "required_resource_credential": {
        "neutral_context": (
            "The archive room requires a physical key card, and your card is on your "
            "dresser at home."
        ),
        "precondition": "You must have the physical card at the locked archive door.",
    },
    "required_resource_membership": {
        "neutral_context": (
            "The club requires a physical membership card at the entrance, and your "
            "card is in your spare wallet at home."
        ),
        "precondition": "You must bring the physical membership card to enter and buy the item.",
    },
    "required_resource_locker": {
        "neutral_context": (
            "The locker requires a physical library card, and your card is on your "
            "desk at home."
        ),
        "precondition": "You must have the physical library card at the locker to open it.",
    },
    "agent_capability_legal": {
        "neutral_context": (
            "Your assistant is in the next room but is not a notary; a commissioned "
            "notary is available at the bank."
        ),
        "precondition": "Only a commissioned notary can validly notarize the affidavit.",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assemble(paths: list[Path]) -> dict[str, Any]:
    scenarios = []
    provenance = []
    seen = set()
    for path in paths:
        payload = read_json(path)
        model = payload["generator_model"]
        for row in payload["scenarios"]:
            scenario = Scenario(**row)
            if scenario.scenario_id in seen:
                raise ValueError(f"Duplicate scenario ID: {scenario.scenario_id}")
            seen.add(scenario.scenario_id)
            scenarios.append(scenario)
            provenance.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "generator_model": model,
                    "source_file": path.name,
                }
            )
    rows = expand_scenarios(scenarios, revision="candidate_pool_v0")
    counts = Counter(scenario.family for scenario in scenarios)
    if len(set(counts.values())) != 1:
        raise ValueError(f"Candidate pool is not family-balanced: {dict(counts)}")
    return {
        "dataset_id": "goal_affordance_candidate_pool_v0",
        "schema_version": 3,
        "title": "Goal-Affordance generated candidate pool",
        "description": (
            "Uncurated development proposals. Do not use as the final benchmark."
        ),
        "task_kind": "goal_affordance",
        "scoring": "binary_choice",
        "revision": "candidate_pool_v0",
        "n_base_scenarios": len(scenarios),
        "n_cases": len(rows),
        "family_counts": dict(sorted(counts.items())),
        "condition_counts": dict(sorted(Counter(row["condition"] for row in rows).items())),
        "provenance": provenance,
        "scenario_source": [scenario.__dict__ for scenario in scenarios],
        "cases": rows,
    }


def fairness_revision_pool(payload: dict[str, Any]) -> dict[str, Any]:
    revised = []
    provenance = {row["scenario_id"]: row for row in payload["provenance"]}
    source = {row["scenario_id"]: row for row in payload["scenario_source"]}
    for scenario_id, changes in FAIRNESS_REVISIONS.items():
        row = {**source[scenario_id], **changes}
        revised.append(Scenario(**row))
    rows = [
        row
        for scenario in revised
        for row in expand_scenarios([scenario], revision="fairness_revision_v1")
    ]
    return {
        "dataset_id": "goal_affordance_fairness_revision_v1",
        "schema_version": 3,
        "title": "Goal-Affordance fairness-revised challenge candidates",
        "description": (
            "Previously failed candidates revised so hostile questions state every "
            "fact needed to determine the answer."
        ),
        "task_kind": "goal_affordance",
        "scoring": "binary_choice",
        "revision": "fairness_revision_v1",
        "n_base_scenarios": len(revised),
        "n_cases": len(rows),
        "provenance": [provenance[scenario.scenario_id] for scenario in revised],
        "scenario_source": [scenario.__dict__ for scenario in revised],
        "cases": rows,
    }


if __name__ == "__main__":
    sources = sorted(DEVELOPMENT_DIR.glob("candidates_*.json"))
    if not sources:
        raise RuntimeError("No generated candidate files found")
    output = DEVELOPMENT_DIR / "candidate_pool_v0.json"
    payload = assemble(sources)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    revised_output = DEVELOPMENT_DIR / "fairness_revision_v1.json"
    revised_payload = fairness_revision_pool(payload)
    revised_output.write_text(
        json.dumps(revised_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{output} | scenarios={payload['n_base_scenarios']} | "
        f"cases={payload['n_cases']}"
    )
    print(
        f"{revised_output} | scenarios={revised_payload['n_base_scenarios']} | "
        f"cases={revised_payload['n_cases']}"
    )
