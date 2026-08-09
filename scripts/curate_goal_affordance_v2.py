"""Assemble empirically promising v2 pairs for confirmation runs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from goal_affordance_v2_data import validate_rows

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_DIR = ROOT / "results" / "goal_affordance_v2_development"
SELECTION = {
    "vehicle_inspection_ko": "candidate_pool_v0.json",
    "vehicle_tire_air_ko": "candidate_pool_v0.json",
    "agent_authorization_document_approval_account": "candidate_pool_v2_high_load.json",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_selection() -> dict[str, Any]:
    payloads = {
        filename: read_json(DEVELOPMENT_DIR / filename)
        for filename in set(SELECTION.values())
    }
    rows = []
    provenance = []
    for pair_id, filename in SELECTION.items():
        source = payloads[filename]
        pair_rows = [row for row in source["cases"] if row["pair_id"] == pair_id]
        if len(pair_rows) != 4:
            raise ValueError(f"{pair_id}: expected four rows, got {len(pair_rows)}")
        rows.extend(pair_rows)
        provenance.append(
            {
                "pair_id": pair_id,
                "source_dataset_id": source["dataset_id"],
                "source_file": filename,
            }
        )
    validate_rows(rows)
    return {
        "dataset_id": "goal_affordance_traps_v2_confirmation_v0",
        "schema_version": 3,
        "title": "Goal-Affordance v2 empirical confirmation candidates",
        "description": (
            "Three ambiguity-screened pairs that showed at least one intuitive-to-"
            "reflective recovery in initial frontier evaluation."
        ),
        "task_kind": "goal_affordance",
        "scoring": "binary_choice",
        "revision": "confirmation_v0",
        "n_base_surfaces": len(SELECTION),
        "n_cases": len(rows),
        "condition_counts": dict(
            sorted(Counter(row["condition"] for row in rows).items())
        ),
        "family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
        "provenance": provenance,
        "cases": rows,
    }


if __name__ == "__main__":
    output = DEVELOPMENT_DIR / "confirmation_v0.json"
    payload = build_selection()
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{output} | pairs={payload['n_base_surfaces']} | cases={payload['n_cases']}")
