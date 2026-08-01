"""Build the curated 60-scenario Goal-Affordance Traps v1 dataset.

The first build selects from saved development proposals. Once written, the
canonical JSON is self-contained: later rebuilds can expand its
``scenario_source`` without requiring API-generation artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from goal_affordance_data import SEED_SCENARIOS, Scenario, expand_scenarios

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_DIR = ROOT / "results" / "goal_affordance_development"
DESTINATION = (
    ROOT / "src" / "mindscopex_analysis" / "data" / "goal_affordance_traps_v1.json"
)

SELECTED_GENERATED_IDS = {
    # Three GPT + three Claude + two Gemini proposals per family.
    "target_transport_cat_exam_room",
    "target_transport_museum_model_lab",
    "target_transport_citrus_tree_greenhouse",
    "target_transport_cello_recital",
    "target_transport_foster_kitten_vet",
    "target_transport_sourdough_starter",
    "target_transport_electronics",
    "target_transport_donation",
    "tool_transport_probe_calibration_kit",
    "tool_transport_inspection_mirror_serial",
    "tool_transport_triangular_driver_guard",
    "tool_transport_torque_wrench_bleachers",
    "tool_transport_beehive_smoker",
    "tool_transport_pipette_freezer_samples",
    "tool_transport_maintenance",
    "tool_transport_landscaping",
    "required_resource_conference_badge",
    "required_resource_signed_release_archive",
    "required_resource_label_roll_stockroom",
    "required_resource_kiln_firing_log",
    "required_resource_grout_sealant",
    "required_resource_notary_signature",
    "required_resource_credential",
    "required_resource_locker",
    "agent_capability_named_form_certifier",
    "agent_capability_pallet_operator",
    "agent_capability_fluent_intake_staff",
    "agent_capability_lifeguard_pool_reopen",
    "agent_capability_translator_deposition",
    "agent_capability_specialist",
    "agent_capability_legal",
    "agent_capability_technical",
    "prerequisite_state_quiet_recording_room",
    "prerequisite_state_dry_scanner_glass",
    "prerequisite_state_dry_base_coat",
    "prerequisite_state_aquarium_dechlorinate",
    "prerequisite_state_construction",
    "prerequisite_state_artistic",
    "means_end_conflict_surprise_address",
    "means_end_conflict_undiluted_lemonade",
    "means_end_conflict_archive_backup_move",
    "means_end_conflict_quiet_hallway_recording",
    "means_end_conflict_seed_saving_hybrid",
    "means_end_conflict_night_shift_handover",
    "means_end_conflict_preservation",
    "means_end_conflict_integrity",
}

CURATION_REVISIONS: dict[str, dict[str, str]] = {
    "target_transport_cat_exam_room": {
        "counterfactual_goal": (
            "Get yourself alone to the examination room as quickly as possible."
        ),
        "lure_action": "Walk through the staff door alone.",
    },
    "target_transport_foster_kitten_vet": {
        "counterfactual_goal": (
            "Go to the clinic alone now to claim the early slot before bringing the kitten."
        ),
        "lure_action": "Walk to the clinic alone now to take the open slot.",
    },
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
    "tool_transport_landscaping": {
        "neutral_context": (
            "The high branches require long-reach pole shears, which are locked in "
            "the backyard garden shed."
        ),
        "precondition": "The pole shears must be brought to the driveway before pruning.",
        "counterfactual_goal": (
            "Inspect from below which high branches currently overhang the driveway."
        ),
    },
    "required_resource_credential": {
        "neutral_context": (
            "The archive room requires a physical key card, and your card is on your "
            "dresser at home."
        ),
        "precondition": "You must have the physical card at the locked archive door.",
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
    "agent_capability_technical": {
        "neutral_context": (
            "The contract requires a certified medical translator. Your nearby friend "
            "speaks conversational Japanese but has no medical or legal training."
        ),
        "precondition": (
            "A certified medical translator must handle the technical contract."
        ),
    },
    "agent_capability_specialist": {
        "neutral_context": (
            "Removing the limb beside the power lines requires a certified arborist. "
            "A nearby handyman has a chainsaw but is not certified for utility work."
        ),
        "precondition": (
            "A certified arborist must perform the power-line limb removal."
        ),
        "counterfactual_goal": (
            "Hire the nearby handyman for an ordinary yard job away from the power lines."
        ),
    },
    "agent_capability_translator_deposition": {
        "counterfactual_goal": (
            "Let the bilingual paralegal translate the witness's next answer now "
            "as an informal preview, not as sworn testimony."
        ),
    },
    "prerequisite_state_construction": {
        "neutral_context": (
            "The deck was soaked by heavy rain two hours ago, and this stain bonds "
            "only after the wood has been completely dry for twenty-four hours."
        ),
        "precondition": "The deck must finish drying before the stain is applied.",
        "counterfactual_goal": (
            "Apply some stain to the wet deck now to observe how poorly it bonds."
        ),
        "lure_action": "Start staining the deck now.",
    },
    "prerequisite_state_artistic": {
        "neutral_context": (
            "The mold needs release agent to prevent sticking, but none has been "
            "applied and the plaster is already mixed."
        ),
        "precondition": "Release agent must coat the mold before the plaster is poured.",
        "counterfactual_goal": (
            "Make a plaster cast that intentionally sticks permanently inside this mold."
        ),
    },
    "prerequisite_state_aquarium_dechlorinate": {
        "counterfactual_goal": (
            "Stop the pump running for now without adding any water to the aquarium."
        ),
        "lure_action": "Unplug the pump for now.",
    },
    "prerequisite_state_firmware_battery": {
        "salient_cue": (
            "Opening the update screen now feels faster than waiting for the battery "
            "to charge."
        ),
        "counterfactual_goal": (
            "Open the update screen now only to verify that it loads; do not begin "
            "the installation."
        ),
        "lure_action": "Open the update screen now.",
    },
    "prerequisite_state_frozen_sample_thaw": {
        "counterfactual_goal": (
            "Shake the sealed vial once right now only to check whether the ice moves, "
            "not to mix the solution."
        ),
    },
    "means_end_conflict_preservation": {
        "neutral_context": (
            "The fragile map is flaking at the edges, and the fast automatic feeder "
            "often snags and tears thin aged paper."
        ),
        "precondition": "The map must not pass through a mechanism that may tear it.",
        "counterfactual_goal": (
            "Digitize the map as fast as possible even if the original may be torn."
        ),
    },
    "means_end_conflict_integrity": {
        "neutral_context": (
            "The antique floor is dusty, and a steam mop would clean quickly but "
            "strip off its original wax finish."
        ),
        "precondition": "The cleaning method must leave the original wax finish intact.",
        "counterfactual_goal": (
            "Remove the dust and the old wax finish as quickly as possible."
        ),
    },
}

HUMAN_ADDITIONS = (
    Scenario(
        scenario_id="prerequisite_state_adhesive_cure",
        family="prerequisite_state",
        goal="I want the new adhesive hook to hold a weighted test pouch for a full hour.",
        neutral_context=(
            "The hook was attached one minute ago, and its adhesive needs twenty-four "
            "hours to cure before carrying weight."
        ),
        salient_cue=(
            "The pouch is ready, so hanging it now would start the one-hour test immediately."
        ),
        precondition="The adhesive must finish curing before the load test begins.",
        counterfactual_goal=(
            "Observe immediately how the uncured hook responds when the test pouch is added."
        ),
        correct_action="Wait for the adhesive to cure before hanging the pouch.",
        lure_action="Hang the test pouch on the hook now.",
        rationale=(
            "Starting the timed test early prevents the hook from satisfying its load goal."
        ),
    ),
    Scenario(
        scenario_id="prerequisite_state_frozen_sample_thaw",
        family="prerequisite_state",
        goal="I want to mix this frozen reference solution evenly before measuring it.",
        neutral_context=(
            "The sealed vial is still mostly frozen, and the protocol requires it to "
            "thaw completely before mixing."
        ),
        salient_cue=(
            "Shaking it now would begin the mixing step instead of waiting for the ice to melt."
        ),
        precondition="The solution must be fully liquid before it can be mixed evenly.",
        counterfactual_goal="Check right now whether the ice moves inside the sealed vial.",
        correct_action="Wait for the vial to thaw completely, then mix it.",
        lure_action="Shake the frozen vial now.",
        rationale="A mostly frozen vial cannot be mixed into an even solution.",
    ),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def development_scenarios() -> tuple[list[Scenario], list[dict[str, str]]]:
    scenarios = list(SEED_SCENARIOS) + list(HUMAN_ADDITIONS)
    provenance = [
        {"scenario_id": row.scenario_id, "origin": "human_seed"}
        for row in SEED_SCENARIOS
    ]
    provenance.extend(
        {"scenario_id": row.scenario_id, "origin": "human_replacement"}
        for row in HUMAN_ADDITIONS
    )
    found = set()
    for path in sorted(DEVELOPMENT_DIR.glob("candidates_*.json")):
        payload = read_json(path)
        model = payload["generator_model"]
        for raw in payload["scenarios"]:
            scenario_id = raw["scenario_id"]
            if scenario_id not in SELECTED_GENERATED_IDS:
                continue
            scenarios.append(Scenario(**raw))
            provenance.append(
                {
                    "scenario_id": scenario_id,
                    "origin": "frontier_proposal_human_curated",
                    "proposer_model": model,
                }
            )
            found.add(scenario_id)
    missing = SELECTED_GENERATED_IDS - found
    if missing:
        raise RuntimeError(f"Missing selected development proposals: {sorted(missing)}")
    scenarios = [
        Scenario(**{**scenario.__dict__, **CURATION_REVISIONS.get(scenario.scenario_id, {})})
        for scenario in scenarios
    ]
    return scenarios, provenance


def canonical_scenarios() -> tuple[list[Scenario], list[dict[str, str]]]:
    payload = read_json(DESTINATION)
    scenarios = [Scenario(**row) for row in payload["scenario_source"]]
    return scenarios, list(payload["provenance"])


def build_payload(
    scenarios: list[Scenario], provenance: list[dict[str, str]]
) -> dict[str, Any]:
    rows = expand_scenarios(scenarios, revision="v1_1")
    scenario_counts = Counter(scenario.family for scenario in scenarios)
    if scenario_counts != Counter(
        {
            "target_transport": 10,
            "tool_transport": 10,
            "required_resource": 10,
            "agent_capability": 10,
            "prerequisite_state": 10,
            "means_end_conflict": 10,
        }
    ):
        raise ValueError(f"Unexpected base-scenario balance: {dict(scenario_counts)}")
    return {
        "dataset_id": "goal_affordance_traps_v1",
        "schema_version": 3,
        "title": "Goal-Affordance Traps v1",
        "description": (
            "Sixty everyday goal/precondition scenarios, each rendered as hostile, "
            "explicit, neutral, and goal-counterfactual binary-choice conditions."
        ),
        "task_kind": "goal_affordance",
        "scoring": "binary_choice",
        "source": {
            "authors": "MindScopeX project",
            "year": 2026,
            "title": "Goal-Affordance Traps v1",
            "license": "Apache-2.0 (repository-generated and curated content)",
            "generation_note": (
                "Human seeds plus GPT/Claude/Gemini proposals; every selected scenario "
                "was human-curated and normalized before evaluation."
            ),
        },
        "generated_by": "scripts/build_goal_affordance_dataset.py",
        "revision": "v1.1",
        "n_base_scenarios": len(scenarios),
        "n_cases": len(rows),
        "family_counts": dict(
            sorted(Counter(row["family"] for row in rows).items())
        ),
        "base_family_counts": dict(sorted(scenario_counts.items())),
        "condition_counts": dict(
            sorted(Counter(row["condition"] for row in rows).items())
        ),
        "provenance": provenance,
        "scenario_source": [scenario.__dict__ for scenario in scenarios],
        "cases": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-selection",
        action="store_true",
        help="Rebuild the curated selection from saved development proposals.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if DESTINATION.is_file() and not args.refresh_selection:
        selected_scenarios, selected_provenance = canonical_scenarios()
    else:
        selected_scenarios, selected_provenance = development_scenarios()
    dataset = build_payload(selected_scenarios, selected_provenance)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{DESTINATION} | scenarios={dataset['n_base_scenarios']} | "
        f"cases={dataset['n_cases']}"
    )
