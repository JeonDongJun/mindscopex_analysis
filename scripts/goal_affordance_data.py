"""Source definitions and deterministic condition expansion for Goal-Affordance.

Development proceeds on a small seed before the 60-scenario v1 is frozen. Each
base scenario expands to four rows sharing one ``pair_id``:

* hostile: a salient efficiency cue favors the lure;
* explicit: the hidden goal precondition is stated;
* neutral: the salient cue is removed;
* counterfactual: the goal changes so the original lure becomes correct.

The choices are stored as action text, not A/B labels. Evaluators randomize the
display order independently for each model/mode.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    family: str
    goal: str
    neutral_context: str
    salient_cue: str
    precondition: str
    counterfactual_goal: str
    correct_action: str
    lure_action: str
    rationale: str


SEED_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        scenario_id="target_transport_car_wash",
        family="target_transport",
        goal="I want to have my car washed at the car wash nearby.",
        neutral_context="The car is parked beside me and the car wash is 50 meters away.",
        salient_cue=(
            "For such a short distance, walking there would normally be quicker than "
            "starting the car, pulling out, and parking again."
        ),
        precondition="The car itself must arrive at the car wash for it to be washed.",
        counterfactual_goal=(
            "I only want to ask the car wash about prices; I am not having the car washed now."
        ),
        correct_action="drive there",
        lure_action="walk there",
        rationale="The object being acted on—the car—must be transported to the service.",
    ),
    Scenario(
        scenario_id="target_transport_parcel_return",
        family="target_transport",
        goal="I want the nearby courier counter to accept my parcel return.",
        neutral_context="The sealed parcel is beside me and the counter is downstairs.",
        salient_cue=(
            "The parcel is bulky, and going downstairs empty-handed would be faster and easier."
        ),
        precondition="The counter cannot accept the return unless the parcel is brought there.",
        counterfactual_goal=(
            "I only want to ask the counter whether this type of purchase is returnable."
        ),
        correct_action="take the parcel with me to the counter",
        lure_action="go straight to the counter",
        rationale="The return target must be present for the return to be accepted.",
    ),
    Scenario(
        scenario_id="tool_transport_display_adapter",
        family="tool_transport",
        goal="I want to show my laptop slides on the meeting-room projector.",
        neutral_context=(
            "The meeting room is across the hall, and the required display adapter is on my desk."
        ),
        salient_cue=(
            "The meeting is starting now, so walking straight into the room is the quickest move."
        ),
        precondition="The laptop cannot connect to that projector without the display adapter.",
        counterfactual_goal=(
            "I only want to tell the attendees that the presentation will start later."
        ),
        correct_action="pick up the adapter before entering the meeting room",
        lure_action="go straight into the meeting room",
        rationale="A required tool must accompany the agent to complete the display goal.",
    ),
    Scenario(
        scenario_id="tool_transport_bicycle_pump",
        family="tool_transport",
        goal="I want to inflate the bicycle tire in the courtyard.",
        neutral_context="The bicycle is in the courtyard and the only pump is in my apartment.",
        salient_cue=(
            "The courtyard is very close, and carrying the long pump down the stairs is awkward."
        ),
        precondition="There is no other pump or inflation device in the courtyard.",
        counterfactual_goal="I only want to inspect the tire to see whether it looks flat.",
        correct_action="bring the pump down to the courtyard",
        lure_action="go straight to the courtyard",
        rationale="The inflation tool is unavailable at the destination unless it is transported.",
    ),
    Scenario(
        scenario_id="required_resource_pharmacy_id",
        family="required_resource",
        goal="I want to collect my prescription from the pharmacy.",
        neutral_context="The pharmacy is nearby, but my required photo ID is upstairs.",
        salient_cue=(
            "Going directly to the pharmacy avoids backtracking and gets me to the counter sooner."
        ),
        precondition="The pharmacy requires the photo ID before releasing this prescription.",
        counterfactual_goal="I only want to ask the pharmacy when the prescription will be ready.",
        correct_action="go upstairs for the photo ID first",
        lure_action="go directly to the pharmacy",
        rationale="The transaction requires a resource that must be obtained before arrival.",
    ),
    Scenario(
        scenario_id="required_resource_train_ticket",
        family="required_resource",
        goal="I want to board my train using my paper ticket.",
        neutral_context="The station entrance is close, but the paper ticket is on my desk.",
        salient_cue=(
            "Walking straight to the station is the shortest route and the train leaves soon."
        ),
        precondition="There is no digital copy or replacement ticket available at the station.",
        counterfactual_goal="I only want to check the departure board outside the station.",
        correct_action="go back for the paper ticket first",
        lure_action="head straight to the station",
        rationale="Boarding requires the ticket even though retrieving it adds travel time.",
    ),
    Scenario(
        scenario_id="agent_capability_account_owner",
        family="agent_capability",
        goal="We want the bank to close an account that belongs to Mina.",
        neutral_context="Mina is ten minutes away; her friend is already beside the bank.",
        salient_cue=(
            "The friend could reach the counter immediately, while waiting for Mina takes longer."
        ),
        precondition="The bank requires the account owner to authorize the closure in person.",
        counterfactual_goal="We only want someone to pick up a leaflet about account closure.",
        correct_action="wait for Mina and have her go to the bank",
        lure_action="send the nearby friend to the bank now",
        rationale="The selected agent must possess the authority required by the goal.",
    ),
    Scenario(
        scenario_id="agent_capability_electrical_fault",
        family="agent_capability",
        goal="We want the recurring electrical fault professionally repaired.",
        neutral_context="A licensed electrician is farther away; an untrained coworker is nearby.",
        salient_cue=(
            "The coworker could look at it immediately and calling the electrician causes a delay."
        ),
        precondition="The repair requires a licensed electrician and is not a visual inspection.",
        counterfactual_goal="We only want someone to note which indicator light is blinking.",
        correct_action="wait for the licensed electrician",
        lure_action="have the nearby coworker handle it now",
        rationale="Availability is not a substitute for the capability required by the repair.",
    ),
    Scenario(
        scenario_id="prerequisite_state_preheat_oven",
        family="prerequisite_state",
        goal="I want to bake the cake according to a recipe that assumes a preheated oven.",
        neutral_context="The filled cake pan is ready and the oven is currently cold.",
        salient_cue=(
            "Putting the pan in immediately starts sooner and avoids waiting for preheating."
        ),
        precondition=(
            "The recipe's baking time and result require the oven to reach temperature first."
        ),
        counterfactual_goal=(
            "For an experiment, I want to observe what happens when this cake starts "
            "in a cold oven."
        ),
        correct_action="wait for the oven to preheat before putting the cake in",
        lure_action="put the cake in now",
        rationale="The required initial state must be reached before the main action begins.",
    ),
    Scenario(
        scenario_id="prerequisite_state_firmware_battery",
        family="prerequisite_state",
        goal="I want the device firmware update to complete without losing power.",
        neutral_context="The device battery is at 5 percent and the charger is available.",
        salient_cue=(
            "Starting the update now is faster than waiting for the battery to charge."
        ),
        precondition=(
            "The device cannot be charged during this update and power loss can corrupt it."
        ),
        counterfactual_goal=(
            "I only want to verify that the update screen opens; I will cancel before installation."
        ),
        correct_action="charge the device before starting the update",
        lure_action="start the update now",
        rationale="Sufficient battery state is a prerequisite for completing the update safely.",
    ),
    Scenario(
        scenario_id="means_end_conflict_exercise_stairs",
        family="means_end_conflict",
        goal="I want this trip upstairs to provide some physical exercise.",
        neutral_context="Both the stairs and the elevator reach the same floor.",
        salient_cue="The elevator is faster and requires less effort than climbing the stairs.",
        precondition="The chosen trip itself must involve physical exertion to satisfy the goal.",
        counterfactual_goal="I want to reach the upstairs floor as quickly and easily as possible.",
        correct_action="use the stairs",
        lure_action="use the elevator",
        rationale="The locally efficient means conflicts with the stated exercise objective.",
    ),
    Scenario(
        scenario_id="means_end_conflict_refill_bottle",
        family="means_end_conflict",
        goal="I want to avoid using a new single-use plastic bottle for my water.",
        neutral_context=(
            "A refill station is farther down the hall; bottled water is sold beside me."
        ),
        salient_cue=(
            "Buying the nearby bottled water is quicker than walking to the refill station."
        ),
        precondition="Buying that water would require taking a new single-use plastic bottle.",
        counterfactual_goal="I want to get drinking water with the least possible walking.",
        correct_action="walk to the refill station",
        lure_action="buy the nearby bottled water",
        rationale="The shortest action directly violates the stated waste-reduction objective.",
    ),
)


def _question(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if part.strip()) + " What should I do?"


def expand_scenarios(
    scenarios: Iterable[Scenario],
    *,
    revision: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        common = {
            "pair_id": scenario.scenario_id,
            "template_id": f"{scenario.family}_{revision}",
            "family": scenario.family,
            "rationale": scenario.rationale,
            "revision": revision,
        }
        rows.extend(
            [
                {
                    **common,
                    "case_id": f"{scenario.scenario_id}_hostile",
                    "condition": "hostile",
                    "question": _question(
                        scenario.goal,
                        scenario.neutral_context,
                        scenario.salient_cue,
                    ),
                    "correct_answer": scenario.correct_action,
                    "lure_answer": scenario.lure_action,
                    "note": "hidden_precondition; intended_lure=local_efficiency",
                },
                {
                    **common,
                    "case_id": f"{scenario.scenario_id}_explicit",
                    "condition": "explicit",
                    "question": _question(
                        scenario.goal,
                        scenario.neutral_context,
                        scenario.salient_cue,
                        f"One requirement matters here: {scenario.precondition}",
                    ),
                    "correct_answer": scenario.correct_action,
                    "lure_answer": scenario.lure_action,
                    "note": "precondition_explicit",
                },
                {
                    **common,
                    "case_id": f"{scenario.scenario_id}_neutral",
                    "condition": "neutral",
                    "question": _question(
                        scenario.goal,
                        scenario.neutral_context,
                    ),
                    "correct_answer": scenario.correct_action,
                    "lure_answer": scenario.lure_action,
                    "note": "efficiency_cue_removed",
                },
                {
                    **common,
                    "case_id": f"{scenario.scenario_id}_counterfactual",
                    "condition": "counterfactual",
                    "question": _question(
                        f"My only goal right now is this: {scenario.counterfactual_goal}",
                        (
                            "I do not need to complete any other task suggested by "
                            "the situation."
                        ),
                        scenario.neutral_context,
                        scenario.salient_cue,
                    ),
                    "correct_answer": scenario.lure_action,
                    "lure_answer": scenario.correct_action,
                    "note": (
                        "goal_changed_and_exclusive; hostile_lure_becomes_correct"
                    ),
                },
            ]
        )
    validate_condition_rows(rows)
    return rows


def validate_condition_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Goal-Affordance rows cannot be empty")
    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Goal-Affordance case IDs must be unique")
    questions = [row["question"] for row in rows]
    if len(questions) != len(set(questions)):
        raise ValueError("Goal-Affordance questions must be unique")

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)
        if row["correct_answer"].casefold() == row["lure_answer"].casefold():
            raise ValueError(f"{row['case_id']}: correct and lure are identical")
        if not row["question"].endswith("What should I do?"):
            raise ValueError(f"{row['case_id']}: missing common decision question")

    expected = {"hostile", "explicit", "neutral", "counterfactual"}
    for pair_id, group in by_pair.items():
        by_condition = {row["condition"]: row for row in group}
        if set(by_condition) != expected:
            raise ValueError(f"{pair_id}: expected four conditions, got {sorted(by_condition)}")
        hostile = by_condition["hostile"]
        for condition in ("explicit", "neutral"):
            row = by_condition[condition]
            if (
                row["correct_answer"] != hostile["correct_answer"]
                or row["lure_answer"] != hostile["lure_answer"]
            ):
                raise ValueError(f"{pair_id}/{condition}: answer mapping changed")
        counterfactual = by_condition["counterfactual"]
        if (
            counterfactual["correct_answer"] != hostile["lure_answer"]
            or counterfactual["lure_answer"] != hostile["correct_answer"]
        ):
            raise ValueError(f"{pair_id}: counterfactual does not swap correct/lure")

    family_counts = Counter(row["family"] for row in rows if row["condition"] == "hostile")
    if len(set(family_counts.values())) != 1:
        raise ValueError(f"Goal-Affordance family imbalance: {dict(family_counts)}")


def development_payload(
    scenarios: Iterable[Scenario],
    *,
    dataset_id: str,
    revision: str,
) -> dict[str, Any]:
    scenario_list = list(scenarios)
    rows = expand_scenarios(scenario_list, revision=revision)
    return {
        "dataset_id": dataset_id,
        "schema_version": 3,
        "title": "Goal-Affordance Traps development seed",
        "description": (
            "Development-only paired scenarios for calibrating salient goal/precondition "
            "traps before goal_affordance_traps_v1 is frozen."
        ),
        "task_kind": "goal_affordance",
        "scoring": "binary_choice",
        "source": {
            "authors": "MindScopeX project",
            "year": 2026,
            "title": "Goal-Affordance Traps",
            "license": "Apache-2.0 (repository-generated content)",
        },
        "generated_by": "scripts/goal_affordance_data.py",
        "revision": revision,
        "n_base_scenarios": len(scenario_list),
        "n_cases": len(rows),
        "family_counts": dict(
            sorted(Counter(scenario.family for scenario in scenario_list).items())
        ),
        "condition_counts": dict(sorted(Counter(row["condition"] for row in rows).items())),
        "cases": rows,
        "scenario_source": [asdict(scenario) for scenario in scenario_list],
    }


def write_development_seed(path: Path, *, revision: str = "seed_v1") -> Path:
    payload = development_payload(
        SEED_SCENARIOS,
        dataset_id=f"goal_affordance_{revision}",
        revision=revision,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    destination = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "goal_affordance_development"
        / "seed_v1.json"
    )
    write_development_seed(destination)
    print(destination)
