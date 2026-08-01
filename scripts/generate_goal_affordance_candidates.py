"""Generate diverse Goal-Affordance candidate scenarios through OpenRouter.

Generated proposals are development artifacts, not automatically trusted data.
They must pass local invariants, control-condition checks, and frontier solver
evaluation before curation into the committed dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluate_goal_affordance import (
    API_URL,
    MODELS_URL,
    ROOT,
    extract_content,
    http_json,
    load_dotenv,
)

FAMILIES = {
    "target_transport": (
        "The goal applies to an object or being that must reach a destination, but "
        "the lure moves only the decision-maker."
    ),
    "tool_transport": (
        "A tool needed at the destination is elsewhere; the lure goes directly to "
        "the destination without first taking the tool."
    ),
    "required_resource": (
        "A credential, ticket, document, component, or consumable is required; the "
        "lure optimizes arrival while omitting the resource."
    ),
    "agent_capability": (
        "Only a particular authorized or capable agent can achieve the goal; the "
        "lure chooses a nearer but ineligible agent."
    ),
    "prerequisite_state": (
        "A safe or functional prerequisite state must be established first; the "
        "lure starts the focal action immediately."
    ),
    "means_end_conflict": (
        "The tempting means is locally faster or easier but directly defeats a "
        "stated non-speed objective."
    ),
}
DEFAULT_MODELS = (
    "openai/gpt-5.6-sol",
    "anthropic/claude-opus-5",
    "google/gemini-3-flash-preview",
)


def candidate_schema(per_family: int) -> dict[str, Any]:
    scenario = {
        "type": "object",
        "properties": {
            "scenario_id": {"type": "string"},
            "family": {"type": "string", "enum": list(FAMILIES)},
            "goal": {"type": "string"},
            "neutral_context": {"type": "string"},
            "salient_cue": {"type": "string"},
            "precondition": {"type": "string"},
            "counterfactual_goal": {"type": "string"},
            "correct_action": {"type": "string"},
            "lure_action": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": [
            "scenario_id",
            "family",
            "goal",
            "neutral_context",
            "salient_cue",
            "precondition",
            "counterfactual_goal",
            "correct_action",
            "lure_action",
            "rationale",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "goal_affordance_candidates",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "scenarios": {
                        "type": "array",
                        "items": scenario,
                    }
                },
                "required": ["scenarios"],
                "additionalProperties": False,
            },
        },
    }


def prompt(per_family: int) -> str:
    family_text = "\n".join(f"- {name}: {definition}" for name, definition in FAMILIES.items())
    return f"""Design hard but unambiguous benchmark items for goal-affordance reasoning.

Create exactly {per_family} scenarios for each of these six families:
{family_text}

Each scenario will later become four binary-choice conditions. Follow every constraint:
1. The hostile condition combines goal + neutral_context + salient_cue. The cue makes the
   lure feel like the locally fastest, easiest, nearest, or default action, but the lure
   cannot actually accomplish the stated goal because of exactly one implicit precondition.
2. The precondition sentence makes that missing fact explicit without changing anything else.
3. Removing salient_cue leaves a neutral, uniquely answerable item.
4. Changing only to counterfactual_goal makes the original lure_action uniquely correct and
   the original correct_action inferior. This must be a real goal change, not permission to fail.
5. An attentive adult should agree on the answer. Do not depend on jurisdiction, brand-specific
   policy, hidden medical facts, moral taste, obscure trivia, unsafe conduct, or assumptions
   not stated in neutral_context/precondition.
6. Keep correct_action and lure_action short, parallel, and plausible as immediate actions.
   Do not say "without the required X" in the lure; that gives away the trap.
7. Use natural everyday language. Do not label the cue as a trick or mention this benchmark.
8. Seek genuinely subtle goal/means confusions, not trivial forgotten-object checklists.
9. Avoid these already used examples: car wash, parcel return, projector adapter, bicycle pump,
   pharmacy photo ID, paper train ticket, bank account owner, electrician, oven preheating,
   firmware battery, exercise stairs, and refillable water bottle.
10. scenario_id must be unique snake_case and start with its family name.

Return only the required JSON."""


def model_row(model_id: str) -> dict[str, Any]:
    rows = http_json(MODELS_URL).get("data", [])
    matching = [row for row in rows if row["id"] == model_id]
    if not matching:
        raise ValueError(f"Model not found: {model_id}")
    row = matching[0]
    efforts = (row.get("reasoning") or {}).get("supported_efforts") or []
    if "high" not in efforts:
        raise ValueError(f"{model_id} does not support high reasoning")
    if "response_format" not in (row.get("supported_parameters") or []):
        raise ValueError(f"{model_id} does not support structured output")
    return row


def generate(model: str, per_family: int, max_tokens: int, timeout: float) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    metadata = model_row(model)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a psychometrics researcher designing controlled reasoning "
                    "stimuli. Prioritize validity, subtlety, and diverse mechanisms."
                ),
            },
            {"role": "user", "content": prompt(per_family)},
        ],
        "reasoning": {"effort": "high", "exclude": True},
        "response_format": candidate_schema(per_family),
        "max_tokens": max_tokens,
    }
    try:
        response = http_json(
            API_URL,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/",
                "X-Title": "MindScopeX goal-affordance candidate generation",
            },
            payload=payload,
            timeout=timeout,
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:2000]}") from exc
    if response.get("error"):
        raise RuntimeError(json.dumps(response["error"], ensure_ascii=False))
    content = extract_content(response["choices"][0]["message"])
    parsed = json.loads(content)
    scenarios = parsed["scenarios"]
    expected = per_family * len(FAMILIES)
    if len(scenarios) != expected:
        raise ValueError(f"Expected {expected} scenarios, got {len(scenarios)}")
    counts = {family: 0 for family in FAMILIES}
    ids = set()
    for scenario in scenarios:
        family = scenario["family"]
        counts[family] += 1
        scenario_id = scenario["scenario_id"]
        if scenario_id in ids:
            raise ValueError(f"Duplicate scenario ID: {scenario_id}")
        ids.add(scenario_id)
        if not scenario_id.startswith(f"{family}_"):
            raise ValueError(f"Scenario ID does not start with family: {scenario_id}")
    if any(count != per_family for count in counts.values()):
        raise ValueError(f"Unbalanced family counts: {counts}")
    usage = response.get("usage") or {}
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "generator_model": model,
        "generator_canonical_slug": metadata.get("canonical_slug"),
        "reasoning_effort": "high",
        "per_family": per_family,
        "family_counts": counts,
        "usage": usage,
        "scenarios": scenarios,
    }


def safe_model_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=DEFAULT_MODELS, required=True)
    parser.add_argument("--per-family", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.per_family < 1:
        parser.error("--per-family must be positive")
    if args.output is None:
        args.output = (
            ROOT
            / "results"
            / "goal_affordance_development"
            / f"candidates_{safe_model_name(args.model)}.json"
        )
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    result = generate(
        cli_args.model,
        cli_args.per_family,
        cli_args.max_tokens,
        cli_args.timeout,
    )
    cli_args.output.parent.mkdir(parents=True, exist_ok=True)
    cli_args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{cli_args.output} | scenarios={len(result['scenarios'])} | "
        f"cost=${float(result['usage'].get('cost') or 0):.4f}"
    )
