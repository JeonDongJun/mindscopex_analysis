"""Generate high-load Korean Goal-Affordance v2 candidates through OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluate_goal_affordance import API_URL, MODELS_URL, ROOT
from openrouter_common import extract_content, http_json, load_dotenv

DEFAULT_MODELS = (
    "openai/gpt-5.6-sol",
    "anthropic/claude-opus-5",
    "google/gemini-3-flash-preview",
)
FAMILIES = (
    "target_identity_binding",
    "resource_compatibility",
    "agent_authorization",
    "prerequisite_sequence",
)


def response_schema() -> dict[str, Any]:
    scenario = {
        "type": "object",
        "properties": {
            "scenario_id": {"type": "string"},
            "family": {"type": "string", "enum": list(FAMILIES)},
            "hostile_question": {"type": "string"},
            "explicit_question": {"type": "string"},
            "neutral_question": {"type": "string"},
            "counterfactual_question": {"type": "string"},
            "correct_action": {"type": "string"},
            "lure_action": {"type": "string"},
            "critical_fact": {"type": "string"},
            "heuristic": {"type": "string"},
            "rationale": {"type": "string"},
            "ambiguity_check": {"type": "string"},
        },
        "required": [
            "scenario_id",
            "family",
            "hostile_question",
            "explicit_question",
            "neutral_question",
            "counterfactual_question",
            "correct_action",
            "lure_action",
            "critical_fact",
            "heuristic",
            "rationale",
            "ambiguity_check",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "goal_affordance_v2_candidates",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "scenarios": {"type": "array", "items": scenario}
                },
                "required": ["scenarios"],
                "additionalProperties": False,
            },
        },
    }


def generation_prompt(count: int) -> str:
    return f"""한국어 Goal-Affordance reasoning challenge 후보를 정확히 {count}개 작성하라.

목표 현상:
- 직관적 즉답은 가까움, 즉시 사용 가능함, 기본 행동에 끌려 lure를 고른다.
- 문장을 다시 읽고 목표에 묶인 대상·도구·사람·상태를 추적하면 correct가 유일하다.
- 최신 frontier model도 direct 조건에서 실수할 만큼 주의 부하가 있어야 한다.

family:
- target_identity_binding: 비슷한 대상이 여럿이고 목표는 그중 하나에만 적용된다.
- resource_compatibility: 가까운 자원은 호환되지 않고 목표에는 다른 자원이 필요하다.
- agent_authorization: 가까운 사람은 편리하지만 목표 수행 자격은 다른 사람에게 있다.
- prerequisite_sequence: 즉시 시작하는 행동이 명시된 선행상태 때문에 목표를 실패시킨다.

절대 조건:
1. hostile_question은 자연스러운 2~4문장 한국어다.
2. 정답을 결정하는 모든 사실을 hostile에 명시한다. correct를 고르기 위해 explicit에서
   처음 공개되는 사실이나 상식 밖 가정을 사용하지 않는다.
3. 결정적 사실은 앞쪽에, 거리·속도·가용성 lure는 마지막 문장에 둔다.
4. 최소 두 대상/자원/사람을 추적해야 한다. 단순히 물건 하나를 잊는 문제는 피한다.
5. correct_action과 lure_action은 길이와 문법이 비슷한 즉시 행동이다.
6. explicit_question은 같은 사실관계를 유지하면서 목표-필수조건 연결만 한 번 더 직접 말한다.
7. neutral_question은 lure 단서만 제거하며 correct/lure 정답 관계를 유지한다.
8. counterfactual_question은 목표만 바꿔 원래 lure_action이 유일한 정답이 되게 한다.
9. 안전·법률·의료 조언, 브랜드 정책, 지역 규정, 도덕 취향에 의존하지 않는다.
10. attentive adult가 hostile의 정답에 동의해야 한다. '담당자가 도와줄 수도 있다' 같은
    대안 행동이 남지 않도록 ambiguity_check에서 반례를 검사한다.
11. 세차장·주유·타이어 공기·차량 검사·기록보관실 출입카드 예시는 사용하지 않는다.
12. scenario_id는 family로 시작하는 고유 snake_case다.

각 family를 같은 수로 구성하고 요구된 JSON만 반환하라."""


def model_metadata(model: str) -> dict[str, Any]:
    rows = http_json(MODELS_URL).get("data", [])
    matching = [row for row in rows if row["id"] == model]
    if not matching:
        raise ValueError(f"Model not found: {model}")
    row = matching[0]
    if "high" not in (row.get("reasoning") or {}).get("supported_efforts", []):
        raise ValueError(f"{model} does not support high reasoning")
    return row


def generate(model: str, count: int, max_tokens: int, timeout: float) -> dict[str, Any]:
    if count % len(FAMILIES):
        raise ValueError(f"count must be divisible by {len(FAMILIES)}")
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    metadata = model_metadata(model)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "당신은 인지과학 benchmark 설계자다. 높은 오류율보다 모호성 없는 "
                    "직관-심사숙고 대비 타당성을 우선한다."
                ),
            },
            {"role": "user", "content": generation_prompt(count)},
        ],
        "reasoning": {"effort": "high", "exclude": True},
        "response_format": response_schema(),
        "max_tokens": max_tokens,
    }
    try:
        response = http_json(
            API_URL,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/",
                "X-Title": "MindScopeX Goal-Affordance v2 generation",
            },
            payload=payload,
            timeout=timeout,
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:2000]}") from exc
    if response.get("error"):
        raise RuntimeError(json.dumps(response["error"], ensure_ascii=False))
    parsed = json.loads(extract_content(response["choices"][0]["message"]))
    scenarios = parsed["scenarios"]
    if len(scenarios) != count:
        raise ValueError(f"Expected {count} scenarios, got {len(scenarios)}")
    ids = [row["scenario_id"] for row in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate generated scenario IDs")
    counts = {family: 0 for family in FAMILIES}
    for row in scenarios:
        family = row["family"]
        counts[family] += 1
        if not row["scenario_id"].startswith(f"{family}_"):
            raise ValueError(f"Bad scenario ID: {row['scenario_id']}")
    expected = count // len(FAMILIES)
    if any(value != expected for value in counts.values()):
        raise ValueError(f"Unbalanced generated families: {counts}")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "generator_model": model,
        "generator_canonical_slug": metadata.get("canonical_slug"),
        "reasoning_effort": "high",
        "family_counts": counts,
        "usage": response.get("usage") or {},
        "scenarios": scenarios,
    }


def safe_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=DEFAULT_MODELS, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = (
            ROOT
            / "results"
            / "goal_affordance_v2_development"
            / f"generated_high_load_{safe_name(args.model)}.json"
        )
    return args


if __name__ == "__main__":
    cli = parse_args()
    result = generate(cli.model, cli.count, cli.max_tokens, cli.timeout)
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{cli.output} | scenarios={len(result['scenarios'])} | "
        f"cost=${float(result['usage'].get('cost') or 0):.4f}"
    )
