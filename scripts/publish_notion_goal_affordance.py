"""Append the Goal-Affordance v1.1 result summary to the 7/30 Notion page."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAGE_ID = "3ad0df78-43fe-8154-9a5a-c6ea23c761c8"
NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2026-03-11"
MARKER = "[GA-V1.1-2026-07-31]"
V2_MARKER = "[GA-V2.0-2026-08-02]"


def env_value(name: str) -> str:
    for raw in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip("'\"")
    raise RuntimeError(f"{name} is missing from .env")


HEADERS = {
    "Authorization": f"Bearer {env_value('NOTION_TOKEN')}",
    "Notion-Version": NOTION_VERSION,
    "Content-Type": "application/json",
}


def request(
    method: str, url: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, data=data, headers=HEADERS, method=method)
            with urllib.request.urlopen(req, timeout=60) as response:
                body = response.read()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 or 500 <= exc.code < 600:
                time.sleep(min(2**attempt, 10))
                continue
            raise RuntimeError(f"HTTP {exc.code}: {detail[:1200]}") from exc
    raise RuntimeError("Notion API retries exhausted")


def rich_text(text: str, *, bold: bool = False) -> list[dict[str, Any]]:
    return [
        {
            "type": "text",
            "text": {"content": text},
            "annotations": {"bold": bold},
        }
    ]


def block(kind: str, text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": kind,
        kind: {"rich_text": rich_text(text)},
    }


def page_contains_marker(marker: str = MARKER) -> bool:
    cursor = None
    while True:
        url = f"{NOTION_API}/blocks/{PAGE_ID}/children?page_size=100"
        if cursor:
            url += f"&start_cursor={cursor}"
        payload = request("GET", url)
        for item in payload.get("results", []):
            value = item.get(item.get("type", ""), {})
            text = "".join(part.get("plain_text", "") for part in value.get("rich_text", []))
            if marker in text:
                return True
        if not payload.get("has_more"):
            return False
        cursor = payload.get("next_cursor")


def publish() -> bool:
    if page_contains_marker():
        return False
    children = [
        block("heading_2", "Goal-Affordance Traps v1.1 — 2026-07-31"),
        block(
            "paragraph",
            (
                f"{MARKER} 목표의 필수조건보다 가까움·편의 단서가 먼저 선택되는지를 "
                "검사하는 paired binary-choice 데이터셋을 확정했다."
            ),
        ),
        block(
            "bulleted_list_item",
            (
                "구성: 60 base scenario × hostile/explicit/neutral/counterfactual "
                "= 240 cases, 6 family 각 10 scenario."
            ),
        ),
        block(
            "bulleted_list_item",
            (
                "Frontier 검증: GPT-5.6-sol, Claude Opus 5, Gemini 3 Flash Preview의 "
                "direct/high 1,440응답, API 오류 0."
            ),
        ),
        block(
            "bulleted_list_item",
            (
                "direct hostile lure: GPT 1/60, Claude 2/60, Gemini 1/60. "
                "high 전체는 3/180으로 direct 4/180보다 1건만 감소."
            ),
        ),
        block(
            "bulleted_list_item",
            (
                "confirmed challenge: required_resource_credential 1쌍. 세 direct "
                "모델 모두 lure, explicit/neutral/counterfactual은 9/9 정답, "
                "A/B 순서 반전에서도 재현."
            ),
        ),
        block(
            "bulleted_list_item",
            (
                "해석: frontier도 함정에 빠질 수 있다는 존재 증거는 있으나 전체 "
                "60쌍이 challenge는 아니다. core와 challenge manifest를 분리한다."
            ),
        ),
        block(
            "paragraph",
            (
                "정본 문서: docs/datasets.md · 데이터: "
                "src/mindscopex_analysis/data/goal_affordance_traps_v1.json · "
                "독립 검토자 2인의 blind 인간 검수는 후속 gate로 남긴다."
            ),
        ),
    ]
    request(
        "PATCH",
        f"{NOTION_API}/blocks/{PAGE_ID}/children",
        {"children": children},
    )
    return True


def publish_v2() -> bool:
    if page_contains_marker(V2_MARKER):
        return False
    children = [
        block("heading_2", "Goal-Affordance Traps v2 micro-challenge — 2026-08-02"),
        block(
            "paragraph",
            (
                f"{V2_MARKER} 이미지형 거리 함정을 최신 frontier에 맞게 다시 "
                "설계하고 반복 검증했다."
            ),
        ),
        block(
            "bulleted_list_item",
            (
                "최종 구성: 한국어 타이어 공기 scenario 1개 × hostile/explicit/"
                "neutral/counterfactual = 4 cases. broad benchmark가 아닌 micro-challenge."
            ),
        ),
        block(
            "bulleted_list_item",
            (
                "5회 반복 hostile: intuitive 8/15 lure(Claude 5/5, Gemini 3/5, "
                "GPT 0/5), reflective 0/15 lure."
            ),
        ),
        block(
            "bulleted_list_item",
            (
                "explicit/neutral/counterfactual과 A/B 순서 반전은 모두 통과. "
                "반복 호출은 독립 문항이 아니라 한 문항의 응답 확률 추정."
            ),
        ),
        block(
            "bulleted_list_item",
            (
                "짧은 50m 원형 8개는 frontier가 72/72 정답. 새 semantic cluster를 "
                "확보하기 전까지 v2로 일반 오류율을 추정하지 않는다."
            ),
        ),
        block(
            "paragraph",
            (
                "정본: docs/datasets.md · 데이터: src/mindscopex_analysis/data/"
                "goal_affordance_traps_v2.json · 보고서: results/"
                "goal_affordance_traps_v2_final_20260802/"
            ),
        ),
    ]
    request(
        "PATCH",
        f"{NOTION_API}/blocks/{PAGE_ID}/children",
        {"children": children},
    )
    return True


if __name__ == "__main__":
    v1_changed = publish()
    v2_changed = publish_v2()
    print(
        f"https://www.notion.so/{PAGE_ID.replace('-', '')} | "
        f"v1={'updated' if v1_changed else 'present'} | "
        f"v2={'updated' if v2_changed else 'present'}"
    )
