"""Publish the final OpenRouter reasoning evaluation to the 7/26 Notion page."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
OUTPUT_DIR = ROOT / "results" / "openrouter_deliberation_redesign_20260726"
PAGE_ID = "3a90df78-43fe-8112-aed8-f494256ad91a"
NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2026-03-11"


def env_value(name: str) -> str:
    for raw in ENV_PATH.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip('"').strip("'")
    raise RuntimeError(f"{name} is missing from .env")


TOKEN = env_value("NOTION_TOKEN")
BASE_HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Notion-Version": NOTION_VERSION,
}


def request(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
    raw: bytes | None = None,
    retries: int = 6,
) -> dict[str, Any]:
    request_headers = dict(BASE_HEADERS)
    request_headers.update(headers or {})
    data = raw
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers=request_headers,
                method=method,
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                body = response.read()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 or 500 <= exc.code < 600:
                delay = float(
                    exc.headers.get("Retry-After") or min(2**attempt, 10)
                )
                time.sleep(delay)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {detail[:1200]}") from exc
    raise RuntimeError(f"Notion API retries exhausted: {method} {url}")


def rich_text(
    text: str,
    *,
    bold: bool = False,
    code: bool = False,
    color: str = "default",
) -> dict[str, Any]:
    return {
        "type": "text",
        "text": {"content": str(text)},
        "annotations": {
            "bold": bold,
            "italic": False,
            "strikethrough": False,
            "underline": False,
            "code": code,
            "color": color,
        },
    }


def text_block(
    block_type: str,
    text: str,
    *,
    color: str = "default",
    bold: bool = False,
) -> dict[str, Any]:
    return {
        "object": "block",
        "type": block_type,
        block_type: {
            "rich_text": [rich_text(text, bold=bold)],
            "color": color,
        },
    }


def bullets(items: list[str]) -> list[dict[str, Any]]:
    return [text_block("bulleted_list_item", item) for item in items]


def table(
    headers: list[str],
    rows: list[list[str]],
) -> dict[str, Any]:
    def table_row(cells: list[str], *, header: bool = False) -> dict[str, Any]:
        return {
            "object": "block",
            "type": "table_row",
            "table_row": {
                "cells": [
                    [rich_text(cell, bold=header)]
                    for cell in cells
                ]
            },
        }

    return {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": len(headers),
            "has_column_header": True,
            "has_row_header": False,
            "children": [
                table_row(headers, header=True),
                *(table_row(row) for row in rows),
            ],
        },
    }


def upload_file(path: Path) -> str:
    created = request(
        "POST",
        f"{NOTION_API}/file_uploads",
        {
            "mode": "single_part",
            "filename": path.name,
            "content_type": "image/png",
        },
    )
    upload_id = created["id"]
    upload_url = (
        created.get("upload_url")
        or f"{NOTION_API}/file_uploads/{upload_id}/send"
    )
    boundary = "----CodexBoundary" + uuid.uuid4().hex
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode()
    body = prefix + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    uploaded = request(
        "POST",
        upload_url,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        raw=body,
    )
    if uploaded.get("status") != "uploaded":
        raise RuntimeError(f"File upload failed: {uploaded.get('status')}")
    return upload_id


def image_block(upload_id: str, caption: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "image",
        "image": {
            "type": "file_upload",
            "file_upload": {"id": upload_id},
            "caption": [rich_text(caption)],
        },
    }


def build_blocks(
    accuracy_upload: str,
    lure_upload: str,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [
        text_block(
            "paragraph",
            "재설계 검증 결과: reasoning을 충분히 주면 세 모델 모두 함정 회피 "
            "정답률이 상승했다. 다만 GPT와 Gemini의 남은 의미 함정 오답은 "
            "대부분 사실을 알고도 최종 답에서 거짓 전제를 명시적으로 고치지 "
            "않은 경우였다.",
            color="blue_background",
            bold=True,
        ),
        text_block(
            "paragraph",
            "최종 업데이트: 2026-07-26 · OpenRouter API · temperature 0 · "
            "문항·조건별 3회 반복",
        ),
        text_block("heading_2", "한눈에 보는 결론"),
    ]
    blocks += bullets(
        [
            "전체 함정 회피율은 GPT 94.3%→96.7%(+2.3%p), Claude "
            "98.0%→100.0%(+2.0%p), Gemini 75.7%→86.7%(+11.0%p)로 "
            "모두 상승했다.",
            "거짓 전제 최종 교정률은 GPT +4.7%p, Claude +4.0%p, "
            "Gemini +22.0%p 상승했다.",
            "CRT 50문항은 모든 모델·조건에서 100%였다. 관측된 차이는 전부 "
            "semantic illusion에서 발생했다.",
            "paired McNemar 검정에서 Claude(p=0.031)와 Gemini(p<0.001)는 "
            "유의했고, GPT(p=0.092)는 개선 방향이지만 5% 유의수준에는 "
            "도달하지 않았다.",
        ]
    )
    blocks += [
        text_block("heading_2", "Reasoning에 따른 정답률 변화"),
        image_block(
            accuracy_upload,
            "전체 함정 회피율과 거짓 전제 최종 교정률: "
            "reasoning none vs 충분한 reasoning",
        ),
        table(
            [
                "모델",
                "전체 none",
                "전체 reasoning",
                "Δ",
                "의미 함정 none",
                "의미 함정 reasoning",
                "Δ",
            ],
            [
                [
                    "GPT-5.6 Sol",
                    "94.3%",
                    "96.7%",
                    "+2.3%p",
                    "88.7%",
                    "93.3%",
                    "+4.7%p",
                ],
                [
                    "Claude Opus 5",
                    "98.0%",
                    "100.0%",
                    "+2.0%p",
                    "96.0%",
                    "100.0%",
                    "+4.0%p",
                ],
                [
                    "Gemini 3 Flash",
                    "75.7%",
                    "86.7%",
                    "+11.0%p",
                    "51.3%",
                    "73.3%",
                    "+22.0%p",
                ],
            ],
        ),
        text_block("heading_2", "Lure 오답은 얼마나 남았나"),
        image_block(
            lure_upload,
            "지식 통제로 확인된 lure 비율과 의미 함정 오답 중 lure 비중",
        ),
        table(
            [
                "모델",
                "알고도 lure none",
                "reasoning",
                "의미 함정 오답 none",
                "reasoning",
                "오답 중 lure none→reasoning",
            ],
            [
                [
                    "GPT-5.6 Sol",
                    "11.3%",
                    "6.7%",
                    "17건",
                    "10건",
                    "100.0% → 100.0%",
                ],
                [
                    "Claude Opus 5",
                    "4.1%",
                    "0.0%",
                    "6건",
                    "0건",
                    "100.0% → 해당 없음",
                ],
                [
                    "Gemini 3 Flash",
                    "46.5%",
                    "25.0%",
                    "73건",
                    "40건",
                    "91.8% → 90.0%",
                ],
            ],
        ),
        text_block(
            "paragraph",
            "“알고도 lure”는 같은 사실을 묻는 중립적 knowledge-control 질문을 "
            "맞혔지만, 함정 질문의 최종 답에서는 거짓 전제를 명시적으로 "
            "고치지 못한 비율이다.",
        ),
        text_block(
            "paragraph",
            "오답 중 lure 비중이 높게 유지된 것은 reasoning이 효과가 없다는 "
            "뜻이 아니다. reasoning은 오답의 절대 개수를 줄였지만, 끝까지 "
            "남은 실패가 주로 전제 교정 누락 유형이었다는 뜻이다.",
        ),
        text_block("heading_2", "데이터셋 구성"),
        table(
            ["구분", "문항 수", "문항 형태", "채점"],
            [
                [
                    "CRT",
                    "50",
                    "직관적인 lure 답을 유도하는 계산·언어 문제",
                    "최종 수치/정답 일치",
                ],
                [
                    "Semantic illusion",
                    "50",
                    "질문 안에 잘못된 사실적 전제가 포함된 문제",
                    "최종 답의 명시적 전제 교정",
                ],
                [
                    "Knowledge control",
                    "모델당 50",
                    "동일 사실을 함정 없이 중립적으로 질문",
                    "지식 부족과 lure 취약성 분리",
                ],
            ],
        ),
    ]
    blocks += bullets(
        [
            "대상 응답: 100문항 × 3회 × 2조건 × 3모델 = 1,800건",
            "Knowledge-control 응답: 50문항 × 3모델 = 150건",
            "총 생성 1,950건, structured judge 판정 1,050건",
            "중복 CRT를 제거하고 애매한 verbal_crt_010 문항은 제외했다.",
        ]
    )
    blocks += [
        text_block("heading_2", "실험 조건"),
        table(
            ["항목", "reasoning none", "충분한 reasoning"],
            [
                [
                    "API 설정",
                    "reasoning.effort=none",
                    "reasoning.effort=high",
                ],
                [
                    "응답 형식",
                    "최종 답변만 요청",
                    "<verification> 검증 후 self-contained <final>",
                ],
                [
                    "전제 오류 지시",
                    "별도 강조 없음",
                    "검증과 최종 답 모두에서 명시적으로 교정",
                ],
                [
                    "공통 설정",
                    "temperature 0, max_tokens 8,192",
                    "temperature 0, max_tokens 8,192",
                ],
            ],
        ),
    ]
    blocks += bullets(
        [
            "평가 모델: OpenAI GPT-5.6 Sol, Anthropic Claude Opus 5, "
            "Google Gemini 3 Flash Preview",
            "의미 함정과 knowledge control은 GPT-5.4 Mini high-reasoning "
            "structured judge로 판정했다.",
            "반복 응답은 temperature 0에서도 provider-side 비결정성이 남을 수 "
            "있어 각각 독립 호출했다.",
            "총 API 비용: $9.36",
        ]
    )
    blocks += [text_block("heading_2", "해석")]
    blocks += bullets(
        [
            "reasoning은 최종 정답률을 올렸다. 특히 Gemini의 의미 함정 교정률 "
            "개선이 가장 컸다.",
            "GPT는 검증 과정에서 전제를 감지한 비율이 98.0%였지만 최종 교정률은 "
            "93.3%였다. Gemini도 98.7% 대 73.3%로 차이가 컸다. 즉 "
            "“알아챔”과 “최종 답에 명시”는 별도 실패 지점이다.",
            "이번 비교는 high reasoning과 명시적인 검증·전제 교정 프롬프트를 "
            "함께 적용한 실용적 intervention이다. hidden reasoning effort "
            "하나만의 순수 인과효과로 해석할 수는 없다.",
            "CRT가 전 조건 100%였으므로 frontier 모델 비교에는 현재 CRT보다 "
            "새로운 비노출 문항이나 더 어려운 함정 세트가 필요하다.",
        ]
    )
    blocks += [
        text_block("heading_2", "재현 정보"),
        text_block(
            "paragraph",
            "로컬 산출물: "
            "results/openrouter_deliberation_redesign_20260726/report.md · "
            "summary.csv · paired.csv",
        ),
        text_block(
            "paragraph",
            "실행 스크립트: scripts/evaluate_openrouter_deliberation.py",
        ),
    ]
    return blocks


def main() -> None:
    old_page = request(
        "GET",
        f"{NOTION_API}/blocks/{PAGE_ID}/children?page_size=100",
    )
    if old_page.get("has_more"):
        raise RuntimeError("Unexpected pagination; aborting safely")
    old_ids = [block["id"] for block in old_page.get("results", [])]
    print(f"Old blocks snapshotted: {len(old_ids)}", flush=True)

    accuracy_upload = upload_file(OUTPUT_DIR / "accuracy_by_reasoning.png")
    lure_upload = upload_file(OUTPUT_DIR / "lure_rates_by_reasoning.png")
    print("Charts uploaded", flush=True)

    blocks = build_blocks(accuracy_upload, lure_upload)
    for index in range(0, len(blocks), 50):
        request(
            "PATCH",
            f"{NOTION_API}/blocks/{PAGE_ID}/children",
            {"children": blocks[index : index + 50]},
        )
        time.sleep(0.4)
    print(f"New blocks appended: {len(blocks)}", flush=True)

    verification = request(
        "GET",
        f"{NOTION_API}/blocks/{PAGE_ID}/children?page_size=100",
    )
    new_ids = {
        block["id"] for block in verification.get("results", [])
    } - set(old_ids)
    if len(new_ids) < len(blocks):
        raise RuntimeError(
            f"Verification failed: {len(new_ids)}/{len(blocks)}"
        )
    print("New content verified", flush=True)

    for index, block_id in enumerate(old_ids, 1):
        request("DELETE", f"{NOTION_API}/blocks/{block_id}")
        time.sleep(0.36)
        if index % 10 == 0:
            print(
                f"Old blocks removed: {index}/{len(old_ids)}",
                flush=True,
            )

    request(
        "PATCH",
        f"{NOTION_API}/pages/{PAGE_ID}",
        {
            "properties": {
                "title": {
                    "title": [
                        rich_text(
                            "Frontier 모델 함정 문제 재검증 — "
                            "reasoning none vs sufficient"
                        )
                    ]
                }
            }
        },
    )
    final_page = request(
        "GET",
        f"{NOTION_API}/blocks/{PAGE_ID}/children?page_size=100",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "top_level_blocks": len(final_page.get("results", [])),
                "page_url": (
                    "https://www.notion.so/" + PAGE_ID.replace("-", "")
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
