"""Compare OpenRouter frontier models with reasoning disabled vs high effort.

The script evaluates every normalized lure dataset committed under
``src/mindscopex_analysis/data``. It uses one independent chat-completion
request per model, reasoning mode, and item; appends successful responses to a
resume-safe JSONL file; and writes paired CSV/Markdown reports.

The API key is read only from ``OPENROUTER_API_KEY``. Do not put it in a config
file or command-line argument.

Examples:
    $env:OPENROUTER_API_KEY = "..."
    uv run python scripts/evaluate_openrouter_reasoning.py

    # Fast validation run, then resume the same directory for a full run.
    uv run python scripts/evaluate_openrouter_reasoning.py --limit-per-dataset 1
    uv run python scripts/evaluate_openrouter_reasoning.py --output-dir results/my_run

The default model set is deliberately restricted to strong GPT, Claude, and
Gemini models whose OpenRouter metadata permits reasoning to be disabled. Some
stronger Gemini Pro endpoints mark reasoning as mandatory and cannot support
this within-model comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
import urllib.error
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from openrouter_common import (
    append_jsonl,
    extract_content,
    http_json,
    read_json,
    utc_now,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "src" / "mindscopex_analysis" / "data"
DEFAULT_MODELS = (
    "openai/gpt-5.6-sol",
    "anthropic/claude-opus-5",
    "google/gemini-3-flash-preview",
)
DEFAULT_SEMANTIC_JUDGE_MODEL = "openai/gpt-5.4-mini"
MODES = ("none", "high")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
SYSTEM_PROMPT = (
    "Answer the user's question accurately. Return only the concise final answer, "
    "with no explanation, analysis, or working."
)
JUDGE_VERSION = 2
JUDGE_SYSTEM_PROMPT = """You grade answers to questions containing a false premise.
Classify what the ANSWER did using exactly one verdict:
- corrected_false_premise: the answer explicitly corrects/denies the false premise, or gives
  an unambiguous negating answer such as zero/none/no such person that necessarily denies it.
- fell_for_false_premise: the answer responds inside the false frame without identifying or
  negating the error. Giving a true detail about a confused entity still belongs here.
- unclear: the answer is empty, irrelevant, or impossible to determine.
Use the supplied authoritative reference only as grading context. Be strict and concise."""
USER_SUFFIX = "\n\nReply with only the concise final answer."
RETRYABLE_HTTP_CODES = {408, 409, 429, 500, 502, 503, 504}
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}
CORRECT_CASE_PATTERNS = {
    "verbal_crt_003": (r"\bsurvivors\b.*\b(?:are|were) not buried\b",),
    "verbal_crt_004": (
        r"\bnone of them\b",
        r"\b(?:do|does) not (?:grow|have) bananas?\b",
        r"\bbananas? do not grow\b",
        r"\bno bananas?\b",
    ),
    "verbal_crt_005": (r"\bno stairs\b", r"\bthere (?:were|are) no stairs\b"),
    "verbal_crt_007": (r"\bno smoke\b",),
    "verbal_crt_009": (r"\bimpossible\b", r"\bcannot marry\b", r"\bhe is dead\b"),
}


@dataclass(frozen=True)
class Case:
    dataset_id: str
    case_id: str
    family: str
    question: str
    scoring: str
    correct_answer: str
    lure_answer: str
    reference_answer: str

    @property
    def key(self) -> str:
        return f"{self.dataset_id}/{self.case_id}"


def load_cases(dataset_ids: list[str] | None, limit_per_dataset: int | None) -> list[Case]:
    paths = sorted(DATA_DIR.glob("*.json"))
    available = {path.stem: path for path in paths}
    wanted = dataset_ids or sorted(available)
    unknown = sorted(set(wanted) - set(available))
    if unknown:
        raise ValueError(f"Unknown datasets {unknown}; available: {sorted(available)}")

    cases: list[Case] = []
    seen: set[str] = set()
    for dataset_id in wanted:
        payload = read_json(available[dataset_id])
        scoring = str(payload.get("scoring", "logprob_margin"))
        rows = payload["cases"]
        if limit_per_dataset is not None:
            rows = rows[:limit_per_dataset]
        for row in rows:
            case = Case(
                dataset_id=dataset_id,
                case_id=str(row["case_id"]),
                family=str(row["family"]),
                question=str(row["question"]).strip(),
                scoring=scoring,
                correct_answer=str(row.get("correct_answer", "")).strip(),
                lure_answer=str(row.get("lure_answer", "")).strip(),
                reference_answer=str(row.get("reference_answer", "")).strip(),
            )
            if case.key in seen:
                raise ValueError(f"Duplicate case key: {case.key}")
            if scoring == "logprob_margin" and (
                not case.correct_answer or not case.lure_answer
            ):
                raise ValueError(f"{case.key} has no correct/lure answers")
            seen.add(case.key)
            cases.append(case)
    return cases


def normalized_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("`", "'")
    contractions = {
        "don't": "do not",
        "doesn't": "does not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "can't": "cannot",
    }
    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    for word, value in NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", value, text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_answer(text: str) -> str:
    """Collapse punctuation-only variants for semantic-judge consistency."""

    return re.sub(r"[^a-z0-9]+", " ", normalized_text(text)).strip()


def answer_patterns(answer: str) -> tuple[str, ...]:
    normalized = normalized_text(answer)
    patterns = [rf"(?<!\w){re.escape(normalized)}(?!\w)"]

    currency = re.fullmatch(r"\$\s*(\d+(?:\.\d+)?)", normalized)
    if currency:
        value = Decimal(currency.group(1))
        plain = format(value.normalize(), "f")
        if "." in plain:
            integer, fraction = plain.split(".", 1)
            number_pattern = rf"{re.escape(integer)}\.{re.escape(fraction)}0*"
        else:
            number_pattern = rf"{re.escape(plain)}(?:\.0+)?"
        patterns.extend(
            (
                rf"(?<!\w)\$\s*{number_pattern}(?![\d.])",
                rf"(?<!\w){number_pattern}\s*(?:dollars?|usd)(?!\w)",
            )
        )

    cents = re.fullmatch(r"(\d+) cents?", normalized)
    if cents:
        value = int(cents.group(1))
        patterns.extend(
            (
                rf"(?<!\w){value}\s*(?:cent|cents|c)(?!\w)",
                rf"(?<!\w){value}\s*¢",
                rf"\$\s*{value / 100:.2f}(?!\d)",
            )
        )

    named_currency = re.fullmatch(r"(\d+(?:\.\d+)?) (dollars?|euros?)", normalized)
    if named_currency:
        raw_value, currency_name = named_currency.groups()
        value = format(Decimal(raw_value).normalize(), "f")
        symbol = "$" if currency_name.startswith("dollar") else "€"
        currency_root = "dollar" if currency_name.startswith("dollar") else "euro"
        patterns.extend(
            (
                rf"(?<!\w){re.escape(symbol)}\s*{re.escape(value)}(?:\.0+)?(?![\d.])",
                rf"(?<!\w){re.escape(value)}(?:\.0+)?\s*{currency_root}s?(?!\w)",
            )
        )

    quantity = re.fullmatch(
        r"(\d+(?:\.\d+)?) (seconds?|minutes?|hours?|days?|weeks?|months?|years?)",
        normalized,
    )
    if quantity:
        value, unit = quantity.groups()
        root = unit.rstrip("s")
        patterns.append(rf"(?<!\w){value}\s*{root}s?(?!\w)")

    ordinal_day = re.fullmatch(r"(\d+)(?:st|nd|rd|th) day", normalized)
    if ordinal_day:
        value = ordinal_day.group(1)
        patterns.extend(
            (
                rf"(?<!\w){value}(?:st|nd|rd|th)?\s+day(?!\w)",
                rf"(?<!\w)day\s+{value}(?!\w)",
            )
        )

    ordinal_words = {"first": "1", "second": "2", "third": "3"}
    if normalized in ordinal_words:
        value = ordinal_words[normalized]
        patterns.append(rf"(?<!\w){value}(?:st|nd|rd|th)(?!\w)")

    leading_number = re.match(r"^[€$]?\s*(\d+(?:\.\d+)?)(?:\s|$)", normalized)
    if leading_number:
        value = leading_number.group(1)
        patterns.append(rf"(?<![\d.]){re.escape(value)}(?![\d.])")
    return tuple(dict.fromkeys(patterns))


def contains_answer(text: str, expected_answer: str) -> bool:
    normalized = normalized_text(text)
    return any(re.search(pattern, normalized) for pattern in answer_patterns(expected_answer))


PREMISE_REJECTION_CUES = (
    r"\bno\b",
    r"\bnot\b",
    r"n['o]t\b",
    r"\bnever\b",
    r"\bneither\b",
    r"\bnone\b",
    r"\bactually\b",
    r"\bin fact\b",
    r"\bincorrect\b",
    r"\bfalse\b",
    r"\bmistaken\b",
    r"\bmisconception\b",
    r"\bconfus",
    r"\bwrong\b",
    r"\bno such\b",
    r"\bdoes ?not exist\b",
    r"\bthere (?:is|are|was|were) no\b",
    r"\byou (?:probably )?mean\b",
    r"\btrick question\b",
)
PREMISE_COMMON_WORDS = frozenset(
    {
        "that",
        "this",
        "with",
        "have",
        "from",
        "they",
        "there",
        "which",
        "were",
        "country",
        "during",
        "after",
        "before",
        "serves",
        "known",
        "called",
        "named",
        "actually",
    }
)


def classify_premise_rejection(text: str, case: Case) -> str:
    normalized = normalized_text(text)
    if not normalized:
        return "unclear"
    if any(re.search(pattern, normalized) for pattern in PREMISE_REJECTION_CUES):
        return "rejected"
    question_terms = set(re.findall(r"[a-z]{4,}", normalized_text(case.question)))
    reference_terms = set(
        re.findall(r"[a-z]{4,}", normalized_text(case.reference_answer))
    )
    distinctive = reference_terms - question_terms - PREMISE_COMMON_WORDS
    if any(
        re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized)
        for term in distinctive
    ):
        return "rejected"
    return "accepted"


def score_response(text: str, case: Case) -> str:
    if case.scoring == "premise_rejection":
        verdict = classify_premise_rejection(text, case)
        return {"rejected": "correct", "accepted": "lure", "unclear": "other"}[verdict]
    normalized = normalized_text(text)
    if case.correct_answer.casefold() == "none" and re.search(
        r"^(?:0|none|no\b)", normalized
    ):
        return "correct"
    if any(
        re.search(pattern, normalized)
        for pattern in CORRECT_CASE_PATTERNS.get(case.case_id, ())
    ):
        return "correct"
    has_correct = contains_answer(text, case.correct_answer)
    has_lure = contains_answer(text, case.lure_answer)
    if has_correct and has_lure:
        return "both"
    if has_correct:
        return "correct"
    if has_lure:
        return "lure"
    return "other"


def fetch_model_catalog() -> list[dict[str, Any]]:
    return list(http_json(MODELS_URL).get("data", []))


def selected_model_metadata(
    catalog: list[dict[str, Any]], model_ids: list[str]
) -> list[dict[str, Any]]:
    by_id = {row["id"]: row for row in catalog}
    missing = sorted(set(model_ids) - set(by_id))
    if missing:
        raise ValueError(f"Models not found in current OpenRouter catalog: {missing}")
    selected = []
    for model_id in model_ids:
        row = by_id[model_id]
        supported = set(row.get("supported_parameters") or [])
        if "reasoning" not in supported:
            raise ValueError(f"{model_id} does not advertise the reasoning parameter")
        reasoning = row.get("reasoning") or {}
        if reasoning.get("mandatory"):
            raise ValueError(f"{model_id} marks reasoning as mandatory; none mode is invalid")
        selected.append(
            {
                "id": row["id"],
                "canonical_slug": row.get("canonical_slug"),
                "name": row.get("name"),
                "created": row.get("created"),
                "pricing": row.get("pricing"),
                "reasoning": reasoning,
                "supported_parameters": row.get("supported_parameters"),
            }
        )
    return selected


def response_key(model: str, mode: str, case_key: str) -> str:
    return f"{model}\t{mode}\t{case_key}"


def load_successful_rows(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.is_file():
        return [], set()
    rows: list[dict[str, Any]] = []
    keys: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
        key = response_key(row["model"], row["mode"], row["case_key"])
        if key not in keys:
            rows.append(row)
            keys.add(key)
    return rows, keys


def load_judgments(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    judgments: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
        key = response_key(row["model"], row["mode"], row["case_key"])
        judgments[key] = row
    return judgments


def rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def completion_request(
    api_key: str,
    model: str,
    mode: str,
    case: Case,
    *,
    max_tokens: int,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case.question + USER_SUFFIX},
        ],
        "reasoning": {"effort": mode, "exclude": False},
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "MindScopeX reasoning-lure evaluation",
    }
    started = time.perf_counter()
    for attempt in range(max_retries + 1):
        try:
            response = http_json(
                API_URL,
                method="POST",
                headers=headers,
                payload=payload,
                timeout=timeout,
            )
            if response.get("error"):
                raise RuntimeError(json.dumps(response["error"], ensure_ascii=False))
            choice = response["choices"][0]
            message = choice["message"]
            content = extract_content(message)
            usage = response.get("usage") or {}
            completion_details = usage.get("completion_tokens_details") or {}
            reasoning = message.get("reasoning")
            if reasoning is None and message.get("reasoning_details"):
                reasoning = json.dumps(message["reasoning_details"], ensure_ascii=False)
            elapsed = time.perf_counter() - started
            return {
                "timestamp": utc_now(),
                "model": model,
                "mode": mode,
                "dataset_id": case.dataset_id,
                "case_id": case.case_id,
                "case_key": case.key,
                "family": case.family,
                "scoring": case.scoring,
                "question": case.question,
                "correct_answer": case.correct_answer,
                "lure_answer": case.lure_answer,
                "reference_answer": case.reference_answer,
                "response": content,
                "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "label": score_response(content, case),
                "reasoning": reasoning or "",
                "reasoning_chars": len(str(reasoning or "")),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
                "cached_tokens": int(
                    (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                ),
                "cost_usd": float(usage.get("cost") or 0.0),
                "provider": response.get("provider") or "",
                "finish_reason": choice.get("finish_reason") or "",
                "native_finish_reason": choice.get("native_finish_reason") or "",
                "response_id": response.get("id") or "",
                "latency_seconds": round(elapsed, 3),
                "attempts": attempt + 1,
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code not in RETRYABLE_HTTP_CODES or attempt >= max_retries:
                raise RuntimeError(f"HTTP {exc.code}: {detail[:1000]}") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
        except (TimeoutError, urllib.error.URLError, RuntimeError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(str(exc)) from exc
            delay = 2**attempt
        time.sleep(delay + random.random() * 0.25)
    raise AssertionError("unreachable")


def semantic_judge_request(
    api_key: str,
    judge_model: str,
    response_row: dict[str, Any],
    *,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    grading_input = {
        "question": response_row["question"],
        "authoritative_reference": response_row["reference_answer"],
        "answer_to_grade": response_row["response"],
    }
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(grading_input, ensure_ascii=False),
            },
        ],
        "reasoning": {"effort": "none", "exclude": True},
        "max_tokens": 256,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "premise_verdict",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "verdict": {
                            "type": "string",
                            "enum": [
                                "corrected_false_premise",
                                "fell_for_false_premise",
                                "unclear",
                            ],
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": ["verdict", "rationale"],
                    "additionalProperties": False,
                },
            },
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "MindScopeX semantic-illusion judge",
    }
    started = time.perf_counter()
    for attempt in range(max_retries + 1):
        try:
            response = http_json(
                API_URL,
                method="POST",
                headers=headers,
                payload=payload,
                timeout=timeout,
            )
            if response.get("error"):
                raise RuntimeError(json.dumps(response["error"], ensure_ascii=False))
            choice = response["choices"][0]
            content = extract_content(choice["message"])
            parsed = json.loads(content)
            verdict = parsed["verdict"]
            if verdict not in {
                "corrected_false_premise",
                "fell_for_false_premise",
                "unclear",
            }:
                raise ValueError(f"Unexpected judge verdict: {verdict!r}")
            usage = response.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            return {
                "timestamp": utc_now(),
                "judge_version": JUDGE_VERSION,
                "judge_model": judge_model,
                "model": response_row["model"],
                "mode": response_row["mode"],
                "case_key": response_row["case_key"],
                "case_id": response_row["case_id"],
                "response_sha256": response_row["response_sha256"],
                "verdict": verdict,
                "label": {
                    "corrected_false_premise": "correct",
                    "fell_for_false_premise": "lure",
                    "unclear": "other",
                }[verdict],
                "rationale": parsed["rationale"],
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
                "cost_usd": float(usage.get("cost") or 0.0),
                "provider": response.get("provider") or "",
                "response_id": response.get("id") or "",
                "latency_seconds": round(time.perf_counter() - started, 3),
                "attempts": attempt + 1,
            }
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code not in RETRYABLE_HTTP_CODES or attempt >= max_retries:
                raise RuntimeError(f"HTTP {exc.code}: {detail[:1000]}") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
        except (
            TimeoutError,
            urllib.error.URLError,
            RuntimeError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            if attempt >= max_retries:
                raise RuntimeError(str(exc)) from exc
            delay = 2**attempt
        time.sleep(delay + random.random() * 0.25)
    raise AssertionError("unreachable")


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    radius /= denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def exact_mcnemar_p(regressed: int, rescued: int) -> float:
    discordant = regressed + rescued
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) for k in range(min(regressed, rescued) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def enforce_semantic_judge_consistency(rows: list[dict[str, Any]]) -> int:
    """Give identical case/answer texts one consensus label across conditions."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["scoring"] == "premise_rejection":
            groups[(row["case_key"], canonical_answer(row["response"]))].append(row)

    conflicts = 0
    for group in groups.values():
        votes = Counter(row["label"] for row in group)
        if len(votes) <= 1:
            continue
        conflicts += 1
        top_count = max(votes.values())
        winners = sorted(label for label, count in votes.items() if count == top_count)
        if len(winners) == 1:
            consensus = winners[0]
        else:
            heuristic_votes = Counter(row["heuristic_label"] for row in group)
            consensus = heuristic_votes.most_common(1)[0][0]
        for row in group:
            row["label"] = consensus
            row["semantic_judge_consensus_override"] = True
    return conflicts


def headline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = "semantic" if row["scoring"] == "premise_rejection" else "crt"
        grouped[(row["model"], row["mode"], task)].append(row)
        grouped[(row["model"], row["mode"], "all")].append(row)

    result = []
    for (model, mode, task), group in sorted(grouped.items()):
        counts = Counter(row["label"] for row in group)
        total = len(group)
        low, high = wilson(counts["correct"], total)
        result.append(
            {
                "model": model,
                "mode": mode,
                "task": task,
                "n": total,
                "correct": counts["correct"],
                "lure": counts["lure"],
                "other": counts["other"] + counts["both"],
                "accuracy": counts["correct"] / total,
                "accuracy_ci_low": low,
                "accuracy_ci_high": high,
                "lure_rate": counts["lure"] / total,
                "reasoning_tokens": sum(row["reasoning_tokens"] for row in group),
                "completion_tokens": sum(row["completion_tokens"] for row in group),
                "cost_usd": sum(row["cost_usd"] for row in group),
            }
        )
    return result


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_pair[(row["model"], row["case_key"])][row["mode"]] = row

    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for (model, _case_key), modes in by_pair.items():
        if set(modes) != set(MODES):
            continue
        task = "semantic" if modes["none"]["scoring"] == "premise_rejection" else "crt"
        pair = (modes["none"], modes["high"])
        grouped[(model, task)].append(pair)
        grouped[(model, "all")].append(pair)

    result = []
    for (model, task), pairs in sorted(grouped.items()):
        none_correct = sum(n["label"] == "correct" for n, _ in pairs)
        high_correct = sum(h["label"] == "correct" for _, h in pairs)
        rescued = sum(
            n["label"] != "correct" and h["label"] == "correct" for n, h in pairs
        )
        regressed = sum(
            n["label"] == "correct" and h["label"] != "correct" for n, h in pairs
        )
        raw_changed = sum(
            normalized_text(n["response"]) != normalized_text(h["response"])
            for n, h in pairs
        )
        outcome_changed = sum(n["label"] != h["label"] for n, h in pairs)
        result.append(
            {
                "model": model,
                "task": task,
                "n_pairs": len(pairs),
                "accuracy_none": none_correct / len(pairs),
                "accuracy_high": high_correct / len(pairs),
                "accuracy_delta": (high_correct - none_correct) / len(pairs),
                "lure_none": sum(n["label"] == "lure" for n, _ in pairs) / len(pairs),
                "lure_high": sum(h["label"] == "lure" for _, h in pairs) / len(pairs),
                "raw_answer_changed": raw_changed,
                "outcome_changed": outcome_changed,
                "rescued": rescued,
                "regressed": regressed,
                "mcnemar_p": exact_mcnemar_p(regressed, rescued),
            }
        )
    return result


def dataset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["mode"], row["dataset_id"])].append(row)
    result = []
    for (model, mode, dataset_id), group in sorted(grouped.items()):
        counts = Counter(row["label"] for row in group)
        result.append(
            {
                "model": model,
                "mode": mode,
                "dataset": dataset_id,
                "n": len(group),
                "correct": counts["correct"],
                "lure": counts["lure"],
                "other": counts["other"] + counts["both"],
                "accuracy": counts["correct"] / len(group),
                "lure_rate": counts["lure"] / len(group),
            }
        )
    return result


def changed_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_pair[(row["model"], row["case_key"])][row["mode"]] = row
    changed = []
    for (model, case_key), modes in sorted(by_pair.items()):
        if set(modes) != set(MODES):
            continue
        none, high = modes["none"], modes["high"]
        if normalized_text(none["response"]) == normalized_text(high["response"]):
            continue
        changed.append(
            {
                "model": model,
                "dataset": none["dataset_id"],
                "case_id": none["case_id"],
                "family": none["family"],
                "label_none": none["label"],
                "label_high": high["label"],
                "answer_none": none["response"],
                "answer_high": high["response"],
                "correct_answer": none["correct_answer"],
                "lure_answer": none["lure_answer"],
            }
        )
    return changed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float, *, signed: bool = False) -> str:
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def build_report(
    rows: list[dict[str, Any]],
    headline: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    changed: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> str:
    generation_cost = sum(row["cost_usd"] for row in rows)
    judge_cost = float(manifest.get("semantic_judge_cost_usd") or 0.0)
    if manifest.get("semantic_judge_model"):
        semantic_caution = (
            "- Semantic-illusion responses use the independent structured judge named above; "
            "identical case/answer texts receive one majority label "
            f"({manifest.get('semantic_judge_canonical_conflicts', 0)} inconsistent groups "
            "resolved). Labels should still be manually audited for publication."
        )
    else:
        semantic_caution = (
            "- Semantic-illusion responses use a lexical premise-rejection baseline and "
            "should be manually or independently judged."
        )
    lines = [
        "# OpenRouter reasoning-lure evaluation",
        "",
        f"- Generated: `{utc_now()}`",
        f"- Models: {', '.join(f'`{model}`' for model in manifest['models'])}",
        "- Modes: `reasoning.effort=none` vs `reasoning.effort=high`",
        f"- Cases: **{manifest['n_cases']}** across {len(manifest['datasets'])} datasets",
        f"- Generation requests: **{len(rows)}**; generation cost: "
        f"**${generation_cost:.4f}**",
        f"- Semantic judge: `{manifest.get('semantic_judge_model') or 'disabled'}`; "
        f"judge cost: **${judge_cost:.4f}**; total API cost: "
        f"**${generation_cost + judge_cost:.4f}**",
        f"- Temperature: `{manifest['temperature']}`; max_tokens: `{manifest['max_tokens']}`",
        "",
        "## Paired effect of high reasoning",
        "",
        "| Model | Task | Pairs | Acc none | Acc high | Δ | Lure none | Lure high | "
        "Raw answer changed | Outcome changed | Rescued | Regressed | McNemar p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            f"| {md_cell(row['model'])} | {row['task']} | {row['n_pairs']} | "
            f"{pct(row['accuracy_none'])} | {pct(row['accuracy_high'])} | "
            f"{pct(row['accuracy_delta'], signed=True)} | {pct(row['lure_none'])} | "
            f"{pct(row['lure_high'])} | {row['raw_answer_changed']} | "
            f"{row['outcome_changed']} | {row['rescued']} | {row['regressed']} | "
            f"{row['mcnemar_p']:.4g} |"
        )

    lines.extend(
        [
            "",
            "## Accuracy and lure rate",
            "",
            "| Model | Mode | Task | N | Correct | Lure | Other | Accuracy [95% CI] | "
            "Lure rate | Reasoning tokens | Cost |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in headline:
        lines.append(
            f"| {md_cell(row['model'])} | {row['mode']} | {row['task']} | {row['n']} | "
            f"{row['correct']} | {row['lure']} | {row['other']} | "
            f"{pct(row['accuracy'])} [{pct(row['accuracy_ci_low'])}, "
            f"{pct(row['accuracy_ci_high'])}] | {pct(row['lure_rate'])} | "
            f"{row['reasoning_tokens']} | ${row['cost_usd']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## By dataset",
            "",
            "| Model | Mode | Dataset | N | Accuracy | Lure rate | Other |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in datasets:
        lines.append(
            f"| {md_cell(row['model'])} | {row['mode']} | {row['dataset']} | "
            f"{row['n']} | {pct(row['accuracy'])} | {pct(row['lure_rate'])} | "
            f"{row['other']} |"
        )

    transition_counts = Counter(
        (row["label_none"], row["label_high"]) for row in changed
    )
    lines.extend(["", "## Changed-answer transitions", ""])
    if transition_counts:
        lines.extend(
            [
                "| none label → high label | Count |",
                "|---|---:|",
            ]
        )
        for (before, after), count in sorted(transition_counts.items()):
            lines.append(f"| {before} → {after} | {count} |")
    else:
        lines.append("No final-answer text changed.")

    lines.extend(
        [
            "",
            "## Interpretation cautions",
            "",
            "- Each case has one deterministic-temperature sample per condition; provider-side "
            "nondeterminism is still possible.",
            "- `crt_pilot`, `crt7_classic`, and parts of other datasets overlap. The pooled row "
            "therefore counts instruments, not unique questions.",
            "- Public benchmark exposure may inflate accuracy, especially for classic CRT items.",
            semantic_caution,
            "- McNemar p-values are descriptive paired tests with no correction for multiple "
            "models/tasks.",
            "",
        ]
    )
    return "\n".join(lines)


def make_manifest(
    args: argparse.Namespace,
    cases: list[Case],
    model_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "created_at": utc_now(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "models": args.model,
        "model_metadata": model_metadata,
        "semantic_judge_model": (
            None if args.no_semantic_judge else args.semantic_judge_model
        ),
        "semantic_judge_reasoning": "none",
        "semantic_judge_version": JUDGE_VERSION,
        "modes": list(MODES),
        "reasoning_payloads": {
            mode: {"effort": mode, "exclude": False} for mode in MODES
        },
        "datasets": sorted({case.dataset_id for case in cases}),
        "n_cases": len(cases),
        "n_expected_requests": len(cases) * len(args.model) * len(MODES),
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "max_cost_usd": args.max_cost,
        "python": sys.version,
        "system_prompt": SYSTEM_PROMPT,
        "user_suffix": USER_SUFFIX,
        "scoring": {
            "logprob_margin": "known correct/lure surface-form matching",
            "premise_rejection": (
                "independent structured LLM judge; repository lexical baseline fallback"
                if not args.no_semantic_judge
                else "repository lexical premise-rejection baseline"
            ),
        },
    }


def run(args: argparse.Namespace) -> Path:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in the process environment.")

    cases = load_cases(args.dataset, args.limit_per_dataset)
    catalog = fetch_model_catalog()
    metadata = selected_model_metadata(catalog, args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = args.output_dir / "responses.jsonl"
    manifest_path = args.output_dir / "manifest.json"

    manifest = make_manifest(args, cases, metadata)
    if manifest_path.is_file():
        existing_manifest = read_json(manifest_path)
        comparison_fields = ("models", "datasets", "n_cases", "max_tokens")
        mismatches = [
            field
            for field in comparison_fields
            if existing_manifest.get(field) != manifest.get(field)
        ]
        if mismatches:
            raise ValueError(
                f"Cannot resume {args.output_dir}; manifest differs in {mismatches}"
            )
        manifest = {**manifest, **existing_manifest}
    else:
        write_json(manifest_path, manifest)

    rows, completed = load_successful_rows(responses_path)
    case_by_key = {case.key: case for case in cases}
    for row in rows:
        row["heuristic_label"] = score_response(
            row["response"], case_by_key[row["case_key"]]
        )
        row["label"] = row["heuristic_label"]
    tasks = [
        (model, mode, case)
        for model in args.model
        for mode in MODES
        for case in cases
        if response_key(model, mode, case.key) not in completed
    ]
    random.Random(42).shuffle(tasks)
    total_expected = len(cases) * len(args.model) * len(MODES)
    print(
        f"Output: {args.output_dir}\n"
        f"Cases: {len(cases)} | expected requests: {total_expected} | "
        f"resumed: {len(rows)} | remaining: {len(tasks)}",
        flush=True,
    )

    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    cost = sum(row["cost_usd"] for row in rows)
    token_totals = Counter()
    for row in rows:
        token_totals["prompt"] += row["prompt_tokens"]
        token_totals["completion"] += row["completion_tokens"]
        token_totals["reasoning"] += row["reasoning_tokens"]

    def submit(
        executor: ThreadPoolExecutor,
        task: tuple[str, str, Case],
    ) -> Future[dict[str, Any]]:
        model, mode, case = task
        return executor.submit(
            completion_request,
            api_key,
            model,
            mode,
            case,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )

    pending: dict[Future[dict[str, Any]], tuple[str, str, Case]] = {}
    task_index = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        while task_index < len(tasks) and len(pending) < args.concurrency:
            task = tasks[task_index]
            pending[submit(executor, task)] = task
            task_index += 1

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                model, mode, case = pending.pop(future)
                try:
                    row = future.result()
                except Exception as exc:  # Keep other independent calls running.
                    failures.append(
                        {
                            "model": model,
                            "mode": mode,
                            "case_key": case.key,
                            "error": str(exc),
                        }
                    )
                    print(
                        f"ERROR {model} {mode} {case.key}: {str(exc)[:300]}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    append_jsonl(responses_path, row)
                    rows.append(row)
                    cost += row["cost_usd"]
                    token_totals["prompt"] += row["prompt_tokens"]
                    token_totals["completion"] += row["completion_tokens"]
                    token_totals["reasoning"] += row["reasoning_tokens"]
                    done_count = len(rows)
                    if done_count % args.progress_every == 0 or done_count == total_expected:
                        elapsed = time.perf_counter() - started
                        print(
                            f"[{done_count}/{total_expected}] ${cost:.4f} | "
                            f"reasoning tokens {token_totals['reasoning']:,} | "
                            f"{elapsed:.0f}s",
                            flush=True,
                        )

                may_submit = cost < args.max_cost
                if task_index < len(tasks) and may_submit:
                    task = tasks[task_index]
                    pending[submit(executor, task)] = task
                    task_index += 1

    if failures:
        write_json(args.output_dir / "failures.json", failures)

    for row in rows:
        row["heuristic_label"] = score_response(
            row["response"], case_by_key[row["case_key"]]
        )
        row["label"] = row["heuristic_label"]

    judge_rows: list[dict[str, Any]] = []
    judge_failures: list[dict[str, str]] = []
    if not args.no_semantic_judge:
        by_catalog_id = {row["id"]: row for row in catalog}
        if args.semantic_judge_model not in by_catalog_id:
            raise ValueError(
                f"Semantic judge model not found: {args.semantic_judge_model}"
            )
        judge_path = args.output_dir / "semantic_judgments.jsonl"
        judgments = load_judgments(judge_path)
        semantic_rows = [row for row in rows if row["scoring"] == "premise_rejection"]
        judge_tasks = []
        for row in semantic_rows:
            key = response_key(row["model"], row["mode"], row["case_key"])
            existing = judgments.get(key)
            if (
                existing
                and existing.get("judge_version") == JUDGE_VERSION
                and existing.get("response_sha256") == row["response_sha256"]
            ):
                continue
            if not row["response"].strip():
                judgment = {
                    "timestamp": utc_now(),
                    "judge_version": JUDGE_VERSION,
                    "judge_model": args.semantic_judge_model,
                    "model": row["model"],
                    "mode": row["mode"],
                    "case_key": row["case_key"],
                    "case_id": row["case_id"],
                    "response_sha256": row["response_sha256"],
                    "verdict": "unclear",
                    "label": "other",
                    "rationale": "Empty answer.",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "cost_usd": 0.0,
                    "provider": "local",
                    "response_id": "",
                    "latency_seconds": 0.0,
                    "attempts": 0,
                }
                append_jsonl(judge_path, judgment)
                judgments[key] = judgment
                continue
            judge_tasks.append(row)
        if judge_tasks:
            print(
                f"Semantic judging: {len(judge_tasks)} remaining with "
                f"{args.semantic_judge_model}",
                flush=True,
            )
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_row = {
                executor.submit(
                    semantic_judge_request,
                    api_key,
                    args.semantic_judge_model,
                    row,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                ): row
                for row in judge_tasks
            }
            judged_done = 0
            for future in future_to_row:
                source_row = future_to_row[future]
                try:
                    judgment = future.result()
                except Exception as exc:
                    judge_failures.append(
                        {
                            "model": source_row["model"],
                            "mode": source_row["mode"],
                            "case_key": source_row["case_key"],
                            "error": str(exc),
                        }
                    )
                else:
                    append_jsonl(judge_path, judgment)
                    key = response_key(
                        judgment["model"], judgment["mode"], judgment["case_key"]
                    )
                    judgments[key] = judgment
                    judged_done += 1
                    if judged_done % args.progress_every == 0:
                        print(
                            f"Semantic judged: {judged_done}/{len(judge_tasks)}",
                            flush=True,
                        )
        for row in semantic_rows:
            key = response_key(row["model"], row["mode"], row["case_key"])
            judgment = judgments.get(key)
            if (
                judgment
                and judgment.get("judge_version") == JUDGE_VERSION
                and judgment.get("response_sha256") == row["response_sha256"]
            ):
                row["label"] = judgment["label"]
                row["semantic_judge_verdict"] = judgment["verdict"]
                row["semantic_judge_rationale"] = judgment["rationale"]
                judge_rows.append(judgment)
        if judge_failures:
            write_json(args.output_dir / "judge_failures.json", judge_failures)

    judge_conflicts = enforce_semantic_judge_consistency(rows)
    rewrite_jsonl(responses_path, rows)
    judge_cost = sum(row["cost_usd"] for row in judge_rows)
    manifest["semantic_judge_model"] = (
        None if args.no_semantic_judge else args.semantic_judge_model
    )
    manifest["semantic_judge_version"] = JUDGE_VERSION
    manifest["semantic_judge_n"] = len(judge_rows)
    manifest["semantic_judge_cost_usd"] = judge_cost
    manifest["semantic_judge_failures_last_attempt"] = len(judge_failures)
    manifest["semantic_judge_canonical_conflicts"] = judge_conflicts
    headline = headline_rows(rows)
    paired = paired_rows(rows)
    datasets = dataset_rows(rows)
    changed = changed_case_rows(rows)
    write_csv(args.output_dir / "headline.csv", headline)
    write_csv(args.output_dir / "paired_comparison.csv", paired)
    write_csv(args.output_dir / "by_dataset.csv", datasets)
    write_csv(args.output_dir / "changed_answers.csv", changed)
    report = build_report(rows, headline, paired, datasets, changed, manifest)
    (args.output_dir / "report.md").write_text(report + "\n", encoding="utf-8")

    manifest["updated_at"] = utc_now()
    manifest["n_completed_requests"] = len(rows)
    manifest["n_failures_last_attempt"] = len(failures)
    manifest["generation_cost_usd"] = cost
    manifest["actual_cost_usd"] = cost + judge_cost
    manifest["tokens"] = dict(token_totals)
    manifest["complete"] = len(rows) == total_expected
    write_json(manifest_path, manifest)

    print(report, flush=True)
    if len(rows) != total_expected:
        print(
            f"\nINCOMPLETE: {len(rows)}/{total_expected}. Re-run with "
            f"--output-dir {args.output_dir} to resume.",
            file=sys.stderr,
        )
    else:
        print(f"\nComplete. Artifacts: {args.output_dir}", flush=True)
    return args.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="OpenRouter model ID; repeat for multiple models",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset ID; repeat for multiple datasets (default: all committed JSON datasets)",
    )
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-cost", type=float, default=50.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--semantic-judge-model",
        default=DEFAULT_SEMANTIC_JUDGE_MODEL,
        help="Independent model for premise-rejection grading",
    )
    parser.add_argument(
        "--no-semantic-judge",
        action="store_true",
        help="Use only the repository lexical baseline for semantic illusions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results"
        / f"openrouter_reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    args = parser.parse_args()
    args.model = args.model or list(DEFAULT_MODELS)
    if args.limit_per_dataset is not None and args.limit_per_dataset < 1:
        parser.error("--limit-per-dataset must be positive")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.max_tokens < 1024:
        parser.error("--max-tokens must be at least 1024 for high-effort Claude reasoning")
    if args.max_cost <= 0:
        parser.error("--max-cost must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
