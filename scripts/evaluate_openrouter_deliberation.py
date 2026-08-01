"""Compare direct answers with explicit, visible deliberation on lure tasks.

This redesign avoids treating hidden API reasoning as equivalent to a visible,
self-contained correction.  It evaluates a balanced 100-item suite:

* all 50 Hagendorff semantic illusions;
* 50 CRT items, deduplicated and excluding the ambiguous egg-yolk item.

The ``direct`` condition disables API reasoning and requests only a final
answer.  The ``deliberate`` condition enables high reasoning, asks for a
user-visible verification of calculations/entities/premises, and then requires
a self-contained final answer.  Each target is repeated three times by default.

Semantic responses are graded once with a structured high-reasoning judge on
three separate outcomes: factual adequacy, premise detection anywhere in the
response, and explicit premise correction in the final answer.  A direct
knowledge-control question is also run once per semantic item and model.

The API key must be supplied through ``OPENROUTER_API_KEY``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
import urllib.error
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluate_openrouter_reasoning import (
    API_URL,
    DEFAULT_MODELS,
    DEFAULT_SEMANTIC_JUDGE_MODEL,
    RETRYABLE_HTTP_CODES,
    Case,
    answer_patterns,
    classify_premise_rejection,
    exact_mcnemar_p,
    fetch_model_catalog,
    load_cases,
    normalized_text,
    score_response,
    selected_model_metadata,
)
from openrouter_common import (
    append_jsonl,
    extract_content,
    http_json,
    read_jsonl,
    utc_now,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / f"openrouter_deliberation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
CONDITIONS = ("direct", "deliberate")
JUDGE_VERSION = 2
DIRECT_SYSTEM = """Answer the user's question accurately.
Return only the final answer inside <final> and </final>.
Do not include analysis, explanation, or any text outside those tags."""

DELIBERATE_SYSTEM = """Answer the user's question accurately.
First write a concise, user-visible verification inside <verification> and </verification>.
The verification must check the relevant calculation, identity, factual assumption, and
question premise instead of merely repeating the question.
Then write a self-contained final answer inside <final> and </final>.
If the question contains an invalid premise, explicitly correct it in both the verification
and the final answer. Do not silently answer inside a false frame."""

SEMANTIC_JUDGE_SYSTEM = """You audit answers to questions that may contain a false premise.
Use the dataset reference to identify the invariant premise correction. Some references contain
explicitly dated or stale ancillary details; do not mark a response wrong merely because it gives
plausible newer information. Evaluate three distinct properties:

1. answer_factually_adequate: The response gives accurate, useful information responsive to
   the likely user need. A bare slot answer can be factually adequate even if it fails to
   mention a false premise.
2. overall_premise_detected: Anywhere in the complete response, it clearly recognizes or
   corrects the false premise. Merely giving a fact compatible with the intended question
   does not count.
3. final_premise_corrected: The text in final_answer is self-contained and explicitly
   corrects or unambiguously negates the false premise. A bare answer such as a country,
   person, date, title, or number does not count unless that answer itself necessarily
   negates the premise (for example zero, none, never, or no).

Judge the supplied text, not hidden intentions. Be strict and consistent."""

CONTROL_JUDGE_SYSTEM = """Judge whether an answer to a factual control question is accurate
and responsive. Use the dataset reference for the invariant fact, but allow plausible newer
information when the reference is explicitly dated or stale. Ignore style and brevity."""


@dataclass(frozen=True)
class EvalTask:
    model: str
    kind: str
    condition: str
    repeat: int
    case: Case
    question: str

    @property
    def key(self) -> str:
        return "\t".join(
            (self.model, self.kind, self.condition, str(self.repeat), self.case.key)
        )


def first_number(case_id: str) -> int:
    match = re.search(r"(\d+)$", case_id)
    return int(match.group(1)) if match else 0


_BALANCED_CRT_SEED_DATASETS = {
    "crt2",
    "crt7_classic",
    "crt_pilot",
    "verbal_crt",
    "yax_crt_isomorph",
}


def select_balanced_cases() -> tuple[list[Case], list[Case]]:
    all_cases = load_cases(None, None)
    semantic = [case for case in all_cases if case.scoring == "premise_rejection"]
    if len(semantic) != 50:
        raise ValueError(f"Expected 50 semantic illusions, got {len(semantic)}")

    # Keep the original published/pilot seed set fixed so adding a new JSON
    # dataset does not silently change this report's historical 50-item arm.
    # New synthetic sets are evaluated explicitly with their dataset IDs.
    crt: list[Case] = []
    seen_questions: set[str] = set()
    for case in all_cases:
        if case.dataset_id not in _BALANCED_CRT_SEED_DATASETS:
            continue
        if case.case_id == "verbal_crt_010":
            continue
        normalized = normalized_text(case.question)
        if normalized in seen_questions:
            continue
        seen_questions.add(normalized)
        crt.append(case)

    # Add deterministic, evenly spaced Hagendorff variants until the CRT arm
    # contains 50 items. Rotate families so one template cannot dominate.
    hag = [case for case in all_cases if case.dataset_id == "hagendorff_crt"]
    by_family: dict[str, list[Case]] = defaultdict(list)
    for case in hag:
        by_family[case.family].append(case)
    for family in by_family:
        by_family[family].sort(key=lambda case: first_number(case.case_id))

    family_names = sorted(by_family)
    cursors = {family: 0 for family in family_names}
    while len(crt) < 50:
        made_progress = False
        for family in family_names:
            rows = by_family[family]
            cursor = cursors[family]
            if cursor >= len(rows):
                continue
            # Spacing by 7 gives broad numerical/surface coverage without
            # simply taking the first near-duplicate prompts.
            index = (cursor * 7) % len(rows)
            candidate = rows[index]
            cursors[family] += 1
            normalized = normalized_text(candidate.question)
            if normalized in seen_questions:
                continue
            seen_questions.add(normalized)
            crt.append(candidate)
            made_progress = True
            if len(crt) == 50:
                break
        if not made_progress:
            raise ValueError("Unable to construct 50 unique CRT cases")
    return crt, semantic


def parse_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.I | re.S)
    return match.group(1).strip() if match else ""


def extract_final(text: str) -> tuple[str, bool]:
    tagged = parse_tag(text, "final")
    if tagged:
        return tagged, True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (lines[-1] if lines else ""), False


STRICT_CORRECTION_PATTERNS = (
    r"\bno\b",
    r"\bnot\b",
    r"n['’]t\b",
    r"\bnever\b",
    r"\bnone\b",
    r"\bzero\b",
    r"(?<!\d)0(?!\d)",
    r"\bactually\b",
    r"\bin fact\b",
    r"\bincorrect\b",
    r"\bfalse\b",
    r"\bmistaken\b",
    r"\bmisconception\b",
    r"\bconfus",
    r"\bwrong\b",
    r"\brather\b",
    r"\binstead\b",
    r"\bhowever\b",
    r"\bwhereas\b",
    r"\bbut\b",
    r"\byou (?:probably )?mean\b",
    r"\bif you meant\b",
    r"\bproperly\b",
    r"\breal(?:ly)? (?:called|named)\b",
    r"\b(?:mixing|mixed|combining|combined) (?:up )?\b",
    r"\btrick question\b",
)


def strict_premise_correction(text: str) -> bool:
    """Conservative explicit-correction detector with no reference leakage."""

    normalized = normalized_text(text)
    return any(re.search(pattern, normalized) for pattern in STRICT_CORRECTION_PATTERNS)


def broad_premise_correction(text: str, case: Case) -> bool:
    """Reference-aware upper-bound detector used for sensitivity analysis."""

    return classify_premise_rejection(text, case) == "rejected"


LOCAL_CRT_CORRECT_PATTERNS = {
    "verbal_crt_002": (r"\bsecond place\b",),
    "verbal_crt_003": (
        r"\bdo not bury (?:the )?survivors\b",
        r"\bdo not bury (?:them|anyone)\b",
        r"\bsurvivors\b.*\b(?:alive|not buried)\b",
    ),
    "verbal_crt_004": (
        r"\bno banana\b",
        r"\bcoconut\b.*\b(?:does not|do not|doesn't|don't) (?:grow|have) bananas?\b",
    ),
    "verbal_crt_005": (r"\b(?:no|not any|were not|weren't) stairs\b",),
    "verbal_crt_006": (
        r"\bmoses\b.*\b(?:no|not|none|0)\b.*\banimals?\b",
        r"\bnoah\b.*\bark\b",
    ),
    "verbal_crt_007": (r"\b(?:no|not produce) smoke\b",),
    "verbal_crt_008": (r"\bmatch\b",),
    "verbal_crt_009": (
        r"\b(?:dead|deceased|impossible|cannot marry|can't marry)\b",
    ),
}


def _first_match_position(text: str, expected: str) -> int | None:
    normalized = normalized_text(text)
    positions = [
        match.start()
        for pattern in answer_patterns(expected)
        if (match := re.search(pattern, normalized))
    ]
    return min(positions) if positions else None


def _first_decimal(text: str) -> str:
    plain = re.sub(r"[*_`]", "", text)
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", plain)
    if not match:
        return ""
    raw = match.group(0).replace(",", "")
    try:
        return str(float(raw))
    except ValueError:
        return ""


def score_crt_visible_final(text: str, case: Case) -> str:
    """Score the asserted final result, not lure values mentioned in explanation."""

    normalized = normalized_text(text)
    if case.correct_answer.casefold() == "none" and (
        re.search(r"\b(?:0|zero|none|no)\b", normalized)
        or re.search(r"\bempty\b", normalized)
    ):
        return "correct"
    if any(
        re.search(pattern, normalized)
        for pattern in LOCAL_CRT_CORRECT_PATTERNS.get(case.case_id, ())
    ):
        return "correct"

    correct_number = _first_decimal(case.correct_answer)
    lure_number = _first_decimal(case.lure_answer)
    normalized_correct = normalized_text(case.correct_answer)
    normalized_lure = normalized_text(case.lure_answer)
    has_specific_correct = bool(
        normalized_correct
        and re.search(rf"(?<!\w){re.escape(normalized_correct)}(?!\w)", normalized)
    )
    has_specific_lure = bool(
        normalized_lure
        and re.search(rf"(?<!\w){re.escape(normalized_lure)}(?!\w)", normalized)
    )
    if has_specific_correct and not has_specific_lure:
        return "correct"
    if has_specific_lure and not has_specific_correct:
        return "lure"

    ordinal_words = {"1.0": "first", "2.0": "second", "3.0": "third"}
    correct_unit = re.search(r"\b(weeks?|days?|hours?|minutes?)\b", normalized_correct)
    if correct_number in ordinal_words and correct_unit:
        unit_root = correct_unit.group(1).rstrip("s")
        if re.search(
            rf"\b{ordinal_words[correct_number]}\s+{unit_root}s?\b",
            normalized,
        ):
            return "correct"

    asserted_number = _first_decimal(text)
    if asserted_number and correct_number and asserted_number == correct_number:
        return "correct"
    if asserted_number and lure_number and asserted_number == lure_number:
        return "lure"

    label = score_response(text, case)
    if label != "both":
        return label
    correct_position = _first_match_position(text, case.correct_answer)
    lure_position = _first_match_position(text, case.lure_answer)
    if correct_position is not None and (
        lure_position is None or correct_position < lure_position
    ):
        return "correct"
    if lure_position is not None:
        return "lure"
    return "other"


def retry_json_request(
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
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
            return response
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code not in RETRYABLE_HTTP_CODES or attempt >= max_retries:
                raise RuntimeError(f"HTTP {exc.code}: {detail[:1000]}") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
        except (TimeoutError, urllib.error.URLError, RuntimeError, KeyError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(str(exc)) from exc
            delay = 2**attempt
        time.sleep(delay + random.random() * 0.25)
    raise AssertionError("unreachable")


def generation_request(
    api_key: str,
    task: EvalTask,
    *,
    max_tokens: int,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    system = DELIBERATE_SYSTEM if task.condition == "deliberate" else DIRECT_SYSTEM
    effort = "high" if task.condition == "deliberate" else "none"
    payload = {
        "model": task.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": task.question},
        ],
        "reasoning": {"effort": effort, "exclude": False},
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "MindScopeX visible-deliberation evaluation",
    }
    started = time.perf_counter()
    response = retry_json_request(
        payload,
        headers,
        timeout=timeout,
        max_retries=max_retries,
    )
    choice = response["choices"][0]
    message = choice["message"]
    content = extract_content(message)
    final, final_tag_found = extract_final(content)
    verification = parse_tag(content, "verification")
    reasoning = message.get("reasoning")
    if reasoning is None and message.get("reasoning_details"):
        reasoning = json.dumps(message["reasoning_details"], ensure_ascii=False)
    usage = response.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "timestamp": utc_now(),
        "task_key": task.key,
        "model": task.model,
        "kind": task.kind,
        "condition": task.condition,
        "repeat": task.repeat,
        "dataset_id": task.case.dataset_id,
        "case_id": task.case.case_id,
        "case_key": task.case.key,
        "family": task.case.family,
        "scoring": task.case.scoring,
        "question": task.question,
        "original_question": task.case.question,
        "correct_answer": task.case.correct_answer,
        "lure_answer": task.case.lure_answer,
        "reference_answer": task.case.reference_answer,
        "response": content,
        "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "verification": verification,
        "final_answer": final,
        "final_tag_found": final_tag_found,
        "crt_label": (
            score_response(final, task.case)
            if task.kind == "target" and task.case.scoring != "premise_rejection"
            else ""
        ),
        "reasoning": reasoning or "",
        "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cached_tokens": int(details.get("cached_tokens") or 0),
        "cost_usd": float(usage.get("cost") or 0.0),
        "finish_reason": choice.get("finish_reason") or "",
        "native_finish_reason": choice.get("native_finish_reason") or "",
        "provider": response.get("provider") or "",
        "response_id": response.get("id") or "",
        "latency_seconds": round(time.perf_counter() - started, 3),
    }


def judge_request(
    api_key: str,
    judge_model: str,
    row: dict[str, Any],
    *,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    is_control = row["kind"] == "knowledge_control"
    if is_control:
        system = CONTROL_JUDGE_SYSTEM
        grading_input = {
            "control_question": row["question"],
            "authoritative_reference": row["reference_answer"],
            "answer": row["final_answer"],
        }
        properties: dict[str, Any] = {
            "answer_factually_adequate": {"type": "boolean"},
            "rationale": {"type": "string"},
        }
    else:
        system = SEMANTIC_JUDGE_SYSTEM
        grading_input = {
            "question": row["question"],
            "authoritative_reference": row["reference_answer"],
            "complete_response": row["response"],
            "verification": row["verification"],
            "final_answer": row["final_answer"],
        }
        properties = {
            "answer_factually_adequate": {"type": "boolean"},
            "overall_premise_detected": {"type": "boolean"},
            "final_premise_corrected": {"type": "boolean"},
            "rationale": {"type": "string"},
        }
    required = list(properties)
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(grading_input, ensure_ascii=False)},
        ],
        "reasoning": {"effort": "high", "exclude": True},
        # High-effort judging can consume the entire completion budget as hidden
        # reasoning before emitting the structured result. Keep enough headroom
        # for the final JSON, especially when grading long model answers.
        "max_tokens": 8192,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "deliberation_audit",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "MindScopeX deliberation judge",
    }
    started = time.perf_counter()
    response = retry_json_request(
        payload,
        headers,
        timeout=timeout,
        max_retries=max_retries,
    )
    choice = response["choices"][0]
    parsed = json.loads(extract_content(choice["message"]))
    usage = response.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "timestamp": utc_now(),
        "judge_version": JUDGE_VERSION,
        "judge_key": row["task_key"],
        "judge_model": judge_model,
        "model": row["model"],
        "kind": row["kind"],
        "condition": row["condition"],
        "repeat": row["repeat"],
        "case_key": row["case_key"],
        "response_sha256": row["response_sha256"],
        **parsed,
        "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
        "cost_usd": float(usage.get("cost") or 0.0),
        "provider": response.get("provider") or "",
        "response_id": response.get("id") or "",
        "latency_seconds": round(time.perf_counter() - started, 3),
    }


def run_concurrent(
    tasks: list[Any],
    worker: Any,
    *,
    concurrency: int,
    progress_every: int,
    on_success: Any,
) -> list[tuple[Any, str]]:
    failures: list[tuple[Any, str]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        pending: dict[Future[Any], Any] = {
            executor.submit(worker, task): task for task in tasks
        }
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - persist all API failures
                    failures.append((task, str(exc)))
                else:
                    on_success(result)
                completed += 1
                if completed % progress_every == 0 or completed == len(tasks):
                    print(
                        f"Progress: {completed}/{len(tasks)}; failures={len(failures)}",
                        flush=True,
                    )
    return failures


def load_generation_map(path: Path) -> dict[str, dict[str, Any]]:
    return {row["task_key"]: row for row in read_jsonl(path)}


def load_judgment_map(path: Path) -> dict[str, dict[str, Any]]:
    return {row["judge_key"]: row for row in read_jsonl(path)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    rows: list[dict[str, Any]],
    judgments: dict[str, dict[str, Any]],
    models: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_by_key = {case.key: case for case in load_cases(None, None)}
    controls: dict[tuple[str, str], bool] = {}
    for row in rows:
        if row["kind"] != "knowledge_control":
            continue
        judgment = judgments.get(row["task_key"], {})
        controls[(row["model"], row["case_key"])] = bool(
            judgment.get("answer_factually_adequate")
        )

    summary: list[dict[str, Any]] = []
    for model in models:
        for condition in CONDITIONS:
            group = [
                row
                for row in rows
                if row["model"] == model
                and row["condition"] == condition
                and row["kind"] == "target"
            ]
            crt = [row for row in group if row["scoring"] != "premise_rejection"]
            semantic = [row for row in group if row["scoring"] == "premise_rejection"]
            sem_judgments = [judgments[row["task_key"]] for row in semantic]
            crt_correct = sum(
                score_crt_visible_final(
                    row["final_answer"], case_by_key[row["case_key"]]
                )
                == "correct"
                for row in crt
            )
            final_corrected = sum(
                bool(judgment["final_premise_corrected"])
                for judgment in sem_judgments
            )
            detected = sum(
                bool(judgment["overall_premise_detected"])
                for judgment in sem_judgments
            )
            adequate = sum(
                bool(judgment["answer_factually_adequate"])
                for judgment in sem_judgments
            )
            qualified = [
                (row, judgment)
                for row, judgment in zip(semantic, sem_judgments, strict=True)
                if controls.get((model, row["case_key"]), False)
            ]
            qualified_lure = sum(
                not bool(judgment["final_premise_corrected"])
                for _, judgment in qualified
            )
            semantic_incorrect = len(semantic) - final_corrected
            summary.append(
                {
                    "model": model,
                    "condition": condition,
                    "target_responses": len(group),
                    "crt_n": len(crt),
                    "crt_correct": crt_correct,
                    "crt_accuracy": crt_correct / len(crt),
                    "semantic_n": len(semantic),
                    "semantic_answer_adequacy": adequate / len(semantic),
                    "semantic_overall_detection": detected / len(semantic),
                    "semantic_final_correction": final_corrected / len(semantic),
                    "semantic_knowledge_qualified_n": len(qualified),
                    "semantic_qualified_lure_count": qualified_lure,
                    "semantic_qualified_lure_rate": (
                        qualified_lure / len(qualified) if qualified else 0.0
                    ),
                    "semantic_lure_among_incorrect": (
                        qualified_lure / semantic_incorrect
                        if semantic_incorrect
                        else 0.0
                    ),
                    "combined_trap_avoidance": (
                        (crt_correct + final_corrected) / len(group)
                    ),
                    "reasoning_tokens": sum(row["reasoning_tokens"] for row in group),
                    "zero_reasoning_responses": sum(
                        row["reasoning_tokens"] == 0 for row in group
                    ),
                    "length_finishes": sum(
                        row["finish_reason"] == "length" for row in group
                    ),
                    "tag_compliance": sum(row["final_tag_found"] for row in group)
                    / len(group),
                    "generation_cost_usd": sum(row["cost_usd"] for row in group),
                }
            )

    paired: list[dict[str, Any]] = []
    by_pair: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["kind"] == "target":
            by_pair[(row["model"], row["repeat"], row["case_key"])][
                row["condition"]
            ] = row
    for model in models:
        pairs = [
            pair
            for (pair_model, _, _), pair in by_pair.items()
            if pair_model == model and set(pair) == set(CONDITIONS)
        ]
        for task_kind in ("crt", "semantic", "all"):
            selected = []
            for pair in pairs:
                is_semantic = pair["direct"]["scoring"] == "premise_rejection"
                if task_kind == "crt" and is_semantic:
                    continue
                if task_kind == "semantic" and not is_semantic:
                    continue
                selected.append(pair)

            def success(row: dict[str, Any]) -> bool:
                if row["scoring"] != "premise_rejection":
                    return (
                        score_crt_visible_final(
                            row["final_answer"], case_by_key[row["case_key"]]
                        )
                        == "correct"
                    )
                return bool(judgments[row["task_key"]]["final_premise_corrected"])

            rescued = sum(
                not success(pair["direct"]) and success(pair["deliberate"])
                for pair in selected
            )
            regressed = sum(
                success(pair["direct"]) and not success(pair["deliberate"])
                for pair in selected
            )
            paired.append(
                {
                    "model": model,
                    "task": task_kind,
                    "pairs": len(selected),
                    "direct_success": sum(success(pair["direct"]) for pair in selected)
                    / len(selected),
                    "deliberate_success": sum(
                        success(pair["deliberate"]) for pair in selected
                    )
                    / len(selected),
                    "rescued": rescued,
                    "regressed": regressed,
                    "mcnemar_p": exact_mcnemar_p(regressed, rescued),
                    "final_answer_changed": sum(
                        normalized_text(pair["direct"]["final_answer"])
                        != normalized_text(pair["deliberate"]["final_answer"])
                        for pair in selected
                    ),
                }
            )
    return summary, paired


def aggregate_local_sensitivity(
    rows: list[dict[str, Any]],
    judgments: dict[str, dict[str, Any]],
    models: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate complete generations without requiring complete LLM judgments."""

    case_by_key = {case.key: case for case in load_cases(None, None)}
    summary: list[dict[str, Any]] = []
    for model in models:
        for condition in CONDITIONS:
            group = [
                row
                for row in rows
                if row["model"] == model
                and row["condition"] == condition
                and row["kind"] == "target"
            ]
            crt = [row for row in group if row["scoring"] != "premise_rejection"]
            semantic = [row for row in group if row["scoring"] == "premise_rejection"]
            strict_final = sum(
                strict_premise_correction(row["final_answer"]) for row in semantic
            )
            broad_final = sum(
                broad_premise_correction(
                    row["final_answer"], case_by_key[row["case_key"]]
                )
                for row in semantic
            )
            strict_overall = sum(
                strict_premise_correction(row["response"]) for row in semantic
            )
            broad_overall = sum(
                broad_premise_correction(row["response"], case_by_key[row["case_key"]])
                for row in semantic
            )
            judged = [
                judgments[row["task_key"]]
                for row in semantic
                if row["task_key"] in judgments
            ]
            crt_correct = sum(
                score_crt_visible_final(
                    row["final_answer"], case_by_key[row["case_key"]]
                )
                == "correct"
                for row in crt
            )
            summary.append(
                {
                    "model": model,
                    "condition": condition,
                    "target_responses": len(group),
                    "crt_n": len(crt),
                    "crt_accuracy": crt_correct / len(crt),
                    "semantic_n": len(semantic),
                    "semantic_strict_final_correction": strict_final / len(semantic),
                    "semantic_broad_final_correction": broad_final / len(semantic),
                    "semantic_strict_overall_detection": strict_overall / len(semantic),
                    "semantic_broad_overall_detection": broad_overall / len(semantic),
                    "judge_coverage_n": len(judged),
                    "judge_final_correction": (
                        sum(bool(row["final_premise_corrected"]) for row in judged)
                        / len(judged)
                        if judged
                        else 0.0
                    ),
                    "judge_overall_detection": (
                        sum(bool(row["overall_premise_detected"]) for row in judged)
                        / len(judged)
                        if judged
                        else 0.0
                    ),
                    "reasoning_tokens": sum(row["reasoning_tokens"] for row in group),
                    "zero_reasoning_responses": sum(
                        row["reasoning_tokens"] == 0 for row in group
                    ),
                    "length_finishes": sum(
                        row["finish_reason"] == "length" for row in group
                    ),
                    "tag_compliance": sum(row["final_tag_found"] for row in group)
                    / len(group),
                    "generation_cost_usd": sum(row["cost_usd"] for row in group),
                }
            )

    by_pair: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["kind"] == "target":
            by_pair[(row["model"], row["repeat"], row["case_key"])][
                row["condition"]
            ] = row

    paired: list[dict[str, Any]] = []
    for model in models:
        model_pairs = [
            pair
            for (pair_model, _, _), pair in by_pair.items()
            if pair_model == model and set(pair) == set(CONDITIONS)
        ]
        for metric in ("strict", "broad"):
            for task_kind in ("crt", "semantic", "all"):
                selected = []
                for pair in model_pairs:
                    is_semantic = pair["direct"]["scoring"] == "premise_rejection"
                    if task_kind == "crt" and is_semantic:
                        continue
                    if task_kind == "semantic" and not is_semantic:
                        continue
                    selected.append(pair)

                def success(row: dict[str, Any]) -> bool:
                    if row["scoring"] != "premise_rejection":
                        return (
                            score_crt_visible_final(
                                row["final_answer"], case_by_key[row["case_key"]]
                            )
                            == "correct"
                        )
                    if metric == "strict":
                        return strict_premise_correction(row["final_answer"])
                    return broad_premise_correction(
                        row["final_answer"], case_by_key[row["case_key"]]
                    )

                rescued = sum(
                    not success(pair["direct"]) and success(pair["deliberate"])
                    for pair in selected
                )
                regressed = sum(
                    success(pair["direct"]) and not success(pair["deliberate"])
                    for pair in selected
                )
                paired.append(
                    {
                        "model": model,
                        "metric": metric,
                        "task": task_kind,
                        "pairs": len(selected),
                        "direct_success": sum(
                            success(pair["direct"]) for pair in selected
                        )
                        / len(selected),
                        "deliberate_success": sum(
                            success(pair["deliberate"]) for pair in selected
                        )
                        / len(selected),
                        "rescued": rescued,
                        "regressed": regressed,
                        "mcnemar_p": exact_mcnemar_p(regressed, rescued),
                    }
                )
    return summary, paired


def build_local_report(
    manifest: dict[str, Any],
    summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    *,
    judgment_count: int,
) -> str:
    lines = [
        "# Direct vs visible-deliberation: complete local sensitivity analysis",
        "",
        f"- Complete generations: **{manifest['generation_requests']}**",
        f"- Available high-reasoning judge audits: **{judgment_count} / 1050**",
        "- Strict correction: explicit negation/contrast cue; no reference-answer leakage",
        "- Broad correction: reference-aware lexical detector; reported as an upper bound",
        "",
        "## Complete-generation outcomes",
        "",
        "| Model | Condition | CRT | Semantic strict final | Semantic broad final | "
        "Strict detected anywhere | Broad detected anywhere | Judge audit (n) | "
        "Reasoning tokens | Zero-reasoning |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['condition']} | {pct(row['crt_accuracy'])} | "
            f"{pct(row['semantic_strict_final_correction'])} | "
            f"{pct(row['semantic_broad_final_correction'])} | "
            f"{pct(row['semantic_strict_overall_detection'])} | "
            f"{pct(row['semantic_broad_overall_detection'])} | "
            f"{pct(row['judge_final_correction'])} ({row['judge_coverage_n']}) | "
            f"{row['reasoning_tokens']} | {row['zero_reasoning_responses']} |"
        )
    lines.extend(
        [
            "",
            "## Paired final-answer effect",
            "",
            "| Model | Metric | Task | Pairs | Direct | Deliberate | Rescued | "
            "Regressed | McNemar p |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            f"| {row['model']} | {row['metric']} | {row['task']} | "
            f"{row['pairs']} | {pct(row['direct_success'])} | "
            f"{pct(row['deliberate_success'])} | {row['rescued']} | "
            f"{row['regressed']} | {row['mcnemar_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "The API credit balance was exhausted after 628 structured audits. Complete "
            "results therefore use a conservative strict detector and a reference-aware "
            "upper-bound detector. The available judge subset is reported only as an audit, "
            "not substituted for missing labels.",
        ]
    )
    return "\n".join(lines)


def pct(value: float) -> str:
    return f"{value:.1%}"


def build_report(
    manifest: dict[str, Any],
    summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> str:
    lines = [
        "# Direct vs visible-deliberation lure evaluation",
        "",
        f"- Models: {', '.join(f'`{model}`' for model in manifest['models'])}",
        f"- Balanced target suite: **{manifest['target_cases']}** cases "
        f"({manifest['crt_cases']} CRT + {manifest['semantic_cases']} semantic)",
        f"- Repeats: **{manifest['repeats']}** per target and condition",
        "- Direct: `reasoning.effort=none`, final answer only",
        "- Deliberate: `reasoning.effort=high`, visible verification + self-contained final",
        f"- Maximum completion tokens: **{manifest['max_tokens']}**",
        "",
        "## Dataset",
        "",
        "| Split | Cases | Shape | Scoring |",
        "|---|---:|---|---|",
        "| CRT | 50 | Short calculation/verbal questions with an intuitive lure answer | "
        "Exact/numeric final answer |",
        "| Semantic illusion | 50 | Questions containing a false factual premise | "
        "Explicit premise correction in the final answer |",
        "| Knowledge control | 50 per model | Neutral questions covering the same facts "
        "as the semantic items | Separates lack of knowledge from lure susceptibility |",
        "",
        "Each target case was requested three times in both conditions: 100 cases × "
        "3 repeats × 2 conditions × 3 models = 1,800 target responses. The 150 "
        "knowledge-control responses bring generation to 1,950 calls.",
        "",
        "## Reasoning effect at a glance",
        "",
        "| Model | Combined direct → deliberate | Δ | Semantic final correction "
        "direct → deliberate | Δ | Lure among semantic failures direct → deliberate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    by_model_condition = {
        (row["model"], row["condition"]): row for row in summary
    }
    for model in manifest["models"]:
        direct = by_model_condition[(model, "direct")]
        deliberate = by_model_condition[(model, "deliberate")]
        combined_delta = (
            deliberate["combined_trap_avoidance"]
            - direct["combined_trap_avoidance"]
        ) * 100
        semantic_delta = (
            deliberate["semantic_final_correction"]
            - direct["semantic_final_correction"]
        ) * 100
        lines.append(
            f"| {model} | {pct(direct['combined_trap_avoidance'])} → "
            f"{pct(deliberate['combined_trap_avoidance'])} | "
            f"{combined_delta:+.1f} pp | "
            f"{pct(direct['semantic_final_correction'])} → "
            f"{pct(deliberate['semantic_final_correction'])} | "
            f"{semantic_delta:+.1f} pp | "
            f"{pct(direct['semantic_lure_among_incorrect'])} → "
            f"{pct(deliberate['semantic_lure_among_incorrect'])} |"
        )
    lines.extend(
        [
        "",
        "## Outcomes",
        "",
        "| Model | Condition | CRT accuracy | Semantic factual adequacy | "
        "Premise detected anywhere | Premise corrected in final | "
        "Known-case lure rate | Lure among failures | Combined trap avoidance | "
        "Reasoning tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['condition']} | {pct(row['crt_accuracy'])} | "
            f"{pct(row['semantic_answer_adequacy'])} | "
            f"{pct(row['semantic_overall_detection'])} | "
            f"{pct(row['semantic_final_correction'])} | "
            f"{pct(row['semantic_qualified_lure_rate'])} | "
            f"{pct(row['semantic_lure_among_incorrect'])} | "
            f"{pct(row['combined_trap_avoidance'])} | "
            f"{row['reasoning_tokens']} |"
        )
    lines.extend(
        [
            "",
            "## Paired intervention effect",
            "",
            "| Model | Task | Pairs | Direct | Deliberate | Rescued | Regressed | "
            "McNemar p | Final changed |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            f"| {row['model']} | {row['task']} | {row['pairs']} | "
            f"{pct(row['direct_success'])} | {pct(row['deliberate_success'])} | "
            f"{row['rescued']} | {row['regressed']} | {row['mcnemar_p']:.4g} | "
            f"{row['final_answer_changed']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Semantic factual adequacy and explicit premise correction are intentionally "
            "reported separately.",
            "- Knowledge-qualified lure counts failures to correct only when the same model "
            "answered the matched control question adequately.",
            "- Lure among failures divides those knowledge-qualified lure failures by all "
            "semantic final-correction failures. It is undefined when there are no failures; "
            "the table displays 0.0% in that case.",
            "- Repeats are independent API requests at temperature 0; provider-side "
            "nondeterminism can remain.",
            "- The intervention intentionally combines API high reasoning with an explicit "
            "verification/correction instruction. It measures the practical effect of visible "
            "deliberation, not an isolated causal effect of hidden reasoning effort.",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    if args.analyze_only:
        manifest_path = args.output_dir / "manifest.json"
        generation_path = args.output_dir / "responses.jsonl"
        judgment_path = args.output_dir / "judgments.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = list(load_generation_map(generation_path).values())
        judgments = load_judgment_map(judgment_path)
        manifest["generation_requests"] = len(rows)
        case_by_key = {case.key: case for case in load_cases(None, None)}
        calibration: dict[str, Any] = {}
        for name, predictor in (
            ("strict", lambda row: strict_premise_correction(row["final_answer"])),
            (
                "broad",
                lambda row: broad_premise_correction(
                    row["final_answer"], case_by_key[row["case_key"]]
                ),
            ),
        ):
            true_positive = false_positive = true_negative = false_negative = 0
            for row in rows:
                if (
                    row["kind"] != "target"
                    or row["scoring"] != "premise_rejection"
                    or row["task_key"] not in judgments
                ):
                    continue
                predicted = bool(predictor(row))
                expected = bool(
                    judgments[row["task_key"]]["final_premise_corrected"]
                )
                if predicted and expected:
                    true_positive += 1
                elif predicted:
                    false_positive += 1
                elif expected:
                    false_negative += 1
                else:
                    true_negative += 1
            total = true_positive + false_positive + true_negative + false_negative
            calibration[name] = {
                "n": total,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "true_negative": true_negative,
                "false_negative": false_negative,
                "accuracy": (true_positive + true_negative) / total,
                "precision": (
                    true_positive / (true_positive + false_positive)
                    if true_positive + false_positive
                    else 0.0
                ),
                "recall": (
                    true_positive / (true_positive + false_negative)
                    if true_positive + false_negative
                    else 0.0
                ),
            }
        summary, paired = aggregate_local_sensitivity(
            rows, judgments, list(manifest["models"])
        )
        write_csv(args.output_dir / "local_summary.csv", summary)
        write_csv(args.output_dir / "local_paired.csv", paired)
        report = build_local_report(
            manifest,
            summary,
            paired,
            judgment_count=len(judgments),
        )
        (args.output_dir / "local_report.md").write_text(
            report + "\n", encoding="utf-8"
        )
        write_json(
            args.output_dir / "audit_manifest.json",
            {
                **manifest,
                "available_judgments": len(judgments),
                "complete_generations": len(rows) == 1950,
                "complete_judgments": len(judgments) == 1050,
                "generation_cost_usd": sum(row["cost_usd"] for row in rows),
                "judge_cost_usd": sum(
                    row["cost_usd"] for row in judgments.values()
                ),
                "actual_cost_usd": sum(row["cost_usd"] for row in rows)
                + sum(row["cost_usd"] for row in judgments.values()),
                "local_scorer_calibration": calibration,
            },
        )
        print(report, flush=True)
        return args.output_dir

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not args.dry_run and not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY before running")

    crt, semantic = select_balanced_cases()
    if args.limit_per_task_type is not None:
        crt = crt[: args.limit_per_task_type]
        semantic = semantic[: args.limit_per_task_type]
    target_cases = crt + semantic
    catalog = fetch_model_catalog()
    metadata = selected_model_metadata(catalog, args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    generation_path = args.output_dir / "responses.jsonl"
    judgment_path = args.output_dir / "judgments.jsonl"

    manifest = {
        "created_at": utc_now(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "models": args.model,
        "model_metadata": metadata,
        "conditions": list(CONDITIONS),
        "repeats": args.repeats,
        "target_cases": len(target_cases),
        "crt_cases": len(crt),
        "semantic_cases": len(semantic),
        "excluded_cases": ["verbal_crt/verbal_crt_010"],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "judge_model": args.judge_model,
        "judge_version": JUDGE_VERSION,
        "direct_system": DIRECT_SYSTEM,
        "deliberate_system": DELIBERATE_SYSTEM,
        "selected_case_keys": [case.key for case in target_cases],
    }
    write_json(manifest_path, manifest)
    if args.dry_run:
        print(
            f"Dry run OK: {len(crt)} CRT + {len(semantic)} semantic; "
            f"{len(target_cases)} targets",
            flush=True,
        )
        return args.output_dir

    tasks: list[EvalTask] = []
    for model in args.model:
        for case in target_cases:
            for repeat in range(1, args.repeats + 1):
                for condition in CONDITIONS:
                    tasks.append(
                        EvalTask(model, "target", condition, repeat, case, case.question)
                    )
        for case in semantic:
            payload = json.loads(
                (
                    ROOT / "src" / "mindscopex_analysis" / "data" / f"{case.dataset_id}.json"
                ).read_text(encoding="utf-8")
            )
            case_payload = next(
                item for item in payload["cases"] if item["case_id"] == case.case_id
            )
            tasks.append(
                EvalTask(
                    model,
                    "knowledge_control",
                    "direct",
                    0,
                    case,
                    case_payload["control_question"],
                )
            )

    existing = load_generation_map(generation_path)
    pending = [task for task in tasks if task.key not in existing]
    print(
        f"Generation: {len(pending)} pending / {len(tasks)} total",
        flush=True,
    )

    def generation_worker(task: EvalTask) -> dict[str, Any]:
        return generation_request(
            api_key,
            task,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )

    generation_failures = run_concurrent(
        pending,
        generation_worker,
        concurrency=args.concurrency,
        progress_every=args.progress_every,
        on_success=lambda row: append_jsonl(generation_path, row),
    )
    rows = list(load_generation_map(generation_path).values())
    current_cost = sum(row["cost_usd"] for row in rows)
    if current_cost > args.max_cost:
        raise RuntimeError(
            f"Generation cost ${current_cost:.2f} exceeded cap ${args.max_cost:.2f}"
        )

    semantic_rows = [
        row
        for row in rows
        if row["kind"] == "knowledge_control"
        or (
            row["kind"] == "target"
            and row["scoring"] == "premise_rejection"
        )
    ]
    existing_judgments = load_judgment_map(judgment_path)
    judge_pending = [
        row
        for row in semantic_rows
        if row["task_key"] not in existing_judgments
        or existing_judgments[row["task_key"]].get("judge_version") != JUDGE_VERSION
        or existing_judgments[row["task_key"]].get("response_sha256")
        != row["response_sha256"]
    ]
    print(
        f"Judging: {len(judge_pending)} pending / {len(semantic_rows)} total",
        flush=True,
    )

    def judge_worker(row: dict[str, Any]) -> dict[str, Any]:
        return judge_request(
            api_key,
            args.judge_model,
            row,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )

    judge_failures = run_concurrent(
        judge_pending,
        judge_worker,
        concurrency=args.concurrency,
        progress_every=args.progress_every,
        on_success=lambda row: append_jsonl(judgment_path, row),
    )
    judgments = load_judgment_map(judgment_path)

    expected_generation = len(tasks)
    expected_judgments = len(semantic_rows)
    if len(rows) != expected_generation or len(judgments) != expected_judgments:
        write_json(
            args.output_dir / "failures.json",
            {
                "generation": [
                    {"task": asdict(task), "error": error}
                    for task, error in generation_failures
                ],
                "judging": [
                    {"task_key": row["task_key"], "error": error}
                    for row, error in judge_failures
                ],
            },
        )
        raise RuntimeError(
            f"Incomplete run: generations {len(rows)}/{expected_generation}, "
            f"judgments {len(judgments)}/{expected_judgments}. Re-run to resume."
        )

    summary, paired = aggregate(rows, judgments, args.model)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "paired.csv", paired)
    report = build_report(manifest, summary, paired)
    (args.output_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    manifest.update(
        {
            "updated_at": utc_now(),
            "generation_requests": len(rows),
            "judge_requests": len(judgments),
            "generation_cost_usd": sum(row["cost_usd"] for row in rows),
            "judge_cost_usd": sum(row["cost_usd"] for row in judgments.values()),
            "actual_cost_usd": sum(row["cost_usd"] for row in rows)
            + sum(row["cost_usd"] for row in judgments.values()),
            "generation_failures_last_attempt": len(generation_failures),
            "judge_failures_last_attempt": len(judge_failures),
            "complete": True,
        }
    )
    write_json(manifest_path, manifest)
    print(report, flush=True)
    print(f"\nComplete. Artifacts: {args.output_dir}", flush=True)
    return args.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-cost", type=float, default=15.0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--limit-per-task-type", type=int)
    parser.add_argument("--judge-model", default=DEFAULT_SEMANTIC_JUDGE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    args.model = args.model or list(DEFAULT_MODELS)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.max_tokens < 2048:
        parser.error("--max-tokens must be at least 2048")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.max_cost <= 0:
        parser.error("--max-cost must be positive")
    if args.limit_per_task_type is not None and args.limit_per_task_type < 1:
        parser.error("--limit-per-task-type must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
