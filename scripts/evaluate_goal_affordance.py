"""Evaluate goal-affordance traps with exact, counterbalanced binary choices.

The evaluator compares the weakest available reasoning effort ("direct") with
high effort ("deliberate") for current GPT, Claude, and Gemini endpoints. It
uses strict JSON output, keeps option order fixed within a case across modes,
and writes resume-safe JSONL plus aggregate and pair-level reports.

The API key is loaded from ``OPENROUTER_API_KEY`` or the repository ``.env``.
It is never written to an artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from openrouter_common import (
    append_jsonl,
    extract_content,
    http_json,
    load_dotenv,
    read_json,
    utc_now,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_INPUT = ROOT / "results" / "goal_affordance_development" / "seed_v0.json"
DEFAULT_MODELS = (
    "openai/gpt-5.6-sol",
    "anthropic/claude-opus-5",
    "google/gemini-3-flash-preview",
)
MODES = ("direct", "deliberate")
MODE_CHOICES = (*MODES, "intuitive_prompted", "deliberate_prompted")
CONDITIONS = ("hostile", "explicit", "neutral", "counterfactual")
RETRYABLE_HTTP_CODES = {408, 409, 429, 500, 502, 503, 504}
SYSTEM_PROMPT = (
    "Answer the user's question by choosing exactly one of the provided options. "
    "Return only the JSON object required by the response schema."
)
INTUITIVE_SYSTEM_PROMPT = (
    "Choose immediately from your first impression. Do not analyze hidden requirements "
    "or reconsider the options. Return only the JSON object required by the response schema."
)
DELIBERATE_SYSTEM_PROMPT = (
    "Before choosing, carefully reread the stated goal and check which option can "
    "actually accomplish it. Explicitly account for any object that must move, "
    "required resource, eligible agent, or prerequisite state. Reason internally, "
    "then return only the JSON object required by the response schema."
)
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "goal_affordance_choice",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"choice": {"type": "string", "enum": ["A", "B"]}},
            "required": ["choice"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class Case:
    case_id: str
    pair_id: str
    semantic_id: str
    language: str
    family: str
    condition: str
    question: str
    correct_answer: str
    lure_answer: str


@dataclass(frozen=True)
class Task:
    model: str
    mode: str
    effort: str
    case: Case

    @property
    def key(self) -> str:
        return f"{self.model}\t{self.mode}\t{self.case.case_id}"


def load_cases(
    path: Path,
    conditions: set[str] | None,
    pair_ids: set[str] | None = None,
) -> tuple[dict[str, Any], list[Case]]:
    payload = read_json(path)
    cases = [
        Case(
            case_id=str(row["case_id"]),
            pair_id=str(row["pair_id"]),
            semantic_id=str(row.get("semantic_id", row["pair_id"])),
            language=str(row.get("language", "en")),
            family=str(row["family"]),
            condition=str(row["condition"]),
            question=str(row["question"]).strip(),
            correct_answer=str(row["correct_answer"]).strip(),
            lure_answer=str(row["lure_answer"]).strip(),
        )
        for row in payload["cases"]
        if (conditions is None or row["condition"] in conditions)
        and (pair_ids is None or row["pair_id"] in pair_ids)
    ]
    if not cases:
        raise ValueError("No cases remain after condition filtering")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case IDs")
    for case in cases:
        if case.condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {case.condition}")
        if not case.correct_answer or not case.lure_answer:
            raise ValueError(f"{case.case_id} has an empty answer")
        if case.correct_answer == case.lure_answer:
            raise ValueError(f"{case.case_id} has identical answer choices")
    return payload, cases


def fetch_model_metadata(model_ids: Iterable[str]) -> list[dict[str, Any]]:
    catalog = http_json(MODELS_URL).get("data", [])
    by_id = {row["id"]: row for row in catalog}
    missing = sorted(set(model_ids) - set(by_id))
    if missing:
        raise ValueError(f"Models missing from OpenRouter catalog: {missing}")
    selected = []
    for model_id in model_ids:
        row = by_id[model_id]
        parameters = set(row.get("supported_parameters") or [])
        if not {"reasoning", "response_format"}.issubset(parameters):
            raise ValueError(f"{model_id} lacks reasoning or response_format support")
        reasoning = row.get("reasoning") or {}
        efforts = list(reasoning.get("supported_efforts") or [])
        if not efforts:
            raise ValueError(f"{model_id} does not publish supported reasoning efforts")
        selected.append(
            {
                "id": model_id,
                "name": row.get("name"),
                "canonical_slug": row.get("canonical_slug"),
                "reasoning": reasoning,
                "pricing": row.get("pricing") or {},
                "supported_parameters": row.get("supported_parameters") or [],
            }
        )
    return selected


def effort_for_mode(model_row: dict[str, Any], mode: str) -> str:
    efforts = model_row["reasoning"]["supported_efforts"]
    if mode == "direct":
        for candidate in ("none", "minimal", "low"):
            if candidate in efforts:
                return candidate
    elif mode == "intuitive_prompted":
        for candidate in ("none", "minimal", "low"):
            if candidate in efforts:
                return candidate
    elif mode in {"deliberate", "deliberate_prompted"} and "high" in efforts:
        return "high"
    raise ValueError(f"No valid {mode} effort for {model_row['id']}: {efforts}")


def correct_is_a(case: Case) -> bool:
    digest = hashlib.sha256(f"goal-affordance-order-v1|{case.case_id}".encode()).digest()
    return digest[0] % 2 == 0


def presented_options(case: Case, *, reverse: bool = False) -> tuple[str, str, str]:
    correct_at_a = correct_is_a(case) != reverse
    if correct_at_a:
        return case.correct_answer, case.lure_answer, "A"
    return case.lure_answer, case.correct_answer, "B"


def parse_choice(content: str) -> str:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON response: {content[:300]}") from exc
    choice = parsed.get("choice") if isinstance(parsed, dict) else None
    if choice not in {"A", "B"}:
        raise ValueError(f"Invalid choice response: {content[:300]}")
    return str(choice)


def estimated_cost(usage: dict[str, Any], pricing: dict[str, Any]) -> float:
    if usage.get("cost") is not None:
        return float(usage["cost"])
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return prompt * float(pricing.get("prompt") or 0) + completion * float(
        pricing.get("completion") or 0
    )


def request_task(
    api_key: str,
    task: Task,
    pricing: dict[str, Any],
    *,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    reverse_options: bool,
) -> dict[str, Any]:
    option_a, option_b, correct_choice = presented_options(
        task.case, reverse=reverse_options
    )
    user_prompt = (
        f"{task.case.question}\n\n"
        f"Option A: {option_a}\n"
        f"Option B: {option_b}\n"
        "Choose exactly one option."
    )
    payload = {
        "model": task.model,
        "messages": [
            {
                "role": "system",
                "content": {
                    "intuitive_prompted": INTUITIVE_SYSTEM_PROMPT,
                    "deliberate_prompted": DELIBERATE_SYSTEM_PROMPT,
                }.get(task.mode, SYSTEM_PROMPT),
            },
            {"role": "user", "content": user_prompt},
        ],
        "reasoning": {"effort": task.effort, "exclude": True},
        "response_format": SCHEMA,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "MindScopeX goal-affordance evaluation",
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
            api_choice = response["choices"][0]
            message = api_choice["message"]
            content = extract_content(message)
            selected_choice = parse_choice(content)
            usage = response.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            elapsed = time.perf_counter() - started
            return {
                "timestamp": utc_now(),
                "model": task.model,
                "mode": task.mode,
                "effort": task.effort,
                "case_id": task.case.case_id,
                "pair_id": task.case.pair_id,
                "semantic_id": task.case.semantic_id,
                "language": task.case.language,
                "family": task.case.family,
                "condition": task.case.condition,
                "question": task.case.question,
                "option_a": option_a,
                "option_b": option_b,
                "correct_choice": correct_choice,
                "selected_choice": selected_choice,
                "selected_answer": option_a if selected_choice == "A" else option_b,
                "label": "correct" if selected_choice == correct_choice else "lure",
                "raw_response": content,
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
                "cost_usd": estimated_cost(usage, pricing),
                "provider": response.get("provider") or "",
                "finish_reason": api_choice.get("finish_reason") or "",
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
        except (TimeoutError, urllib.error.URLError, RuntimeError, ValueError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(str(exc)) from exc
            delay = 2**attempt
        time.sleep(delay + random.random() * 0.25)
    raise AssertionError("unreachable")


def load_rows(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.is_file():
        return [], set()
    rows = []
    keys = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL at line {line_number}") from exc
        key = f"{row['model']}\t{row['mode']}\t{row['case_id']}"
        if key not in keys:
            keys.add(key)
            rows.append(row)
    return rows, keys


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["mode"], row["condition"])].append(row)
    summary = []
    for (model, mode, condition), group in sorted(groups.items()):
        counts = Counter(row["label"] for row in group)
        total = len(group)
        summary.append(
            {
                "model": model,
                "mode": mode,
                "effort": group[0]["effort"],
                "condition": condition,
                "n": total,
                "correct": counts["correct"],
                "lure": counts["lure"],
                "lure_rate": counts["lure"] / total,
                "reasoning_tokens": sum(row["reasoning_tokens"] for row in group),
                "cost_usd": sum(row["cost_usd"] for row in group),
            }
        )
    return summary


def pair_analysis(
    rows: list[dict[str, Any]], expected_models: list[str]
) -> list[dict[str, Any]]:
    lookup = {
        (row["pair_id"], row["condition"], row["model"], row["mode"]): row["label"]
        for row in rows
    }
    pair_meta = {}
    for row in rows:
        pair_meta[row["pair_id"]] = {"family": row["family"]}
    output = []
    controls = ("explicit", "neutral", "counterfactual")
    for pair_id, meta in sorted(pair_meta.items()):
        hostile_direct_lures = sum(
            lookup.get((pair_id, "hostile", model, "direct")) == "lure"
            for model in expected_models
        )
        hostile_deliberate_lures = sum(
            lookup.get((pair_id, "hostile", model, "deliberate")) == "lure"
            for model in expected_models
        )
        control_total = len(expected_models) * len(controls)
        control_correct = sum(
            lookup.get((pair_id, condition, model, "direct")) == "correct"
            for model in expected_models
            for condition in controls
        )
        complete = all(
            (pair_id, condition, model, mode) in lookup
            for condition in CONDITIONS
            for model in expected_models
            for mode in MODES
        )
        output.append(
            {
                "pair_id": pair_id,
                "family": meta["family"],
                "complete": complete,
                "hostile_direct_lure_models": hostile_direct_lures,
                "hostile_deliberate_lure_models": hostile_deliberate_lures,
                "direct_control_correct": control_correct,
                "direct_control_total": control_total,
                "deliberate_recovery": hostile_direct_lures - hostile_deliberate_lures,
                "challenge": bool(
                    complete
                    and hostile_direct_lures >= 2
                    and control_correct == control_total
                    and hostile_deliberate_lures < hostile_direct_lures
                ),
            }
        )
    return output


def write_summary_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    fields = list(summary[0]) if summary else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def build_report(
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> str:
    lines = [
        "# Goal-affordance frontier evaluation",
        "",
        f"- Successful responses: {len(rows)}",
        f"- Errors: {len(errors)}",
        f"- API cost: ${sum(row['cost_usd'] for row in rows):.4f}",
        "",
        "## Condition results",
        "",
        "| Model | Mode (effort) | Condition | n | Lure | Lure rate |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['mode']} ({row['effort']}) | "
            f"{row['condition']} | {row['n']} | {row['lure']} | "
            f"{row['lure_rate']:.1%} |"
        )
    complete_pairs = [row for row in pairs if row["complete"]]
    challenges = [row for row in complete_pairs if row["challenge"]]
    lines.extend(
        [
            "",
            "## Pair-level validation",
            "",
            f"- Complete four-condition/two-mode pairs: {len(complete_pairs)}",
            f"- Strict challenge pairs: {len(challenges)}",
            (
                "- Strict criterion: hostile lure for at least 2/3 direct models, all "
                "direct controls correct, and fewer hostile lures under deliberate mode."
            ),
        ]
    )
    if challenges:
        lines.extend(
            [
                "",
                "| Pair | Family | Direct hostile lures | Deliberate hostile lures |",
                "|---|---|---:|---:|",
            ]
        )
        for row in challenges:
            lines.append(
                f"| {row['pair_id']} | {row['family']} | "
                f"{row['hostile_direct_lure_models']} | "
                f"{row['hostile_deliberate_lure_models']} |"
            )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Path:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    conditions = set(args.condition) if args.condition else None
    pair_ids = set(args.pair) if args.pair else None
    source, cases = load_cases(args.input, conditions, pair_ids)
    model_meta = fetch_model_metadata(args.model)
    metadata_by_id = {row["id"]: row for row in model_meta}
    modes = args.mode or list(MODES)
    tasks = [
        Task(
            model=model,
            mode=mode,
            effort=effort_for_mode(metadata_by_id[model], mode),
            case=case,
        )
        for model in args.model
        for mode in modes
        for case in cases
    ]
    if args.limit:
        tasks = tasks[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    response_path = args.output_dir / "responses.jsonl"
    error_path = args.output_dir / "errors.jsonl"
    rows, completed = load_rows(response_path)
    errors, _ = load_rows(error_path)
    pending = [task for task in tasks if task.key not in completed]
    input_sha256 = hashlib.sha256(args.input.read_bytes()).hexdigest()
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "created_or_resumed_at": utc_now(),
        "input": str(args.input.resolve()),
        "input_sha256": input_sha256,
        "source_dataset_id": source.get("dataset_id"),
        "models": model_meta,
        "modes": {
            model: {mode: effort_for_mode(metadata_by_id[model], mode) for mode in modes}
            for model in args.model
        },
        "conditions": sorted(conditions) if conditions else list(CONDITIONS),
        "expected_responses": len(tasks),
        "option_order": (
            "reverse of sha256(goal-affordance-order-v1|case_id)"
            if args.reverse_options
            else "sha256(goal-affordance-order-v1|case_id); fixed across models/modes"
        ),
        "system_prompts": {
            "direct_and_deliberate": SYSTEM_PROMPT,
            "intuitive_prompted": INTUITIVE_SYSTEM_PROMPT,
            "deliberate_prompted": DELIBERATE_SYSTEM_PROMPT,
        },
        "response_schema": SCHEMA,
    }
    if manifest_path.is_file():
        previous = read_json(manifest_path)
        for field in ("input_sha256", "models", "modes", "conditions", "expected_responses"):
            if previous.get(field) != manifest.get(field):
                raise RuntimeError(
                    f"Resume manifest mismatch for {field}; use a new output directory"
                )
    write_json(manifest_path, manifest)

    print(
        f"Cases {len(cases)} | tasks {len(tasks)} | pending {len(pending)} | "
        f"models {len(args.model)}",
        flush=True,
    )
    for model in args.model:
        mapping = ", ".join(
            f"{mode}={effort_for_mode(metadata_by_id[model], mode)}" for mode in modes
        )
        print(f"{model}: {mapping}", flush=True)

    spent = sum(row["cost_usd"] for row in rows)
    futures: dict[Future[dict[str, Any]], Task] = {}
    task_iter = iter(pending)
    stop_submitting = False
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:

        def submit_one() -> bool:
            nonlocal stop_submitting
            if stop_submitting or spent >= args.max_cost:
                stop_submitting = True
                return False
            try:
                task = next(task_iter)
            except StopIteration:
                return False
            future = executor.submit(
                request_task,
                api_key,
                task,
                metadata_by_id[task.model]["pricing"],
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
                reverse_options=args.reverse_options,
            )
            futures[future] = task
            return True

        for _ in range(args.concurrency):
            if not submit_one():
                break
        processed = 0
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                task = futures.pop(future)
                processed += 1
                try:
                    row = future.result()
                    rows.append(row)
                    completed.add(task.key)
                    spent += row["cost_usd"]
                    append_jsonl(response_path, row)
                except Exception as exc:  # noqa: BLE001 - preserve task-level failures
                    error = {
                        "timestamp": utc_now(),
                        "model": task.model,
                        "mode": task.mode,
                        "case_id": task.case.case_id,
                        "error": str(exc),
                    }
                    errors.append(error)
                    append_jsonl(error_path, error)
                if processed % args.progress_every == 0 or not futures:
                    print(
                        f"processed {processed}/{len(pending)} | "
                        f"success {len(rows)}/{len(tasks)} | cost ${spent:.4f}",
                        flush=True,
                    )
                submit_one()

    summary = summarize_rows(rows)
    pairs = pair_analysis(rows, args.model)
    write_json(args.output_dir / "summary.json", summary)
    write_summary_csv(args.output_dir / "summary.csv", summary)
    write_json(args.output_dir / "pair_analysis.json", pairs)
    write_json(args.output_dir / "errors.json", errors)
    report = build_report(rows, summary, pairs, errors)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    manifest["completed_responses"] = len(rows)
    manifest["complete"] = len(rows) == len(tasks)
    manifest["total_cost_usd"] = spent
    manifest["finished_at"] = utc_now()
    write_json(manifest_path, manifest)
    print(report, flush=True)
    return args.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--mode", action="append", choices=MODE_CHOICES, default=None)
    parser.add_argument("--condition", action="append", choices=CONDITIONS, default=None)
    parser.add_argument("--pair", action="append", default=None)
    parser.add_argument(
        "--reverse-options",
        action="store_true",
        help="Flip the deterministic A/B order for every selected case.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-cost", type=float, default=20.0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results"
        / f"goal_affordance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    args = parser.parse_args()
    args.model = args.model or list(DEFAULT_MODELS)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.max_tokens < 256:
        parser.error("--max-tokens must be at least 256")
    if args.max_cost <= 0:
        parser.error("--max-cost must be positive")
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Interrupted; rerun with the same --output-dir to resume.", file=sys.stderr)
        raise
