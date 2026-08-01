"""Audit every committed reasoning-lure JSON and emit a compact Markdown report.

The canonical human documentation lives in ``docs/datasets.md``. This script
checks the machine-verifiable facts that tend to drift when a dataset is added:
counts, family counts, scoreable answers, IDs, controls, schema-v2 metadata, and
exact cross-dataset question duplicates.

Usage::

    uv run python scripts/audit_datasets.py
    uv run python scripts/audit_datasets.py --check

``--check`` exits non-zero on integrity errors. Known/documented overlaps and
legacy metadata gaps are warnings, not errors.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "src" / "mindscopex_analysis" / "data"


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _read_datasets() -> list[tuple[str, dict[str, Any]]]:
    datasets: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(DATA_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path.name}: root must be an object")
        datasets.append((path.stem, payload))
    return datasets


def audit() -> dict[str, Any]:
    datasets = _read_datasets()
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    global_ids: dict[str, list[str]] = defaultdict(list)
    global_questions: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for loader_id, payload in datasets:
        cases = payload.get("cases")
        if not isinstance(cases, list):
            errors.append(f"{loader_id}: cases must be a list")
            continue

        embedded_id = str(payload.get("dataset_id", "")).strip()
        if embedded_id != loader_id:
            warnings.append(
                f"{loader_id}: embedded dataset_id is {embedded_id!r} (legacy alias)"
            )

        declared_n = payload.get("n_cases")
        if declared_n is None:
            warnings.append(f"{loader_id}: n_cases metadata is missing")
        elif int(declared_n) != len(cases):
            errors.append(f"{loader_id}: n_cases={declared_n}, actual={len(cases)}")

        actual_families = Counter()
        n_controls = 0
        n_references = 0
        pair_ids: set[str] = set()
        template_ids: set[str] = set()
        scoring = str(payload.get("scoring", "logprob_margin"))

        for index, case in enumerate(cases, start=1):
            if not isinstance(case, dict):
                errors.append(f"{loader_id}: case #{index} is not an object")
                continue
            case_id = str(case.get("case_id", "")).strip()
            family = str(case.get("family", "")).strip()
            question = str(case.get("question", "")).strip()
            if not case_id or not family or not question:
                errors.append(f"{loader_id}: case #{index} lacks case_id/family/question")
                continue

            global_ids[case_id].append(loader_id)
            global_questions[_normalized(question)].append((loader_id, case_id))
            actual_families[family] += 1

            control = str(case.get("control_question", "")).strip()
            if control:
                n_controls += 1
                if _normalized(control) == _normalized(question):
                    errors.append(f"{loader_id}/{case_id}: control duplicates hostile")
            if str(case.get("reference_answer", "")).strip():
                n_references += 1

            pair_id = str(case.get("pair_id", "")).strip()
            template_id = str(case.get("template_id", "")).strip()
            condition = str(case.get("condition", "")).strip()
            if pair_id:
                pair_ids.add(pair_id)
            if template_id:
                template_ids.add(template_id)

            if int(payload.get("schema_version", 1)) >= 2:
                if not pair_id or not template_id or not condition:
                    errors.append(
                        f"{loader_id}/{case_id}: schema v2 requires "
                        "pair_id/template_id/condition"
                    )

            correct = str(case.get("correct_answer", "")).strip()
            lure = str(case.get("lure_answer", "")).strip()
            if scoring in {"binary_choice", "logprob_margin"}:
                if not correct or not lure:
                    errors.append(f"{loader_id}/{case_id}: missing correct/lure")
                elif correct.casefold() == lure.casefold():
                    errors.append(f"{loader_id}/{case_id}: correct equals lure")

        declared_families = payload.get("family_counts")
        if declared_families is None:
            warnings.append(f"{loader_id}: family_counts metadata is missing")
        elif dict(sorted(declared_families.items())) != dict(sorted(actual_families.items())):
            errors.append(
                f"{loader_id}: family_counts mismatch "
                f"(declared={declared_families}, actual={dict(actual_families)})"
            )

        rows.append(
            {
                "loader_id": loader_id,
                "embedded_id": embedded_id,
                "schema": int(payload.get("schema_version", 1)),
                "n": len(cases),
                "scoring": scoring,
                "families": dict(sorted(actual_families.items())),
                "controls": n_controls,
                "references": n_references,
                "pairs": len(pair_ids),
                "templates": len(template_ids),
            }
        )

    for case_id, owners in global_ids.items():
        if len(owners) > 1:
            errors.append(f"global duplicate case_id {case_id!r}: {owners}")

    exact_overlaps = [
        group
        for group in global_questions.values()
        if len({dataset_id for dataset_id, _case_id in group}) > 1
    ]
    for group in exact_overlaps:
        warnings.append(
            "exact cross-dataset question overlap: "
            + ", ".join(f"{dataset_id}/{case_id}" for dataset_id, case_id in group)
        )

    return {
        "datasets": rows,
        "total_cases": sum(row["n"] for row in rows),
        "exact_overlaps": exact_overlaps,
        "errors": errors,
        "warnings": warnings,
    }


def _families_text(families: dict[str, int]) -> str:
    return ", ".join(f"{family} ({count})" for family, count in families.items())


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"Total: **{len(report['datasets'])} datasets, {report['total_cases']} cases**",
        "",
        "| loader ID | schema | n | scoring | controls | pairs | templates | families |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in report["datasets"]:
        lines.append(
            f"| `{row['loader_id']}` | {row['schema']} | {row['n']} | "
            f"{row['scoring']} | {row['controls']} | {row['pairs']} | "
            f"{row['templates']} | {_families_text(row['families'])} |"
        )

    lines.extend(["", f"Errors: **{len(report['errors'])}**"])
    for error in report["errors"]:
        lines.append(f"- ERROR: {error}")
    lines.extend(["", f"Warnings: **{len(report['warnings'])}**"])
    for warning in report["warnings"]:
        lines.append(f"- WARN: {warning}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Exit non-zero on integrity errors.")
    args = parser.parse_args(argv)
    report = audit()
    print(markdown(report), end="")
    return 1 if args.check and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
