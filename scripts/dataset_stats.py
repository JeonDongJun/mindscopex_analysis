"""Emit a Markdown catalog (counts, source, license, examples) for every
normalized lure dataset. Used to keep ``docs/datasets.md`` statistics in sync.

    uv run python scripts/dataset_stats.py            # print to stdout
    uv run python scripts/dataset_stats.py --overview # compact table only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mindscopex_analysis.lure_datasets import (  # noqa: E402
    LureDatasetInfo,
    load_lure_dataset,
    lure_dataset_catalog,
)


def _overview_table(catalog: list[LureDatasetInfo]) -> list[str]:
    lines = [
        "| dataset | n | task_kind | scoring | families |",
        "|---------|--:|-----------|---------|----------|",
    ]
    for info in catalog:
        families = ", ".join(f"{k} ({v})" for k, v in info.family_counts.items())
        lines.append(
            f"| `{info.dataset_id}` | {info.n_cases} | {info.task_kind} "
            f"| {info.scoring} | {families} |"
        )
    return lines


def _example_block(dataset_id: str, scoring: str) -> list[str]:
    cases = load_lure_dataset(dataset_id)
    if not cases:
        return []
    case = cases[0]
    question = case.prompt.removesuffix("\nAnswer:")
    lines = ["", "_Example_", "", f"- **prompt**: {question}"]
    if scoring == "logprob_margin":
        correct = case.correct_answer.strip()
        lure = case.lure_answer.strip()
        lines.append(f"- **correct**: `{correct}`  ·  **lure**: `{lure}`")
    if case.control_prompt:
        control = case.control_prompt.removesuffix("\nAnswer:")
        lines.append(f"- **control**: {control}")
    if scoring == "premise_rejection" and case.note:
        lines.append(f"- **note**: {case.note}")
    return lines


def build_markdown(*, overview_only: bool) -> str:
    catalog = lure_dataset_catalog()
    total = sum(info.n_cases for info in catalog)
    out: list[str] = []
    out.append(f"Total: **{len(catalog)} datasets, {total} cases**")
    out.append("")
    out.extend(_overview_table(catalog))
    if overview_only:
        return "\n".join(out) + "\n"

    for info in catalog:
        out.append("")
        out.append(f"### `{info.dataset_id}` — {info.title}")
        out.append("")
        out.append(info.description)
        out.append("")
        out.append(f"- **cases**: {info.n_cases}  ·  **scoring**: {info.scoring}")
        families = ", ".join(f"{k} ({v})" for k, v in info.family_counts.items())
        out.append(f"- **families**: {families}")
        src = info.source
        if src:
            cite = f"{src.get('authors', '')} ({src.get('year', '')}). {src.get('title', '')}."
            out.append(f"- **source**: {cite.strip()}")
            if src.get("venue"):
                out.append(f"- **venue**: {src['venue']}")
            if src.get("doi"):
                out.append(f"- **doi**: https://doi.org/{src['doi']}")
            if src.get("project_url"):
                out.append(f"- **data**: {src['project_url']}")
            if src.get("license"):
                out.append(f"- **license**: {src['license']}")
        out.extend(_example_block(info.dataset_id, info.scoring))
        out.append("")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overview", action="store_true", help="Print only the overview table.")
    args = parser.parse_args(argv)
    sys.stdout.write(build_markdown(overview_only=args.overview))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
