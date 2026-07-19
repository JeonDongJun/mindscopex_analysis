"""Cross-model analysis of ``00`` CRT text-response runs.

Each ``crt_text_responses`` run writes a per-model ``summary.json`` /
``family_summary.json``. This script merges several runs (e.g. the four
``00_hagendorff_full_*`` models) into one comparison: headline accuracy / lure
rate per model x reasoning mode, the thinking-vs-non-thinking effect, per-family
difficulty, and data-quality flags. Stdlib only, so it runs locally on the
downloaded artifacts without loading any model.

Usage:
    uv run python scripts/analyze_crt_responses.py results/runs --output results/analysis
    uv run python scripts/analyze_crt_responses.py run_a/ run_b/ ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

COUNT_FIELDS = (
    "total",
    "correct",
    "incorrect",
    "lure",
    "hallucination",
    "both",
    "other",
    "format_failures",
    "protocol_failures",
    "retried_responses",
    "retry_attempts",
)

# Rough capability order for Qwen3.5 analysis profiles; unknown sizes sort last.
_SIZE_ORDER = {"0.8B": 0, "2B": 1, "9B": 2, "27B": 3, "35B-A3B": 4, "35B": 4}


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
    radius /= denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _size_key(model: str) -> tuple[int, str]:
    match = re.search(r"(\d+(?:\.\d+)?B(?:-A\d+B)?)", model)
    token = match.group(1) if match else ""
    return (_SIZE_ORDER.get(token, 99), model)


def find_summary_files(paths: list[Path]) -> list[Path]:
    """Locate ``summary.json`` files under the given files/dirs (recursive)."""

    found: list[Path] = []
    for path in paths:
        if path.is_file() and path.name == "summary.json":
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(path.rglob("summary.json")))
    # de-dup while preserving order
    seen: set[Path] = set()
    unique = []
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Sum count fields by ``keys`` across runs and recompute rates + Wilson CIs."""

    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in keys)
        entry = grouped.setdefault(
            key,
            {**{field: row.get(field, "") for field in keys}, **{c: 0 for c in COUNT_FIELDS}},
        )
        for field in COUNT_FIELDS:
            entry[field] += int(row.get(field, 0) or 0)

    for entry in grouped.values():
        total = entry["total"]
        entry["accuracy"] = entry["correct"] / total if total else 0.0
        entry["lure_rate"] = entry["lure"] / total if total else 0.0
        entry["hallucination_rate"] = entry["hallucination"] / total if total else 0.0
        entry["accuracy_ci_low"], entry["accuracy_ci_high"] = wilson_interval(
            entry["correct"], total
        )
        entry["lure_rate_ci_low"], entry["lure_rate_ci_high"] = wilson_interval(
            entry["lure"], total
        )
    return list(grouped.values())


def thinking_effect(headline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-model accuracy delta of thinking vs non_thinking mode."""

    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for row in headline:
        by_model.setdefault(row["model"], {})[row["mode"]] = row
    rows: list[dict[str, Any]] = []
    for model, modes in by_model.items():
        think = modes.get("thinking")
        plain = modes.get("non_thinking")
        rows.append(
            {
                "model": model,
                "acc_non_thinking": plain["accuracy"] if plain else None,
                "acc_thinking": think["accuracy"] if think else None,
                "acc_delta": (think["accuracy"] - plain["accuracy"] if think and plain else None),
            }
        )
    rows.sort(key=lambda row: _size_key(row["model"]))
    return rows


def observations(headline: list[dict[str, Any]], family: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    if not headline:
        return ["No summary rows found."]

    best = max(headline, key=lambda row: row["accuracy"])
    worst = min(headline, key=lambda row: row["accuracy"])
    notes.append(
        f"Highest accuracy: {best['model']} / {best['mode']} "
        f"({best['accuracy']:.1%}, N={best['total']}). "
        f"Lowest: {worst['model']} / {worst['mode']} ({worst['accuracy']:.1%})."
    )

    deltas = [row["acc_delta"] for row in thinking_effect(headline) if row["acc_delta"] is not None]
    if deltas:
        mean_delta = sum(deltas) / len(deltas)
        positive = sum(1 for d in deltas if d > 0)
        direction = "helps" if mean_delta > 0 else "does not help"
        notes.append(
            f"Thinking mode {direction} on average: mean accuracy delta "
            f"{mean_delta:+.1%} across {len(deltas)} models ({positive} positive)."
        )

    if family:
        pooled: dict[str, dict[str, int]] = {}
        for row in family:
            acc = pooled.setdefault(row["family"], {"lure": 0, "total": 0})
            acc["lure"] += int(row.get("lure", 0) or 0)
            acc["total"] += int(row.get("total", 0) or 0)
        rates = {
            fam: counts["lure"] / counts["total"]
            for fam, counts in pooled.items()
            if counts["total"]
        }
        if rates:
            hardest = max(rates, key=rates.get)
            easiest = min(rates, key=rates.get)
            notes.append(
                f"Most lure-prone family (pooled): {hardest} ({rates[hardest]:.1%}); "
                f"least: {easiest} ({rates[easiest]:.1%})."
            )

    flagged = [
        f"{row['model']}/{row['mode']}"
        for row in headline
        if row["format_failures"] or row["protocol_failures"]
    ]
    if flagged:
        notes.append(
            "Data-quality caveat — format/protocol failures present in: "
            + ", ".join(flagged)
            + " (inspect summary.md before trusting hallucination counts)."
        )
    return notes


def _fmt_pct(value: Any) -> str:
    return "-" if value is None else f"{value:.1%}"


def build_report(
    headline: list[dict[str, Any]],
    family: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> str:
    headline = sorted(headline, key=lambda row: (_size_key(row["model"]), row["mode"]))
    family = sorted(family, key=lambda row: (_size_key(row["model"]), row["mode"], row["family"]))
    lines = ["# CRT text-response cross-model analysis", ""]

    lines.append("## Sources")
    lines.append("")
    for source in sources:
        commit = (source.get("git_commit") or "")[:8]
        lines.append(
            f"- `{source['run']}` — models {source.get('model_ids', '?')}, "
            f"dataset `{source.get('dataset', '?')}`, seeds {source.get('seeds', '?')}"
            + (f", commit `{commit}`" if commit else "")
        )
    lines.append("")

    lines.append("## Observations")
    lines.append("")
    lines.extend(f"- {note}" for note in observations(headline, family))
    lines.append("")

    lines.append("## Headline (model x mode)")
    lines.append("")
    lines.append("| Model | Mode | N | Accuracy [95% CI] | Lure | Hallucination | Retried |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in headline:
        acc_ci = f"[{row['accuracy_ci_low']:.1%}, {row['accuracy_ci_high']:.1%}]"
        lines.append(
            f"| {row['model']} | {row['mode']} | {row['total']} | "
            f"{row['accuracy']:.1%} {acc_ci} | "
            f"{row['lure_rate']:.1%} | {row['hallucination_rate']:.1%} | "
            f"{row['retried_responses']} |"
        )
    lines.append("")

    lines.append("## Thinking effect (accuracy)")
    lines.append("")
    lines.append("| Model | non_thinking | thinking | Δ (think − non) |")
    lines.append("|---|---:|---:|---:|")
    for row in thinking_effect(headline):
        delta = row["acc_delta"]
        delta_str = "-" if delta is None else f"{delta:+.1%}"
        lines.append(
            f"| {row['model']} | {_fmt_pct(row['acc_non_thinking'])} | "
            f"{_fmt_pct(row['acc_thinking'])} | {delta_str} |"
        )
    lines.append("")

    if family:
        lines.append("## By CRT family")
        lines.append("")
        lines.append("| Model | Mode | Family | N | Accuracy | Lure rate [95% CI] |")
        lines.append("|---|---|---|---:|---:|---:|")
        for row in family:
            lines.append(
                f"| {row['model']} | {row['mode']} | {row['family']} | {row['total']} | "
                f"{row['accuracy']:.1%} | {row['lure_rate']:.1%} "
                f"[{row['lure_rate_ci_low']:.1%}, {row['lure_rate_ci_high']:.1%}] |"
            )
        lines.append("")

    return "\n".join(lines)


def analyze(paths: list[Path]) -> dict[str, Any]:
    summary_files = find_summary_files(paths)
    headline_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for summary_path in summary_files:
        run_dir = summary_path.parent
        headline_rows.extend(_load_json(summary_path))
        family_path = run_dir / "family_summary.json"
        if family_path.is_file():
            family_rows.extend(_load_json(family_path))
        manifest_path = run_dir / "manifest.json"
        manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
        sources.append(
            {
                "run": manifest.get("run_name", run_dir.name),
                "model_ids": manifest.get("model_ids"),
                "dataset": manifest.get("dataset"),
                "seeds": manifest.get("seeds"),
                "git_commit": manifest.get("git_commit"),
            }
        )
    return {
        "n_runs": len(summary_files),
        "headline": merge_rows(headline_rows, ("model", "mode")),
        "family": merge_rows(family_rows, ("model", "mode", "family")),
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+", type=Path, help="Run dirs or a parent (e.g. results/runs)"
    )
    parser.add_argument("--output", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()

    result = analyze(args.paths)
    if result["n_runs"] == 0:
        raise SystemExit(
            "No summary.json found. Run 00 first, e.g.\n"
            "  ./experiments/run_colab.sh experiments/suites/full_all.toml "
            "-s mindscopex --gpu A100\n"
            "then point this script at results/runs."
        )

    report = build_report(result["headline"], result["family"], result["sources"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "crt_analysis.md").write_text(report, encoding="utf-8")

    columns = [
        "model",
        "mode",
        "total",
        "accuracy",
        "accuracy_ci_low",
        "accuracy_ci_high",
        "lure_rate",
        "hallucination_rate",
        "format_failures",
        "protocol_failures",
        "retried_responses",
    ]
    with (args.output / "crt_analysis.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(result["headline"], key=lambda r: (_size_key(r["model"]), r["mode"])):
            writer.writerow(row)

    print(report)
    print(f"\nWrote {args.output / 'crt_analysis.md'} and {args.output / 'crt_analysis.csv'}")


if __name__ == "__main__":
    main()
