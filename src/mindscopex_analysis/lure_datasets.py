"""Uniform loader for the normalized reasoning-lure datasets in ``data/``.

Every dataset the project uses is stored as one self-describing JSON file
under ``mindscopex_analysis/data`` (see ``scripts/build_datasets.py`` for how
they are generated). This module loads any of them into the common
:class:`~mindscopex_analysis.cases.LureCase` format so notebooks and workflows
can treat every source the same way.

    from mindscopex_analysis import load_lure_dataset, available_lure_datasets

    available_lure_datasets()                 # ['crt_pilot', 'hagendorff_crt', ...]
    cases = load_lure_dataset("hagendorff_crt")

Datasets scored by ``logprob_margin`` (the CRT sets) carry short, distinct
``correct``/``lure`` answers ready for the teacher-forced margin scorer.
Datasets scored by ``premise_rejection`` (the semantic-illusion set) have empty
answer strings and keep the authoritative correction in the case ``note``; they
are meant for a free-form accept-vs-reject judge, not the margin scorer.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Any

from mindscopex_analysis.cases import LureCase

_DATA_PACKAGE = "mindscopex_analysis"
_DATA_SUBDIR = "data"

_VALID_SCORING = {"logprob_margin", "premise_rejection"}


@dataclass(frozen=True)
class LureDatasetInfo:
    """Metadata summary for one normalized lure dataset (no cases)."""

    dataset_id: str
    title: str
    description: str
    task_kind: str
    scoring: str
    n_cases: int
    family_counts: dict[str, int]
    source: dict[str, Any]


def _data_dir():
    return files(_DATA_PACKAGE).joinpath(_DATA_SUBDIR)


def available_lure_datasets() -> list[str]:
    """Return the sorted loader keys (JSON file stems) available in ``data/``."""

    names: list[str] = []
    for entry in _data_dir().iterdir():
        name = entry.name
        if name.endswith(".json"):
            names.append(name[: -len(".json")])
    return sorted(names)


@cache
def _read_raw(dataset_id: str) -> dict[str, Any]:
    resource = _data_dir().joinpath(f"{dataset_id}.json")
    if not resource.is_file():
        available = ", ".join(available_lure_datasets())
        raise FileNotFoundError(f"Unknown lure dataset {dataset_id!r}. Available: {available}")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError(f"{dataset_id!r} is not a normalized lure dataset (no 'cases' list)")
    return payload


def _scoring(payload: dict[str, Any]) -> str:
    scoring = payload.get("scoring", "logprob_margin")
    if scoring not in _VALID_SCORING:
        raise ValueError(f"dataset {payload.get('dataset_id')!r} has unknown scoring {scoring!r}")
    return scoring


def lure_dataset_info(dataset_id: str) -> LureDatasetInfo:
    """Return the metadata summary for one dataset."""

    payload = _read_raw(dataset_id)
    cases = payload["cases"]
    family_counts = payload.get("family_counts")
    if not isinstance(family_counts, dict):
        family_counts = {}
        for case in cases:
            family = case.get("family", "")
            family_counts[family] = family_counts.get(family, 0) + 1
        family_counts = dict(sorted(family_counts.items()))
    return LureDatasetInfo(
        dataset_id=dataset_id,
        title=payload.get("title", dataset_id),
        description=payload.get("description", ""),
        task_kind=payload.get("task_kind", "unknown"),
        scoring=_scoring(payload),
        n_cases=len(cases),
        family_counts=family_counts,
        source=payload.get("source", {}),
    )


def lure_dataset_catalog() -> list[LureDatasetInfo]:
    """Return metadata for every available dataset (handy for docs/overviews)."""

    return [lure_dataset_info(name) for name in available_lure_datasets()]


def _answer_prompt(text: str) -> str:
    return text.strip() + "\nAnswer:"


def _case_from_row(row: dict[str, Any], *, scoring: str, index: int) -> LureCase:
    case_id = row.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"case #{index} is missing a string case_id")
    for field in ("family", "question"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"case {case_id!r} has empty {field!r}")

    correct = str(row.get("correct_answer", "")).strip()
    lure = str(row.get("lure_answer", "")).strip()
    if scoring == "logprob_margin":
        if not correct or not lure:
            raise ValueError(f"case {case_id!r} needs non-empty correct/lure for logprob_margin")
        if correct.casefold() == lure.casefold():
            raise ValueError(f"case {case_id!r} has identical correct and lure answers")

    control = str(row.get("control_question", "")).strip()
    reference = str(row.get("reference_answer", "")).strip()
    note = str(row.get("note", "")).strip()
    if reference:
        note = f"{note} reference_answer: {reference}".strip()

    return LureCase(
        case_id=case_id,
        family=row["family"].strip(),
        prompt=_answer_prompt(row["question"]),
        correct_answer=(" " + correct) if correct else "",
        lure_answer=(" " + lure) if lure else "",
        control_prompt=_answer_prompt(control) if control else "",
        note=note,
    )


def load_lure_dataset(dataset_id: str) -> list[LureCase]:
    """Load one normalized dataset as a list of :class:`LureCase`.

    Answers gain the leading space and each prompt the ``\\nAnswer:`` delimiter
    that the logprob scorer expects. Premise-rejection datasets keep empty
    answer strings and fold their free-form correction into the case note.
    """

    payload = _read_raw(dataset_id)
    scoring = _scoring(payload)
    cases: list[LureCase] = []
    seen: set[str] = set()
    for index, row in enumerate(payload["cases"], start=1):
        if not isinstance(row, dict):
            raise TypeError(f"{dataset_id!r} case #{index} is not an object")
        case = _case_from_row(row, scoring=scoring, index=index)
        if case.case_id in seen:
            raise ValueError(f"{dataset_id!r} has duplicate case_id {case.case_id!r}")
        seen.add(case.case_id)
        cases.append(case)
    return cases


def load_all_lure_cases() -> dict[str, list[LureCase]]:
    """Return ``{dataset_id: cases}`` for every available dataset."""

    return {name: load_lure_dataset(name) for name in available_lure_datasets()}


def lure_dataset_cases(
    dataset_id: str,
    *,
    families: Sequence[str] | None = None,
    limit_per_family: int | None = None,
) -> list[LureCase]:
    """Load a dataset, optionally restricting families and capping items per family.

    A convenience over :func:`load_lure_dataset` for notebooks and experiment
    presets. For example ``lure_dataset_cases("hagendorff_crt", limit_per_family=3)``
    mirrors the old ``nature_smoke`` preset (3 items per CRT type), and
    ``lure_dataset_cases("hagendorff_crt")`` is the full 150-item set with matched
    controls already attached.
    """

    cases = load_lure_dataset(dataset_id)
    if families is not None:
        wanted = list(dict.fromkeys(families))
        present = {case.family for case in cases}
        unknown = [family for family in wanted if family not in present]
        if unknown:
            raise ValueError(
                f"{dataset_id!r} has no families {unknown}; present: {sorted(present)}"
            )
        order = {family: rank for rank, family in enumerate(wanted)}
        cases = sorted(
            (case for case in cases if case.family in order),
            key=lambda case: order[case.family],
        )
    if limit_per_family is not None:
        if limit_per_family < 1:
            raise ValueError("limit_per_family must be positive or None")
        counts: dict[str, int] = {}
        limited: list[LureCase] = []
        for case in cases:
            taken = counts.get(case.family, 0)
            if taken < limit_per_family:
                limited.append(case)
                counts[case.family] = taken + 1
        cases = limited
    return cases
