"""Public benchmark dataset loaders used by the CRT experiments."""

from __future__ import annotations

import ast
import hashlib
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mindscopex_analysis.cases import LureCase

NatureCRTType = Literal["crt1", "crt2", "crt3"]
NaturePromptStyle = Literal["task_only", "question_answer"]

NATURE_CRT150_SOURCE_URL = "https://osf.io/download/z6kmw/"
NATURE_CRT150_SOURCE_SHA256 = "cdf4617e8dec63546762cbe2b3cae6b6c7f640adfb6002bf5fc226f5871a4125"
NATURE_CRT150_DOI = "https://doi.org/10.1038/s43588-023-00527-x"
NATURE_CRT150_OSF_URL = "https://osf.io/w5vhp/"

_NATURE_TYPE_FAMILY = {
    "crt1": "nature_crt_difference",
    "crt2": "nature_crt_rate",
    "crt3": "nature_crt_growth",
}
_NATURE_ASSIGNMENT_END = {"crt1": "crt2", "crt2": "crt3", "crt3": "si"}


@dataclass(frozen=True)
class NatureCRTItem:
    """One published item from the Hagendorff et al. CRT-150 benchmark."""

    item_id: str
    crt_type: NatureCRTType
    number: int
    task: str
    correct_answer: str
    lure_answer: str

    def as_lure_case(
        self,
        *,
        prompt_style: NaturePromptStyle = "task_only",
    ) -> LureCase:
        """Convert the published item to the common experiment case format."""

        if prompt_style == "task_only":
            prompt = self.task.strip()
        elif prompt_style == "question_answer":
            prompt = f"Question: {self.task.strip()}\nAnswer:"
        else:
            raise ValueError(f"Unknown prompt_style={prompt_style!r}")

        return LureCase(
            case_id=self.item_id,
            family=_NATURE_TYPE_FAMILY[self.crt_type],
            prompt=prompt,
            correct_answer=" " + self.correct_answer.strip(),
            lure_answer=" " + self.lure_answer.strip(),
            note=(
                f"Hagendorff et al. (2023) Nature CRT-150 {self.crt_type.upper()} "
                f"item {self.number}; {NATURE_CRT150_DOI}"
            ),
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_nature_cache_dir() -> Path:
    return Path.home() / ".cache" / "mindscopex_analysis" / "nature_crt150"


def download_nature_crt150_source(
    cache_dir: str | Path | None = None,
    *,
    force: bool = False,
    timeout: float = 30.0,
) -> Path:
    """Download and checksum the public OSF source containing the CRT-150 items."""

    destination_dir = Path(cache_dir) if cache_dir is not None else _default_nature_cache_dir()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "LLMs_responses.py"

    if destination.is_file() and not force:
        actual = _file_sha256(destination)
        if actual == NATURE_CRT150_SOURCE_SHA256:
            return destination
        raise ValueError(
            f"Cached Nature CRT-150 checksum mismatch: expected "
            f"{NATURE_CRT150_SOURCE_SHA256}, got {actual}. Use force=True to redownload."
        )

    temporary = destination.with_suffix(".py.tmp")
    request = urllib.request.Request(
        NATURE_CRT150_SOURCE_URL,
        headers={"User-Agent": "mindscopex-analysis/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            temporary.write_bytes(response.read())
        actual = _file_sha256(temporary)
        if actual != NATURE_CRT150_SOURCE_SHA256:
            raise ValueError(
                f"Downloaded Nature CRT-150 checksum mismatch: expected "
                f"{NATURE_CRT150_SOURCE_SHA256}, got {actual}."
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _literal_list_assignment(source: str, name: str, next_name: str) -> list[dict[str, Any]]:
    start_match = re.search(rf"(?m)^{re.escape(name)}\s*=", source)
    if start_match is None:
        raise ValueError(f"Nature CRT-150 source is missing {name!r}")

    end_match = re.search(rf"(?m)^{re.escape(next_name)}\s*=", source[start_match.end() :])
    if end_match is None:
        raise ValueError(f"Nature CRT-150 source is missing {next_name!r}")
    end = start_match.end() + end_match.start()

    module = ast.parse(source[start_match.start() : end])
    assignment = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ),
        None,
    )
    if assignment is None:
        raise ValueError(f"Could not parse Nature CRT-150 assignment {name!r}")
    value = ast.literal_eval(assignment.value)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise TypeError(f"Nature CRT-150 assignment {name!r} is not a list of records")
    return value


def parse_nature_crt150_source(
    source: str,
    *,
    expected_per_type: int | None = 50,
) -> tuple[NatureCRTItem, ...]:
    """Parse only literal CRT records from the published Python source without executing it."""

    items: list[NatureCRTItem] = []
    for crt_type in ("crt1", "crt2", "crt3"):
        rows = _literal_list_assignment(source, crt_type, _NATURE_ASSIGNMENT_END[crt_type])
        if expected_per_type is not None and len(rows) != expected_per_type:
            raise ValueError(
                f"Expected {expected_per_type} {crt_type.upper()} items, found {len(rows)}"
            )

        expected_numbers = list(range(1, len(rows) + 1))
        numbers = [row.get("number") for row in rows]
        if numbers != expected_numbers:
            raise ValueError(f"Unexpected item numbering for {crt_type}: {numbers}")

        for row in rows:
            task = row.get("task")
            correct = row.get("correct")
            intuitive = row.get("intuitive")
            required_values = (task, correct, intuitive)
            if not all(isinstance(value, str) and value.strip() for value in required_values):
                raise ValueError(
                    f"Incomplete Nature CRT-150 record: {crt_type} item {row.get('number')}"
                )
            number = int(row["number"])
            items.append(
                NatureCRTItem(
                    item_id=f"nature_{crt_type}_{number:03d}",
                    crt_type=crt_type,
                    number=number,
                    task=task.strip(),
                    correct_answer=correct.strip(),
                    lure_answer=intuitive.strip(),
                )
            )
    return tuple(items)


def load_nature_crt150_items(
    source_path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> tuple[NatureCRTItem, ...]:
    """Load all 150 published CRT items from a local file or the checksum-pinned OSF source."""

    path = (
        Path(source_path)
        if source_path is not None
        else download_nature_crt150_source(cache_dir, force=force_download)
    )
    return parse_nature_crt150_source(path.read_text(encoding="utf-8"))


def nature_crt150_cases(
    *,
    source_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    crt_types: tuple[NatureCRTType, ...] = ("crt1", "crt2", "crt3"),
    limit_per_type: int | None = None,
    prompt_style: NaturePromptStyle = "task_only",
) -> list[LureCase]:
    """Return a type-balanced selection of Nature CRT-150 items as experiment cases."""

    valid_types = set(_NATURE_TYPE_FAMILY)
    unknown = set(crt_types) - valid_types
    if unknown:
        raise ValueError(f"Unknown Nature CRT types: {sorted(unknown)}")
    if len(set(crt_types)) != len(crt_types):
        raise ValueError("crt_types must not contain duplicates")
    if limit_per_type is not None and limit_per_type < 1:
        raise ValueError("limit_per_type must be positive or None")

    items = load_nature_crt150_items(source_path, cache_dir=cache_dir)
    cases: list[LureCase] = []
    for crt_type in crt_types:
        selected = [item for item in items if item.crt_type == crt_type]
        if limit_per_type is not None:
            selected = selected[:limit_per_type]
        cases.extend(item.as_lure_case(prompt_style=prompt_style) for item in selected)
    return cases
