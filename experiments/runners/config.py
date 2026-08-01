"""Small TOML config helpers shared by experiment jobs and launchers."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a table: {path}")
    return data


def table(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"[{name}] must be a table")
    return value


def bool_or_none_list(value: Any, *, default: tuple[bool | None, ...]) -> tuple[bool | None, ...]:
    if value is None:
        return default
    if not isinstance(value, list):
        raise TypeError(f"Expected list[bool | null], got {value!r}")
    if not all(item is None or isinstance(item, bool) for item in value):
        raise TypeError(f"Expected list[bool | null], got {value!r}")
    return tuple(value)


def int_list(value: Any, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise TypeError(f"Expected list[int], got {value!r}")
    if not value:
        raise ValueError("Seed list must not be empty")
    return tuple(value)


def run_name(config: dict[str, Any]) -> str:
    name = table(config, "run").get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("[run].name must be a non-empty string")
    return name.strip()


def job_name(config: dict[str, Any]) -> str:
    job = table(config, "run").get("job")
    if not isinstance(job, str) or not job.strip():
        raise ValueError("[run].job must be a non-empty string")
    return job.strip()
