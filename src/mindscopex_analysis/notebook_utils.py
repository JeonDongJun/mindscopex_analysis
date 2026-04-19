"""노트북 실험 설정 로드·해석 (경량)."""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml

_REPO_MARKER = Path("src") / "mindscopex_analysis" / "__init__.py"


def find_repo_root(cwd: Path | None = None) -> Path:
    """`src/mindscopex_analysis` 이 있는 저장소 루트를 찾는다.

    Colab 등에서 cwd 가 `/content` 만 잡히는 경우, 상위로 올라가며 탐색한다.
    환경 변수 ``MINDSCOPEX_ROOT`` (구 ``PERSONA_INTERP_ROOT``) 가 있으면 그 경로를 우선한다.
    """
    env = (
        os.environ.get("MINDSCOPEX_ROOT", "").strip()
        or os.environ.get("PERSONA_INTERP_ROOT", "").strip()
    )
    if env:
        root = Path(env).expanduser().resolve()
        if (root / _REPO_MARKER).is_file():
            return root
        raise FileNotFoundError(
            f"MINDSCOPEX_ROOT={env!r} 아래에 {_REPO_MARKER.as_posix()} 가 없습니다."
        )
    start = (cwd or Path.cwd()).resolve()
    for base in [start, *start.parents]:
        if (base / _REPO_MARKER).is_file():
            return base
    raise FileNotFoundError(
        "저장소 루트를 찾지 못했습니다. "
        "Colab에서는 `%%cd` 로 clone 한 프로젝트 루트로 이동하거나, "
        "`import os; os.environ['MINDSCOPEX_ROOT']='/content/당신의/colab'` 후 첫 셀을 다시 실행하세요."
    )


def project_root_from_notebook(cwd: Path) -> Path:
    """노트북이 열린 위치와 무관하게 저장소 루트 반환."""
    return find_repo_root(cwd)


def load_yaml_merged(path: Path, base: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return base
    with path.open(encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    return deep_merge(base, y)


def deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def dtype_from_str(s: str) -> torch.dtype:
    m = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    return m.get(s.lower(), torch.float32)


def resolve_target_layer(spec: str | int, n_layers: int) -> int:
    if isinstance(spec, int):
        if not 0 <= spec < n_layers:
            raise ValueError(f"target_layer {spec} out of range [0, {n_layers})")
        return spec
    s = str(spec).lower()
    if s in ("mid", "middle"):
        return n_layers // 2
    if s == "first":
        return 0
    if s == "last":
        return n_layers - 1
    raise ValueError(f"Unknown target_layer spec: {spec!r}")
