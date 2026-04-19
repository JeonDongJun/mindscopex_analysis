from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def cfg_to_plain(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def write_run_manifest(output_dir: Path, cfg: DictConfig, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg_to_plain(cfg),
    }
    if extra:
        payload.update(extra)
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_layer_vectors_npz(
    output_dir: Path,
    filename: str,
    persona_ids: list[str],
    layer_indices: list[int],
    vectors: list[np.ndarray],
) -> Path:
    """vectors[i] shape (n_layers,) — 페르소나별 레이어 스칼라 프로파일."""
    ensure_dir(output_dir / "arrays")
    out = output_dir / "arrays" / filename
    np.savez(
        out,
        persona_ids=np.array(persona_ids, dtype=object),
        layer_indices=np.array(layer_indices, dtype=np.int32),
        profiles=np.stack(vectors, axis=0),
    )
    return out


def save_metrics_table_json(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_dir(output_dir / "tables")
    path = output_dir / "tables" / "layer_metrics.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
