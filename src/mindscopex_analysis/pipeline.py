from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from mindscopex_analysis.capture import capture_layer_profile, load_language_model
from mindscopex_analysis.io import (
    save_layer_vectors_npz,
    save_metrics_table_json,
    write_run_manifest,
)
from mindscopex_analysis.patching import run_activation_patching_stub
from mindscopex_analysis.sae_bridge import load_sae_for_layer_stub
from mindscopex_analysis.visualize.layers import (
    plot_layer_profiles_plotly,
    plot_persona_layer_heatmap_plotly,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_prompt(prefix: str, task: str) -> str:
    parts = [s for s in (prefix.strip(), task.strip()) if s]
    return "\n\n".join(parts)


def run_persona_comparison(cfg: DictConfig, output_dir: Path | None = None) -> dict[str, Any]:
    out = Path(output_dir) if output_dir is not None else Path.cwd()
    _set_seed(int(cfg.seed))

    lm = load_language_model(cfg)
    personas = list(cfg.prompts.personas)
    task = str(cfg.prompts.shared_task)

    layer_indices: list[int] | None = None
    profiles: dict[str, list[float]] = {}
    rows: list[dict[str, Any]] = []

    for p in personas:
        pid = str(p.id)
        prompt = _build_prompt(str(p.prefix), task)
        li, scalars, _ = capture_layer_profile(lm, prompt, cfg)
        layer_indices = li
        profiles[pid] = scalars
        rows.append(
            {
                "persona_id": pid,
                "label": str(p.label),
                "prompt_chars": len(prompt),
                "layer_metric": "capture.reduce=" + str(cfg.capture.reduce),
                "per_layer": {str(i): float(s) for i, s in zip(li, scalars, strict=True)},
            }
        )

    assert layer_indices is not None

    write_run_manifest(
        out,
        cfg,
        extra={
            "persona_ids": list(profiles.keys()),
            "n_layers": len(layer_indices),
            "patching": run_activation_patching_stub(cfg),
            "sae": load_sae_for_layer_stub(cfg),
        },
    )

    if bool(cfg.artifacts.save_metrics_json):
        save_metrics_table_json(out, rows)

    persona_ids = list(profiles.keys())
    vec_list = [np.array(profiles[pid], dtype=np.float64) for pid in persona_ids]

    if bool(cfg.artifacts.save_arrays_npz):
        save_layer_vectors_npz(
            out,
            "layer_profiles.npz",
            persona_ids=persona_ids,
            layer_indices=layer_indices,
            vectors=vec_list,
        )

    labels = [str(p.label) for p in personas]
    matrix = np.stack(vec_list, axis=0)

    fig_dir = out / "figures"
    if bool(cfg.artifacts.figures.layer_profile_plotly):
        plot_layer_profiles_plotly(
            layer_indices,
            profiles,
            title="페르소나별 레이어 활성화 프로파일",
            out_path=fig_dir / "layer_profiles.html",
            export_png=bool(cfg.artifacts.export_png),
        )
    if bool(cfg.artifacts.figures.layer_heatmap_plotly):
        plot_persona_layer_heatmap_plotly(
            labels,
            layer_indices,
            matrix,
            title="페르소나 × 레이어 히트맵",
            out_path=fig_dir / "persona_layer_heatmap.html",
            export_png=bool(cfg.artifacts.export_png),
        )

    return {
        "output_dir": str(out.resolve()),
        "n_layers": len(layer_indices),
        "personas": persona_ids,
    }
