from __future__ import annotations

from typing import Any

from omegaconf import DictConfig


def load_sae_for_layer_stub(cfg: DictConfig) -> dict[str, Any]:
    """
    sae-lens 연동 지점. `uv sync --extra sae` 후 SAE 가중치·훅을 여기서 로드.
    """
    if not bool(cfg.sae.enabled):
        return {"status": "skipped"}
    try:
        import sae_lens  # noqa: F401
    except ImportError as e:
        return {
            "status": "error",
            "message": "sae-lens 미설치. uv sync --extra sae",
            "detail": str(e),
        }
    return {
        "status": "todo",
        "release": cfg.sae.release,
        "hook_layer": cfg.sae.hook_layer,
    }
