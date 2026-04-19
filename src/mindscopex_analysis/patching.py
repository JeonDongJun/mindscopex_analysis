from __future__ import annotations

from typing import Any

from omegaconf import DictConfig


def run_activation_patching_stub(cfg: DictConfig) -> dict[str, Any]:
    """
    Activation patching 본구현 전 자리표시자.
    nnsight `.edit` / 교차 페르소나 잔차 주입 등은 여기에 연결.
    """
    if not bool(cfg.patching.enabled):
        return {"status": "skipped", "reason": "patching.enabled is false"}
    return {
        "status": "todo",
        "donor": cfg.patching.donor_persona_id,
        "receiver": cfg.patching.receiver_persona_id,
        "target_layer": int(cfg.patching.target_layer),
        "hint": "capture.py에서 donor/receiver 순전파 hidden을 저장한 뒤 receiver trace에서 해당 레이어 output을 donor 텐서로 치환.",
    }
