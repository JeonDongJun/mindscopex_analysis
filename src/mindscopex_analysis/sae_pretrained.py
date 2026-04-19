"""SAELens(`sae-lens`) 사전학습 SAE 로드·모델 매칭."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Qwen 등은 SAELens registry에 없을 수 있음 — 자동 매칭 실패 시 수동 release/sae_id 사용


@dataclass
class PretrainedSAELoadResult:
    """단일 (모델 태그 × 레이어) 에 대한 로드 결과."""

    sae: Any | None
    release: str | None
    sae_id: str | None
    ok: bool
    message: str


def _models_match(registry_model: str, user_model: str) -> bool:
    if registry_model == user_model:
        return True
    a, b = registry_model.strip(), user_model.strip()
    if a.split("/")[-1] == b.split("/")[-1]:
        return True
    return False


def _sae_id_matches_layer(sae_id: str, layer: int) -> bool:
    if re.search(rf"blocks\.{layer}\.", sae_id):
        return True
    if re.search(rf"(^|[^0-9])layer_{layer}([^0-9]|$)", sae_id) or re.search(
        rf"layer_{layer}_", sae_id
    ):
        return True
    return False


def _dtype_str_for_sae(device: str) -> str:
    return "bfloat16" if device == "cuda" else "float32"


def list_registry_pairs_for_model(model_hf_id: str) -> list[tuple[str, str]]:
    """SAELens `pretrained_saes.yaml` 에서 `model` 필드가 맞는 (release, sae_id) 목록."""
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

    out: list[tuple[str, str]] = []
    for release, info in get_pretrained_saes_directory().items():
        if not _models_match(info.model, model_hf_id):
            continue
        for sid in info.saes_map:
            out.append((release, sid))
    return out


def load_sae_lens_or_explain(
    *,
    model_id: str,
    d_in: int,
    layer: int,
    device: str,
    mode: str,
    manual_release: str | None,
    manual_sae_id: str | None,
    hook_substr: str = "hook_resid_pre",
    max_tries: int = 24,
) -> PretrainedSAELoadResult:
    """
    Parameters
    ----------
    mode
        ``auto`` — registry 에서 model·레이어·훅이 맞는 SAE 를 순차 시도해 ``d_in`` 일치하는 것 로드.
        ``manual`` — ``release`` / ``sae_id`` 필수, ``d_in`` 과 일치 검사.
        ``off`` — 사용 안 함 (호출부에서 처리).
    """
    try:
        from sae_lens import SAE
    except ImportError as e:
        return PretrainedSAELoadResult(
            None, None, None, False, f"sae-lens 미설치: {e}  →  pip install sae-lens 또는 uv sync --extra sae"
        )

    dtype = _dtype_str_for_sae(device)

    if mode == "manual":
        if not manual_release or not manual_sae_id:
            return PretrainedSAELoadResult(
                None,
                None,
                None,
                False,
                "mode=manual 인데 analysis.pretrained_sae.release / sae_id 가 비어 있습니다.",
            )
        try:
            sae = SAE.from_pretrained(
                manual_release,
                manual_sae_id,
                device=device,
                dtype=dtype,
            )
        except Exception as e:
            return PretrainedSAELoadResult(
                None, manual_release, manual_sae_id, False, f"수동 로드 실패: {e}"
            )
        got = int(getattr(sae.cfg, "d_in", -1))
        if got != d_in:
            return PretrainedSAELoadResult(
                None,
                manual_release,
                manual_sae_id,
                False,
                f"d_in 불일치: SAE={got}, 활성={d_in} (다른 레이어/모델용 가중치일 수 있음)",
            )
        return PretrainedSAELoadResult(
            sae, manual_release, manual_sae_id, True, "수동 로드 성공 (d_in 일치)"
        )

    if mode != "auto":
        return PretrainedSAELoadResult(
            None, None, None, False, f"알 수 없는 mode={mode!r} (auto|manual|off)"
        )

    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

    candidates: list[tuple[str, str]] = []
    for release, info in get_pretrained_saes_directory().items():
        if not _models_match(info.model, model_id):
            continue
        for sid in info.saes_map:
            if not _sae_id_matches_layer(sid, layer):
                continue
            if hook_substr and hook_substr not in sid:
                continue
            candidates.append((release, sid))

    if not candidates:
        n = len(list_registry_pairs_for_model(model_id))
        hint = (
            f"registry 에 이 모델용 항목이 {n} 개입니다."
            if n
            else "registry 에 이 HF 모델 id 와 일치하는 항목이 없습니다."
        )
        return PretrainedSAELoadResult(
            None,
            None,
            None,
            False,
            f"자동 매칭 실패 (layer={layer}, hook~{hook_substr!r}). {hint} "
            "Qwen 등은 공개 SAE 가 없을 수 있어 analysis.pretrained_sae 를 manual 로 지정하거나 "
            "GPT-2 / Pythia 등 지원 모델로 프리셋을 바꿔 보세요.",
        )

    last_err: str | None = None
    for i, (release, sae_id) in enumerate(candidates[:max_tries]):
        try:
            sae = SAE.from_pretrained(release, sae_id, device=device, dtype=dtype)
        except Exception as e:
            last_err = str(e)
            continue
        got = int(getattr(sae.cfg, "d_in", -1))
        if got != d_in:
            last_err = f"d_in={got} != {d_in} @ {release}/{sae_id}"
            continue
        return PretrainedSAELoadResult(
            sae,
            release,
            sae_id,
            True,
            f"자동 로드 성공: {release} / {sae_id} (d_in={d_in})",
        )

    return PretrainedSAELoadResult(
        None,
        None,
        None,
        False,
        f"후보 {min(len(candidates), max_tries)}개 시도했으나 d_in={d_in} 과 맞지 않음. 마지막: {last_err}",
    )


def encode_residuals(sae: Any, h: Any) -> Any:
    """활성 텐서 (tokens, d_in) → SAE latents (tokens, d_sae)."""
    import torch

    with torch.no_grad():
        return sae.encode(h)
