"""노트북 `01_qwen_neuron_comparison` 용 실험 프리셋.

노트북 상단에서 `ACTIVE_PRESET` 만 바꾸거나, YAML로 덮어쓸 수 있습니다.
"""

from __future__ import annotations

# Hydra / YAML 과 무관한 기본 골격
DEFAULT_EXPERIMENT_BASE: dict = {
    "runtime": {
        "device": None,
        "dtype": "float32",
        "max_length": 512,
        "trust_remote_code": True,
    },
    "analysis": {
        "target_layer": "mid",
        "heatmap_top_neurons": 200,
        "differential_top_k": 30,
        "sae_top_features": 50,
        "max_prompts_per_group": None,
        "sae_train": {
            "epochs": 3,
            "k": 32,
            "lr": 3e-4,
            "l1_coeff": 0.0,
        },
    },
    "tags_for_sae_plot": {
        "primary": "reasoning",
        "baseline": "non_reasoning",
    },
}

_DEFAULT_PROMPTS = {
    "intuitive": [
        "하늘은 왜 파란색인가요?",
        "봄에는 어떤 꽃이 피나요?",
        "물은 높은 곳에서 낮은 곳으로 흐른다. 맞나요?",
        "사과의 색깔은 보통 무엇인가요?",
    ],
    "logical": [
        "A는 B보다 크고, B는 C보다 큽니다. A와 C 중 어느 것이 더 큰가요? 단계별로 추론하세요.",
        "모든 고양이는 동물입니다. 일부 동물은 날 수 있습니다. 따라서 일부 고양이는 날 수 있을까요? 논리적으로 설명하세요.",
        "5명이 원형 테이블에 앉아있습니다. A는 B 옆에, C는 D 맞은편에 앉아있습니다. E는 어디에 앉아있나요?",
        "한 농부가 늑대, 양, 양배추를 강 건너편으로 옮기려 합니다. 보트에는 한 번에 하나만 실을 수 있습니다. 어떻게 해야 할까요?",
    ],
}

MODEL_PRESETS: dict[str, dict] = {
    # Qwen2.5 Instruct vs Qwen3 (추론 강화) — 규모 유사
    "qwen_reasoning_pair": {
        "models": {
            "non_reasoning": "Qwen/Qwen2.5-0.5B-Instruct",
            "reasoning": "Qwen/Qwen3-0.6B",
        },
        "prompt_groups": {k: list(v) for k, v in _DEFAULT_PROMPTS.items()},
        "tags_for_sae_plot": {"primary": "reasoning", "baseline": "non_reasoning"},
    },
    # 단일 Qwen — 직관 vs 논리만 비교 (모델 고정)
    "qwen_single_instruct": {
        "models": {"qwen_instruct": "Qwen/Qwen2.5-0.5B-Instruct"},
        "prompt_groups": {k: list(v) for k, v in _DEFAULT_PROMPTS.items()},
        "tags_for_sae_plot": {"primary": "qwen_instruct", "baseline": "qwen_instruct"},
    },
    # 동일 패밀리 3종 — 크기 스윕 (VRAM 여유 시)
    "qwen_size_sweep": {
        "models": {
            "q05": "Qwen/Qwen2.5-0.5B-Instruct",
            "q15": "Qwen/Qwen2.5-1.5B-Instruct",
            "q3": "Qwen/Qwen2.5-3B-Instruct",
        },
        "prompt_groups": {k: list(v) for k, v in _DEFAULT_PROMPTS.items()},
        "tags_for_sae_plot": {"primary": "q15", "baseline": "q05"},
    },
    # Mistral 계열 예시 (7B — GPU 권장). 모델 ID는 환경에 맞게 수정.
    "mistral_pair_example": {
        "models": {
            "mistral_inst": "mistralai/Mistral-7B-Instruct-v0.3",
            "mistral_base": "mistralai/Mistral-7B-v0.1",
        },
        "prompt_groups": {k: list(v) for k, v in _DEFAULT_PROMPTS.items()},
        "tags_for_sae_plot": {"primary": "mistral_inst", "baseline": "mistral_base"},
        "runtime": {"dtype": "bfloat16"},
    },
}

PRESET_CHOICES = tuple(MODEL_PRESETS.keys())
