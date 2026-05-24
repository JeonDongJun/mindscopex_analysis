# MindScopeX Analysis

이 저장소는 이제 Qwen 모델 내부 분석을 작게 반복하기 위한 레포입니다.

## 현재 초점

- `nnsight.LanguageModel` 로 Qwen residual stream activation 캡처
- Hugging Face의 Qwen-Scope SAE checkpoint 로 feature activation 추출
- 노트북은 목적별 최소 흐름만 유지: `01_qwen_scope_activation_mvp.ipynb`, `02_bat_ball_lure_feature_ablation.ipynb`

## 핵심 경로

| 경로 | 설명 |
|------|------|
| `src/mindscopex_analysis/models.py` | Qwen/NNsight 로딩, dtype/device 기본값 |
| `src/mindscopex_analysis/activations.py` | NNsight trace 기반 residual stream 캡처 |
| `src/mindscopex_analysis/qwen_scope.py` | Qwen-Scope SAE 로드, TopK feature 요약, layer scan |
| `src/mindscopex_analysis/effects.py` | 답변 logprob margin과 feature decoder-direction ablation |
| `notebooks/01_qwen_scope_activation_mvp.ipynb` | activation 캡처부터 layer 후보 선정까지의 MVP |
| `notebooks/02_bat_ball_lure_feature_ablation.ipynb` | bat-and-ball 함정 답 feature ablation 실험 |

## 로컬 실행

- 설치: `uv sync --extra dev`
- 노트북: `uv run jupyter lab notebooks/`
- 확인: `make lint` / `make smoke`

기본 모델은 Qwen-Scope SAE와 정확히 맞는 `Qwen/Qwen3-1.7B-Base` 입니다. 더 작은 `Qwen3-0.6B`는 현재 Qwen-Scope 공개 SAE가 없어 이 레포의 기본 경로로 쓰지 않습니다.
