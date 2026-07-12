# MindScopeX Analysis

이 저장소는 이제 Qwen 모델 내부 분석을 작게 반복하기 위한 레포입니다.

## 현재 초점

- `nnsight.LanguageModel` 로 Qwen residual stream activation 캡처
- Hugging Face의 Qwen-Scope SAE checkpoint 로 feature activation 추출
- 노트북은 목적별 최소 흐름만 유지: `01`부터 `13`까지 activation, feature search, steering, control, transfer 실험으로 분리

## 핵심 경로

| 경로 | 설명 |
|------|------|
| `src/mindscopex_analysis/models.py` | Qwen/NNsight 로딩, dtype/device 기본값 |
| `src/mindscopex_analysis/activations.py` | NNsight trace 기반 residual stream 캡처 |
| `src/mindscopex_analysis/qwen_scope.py` | Qwen-Scope SAE 로드, TopK feature 요약, layer scan |
| `src/mindscopex_analysis/effects.py` | 답변 logprob margin과 feature decoder-direction ablation |
| `src/mindscopex_analysis/lure_datasets.py` | `data/*.json` 통일 로더 (`load_lure_dataset`, `lure_dataset_cases`) |
| `src/mindscopex_analysis/data/*.json` | 실험용 lure 데이터셋 (CRT/의미착각), 공통 스키마 |
| `scripts/build_datasets.py` | 원본 fetch + 정규화(1회성). `docs/datasets.md`가 카탈로그 정본 |
| `notebooks/01_qwen_scope_activation_mvp.ipynb` | activation 캡처부터 layer 후보 선정까지의 MVP |
| `notebooks/02_bat_ball_lure_feature_ablation.ipynb` | bat-and-ball 함정 답 feature ablation 실험 |

## 실험 지도

1. `01` / `02`: 기본 activation 캡처와 bat-and-ball feature ablation.
2. `03` / `04`: layer sweep과 coefficient dose response.
3. `05`: 제거, 억제, 증폭, projection removal 비교.
4. `06` / `07` / `08` / `09`: control, paraphrase, answer format, token position 강건성.
5. `10` / `13`: CRT 및 semantic/logic lure 전이성.
6. `11`: matched-control residual delta로 우회 가능성 확인.
7. `12`: decoder geometry로 feature family 후보 확인.

## 로컬 실행

- 설치: `uv sync --extra dev`
- 노트북: `uv run jupyter lab notebooks/`
- 확인: `make lint` / `make smoke`

기본 모델은 Qwen-Scope SAE와 정확히 맞는 `Qwen/Qwen3-1.7B-Base` 입니다. 더 작은 `Qwen3-0.6B`는 현재 Qwen-Scope 공개 SAE가 없어 이 레포의 기본 경로로 쓰지 않습니다.
