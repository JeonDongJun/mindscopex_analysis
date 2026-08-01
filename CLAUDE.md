# MindScopeX Analysis

이 저장소는 Qwen3.5와 Qwen-Scope SAE로 reasoning lure를 분석하는 연구 레포입니다.
노트북은 탐색용이고, TOML 기반 `experiments/`가 재현 가능한 통제 실험의 실행
경로입니다.

## 현재 초점

- `nnsight.LanguageModel`로 Qwen residual stream activation 캡처
- Qwen-Scope SAE checkpoint로 feature activation과 decoder direction 추출
- discovery/held-out split, random-direction null, matched control, behavioral readout 분리
- 노트북 00–13과 batch job이 같은 `src/mindscopex_analysis` 코어 사용

## 핵심 경로

| 경로 | 설명 |
|------|------|
| `src/mindscopex_analysis/models.py` | Qwen/NNsight 로딩, dtype/device 기본값 |
| `src/mindscopex_analysis/activations.py` | NNsight trace 기반 residual stream 캡처 |
| `src/mindscopex_analysis/qwen_scope.py` | Qwen-Scope SAE 로드, TopK feature 요약, layer scan |
| `src/mindscopex_analysis/effects.py` | 답변 logprob margin과 feature decoder-direction ablation |
| `src/mindscopex_analysis/research.py` | split, null, 일반화 feature, specificity, behavioral readout |
| `src/mindscopex_analysis/lure_datasets.py` | `data/*.json` 통일 로더 (`load_lure_dataset`, `lure_dataset_cases`) |
| `src/mindscopex_analysis/data/*.json` | 실험용 lure 데이터셋 (CRT/의미착각), 공통 스키마 |
| `scripts/build_datasets.py` | 원본 fetch + 정규화(1회성). `docs/datasets.md`가 카탈로그 정본 |
| `experiments/jobs/research_experiments.py` | 재현 가능한 통제 연구 batch job |
| `experiments/runners/launch_colab.py` | config/suite 실행과 Colab artifact 회수 |
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
- Colab smoke: `./experiments/run_colab.sh experiments/suites/smoke.toml -s mindscopex-smoke`

기본 mechanistic pair는 정확히 대응하는 `Qwen/Qwen3.5-27B`와
`Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50`입니다. 2B/9B/35B-A3B profile의 공식 SAE는
Base checkpoint용이므로 post-trained behavior model의 feature로 직접 해석하지 않습니다.
