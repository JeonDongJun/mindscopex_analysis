# CRT lure-feature 통제 연구 설계 (research_experiments)

이 문서는 논문의 핵심 주장을 **엄밀하게** 검증하는 통제 연구(`research_experiments` job)의
설계·실행·해석을 정리한다. `notebooks/`의 탐색은 "무엇을 볼 수 있나"를 보여주지만 단일 문항
발견에 그친다(약점은 §1). 이 job은 "그 효과가 실재하는가"를 대조군·train/test 분할·행동
지표로 확인한다. 노트북 대비 필요성 검토는 [notebook_paper_audit.md](notebook_paper_audit.md),
데이터 원칙은 [datasets.md](datasets.md), 지표 정의는 [metrics_guide.md](metrics_guide.md)를 따른다.

## 1. 왜 이 설계인가 — 단일-case 접근의 약점

기존 파이프라인(bat-and-ball 한 문항에서 `margin_delta` 최대 feature를 고르고 그 값을 효과로
보고)에는 네 가지 구멍이 있다. 통제 연구는 각각을 정면으로 막는다.

| 약점 | 문제 | 보강 |
|------|------|------|
| **순환성(selection-on-outcome)** | top-N에서 최댓값을 고르면 항상 뭔가 나온다 | **random-direction null** 대비 z-score |
| **logprob ≠ 행동** | 2개 고정 문자열 margin은 "정답 회복" 주장을 직접 못 함 | **free-generation 정답률** readout |
| **n=1 발견** | 한 문항 발견을 4문항에 적용 | **train/test 분할**, train에서만 발견·계수 선택 |
| **control 미활용** | 특이성의 최강 증거가 각주 | **matched-control** hostile vs control 대조를 headline으로 |

## 2. 데이터 (docs/datasets.md 원칙 4·2·3)

- **주 데이터셋 `hagendorff_crt`** (150문항, `crt_difference`/`crt_rate`/`crt_growth` 각 50,
  **matched control 포함**). family-stratified train/test 분할(`split_lure_cases`, 기본 6:4).
  **발견과 계수 선택은 train split에서만** 하고 held-out test에 그대로 적용한다(원칙 4).
- **일반화 검증(선택)**: `verbal_crt`·`crt2`(비산술 → 산술 특화 여부), `yax_crt_isomorph`
  vs `crt7_classic`(표면 새로 만든 isomorph → pretraining 오염 강건성, 원칙 3). `[data].dataset`
  으로 교체.
- `crt_pilot`(9)은 smoke 전용(원칙 1).

## 3. 실험 (E1–E4 + behavioral)

공통 부호: `margin = logprob(lure) − logprob(correct)` (>0 함정 선호),
`margin_delta = baseline − intervened` (>0 개입이 함정 선호를 낮춤).

### E1 `phenomenon` — 현상 확립

- **의의.** 분석 대상(base model)이 실제로 함정 답을 선호하는지부터 확인한다. 이게 없으면
  이후 개입은 무의미하다.
- **산출.** `phenomenon.csv`(case별 `baseline_margin`, `lure_preferred`), 요약(`mean_margin`,
  `frac_lure_preferred`), `phenomenon.png`(margin 분포).
- **해석.** `frac_lure_preferred`가 0.5보다 크게 높으면 함정 우세가 성립. family별로 나눠 보면
  어느 구조에서 특히 강한지 보인다.

### E2 `discover` — 일반화되는 feature + null

- **의의.** 한 문항이 아니라 **train split 여러 문항에 걸쳐** 함정 margin을 올리는 feature를
  찾는다. 각 layer의 최고 feature 효과를 **동일 크기 무작위 방향 제거(null)** 와 비교해
  "이 방향이 특별한가"를 z-score로 계량한다.
- **산출.** `discover_localization.csv`(layer별 `mean_margin_delta`, `null_mean`, `null_z`,
  `null_percentile`, `frac_positive`), `discover_features.csv`(best layer의 상위 feature),
  `study_feature.json`(이후 단계가 쓰는 feature), `discover_localization.png`(layer vs null band).
- **해석.** 특정 layer에서 feature 곡선이 null 위로 크게 뜨고(`null_z` 큼, `null_percentile`
  ≈1) `frac_positive`가 높으면 → 국소적이고 일반화되는 lure 방향. null과 겹치면 근거 약함.

### E3-margin `causal_heldout` — held-out 인과(margin)

- **의의.** train에서 고른 feature를 **한 번도 안 본 test 문항**에 적용한다. 여기서의
  `mean_margin_delta > 0`, `frac_positive` 높음이 순환성 없는 정직한 효과다.
- **산출.** `causal_heldout.csv`(test case별 baseline/edited margin, `margin_delta`), 요약.
- **해석.** train 대비 효과가 유지되면 일반화. 크게 줄면 train 과적합(feature가 train 특이적).

### E3-behavioral `behavioral` — 자유 생성 정답률 (논문 제목 직결)

- **의의.** teacher-forced margin이 아니라 **실제 생성**에서 lure feature를 억제(음의 steering
  계수)했을 때 답이 함정→정답으로 바뀌는지 본다. "CoT 없이 정답 회복"을 직접 측정.
- **산출.** `behavioral.csv`(계수별 baseline/steered `accuracy`·`lure_rate`와 delta),
  case별 생성 텍스트(`behavioral/generations.json`), `behavioral.png`.
- **해석.** 음의 계수를 키울수록 `steered_accuracy` 상승·`steered_lure_rate` 하락이면 행동
  수준 인과. 계수를 너무 키우면 텍스트가 무너져 `other`가 늘 수 있으니 정확도·lure율을 함께 본다.

### E4 `control_specificity` — 특이성 (matched control)

- **의의.** 같은 feature가 **함정 문항(hostile)** margin은 크게 낮추지만 **matched control**
  (함정 문구만 제거, 직관답이 곧 정답) margin은 거의 안 바꿔야 "산술 일반 feature가 아니라
  lure feature"라 말할 수 있다.
- **산출.** `control_specificity.csv`(case별 `hostile_margin_delta`, `control_margin_delta`,
  `specificity_gap`), 요약.
- **해석.** `mean_specificity_gap`(= hostile − control)이 크게 양수면 특이성 충족. gap이 ≈0이면
  feature는 lure가 아니라 문제 형식 전반에 반응하는 것.

## 4. 실행

사전 준비는 [colab_cli_workflow.md](colab_cli_workflow.md)와 동일하다.

```bash
cd ~/dev/colab

# (A) 통제 연구 전체 (margin phase -> generation phase, 한 번에). 오래 걸리므로 타임아웃 상향.
./experiments/run_colab.sh experiments/suites/study.toml -s mindscopex --gpu A100 --exec-timeout 9000

# (B) 단계별 (하나씩 확인). feature가 필요한 단계는 train split에서 자동으로 먼저 발견한다.
./experiments/run_colab.sh experiments/configs/study_phenomenon_2b.toml        -s mindscopex
./experiments/run_colab.sh experiments/configs/study_discover_2b.toml          -s mindscopex
./experiments/run_colab.sh experiments/configs/study_causal_heldout_2b.toml    -s mindscopex
./experiments/run_colab.sh experiments/configs/study_control_specificity_2b.toml -s mindscopex
./experiments/run_colab.sh experiments/configs/study_behavioral_2b.toml        -s mindscopex

# (C) 가장 싼 점검 (T4): 18문항·단일 layer 발견 경로만
./experiments/run_colab.sh experiments/configs/study_smoke.toml -s mindscopex-smoke --gpu T4
```

운영 노트:

- **모델 2단계.** margin 계열 kind는 nnsight 모델을, `behavioral`은 HF 생성 모델을 쓴다. job이
  margin phase를 끝내고 모델을 내린 뒤 generation phase 모델을 올린다(순차 로드로 peak 메모리
  절약). 그래서 2b/9b에서 실용적이며, **27b 생성 단계는 권장하지 않는다**(발견/held-out margin은
  가능).
- **feature 재현/고정.** `study_discover_2b`로 확인한 (layer, feature_id)를 causal/behavioral
  config의 `[feature].layer`·`feature_id`에 적으면 재발견을 건너뛰고 그 feature로 고정 재현한다.
- **속도.** `[discover].max_cases`·`max_candidates`·`candidate_top_n`이 비용을 지배한다. 정밀도를
  높이려면 키우고(느림), 점검은 줄인다. `[data].limit_per_family`로 전체 규모도 줄일 수 있다.

## 5. 결과를 논문에 반영 (권장 새 figure set)

figure는 초안이므로 아래 통제 연구 산출로 교체하면 훨씬 강한 주장이 된다. `paper/data/`에 복사하고
`make paper-ko`. 컬럼 정본은 [../paper/data/README.md](../paper/data/README.md).

| 산출 CSV | 대체/추가할 그림 | 읽는 법 |
|----------|------------------|---------|
| `phenomenon.csv` | 현상: baseline margin 분포 | 0 오른쪽 질량 = 함정 우세 |
| `discover_localization.csv` | §finding: layer vs null band | feature가 null 위로 뜨는 layer |
| `causal_heldout.csv` | §effect: held-out 효과 | 점들이 왼쪽(정답)으로 이동 |
| `behavioral.csv` | §effect(행동): steering vs 정답률 | 음의 계수에서 정확도↑ lure율↓ |
| `control_specificity.csv` | §transfer/특이성: hostile vs control | hostile delta ≫ control delta |

초안의 기존 4개 그림(`layer_sweep`/`dose_response`/`intervention_modes`/`crt_transfer`,
`paper/_analysis.py`의 placeholder)은 그대로 두어도 빌드되지만, 논문 headline은 위 통제 산출로
교체하길 권한다.

## 6. 관련 파일

| 경로 | 내용 |
|------|------|
| `src/mindscopex_analysis/research.py` | 분할·null·일반화 발견·특이성·생성 steering primitive |
| `tests/test_research.py` | 순수 로직 단위 테스트(분할 결정성·null 통계·block finder) |
| `experiments/jobs/research_experiments.py` | 통제 연구 job(모든 kind) |
| `experiments/configs/study_*.toml` | 전체/단계별/smoke config |
| `experiments/suites/study.toml` | 통제 연구 실행 묶음 |
