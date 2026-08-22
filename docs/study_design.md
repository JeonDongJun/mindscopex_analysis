# CRT lure-feature 통제 연구 설계 (research_experiments)

이 문서는 논문의 핵심 주장을 **엄밀하게** 검증하는 통제 연구(`research_experiments` job)의
설계·실행·해석을 정리한다. `notebooks/`의 탐색은 "무엇을 볼 수 있나"를 보여주지만 단일 문항
발견에 그친다(약점은 §1). 이 job은 "그 효과가 실재하는가"를 대조군·train/test 분할·행동
지표로 확인한다. 노트북 대비 필요성 검토는 [notebook_paper_audit.md](notebook_paper_audit.md),
데이터 원칙은 [datasets.md](datasets.md), 지표 정의는 [metrics_guide.md](metrics_guide.md)를 따른다.

## 1. 왜 이 설계인가 — 단일-case 접근의 약점

기존 파이프라인(bat-and-ball 한 문항에서 `margin_delta` 최대 feature를 고르고 그 값을 효과로
보고)에는 네 가지 구멍이 있다. 통제 연구는 각각을 정면으로 막는다.

| 약점 | 문제 | 보강 (구현됨) |
|------|------|------|
| **순환성(selection-on-outcome)** | top-N에서 최댓값을 고르면 항상 뭔가 나온다 | **peer-feature null + selection-adjusted best-of-k** (`nulls.py`) |
| **logprob ≠ 행동** | 2개 고정 문자열 margin은 "정답 회복" 주장을 직접 못 함 | constrained **correct-vs-lure 생성** readout |
| **n=1 발견** | 한 문항 발견을 4문항에 적용 | **train/test 분할**, train에서만 발견·계수 선택 |
| **control 미활용** | 특이성의 최강 증거가 각주 | **cue effect**를 목적함수로 승격 (§3.5) |
| **SAE 활성 오독** | TopK 밖 feature를 pre-activation으로 ablation | **sparse activation** 사용 (`qwen_scope_sparse_feature_values`) |
| **단일 레이어·단일 feature** | 최신 MI 기준에서 가장 약한 인과 주장 | **multi-site ablation**과 **coactivation module** 경로 추가 |

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
  "이 방향이 특별한가"를 z-score로 계량한다. `max_cases`로 일부만 쓸 때도 family별
  round-robin으로 균형 있게 뽑는다.
- **산출.** `discover_localization.csv`(layer별 `mean_margin_delta`, `null_mean`, `null_z`,
  `null_percentile`, `frac_positive`), `discover_features.csv`(best layer의 상위 feature),
  `study_feature.json`(이후 단계가 쓰는 feature), `discover_localization.png`(layer vs null band).
- **해석.** 특정 layer에서 feature 곡선이 null 위로 크게 뜨고(`null_z` 큼, `null_percentile`
  ≈1) `frac_positive`가 높으면 → 국소적이고 일반화되는 lure 방향. null과 겹치면 근거 약함.

> **현재 null의 한계.** feature 선택은 discovery 문항 전체에서 여러 후보와 layer를 검색해
> 최대 평균 효과를 고르지만, 현재 null은 각 layer 대표 feature를 `subset[0]` 한 문항에서만
> 동일 norm의 random direction과 비교한다. 따라서 `null_z`는 “이 대표 문항에서 임의 방향보다
> 큰가”를 보는 진단값이지, 후보·layer 검색의 winner's curse를 보정한 p-value가 아니다.
> 확인 실험에서는 각 null 반복마다 discovery 전체와 후보 검색을 재현하고 최댓값을 취하는
> selection-adjusted max null로 교체한다. 계산과 해석은
> [random_direction_null.md](random_direction_null.md)에 자세히 정리했다.

### E3-margin `causal_heldout` — held-out 인과(margin)

- **의의.** train에서 고른 feature를 **한 번도 안 본 test 문항**에 적용한다. 여기서의
  `mean_margin_delta > 0`, `frac_positive` 높음이 순환성 없는 정직한 효과다.
- **산출.** `causal_heldout.csv`(test case별 baseline/edited margin, `margin_delta`), 요약.
- **해석.** train 대비 효과가 유지되면 일반화. 크게 줄면 train 과적합(feature가 train 특이적).

### E3-behavioral `behavioral` — correct-vs-lure 제한 생성 (논문 제목 직결)

- **의의.** 각 문항의 정답 또는 lure 문자열만 생성할 수 있게 token-level constraint를 건 뒤,
  lure feature를 억제(음의 steering 계수)했을 때 선택이 함정→정답으로 바뀌는지 본다.
  Qwen의 `enable_thinking=False`는 chat-template 옵션이라 SAE와 같은 Base 체크포인트의 plain
  completion에는 적용되지 않는다. 제한 생성은 같은 체크포인트를 유지하면서 `<think>`와
  `both`/`other`를 원천 차단하며, held-out 일부만 쓸 때도 family-balanced subset을 사용한다.
- **산출.** `behavioral.csv`(계수별 baseline/steered `accuracy`·`lure_rate`와 delta),
  case별 생성 텍스트(`behavioral/generations.json`), `behavioral.png`.
- **해석.** 음의 계수를 키울수록 `steered_accuracy` 상승·`steered_lure_rate` 하락이면 행동
  수준 인과. 다만 이는 자유 응답이 아니라 두 후보 사이의 강제 선택이므로 teacher-forced margin과
  독립적인 자유 생성 검증으로 과장하지 않는다.

### E4 `control_specificity` — 특이성 (matched control)

- **의의.** 같은 feature가 **함정 문항(hostile)** margin은 크게 낮추지만 **matched control**
  (함정 문구만 제거, 직관답이 곧 정답) margin은 거의 안 바꿔야 "산술 일반 feature가 아니라
  lure feature"라 말할 수 있다.
- **산출.** `control_specificity.csv`(case별 `hostile_margin_delta`, `control_margin_delta`,
  `specificity_gap`), 요약.
- **해석.** `mean_specificity_gap`(= hostile − control)이 크게 양수면 특이성 충족. gap이 ≈0이면
  feature는 lure가 아니라 문제 형식 전반에 반응하는 것.

### E5 `condition_specificity` — 단서 특이성 (multi-condition 세트)

`goal_affordance_traps`처럼 matched `control_prompt`가 없고 조건이 case_id 접미사로 인코딩된
세트용. 같은 시나리오의 세 쌍둥이에 같은 개입을 적용한다.

- `hostile` (단서 있음) vs `neutral` (답·정답 동일, 단서만 제거) → **cue effect** = 두 delta의 차
- `counterfactual` (목표를 바꿔 correct/lure 스왑) → 진짜 단서 feature라면 **부호가 뒤집혀야** 함

> **주의.** `counterfactual`은 답 매핑이 뒤바뀌므로 **discovery control로 쓰면 안 된다**.
> 차분이 상쇄가 아니라 합산이 되어 검증이 순환한다. `_pair_with_controls`가 답 매핑이
> 다른 쌍둥이를 거부한다.

## 3.5 Discovery 목적함수 — hostile margin vs cue effect

`[discover].objective`로 고른다.

| 값 | 순위 기준 | 언제 |
|---|---|---|
| `hostile_margin` (기본) | `baseline − ablated` (hostile) | matched control이 없는 세트 |
| `cue_effect` | `delta(hostile) − delta(neutral)` | 조건 쌍둥이가 있는 세트 |

**왜 필요한가.** hostile margin은 *모델의 기저 선호* + *단서가 밀어올린 양*이다. 전자를 흔들기만
하는 feature도 똑같이 높은 점수를 받으므로, 특이성 게이트가 구조적으로 실패한다. 차분을 순위
기준으로 삼으면 게이트가 곧 목적함수가 된다.

**두 개의 안전장치**(둘 다 실제 실패에서 유래했다):

1. **부호 게이트.** 차분은 통제군을 망가뜨려도 최대화된다. hostile 팔이 claimed 방향으로
   움직이는 후보만 경쟁한다. 없으면 실행이 실패한다 — 통제군이 만든 cue effect는 증거가 아니다.
2. **강한 null 필수.** 값싼 per-layer 스크린은 여전히 raw hostile margin을 채점하므로,
   `objective="cue_effect"`는 `[null].selection_adjusted=true`를 요구한다. 아니면 winner가
   이 목적함수가 대체하려던 바로 그 통계로 뽑힌다.

## 3.6 Null 모델 — 무엇과 비교하는가

`margin_delta`는 단독으로 의미가 없다. "**무엇을 대신 지웠을 때보다** 큰가"가 질문이다.
`src/mindscopex_analysis/nulls.py`가 세 층위를 제공한다.

| null | 방향의 출처 | 답하는 질문 | 난이도 |
|---|---|---|---|
| Gaussian | 등방 난수, norm 매칭 | "같은 길이 아무 벡터보다 나은가" | 매우 쉬움 |
| **peer feature** | **같은 자리에서 함께 켜지는 다른 feature의 decoder** | "여기서 켜지는 실제 feature보다 나은가" | 적절 |
| **selection-adjusted** | 위 분포에서 **best-of-k**를 부트스트랩 | "같은 방식으로 뽑은 챔피언보다 나은가" | 정확 |

고차원에서 등방 난수는 어떤 의미 방향과도 거의 직교하므로(기대 |cos| ≈ √(2/πd)), Gaussian을
이기는 것은 정보량이 적다. **headline 통계는 z가 아니라 경험적 percentile**을 쓴다 — 이 delta
분포는 heavy-tailed라 소수 draw에 가우시안을 맞추면 관측된 적 없는 꼬리를 외삽한다.

> **percentile 1.0의 함정.** 부트스트랩 최댓값은 표본 최댓값을 넘을 수 없으므로, 어떤 draw도
> 관측을 못 이기면 percentile은 **구조적으로 1.0**이다. 정보는 `selection_max_mean` /
> `selection_max_p95`(그만큼 검색하면 공짜로 얻는 점수)에 있다.

## 3.7 단일 feature를 넘어서

| job | 질문 |
|---|---|
| `feature_diagnostics` | 이 feature가 positional/dense/token-identity **artifact인가?** |
| `feature_falsification` | 이 feature는 **cue를 읽는가, 형식을 읽는가?** (반증 프로파일) |
| `multisite_ablation` | 단일 레이어가 효과를 **과소평가**하는가? (인접 레이어 윈도우 + self-repair) |
| `cross_layer_siblings` | 다른 레이어의 **진짜 대응 feature**를 찾아 공동 ablation하면 달라지는가? |
| `feature_modules` | 단일 feature가 아니라 **coactivating 집합**이 매개하는가? |
| `reasoning_trajectory` | 이 표상은 **추론 과정 중** 어떻게 변하는가? (마지막 토큰 밖) |

`feature_modules`의 두 함정 회피: 모듈은 단일 feature보다 **필연적으로 더 많은 norm을 제거**하므로
null도 **모듈**이어야 한다(같은 크기·발화빈도·제거 norm). 그리고 모든 margin을 correct/lure
logprob으로 분해해 "함정 억제"와 "모델 손상"을 구분한다.

### `feature_falsification` — 인과 테스트가 못 하는 질문

ablation은 "이 방향을 지우면 margin이 움직이는가"만 답한다. 형식(template)을 읽는 feature도
그 테스트를 통과할 수 있다. 이 job은 **통과하면 안 되는 조건**을 명시적으로 건다.

| 반증 축 | cue feature라면 | template feature라면 |
|---|---|---|
| 조건 프로파일 (hostile vs neutral) | hostile에서만 발화 | 둘 다 동일하게 발화 |
| paraphrase (`template_id` 교차) | 표현이 바뀌어도 유지 | 템플릿 따라 흔들림 |
| 다른 과제 (`hagendorff_crt`) | 거의 발화 안 함 | 어디서나 발화 |
| 답 길이 상관 | 무관 | 길이를 읽고 있음 (v1에서 r=−0.56 관측) |
| FP/FN 감사 | 두 오류 모두 낮음 | 임계값을 어디에 둬도 한쪽이 폭발 |

임계값은 **discovery split에서만** 정하고 held-out에 적용한다 — 아니면 감사 자체가 순환이다.

### `cross_layer_siblings` — transplant는 "그 레이어의 feature"가 아니다

`multisite_ablation`은 한 레이어의 decoder direction을 이웃 레이어에 그대로 옮긴다. residual
stream이 basis를 공유하므로 유효한 진단이지만, **레이어마다 SAE와 번호가 다르므로** "L31의 그
feature"라는 주장은 아니다. 이 job은 대응 feature를 먼저 **식별**한다.

세 신호의 **가중 기하평균**으로 순위를 매기고, 어느 항이든 0 이하이면 점수를 0으로 만든다 —
사전이 overcomplete라 decoder cosine 하나만으로는 기하학적 우연이 1등을 할 수 있다.

| 신호 | 묻는 것 |
|---|---|
| decoder cosine | 같은 방향을 가리키는가 |
| activation corr | **같은 항목에서** 같은 세기로 켜지는가 |
| effect corr | 각각 지웠을 때 **항목별로** margin이 같은 방향으로 움직이는가 |

그다음 공동 ablation(clean / A / B / A+B)을 **norm-matched 무작위 쌍**과 함께 돌리고
**difference-in-differences**를 본다. joint 조건은 어느 한쪽보다 반드시 더 많은 norm을 제거하고
네트워크는 비선형이므로, `joint − ΣA,B`를 0과 비교하면 **아무 방향 쌍에서나 superadditive**가
나온다. null 쌍의 상호작용을 빼는 것이 이 숫자를 의미 있게 만든다. null 쌍은 **서로 독립인 두
방향**을 쓴다 — 진짜 sibling 쌍은 정렬되어 있는 것이 본질이므로, 그 정렬을 null에 넣으면 검출
대상을 미리 빼버리는 셈이다.

부수적으로 **sibling repair**를 활동값 자체에서 측정한다: A를 지운 뒤 B가 *더 세게* 켜지면
보상(self-repair)이다. margin에서 역산하지 않는다.

### `reasoning_trajectory` — 마지막 토큰 밖을 보는 유일한 job

이 연구의 모든 인과 측정은 **마지막 프롬프트 토큰** 하나를 읽는다. 그것은 "답하기 직전"의
표상이지 추론 자체가 아니다. 그런데 행동 결과는 정확히 추론에서 함정이 해소된다고 말한다
(2B: thinking off 55% → on 21%).

이 job은 생성된 trace를 따라 feature를 샘플링한다. 위치는 **항목 상대 분위수**로 잡는다 —
trace 길이가 제각각이라 절대 오프셋은 비교 불가능하다.

| phase | 위치 |
|---|---|
| `prompt_last` | 다른 모든 실험이 읽는 지점 (기준점) |
| `reasoning_0…100` | 생성 trace를 가로지르는 분위수 |
| `pre_answer` | 답이 실제로 나오는 토큰 (thinking arm만) |

**반증 가능한 예측**: lure 표상을 숙고가 억제하는 것이라면 thinking arm은 하강 궤적을 보이고
non-thinking arm은 그렇지 않아야 한다. 단순 위치 feature라면 두 조건에서 **똑같이** 흐른다.
그래서 2B(행동이 바뀜)와 27B(안 바뀜)를 **짝으로** 읽어야 한다 — 어느 한쪽만으로는 해석되지 않는다.

> **명시된 근사.** trace는 chat template으로 생성하지만 읽기는 template 없는 `prompt + trace`
> 문자열에서 한다. SAE가 학습된 Base 체크포인트에 chat template이 없기 때문이다. 두 조건을
> 같은 방식으로 읽으므로 **조건 간 비교는 유효**하지만, 절대 위치는 behavior 모델이 본 것과 다르다.

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

# (D) 후속 검증. 발견된 feature(27B L15 #81663 / 2B L17 #2144)를 config에 박아 두고 돌린다.
./experiments/run_colab.sh experiments/configs/diag_affordance_27b.toml       -s mindscopex  # degeneracy
./experiments/run_colab.sh experiments/configs/falsify_affordance_27b.toml    -s mindscopex  # 반증 프로파일
./experiments/run_colab.sh experiments/configs/multisite_affordance_27b.toml  -s mindscopex  # 인접 레이어
./experiments/run_colab.sh experiments/configs/siblings_affordance_27b.toml   -s mindscopex  # 대응 feature + DiD
./experiments/run_colab.sh experiments/configs/modules_affordance_27b.toml    -s mindscopex  # 모듈
./experiments/run_colab.sh experiments/configs/trajectory_affordance_2b.toml  -s mindscopex  # 궤적 (행동이 바뀌는 쪽)
./experiments/run_colab.sh experiments/configs/trajectory_affordance_27b.toml -s mindscopex  # 궤적 (대조군)
```

> `trajectory_*`는 **둘 다** 돌려야 한다. 2B만 보면 "궤적이 다르다"가 함정 해소인지 형식 차이인지
> 구분되지 않는다.

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

## 6. Claim level — 각 실험이 무엇을 지지하는가

주장 강도를 네 단계로 구분한다. 근거 없이 상위 단계 표현을 쓰지 않는다.

| Level | 주장 | 필요한 증거 | 지지하는 실험 |
|---|---|---|---|
| **1** Activation association | feature가 lure 조건과 연관됨 | 활성 빈도·상관 | `discover` |
| **2** Generalizing representation | held-out lure 문항으로 일반화되고 falsification을 통과 | held-out 재현 + artifact 배제 | `causal_heldout`, `feature_diagnostics` |
| **3** Causal feature | 개입이 held-out lure 선호를 바꿈 | held-out CI가 0 제외 | `causal_heldout` + peer/selection null |
| **4** Lure mechanism/module | lexical·template 통제, counterfactual 부호 뒤집힘, cross-layer, 행동 지표까지 통과 | 전 게이트 | 전체 |

**허용 표현**: `lure-associated` → `lure-sensitive` → `causally lure-supporting` → `lure-related module`.
**금지**: "the reasoning feature", "the model's intuition neuron", "the feature responsible for reasoning".

### 현재 도달 수준 (2026-08, goal_affordance_traps_v1 · 27B)

| Level | 상태 |
|---|---|
| 1 | ✅ |
| 2 | ⚠️ 부분 — token-identity는 배제됐지만 **positional 성분이 큼**(무관 과제에서도 92% 세기로 발화) |
| 3 | ❌ — held-out cue effect가 **0과 구별 안 됨** (최선 +0.092, p=.295, 양수 14/25) |
| 4 | ❌ — 단서 특이성 실패 |

**요약**: 27B L15에서 goal-affordance 단서 효과(≈5.28 nat)의 유의미한 몫을 설명하는 **단일 SAE
feature는 없다**. 최선 추정치도 2% 미만이고 0과 구별되지 않는다. 목적함수를 cue effect로 바꿔도
결과는 나아지지 않았고(오히려 통제군이 만든 효과가 뽑혔다), 2B/9B CRT의 결론과 일치한다.

**검정력 한계**: 현재 설계(held-out n=25, per-item sd 0.18~0.43)는 단서의 **1.4~3.4% 이상만**
탐지 가능하다. 그 미만을 주장하려면 문항 수를 늘려야 한다.

> 이것은 유효한 연구 결과다. 목표는 lure feature를 찾아내는 것이 아니라, 발견한 표상이 실제
> 인과 기제인지 artifact인지 **구별할 수 있는 시스템**을 만드는 것이다.

## 7. 관련 파일

| 경로 | 내용 |
|------|------|
| `src/mindscopex_analysis/research.py` | 분할·일반화 발견(cue effect 포함)·특이성·생성 steering |
| `src/mindscopex_analysis/nulls.py` | peer-feature null, selection-adjusted best-of-k, null 패널 |
| `src/mindscopex_analysis/modules.py` | coactivation 그래프·모듈·frequency-matched 모듈 null |
| `src/mindscopex_analysis/effects.py` | margin, 단일/다중 지점 개입(`EditSite`) |
| `src/mindscopex_analysis/qwen_scope.py` | SAE 로드, **sparse activation vs pre-activation** |
| `experiments/jobs/research_experiments.py` | 통제 연구 job(모든 kind) |
| `experiments/jobs/feature_diagnostics.py` | degeneracy 진단(위치·logit lens·밀도·null) |
| `experiments/jobs/feature_falsification.py` | 반증 프로파일(조건·paraphrase·답 길이·FP/FN) |
| `experiments/jobs/multisite_ablation.py` | 다층 윈도우 ablation + self-repair 감사 |
| `experiments/jobs/cross_layer_siblings.py` | 대응 feature 식별 + 조건부 공동 ablation(DiD) |
| `experiments/jobs/feature_modules.py` | coactivation 모듈 발견 + joint ablation |
| `experiments/jobs/reasoning_trajectory.py` | 추론 trace를 따라간 feature 궤적(thinking on/off) |
| `src/mindscopex_analysis/trajectory.py` | 샘플링 위치·phase 라벨·cue span (모델 불필요) |
| `src/mindscopex_analysis/siblings.py` | sibling 점수·순위·difference-in-differences |
| `tests/test_research.py`, `test_nulls.py`, `test_modules.py`, `test_cue_effect.py`, `test_qwen_scope.py`, `test_trajectory_siblings.py` | 모델 없이 도는 순수 로직 테스트 |
| `experiments/configs/study_*.toml`, `cue_*.toml`, `diag_*.toml`, `modules_*.toml`, `falsify_*.toml`, `siblings_*.toml`, `trajectory_*.toml` | 단계별 config |
