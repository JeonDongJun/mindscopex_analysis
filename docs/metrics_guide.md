# 지표 해석 가이드

Qwen-Scope feature ablation 실험에서 출력되는 수치를 읽는 법을 정리합니다.

---

## 핵심 용어

### Logprob

모델이 특정 토큰 시퀀스를 생성할 확률의 로그값입니다.  
값이 클수록(0에 가까울수록) 더 높은 확률을 뜻합니다.

```
logprob("5 cents")  = -3.061   ← 정답
logprob("10 cents") = -3.079   ← 함정 답
```

---

### Margin

두 답변 중 어느 쪽을 모델이 더 선호하는지 나타냅니다.

```
margin = logprob(lure) - logprob(correct)
```

| margin | 의미 |
|--------|------|
| `< 0` | 정답이 더 높은 확률 → 모델이 올바른 답 선호 |
| `≈ 0` | 두 답변 선호도가 거의 같음 → 불안정한 상태 |
| `> 0` | 함정 답이 더 높은 확률 → 모델이 함정에 빠진 상태 |

`mean_margin`은 답변 토큰 수로 나눈 길이 정규화 버전입니다.  
토큰 수가 다른 답변을 비교할 때 사용합니다.

---

### Ablation (제거 개입)

특정 feature의 decoder direction 벡터를 residual stream에서 빼는 개입입니다.

```
h[last_token] -= coefficient × feature_value × W_dec[:, feature_id]
```

- `feature_value`: SAE가 해당 feature에 할당한 활성화 강도
- `W_dec[:, feature_id]`: SAE decoder에서 해당 feature의 방향 벡터
- `coefficient`: 개입 강도 조절 (기본값 1.0)

---

### margin_delta

feature를 제거했을 때 margin이 얼마나 바뀌었는지를 나타냅니다.

```
margin_delta = baseline_margin - ablated_margin
```

| margin_delta | 해석 |
|---|---|
| `>> 0` | 제거 후 함정 우위가 크게 줄었음 → **함정 답 기여 feature** |
| `≈ 0` | 제거해도 선호도 변화 없음 → 이 답변 판단에 중립 |
| `<< 0` | 제거 후 함정 우위가 오히려 늘었음 → 정답을 지지하던 feature |

---

### correct_logprob_delta / lure_logprob_delta

margin_delta가 어느 방향에서 왔는지 분해하는 지표입니다.

```
correct_logprob_delta = ablated_correct_logprob - baseline_correct_logprob
lure_logprob_delta    = ablated_lure_logprob    - baseline_lure_logprob
```

| 지표 | 값의 방향 | 의미 |
|------|-----------|------|
| `correct_logprob_delta` | `> 0` | 제거 후 정답 logprob **상승** → feature가 정답 확률을 억누르고 있었음 |
| `correct_logprob_delta` | `< 0` | 제거 후 정답 logprob **하락** → feature가 정답을 도왔음 |
| `lure_logprob_delta` | `< 0` | 제거 후 함정 logprob **하락** → feature가 함정 답을 끌어올리고 있었음 |
| `lure_logprob_delta` | `> 0` | 제거 후 함정 logprob **상승** → feature가 함정 답을 억눌렀음 |

---

### cue_effect

```
cue_effect = margin_delta(hostile) - margin_delta(neutral)
```

`hostile`은 단서가 있고 `neutral`은 같은 시나리오에서 단서만 없앤 쌍둥이다(답과 정답은 동일).
hostile margin에는 *모델의 기저 선호*가 섞여 있으므로, 그것만 흔드는 feature도 `margin_delta`를
키운다. 차분은 그 공통 성분을 상쇄하고 **단서가 만든 몫만** 남긴다.

> 반드시 `mean_hostile_delta`와 함께 읽는다. 차분은 통제군을 망가뜨려도 커지므로, hostile 팔이
> 0 이하인데 cue_effect가 양수라면 그것은 통제군 효과이지 함정 효과가 아니다.

### percentile / p (null 대비)

| 컬럼 | 의미 |
|---|---|
| `gaussian_percentile` | 같은 norm의 **난수 방향**들 대비 순위 (쉬운 기준) |
| `peer_feature_percentile` | 같은 자리에서 켜지는 **다른 SAE feature**들 대비 순위 (적절한 기준) |
| `selection_adjusted_p` | 같은 방식으로 뽑은 **best-of-k** 대비 p값 (검색 규모 보정) |
| `selection_max_mean` / `selection_max_p95` | 그만큼 검색하면 **공짜로 얻는** 점수 |

z-score가 아니라 **경험적 percentile**을 headline으로 쓴다. delta 분포는 heavy-tailed라서
소수 draw에 가우시안을 맞추면 관측된 적 없는 꼬리를 외삽한다.

> `percentile = 1.0`을 "완벽한 결과"로 읽지 않는다. 부트스트랩 최댓값은 표본 최댓값을 넘을 수
> 없으므로, 어떤 draw도 관측을 못 이기면 1.0이 **구조적으로** 나온다. 실제 정보는
> `selection_max_mean`과의 거리다.

### 검정 방법

n이 25 내외이고 per-item 분포가 heavy-tailed이므로 정규 가정에 의존하지 않는다.

- **sign-flip randomization** — 짝지은 per-item 통계의 평균에 대한 정확 검정
- **bootstrap CI** — 백분위수법, 시드 고정
- **empirical percentile** — null 대비 순위

### combined_score (sibling 순위)

`cross_layer_siblings`가 다른 레이어의 대응 feature를 고를 때 쓰는 점수. 세 신호의 **가중
기하평균**이고, 어느 항이든 0 이하면 점수는 **0**이다.

| 컬럼 | 의미 |
|---|---|
| `decoder_cosine` | 두 decoder 방향이 같은 쪽을 가리키는가 |
| `activation_corr` | **같은 항목에서** 같은 세기로 켜지는가 |
| `effect_corr` | 각각 지웠을 때 **항목별로** margin이 같이 움직이는가 |
| `combined_score` | 위 셋의 가중 기하평균 (0이면 "sibling 아님") |

곱 형태를 쓰는 이유: 평균이면 `cosine=0.95`짜리 기하학적 우연이 나머지 둘이 0에 가까워도 1등을
한다. 사전이 overcomplete라 이 실패는 이론적 가능성이 아니라 기본값에 가깝다.
`min_score` 미만은 순위에 넣지 않고 **버린다** — "sibling이 없었다"와 "제일 나은 게 나빴다"를
구분하기 위해서다.

### difference_in_differences (공동 ablation)

```
DiD = (joint − A − B)real − (joint − A − B)null
```

joint 조건은 어느 한쪽보다 **반드시 더 많은 norm을 제거**하고 네트워크는 비선형이므로,
`joint − ΣA,B`를 그냥 0과 비교하면 **아무 방향 쌍에서나 superadditive**가 나온다. norm을 맞춘
무작위 쌍의 상호작용을 빼야 숫자가 의미를 갖는다. null 쌍은 **서로 독립인 두 방향**이다 —
진짜 sibling 쌍의 정렬을 null에 넣으면 검출 대상을 미리 빼버리는 셈이다.

### sibling_repair

A를 지운 forward에서 B의 **활동값 자체**를 다시 읽어 `b_after − b_before`로 계산한다.
양수면 A가 사라지자 B가 더 세게 켜진 것 — 보상(self-repair)이다. margin에서 역산하지 않는다.

### 반증 프로파일 (`feature_falsification`)

인과 테스트는 "지우면 움직이는가"만 답한다. 형식(template) feature도 그것을 통과한다.
아래는 **통과하면 안 되는** 조건들이다.

| 컬럼/축 | cue feature | template feature |
|---|---|---|
| 조건별 발화 (hostile vs neutral) | 차이 큼 | 차이 없음 |
| `template_id` 간 분산 | 작음 | 큼 |
| 다른 과제(`hagendorff_crt`) 발화율 | 낮음 | 높음 |
| 답 길이 상관 | ~0 | 큼 |
| FP/FN | 둘 다 낮음 | 임계값을 어디 둬도 한쪽 폭발 |

임계값은 **discovery split에서만** 정해 held-out에 적용한다. held-out에서 임계값을 고르면
감사 자체가 순환이 된다.

### reasoning_drift / drift_difference

`reasoning_trajectory`가 내는 궤적 요약.

| 값 | 의미 |
|---|---|
| `phase_means` | phase별 평균 활동 (`prompt_last`, `reasoning_0…100`, `pre_answer`) |
| `reasoning_drift` | 마지막 reasoning phase − 첫 reasoning phase. **음수 = 숙고하며 사그라듦** |
| `drift_difference` | thinking − non-thinking. 위치 feature면 **0에 가깝다** |

`drift_difference` 하나만으로는 해석하지 않는다. 행동이 바뀌는 2B와 바뀌지 않는 27B를 **짝으로**
읽어야 "함정 해소"와 "형식 차이"가 갈린다.

## 결과 읽기 순서

1. **`margin_delta` 부호** 확인 — 양수여야 lure 기여 feature
2. **`margin_delta` 크기** 확인 — `|baseline_margin|` 대비 얼마나 큰가?
3. **`lure_logprob_delta < 0`** 확인 — 함정 답 자체의 logprob이 내려갔는가?
4. **`correct_logprob_delta > 0`** 확인 — 정답 logprob이 올라갔는가?
5. 3·4 둘 다 충족 → **lure-sensitive 후보** (여기까지는 claim level 1~2)
6. **null 통과 확인** — `peer_feature_percentile`과 `selection_adjusted_p`
7. **held-out 재현 확인** — 신뢰구간이 0을 제외하는가 (claim level 3)
8. **특이성 확인** — `cue_effect`, counterfactual 부호 뒤집힘 (claim level 4)
9. **반증 통과 확인** — 조건 프로파일·paraphrase·답 길이·FP/FN이 전부 cue 쪽인가
10. **단일 feature 밖 확인** — `difference_in_differences`, 모듈, `drift_difference`

> 1~5만 보고 "함정 메커니즘"이라 부르지 않는다. 단계별 허용 표현은
> [study_design.md](study_design.md) §6을 따른다. 9~10은 claim level을 올리지는 않지만,
> **떨어뜨릴 수는 있다** — 반증 축 하나만 걸려도 앞 단계의 결론이 무효가 된다.

---

## 주의 사항

- **feature_value가 크다고 margin_delta가 큰 것은 아닙니다.**  
  두 답변 logprob이 함께 비슷한 크기로 내려가면 margin은 거의 변하지 않습니다.

- **`mean_margin` vs `margin`의 차이:**  
  `"10 cents"`는 `"5 cents"`보다 토큰이 많아 단순 logprob 합산이 불리합니다.  
  길이 효과를 제거하려면 `mean_margin_delta`를 참고하세요.

- **답이 문장이면 길이 교란이 실제로 큽니다.**
  `goal_affordance_traps`처럼 답이 행동 문구인 세트에서는 길이와 margin의 상관이 r=-0.56
  (분산의 31%)까지 갑니다. **현상 크기**를 보고할 때는 반드시 이 교란을 언급하세요.
  다만 **개입 효과(delta)** 는 같은 답 문자열끼리의 차분이라 길이가 상쇄됩니다.

- **SAE 활성은 pre-activation이 아닙니다.**
  Qwen-Scope는 TopK SAE라 TopK 밖 feature의 실제 기여는 0입니다. 개입 스케일에는
  `qwen_scope_sparse_feature_values()`를 쓰고, `qwen_scope_feature_preactivations()`는
  진단용으로만 쓰세요.

- **baseline_margin은 모든 feature에 동일합니다.**  
  개입 없이 측정한 단일 값이므로 feature별로 달라지지 않습니다.

---

## 관련 파일

| 파일 | 내용 |
|------|------|
| `src/mindscopex_analysis/effects.py` | `AnswerMargin`, `FeatureAblationResult`, `EditSite`, `rank_lure_feature_effects()` |
| `src/mindscopex_analysis/nulls.py` | percentile·selection 보정·peer null |
| `src/mindscopex_analysis/modules.py` | coactivation 모듈과 모듈 null |
| `src/mindscopex_analysis/siblings.py` | sibling 점수·순위·difference-in-differences |
| `src/mindscopex_analysis/trajectory.py` | 샘플링 위치와 phase 라벨 |
| `docs/study_design.md` | claim level과 각 실험이 지지하는 주장 |
| `notebooks/02_bat_ball_lure_feature_ablation.ipynb` | bat-and-ball 실험 전체 흐름 |
| `outputs/` | Colab runtime에서 생성되는 JSON, Markdown, feature handle (git 미추적) |
| `results/` | Colab CLI로 로컬에 회수한 output notebook, log, archive (git 미추적) |

원격 결과 회수 절차는 [colab_cli_workflow.md](colab_cli_workflow.md)를 참고하세요.
