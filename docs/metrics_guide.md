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

## 결과 읽기 순서

1. **`margin_delta` 부호** 확인 — 양수여야 lure 기여 feature
2. **`margin_delta` 크기** 확인 — `|baseline_margin|` 대비 얼마나 큰가?
3. **`lure_logprob_delta < 0`** 확인 — 함정 답 자체의 logprob이 내려갔는가?
4. **`correct_logprob_delta > 0`** 확인 — 정답 logprob이 올라갔는가?
5. 3·4 둘 다 충족하는 feature → **함정 메커니즘 핵심 후보**

---

## 주의 사항

- **feature_value가 크다고 margin_delta가 큰 것은 아닙니다.**  
  두 답변 logprob이 함께 비슷한 크기로 내려가면 margin은 거의 변하지 않습니다.

- **`mean_margin` vs `margin`의 차이:**  
  `"10 cents"`는 `"5 cents"`보다 토큰이 많아 단순 logprob 합산이 불리합니다.  
  길이 효과를 제거하려면 `mean_margin_delta`를 참고하세요.

- **baseline_margin은 모든 feature에 동일합니다.**  
  개입 없이 측정한 단일 값이므로 feature별로 달라지지 않습니다.

---

## 관련 파일

| 파일 | 내용 |
|------|------|
| `src/mindscopex_analysis/effects.py` | `AnswerMargin`, `FeatureAblationResult`, `rank_lure_feature_effects()` 정의 |
| `notebooks/02_bat_ball_lure_feature_ablation.ipynb` | bat-and-ball 실험 전체 흐름 |
| `results/` | 각 실험 실행 결과 분석 파일 (git 미추적) |
