# Random-direction null: 현재 의미와 selection-adjusted 확장

이 문서는 `research_experiments`의 feature discovery에서 사용하는 random-direction
null이 무엇을 통제하고, 현재 `null_z`를 왜 진단값으로만 해석해야 하며, 다음 실험에서
어떻게 max-null 검정으로 바꿀지를 정리한다.

## 현재 discovery와 null 계산

각 문항의 margin과 개입 효과는 다음 부호를 쓴다.

```text
margin = logprob(lure) - logprob(correct)
margin_delta = baseline_margin - intervened_margin
```

따라서 `margin_delta > 0`이면 feature 제거가 lure 선호를 낮췄다는 뜻이다.

현재 코드는 각 layer에서 다음 순서로 feature를 고른다.

1. discovery 문항마다 마지막 token의 SAE activation 상위 `candidate_top_n`을 모은다.
2. 적어도 `min_active_cases` 문항에서 활성화된 feature만 남긴다.
3. activation 빈도순으로 최대 `max_candidates`개를 평가한다.
4. 각 후보 feature를 discovery 문항 전체에서 제거하고 평균 `margin_delta`를 구한다.
5. 평균 `margin_delta`가 가장 큰 feature를 그 layer의 대표 feature로 고른다.
6. 모든 layer 대표 중 평균 `margin_delta`가 가장 큰 feature를 최종 선택한다.

반면 현재 random-direction null은 **각 layer의 선택된 대표 feature 하나**에 대해
discovery subset의 첫 문항 `subset[0]`만 사용한다.

1. 그 문항에서 실제 feature 제거 벡터의 L2 norm을 계산한다.
2. 같은 layer residual 공간에서 동일한 L2 norm을 가진 Gaussian random direction을
   `null_samples`번 뽑는다.
3. 각 random direction을 그 한 문항에 적용해 `margin_delta` 분포를 만든다.
4. 선택 feature의 그 한 문항 효과와 random 분포를 비교한다.

```text
z = (representative_case_delta - mean(random_deltas))
    / std(random_deltas)
```

`null_percentile`은 random delta 중 관측 delta보다 작은 값의 비율이다.

## 현재 null이 답하는 질문

현재 `null_z`는 다음 질문에는 답한다.

> 이미 선택된 feature가 대표 문항 하나에서 만든 변화는, 같은 크기의 임의 residual
> 방향을 제거했을 때보다 큰가?

동일 norm을 맞추므로 단순히 큰 벡터를 제거해서 효과가 커지는 문제를 일부 통제한다.
레이어별로 residual 공간이 다르기 때문에 null도 같은 layer 안에서 생성한다.

## 현재 null이 답하지 못하는 질문

현재 값은 다음 질문에 대한 유효한 유의확률이 아니다.

> 여러 feature와 layer를 검색해 가장 큰 효과를 골랐을 때도, 그 최댓값이 우연으로
> 설명되기 어려운가?

이유는 세 가지다.

### 1. 선택 통계와 null 통계가 다르다

feature는 discovery 문항 전체의 **평균 효과**로 선택하지만 null은 첫 문항 하나의
효과만 계산한다. 관측 통계와 귀무분포 통계의 단위가 일치하지 않는다.

### 2. 후보 검색의 최댓값 편향을 재현하지 않는다

예를 들어 서로 효과가 없는 후보 24개를 검색해도 그중 최대 평균값은 보통 0보다 크다.
여러 layer까지 검색하면 최대값은 더 커진다. 현재 null은 임의 방향 하나씩을 평가할 뿐,
각 null 반복에서 동일하게 후보와 layer의 최댓값을 고르지 않는다.

### 3. 선택된 문항·활성값 구조를 충분히 보존하지 않는다

실제 feature 제거 크기는 문항별 feature activation에 따라 달라진다. 현재 null은 대표
문항의 한 activation만 이용하므로 discovery 전체에서의 per-case 개입 크기 분포와
feature 활성 빈도를 보존하지 않는다.

따라서 현재 `null_z`가 크더라도 “후보 검색을 보정한 통계적 유의성”이라고 쓰면 안 된다.
다만 held-out test 효과와 별개로, 선택 방향이 대표 문항에서 완전히 일반적인 residual
교란인지 확인하는 진단값으로는 유용하다.

## 권장 selection-adjusted max null

확인 통계는 실제 선택 기준과 동일하게 둔다.

```text
T_observed = max over layers and candidate features(
    mean discovery margin_delta
)
```

null 반복 `b`에서도 discovery 검색을 재현한다.

```text
for b in 1..B:
    for each layer:
        후보 수와 per-case 개입 크기를 실제 검색과 동일하게 보존
        각 null 후보를 discovery 문항 전체에서 평가
        layer 내 최대 mean_margin_delta 선택
    T_null[b] = layer 전체의 최대값
```

selection-adjusted p-value는 다음처럼 계산한다.

```text
p_max = (1 + count(T_null >= T_observed)) / (B + 1)
```

`B=200`은 개발용 최소치이고, 최종 보고에는 가능하면 `B>=1000`을 권장한다. z-score는
분포가 정규라는 보장이 없으므로 보조값으로만 두고 `p_max`와 empirical percentile을
주요 값으로 보고한다.

## null 후보를 만드는 두 방법

### Gaussian max null

- 각 실제 후보마다 random unit direction을 만든다.
- 실제 후보의 문항별 activation과 decoder norm으로 개입 크기를 맞춘다.
- 이해하기 쉽지만 후보 수 × layer 수 × discovery 문항 수 × 반복 수만큼 forward가
  필요해 매우 비싸다.

### Feature-direction permutation null

- 같은 layer의 후보 activation profile은 유지한다.
- SAE decoder direction과 feature activation profile의 대응을 무작위로 섞는다.
- activation 빈도, 문항별 값, decoder norm 분포를 더 잘 보존한다.
- “특정 activation과 특정 causal direction의 정렬이 우연인가”를 검정하며 계산 캐시를
  활용하기 쉽다.

첫 구현은 permutation max null을 권장하고, 작은 subset에서 Gaussian max null과 결과가
같은 방향인지 확인한다.

## 최종 해석 순서

1. `p_max`: discovery 검색 전체를 고려해도 선택 효과가 특이한가?
2. held-out `mean_margin_delta`와 bootstrap CI: 보지 않은 문항에 인과 효과가 유지되는가?
3. `specificity_gap`: matched control보다 hostile에서 더 크게 작동하는가?
4. behavioral accuracy/lure-rate: logprob 변화가 실제 binary choice로 이어지는가?

현재 random-direction `null_z`만 통과하고 held-out·specificity가 실패하면 lure feature의
증거가 아니다. 반대로 discovery max-null과 held-out, specificity가 함께 통과해야
“검색 우연을 넘어 일반화되는 lure-related causal direction”이라고 해석할 수 있다.
