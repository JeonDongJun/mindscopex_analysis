# Reasoning-lure 데이터셋 가이드

> **문서 상태:** 2026-07-31 검토 완료
> **정본(canonical doc):** 이 파일 하나에서 데이터셋 목록, 형태, 품질 상태, 사용 원칙,
> 확장 계획을 함께 관리한다.
> **기계 정본:** `src/mindscopex_analysis/data/*.json`
> **생성·정규화:** `scripts/build_datasets.py`
> **수치 표:** `scripts/dataset_stats.py`
> **무결성 감사:** `scripts/audit_datasets.py`

이 문서는 사람이 다음 질문에 빠르게 답할 수 있도록 작성했다.

- 지금 어떤 데이터셋이 있는가?
- 각 데이터셋은 어떤 함정과 정답을 표현하는가?
- 어떤 실험에 어떤 데이터셋을 써야 하는가?
- control은 무엇을 의미하고 어떻게 채점하는가?
- 현재 확인된 중복·라이선스·품질 한계는 무엇인가?
- 다음 데이터셋을 추가할 때 어떤 절차로 문서화하고 검증해야 하는가?

---

## 1. 한눈에 보는 결론

현재 정규화된 데이터는 **11개 데이터셋, 661 cases**다.

| 목적 | 우선 사용할 데이터셋 | 이유 |
|---|---|---|
| 코드·출력 smoke test | `crt_pilot` | 9문항으로 작고 여러 CRT 유형을 포함 |
| 기존 연구와 외부 비교 | `hagendorff_crt` | 150문항, 3 family 균형, matched control |
| 새 수치·표면형 core 검증 | `crt_fresh_v2` | 150문항, 수식 검증, schema v2 metadata |
| 고전 문항 노출 영향 | `crt7_classic` vs `yax_crt_isomorph` | 원본 7개와 신규 isomorph 7개 대응 |
| 비수학 반성 사고 | `crt2`, `verbal_crt` | 산술에 의존하지 않는 verbal lure |
| 거짓 전제 수용 여부 | `hagendorff_semantic_illusion` | premise rejection 50문항 |
| 목표-수단 필수조건 추론 | `goal_affordance_traps_v1` | 60 scenario × 4 condition의 paired binary choice |
| 직관→심사숙고 micro challenge | `goal_affordance_traps_v2` | 한국어 semantic cluster 1개에서 반복 검증된 4-condition set |

중요한 제한:

- `crt_fresh_v1`은 생성 파이프라인 파일럿이고 `crt_fresh_v2`가 후속 core다.
  **둘을 합쳐 180개의 독립 문항처럼 사용하지 않는다.**
- `crt_pilot`과 `crt7_classic`에는 정확히 같은 문항 2개가 있다.
- 현재 `crt_fresh_v2`는 프로그램 검증과 표면 품질 검사는 통과했지만,
  전 문항 인간 검수와 frontier-model 검증은 아직 완료되지 않았다.
- 공개 데이터셋은 pretraining 노출 가능성이 있으므로 공개 세트 성능만으로 새로운
  reasoning 일반화를 주장하지 않는다.

---

## 2. 무엇을 검토하려는 데이터인가

### 2.1 행동 현상

모델이 문제의 정답보다 직관적으로 떠오르는 함정 답(lure)을 더 선호하는지 측정한다.

```text
margin = logprob(lure) - logprob(correct)
```

- `margin > 0`: lure 선호
- `margin < 0`: correct 선호

### 2.2 Reasoning/deliberation 효과

같은 문항을 direct/no-deliberation과 reasoning/deliberation 조건으로 실행해, 추가 추론이
lure 선택을 줄이는지 본다.

### 2.3 SAE feature의 인과 효과

discovery split에서 lure-related feature를 찾고 held-out 문항에서 제거·steering했을 때
margin과 binary-choice 행동이 변하는지 본다.

### 2.4 함정 구조 특이성

hostile과 control을 비교해 feature가 단순 숫자·문장 형식이 아니라 함정 구조에
특이적으로 작동하는지 본다.

### 2.5 표면형·도메인 일반화

- 고전 CRT → fresh isomorph
- 산술 CRT → verbal CRT
- CRT → semantic illusion
- 공개 문항 → 신규 합성 core

로 넘어가도 현상이 유지되는지 확인한다.

---

## 3. 전체 인벤토리

<!-- `uv run python scripts/dataset_stats.py --overview`로 수치를 확인한다. -->

| dataset | n | schema | task / scoring | controls | 연구 역할 | 상태 |
|---|---:|---:|---|---:|---|---|
| `crt_pilot` | 9 | 1 | CRT / margin | 3 | smoke·format test | 사용 가능, 정식 benchmark 아님 |
| `hagendorff_crt` | 150 | 1 | CRT / margin | 150 | 기존 연구 비교·주 external set | 사용 가능 |
| `crt_fresh_v1` | 30 | 1 | CRT / margin | 30 | 합성 generator pilot | v2로 대체됨 |
| `crt_fresh_v2` | 150 | 2 | CRT / margin | 150 | 신규 synthetic core | 자동 검증 완료, 인간/API 검수 대기 |
| `goal_affordance_traps_v1` | 240 | 3 | goal affordance / binary choice | 180 paired conditions | 목표-필수조건·조건 효과 | v1.1 build·frontier 검증 완료 |
| `goal_affordance_traps_v2` | 4 | 3 | goal affordance / binary choice | 3 paired conditions | intuitive↔reflective micro challenge | 반복·A/B 반전 검증 완료, broad benchmark 아님 |
| `crt7_classic` | 7 | 1 | CRT / margin | 0 | 고전 기준·노출 비교 | fair-use 주의 |
| `yax_crt_isomorph` | 7 | 1 | CRT / margin | 0 | CRT-7 신규 표면형 비교 | 사용 가능 |
| `crt2` | 4 | 1 | verbal CRT / margin | 0 | 짧은 비수학 전이 | 사용 가능 |
| `verbal_crt` | 10 | 1 | verbal CRT / margin | 0 | 비수학 반성 사고 | 생성 채점 병행 권장 |
| `hagendorff_semantic_illusion` | 50 | 1 | semantic illusion / premise rejection | 50 | 거짓 전제 수용·거부 | judge 검수 권장 |

Family 합계는 다음과 같다.

| dataset | family 구성 |
|---|---|
| `crt_pilot` | arithmetic 1, counting 1, growth 1, percentage 1, rate 3, verbal 2 |
| `hagendorff_crt` | difference 50, growth 50, rate 50 |
| `crt_fresh_v1` | difference 10, growth 10, rate 10 |
| `crt_fresh_v2` | difference 50, growth 50, rate 50 |
| `goal_affordance_traps_v1` | agent capability 40, means-end conflict 40, prerequisite state 40, required resource 40, target transport 40, tool transport 40 |
| `goal_affordance_traps_v2` | goal-bound vehicle 4; 독립 semantic cluster 1개 |
| `crt7_classic` | arithmetic 1, counting 1, difference 1, growth 1, percentage 1, rate 2 |
| `yax_crt_isomorph` | arithmetic 1, counting 1, difference 1, growth 1, percentage 1, rate 2 |
| `crt2` | verbal 4 |
| `verbal_crt` | verbal 10 |
| `hagendorff_semantic_illusion` | semantic illusion 50 |

---

## 4. 공통 데이터 형태

### 4.1 Dataset 파일

각 데이터셋은 `src/mindscopex_analysis/data/<loader_id>.json` 하나로 저장한다.

```jsonc
{
  "dataset_id": "crt_fresh_v2",
  "schema_version": 2,
  "title": "...",
  "description": "...",
  "task_kind": "crt",
  "scoring": "logprob_margin",
  "source": {
    "authors": "...",
    "year": 2026,
    "title": "...",
    "venue": "...",
    "doi": "",
    "project_url": "",
    "download_url": "",
    "license": "...",
    "license_note": "..."
  },
  "generated_by": "scripts/build_datasets.py",
  "n_cases": 150,
  "family_counts": {
    "crt_difference": 50,
    "crt_growth": 50,
    "crt_rate": 50
  },
  "cases": []
}
```

### 4.2 Case 필드

| 필드 | 필수 | 의미 |
|---|---|---|
| `case_id` | 예 | 저장소 전체에서 고유한 case ID |
| `pair_id` | schema v2+ | hostile/control/paraphrase를 묶는 논리적 원형 ID |
| `template_id` | schema v2+ | 생성·문장 template ID |
| `condition` | schema v2+ | 기본 `hostile`; explicit/counterfactual/neutral 등 |
| `family` | 예 | `crt_difference`, `crt_rate`, `semantic_illusion` 등 |
| `question` | 예 | 모델에 주는 본문. JSON에는 `Answer:`를 넣지 않음 |
| `correct_answer` | margin task | 짧은 정답 표면형 |
| `lure_answer` | margin task | 짧고 명시적인 함정 답 표면형 |
| `control_question` | 선택 | 함정 구조를 변경한 대조 질문 |
| `reference_answer` | premise task | 자유서술 기준 답변·정정문 |
| `rationale` | schema v3 | pair 전체가 공유하는 필수조건·정답 근거 |
| `revision` | schema v3 | scenario/case 내용 revision |
| `note` | 선택 | 출처, 검증 방식, 생성 parameter, 제한 |

로더는 margin/binary-choice 답 앞에 공백을 붙이고 질문 끝에 `\nAnswer:`를 추가한다.

```python
from mindscopex_analysis import load_lure_dataset

cases = load_lure_dataset("crt_fresh_v2")
case = cases[0]

case.prompt          # "...?\nAnswer:"
case.correct_answer  # " $6"
case.lure_answer     # " $12"
case.pair_id
case.template_id
case.condition
```

### 4.3 Schema 버전

- **v1:** 공통 hostile case와 선택적 control 중심.
- **v2:** `pair_id`, `template_id`, `condition`을 필수로 추가.
- **v3:** 여러 condition을 독립 row로 저장하고 pair 수준 `rationale`·`revision`을 추가.
- 새 합성 데이터셋은 v2 이상을 사용한다. 네 조건을 독립 row로 저장하는
  `goal_affordance_traps_v1`은 v3다.
- 같은 scenario의 condition·paraphrase·answer-order 변형은 같은 `pair_id`를 공유한다.

### 4.4 `crt_pilot` ID 예외

파일·로더 ID는 `crt_pilot`이지만 JSON 내부 `dataset_id`는 과거 호환을 위해
`mindscopex_crt_pilot_v1`로 남아 있다. 새 코드와 문서에서는 항상 `crt_pilot`을 사용한다.
이 불일치는 감사 스크립트에서 legacy warning으로 표시한다.

---

## 5. Control의 세 가지 의미

`control_question`이 있다고 모두 같은 대조군은 아니다.

| control 유형 | 데이터셋 | hostile과의 관계 | 채점 시 주의 |
|---|---|---|---|
| **lure-becomes-correct matched control** | `hagendorff_crt`, `crt_fresh_v1/v2` | 함정 관계를 제거하면 hostile lure 값이 control 정답이 됨 | specificity 분석에 가장 적합 |
| **explicit/disambiguated control** | `crt_pilot` 일부 | 숨은 관계를 직접 말해 주며 hostile correct가 유지됨 | smoke 및 지시 명확화 확인용 |
| **semantic sanity control** | `hagendorff_semantic_illusion` | 거짓 전제가 없는 관련 질문 | premise rejection 비교용 |

현재 JSON은 control 정답을 별도 필드로 저장하지 않는다. 따라서 새 데이터셋에서는
`note`에 control 관계를 기록하고, 다음 schema 개정 시 `control_type`과
`control_answer`를 명시 필드로 승격하는 것이 좋다.

---

## 6. 채점 방식

### 6.1 `logprob_margin`

CRT 계열의 기본 채점이다.

```text
margin = logprob(lure) - logprob(correct)
```

두 후보의 길이가 다르면 합계 logprob가 표면 길이에 민감할 수 있으므로
`mean_margin`도 함께 보관한다. 문장형 답이 많은 `verbal_crt`는 constrained generation
또는 별도 judge 결과도 함께 본다.

### 6.2 Binary-choice generation

correct 또는 lure token sequence만 생성할 수 있게 제한한다.

- `<think>`와 형식 이탈을 차단한다.
- `other`가 생기지 않는다.
- 자유 생성 능력 평가가 아니라 두 후보 사이의 행동 선택이다.

`goal_affordance_traps_v1`도 `binary_choice`지만 외부 API에서는 A/B JSON schema를
사용한다. 같은 case의 direct/deliberate mode에는 동일한 option order를 적용하고,
case 간에는 안정 해시로 정답 위치를 counterbalance한다. 분석 단위는 240개 row가 아니라
기본 60개 `pair_id`이며, condition 비교도 pair 안에서 수행한다.

### 6.3 `premise_rejection`

`hagendorff_semantic_illusion`은 짧은 correct/lure 문자열 대신 거짓 전제를 거부했는지 본다.

```python
from mindscopex_analysis import classify_premise_rejection

classify_premise_rejection("No, Noah built the ark, not Moses.")  # rejected
classify_premise_rejection("Two of each animal.")                 # accepted
```

어휘 기반 판정은 baseline이다. 최종 결과는 독립 judge 또는 인간 감사가 필요하다.
원 논문의 `reference_answer`에는 작성 당시의 인물명 등 시간이 지나면 낡는 정보가 포함될 수
있으므로, 특정 고유명사 일치보다 **거짓 전제를 명시적으로 거부했는지**를 우선 채점한다.

---

## 7. 데이터셋별 상세

### 7.1 `crt_fresh_v2` — 주 synthetic core

- **크기:** 150
- **구성:** difference/rate/growth 각 50
- **형태:** family별 wording template 5개, template당 10문항
- **control:** 전 문항 lure-becomes-correct matched control
- **metadata:** 150개 고유 `pair_id`, 15개 `template_id`, `condition=hostile`
- **정답 생성:** 폐쇄형 수식
- **포함 기준:** frontier model 실패 여부와 무관

빌드 단계 검증:

- correct/lure/control 수식 assertion
- case/question/control/pair 중복 검사
- family와 template 균형 검사
- 문장 시작 대소문자, 부정관사, 이중 공백 검사
- 기존 비합성 데이터셋과 exact/near-duplicate 검사

해석 제한:

- 세 family 모두 고전 CRT 구조의 합성 isomorph다.
- “새로운 reasoning 유형 150개”가 아니라 **세 구조의 새 instance 150개**다.
- 전 문항 인간 자연스러움 검수와 frontier-model behavior 검증은 대기 중이다.
- v1과 합산하지 않는다.

예:

```text
The combined prices of the portable projector and the wireless presenter
total $39. The price of the portable projector is $27 more than the price
of the wireless presenter. What is the price of the wireless presenter?

correct: $6
lure:    $12
```

### 7.2 `hagendorff_crt` — 주 external benchmark

- **크기:** 150
- **구성:** difference/rate/growth 각 50
- **control:** 전 문항 lure-becomes-correct matched control
- **용도:** 기존 연구와 행동 비교, external validity
- **출처:** Hagendorff, Fabi & Kosinski (2023), *Nature Computational Science*
- **DOI:** <https://doi.org/10.1038/s43588-023-00527-x>
- **데이터:** <https://osf.io/w5vhp/>
- **라이선스:** CC BY 4.0

주의:

- OSF 원본 control의 일부 `correct` 필드는 hostile 값을 복사한 stale metadata다.
- 저장소는 control을 `control_question` 자극으로 사용하고, 실제 control 정답 관계는
  hostile lure가 control 정답이라는 원 구조를 따른다.
- 공개된 데이터이므로 pretraining 노출 가능성이 있다.

### 7.3 `crt_fresh_v1` — generator pilot

- **크기:** 30, family별 10
- **용도:** 합성 생성기와 control 수식 검증
- **상태:** v2가 후속 core이므로 신규 본 실험에는 v2 사용
- **라이선스:** repository Apache-2.0

v1과 v2는 같은 계보이며 표면과 구조가 강하게 겹친다. v1은 회귀·재현용으로만 보존한다.

### 7.4 `crt_pilot` — smoke set

- **크기:** 9
- **용도:** 로딩, generation, parser, plot이 작동하는지 확인
- **정식 benchmark 여부:** 아님
- **control:** 3개 explicit/disambiguated control

정식 성능 표나 독립 표본 수 계산에는 포함하지 않는다.

### 7.5 `crt7_classic` — 고전 CRT-7

- **크기:** 7
- **구성:** Frederick 원본 3 + Toplak 확장 4
- **용도:** 고전 기준, `yax_crt_isomorph`와 원본/isomorph 비교
- **라이선스:** 원 논문에 오픈 데이터 라이선스 없음

학술적 fair use 참조 자료로 취급한다. 엄격한 재배포 정책이 있는 공개 전에 검토가 필요하다.
`crt_pilot`의 bat-and-ball과 machines 항목은 이 세트와 정확히 중복된다.

### 7.6 `yax_crt_isomorph` — CRT-7 신규 표면형

- **크기:** 7
- **용도:** 고전 CRT-7 구조를 새로운 서사로 바꾼 contamination-robustness 비교
- **출처:** Yax et al. (2024), *Communications Psychology*
- **DOI:** <https://doi.org/10.1038/s44271-024-00091-8>
- **라이선스:** 논문 CC BY 4.0; 동반 repo GPL-3.0

동반 repo의 `.npy`·응답 파일은 vendoring하지 않고 공개 논문의 stimuli만 포함한다.
원 소스에는 pure-math control과 solved-example 조건도 있으나 현재 JSON에는 싣지 않았다.

### 7.7 `crt2` — 짧은 verbal CRT

- **크기:** 4
- **용도:** 수학 의존도를 낮춘 반성 사고 전이
- **출처:** Thomson & Oppenheimer (2016)
- **라이선스:** CC BY 3.0

`verbal_crt`, `crt_pilot`과 의미상 유사한 문항이 있으므로 pooled count에 주의한다.

### 7.8 `verbal_crt` — 비수학 CRT

- **크기:** 10
- **용도:** lure feature가 산술 특화인지 검토
- **출처:** Sirota et al. (2021)
- **DOI:** <https://doi.org/10.1002/bdm.2213>
- **데이터:** <https://osf.io/xehbv/>
- **라이선스:** CC BY 4.0

답이 문장인 항목은 logprob surface form에 민감하므로 generation/judge를 병행한다.
egg-yolk 문항은 문구상 모호성이 있어 balanced OpenRouter suite에서는 제외한다.

### 7.9 `hagendorff_semantic_illusion` — 거짓 전제

- **크기:** 50
- **채점:** premise rejection
- **control:** 50개 semantic sanity question
- **출처·라이선스:** `hagendorff_crt`와 동일

짧은 lure token이 없으므로 CRT margin 실험에 직접 섞지 않는다. 자유 응답에서 전제를
거부·정정했는지를 별도 채점한다.

### 7.10 `goal_affordance_traps_v1` — 목표-필수조건 추론

- **크기:** 60 base scenario × 4 condition = 240 case
- **family:** 6개, family당 10 scenario
- **condition:** `hostile`, `explicit`, `neutral`, `counterfactual` 각 60
- **채점:** correct/lure 두 후보의 counterbalanced binary choice
- **출처·라이선스:** repository-generated, Apache-2.0
- **생성:** `scripts/build_goal_affordance_dataset.py`
- **평가:** `scripts/evaluate_goal_affordance.py`
- **현재 revision:** v1.1

hostile은 가까움·편의 같은 salient cue가 목표의 필수조건을 가리는 조건이고,
explicit/neutral은 필수조건 또는 함정 단서를 통제한다. counterfactual은 목표를 바꿔
원래 lure가 정답이 되게 해 단순 option bias를 분리한다. build invariant, family/condition
균형, 세 frontier 계열의 direct/high 평가를 통과했다. 개발 과정에서 중요한 정답 사실이
explicit에만 있던 후보와 목표-행동이 맞지 않던 counterfactual은 제거하거나 수정했다.
내부 수동 curation은 완료했지만, 연구 발표 전 독립 검토자 2인의 blind 검수는 여전히
권장한다.

#### Frontier 검증 결과 — 2026-07-31

평가 설정:

| 계열 | exact model ID | direct | deliberate |
|---|---|---|---|
| OpenAI | `openai/gpt-5.6-sol` | `none` | `high` |
| Anthropic | `anthropic/claude-opus-5` | `low` | `high` |
| Google | `google/gemini-3-flash-preview` | `minimal` | `high` |

Claude는 `none`, Gemini는 `none`을 지원하지 않으므로 각 endpoint가 지원하는 최저
effort를 direct로 썼다. 응답은 strict JSON A/B, 동일 case의 direct/high 옵션 순서는
동일, case 간 위치는 SHA-256으로 counterbalance했다. 최종 full run은 1,440/1,440
응답 성공, API 오류 0, 비용 `$2.1045`였다.

| condition | GPT direct / high | Claude direct / high | Gemini direct / high |
|---|---:|---:|---:|
| hostile | 1/60 / 0/60 | 2/60 / 2/60 | 1/60 / 1/60 |
| explicit | 0/60 / 0/60 | 0/60 / 0/60 | 0/60 / 0/60 |
| neutral | 0/60 / 1/60 | 0/60 / 0/60 | 0/60 / 0/60 |
| counterfactual | 0/60 / 0/60 | 0/60 / 1/60 | 0/60 / 1/60 |

표의 값은 lure 선택 수다. direct hostile 전체는 4/180, high hostile은 3/180으로
감소했지만 표본 오류가 너무 적어 reasoning의 일반적 개선 효과를 주장할 수 없다.
오히려 Claude high에서 `tool_transport_torque_wrench_bleachers`가 새로 실패했고,
일부 control도 high에서만 단발 실패했다.

확인된 challenge subset은 `required_resource_credential` 한 pair다.

- hostile: 세 direct 모델 모두 가까운 archivist에게 먼저 가는 lure 선택
- explicit/neutral/counterfactual direct: 9/9 정답
- high: GPT만 회복, Claude/Gemini는 lure 유지
- A/B 순서 반전: 동일한 3 direct 실패와 2 high 실패 재현

따라서 “frontier 모델도 goal-affordance 함정에 빠질 수 있다”는 존재 증거는 얻었지만,
60개 전체가 frontier challenge라는 뜻은 아니다. core 60쌍은 구조·전이 평가용으로
유지하고 confirmed challenge는 별도 manifest로 관리한다.

주요 아티팩트:

- 최종 full run: `results/goal_affordance_traps_v1_1_frontier_final_20260731/`
- option-order 반전: `results/goal_affordance_traps_v1_1_challenge_reverse_20260731/`
- challenge manifest: 최종 full run 디렉터리의 `challenge_manifest.json`

### 7.11 `goal_affordance_traps_v2` — 직관/심사숙고 micro challenge

- **크기:** 독립 semantic cluster 1개 × 4 condition = 4 case
- **언어:** 한국어
- **condition:** `hostile`, `explicit`, `neutral`, `counterfactual`
- **채점:** counterbalanced binary choice
- **생성:** `scripts/build_goal_affordance_v2_dataset.py`
- **상태:** confirmed micro-challenge, broad benchmark가 아님

hostile 문항은 다음 구조다.

```text
차 타이어에 공기를 넣어야 한다. 정비소는 주차장 건너편이라
차로 돌아가는 것보다 걸어가는 편이 쉽다. 어떻게 가야 할까?

correct: 차를 운전해 정비소로 간다
lure:    정비소까지 걸어간다
```

v2 개발에서 확인한 calibration:

- 이미지와 같은 짧은 “50m 앞, 걸을까/운전할까” 8문항은 최신 frontier가 72/72 정답
- bilingual 20표면형 direct hostile은 8/60 lure였지만 모호성·비재현 항목 제외
- high-load frontier 제안 16문항은 intuitive에서 1/48 lure에 그침
- 부착 차량 부품 6문항 중 타이어 공기만 반복 재현
- 같은 의미 paraphrase는 독립 문항으로 채택하지 않음

최종 조작은 API effort만 비교하지 않는다.

| 조건 | system instruction | effort |
|---|---|---|
| intuitive | 첫인상으로 즉답하고 재검토하지 않음 | endpoint별 최저 (`none`/`low`/`minimal`) |
| reflective | 목표 대상·자원·자격·선행상태를 재점검 | `high` |

독립 호출 5회씩의 hostile 결과:

| model | intuitive lure | reflective lure |
|---|---:|---:|
| `openai/gpt-5.6-sol` | 0/5 | 0/5 |
| `anthropic/claude-opus-5` | 5/5 | 0/5 |
| `google/gemini-3-flash-preview` | 3/5 | 0/5 |
| **pooled** | **8/15 (53.3%)** | **0/15 (0%)** |

정방향·A/B 반전에서 explicit/neutral/counterfactual은 모두 정답이었고 reflective
hostile도 모두 정답이었다. 다만 5회 반복은 한 문항의 응답 확률을 추정하는 반복 측정이지
독립 표본 5개가 아니다. GPT에서는 lure가 전혀 없었으므로 “모든 frontier 모델이
함정에 빠진다”는 결론도 금지한다.

주요 아티팩트:

- 최종 데이터: `src/mindscopex_analysis/data/goal_affordance_traps_v2.json`
- 통합 보고서: `results/goal_affordance_traps_v2_final_20260802/report.md`
- 평가 manifest: 같은 디렉터리의 `evaluation_manifest.json`

---

## 8. 2026-07-31 데이터 감사 결과

실행:

```bash
uv run python scripts/audit_datasets.py --check
```

결과:

```text
11 datasets
661 cases
0 integrity errors
3 documented warnings
```

통과한 항목:

- 파일별 실제 case 수와 선언된 수 일치
- family count 일치
- 저장소 전체 `case_id` 고유
- margin 데이터의 correct/lure 비어 있지 않고 서로 다름
- control이 hostile 질문과 동일한 case 없음
- schema v2의 pair/template/condition 존재
- schema v3 goal-affordance의 60 pair × 4 condition 및 family 균형
- schema v3 v2 micro-challenge의 1 pair × 4 condition과 answer swap 관계

문서화된 warning:

1. `crt_pilot` 파일 ID와 내부 legacy `dataset_id`가 다름.
2. `crt7_classic/crt7_001` = `crt_pilot/bat_ball_original` 정확 중복.
3. `crt7_classic/crt7_002` = `crt_pilot/machines_widgets` 정확 중복.

추가로 사람이 고려해야 할 의미 중복:

- race second-place: `crt2` ≈ `verbal_crt` ≈ `crt_pilot`
- “15 sheep, all but 8”: `crt2` ≈ `crt_pilot`
- 딸 이름 함정: `crt2` ≈ `verbal_crt`
- bat-and-ball / machines / lily / class-rank: classic 계열 간 중복
- Moses / widow 계열: verbal CRT와 semantic lure 사례 간 개념 중복
- `crt_fresh_v1`과 v2: 같은 생성 계보이며 일부 문항은 매우 높은 문자열 유사도

따라서 전체 661을 하나의 독립 pooled benchmark처럼 평균 내지 않는다.

### 품질 상태 표

| dataset | 정답 근거 | control 검증 | 중복 상태 | 인간 전수 검수 | frontier 검증 |
|---|---|---|---|---|---|
| `crt_pilot` | 수작업 고전 문항 | 일부 explicit | classic과 중복 | 제한적 | 목적 아님 |
| `hagendorff_crt` | 공개 원자료 | source 구조 확인 | 공개 노출 가능 | 원 연구 | 기존 모델만 |
| `crt_fresh_v1` | 프로그램 수식 | 프로그램 수식 | v2와 강한 중복 | 표본 검수 | 미실시 |
| `crt_fresh_v2` | 프로그램 수식 | 프로그램 수식 | public near-copy 검사 통과 | **대기** | **대기** |
| `goal_affordance_traps_v1` | pair rationale·build invariant | 4-condition paired 설계 | 독립 scenario 60개 | 내부 curation 완료, 독립 blind 검수 대기 | **3계열 1,440회 + 순서 반전 완료** |
| `goal_affordance_traps_v2` | 목표 대상이 차에 부착됨 | 4-condition + A/B 반전 | v1/후보군과 같은 계보, 독립 cluster 1개 | 내부 검수 완료 | **intuitive/reflective 각 15회 반복** |
| `crt7_classic` | 원 논문 | 없음 | pilot과 정확 중복 | 공개 문항 | 공개 노출 가능 |
| `yax_crt_isomorph` | 공개 논문 | 로컬 미포함 | classic 구조 대응 | 원 연구 | 기존 연구 |
| `crt2` | 공개 논문 | 없음 | verbal/pilot 의미 중복 | 원 연구 | 별도 필요 |
| `verbal_crt` | 공개 부속자료 | 없음 | 일부 의미 중복 | 원 연구 | 별도 필요 |
| `hagendorff_semantic_illusion` | 공개 원자료 | sanity control | CRT와 채점 방식 다름 | 원 연구 | judge 재검수 필요 |

---

## 9. 실험 선택과 split 원칙

### 권장 조합

```text
개발 smoke:
  crt_pilot

기존 연구 비교:
  hagendorff_crt

새 synthetic confirmatory:
  crt_fresh_v2

원본 노출 비교:
  crt7_classic vs yax_crt_isomorph

산술→언어 전이:
  hagendorff/crt_fresh_v2 → crt2/verbal_crt

거짓 전제 전이:
  hagendorff_semantic_illusion
```

### 독립성

1. feature discovery와 coefficient 선택은 train/discovery에서만 한다.
2. held-out에는 선택된 feature와 coefficient를 그대로 적용한다.
3. 같은 `pair_id`의 condition·paraphrase·answer-order 변형은 같은 split에 둔다.
4. paraphrase, seed, 반복 호출을 독립 item처럼 세지 않는다.
5. v1과 v2, classic과 pilot의 중복 문항을 함께 세지 않는다.
6. family별 결과를 먼저 보고 전체 평균은 보조로 사용한다.

---

## 10. 데이터셋 확장 계획

### 완료

#### Phase 1 — `crt_fresh_v1`

- 30문항 파일럿
- difference/rate/growth 각 10
- 수식 기반 correct/lure/control 검증

#### Phase 2 — `crt_fresh_v2`

- 150문항 core
- family별 50, wording template별 균형
- schema v2 metadata
- exact/near-duplicate와 surface-quality build gate

남은 gate:

- 전 문항 인간 자연스러움·단위 검수
- frontier API direct/reasoning behavior 검증
- core/challenge manifest 분리

#### Phase 3 — `goal_affordance_traps_v1`

목표는 세차장 문제처럼 **눈앞의 거리·편의 단서가 최종 목표의 필수조건을 덮는 현상**을
측정하는 것이다.

60개 base scenario, family별 10개와 네 condition을 구현했다.

| family | 예시 능력 |
|---|---|
| `target_transport` | 행위 대상이 목적지에 함께 가야 함 |
| `tool_transport` | 목표 수행에 필요한 도구를 가져가야 함 |
| `required_resource` | 열쇠·신분증·티켓·결제수단 필요 |
| `agent_capability` | 수행 가능한 사람·기계가 함께 가야 함 |
| `prerequisite_state` | 충전·해동·건조·냉각 같은 선행 상태 |
| `means_end_conflict` | 가까움·빠름·편함이 최종 목표와 충돌 |

각 `pair_id`는 네 조건을 가진다.

| condition | 의미 |
|---|---|
| `hostile` | salient cue와 숨은 필수조건을 함께 제시 |
| `explicit` | 필수조건을 한 문장으로 명시 |
| `counterfactual` | 목표를 바꿔 hostile lure가 정답이 되게 함 |
| `neutral` | 함정 단서를 제거하고 나머지 표현 유지 |

이 데이터는 생활 상식의 모호성이 있으므로 프로그램 수식만으로 검증할 수 없다.
독립 검토자 2명과 불일치 adjudication이 필요하다.

#### Phase 3b — `goal_affordance_traps_v2`

v1의 낮은 frontier lure 비율을 보완하기 위해 이미지형 short prompt, bilingual surface,
entity-binding high-load 문제, 부착 차량 부품 family, paraphrase를 단계적으로 시험했다.
최종적으로 새 semantic cluster를 여러 개 확보하지 못했고, 타이어 공기 1개 pair만
intuitive→reflective 회복과 control/A-B 반전을 반복 통과했다.

v2의 올바른 용도는 다음과 같다.

- reasoning instruction·SAE steering의 민감한 micro challenge
- 모델별 응답 확률과 intervention 효과 확인
- 새로운 semantic cluster를 발굴할 때의 positive control

v2 하나로 일반적인 goal-affordance 오류율, 언어 일반화, family 일반화를 추정하지 않는다.

### 완료: Phase 4 — Frontier API 검증

- 서로 다른 제공자 계열 3개 이상
- exact model ID, 날짜, API 설정 기록
- direct/no-deliberation과 reasoning/deliberation 분리
- answer order와 paraphrase 반복
- 같은 모델을 생성기·유일 judge·challenge 선정자로 동시에 사용하지 않음

`frontier_challenge=true` 판정 조건:

1. 프로그램 또는 인간 정답 검증 통과
2. explicit/neutral에서 높은 정답률
3. hostile에서 3개 계열 중 2개 이상 lure 선택
4. answer order를 바꿔도 방향 유지
5. reasoning에서 lure rate 감소 또는 정답 회복

검증된 core 문항은 frontier가 맞혔다는 이유로 삭제하지 않는다. challenge는 별도 난이도
태그이며 데이터셋 전체의 정의가 아니다. v1.1에서는
`required_resource_credential` 한 pair가 이 기준과 A/B 반전 재검증을 통과했다.

### Phase 5 — 오픈웨이트·SAE 확인

- core와 challenge를 별도 보고
- discovery/validation/test 분리
- held-out에 feature와 coefficient 고정
- selection-adjusted max null 사용
- margin, specificity, binary-choice 행동을 함께 확인

---

## 11. 새 데이터셋 추가 절차

### 11.1 먼저 결정할 것

- 측정하려는 실패가 지식 부족인가, lure 편향인가?
- correct와 lure를 짧고 유일하게 정의할 수 있는가?
- hostile과 어떤 control을 짝지을 것인가?
- base item, condition, paraphrase의 독립 단위는 무엇인가?
- 프로그램 검증이 가능한가, 인간 adjudication이 필요한가?

### 11.2 필수 구현

1. 안정적인 `dataset_id`와 파일명 결정.
2. schema v2 이상의 JSON 생성.
3. 모든 case에 `case_id`, `pair_id`, `template_id`, `condition` 기록.
4. source·license·생성 방법 기록.
5. correct/lure/control 관계 검증.
6. family·template 균형과 중복 검사.
7. `tests/test_lure_datasets.py`의 expected count와 전용 test 추가.
8. 이 문서의 인벤토리, 상세 카드, 품질 상태, changelog 업데이트.
9. core를 동결한 뒤 frontier 결과를 별도 manifest에 저장.

### 11.3 문서화 체크리스트

새 데이터셋 상세 카드에는 반드시 다음을 쓴다.

- 한 문장 목적
- case 수와 family
- scoring 방식
- correct/lure 정의
- control 유형과 control 정답 관계
- source와 license
- 자동 검증과 인간 검수 상태
- 알려진 중복과 결합 금지 세트
- 권장 실험 및 금지된 해석
- 대표 예시 1개

### 11.4 검증 명령

```bash
# 특정 또는 전체 JSON 재생성
uv run python scripts/build_datasets.py crt_fresh_v2
uv run python scripts/build_datasets.py
uv run python scripts/build_goal_affordance_dataset.py
uv run python scripts/build_goal_affordance_dataset.py --refresh-selection
uv run python scripts/build_goal_affordance_v2_dataset.py
uv run python scripts/summarize_goal_affordance_v2.py

# Goal-Affordance 세 모델 × direct/high
uv run python scripts/evaluate_goal_affordance.py \
  --input src/mindscopex_analysis/data/goal_affordance_traps_v1.json \
  --output-dir results/goal_affordance_traps_v1_frontier

# 특정 challenge의 A/B 순서 반전
uv run python scripts/evaluate_goal_affordance.py \
  --input src/mindscopex_analysis/data/goal_affordance_traps_v1.json \
  --pair required_resource_credential --reverse-options \
  --output-dir results/goal_affordance_credential_reverse

# 수치 확인
uv run python scripts/dataset_stats.py --overview

# 무결성·정확 중복 확인
uv run python scripts/audit_datasets.py --check

# 로더 및 전체 회귀
uv run python -m unittest tests.test_lure_datasets
uv run python -m unittest discover -s tests

# 포맷
uv run ruff check src tests experiments scripts
```

---

## 12. 라이선스 요약

| dataset | 라이선스·재배포 주의 |
|---|---|
| `crt_pilot` | 저장소 자체 fixture; 고전 문항 포함, 정식 benchmark 아님 |
| `crt_fresh_v1/v2` | repository-generated, Apache-2.0 |
| `goal_affordance_traps_v1/v2` | repository-generated and curated, Apache-2.0 |
| `hagendorff_*` | 논문·부속자료 CC BY 4.0; 논문과 OSF 모두 인용 |
| `crt2` | CC BY 3.0 |
| `verbal_crt` | CC BY 4.0 |
| `crt7_classic` | 원 논문에 오픈 데이터 라이선스 없음; academic fair use |
| `yax_crt_isomorph` | 논문 CC BY 4.0; 동반 repo는 GPL-3.0 |

각 JSON의 `source.license`와 `source.license_note`가 항목별 최종 기록이다.

---

## 13. 변경 기록

| 날짜 | 변경 |
|---|---|
| 2026-08-02 | `goal_affordance_traps_v2` micro-challenge 1 pair/4 cases 추가. intuitive 8/15 vs reflective 0/15, controls·A/B 반전 검증. 카탈로그 11개/661 cases |
| 2026-07-31 | Goal-Affordance v1.1 frontier 1,440회 평가와 A/B 반전 완료. counterfactual 결함 수정, confirmed challenge 1쌍 분리 |
| 2026-07-31 | `goal_affordance_traps_v1` 60 scenario/240 case와 schema v3 추가. 카탈로그를 10개/657 cases로 갱신 |
| 2026-07-30 | 데이터 문서와 확장 계획을 이 파일로 통합. 9개/417 cases 전수 감사, control 의미·중복·품질 상태 명시, 감사 스크립트 추가 |
| 2026-07-26 | `crt_fresh_v1` 30문항과 `crt_fresh_v2` 150문항 추가 |

이후 데이터 추가·삭제·검수 상태 변경은 반드시 이 표와 상단 검토일을 함께 갱신한다.
