# 실험 데이터셋 카탈로그

검토 기준일: 2026-07-12

이 문서는 `src/mindscopex_analysis/data/` 아래에 있는 모든 reasoning-lure 데이터셋을
설명한다. 모든 데이터셋은 **동일한 JSON 스키마**로 저장되고 하나의 로더
(`load_lure_dataset`)로 읽는다. 각 원본은 `scripts/build_datasets.py`가 1회성으로
내려받아 정규화하며, 결과 JSON은 저장소에 커밋되므로 노트북은 네트워크 없이 동작한다.

## 개요

<!-- 아래 표/통계는 `uv run python scripts/dataset_stats.py --overview` 로 재생성 -->

| dataset | n | task_kind | scoring | families |
|---------|--:|-----------|---------|----------|
| `crt2` | 4 | crt | logprob_margin | crt_verbal (4) |
| `crt7_classic` | 7 | crt | logprob_margin | crt_arithmetic (1), crt_counting (1), crt_difference (1), crt_growth (1), crt_percentage (1), crt_rate (2) |
| `crt_pilot` | 9 | crt | logprob_margin | crt_arithmetic (1), crt_counting (1), crt_growth (1), crt_percentage (1), crt_rate (3), crt_verbal (2) |
| `hagendorff_crt` | 150 | crt | logprob_margin | crt_difference (50), crt_growth (50), crt_rate (50) |
| `hagendorff_semantic_illusion` | 50 | semantic_illusion | premise_rejection | semantic_illusion (50) |
| `verbal_crt` | 10 | crt | logprob_margin | crt_verbal (10) |
| `yax_crt_isomorph` | 7 | crt | logprob_margin | crt_arithmetic (1), crt_counting (1), crt_difference (1), crt_growth (1), crt_percentage (1), crt_rate (2) |

현재 총 **7개 데이터셋, 237 cases**.

## 공통 스키마

각 파일은 자기 기술적(self-describing) JSON 하나다.

```jsonc
{
  "dataset_id": "hagendorff_crt",
  "schema_version": 1,
  "title": "...",
  "description": "...",
  "task_kind": "crt" | "semantic_illusion",
  "scoring": "logprob_margin" | "premise_rejection",
  "source": { "authors", "year", "title", "venue", "doi",
              "project_url", "download_url", "source_sha256",
              "license", "license_note" },
  "generated_by": "scripts/build_datasets.py",
  "n_cases": 150,
  "family_counts": { "crt_difference": 50, ... },
  "cases": [
    {
      "case_id": "hagendorff_crt_difference_001",
      "family": "crt_difference",
      "question": "A pear and a fridge together cost $140. ...",
      "correct_answer": "$20.0",
      "lure_answer": "$40.0",
      "control_question": "A pear and a fridge together cost $140. The pear costs $100. ...",  // optional
      "reference_answer": "...",   // optional (premise-rejection 세트용 자유서술 정답)
      "note": "..."                // optional
    }
  ]
}
```

핵심 규칙:

- `correct_answer` / `lure_answer` 는 **맨앞 공백 없는 표면형**으로 저장한다. 로더가
  logprob 채점기가 요구하는 앞 공백(`" $20.0"`)과 `"\nAnswer:"` 구분자를 붙인다.
- `scoring="logprob_margin"`: 짧고 서로 다른 `correct`/`lure` 를 teacher-forced logprob
  margin으로 비교한다(대부분의 CRT).
- `scoring="premise_rejection"`: 짧은 lure 문자열이 없다. `correct_answer`/`lure_answer`
  는 비어 있고, 권위 있는 정정문은 `reference_answer`(로더에서 `note`로 접힘)에 있다.
  자유서술 응답에서 "거짓 전제를 거부했는가"를 판정하는 방식으로 채점한다(의미 착각).

## 로더 사용법

```python
from mindscopex_analysis import (
    available_lure_datasets,   # 7개 dataset id (JSON 파일 stem)
    load_lure_dataset,         # -> list[LureCase]
    lure_dataset_cases,        # -> list[LureCase], family 필터/개수 제한
    lure_dataset_info,         # -> LureDatasetInfo (단일 메타데이터)
    lure_dataset_catalog,      # -> list[LureDatasetInfo] (전체)
    load_all_lure_cases,       # -> {dataset_id: list[LureCase]}
)

cases = load_lure_dataset("hagendorff_crt")   # 150 LureCase, 각 항목에 control_prompt 포함
info = lure_dataset_info("hagendorff_crt")
info.n_cases, info.family_counts, info.scoring, info.source["doi"]

# 노트북/experiment preset용 편의 함수: family별 개수 제한(옛 nature_smoke = 유형별 3문항)
smoke = lure_dataset_cases("hagendorff_crt", limit_per_family=3)     # 9개
rate_only = lure_dataset_cases("hagendorff_crt", families=("crt_rate",))
```

반환 타입은 기존 실험과 동일한 `LureCase`이므로 `answer_logprob_margin`,
`rank_lure_feature_effects`, `control_delta_bypass_rows` 등에 그대로 넣을 수 있다.
노트북 `00`은 `RUN_PRESET`으로 이 데이터셋들을 선택한다(`hagendorff_smoke`,
`hagendorff_full`, `verbal_crt`, `crt7_classic`, `crt2`, `yax_crt_isomorph`, `pilot`).

### premise-rejection 채점 (의미 착각용)

`hagendorff_semantic_illusion`처럼 `scoring="premise_rejection"`인 세트는 짧은 lure가
없어 logprob margin으로 채점하지 않는다. `classify_premise_rejection`이 생성 응답이
거짓 전제를 거부했는지(정답) 아니면 거짓 프레임 안에서 답했는지(함정)를 어휘 기반으로
판정하는 baseline이다(엄밀 채점은 LLM judge 권장).

```python
from mindscopex_analysis import classify_premise_rejection

classify_premise_rejection("No, Noah built the ark, not Moses.")   # 'rejected'
classify_premise_rejection("Two of each animal.")                  # 'accepted'
# reference_answer(각 case JSON의 필드)를 주면 정정된 고유 명사도 부차 신호로 사용
```

## 채택 데이터셋

### `crt_pilot` — MindScopeX CRT pilot (9)

파이프라인/출력 형식 점검용 소형 hand-curated 세트. 정식 벤치마크가 아니라 생성,
응답 파싱, 재시도, 시각화가 정상 동작하는지 확인하는 smoke test다. classic
CRT(bat-and-ball 등) + verbal/counting/percentage 변형을 섞어 둔다.

- 파일: `data/crt_pilot.json` · scoring: logprob_margin
- 예시 — prompt: "A bat and a ball cost $1.10 in total. The bat costs $1.00 more
  than the ball. How much does the ball cost? Answer in cents." / correct `5 cents`
  / lure `10 cents` / control "The bat costs $1.05 ..."

### `hagendorff_crt` — Hagendorff et al. (2023) CRT-150 + matched controls (150)

세 가지 CRT 구조를 각각 50개씩 변형한 150문항. **각 hostile 항목에 matched
control이 `control_question`으로 포함**되어 있다(표면 동일, 함정만 제거).

- family: `crt_difference`(bat-and-ball형 가격차) / `crt_rate`(작업률) /
  `crt_growth`(지수성장), 각 50
- 출처: Hagendorff, Fabi & Kosinski (2023), *Nature Computational Science* 3, 833–838.
  DOI [10.1038/s43588-023-00527-x](https://doi.org/10.1038/s43588-023-00527-x) ·
  데이터 [OSF w5vhp](https://osf.io/w5vhp/) · 원본 `LLMs_responses.py`
  (SHA-256 `cdf4617e…a4125`) · 라이선스 CC BY 4.0(논문/부속자료)
- 예시 — prompt: "A pear and a fridge together cost $140. The pear costs $100 more
  than the fridge. How much does the fridge cost?" / correct `$20.0` / lure `$40.0`
  / control "The pear costs $100. ..."

> **대조군(control) 의미 주의.** OSF 원본의 `*_not_hostile` 리스트는 hostile 항목과
> `number`로 1:1 매칭된다. 함정 문구만 제거했기 때문에 **대조 질문의 실제 정답은 그
> 항목의 `intuitive`(=lure) 값과 같다.** 특히 `crt1`(difference)·`crt3`(growth)
> 원본은 대조군의 `correct` 필드에 hostile 쪽 값을 그대로 복사해 두어(그래서 stale),
> 대조 질문 자체의 정답과 다르다. 이 저장소는 대조군을 **자극 프롬프트**로만
> 쓰는 실험(exp 06 control, exp 11 matched-control residual delta;
> `control_delta_bypass_rows`)을 위해 `control_question`으로만 싣고, 대조군의 별도
> 정답은 저장하지 않는다(필요하면 lure 값이 곧 대조군 정답).

### `hagendorff_semantic_illusion` — Hagendorff et al. (2023) 의미 착각 (50)

거짓 전제를 가진 상식 질문 50개("Who is the dictator of South Korea?"). 문자열
margin이 아니라 **premise-rejection**으로 채점한다: 원본은 자유서술 정정문
(`reference_answer`)과 오도하지 않는 matched control(`control_question`, OSF의
`six_sanity`)을 주지만 짧은 lure 토큰은 없다. 저장소의 기존
`semantic_lure_cases()`(Moses 등 3개, 짧은 lure 보유)와 상호보완적이다.

- 출처/라이선스: `hagendorff_crt`와 동일(OSF w5vhp, CC BY 4.0)
- 예시 — prompt: "Who is the dictator of South Korea?" / control "Is South Korea a
  dictatorship?" / reference_answer "South Korea is a democratic country ...
  President ... Moon Jae-in ..."

### `crt2` — CRT-2 (Thomson & Oppenheimer 2016) (4)

수리 의존도를 낮추고 노출 빈도를 줄인 4문항 대안 CRT. 언어형 함정(insight)이라
"lure feature가 산술 특화인가"를 보는 데 유용하다.

- family: crt_verbal ×4 · scoring: logprob_margin
- 출처: *Judgment and Decision Making* 11(1), 99–113 ·
  [PDF](https://journal.sjdm.org/15/151029/jdm151029.pdf) · 라이선스 **CC BY 3.0**
- 예시 — prompt: "A farmer had 15 sheep and all but 8 died. How many are left?" /
  correct `8` / lure `7`

### `verbal_crt` — Verbal CRT / CRT-V (Sirota et al. 2021) (10)

완전 비수학 10문항 CRT. 산술 없이 반성적 사고만 요구하므로 `crt_arithmetic` 계열과
분리해 lure feature의 도메인 일반성을 검증하기에 좋다. 응답 표면형이 문장이라 일부는
logprob margin보다 생성 판정(judge)이 더 적합할 수 있다.

- family: crt_verbal ×10 · scoring: logprob_margin
- 출처: *J. Behavioral Decision Making* 34(3), 322–343.
  DOI [10.1002/bdm.2213](https://doi.org/10.1002/bdm.2213) ·
  [OSF xehbv](https://osf.io/xehbv/) · 문항 [Supplementary PDF](https://osf.io/download/64x92/) ·
  라이선스 **CC BY 4.0**(응답 CSV는 1=correct/2=intuitive/3=other로 코딩)
- 예시 — prompt: "How many of each animal did Moses put on the ark?" / correct
  `none` / lure `two`

### `crt7_classic` — 고전 CRT-7 (Frederick 2005 + Toplak et al. 2014) (7)

표준 참조 도구. Frederick 원본 3(bat-and-ball, machines, lily) + Toplak 확장 4
(shared-rate, class-rank, pig-trading, Simon-stocks). 여러 항목이 `crt_pilot`과 겹친다.

- family: difference/rate/growth/counting/arithmetic/percentage · scoring: logprob_margin
- 출처: *J. Economic Perspectives* 19(4)(Frederick) + *Thinking & Reasoning* 20(2)
  (Toplak, DOI [10.1080/13546783.2013.844729](https://doi.org/10.1080/13546783.2013.844729))
- 라이선스: **원 논문에 오픈 데이터 라이선스 없음** — stimuli를 학술적 fair use로 전사.
  동일 7문항이 Yax et al.(2024)의 CC-BY 부속자료(GBP 표기)에도 있다.
- 예시 — prompt: "Jerry received both the 15th highest and the 15th lowest mark in
  the class. How many students are in the class?" / correct `29 students` / lure `30 students`

### `yax_crt_isomorph` — Yax et al. (2024) 신규 CRT isomorph (7)

고전 CRT-7 구조에 1:1로 대응하되 표면 서사를 새로 만든 7문항. pretraining 오염을
줄이려는 설계라 `crt7_classic`과 짝지어 **원본 vs isomorph 대조**에 쓴다. 원 소스는
matched pure-math control(`Experiment='crt-math'`)과 solved-example 조건도 제공한다.

- family: difference/rate/growth/counting/arithmetic/percentage · scoring: logprob_margin
- 출처: *Communications Psychology*,
  DOI [10.1038/s44271-024-00091-8](https://doi.org/10.1038/s44271-024-00091-8) ·
  [repo](https://github.com/hrl-team/ReasoningGPT)
- 라이선스: 논문 **CC BY 4.0**; 동반 코드/응답 repo는 **GPL-3.0**.
  → GPL 응답 데이터·`.npy`는 vendoring하지 않고 **공개 논문의 stimuli만** 전사했다.
- 예시 — prompt: "A scarf costs 210 euros more than a hat. The scarf and the hat cost
  220 euros in total. How much does the hat cost?" / correct `5 euros` / lure `10 euros`

### 정규화하지 않은 인접 자료

- **NeuBAROCO** (Ozeki et al. 2024, `kmineshima/NeuBAROCO`, CC BY 4.0): 삼단논법
  belief-bias NLI 790행 + MC 80행. 스키마가 전제/결론/타당성(`gold`)/신념일치
  (`content-type`: congruent/incongruent/symbolic)라 이 저장소의 lure 스키마
  (question/correct/lure)와 맞지 않아 JSON화하지 않았다. 논리 편향 실험을 별도
  스키마로 진행할 때 사용:
  [`NeuBAROCO_NLI.tsv`](https://raw.githubusercontent.com/kmineshima/NeuBAROCO/main/acl2024/NeuBAROCO_NLI.tsv) ·
  [`NeuBAROCO_MC.tsv`](https://raw.githubusercontent.com/kmineshima/NeuBAROCO/main/acl2024/NeuBAROCO_MC.tsv)

### 중복 항목 주의(de-dup)

명시된 서로 다른 published instrument라 그대로 싣지만, 혼합 사용 시 다음 중복을 유의:

- "pass the person in 2nd place": `crt2`#1 ≈ `verbal_crt`#2 ≈ `crt_pilot` race_second_place
- 딸 이름 함정: `crt2`#3(Emily) ≈ `verbal_crt`#1(Mary)
- "15 sheep, all but 8": `crt2`#2 ≈ `crt_pilot` sheep_all_but
- bat-and-ball / machines / lily / class-rank: `crt7_classic` ⊂ `crt_pilot`과 겹침
- Moses / widow's sister: `verbal_crt`#6·#9 ≈ `semantic_lure_cases()`(cases.py 수작업 3종)

## 재생성

```bash
# 원본을 다시 내려받아 정규화 JSON 재생성
uv run python scripts/build_datasets.py                 # 전체
uv run python scripts/build_datasets.py hagendorff_crt  # 일부

# 통계/예시 마크다운 재생성(위 개요 표 갱신용)
uv run python scripts/dataset_stats.py --overview
uv run python scripts/dataset_stats.py                  # 전체 카탈로그
```

## 연구 사용 원칙

1. `crt_pilot`(9)은 코드/출력 형식 점검에만 쓴다.
2. `hagendorff_crt`(150)는 기존 연구와 비교하는 외부 행동 벤치마크로 쓴다.
3. 공개 문항의 pretraining 노출 가능성을 배제할 수 없으므로 공개 세트 결과만으로 새
   reasoning 일반화를 주장하지 않는다. 최종 인과 주장은 별도로 만든 미공개 수치·표면형
   변형에서 재검증한다.
4. feature 탐색과 계수 선택은 discovery split에서만 하고 held-out item에 그대로 적용한다.
5. 유형·item·paraphrase·seed를 구분해 저장하고 paraphrase나 seed를 독립 표본처럼 세지 않는다.
6. 원 Nature 연구는 temperature 0, 당시 OpenAI GPT 계열, 수동 검토를 썼다. 현재 노트북은
   Qwen native thinking switch, Qwen 권장 sampling, 자동 정답/함정 분류를 쓰므로 strict
   replication이 아니라 모델군·내부 분석 방법을 확장한 연구로 보고한다.

## 라이선스 & 인용

- `hagendorff_*`: 논문/부속자료 CC BY 4.0. OSF node에는 별도 node-level license 표시가
  없다. 결과 발표 시 원 논문(DOI 10.1038/s43588-023-00527-x)과 OSF 자료를 모두 인용한다.
- `crt_pilot`: 저장소 자체 작성(고전 CRT 문항은 공개 상식). 벤치마크로 인용하지 않는다.
- `crt2`: CC BY 3.0(Thomson & Oppenheimer 2016).
- `verbal_crt`: CC BY 4.0(Sirota et al. 2021, OSF xehbv).
- `crt7_classic`: **오픈 데이터 라이선스 없음.** Frederick(2005)·Toplak et al.(2014)
  stimuli를 학술적 fair use로 전사. 재배포 정책이 엄격한 곳에 공개하기 전 검토 필요.
- `yax_crt_isomorph`: 논문 CC BY 4.0. 단, 동반 코드/응답 repo는 **GPL-3.0**이므로 그
  repo 파일(`.npy`, `GPTReasoning.csv`)은 이 저장소에 포함하지 않는다. 공개 stimuli만 인용.
- 각 JSON의 `source.license` / `source.license_note` 필드에 항목별 라이선스가 들어 있다.
