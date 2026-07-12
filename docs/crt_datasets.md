# CRT Dataset Notes

> **참고:** 정규화된 전체 데이터셋 카탈로그(공통 JSON 스키마, `load_lure_dataset`
> 로더, 항목별 통계·출처·라이선스·example)는 이제 **[datasets.md](datasets.md)**가
> 정본이다. 이 문서는 `nature_crt150` 런타임 다운로드 경로(`datasets.py`)에 대한
> 기존 설계 메모로 남겨 둔다. 신규 작업은 `datasets.md`와
> `scripts/build_datasets.py`를 따른다.

검토 기준일: 2026-07-05

## 저장소 pilot 세트

빠른 파이프라인 점검용 9문항은
`src/mindscopex_analysis/data/crt_pilot.json`에서 관리한다. `load_pilot_crt_cases()`가
dataset ID, 필수 문자열, 중복 case ID, 정답과 함정 답의 충돌을 검증한다. 이 세트는
정식 벤치마크가 아니라 생성, 응답 파싱, 재시도, 시각화가 정상 작동하는지 확인하는
smoke test다.

## 채택: Nature CRT-150

Hagendorff, Fabi, Kosinski (2023)는 세 가지 CRT 구조를 각각 50개씩 변형한
150문항을 사용했다.

- CRT1: bat-and-ball 형태의 가격 차이 방정식
- CRT2: machines-and-widgets 형태의 작업률 문제
- CRT3: lily-pads 형태의 지수 성장 문제

논문: [Nature Computational Science](https://doi.org/10.1038/s43588-023-00527-x)

공개 자료: [OSF w5vhp](https://osf.io/w5vhp/)

이 프로젝트는 OSF의 `LLMs_responses.py`를 직접 실행하지 않는다. 다음 주소에서 원본을
다운로드하고 SHA-256을 검증한 뒤, Python AST의 literal assignment인 `crt1`, `crt2`,
`crt3`만 읽는다.

- Source URL: `https://osf.io/download/z6kmw/`
- SHA-256: `cdf4617e8dec63546762cbe2b3cae6b6c7f640adfb6002bf5fc226f5871a4125`
- 항목 수 검증: CRT1 50개, CRT2 50개, CRT3 50개
- 각 항목 검증: 연속된 번호, 비어 있지 않은 prompt, correct answer, intuitive answer

```python
from mindscopex_analysis import nature_crt150_cases

# 빠른 유형별 3문항 점검
cases = nature_crt150_cases(limit_per_type=3)

# 전체 150문항
cases = nature_crt150_cases(limit_per_type=None)
```

노트북 `00`에서는 직접 loader 인자를 바꾸기보다 `RUN_PRESET="nature_smoke"`로 유형별
3문항을 먼저 점검한 뒤 `RUN_PRESET="nature_full"`로 150문항을 실행한다. 결과는
`nature_crt_difference`, `nature_crt_rate`, `nature_crt_growth`별로 분리 집계한다.

`prompt_style="task_only"`는 공개 task 문장을 그대로 사용한다. 과거 completion 모델과
같이 `Question:` 및 `Answer:` 경계를 넣으려면 `prompt_style="question_answer"`를 사용한다.

논문 본문과 부속 자료는 CC BY 4.0으로 공개되어 있지만, OSF 프로젝트 자체에는 별도의
node-level license가 표시되어 있지 않다. 이 저장소는 원문을 재배포하지 않고 실행 시
다운로드하며, 결과를 발표할 때 원 논문과 OSF 자료를 모두 인용한다.

## 최신 직접 관련 자료

### Yax et al. (2024)

[Studying and improving reasoning in humans and machines](https://doi.org/10.1038/s44271-024-00091-8)는
고전 CRT-7과 같은 구조를 유지하면서 인물, 행동, 수치를 바꾼 새로운 7문항을 만들었다.
순수 수학식 control, reasoning prompt, solved-example 조건과 인간 응답도 포함한다.

- 상태: peer-reviewed, open access
- 자료: [ReasoningGPT](https://github.com/hrl-team/ReasoningGPT)
- 저장소 라이선스: GPL-3.0
- 권장 용도: Nature CRT-150과 다른 생성 절차를 가진 작은 외부 검증 세트

문항 수가 7개라 주 벤치마크를 대체하기에는 작지만, vignette와 동일 계산의 수학 control을
짝지었다는 점은 lure feature의 문제 이해 효과와 계산 능력 효과를 분리하는 데 유용하다.

### Xie et al. (2024)

[Do Large Language Models Truly Grasp Mathematics?](https://arxiv.org/abs/2410.14979)는
고전 CRT를 목표 지향적으로 변형했을 때 최신 LLM의 정확도가 크게 떨어질 수 있음을
보고한다. 2026-06-29 기준 공식 논문 페이지에서 재사용 가능한 코드나 dataset 링크를
확인하지 못했으므로 자동 loader 대상에는 포함하지 않는다.

### LLMThinkBench (2026 revision)

[LLMThinkBench](https://arxiv.org/abs/2507.04023)는 14개 기초 수학 task를 동적으로
생성하고 accuracy와 token efficiency를 함께 측정한다. ACL 2026 Findings 채택 논문이지만
직관적 오답을 설계한 CRT dataset은 아니다. thinking이 불필요하게 길어지는지 확인하는
인접 robustness benchmark로만 분류한다.

## 연구 사용 원칙

1. 현재 저장소의 9문항 pilot은 코드와 출력 형식 점검에만 사용한다.
2. Nature CRT-150은 기존 연구와 비교하는 외부 행동 벤치마크로 사용한다.
3. 공개 문항의 pretraining 노출 가능성을 배제할 수 없으므로 Nature 결과만으로 새로운
   reasoning 일반화를 주장하지 않는다.
4. feature 탐색과 계수 선택은 discovery split에서만 수행하고, held-out item에 그대로
   적용한다.
5. 최종 인과 주장은 공개 benchmark와 별도로 만든 미공개 수치·표면형 변형에서 재검증한다.
6. 유형, item, paraphrase, sampling seed를 구분해 저장하고 paraphrase나 seed를 독립적인
   문제 표본처럼 계산하지 않는다.

## 원 논문과 현재 실험의 차이

Nature 연구는 temperature 0, 당시 OpenAI GPT 계열, 수동 응답 검토를 사용했다. 현재
노트북은 Qwen3.5의 native thinking switch, Qwen 권장 sampling, 자동 정답/함정 문자열
분류를 사용한다. 따라서 같은 150문항을 쓰더라도 strict replication이 아니라 모델군과
내부 분석 방법을 확장한 연구로 보고해야 한다.
