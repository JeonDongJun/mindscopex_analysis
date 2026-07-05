# Qwen-Scope SAE 지원 현황

확인 기준일: **2026-07-05**

## 확인 범위

이 문서는 Hugging Face의 [Qwen-Scope 공식 컬렉션](https://huggingface.co/collections/Qwen/qwen-scope)과
각 SAE model card를 기준으로 작성했다. `Qwen/*` 공식 namespace에 있고 컬렉션에 포함된
SAE만 집계했으며, 커뮤니티가 별도로 학습한 SAE는 제외했다.

현재 공식 공개 범위는 **Qwen3/Qwen3.5의 7개 모델 변형, 14개 SAE 저장소**다. 모두
transformer block 뒤 residual stream을 hook point로 사용하는 TopK SAE이며, 해당 모델의
모든 레이어에 `layer{n}.sae.pt` 체크포인트를 제공한다.

Hugging Face Models API에서 `author=Qwen`, `search=SAE-Res` 조건으로도 대조했으며 공식
`Qwen/SAE-Res-*` 저장소는 동일하게 14개였다. 각 저장소의 API `lastModified`는
2026-05-13(UTC), 컬렉션 페이지의 최종 갱신 표시는 2026-05-14다.

## Qwen3.5-27B 결론

`Qwen/Qwen3.5-27B`에는 checkpoint가 직접 일치하는 공식 SAE가 있다.

- 모델: [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)
- Top-K 50: [Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50)
- Top-K 100: [Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100)
- SAE width: 81,920
- residual hidden size: 5,120
- 레이어: 0-63, 총 64개 전부

여기서 SAE model card의 `Base model` 표시는 Hugging Face의 parent-model 관계를 뜻한다.
SAE가 연결된 실제 checkpoint는 이름에 `-Base`가 없는 post-trained
`Qwen/Qwen3.5-27B`다. 따라서 Qwen3/Qwen3.5의 Base SAE를 post-trained 모델에 옮겨 쓰는
경우와 달리, 27B 조합은 SAE 학습 대상과 분석 대상 checkpoint가 직접 일치한다.

## 저장소에서 선택한 Qwen3.5 모델군

행동 비교는 공식 post-trained 4종을 모두 사용한다. 내부 분석에서는 checkpoint가 직접
일치하는 공식 SAE를 우선하므로, 2B/9B/35B-A3B는 대응 Base checkpoint를 사용한다.

| 프로필 | 행동 모델 (`00`) | 내부 분석 모델 (`01`-`13`) | K50 SAE | exact behavior match |
|---|---|---|---|---|
| `2b` | `Qwen3.5-2B` | `Qwen3.5-2B-Base` | W32K | 아니오 |
| `9b` | `Qwen3.5-9B` | `Qwen3.5-9B-Base` | W64K | 아니오 |
| `27b` | `Qwen3.5-27B` | `Qwen3.5-27B` | W80K | **예** |
| `35b-a3b` | `Qwen3.5-35B-A3B` | `Qwen3.5-35B-A3B-Base` | W32K | 아니오 |

기본 프로필은 `27b`다. 노트북의 `ANALYSIS_PROFILE_KEY` 하나를 바꾸면 분석 모델, SAE,
layer 수, scan layer, 모델별 feature cache가 함께 바뀐다.

## 공식 SAE 전체 목록

| 계열 | SAE 학습 대상 모델 | checkpoint 단계 | d_model | 레이어 | 공개 SAE |
|---|---|---|---:|---:|---|
| Qwen3 | `Qwen3-1.7B-Base` | Base | 2,048 | 0-27 (28) | [W32K/K50](https://huggingface.co/Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50), [W32K/K100](https://huggingface.co/Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_100) |
| Qwen3 | `Qwen3-8B-Base` | Base | 4,096 | 0-35 (36) | [W64K/K50](https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50), [W64K/K100](https://huggingface.co/Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100) |
| Qwen3 | `Qwen3-30B-A3B-Base` | Base, MoE | 2,048 | 0-47 (48) | [W32K/K50](https://huggingface.co/Qwen/SAE-Res-Qwen3-30B-A3B-Base-W32K-L0_50), [W128K/K100](https://huggingface.co/Qwen/SAE-Res-Qwen3-30B-A3B-Base-W128K-L0_100) |
| Qwen3.5 | `Qwen3.5-2B-Base` | Base | 2,048 | 0-23 (24) | [W32K/K50](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50), [W32K/K100](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100) |
| Qwen3.5 | `Qwen3.5-9B-Base` | Base | 4,096 | 0-31 (32) | [W64K/K50](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50), [W64K/K100](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100) |
| Qwen3.5 | `Qwen3.5-35B-A3B-Base` | Base, MoE | 2,048 | 0-39 (40) | [W32K/K50](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W32K-L0_50), [W128K/K100](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-W128K-L0_100) |
| Qwen3.5 | `Qwen3.5-27B` | **Post-trained, dense** | 5,120 | 0-63 (64) | [W80K/K50](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50), [W80K/K100](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_100) |

`W32K`, `W64K`, `W80K`, `W128K`는 각각 SAE feature dictionary 크기
32,768, 65,536, 81,920, 131,072를 뜻한다. `K50`과 `K100`은 토큰·레이어별로
남기는 non-zero feature 수다. 처음 lure feature 후보를 탐색할 때는 더 희소하고 결과를
읽기 쉬운 K50을 기본으로 사용하고, K100은 reconstruction과 feature 안정성 검증용으로
비교하는 편이 낫다.

## 아직 공식 SAE가 없는 범위

2026-07-05 현재 Qwen-Scope 컬렉션에는 다음 checkpoint와 직접 일치하는 공식 SAE가 없다.

- Qwen3 post-trained 모델 전체: `0.6B`, `1.7B`, `4B`, `8B`, `14B`, `30B-A3B`,
  `32B`, `235B-A22B`와 2507 Instruct/Thinking 변형
- Qwen3 Base 중 `0.6B`, `4B`, `14B`, `32B`, `235B-A22B`
- Qwen3.5 post-trained 중 `0.8B`, `2B`, `4B`, `9B`, `35B-A3B`,
  `122B-A10B`, `397B-A17B`
- Qwen3.5 Base 중 `0.8B`, `4B` 및 더 큰 `122B/397B` 계열
- Qwen3.6와 Qwen3-VL/Omni/Coder 전용 checkpoint
- FP8, GPTQ, AWQ 같은 양자화 checkpoint 전용 SAE

Qwen-Scope model card는 Base SAE를 대응하는 post-trained checkpoint 내부 분석에 쓰는
것이 많은 상황에서 합리적이라고 설명한다. 다만 이 경우 exact match가 아니므로
reconstruction error, explained variance, feature activation 분포와 steering dose response를
먼저 검증해야 한다. 양자화 모델에도 full-precision SAE를 자동으로 호환된다고 가정하지
않는다.

## 이 연구의 권장 선택

### 주 실험: Qwen3.5-27B + W80K/K50

현재 연구 질문에는 다음 조합이 가장 직접적이다.

- behavior와 SAE가 같은 post-trained checkpoint를 사용한다.
- 같은 모델에서 thinking/non-thinking을 비교할 수 있다.
- 64개 전 레이어 SAE가 있어 특정 레이어를 미리 고정하지 않고 탐색할 수 있다.
- K50은 한 토큰에서 50개 feature만 활성화하므로 초기 후보 순위와 ablation이 비교적
  다루기 쉽다.

Qwen3.5는 기존 Qwen3와 다른 multimodal hybrid architecture다. 공식 Transformers 구현을
확인해 NNsight block 경로를 `model.language_model.layers.{layer}`로 변경했으며 dense와 MoE
프로필 모두 같은 language-model 경로를 사용한다. 실제 A100 가중치 실행은 별도로 다음
smoke test를 통과해야 한다.

1. 모델의 실제 transformer block 경로와 총 64개 레이어를 확인한다.
2. 한 레이어 residual 마지막 차원이 5,120인지 확인한다.
3. 해당 레이어 SAE encode/decode shape가 일치하는지 확인한다.
4. 원 residual 대비 SAE reconstruction error와 explained variance를 기록한다.
5. zero coefficient가 baseline logits와 정확히 같고 작은 coefficient에서 결과가 연속적으로
   변하는지 확인한다.

### 비용 절감 및 대조군

- 코드 검증: `Qwen3.5-2B-Base + W32K/K50`
- 중간 규모 exact Base 분석: `Qwen3.5-9B-Base + W64K/K50`
- MoE architecture 대조: `Qwen3.5-35B-A3B-Base + W32K/K50`
- 주 인과 분석: `Qwen3.5-27B + W80K/K50`

이 세 Base 조합은 각 SAE와 정확히 일치하므로 코드와 SAE reconstruction의 positive control로
사용할 수 있다. 다만 그 feature를 대응 post-trained 행동 모델의 feature라고 바로 부르지는
않는다.

## 출처와 갱신 규칙

- 공식 컬렉션: [Qwen-Scope](https://huggingface.co/collections/Qwen/qwen-scope)
- API 대조: [Qwen 공식 SAE-Res 검색](https://huggingface.co/api/models?author=Qwen&search=SAE-Res&limit=100&full=true)
- 기술 보고서: [Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models](https://huggingface.co/papers/2605.11887)
- Qwen3.5 모델 목록: [Qwen3.5 collection](https://huggingface.co/collections/Qwen/qwen35)
- 각 수치의 최종 근거: 위 표에 연결한 개별 SAE model card의 `Model Details`

Qwen-Scope 컬렉션 자체는 Hugging Face 표시상 2026-05-14에 갱신되었으며, 이 문서는
2026-07-05에 다시 확인했다. 새 SAE가 추가되면 “공식 SAE 전체 목록”과 “아직 공식 SAE가
없는 범위”를 함께 갱신한다.
