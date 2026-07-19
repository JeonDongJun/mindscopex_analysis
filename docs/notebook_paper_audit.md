# 노트북 ↔ 논문 실험 검토

작성 기준: `paper/paper_ko.qmd` / `paper/paper.qmd` 초안, `paper/_analysis.py` 그림 로더.

## 요약

논문은 그림 4개로 구성되고, 각 그림은 `paper/data/`의 CSV 1개를 읽습니다
(없으면 placeholder 표시). 즉 **논문에 실제로 필요한 산출물은 CSV 4개와 그 근거가 되는
feature handle 1개**입니다. 노트북 14개 중 이 4개 CSV를 직접 만드는 것은 5개(02·03·04·05·10)
뿐이고, 나머지는 서론의 행동 baseline(00) 또는 논의/한계 절을 뒷받침하는 보조 실험입니다.

| 논문 그림 (`_analysis.py`) | CSV | 필요 컬럼 | 만드는 노트북 |
|---|---|---|---|
| `fig_layer_sweep` (§finding) | `layer_sweep.csv` | `layer`, `margin_delta` | 03 (+02) |
| `fig_dose_response` (§effect) | `dose_response.csv` | `coefficient`, `margin` | 04 |
| `fig_intervention_modes` (§effect) | `intervention_modes.csv` | `mode`, `margin_delta` | 05 |
| `fig_crt_transfer` (§transfer) | `crt_transfer.csv` | `item`, `baseline_margin`, `ablated_margin` | 10 |

## 노트북별 판정과 통제 연구 대응

노트북(`notebooks/`)은 탐색 원본으로 남고, 실제 연구 실행은 통제 연구 job
(`research_experiments`)이 담당한다. 마지막 열은 각 노트북의 의도를 통제 연구가 어떻게
흡수하는지를 보여준다.

| NB | 실험 | 논문 위치 | 필요성 | 통제 연구(research_experiments) 대응 |
|----|------|-----------|--------|--------------------------------------|
| 00 | CRT 텍스트 응답 | §서론 (chat model 행동 baseline) | 필요 | `crt_text_responses` job (유지) |
| 01 | Activation MVP | 예비 layer 스캔 | 불필요 | `discover`의 layer localization이 대체 |
| 02 | bat-ball ablation | §finding (어느 feature) | 필요 | `discover` (train split 다문항 발견) |
| 03 | Layer sweep | §finding | 필요 | `discover` (localization + null) |
| 04 | Coefficient dose response | §effect | 참고 | `behavioral` 계수 sweep(생성). margin dose는 미채택 |
| 05 | Intervention mode 비교 | §effect | 참고 | 탐색 전용 (배치 미채택) |
| 06 | Control prompt 특이성 | §논의 특이성 | 필요 | `control_specificity` (E4) |
| 07 | Paraphrase 강건성 | §transfer | 보조 | held-out 문항 일반화로 대체 (탐색 전용) |
| 08 | Answer format 민감도 | §한계 | 보조 | 탐색 전용 |
| 09 | Token position sweep | §방법 | 보조 | 탐색 전용 |
| 10 | CRT transfer | §transfer | 필요 | `causal_heldout` (held-out split) |
| 11 | Control delta bypass | §한계-우회 | 보조 | 탐색 전용 |
| 12 | Decoder geometry | §논의 | 보조 | 탐색 전용 |
| 13 | Semantic/logic 특이성 | §transfer | 보조 | `[data].dataset` 교체(verbal_crt/crt2 등)로 대체 |

## 왜 통제 연구인가 — 단일-case 접근의 약점

노트북식 "한 문항에서 `margin_delta` 최대 feature 선택 후 그 값을 효과로 보고"에는 네 구멍이
있고, 통제 연구가 각각을 막는다:

- **순환성(selection-on-outcome)** → **random-direction null** 대비 z-score,
- **logprob ≠ 행동** → **free-generation 정답률** readout,
- **n=1 발견** → **train/test 분할**(datasets.md 원칙 4),
- **control 미활용** → **matched-control** hostile vs control 대조를 headline으로.

## 결론

- 통제 연구가 §finding(`discover`)·§effect(`behavioral`)·§transfer(`causal_heldout`)·특이성
  (`control_specificity`)·현상(`phenomenon`)을 담당한다. 설계·실행·해석은
  [study_design.md](study_design.md).
- **00**은 chat-model 행동 baseline으로 유지(`crt_text_responses`).
- **탐색 전용(04·05·08·09·11·12 등)** 은 별도 배치 job 없이 필요할 때 노트북에서 확인한다(축소).

> 모델: **Qwen3.5 분석 프로파일**(2B/9B/27B/35B-A3B, `SAE-Res-Qwen3.5-*`)로 확정. 논문 본문의
> `Qwen/Qwen3-1.7B-Base` + Qwen-Scope 서술은 마이그레이션 이전 표현이므로 실제 실행 대상
> (예: `Qwen/Qwen3.5-2B-Base`)에 맞춰 갱신 대상이다.
