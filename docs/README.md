# 문서 안내

문서는 목적별로 아래 순서로 읽으면 된다.

## 시작과 구조

| 문서 | 역할 |
|---|---|
| [../README.md](../README.md) | 설치, 주요 진입점, 저장소 개요 |
| [architecture.md](architecture.md) | 모듈 경계, 의존 방향, 데이터·실행 흐름의 정본 |
| [../experiments/README.md](../experiments/README.md) | TOML/Colab 배치 실행과 산출물 |

## 연구 설계와 해석

| 문서 | 역할 |
|---|---|
| [study_design.md](study_design.md) | 통제 연구 E1–E4와 behavioral 검증 |
| [datasets.md](datasets.md) | 데이터셋 스키마, 출처, 감사, 선택 원칙의 정본 |
| [metrics_guide.md](metrics_guide.md) | logprob, margin, ablation 지표 해석 |
| [random_direction_null.md](random_direction_null.md) | 현재 null의 범위와 selection-adjusted 확장 |
| [qwen_scope_sae_catalog.md](qwen_scope_sae_catalog.md) | model/SAE checkpoint 호환성 |

## 탐색용 노트북

| 문서 | 역할 |
|---|---|
| [notebook_pipeline.md](notebook_pipeline.md) | notebook 00–13의 순서와 cache 의존성 |
| [notebook_paper_audit.md](notebook_paper_audit.md) | 탐색 notebook과 논문 근거의 차이 |

노트북은 탐색·설명용이고, 논문용 통제 실험의 실행 정본은 `experiments/configs`와
`experiments/jobs/research_experiments.py`다.

## 운영과 참고

| 문서 | 역할 |
|---|---|
| [colab_cli_workflow.md](colab_cli_workflow.md) | Colab CLI 설치·실행·결과 회수 |
| [literature/README.md](literature/README.md) | 문헌 노트 규칙과 색인 |

문서가 충돌하면 데이터는 `datasets.md`, 연구 단계는 `study_design.md`, 코드 경계는
`architecture.md`, 실제 실행 옵션은 CLI `--help`와 TOML config를 우선한다.
