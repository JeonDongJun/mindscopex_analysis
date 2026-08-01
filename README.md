# MindScopeX Analysis

Qwen3.5와 Qwen-Scope SAE를 이용해 reasoning lure를 행동·activation·인과 개입
관점에서 분석하는 연구 코드입니다. 탐색용 노트북과 재현 가능한 TOML 기반 배치 실험을
같은 `mindscopex_analysis` 코어 위에서 실행합니다.

## 빠른 시작

Python 3.11–3.13과 [`uv`](https://docs.astral.sh/uv/)가 필요합니다.

```powershell
uv sync --extra dev
uv run nbstripout --install --attributes .gitattributes
make smoke
```

노트북을 열려면:

```powershell
uv run jupyter lab notebooks/
```

`nbstripout` 필터는 로컬 출력은 유지하면서 커밋되는 노트북의 출력과 execution count를
제거합니다. 새 clone마다 한 번 설치하면 됩니다.

## 실행 경로

목적에 따라 진입점을 고릅니다.

| 목적 | 진입점 | 설명 |
|---|---|---|
| 가장 빠른 로컬 검증 | `make smoke` | 전체 Python 문법 검사와 단위 테스트 |
| 데이터셋 검증 | `uv run python scripts/audit_datasets.py` | 스키마·중복·메타데이터 감사 |
| 탐색과 시각적 확인 | `notebooks/00`–`13` | 행동 기준선부터 feature 기하까지 |
| 재현 가능한 원격 실험 | `experiments/run_colab.sh` | TOML config/suite를 Colab에서 실행 |
| 결과 통합 | `make analyze-crt` | `results/runs`의 CRT summary를 병합 |
| 논문 렌더 | `make paper` / `make paper-ko` | Quarto HTML 출력 |

첫 Colab smoke run:

```bash
./experiments/run_colab.sh experiments/suites/smoke.toml --session mindscopex-smoke
```

통제 연구 전체 실행과 accelerator 전환 규칙은
[`experiments/README.md`](experiments/README.md)를 참고하세요.

## 저장소 구조

```text
src/mindscopex_analysis/   재사용 가능한 모델·SAE·채점·연구 로직
src/mindscopex_analysis/data/
                           checksum/provenance가 포함된 정규화 JSON 데이터
experiments/               TOML config/suite, Colab launcher, 배치 job
notebooks/                 00–13 탐색 파이프라인
scripts/                   데이터 구축·감사·결과 분석·API 평가
tests/                     네트워크와 대형 모델 없이 실행되는 단위 테스트
docs/                      설계, 데이터, 지표, 실행 가이드
paper/                     Quarto 논문과 figure/table 입력
results/                   로컬/원격 실행 산출물(버전 관리 제외)
```

모듈 경계와 데이터 흐름은 [`docs/architecture.md`](docs/architecture.md), 문서 전체
목록과 정본 관계는 [`docs/README.md`](docs/README.md)에 정리되어 있습니다.

## 데이터와 연구 설계

정규화 카탈로그에는 현재 10개 데이터셋, 657개 case가 있습니다. 주 통제 연구는
150개 Hagendorff CRT와 matched control을 사용하며, discovery/held-out split,
random-direction null, control specificity, behavioral readout을 분리합니다.

- 데이터셋 스키마·출처·감사 결과: [`docs/datasets.md`](docs/datasets.md)
- 통제 연구의 단계와 해석: [`docs/study_design.md`](docs/study_design.md)
- 지표 정의: [`docs/metrics_guide.md`](docs/metrics_guide.md)
- Qwen-Scope checkpoint 호환성: [`docs/qwen_scope_sae_catalog.md`](docs/qwen_scope_sae_catalog.md)

기본 mechanistic profile은 정확히 대응하는 다음 pair입니다.

- model: `Qwen/Qwen3.5-27B`
- SAE: `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50`

비용 절감용 2B/9B와 MoE 35B-A3B profile도 제공하지만, 이들의 공식 SAE는 Base
checkpoint용입니다. post-trained behavior model의 feature로 해석하려면 별도의
reconstruction/transfer 검증이 필요합니다.

## 개발 검사

```powershell
make lint
make test
make smoke
```

`make lint`는 `src`, `tests`, `experiments`, `scripts`의 Python 코드를 검사합니다.
단위 테스트는 네트워크 요청이나 모델 다운로드 없이 실행됩니다.
