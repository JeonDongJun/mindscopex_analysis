# MindScopeX Analysis — 뉴런·레이어 활성화·SAE 연구

Claude / 공동 작업자용 저장소 맥락입니다. GitHub에 푸시한 뒤 Colab·로컬에서 `git clone` 으로 동일한 소스를 쓰는 흐름을 전제로 합니다.

## 이 레포가 하는 일

- **Transformer 레이어·뉴런 활성화** 캡처 및 시각화 (Plotly).
- **페르소나·프롬프트 조건** 비교 (`configs/`, Hydra CLI).
- **커스텀 Sparse Autoencoder** 학습 (`src/mindscopex_analysis/sae/`).
- 노트북: `notebooks/01_qwen_neuron_comparison.ipynb` — Qwen 등 프리셋·YAML로 실험.

## 디렉터리 요약

| 경로 | 설명 |
|------|------|
| `src/mindscopex_analysis/` | 패키지 본체 (`capture`, `pipeline`, `sae`, `neurons`, `notebook_presets` …) |
| `configs/` | Hydra 설정, `notebook_neuron_compare.yaml` (노트북 선택 병합) |
| `notebooks/` | 실험 노트북 (맨 위 셀에서 GitHub clone 가능) |
| `scripts/` | `run_experiment.py` 등 CLI 보조 |
| `pyproject.toml` | `uv` / `pip install -e .` 용 메타데이터 (프로젝트명: `mindscopex_analysis`) |

## 로컬 개발

- Python 3.11–3.12 권장. `uv sync` 또는 `pip install -e ".[dev]"` 로 편집 가능 설치.
- CLI: `uv run mindscopex-run` — Makefile의 `make run`, `make run-qwen` 참고.

## Colab / 원격 커널

1. 노트북 **첫 코드 셀**에서 `REPO_URL` 을 본인 GitHub 주소로 바꾼 뒤 실행 → 저장소가 클론되고 `MINDSCOPEX_ROOT` 가 설정됩니다.
2. 이어서 **import 셀**을 실행하면 `mindscopex_analysis` 패키지가 로드됩니다.
3. 의존성: 클론 셀에서 `pip install -e .` 및 `torch`, `transformers` 등이 설치됩니다. 무거운 모델은 GPU 런타임 권장.
4. 구버전 호환: 환경 변수 `PERSONA_INTERP_ROOT` 도 일부 코드에서 여전히 인식합니다.

## 공동 작업 규칙 (권장)

- 실험 결과는 `outputs/`·`multirun/` 에 쌓이며 기본 `.gitignore` 대상 — 커밋하지 않습니다.
- 프리셋 추가는 `src/mindscopex_analysis/notebook_presets.py` 의 `MODEL_PRESETS`.
- 노트북 전용 설정 덮어쓰기는 `configs/notebook_neuron_compare.yaml`.

## 관련 파일

- `CLAUDE.md` (본 문서) — AI·인간 협업 맥락
- 노트북 상단 — clone 셀 + 실험 설명
