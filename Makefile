.PHONY: install install-all run run-qwen notebook lab lint format test clean help

help:  ## 도움말
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## 기본 의존성 설치
	uv sync

install-all:  ## 전체 의존성 설치 (SAE + dev/jupyter)
	uv sync --extra sae --extra dev

# ── 실험 실행 ────────────────────────────────────────────
run:  ## GPT-2 페르소나 비교 실행
	uv run mindscopex-run

run-qwen:  ## Qwen 뉴런 활성화 비교 실행
	uv run mindscopex-run experiment=qwen_neuron_compare

run-custom:  ## 커스텀 오버라이드 (예: make run-custom ARGS="model.device=cuda")
	uv run mindscopex-run $(ARGS)

# ── 노트북 ───────────────────────────────────────────────
notebook:  ## Jupyter Notebook 열기
	uv run jupyter notebook notebooks/

lab:  ## JupyterLab 열기
	uv run jupyter lab notebooks/

# ── 품질 ─────────────────────────────────────────────────
lint:  ## ruff 린트
	uv run ruff check src/

format:  ## ruff 포맷
	uv run ruff format src/

test:  ## pytest 실행
	uv run pytest

# ── 정리 ─────────────────────────────────────────────────
clean:  ## outputs/ 및 캐시 삭제
	uv run python -c "import shutil,pathlib; [shutil.rmtree(p,True) for p in map(pathlib.Path,['outputs','multirun','__pycache__'])]"
