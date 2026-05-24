.PHONY: install notebook lab lint format smoke clean help

help:  ## 도움말
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## 기본 + 노트북 의존성 설치
	uv sync --extra dev

notebook:  ## Jupyter Notebook 열기
	uv run jupyter notebook notebooks/

lab:  ## JupyterLab 열기
	uv run jupyter lab notebooks/

lint:  ## ruff 린트
	uv run ruff check src/

format:  ## ruff 포맷
	uv run ruff format src/

smoke:  ## import/문법 확인
	uv run python -m compileall src

clean:  ## 캐시 삭제
	uv run python -c "import shutil,pathlib; [shutil.rmtree(p,True) for p in map(pathlib.Path,['.ruff_cache','__pycache__'])]"
