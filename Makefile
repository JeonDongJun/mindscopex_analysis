.PHONY: install git-filters notebook lab lint format test smoke clean help paper paper-preview paper-pdf

help:  ## 도움말
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## 기본 + 노트북 의존성 설치
	uv sync --extra dev
	uv run nbstripout --install --attributes .gitattributes

git-filters:  ## Git 커밋에서 노트북 출력 제외
	uv run nbstripout --install --attributes .gitattributes

notebook:  ## Jupyter Notebook 열기
	uv run jupyter notebook notebooks/

lab:  ## JupyterLab 열기
	uv run jupyter lab notebooks/

lint:  ## ruff 린트
	uv run ruff check src/ tests/

format:  ## ruff 포맷
	uv run ruff format src/ tests/

test:  ## 단위 테스트 실행
	uv run python -m unittest discover -s tests -v

smoke:  ## import/문법 확인 + 단위 테스트
	uv run python -m compileall src tests
	uv run python -m unittest discover -s tests

paper:  ## Quarto 논문 HTML 렌더
	uv run quarto render paper/paper.qmd

paper-preview:  ## Quarto 논문 라이브 프리뷰
	uv run quarto preview paper/paper.qmd

paper-pdf:  ## Quarto 논문 PDF 렌더 (tinytex 필요)
	uv run quarto render paper/paper.qmd --to pdf

clean:  ## 캐시 삭제
	uv run python -c "import shutil,pathlib; [shutil.rmtree(p,True) for p in map(pathlib.Path,['.ruff_cache','__pycache__'])]"
