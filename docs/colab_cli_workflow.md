# Colab CLI 실행 결과 회수 가이드

확인 기준일: **2026-07-05**

## 결론

공식 Google Colab CLI로 로컬 노트북이나 Python 파일을 Colab GPU에서 실행하고 결과를
다시 로컬에 저장할 수 있다.

- `colab exec -f notebook.ipynb`: 로컬 노트북을 원격 kernel에서 실행하고 출력이 포함된
  `*_output.ipynb`를 로컬에 만든다.
- `colab download REMOTE LOCAL`: 원격 VM의 JSON, Markdown, 모델 파일 등을 로컬로 받는다.
- `colab log -o FILE`: 세션 실행 기록을 `.ipynb`, `.md`, `.txt`, `.jsonl`로 내보낸다.
- `colab run`: 새 runtime 생성, 스크립트 실행, 결과 회수, runtime 종료를 한 명령 흐름으로
  자동화할 때 사용한다.

공식 CLI는 현재 **Linux와 macOS만 지원**한다. 이 저장소의 로컬 환경이 Windows이므로
WSL2 안에 CLI를 설치하고, 저장소를 WSL 경로에서 실행하는 구성을 권장한다.

## 설치

WSL2 터미널에서 다음을 실행한다.

```bash
uv tool install google-colab-cli
colab version
colab new -s mindscopex --gpu A100
colab status -s mindscopex
```

첫 API 요청에서 Google 인증 절차가 진행된다. 사용할 수 있는 accelerator는 Colab 구독과
잔여 compute unit에 따라 달라지며, 요청한 A100이 항상 할당된다고 가정하면 안 된다.

## 노트북 실행

저장소 루트에서 다음처럼 실행한다.

```bash
colab exec -s mindscopex -f notebooks/00_qwen_crt_text_responses.ipynb
colab exec -s mindscopex -f notebooks/03_layer_sweep_feature_search.ipynb
colab exec -s mindscopex -f notebooks/04_coefficient_dose_response.ipynb
```

각 노트북의 첫 셀이 remote `/content/mindscopex_analysis`를 clone 또는 fast-forward하고
필요한 Qwen3.5 Transformers revision과 현재 package를 설치한다. 따라서 main에 최신 코드가
push된 뒤 실행해야 한다.

`colab exec`의 노트북 실행 결과는 로컬 `*_output.ipynb`로 회수된다. 이 파일은 실험
검토용으로 `results/notebooks/` 아래에 옮겨 보관하는 것을 권장한다. `results/`와 notebook
output은 Git에서 제외되므로 원시 결과가 실수로 push되지 않는다.

## 구조화된 산출물 회수

`00`은 remote repository의 `outputs/` 아래에 JSON과 Markdown을 저장하고, `02`와 `03`은
모델 프로필별 feature handle JSON을 저장한다. 개별 파일은 다음처럼 받는다.

```bash
mkdir -p results/00 results/candidates

colab download -s mindscopex \
  /content/mindscopex_analysis/outputs/00_qwen_crt_text_responses_nature_smoke_qwen_native_seeds-42.json \
  results/00/nature_smoke_qwen_native_seed42.json

colab download -s mindscopex \
  /content/mindscopex_analysis/outputs/00_qwen_crt_text_responses_nature_smoke_qwen_native_seeds-42.md \
  results/00/nature_smoke_qwen_native_seed42.md

colab download -s mindscopex \
  /content/mindscopex_analysis/outputs/candidates/bat_ball_top_feature_answer_instruction_27b.json \
  results/candidates/bat_ball_top_feature_answer_instruction_27b.json
```

여러 파일을 한 번에 회수하려면 remote에서 archive를 만든 뒤 내려받는다.

```bash
cat <<'PY' | colab exec -s mindscopex
from shutil import make_archive
make_archive(
    "/content/mindscopex_outputs",
    "zip",
    "/content/mindscopex_analysis/outputs",
)
PY

colab download -s mindscopex \
  /content/mindscopex_outputs.zip \
  results/mindscopex_outputs.zip
```

## 실행 기록과 종료

```bash
mkdir -p results/logs
colab log -s mindscopex -o results/logs/session.ipynb
colab log -s mindscopex -o results/logs/session.md
colab stop -s mindscopex
```

runtime filesystem은 임시 저장소다. `download` 또는 `log`가 끝나기 전에 `colab stop`을
실행하지 않는다. 오래 걸리는 실험은 중간 결과를 `outputs/`에 주기적으로 저장하고 중요한
checkpoint는 Google Drive에도 복제하는 편이 안전하다.

## 이 저장소에서의 권장 운영

1. 로컬에서 코드와 노트북을 수정하고 main에 push한다.
2. WSL2에서 이름이 있는 Colab session을 생성한다.
3. `00`으로 behavior baseline을 실행하고 JSON/Markdown을 회수한다.
4. 같은 `ANALYSIS_PROFILE_KEY`로 `02` 또는 `03`을 실행해 feature handle을 만든다.
5. `04`~`10`, `13`을 같은 session에서 실행해 모델 재다운로드를 줄인다.
6. `11`, `12`를 보조 분석으로 실행한다.
7. `*_output.ipynb`, `colab log`, `outputs/` archive를 모두 로컬 `results/`에 보관한다.
8. 회수가 끝난 뒤 session을 종료한다.

## 공식 자료

- [Google Developers Blog: Introducing the Google Colab CLI](https://developers.googleblog.com/introducing-the-google-colab-cli/)
- [googlecolab/google-colab-cli](https://github.com/googlecolab/google-colab-cli)
