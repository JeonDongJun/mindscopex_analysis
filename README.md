# MindScopeX Analysis

Focused Qwen interpretability experiments with NNsight and Qwen-Scope.

## Quick Start

```powershell
uv sync --extra dev
uv run nbstripout --install --attributes .gitattributes
uv run jupyter lab notebooks/
```

The `nbstripout` Git filter keeps notebook outputs and execution counts in the local working
copy while removing them from staged and committed notebook blobs. Run the filter-install
command once after each fresh clone; `make install` performs both setup steps as well.

Start with `notebooks/00_qwen_crt_text_responses.ipynb` to inspect model behavior,
then use `notebooks/01_qwen_scope_activation_mvp.ipynb` and
`notebooks/02_bat_ball_lure_feature_ablation.ipynb` for the bat-and-ball
feature ablation workflow.

All experiments use the same concise final-answer instruction. Notebook 00 passes it as a
chat system message; notebooks 01-13 prepend the same text to Base-model prompts so the
Qwen-Scope model/SAE pairing remains unchanged. Instructed feature handles use a separate
cache file to avoid mixing them with earlier unprompted discoveries.

## CRT Datasets

Notebook 00 supports two dataset modes:

- `DATASET_NAME = "pilot"`: the repository's 9-item smoke-test suite, loaded from
  `src/mindscopex_analysis/data/crt_pilot.json`.
- `DATASET_NAME = "nature_crt150"`: the 150 public CRT variants from Hagendorff,
  Fabi, and Kosinski (2023), downloaded from OSF with a pinned SHA-256 checksum.

The full Nature run contains 150 cases and produces 900 responses with the default three
models and two reasoning modes. Use `NATURE_LIMIT_PER_TYPE` and a smaller `model_ids` list
before launching the full benchmark. Dataset provenance, licensing notes, and the review of
newer CRT-related resources are documented in [docs/crt_datasets.md](docs/crt_datasets.md).
The official Qwen-Scope SAE coverage and checkpoint-matching notes are tracked in
[docs/qwen_scope_sae_catalog.md](docs/qwen_scope_sae_catalog.md).

Notebook 00 retries thinking-protocol failures and ambiguous `both` answers with new seeds while
retaining every attempt in the JSON output. It also writes a Markdown report with correct, lure,
and operational hallucination/other counts. Set `INCLUDE_27B_A100 = True` to add
[`Qwen/Qwen3.5-27B`](https://huggingface.co/Qwen/Qwen3.5-27B) on an A100 80GB runtime; keep its
results separate because Qwen3.5 uses a newer multimodal hybrid architecture.

## Experiment Notebooks

- `00_qwen_crt_text_responses.ipynb`: full CRT responses from Qwen models in thinking and non-thinking modes.
- `01_qwen_scope_activation_mvp.ipynb`: activation capture and first Qwen-Scope layer scan.
- `02_bat_ball_lure_feature_ablation.ipynb`: first bat-and-ball lure feature ablation.
- `03_layer_sweep_feature_search.ipynb`: layer-wise feature search.
- `04_coefficient_dose_response.ipynb`: removal/steering dose response.
- `05_intervention_mode_comparison.ipynb`: removal, suppression, amplification, projection removal.
- `06_control_prompt_specificity.ipynb`: matched control prompt specificity.
- `07_paraphrase_robustness.ipynb`: bat-and-ball paraphrase robustness.
- `08_answer_format_sensitivity.ipynb`: answer surface-form sensitivity.
- `09_token_position_sweep.ipynb`: token-position intervention sweep.
- `10_crt_transfer.ipynb`: transfer to other CRT lures.
- `11_control_delta_bypass.ipynb`: bypass with matched-control residual delta.
- `12_decoder_geometry.ipynb`: decoder-direction geometry among candidate features.
- `13_semantic_logic_specificity.ipynb`: specificity against semantic and logic lures.

## What Is Included

- `src/mindscopex_analysis/models.py`: Qwen + NNsight model loading helpers.
- `src/mindscopex_analysis/prompts.py`: shared final-answer instruction and Base-prompt helpers.
- `src/mindscopex_analysis/generation.py`: Qwen CRT generation, retries, classification, and JSON/Markdown persistence.
- `src/mindscopex_analysis/datasets.py`: checksum-pinned public CRT dataset download and parsing.
- `src/mindscopex_analysis/activations.py`: residual stream capture with `lm.trace(...).save()`.
- `src/mindscopex_analysis/qwen_scope.py`: Qwen-Scope SAE loading, TopK feature extraction, and a first-pass layer ranking helper.
- `src/mindscopex_analysis/effects.py`: answer logprob margins and SAE decoder-direction ablation.
- `src/mindscopex_analysis/cases.py`: validated pilot JSON loading plus experimental lure/control cases.
- `src/mindscopex_analysis/workflows.py`: reusable notebook-level experiment loops.

The default pair is:

- Model: `Qwen/Qwen3-1.7B-Base`
- SAE repo: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`

That is the smallest currently available Qwen3 + Qwen-Scope pairing in the public collection.

## Recommended Research Target

- Exact-checkpoint research target: `Qwen/Qwen3.5-27B` with
  `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50`. This official SAE covers all 64 residual layers
  of the same post-trained checkpoint, so it is the cleanest thinking/non-thinking comparison.
- Lower-cost behavioral target: `Qwen/Qwen3-8B`, comparing `enable_thinking=False` and `True`.
  Its official SAE was trained on `Qwen3-8B-Base`, so transfer requires reconstruction checks.
- Exact Base control: `Qwen/Qwen3-8B-Base` with
  `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`.
- Low-cost pilot: the existing Qwen3-1.7B-Base pair remains useful for validating the
  intervention code path.
- Format stress test: Qwen3-0.6B is optional and should not be pooled into the main result.
- Large-model behavior extension: Qwen3.5-27B can be enabled for A100 80GB runs in notebook 00.
  It requires a current Transformers build with Qwen3.5 support. The NNsight residual hook path
  must be smoke-tested before using the 27B SAE intervention results.

## Checks

```powershell
make lint
make test
make smoke
```

`make test` runs the unit tests directly; `make smoke` byte-compiles `src`/`tests` and
then runs the same suite, so it doubles as a fast pre-commit check.
