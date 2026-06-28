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
- `src/mindscopex_analysis/generation.py`: Qwen CRT generation, response parsing, and JSON persistence.
- `src/mindscopex_analysis/activations.py`: residual stream capture with `lm.trace(...).save()`.
- `src/mindscopex_analysis/qwen_scope.py`: Qwen-Scope SAE loading, TopK feature extraction, and a first-pass layer ranking helper.
- `src/mindscopex_analysis/effects.py`: answer logprob margins and SAE decoder-direction ablation.
- `src/mindscopex_analysis/cases.py`: lure/control prompt cases.
- `src/mindscopex_analysis/workflows.py`: reusable notebook-level experiment loops.

The default pair is:

- Model: `Qwen/Qwen3-1.7B-Base`
- SAE repo: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`

That is the smallest currently available Qwen3 + Qwen-Scope pairing in the public collection.

## Recommended Research Target

- Primary behavioral model: `Qwen/Qwen3-8B`, comparing `enable_thinking=False` and `True`
  within the same post-trained checkpoint.
- Primary SAE candidate: `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`, transferred to the
  post-trained 8B checkpoint only after measuring SAE reconstruction quality there.
- Low-cost pilot: the existing Qwen3-1.7B-Base pair remains useful for validating the
  intervention code path.
- Format stress test: Qwen3-0.6B is optional and should not be pooled into the main result.
- Future extension: Qwen3.5-27B has an official checkpoint-matched Qwen-Scope SAE, but its
  current software and single-L40 requirements make it a second-stage target.

## Checks

```powershell
make lint
make test
make smoke
```

`make test` runs the unit tests directly; `make smoke` byte-compiles `src`/`tests` and
then runs the same suite, so it doubles as a fast pre-commit check.
