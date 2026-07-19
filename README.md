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

The full experiment order, cache dependencies, and evidence gates are documented in
[docs/notebook_pipeline.md](docs/notebook_pipeline.md).

All experiments use the same concise final-answer instruction. Notebook 00 passes it as a
chat system message; notebooks 01-13 prepend it to the analysis prompt. Feature handles are
cached per Qwen3.5 profile so feature IDs from different SAE dictionaries cannot be mixed.

## CRT Datasets

Notebook 00 supports three run presets:

- `RUN_PRESET = "pilot"`: the repository's 9-item smoke-test suite, loaded from
  `src/mindscopex_analysis/data/crt_pilot.json`.
- `RUN_PRESET = "nature_smoke"`: three items from each Nature CRT family.
- `RUN_PRESET = "nature_full"`: all 150 public CRT variants from Hagendorff, Fabi,
  and Kosinski (2023), downloaded from OSF with a pinned SHA-256 checksum.

The full Nature run contains 150 cases and produces 1,200 responses with the default four
models and two reasoning modes per seed. Run `nature_smoke` and optionally reduce
`MODEL_IDS_TO_RUN` before launching the full benchmark. Dataset provenance, licensing notes,
and the review of newer CRT-related resources are documented in
[docs/datasets.md](docs/datasets.md).
The official Qwen-Scope SAE coverage and checkpoint-matching notes are tracked in
[docs/qwen_scope_sae_catalog.md](docs/qwen_scope_sae_catalog.md).

Notebook 00 retries thinking-protocol failures and ambiguous `both` answers with new seeds while
retaining every attempt in the JSON output. It also writes a Markdown report with correct, lure,
and operational hallucination/other counts, plus family-level lure rates with Wilson 95%
intervals. `GENERATION_PROTOCOL` separates Qwen-native sampling from a no-system-prompt,
deterministic replication baseline. Multiple-seed inference should cluster by item rather than
treating repeated responses as independent. The default behavior suite is Qwen3.5 2B, 9B,
27B, and 35B-A3B, loaded sequentially. The first cell installs a pinned Transformers revision
with Qwen3.5 support when the runtime does not already provide it.

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

The default mechanistic pair is:

- Model: `Qwen/Qwen3.5-27B`
- SAE repo: `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50`

This is the only selected post-trained behavior checkpoint with a directly matching official
Qwen-Scope SAE. Set `ANALYSIS_PROFILE_KEY` in notebooks 01-13 to `2b`, `9b`, `27b`, or
`35b-a3b`; the matching analysis checkpoint, SAE, layer count, and scan layers change together.

## Recommended Research Target

- Exact-checkpoint research target: `Qwen/Qwen3.5-27B` with
  `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50`. This official SAE covers all 64 residual layers
  of the same post-trained checkpoint, so it is the cleanest thinking/non-thinking comparison.
- Lower-cost exact Base controls: `Qwen3.5-2B-Base` and `Qwen3.5-9B-Base` with their official
  W32K/W64K K50 SAEs.
- MoE Base control: `Qwen3.5-35B-A3B-Base` with the official W32K/K50 SAE.
- Behavior comparison: the corresponding post-trained 2B, 9B, 27B, and 35B-A3B checkpoints
  are all enabled in notebook 00.
- Do not describe a 2B, 9B, or 35B-A3B Base feature as a post-trained-model feature without a
  separate reconstruction and transfer validation; only the selected 27B pair is exact.

## Colab CLI Results

The official Colab CLI can execute local notebooks on a hosted GPU and recover both executed
`*_output.ipynb` notebooks and remote artifacts through `colab download` and `colab log`.
The CLI currently supports Linux and macOS, so Windows development should use WSL2. See
[docs/colab_cli_workflow.md](docs/colab_cli_workflow.md) for the repository-specific commands,
remote paths, and result archival workflow.

## Checks

```powershell
make lint
make test
make smoke
```

`make test` runs the unit tests directly; `make smoke` byte-compiles `src`/`tests` and
then runs the same suite, so it doubles as a fast pre-commit check.
