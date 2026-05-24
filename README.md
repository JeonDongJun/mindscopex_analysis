# MindScopeX Analysis

Focused Qwen interpretability experiments with NNsight and Qwen-Scope.

## Quick Start

```powershell
uv sync --extra dev
uv run jupyter lab notebooks/
```

Start with `notebooks/01_qwen_scope_activation_mvp.ipynb`, then use
`notebooks/02_bat_ball_lure_feature_ablation.ipynb` for the bat-and-ball
feature ablation workflow.

## What Is Included

- `src/mindscopex_analysis/models.py`: Qwen + NNsight model loading helpers.
- `src/mindscopex_analysis/activations.py`: residual stream capture with `lm.trace(...).save()`.
- `src/mindscopex_analysis/qwen_scope.py`: Qwen-Scope SAE loading, TopK feature extraction, and a first-pass layer ranking helper.
- `src/mindscopex_analysis/effects.py`: answer logprob margins and SAE decoder-direction ablation.

The default pair is:

- Model: `Qwen/Qwen3-1.7B-Base`
- SAE repo: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`

That is the smallest currently available Qwen3 + Qwen-Scope pairing in the public collection.

## Checks

```powershell
make lint
make smoke
```
