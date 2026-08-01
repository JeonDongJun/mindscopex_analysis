# Colab experiment runner

This directory is the batch-experiment layer next to the exploratory notebooks.
The notebooks remain useful for inspection and explanation; files here run the
same core `mindscopex_analysis` functions as reproducible Colab jobs.
Repository-wide module boundaries are documented in
[../docs/architecture.md](../docs/architecture.md).

## Layout

- `configs/`: one TOML file per experiment setting.
- `suites/`: ordered groups of config files.
- `jobs/`: Python entry points that run actual experiments.
  - `crt_text_responses.py`: behavioral CRT baseline (notebook 00).
  - `research_experiments.py`: the **controlled study** — train/test split, random-
    direction null, matched-control specificity, and a free-generation accuracy
    readout. See [../docs/study_design.md](../docs/study_design.md).
- `runners/`: local Colab launcher, remote bootstrap renderer, and artifact helpers.
- `run_colab.sh`: WSL-friendly wrapper around the launcher.

## Controlled study

The study discovers the lure feature on a train split and validates it on held-out
items — as teacher-forced margin *and* as free-generation accuracy. Which notebooks
motivated it is reviewed in [../docs/notebook_paper_audit.md](../docs/notebook_paper_audit.md).

```bash
# whole study (long; raise the timeout)
./experiments/run_colab.sh experiments/suites/study.toml -s mindscopex --gpu A100 --exec-timeout 9000

# all model sizes; switches from A100 to H100 before 27B
./experiments/run_colab.sh experiments/suites/study_all.toml -s mindscopex-all --exec-timeout 18000

# one stage at a time
./experiments/run_colab.sh experiments/configs/study_discover_2b.toml -s mindscopex
./experiments/run_colab.sh experiments/configs/study_behavioral_2b.toml -s mindscopex
```

For mixed-accelerator suites such as `study_all.toml`, do not pass `--gpu`: a
command-line override applies to every config. The launcher verifies the current
session hardware and replaces the runtime when the next config requires a
different accelerator. Transient allocation failures are retried; tune this with
`--allocation-attempts` and `--allocation-retry-delay`.

Design, datasets, and interpretation: [../docs/study_design.md](../docs/study_design.md).

## First smoke run

From WSL:

```bash
cd ~/dev/colab
./experiments/run_colab.sh experiments/suites/smoke.toml --session mindscopex-smoke
```

The launcher creates a local run directory under `results/runs/`, uploads the
current workspace as an archive to Colab, runs the configured job, downloads a
zip of the remote artifacts, exports the Colab execution log, and stops the VM
unless `--keep` is passed.

Use `--dry-run` to verify packaging without contacting Colab:

```bash
./experiments/run_colab.sh experiments/suites/smoke.toml --dry-run
```

## Source modes

The default config uses:

```toml
[remote]
source = "archive"
```

This uploads the current local workspace, including uncommitted experiment
changes, while excluding `.git`, `.venv`, `outputs/`, `results/`, and cache
directories. For fully pushed, reproducible runs you can switch to:

```toml
[remote]
source = "git"
repo_url = "https://github.com/JeonDongJun/mindscopex_analysis"
repo_ref = "main"
```

## Artifacts

Each CRT text-response run writes:

- `responses.json`: full generated responses and metadata.
- `summary.md`: readable report.
- `summary.json` and `family_summary.json`: aggregate tables.
- `figures/*.png`: quick review plots.
- `manifest.json`: config, environment, GPU, seed, and git metadata.

Each `research_experiments` run writes:

- `<figure>.csv`: paper-ready CSV per kind (`phenomenon.csv`, `discover_localization.csv`,
  `causal_heldout.csv`, `behavioral.csv`, `control_specificity.csv`) with a matching
  `<figure>.png` preview. Copy the ones you want into `paper/data/`.
- `study_feature.json`: the discovered (or pinned) lure feature (layer, id).
- per-kind `summary.json` / raw rows, and `manifest.json` (config, kinds, split,
  environment, GPU, git).

The local launcher also writes `launcher_manifest.json`, `colab_log.md`, and
`colab_log.jsonl` in the corresponding `results/runs/<timestamp>_<run>/` folder.
