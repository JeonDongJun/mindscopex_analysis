# Colab experiment runner

This directory is the batch-experiment layer next to the exploratory notebooks.
The notebooks remain useful for inspection and explanation; files here run the
same core `mindscopex_analysis` functions as reproducible Colab jobs.

## Layout

- `configs/`: one TOML file per experiment setting.
- `suites/`: ordered groups of config files.
- `jobs/`: Python entry points that run actual experiments.
- `runners/`: local Colab launcher, remote bootstrap renderer, and artifact helpers.
- `run_colab.sh`: WSL-friendly wrapper around the launcher.

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

The local launcher also writes `launcher_manifest.json`, `colab_log.md`, and
`colab_log.jsonl` in the corresponding `results/runs/<timestamp>_<run>/` folder.
