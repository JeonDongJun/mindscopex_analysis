"""Run CRT text-response experiments without opening a notebook."""
# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from experiments.runners.config import (  # noqa: E402
    bool_or_none_list,
    int_list,
    load_toml,
    run_name,
    table,
)
from mindscopex_analysis import (
    CRT_FINAL_ANSWER_SYSTEM_PROMPT,
    DEFAULT_QWEN_CHAT_MODEL_IDS,
    QWEN_LARGE_CHAT_MODEL_IDS,
    clear_device_cache,
    generate_crt_response_suite,
    load_qwen_text_generation_model,
    lure_dataset_cases,
    lure_dataset_info,
    recommended_dtype_name,
    save_crt_markdown_report,
    save_qwen_text_responses,
    summarize_crt_accuracy,
    summarize_crt_accuracy_by_family,
)  # noqa: E402


# preset -> committed dataset id (data/<id>.json) and per-family cap. See docs/datasets.md.
RUN_PRESETS = {
    "pilot": {"dataset": "crt_pilot", "limit_per_family": None},
    "hagendorff_smoke": {"dataset": "hagendorff_crt", "limit_per_family": 3},
    "hagendorff_full": {"dataset": "hagendorff_crt", "limit_per_family": None},
    "verbal_crt": {"dataset": "verbal_crt", "limit_per_family": None},
    "crt7_classic": {"dataset": "crt7_classic", "limit_per_family": None},
    "crt2": {"dataset": "crt2", "limit_per_family": None},
    "yax_crt_isomorph": {"dataset": "yax_crt_isomorph", "limit_per_family": None},
    # Back-compat aliases for the pre-rename preset names.
    "nature_smoke": {"dataset": "hagendorff_crt", "limit_per_family": 3},
    "nature_full": {"dataset": "hagendorff_crt", "limit_per_family": None},
}

PROTOCOLS = {
    "qwen_native": {
        "do_sample": True,
        "system_prompt": CRT_FINAL_ANSWER_SYSTEM_PROMPT,
        "max_retries": 2,
        "retry_protocol_issues": True,
        "retry_both": True,
    },
    "deterministic_replication": {
        "do_sample": False,
        "system_prompt": "",
        "max_retries": 0,
        "retry_protocol_issues": False,
        "retry_both": False,
    },
}


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            [
                "git",
                "-c",
                "filter.nbstripout.clean=cat",
                "-c",
                "filter.nbstripout.smudge=cat",
                "-c",
                "filter.nbstripout.required=false",
                *args,
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return path


def _resolve_model_ids(raw: Any) -> list[str]:
    if raw is None:
        return list(DEFAULT_QWEN_CHAT_MODEL_IDS)
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise TypeError("[model].ids must be a string or list of strings")
    if raw == ["default"]:
        return list(DEFAULT_QWEN_CHAT_MODEL_IDS)
    return raw


def _load_cases(config: dict[str, Any]) -> tuple[list[Any], str, str, str]:
    dataset_cfg = table(config, "dataset")
    preset_name = str(dataset_cfg.get("preset", "hagendorff_smoke"))
    if preset_name not in RUN_PRESETS:
        raise ValueError(f"Unknown dataset preset: {preset_name!r}. Options: {sorted(RUN_PRESETS)}")

    preset = RUN_PRESETS[preset_name]
    dataset_name = preset["dataset"]
    families = dataset_cfg.get("families")
    limit_per_family = dataset_cfg.get("limit_per_family", preset["limit_per_family"])
    cases = lure_dataset_cases(
        dataset_name,
        families=tuple(families) if families else None,
        limit_per_family=limit_per_family,
    )
    source = lure_dataset_info(dataset_name).source
    dataset_reference = source.get("doi") or source.get("project_url") or dataset_name
    return cases, preset_name, dataset_name, dataset_reference


def _save_figures(
    responses: list[Any],
    figures_dir: Path,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_crt_accuracy(responses)
    family_summary = summarize_crt_accuracy_by_family(responses)
    written: list[Path] = []

    if summary:
        labels = [f"{row['model']}\n{row['mode']}" for row in summary]
        accuracy = [row["accuracy"] for row in summary]
        lure_rate = [row["lure_rate"] for row in summary]
        width = max(7.0, len(labels) * 1.1)
        fig, ax = plt.subplots(figsize=(width, 4.5), constrained_layout=True)
        x_positions = range(len(labels))
        ax.bar([x - 0.18 for x in x_positions], accuracy, width=0.36, label="accuracy")
        ax.bar([x + 0.18 for x in x_positions], lure_rate, width=0.36, label="lure rate")
        ax.set_xticks(list(x_positions), labels, rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Rate")
        ax.legend()
        path = figures_dir / "headline_accuracy.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    if family_summary:
        labels = [f"{row['model']}\n{row['mode']}\n{row['family']}" for row in family_summary]
        lure_rate = [row["lure_rate"] for row in family_summary]
        width = max(8.0, len(labels) * 0.8)
        fig, ax = plt.subplots(figsize=(width, 4.8), constrained_layout=True)
        ax.bar(range(len(labels)), lure_rate, color="#8f5fbf")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Lure rate")
        path = figures_dir / "family_lure_rate.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    return written


def run(config_path: Path, output_root: Path) -> Path:
    config = load_toml(config_path)
    name = run_name(config)
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_cases, preset_name, dataset_name, dataset_reference = _load_cases(config)
    model_cfg = table(config, "model")
    generation_cfg = table(config, "generation")
    protocol_name = str(generation_cfg.get("protocol", "qwen_native"))
    if protocol_name not in PROTOCOLS:
        raise ValueError(f"Unknown generation protocol: {protocol_name!r}")
    protocol = {**PROTOCOLS[protocol_name], **generation_cfg.get("overrides", {})}

    model_ids = _resolve_model_ids(model_cfg.get("ids"))
    seeds = int_list(generation_cfg.get("seeds"), default=(42,))
    thinking_modes = bool_or_none_list(
        generation_cfg.get("thinking_modes"),
        default=(False, True),
    )
    dtype = model_cfg.get("dtype", "auto")
    dtype = recommended_dtype_name() if dtype == "auto" else dtype
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 4096))
    max_retries = int(generation_cfg.get("max_retries", protocol["max_retries"]))
    do_sample = bool(generation_cfg.get("do_sample", protocol["do_sample"]))
    retry_protocol_issues = bool(
        generation_cfg.get("retry_protocol_issues", protocol["retry_protocol_issues"])
    )
    retry_both = bool(generation_cfg.get("retry_both", protocol["retry_both"]))
    system_prompt = str(generation_cfg.get("system_prompt", protocol["system_prompt"]))
    device_map = model_cfg.get("device_map", "auto")

    output_json = run_dir / "responses.json"
    report_md = run_dir / "summary.md"
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    family_summary_path = run_dir / "family_summary.json"
    config_copy = run_dir / "config.toml"
    shutil.copyfile(config_path, config_copy)

    manifest = {
        "run_name": name,
        "job": "crt_text_responses",
        "started_at": _timestamp(),
        "config_path": str(config_path),
        "output_dir": str(run_dir),
        "dataset": dataset_name,
        "dataset_preset": preset_name,
        "dataset_reference": dataset_reference,
        "n_cases": len(dataset_cases),
        "model_ids": model_ids,
        "seeds": seeds,
        "thinking_modes": thinking_modes,
        "generation_protocol": protocol_name,
        "max_new_tokens": max_new_tokens,
        "max_retries": max_retries,
        "dtype": dtype,
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_status": _git_value(["status", "--short"]),
    }
    _write_json(manifest_path, manifest)

    responses: list[Any] = []
    save_every = 10
    start = time.time()

    def _on_progress(done: int, total: int, response: Any) -> None:
        responses.append(response)
        short = response.model_id.rsplit("/", 1)[-1]
        print(
            f"[{done}/{total}] {short} {response.mode} {response.case_id} "
            f"-> {response.answer_label}/{response.evaluation_label}",
            flush=True,
        )
        if done % save_every == 0:
            save_qwen_text_responses(responses, output_json)

    for model_id in model_ids:
        model, tokenizer = load_qwen_text_generation_model(
            model_id,
            device_map=device_map,
            dtype=dtype,
        )
        try:
            for seed in seeds:
                print(f"Running {model_id} seed={seed} cases={len(dataset_cases)}", flush=True)
                generate_crt_response_suite(
                    model,
                    tokenizer,
                    dataset_cases,
                    model_id=model_id,
                    thinking_modes=thinking_modes,
                    use_chat_template=True,
                    system_prompt=system_prompt,
                    max_new_tokens=(
                        2048 if model_id in QWEN_LARGE_CHAT_MODEL_IDS else max_new_tokens
                    ),
                    do_sample=do_sample,
                    seed=seed,
                    max_retries=max_retries,
                    retry_protocol_issues=retry_protocol_issues,
                    retry_both=retry_both,
                    progress_callback=_on_progress,
                )
                save_qwen_text_responses(responses, output_json)
                save_crt_markdown_report(
                    responses,
                    report_md,
                    dataset_name=f"{dataset_name}:{protocol_name}",
                    dataset_reference=dataset_reference,
                )
                _write_json(summary_path, summarize_crt_accuracy(responses))
                _write_json(family_summary_path, summarize_crt_accuracy_by_family(responses))
        finally:
            del model
            del tokenizer
            clear_device_cache()

    figure_paths = _save_figures(responses, run_dir / "figures")
    manifest.update(
        {
            "finished_at": _timestamp(),
            "elapsed_seconds": round(time.time() - start, 3),
            "n_responses": len(responses),
            "artifacts": {
                "responses": str(output_json),
                "summary_markdown": str(report_md),
                "summary_json": str(summary_path),
                "family_summary_json": str(family_summary_path),
                "figures": [str(path) for path in figure_paths],
            },
        }
    )
    _write_json(manifest_path, manifest)
    print(f"ARTIFACT_DIR={run_dir}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("outputs/experiments"), type=Path)
    args = parser.parse_args()
    run(args.config, args.output_root)


if __name__ == "__main__":
    main()
