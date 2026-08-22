"""Is the single-layer ablation understating the feature, or is it layer-local?

Two independent literatures predict that ablating one feature at one layer
understates its causal role: cross-layer superposition (a concept is written by
several adjacent layers, so per-layer SAEs learn near-copies and the surviving
copies re-supply the concept downstream) and self-repair / the Hydra effect
(downstream components compensate for an ablated upstream contribution). Either
one produces exactly our phase-2 signature -- a large local statistic and a weak
held-out effect. This job separates them from the third possibility, that the
direction is simply layer-local and the earlier number was all there was.

Design, per held-out item:

  clean          one prompt-only pass, captures at every probe layer -> the clean
                 projection p(l) = h(l)·u that fixes every removal magnitude, and
                 the clean cosine trajectory the self-repair curve is read against
  single_L       projection_remove of u at the feature's own layer
  single_L±k     u transplanted to each window layer alone
  window         all window layers ablated jointly in ONE forward pass
  rand_*         the same layers, token and per-layer removed magnitudes with
                 random unit directions -- both the joint and its singles, so the
                 interaction can be differenced within a draw
  sanity         joint {L real, L+1 at coefficient 0} must reproduce single_L

Superadditivity is NOT tested against additivity: the network is nonlinear, so a
positive interaction is expected even for junk directions, and a window removes
strictly more norm than any single site. The statistic is a difference-in-
differences -- the feature's interaction minus the matched random interaction --
tested with a paired sign-flip randomisation test. Every margin is also split
into its correct and lure logprob deltas, because a direction that merely damages
the model drags both down together while a real lure feature moves mostly the lure.
"""
# ruff: noqa: E402,I001

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch

from experiments.runners.config import load_toml, run_name, table
from mindscopex_analysis import (
    DEFAULT_ANALYSIS_PROFILE_KEY,
    EditSite,
    default_sae_device,
    dtype_from_name,
    family_balanced_subset,
    get_qwen35_analysis_profile,
    instruct_lure_cases,
    load_qwen_language_model,
    load_qwen_scope_sae,
    lure_dataset_cases,
    multi_site_answer_margin,
    recommended_dtype_name,
    sae_decoder_direction,
    split_lure_cases,
    trace_logits_multi_site,
)
from mindscopex_analysis.effects import continuation_token_span


def _log(message: str) -> None:
    print(f"[multi] {message}", flush=True)


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return path


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sign_flip_p(values: Sequence[float], draws: int = 20000, seed: int = 0) -> dict[str, Any]:
    """Two-sided paired sign-flip randomisation test on the per-item statistic.

    Under the null the per-item value is symmetric about zero, so flipping signs
    at random gives the exact reference distribution for its mean. Used instead of
    a t-test because these margins are heavy-tailed at n ~ 25.
    """

    if not values:
        return {"n": 0, "mean": None, "p": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    observed = float(tensor.mean())
    generator = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (draws, tensor.numel()), generator=generator, dtype=torch.float64)
    means = ((signs * 2 - 1) * tensor).mean(dim=1)
    p = float((means.abs() >= abs(observed)).to(torch.float64).mean())
    return {"n": len(values), "mean": observed, "p": p}


# ------------------------------------------------------------------- setup


def _resolve_env(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = table(config, "model")
    profile = get_qwen35_analysis_profile(
        str(model_cfg.get("profile", DEFAULT_ANALYSIS_PROFILE_KEY))
    )
    dtype = model_cfg.get("dtype", "auto")
    dtype = recommended_dtype_name() if dtype == "auto" else str(dtype)
    sae_device = model_cfg.get("sae_device", "auto")
    sae_device = default_sae_device() if sae_device == "auto" else str(sae_device)
    sae_dtype = model_cfg.get("sae_dtype", "auto")
    sae_dtype = dtype if sae_dtype == "auto" else str(sae_dtype)
    return {
        "profile": profile,
        "model_id": profile.analysis_model_id,
        "repo_id": profile.sae_repo_id,
        "dtype": dtype,
        "device_map": model_cfg.get("device_map", "auto"),
        "sae_device": sae_device,
        "sae_dtype": sae_dtype,
    }


def _load_items(config: dict[str, Any]) -> list[Any]:
    data_cfg = table(config, "data")
    dataset = str(data_cfg.get("dataset", "goal_affordance_traps_v1"))
    cases = lure_dataset_cases(dataset)
    conditions = data_cfg.get("conditions") or None
    if conditions:
        suffixes = tuple(f"_{name}" for name in conditions)
        cases = [case for case in cases if case.case_id.endswith(suffixes)]
    if bool(data_cfg.get("instruction", True)):
        cases = instruct_lure_cases(cases)
    _, test = split_lure_cases(
        cases,
        train_frac=float(data_cfg.get("train_frac", 0.6)),
        seed=int(data_cfg.get("split_seed", 0)),
    )
    max_items = int(data_cfg.get("max_items", 0))
    return family_balanced_subset(test, max_cases=max_items) if max_items else test


def _margin_row(margin: Any, baseline: Any) -> dict[str, float]:
    """Margin delta plus its decomposition -- the cheapest damage-vs-lure test."""

    return {
        "margin": float(margin.margin),
        "margin_delta": float(baseline.margin) - float(margin.margin),
        # Sign convention is effects.py's, NOT margin_delta's: the logprob deltas are
        # ablated - baseline, so `correct > 0` means the edit RAISED the correct answer
        # and `lure < 0` means it LOWERED the lure. margin_delta stays baseline - ablated.
        # docs/metrics_guide.md documents this pair; writing them the other way round
        # silently inverts every reader's conclusion.
        "correct_logprob_delta": float(margin.correct.logprob) - float(baseline.correct.logprob),
        "lure_logprob_delta": float(margin.lure.logprob) - float(baseline.lure.logprob),
    }


# ------------------------------------------------------------------- runner


def run(config_path: Path, output_root: Path) -> Path:
    config = load_toml(config_path)
    name = run_name(config)
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config.toml")

    feature_cfg = table(config, "feature")
    feature_id = int(feature_cfg["feature_id"])
    layer = int(feature_cfg["layer"])

    window_cfg = table(config, "window")
    radius = int(window_cfg.get("radius", 1))
    random_draws = int(window_cfg.get("random_draws", 8))
    seed = int(window_cfg.get("seed", 0))

    env = _resolve_env(config)
    profile = env["profile"]
    items = _load_items(config)

    _log(f"loading {env['model_id']} (feature {feature_id} @ layer {layer})")
    lm = load_qwen_language_model(
        env["model_id"], device_map=env["device_map"], dtype=env["dtype"], dispatch=True
    )
    sae = load_qwen_scope_sae(
        env["repo_id"], layer, device=env["sae_device"], dtype=dtype_from_name(env["sae_dtype"])
    )
    direction = sae_decoder_direction(sae, [feature_id]).detach().to(torch.float32).cpu()
    unit = direction / direction.norm().clamp_min(1e-12)
    d_model = int(unit.numel())

    n_layers = int(max(profile.scan_layers)) + 1
    window_layers = [
        candidate
        for candidate in range(layer - radius, layer + radius + 1)
        if 0 <= candidate < n_layers
    ]
    probe_layers = sorted({*window_layers, *range(layer, n_layers, max(1, n_layers // 8))})

    manifest: dict[str, Any] = {
        "run_name": name,
        "job": "multisite_ablation",
        "started_at": _timestamp(),
        "feature_id": feature_id,
        "layer": layer,
        "window_layers": window_layers,
        "probe_layers": probe_layers,
        "random_draws": random_draws,
        "profile": profile.key,
        "model_id": env["model_id"],
        "sae_repo_id": env["repo_id"],
        "n_items": len(items),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    _write_json(run_dir / "manifest.json", manifest)
    _log(f"window {window_layers} | probes {probe_layers} | {len(items)} held-out items")

    margin_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    interaction: list[dict[str, Any]] = []
    started = time.time()

    generator = torch.Generator().manual_seed(seed)
    random_units = torch.randn(random_draws, d_model, generator=generator)
    random_units = random_units / random_units.norm(dim=1, keepdim=True).clamp_min(1e-12)

    for index, case in enumerate(items, start=1):
        _, start = continuation_token_span(lm.tokenizer, case.prompt, case.correct_answer)
        edit_index = start - 1

        # Clean prompt-only pass: the last prompt token's residual is unaffected by
        # the answer (causal attention), so one pass fixes every removal magnitude.
        _, clean_caps = trace_logits_multi_site(
            lm, case.prompt, sites=(), token_index=edit_index, capture_layers=probe_layers
        )
        clean_proj: dict[int, float] = {}
        clean_cos: dict[int, float] = {}
        for probe in probe_layers:
            hidden = clean_caps[(probe, "pre")].to(torch.float32).reshape(-1)
            clean_proj[probe] = float(hidden @ unit)
            clean_cos[probe] = float(hidden @ unit / hidden.norm().clamp_min(1e-12))

        baseline = multi_site_answer_margin(
            lm,
            case.prompt,
            correct_answer=case.correct_answer,
            lure_answer=case.lure_answer,
            sites=(),
        )
        common = {"case_id": case.case_id, "family": case.family}
        margin_rows.append(
            {
                **common,
                "condition": "clean",
                "arm": "clean",
                "draw": -1,
                "margin": float(baseline.margin),
                "margin_delta": 0.0,
                "correct_logprob_delta": 0.0,
                "lure_logprob_delta": 0.0,
            }
        )

        def _real_site(target: int) -> EditSite:
            return EditSite(
                layer=target, direction=unit, coefficient=1.0, intervention_mode="projection_remove"
            )

        def _rand_site(target: int, vector: torch.Tensor) -> EditSite:
            # projection_remove along a random direction would remove a different
            # (much smaller) norm, so subtract the SAME magnitude the real edit did.
            return EditSite(
                layer=target,
                direction=vector,
                feature_value=1.0,
                coefficient=-abs(clean_proj[target]),
                intervention_mode="add_vector",
            )

        def _record(condition: str, arm: str, draw: int, sites: Sequence[EditSite]) -> float:
            margin = multi_site_answer_margin(
                lm,
                case.prompt,
                correct_answer=case.correct_answer,
                lure_answer=case.lure_answer,
                sites=sites,
            )
            row = {
                **common,
                "condition": condition,
                "arm": arm,
                "draw": draw,
                **_margin_row(margin, baseline),
            }
            margin_rows.append(row)
            return row["margin_delta"]

        singles = {
            target: _record(f"single_L{target - layer:+d}", "feature", -1, [_real_site(target)])
            for target in window_layers
        }
        joint = _record("window", "feature", -1, [_real_site(t) for t in window_layers])

        # Silent multi-site failure would fake a null, so assert the plumbing:
        # adding a zero-coefficient site must not change the single-site result.
        if layer + 1 < n_layers:
            zero = EditSite(
                layer=layer + 1,
                direction=unit,
                feature_value=1.0,
                coefficient=0.0,
                intervention_mode="add_vector",
            )
            _record("sanity_zero_site", "sanity", -1, [_real_site(layer), zero])

        rand_joint: list[float] = []
        rand_interaction: list[float] = []
        for draw in range(random_draws):
            vector = random_units[draw]
            rand_singles = {
                target: _record(
                    f"rand_single_L{target - layer:+d}",
                    "random",
                    draw,
                    [_rand_site(target, vector)],
                )
                for target in window_layers
            }
            value = _record(
                "rand_window", "random", draw, [_rand_site(t, vector) for t in window_layers]
            )
            rand_joint.append(value)
            rand_interaction.append(value - sum(rand_singles.values()))

        feature_interaction = joint - sum(singles.values())
        interaction.append(
            {
                **common,
                "baseline_margin": float(baseline.margin),
                "joint_delta": joint,
                "sum_single_deltas": sum(singles.values()),
                "feature_interaction": feature_interaction,
                "rand_joint_mean": _mean(rand_joint),
                "rand_interaction_mean": _mean(rand_interaction),
                # difference-in-differences: the feature's interaction beyond what
                # norm-matched junk directions produce through sheer nonlinearity
                "did": feature_interaction - _mean(rand_interaction),
                "single_L_delta": singles.get(layer, 0.0),
            }
        )

        # Self-repair: does the direction come back downstream of the edit?
        _, edited_caps = trace_logits_multi_site(
            lm,
            case.prompt,
            sites=[_real_site(layer)],
            token_index=edit_index,
            capture_layers=probe_layers,
        )
        for probe in probe_layers:
            if probe < layer:
                continue
            key = (probe, "post") if (probe, "post") in edited_caps else (probe, "pre")
            hidden = edited_caps[key].to(torch.float32).reshape(-1)
            proj = float(hidden @ unit)
            cos = float(hidden @ unit / hidden.norm().clamp_min(1e-12))
            repair_rows.append(
                {
                    **common,
                    "probe_layer": probe,
                    "is_edit_site": probe == layer,
                    "clean_projection": clean_proj[probe],
                    "edited_projection": proj,
                    "clean_cosine": clean_cos[probe],
                    "edited_cosine": cos,
                    # cosine, not the raw dot: the residual norm grows with depth,
                    # so a constant absolute deficit would read as recovery
                    "cosine_recovered_frac": (
                        cos / clean_cos[probe] if abs(clean_cos[probe]) > 1e-9 else None
                    ),
                }
            )

        _log(
            f"[{index}/{len(items)}] {case.case_id} single_L {singles.get(layer, 0.0):+.4f} "
            f"joint {joint:+.4f} DiD {interaction[-1]['did']:+.4f}"
        )
        _write_csv(
            run_dir / "margins.csv",
            margin_rows,
            [
                "case_id",
                "family",
                "condition",
                "arm",
                "draw",
                "margin",
                "margin_delta",
                "correct_logprob_delta",
                "lure_logprob_delta",
            ],
        )
        _write_csv(
            run_dir / "interaction.csv",
            interaction,
            [
                "case_id",
                "family",
                "baseline_margin",
                "single_L_delta",
                "joint_delta",
                "sum_single_deltas",
                "feature_interaction",
                "rand_joint_mean",
                "rand_interaction_mean",
                "did",
            ],
        )
        _write_csv(
            run_dir / "self_repair.csv",
            repair_rows,
            [
                "case_id",
                "family",
                "probe_layer",
                "is_edit_site",
                "clean_projection",
                "edited_projection",
                "clean_cosine",
                "edited_cosine",
                "cosine_recovered_frac",
            ],
        )

    def _column(condition: str, field: str = "margin_delta") -> list[float]:
        return [float(row[field]) for row in margin_rows if row["condition"] == condition]

    sanity = _column("sanity_zero_site")
    single_l = _column("single_L+0")
    summary = {
        "n_items": len(items),
        "layer": layer,
        "feature_id": feature_id,
        "window_layers": window_layers,
        "single_L": _sign_flip_p(single_l),
        "window": _sign_flip_p(_column("window")),
        "did": _sign_flip_p([float(row["did"]) for row in interaction]),
        "mean_feature_interaction": _mean([float(r["feature_interaction"]) for r in interaction]),
        "mean_rand_interaction": _mean([float(r["rand_interaction_mean"]) for r in interaction]),
        "sanity_max_abs_diff_vs_single_L": (
            max(abs(a - b) for a, b in zip(sanity, single_l, strict=True))
            if sanity and len(sanity) == len(single_l)
            else None
        ),
        "decomposition": {
            "single_L_correct_delta": _mean(_column("single_L+0", "correct_logprob_delta")),
            "single_L_lure_delta": _mean(_column("single_L+0", "lure_logprob_delta")),
            "window_correct_delta": _mean(_column("window", "correct_logprob_delta")),
            "window_lure_delta": _mean(_column("window", "lure_logprob_delta")),
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    _write_json(run_dir / "summary.json", summary)
    manifest.update({"finished_at": _timestamp(), "summary": summary})
    _write_json(run_dir / "manifest.json", manifest)
    _log(
        f"done: single_L {summary['single_L']} | window {summary['window']}"
    )
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
