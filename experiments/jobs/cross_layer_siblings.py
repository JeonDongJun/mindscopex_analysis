"""Find a feature's counterpart at other layers, then ablate the pair together.

``multisite_ablation`` transplants one layer's decoder direction to its neighbours.
That answers "does removing this direction at more places do more", but it does
not identify the feature at those layers -- each layer has its own SAE and its own
numbering, so L15 #81663 and L31 #81663 are unrelated by construction.

This job identifies the counterpart first, then intervenes on the identified pair:

    match     for every candidate at the target layer, three independent signals
              measured on the SAME discovery items -- decoder cosine, activation
              correlation, and effect correlation (does ablating each move the
              margin the same way per item). Ranked by their geometric mean, so a
              high cosine alone cannot win; cosine alone is exactly the failure
              mode this exists to avoid, the dictionary being overcomplete.

    co-ablate on held-out items: clean, A alone, B alone, A+B together, and the
              same four with norm-matched random directions at the same two layers.
              The statistic is a difference-in-differences -- the real pair's
              interaction minus the random pair's -- because the joint condition
              removes strictly more norm and the network is non-linear, so a
              positive interaction shows up even for unrelated directions.

    repair    B's own activation with and without A ablated. If removing A makes B
              fire harder, that is the compensation the Hydra-effect literature
              predicts, measured directly rather than inferred from a margin.
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
    answer_logprob_margin,
    capture_layer_residuals,
    clear_device_cache,
    default_sae_device,
    difference_in_differences,
    dtype_from_name,
    family_balanced_subset,
    gaussian_null_directions,
    get_qwen35_analysis_profile,
    instruct_lure_cases,
    load_qwen_language_model,
    load_qwen_scope_sae,
    lure_dataset_cases,
    multi_site_answer_margin,
    pearson,
    qwen_scope_sparse_feature_values,
    rank_siblings,
    recommended_dtype_name,
    sae_decoder_direction,
    split_lure_cases,
    trace_logits_multi_site,
)
from mindscopex_analysis.effects import continuation_token_span
from mindscopex_analysis.modules import rescale_to_norm


def _log(message: str) -> None:
    print(f"[siblings] {message}", flush=True)


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
    if not values:
        return {"n": 0, "mean": None, "p": None}
    tensor = torch.tensor([float(v) for v in values], dtype=torch.float64)
    observed = float(tensor.mean())
    generator = torch.Generator().manual_seed(int(seed))
    signs = torch.randint(0, 2, (draws, tensor.numel()), generator=generator, dtype=torch.float64)
    means = ((signs * 2 - 1) * tensor).mean(dim=1)
    return {
        "n": len(values),
        "mean": observed,
        "p": float((means.abs() >= abs(observed)).to(torch.float64).mean()),
        "frac_positive": sum(1 for v in values if v > 0) / len(values),
    }


def _resolve_env(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = table(config, "model")
    profile = get_qwen35_analysis_profile(
        str(model_cfg.get("profile", DEFAULT_ANALYSIS_PROFILE_KEY))
    )
    dtype = model_cfg.get("dtype", "auto")
    dtype = recommended_dtype_name() if dtype == "auto" else str(dtype)
    sae_device = model_cfg.get("sae_device", "auto")
    sae_device = default_sae_device() if sae_device == "auto" else str(sae_device)
    return {
        "profile": profile,
        "model_id": profile.analysis_model_id,
        "repo_id": profile.sae_repo_id,
        "dtype": dtype,
        "device_map": model_cfg.get("device_map", "auto"),
        "sae_device": sae_device,
        "sae_dtype": model_cfg.get("sae_dtype", dtype),
    }


def _load_splits(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = table(config, "data")
    dataset = str(data_cfg.get("dataset", "goal_affordance_traps_v1"))
    cases = lure_dataset_cases(dataset)
    conditions = data_cfg.get("conditions") or None
    if conditions:
        suffixes = tuple(f"_{name}" for name in conditions)
        cases = [case for case in cases if case.case_id.endswith(suffixes)]
    if bool(data_cfg.get("instruction", True)):
        cases = instruct_lure_cases(cases)
    train, test = split_lure_cases(
        cases,
        train_frac=float(data_cfg.get("train_frac", 0.6)),
        seed=int(data_cfg.get("split_seed", 0)),
    )
    max_match = int(data_cfg.get("max_match_items", 12))
    max_test = int(data_cfg.get("max_test_items", 12))
    return {
        "dataset": dataset,
        "match": family_balanced_subset(train, max_cases=max_match),
        "test": family_balanced_subset(test, max_cases=max_test) if max_test else test,
    }


def _ablate_margin(lm: Any, case: Any, sites: Sequence[EditSite]) -> Any:
    return multi_site_answer_margin(
        lm,
        case.prompt,
        correct_answer=case.correct_answer,
        lure_answer=case.lure_answer,
        sites=sites,
    )


def run(config_path: Path, output_root: Path) -> Path:
    config = load_toml(config_path)
    name = run_name(config)
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config.toml")

    fcfg = table(config, "feature")
    source_feature = int(fcfg["feature_id"])
    source_layer = int(fcfg["layer"])

    mcfg = table(config, "match")
    candidate_top_n = int(mcfg.get("candidate_top_n", 20))
    min_score = float(mcfg.get("min_score", 0.05))
    random_draws = int(mcfg.get("random_draws", 6))
    seed = int(mcfg.get("seed", 0))
    target_layers = [int(v) for v in (mcfg.get("layers") or [])]

    env = _resolve_env(config)
    profile = env["profile"]
    if not target_layers:
        target_layers = [int(v) for v in profile.scan_layers if int(v) != source_layer]
    splits = _load_splits(config)

    manifest: dict[str, Any] = {
        "run_name": name,
        "job": "cross_layer_siblings",
        "started_at": _timestamp(),
        "source_feature": source_feature,
        "source_layer": source_layer,
        "target_layers": target_layers,
        "profile": profile.key,
        "model_id": env["model_id"],
        "sae_repo_id": env["repo_id"],
        "dataset": splits["dataset"],
        "n_match_items": len(splits["match"]),
        "n_test_items": len(splits["test"]),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    _write_json(run_dir / "manifest.json", manifest)

    _log(f"loading {env['model_id']} (source L{source_layer} #{source_feature})")
    lm = load_qwen_language_model(
        env["model_id"], device_map=env["device_map"], dtype=env["dtype"], dispatch=True
    )
    sae_dtype = dtype_from_name(env["sae_dtype"])
    source_sae = load_qwen_scope_sae(
        env["repo_id"], source_layer, device=env["sae_device"], dtype=sae_dtype
    )
    source_direction = sae_decoder_direction(source_sae, [source_feature]).detach()
    source_unit = source_direction.to(torch.float32).cpu() / source_direction.to(
        torch.float32
    ).cpu().norm().clamp_min(1e-12)
    started = time.time()

    # --- source profile on the matching items ------------------------------
    _log(f"profiling the source feature on {len(splits['match'])} discovery items")
    source_values: list[float] = []
    source_effects: list[float] = []
    baselines: list[float] = []
    for case in splits["match"]:
        residual = capture_layer_residuals(lm, [case.prompt], source_layer, token_position="last")
        value = float(
            qwen_scope_sparse_feature_values(residual, source_sae, [source_feature])
            .detach()
            .to(torch.float32)
            .reshape(-1)[0]
        )
        baseline = float(
            answer_logprob_margin(
                lm,
                case.prompt,
                correct_answer=case.correct_answer,
                lure_answer=case.lure_answer,
            ).margin
        )
        ablated = float(
            _ablate_margin(
                lm,
                case,
                [EditSite(source_layer, source_direction, value, 1.0, "remove_activation")],
            ).margin
        )
        source_values.append(value)
        baselines.append(baseline)
        source_effects.append(baseline - ablated)

    # --- candidate matching at each target layer ---------------------------
    sibling_rows: list[dict[str, Any]] = []
    best_by_layer: dict[int, dict[str, Any]] = {}
    for target_layer in target_layers:
        target_sae = load_qwen_scope_sae(
            env["repo_id"], target_layer, device=env["sae_device"], dtype=sae_dtype
        )
        decoder = (
            (
                target_sae.W_dec.T
                if target_sae.W_dec.shape[0] == target_sae.d_model
                else target_sae.W_dec
            )
            .detach()
            .to(torch.float32)
            .cpu()
        )
        units = decoder / decoder.norm(dim=1, keepdim=True).clamp_min(1e-12)
        cosines = units @ source_unit
        candidates = torch.topk(cosines, min(candidate_top_n, cosines.numel())).indices.tolist()
        _log(f"L{target_layer}: scoring {len(candidates)} cosine-nearest candidates")

        # Per-candidate activation and effect on the SAME items, so the two
        # correlations are comparable with the source's.
        target_residuals = [
            capture_layer_residuals(lm, [case.prompt], target_layer, token_position="last")
            for case in splits["match"]
        ]
        for candidate in candidates:
            direction = sae_decoder_direction(target_sae, [int(candidate)]).detach()
            values: list[float] = []
            effects: list[float] = []
            for index, case in enumerate(splits["match"]):
                value = float(
                    qwen_scope_sparse_feature_values(
                        target_residuals[index], target_sae, [int(candidate)]
                    )
                    .detach()
                    .to(torch.float32)
                    .reshape(-1)[0]
                )
                ablated = float(
                    _ablate_margin(
                        lm,
                        case,
                        [EditSite(target_layer, direction, value, 1.0, "remove_activation")],
                    ).margin
                )
                values.append(value)
                effects.append(baselines[index] - ablated)
            sibling_rows.append(
                {
                    "source_layer": source_layer,
                    "source_feature": source_feature,
                    "target_layer": target_layer,
                    "target_feature": int(candidate),
                    "decoder_cosine": float(cosines[int(candidate)]),
                    "activation_corr": pearson(source_values, values),
                    "effect_corr": pearson(source_effects, effects),
                    "mean_activation": _mean(values),
                    "mean_effect": _mean(effects),
                }
            )
        ranked = rank_siblings(
            [row for row in sibling_rows if row["target_layer"] == target_layer],
            min_score=min_score,
        )
        if ranked:
            best_by_layer[target_layer] = ranked[0]
            _log(
                f"L{target_layer}: best #{ranked[0]['target_feature']} "
                f"score {ranked[0]['combined_score']:.3f} "
                f"(cos {ranked[0]['decoder_cosine']:.2f}, "
                f"act {ranked[0]['activation_corr']:+.2f}, eff {ranked[0]['effect_corr']:+.2f})"
            )
        else:
            _log(f"L{target_layer}: no candidate cleared min_score={min_score}")
        del target_sae, decoder, units, target_residuals
        clear_device_cache()

    _write_csv(
        run_dir / "cross_layer_siblings.csv",
        rank_siblings(sibling_rows, min_score=-1.0),
        [
            "source_layer",
            "source_feature",
            "target_layer",
            "target_feature",
            "decoder_cosine",
            "activation_corr",
            "effect_corr",
            "combined_score",
            "mean_activation",
            "mean_effect",
        ],
    )
    if not best_by_layer:
        _write_json(
            run_dir / "coablation_summary.json",
            {
                "note": "no sibling cleared min_score; co-ablation not attempted",
                "min_score": min_score,
            },
        )
        manifest.update({"finished_at": _timestamp(), "siblings": []})
        _write_json(run_dir / "manifest.json", manifest)
        print(f"ARTIFACT_DIR={run_dir}", flush=True)
        return run_dir

    # --- conditional co-ablation on held-out items -------------------------
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    # Two INDEPENDENT draws per null pair, not one direction reused at both layers.
    # A real sibling pair is aligned by construction; baking that alignment into the
    # null would subtract away the very structure the DiD is meant to detect.
    generator_directions = gaussian_null_directions(
        int(source_sae.d_model), random_draws * 2, seed=seed
    )

    for target_layer, sibling in best_by_layer.items():
        target_sae = load_qwen_scope_sae(
            env["repo_id"], target_layer, device=env["sae_device"], dtype=sae_dtype
        )
        target_feature = int(sibling["target_feature"])
        target_direction = sae_decoder_direction(target_sae, [target_feature]).detach()
        _log(f"co-ablation L{source_layer}#{source_feature} + L{target_layer}#{target_feature}")

        a_only: list[float] = []
        b_only: list[float] = []
        joint: list[float] = []
        rand_a: list[float] = []
        rand_b: list[float] = []
        rand_joint: list[float] = []
        repair: list[float] = []

        for item_index, case in enumerate(splits["test"], start=1):
            _, start = continuation_token_span(lm.tokenizer, case.prompt, case.correct_answer)
            edit_index = start - 1
            source_residual = capture_layer_residuals(
                lm, [case.prompt], source_layer, token_position="last"
            )
            a_value = float(
                qwen_scope_sparse_feature_values(source_residual, source_sae, [source_feature])
                .detach()
                .to(torch.float32)
                .reshape(-1)[0]
            )
            target_residual = capture_layer_residuals(
                lm, [case.prompt], target_layer, token_position="last"
            )
            b_value = float(
                qwen_scope_sparse_feature_values(target_residual, target_sae, [target_feature])
                .detach()
                .to(torch.float32)
                .reshape(-1)[0]
            )
            baseline = _ablate_margin(lm, case, [])
            site_a = EditSite(source_layer, source_direction, a_value, 1.0, "remove_activation")
            site_b = EditSite(target_layer, target_direction, b_value, 1.0, "remove_activation")

            def _record(condition: str, sites: Sequence[EditSite], draw: int = -1) -> float:
                margin = _ablate_margin(lm, case, sites)
                delta = float(baseline.margin) - float(margin.margin)
                rows.append(
                    {
                        "case_id": case.case_id,
                        "family": case.family,
                        "target_layer": target_layer,
                        "target_feature": target_feature,
                        "condition": condition,
                        "draw": draw,
                        "margin_delta": delta,
                        "correct_logprob_delta": (
                            float(baseline.correct.logprob) - float(margin.correct.logprob)
                        ),
                        "lure_logprob_delta": (
                            float(baseline.lure.logprob) - float(margin.lure.logprob)
                        ),
                    }
                )
                return delta

            a_only.append(_record("a_only", [site_a]))
            b_only.append(_record("b_only", [site_b]))
            joint.append(_record("joint", [site_a, site_b]))

            # Norm-matched random directions at the SAME two layers: the joint edit
            # removes more norm than either part, so an unmatched null would make any
            # pair look superadditive.
            a_norm = abs(a_value) * float(source_direction.to(torch.float32).norm())
            b_norm = abs(b_value) * float(target_direction.to(torch.float32).norm())
            draw_a: list[float] = []
            draw_b: list[float] = []
            draw_joint: list[float] = []
            for draw in range(random_draws):
                rsite_a = EditSite(
                    source_layer,
                    rescale_to_norm(generator_directions[draw], a_norm),
                    1.0,
                    -1.0,
                    "add_vector",
                )
                rsite_b = EditSite(
                    target_layer,
                    rescale_to_norm(generator_directions[random_draws + draw], b_norm),
                    1.0,
                    -1.0,
                    "add_vector",
                )
                draw_a.append(_record("rand_a", [rsite_a], draw))
                draw_b.append(_record("rand_b", [rsite_b], draw))
                draw_joint.append(_record("rand_joint", [rsite_a, rsite_b], draw))
            rand_a.append(_mean(draw_a))
            rand_b.append(_mean(draw_b))
            rand_joint.append(_mean(draw_joint))

            # Does B fire harder once A is gone? Measured on the activation itself,
            # not inferred from a margin.
            _, captures = trace_logits_multi_site(
                lm,
                case.prompt,
                sites=[site_a],
                token_index=edit_index,
                capture_layers=[target_layer],
            )
            key = (
                (target_layer, "post")
                if (target_layer, "post") in captures
                else (target_layer, "pre")
            )
            edited = captures[key].to(torch.float32)
            b_after = float(
                qwen_scope_sparse_feature_values(edited, target_sae, [target_feature])
                .detach()
                .to(torch.float32)
                .reshape(-1)[0]
            )
            repair.append(b_after - b_value)

            if item_index % 4 == 0:
                _log(f"L{target_layer}: {item_index}/{len(splits['test'])} items")
            _write_csv(
                run_dir / "coablation.csv",
                rows,
                [
                    "case_id",
                    "family",
                    "target_layer",
                    "target_feature",
                    "condition",
                    "draw",
                    "margin_delta",
                    "correct_logprob_delta",
                    "lure_logprob_delta",
                ],
            )

        did = difference_in_differences(joint, [a_only, b_only], rand_joint, [rand_a, rand_b])
        summaries.append(
            {
                "target_layer": target_layer,
                "target_feature": target_feature,
                "sibling_score": sibling["combined_score"],
                "decoder_cosine": sibling["decoder_cosine"],
                "activation_corr": sibling["activation_corr"],
                "effect_corr": sibling["effect_corr"],
                "a_only": _sign_flip_p(a_only, seed=seed),
                "b_only": _sign_flip_p(b_only, seed=seed),
                "joint": _sign_flip_p(joint, seed=seed),
                "difference_in_differences": _sign_flip_p(did, seed=seed),
                "mean_rand_joint": _mean(rand_joint),
                # Positive means B fires HARDER once A is removed: compensation.
                "sibling_repair": _sign_flip_p(repair, seed=seed),
            }
        )
        _write_json(run_dir / "coablation_summary.json", summaries)
        _log(
            f"L{target_layer}: joint {summaries[-1]['joint']['mean']:+.4f} | "
            f"DiD {summaries[-1]['difference_in_differences']['mean']:+.4f} "
            f"(p={summaries[-1]['difference_in_differences']['p']}) | "
            f"repair {summaries[-1]['sibling_repair']['mean']:+.4f}"
        )
        del target_sae
        clear_device_cache()

    manifest.update(
        {
            "finished_at": _timestamp(),
            "elapsed_seconds": round(time.time() - started, 1),
            "siblings": [
                {k: v for k, v in row.items() if k != "combined_score"}
                for row in best_by_layer.values()
            ],
            "summaries": summaries,
        }
    )
    _write_json(run_dir / "manifest.json", manifest)
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
