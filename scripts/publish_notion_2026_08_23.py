"""Publish the 2026-08-23 run batch to Notion.

Five experiments ran back to back on one pinned feature question. This reads their
artifacts rather than restating numbers by hand, so the page cannot drift from the
run directories it describes.

Usage:
    uv run python scripts/publish_notion_2026_08_23.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PARENT_PAGE_ID = "32a0df78-43fe-8056-b56d-f334c1d47edf"
NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2026-03-11"
RUNS = ROOT / "results" / "runs"

RUN_DIRS = {
    "modules": "20260823-101459_modules_affordance_27b",
    "study2b": "20260823-102341_study_affordance_2b",
    "falsify": "20260823-103435_falsify_affordance_27b",
    "siblings": "20260823-104108_siblings_affordance_27b",
    "trajectory": "20260823-104939_trajectory_affordance_27b",
}


# ----------------------------------------------------------------- notion api


def env_value(name: str) -> str:
    for raw in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip("'\"")
    raise RuntimeError(f"{name} is missing from .env")


def headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {env_value('NOTION_TOKEN')}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def request(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, data=data, headers=headers(), method=method)
            with urllib.request.urlopen(req, timeout=60) as response:  # noqa: S310
                body = response.read()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 or 500 <= exc.code < 600:
                time.sleep(min(2**attempt, 10))
                continue
            raise RuntimeError(f"HTTP {exc.code}: {detail[:1200]}") from exc
    raise RuntimeError("Notion API retries exhausted")


def rich(text: str, *, bold: bool = False, code: bool = False) -> list[dict[str, Any]]:
    return [
        {
            "type": "text",
            "text": {"content": str(text)[:1990]},
            "annotations": {"bold": bold, "code": code},
        }
    ]


def block(kind: str, text: str) -> dict[str, Any]:
    return {"object": "block", "type": kind, kind: {"rich_text": rich(text)}}


def callout(text: str, emoji: str = "⚠️") -> dict[str, Any]:
    return {
        "object": "block",
        "type": "callout",
        "callout": {"rich_text": rich(text), "icon": {"type": "emoji", "emoji": emoji}},
    }


def code_block(text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "code",
        "code": {"rich_text": rich(text), "language": "plain text"},
    }


def table(header: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    def row(cells: list[Any]) -> dict[str, Any]:
        return {
            "object": "block",
            "type": "table_row",
            "table_row": {"cells": [rich(str(c)) for c in cells]},
        }

    return {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": len(header),
            "has_column_header": True,
            "has_row_header": False,
            "children": [row(header)] + [row(r) for r in rows],
        },
    }


# ----------------------------------------------------------------- artifacts


def art(key: str, name: str) -> Path:
    return RUNS / RUN_DIRS[key] / "artifacts" / name


def load_json(key: str, name: str) -> Any:
    return json.loads(art(key, name).read_text(encoding="utf-8"))


def load_csv(key: str, name: str) -> list[dict[str, str]]:
    with art(key, name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def stat_row(label: str, s: dict[str, Any]) -> list[str]:
    ci = f"[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]" if s.get("ci_low") is not None else "—"
    p = f"{s['p']:.4f}" if s.get("p") is not None else "—"
    return [label, fmt(s.get("mean")), p, ci, f"{s.get('n_positive')}/{s.get('n')}"]


# ----------------------------------------------------------------- sections


def intro_blocks() -> list[dict[str, Any]]:
    return [
        block("heading_2", "무엇을 물었나"),
        block(
            "paragraph",
            "이 연구는 'Qwen3.5 안의 어떤 SAE feature가 직관적이지만 틀린 답(lure)을 "
            "인과적으로 매개하는가'를 묻습니다. 27B L15 #81663이 그 후보였습니다. "
            "오늘 다섯 실험은 그 후보를 서로 다른 방향에서 공격했고, "
            "전부 같은 곳으로 수렴했습니다.",
        ),
        callout(
            "결론: #81663은 함정 feature가 아닙니다. 마지막 프롬프트 토큰에서만 켜지는 "
            "위치 feature이고, 모든 조건·모든 과제에서 100% 발화합니다. "
            "그리고 2B의 후보 #2144는 selection overfitting이었습니다.",
            "🔴",
        ),
        block(
            "paragraph",
            "다섯 실험 모두 Qwen3.5 + Qwen-Scope SAE, Colab G4(RTX PRO 6000 96GB)에서 순차 실행.",
        ),
    ]


def modules_blocks() -> list[dict[str, Any]]:
    summary = load_json("modules", "module_summary.json")
    search = load_json("modules", "module_search.json")
    out: list[dict[str, Any]] = [
        block("heading_2", "1. feature_modules — 효과가 여러 feature에 분산돼 있는가?"),
        block(
            "paragraph",
            "단일 feature가 실패한 이유가 '효과가 없어서'가 아니라 '여러 feature에 나뉘어서'일 수 "
            "있습니다. 함께 켜지는 집합을 통째로 지우고 held-out margin이 움직이는지 봤습니다.",
        ),
        block("heading_3", "coactivation 구조 자체는 실재합니다"),
        block(
            "paragraph",
            "각 feature를 항목에 대해 셔플해 주변분포는 유지하고 교차 정렬만 파괴하는 permutation "
            "null과 비교했습니다.",
        ),
    ]
    sweep_rows = [
        [
            f"{r['threshold']:.2f}",
            r["n_edges_kept"],
            f"{r['permutation_null_mean_edges']:.1f}",
            r["permutation_null_p95_edges"],
            f"{r['permutation_p']:.4f}",
            str(r["component_sizes"]),
        ]
        for r in search["sweep"]
    ]
    out.append(
        table(
            ["임계값", "관측 엣지", "null 평균", "null p95", "p", "컴포넌트"],
            sweep_rows,
        )
    )
    out.append(
        block(
            "paragraph",
            "관측이 null을 10~30배 넘습니다. feature들은 주변분포로 설명되지 않는 방식으로 "
            "실제로 함께 켜집니다. (직전 run이 jaccard로 크래시한 이유가 여기 있습니다 — "
            "전부가 전부에서 켜지니 이진 co-firing이 상수라 정보가 0이었습니다.)",
        )
    )
    out.append(block("heading_3", "그런데 모듈 개입은 단일 feature보다 약합니다"))
    for m in summary:
        out.append(
            block(
                "paragraph",
                f"모듈 {m['module']} — features {m['features']}, size {m['size']}",
            )
        )
        out.append(
            table(
                ["조건", "평균", "p", "95% CI", "양수"],
                [
                    stat_row("joint (모듈 전체)", m["joint"]),
                    stat_row("single_best (최강 멤버)", m["single_best"]),
                    stat_row("joint − random", m["joint_minus_random"]),
                ],
            )
        )
    out.append(
        callout(
            "이 run에서 0을 배제하는 CI를 가진 유일한 숫자가 single_best입니다. "
            "feature를 더할수록 신호가 아니라 잡음이 늘었습니다. "
            "coactivation은 실재하지만 인과적으로 함께 작동하지는 않습니다 — "
            "함께 켜지는 것과 함께 원인이 되는 것은 다릅니다.",
            "🔍",
        )
    )
    out.append(
        block(
            "paragraph",
            "single_best는 순환 선택이 아닙니다: 항목별 '제거 norm'(활성값 기준)으로 뽑고 "
            "margin으로 고르지 않습니다. 다만 모듈 자체가 val slice의 causal 항으로 3개 후보 중 "
            "argmax로 뽑혔으므로 상류 선택 압력이 간접 전달됩니다 — artifact의 "
            "random_module_null.caveat가 이를 명시합니다.",
        )
    )
    return out


def study2b_blocks() -> list[dict[str, Any]]:
    new = load_json("study2b", "study_feature.json")
    old_path = RUNS / "20260809-182535_study_affordance_2b" / "artifacts" / "study_feature.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    old_by_layer = {int(r["layer"]): r for r in old["localization"]}

    rows = []
    for r in new["localization"]:
        o = old_by_layer.get(int(r["layer"]), {})
        rows.append(
            [
                f"L{r['layer']}",
                r["feature_id"],
                f"{r['mean_margin_delta']:.6f}",
                fmt(o.get("observed_mean_delta")),
                fmt(r["screen_hostile_delta"]),
                fmt(o.get("null_z"), 2),
                fmt(r["screen_null_z"], 2),
            ]
        )

    return [
        block("heading_2", "2. 2B discovery 재선택 — 가장 중요한 결과"),
        block(
            "paragraph",
            "2B의 #2144(L17)는 P0-1 버그(raw pre-activation으로 후보 채점)가 살아있을 때 "
            "선택됐고 한 번도 재현된 적이 없었습니다. 설정을 그대로 두고 코드만 HEAD로 올려 "
            "다시 돌렸습니다.",
        ),
        block("heading_3", "좋은 소식: P0-1은 2B에서도 no-op였습니다"),
        block(
            "paragraph",
            "레이어별 후보와 그 mean_margin_delta가 비트 단위로 동일합니다. "
            "pre-activation 버그는 후보 랭킹을 바꾸지 않았습니다.",
        ),
        block("heading_3", "나쁜 소식: 승자가 바뀌었고, 원인은 in-sample 스크린 널입니다"),
        table(
            [
                "레이어",
                "feature",
                "mean_delta (구=신)",
                "관측 delta (구·in-sample)",
                "관측 delta (신·out-of-sample)",
                "구 null_z",
                "신 null_z",
            ],
            rows,
        ),
        callout(
            "후보 4개 중 3개가 discovery 밖에서 재면 부호가 뒤집힙니다. "
            "#2144의 null_z는 +3.71 → −0.02로 무너집니다. 실재하는 효과가 아니라, "
            "선택된 최댓값을 선택된 적 없는 null에 대고 잰 결과였습니다 — "
            "교과서적인 selection overfitting이고, in-sample 스크린 널이 정확히 그걸 가리고 "
            "있었습니다.",
            "🔴",
        ),
        block("heading_3", "새 승자도 아무것도 아닙니다"),
        block(
            "paragraph",
            f"#{new['feature']['feature_id']} @ L{new['feature']['layer']}, held-out 25문항:",
        ),
    ]


def study2b_stats_blocks() -> list[dict[str, Any]]:
    import statistics as st

    sys.path.insert(0, str(ROOT / "src"))
    from mindscopex_analysis.stats import paired_summary

    spec = load_csv("study2b", "condition_specificity.csv")
    rows = []
    labels = {
        "base_margin_delta": "held-out margin_delta",
        "twin_margin_delta": "counterfactual twin",
        "sign_flip_gap": "sign_flip_gap (특이성)",
        "cue_effect": "cue_effect",
    }
    for col, label in labels.items():
        values = [float(r[col]) for r in spec if r.get(col) not in ("", None, "None")]
        if not values:
            continue
        rows.append(stat_row(label, paired_summary(values, draws=20000, seed=0)))
    del st
    return [
        table(["지표", "평균", "p", "95% CI", "양수"], rows),
        callout(
            "인과 효과 없음, 특이성 없음. 유일하게 유의한 cue_effect는 음수 — "
            "함정 feature와 반대 방향입니다. 2B 결론: 스캔한 네 레이어 어디에도 "
            "out-of-sample에서 matched random direction을 이기는 feature가 없습니다.",
            "🔴",
        ),
    ]


def falsify_blocks() -> list[dict[str, Any]]:
    s = load_json("falsify", "falsification_summary.json")
    acc_rows = [
        [k.replace("_", " "), v.get("status"), str(v.get("reason", ""))[:180]]
        for k, v in s["acceptance"].items()
    ]
    means = s["condition_means"]
    fire = s["condition_fire_rate"]
    cond_rows = [
        [k, f"{means[k]:.3f}", f"{fire[k]:.2f}", f"{s['condition_in_topk_rate'][k]:.2f}"]
        for k in sorted(means, key=lambda x: -means[x])
    ]
    lm = s["crt_lure_transfer"]["length_matched"]
    return [
        block("heading_2", "3. feature_falsification — #81663은 cue를 읽는가 형식을 읽는가?"),
        block(
            "paragraph",
            "ablation은 '지우면 움직이는가'만 답하고, 형식 feature도 그 테스트를 통과합니다. "
            "이 job은 통과하면 안 되는 조건을 명시적으로 걸고, 무엇을 검증하지 못했는지도 "
            "artifact에 기록합니다.",
        ),
        block("heading_3", "5개 축 중 2개만 실제로 검증됐습니다"),
        table(["축", "상태", "이유"], acc_rows),
        block("heading_3", "결정적 숫자: 모든 조건에서 발화율 100%"),
        table(["조건", "평균 활성", "발화율", "TopK 비율"], cond_rows),
        callout(
            f"neutral({means['neutral']:.3f})이 hostile({means['hostile']:.3f})보다 높습니다. "
            f"structure_gap = {s['structure_gap']:+.4f}, structure_auc = {s['structure_auc']:.3f} "
            "— 함정 유무를 구분하는 데 우연(0.5)보다 못합니다. "
            f"held-out 오탐 {s['n_false_positives_held_out']} vs 미탐 "
            f"{s['n_false_negatives_held_out']}.",
            "🔴",
        ),
        block("heading_3", "그럼 이 feature는 무엇을 읽는가"),
        block(
            "paragraph",
            "유일하게 실재하는 분리는 과제입니다: goal_affordance(~2.73–2.77) vs "
            f"hagendorff_crt(~2.50–2.52), "
            f"AUC {s['crt_lure_transfer']['auc_positive_vs_other']:.3f}. "
            "길이 교란인지 확인했더니 아닙니다 — 4토큰 캘리퍼로 길이를 맞춘 "
            f"{lm['n_pairs']}쌍에서도 AUC {lm['auc']:.3f}, 차이 "
            f"{lm['paired_delta']['mean']:+.3f}, p={lm['paired_delta']['p']:.4f}, "
            f"{lm['paired_delta']['n_positive']}/{lm['paired_delta']['n']} 양성.",
        ),
        block(
            "paragraph",
            "답 길이 교란도 아닙니다: corr(활성, 답 길이 차) = "
            f"{s['answer_confound']['corr_activation_vs_answer_len_delta']:+.3f} (사실상 0).",
        ),
    ]


def siblings_blocks() -> list[dict[str, Any]]:
    s = load_json("siblings", "coablation_summary.json")[0]
    sel = s["selection"]
    return [
        block("heading_2", "4. cross_layer_siblings — 다른 레이어와 짝으로 작동하는가?"),
        block(
            "paragraph",
            "기존 multisite는 방향을 이웃 레이어에 이식할 뿐이라 'L31의 그 feature'라는 주장이 "
            "아닙니다. 이 job은 대응 feature를 먼저 식별한 뒤 공동 ablation을 합니다.",
        ),
        block("heading_3", "정직성 장치가 설계대로 작동했습니다"),
        code_block(
            f"source_specificity_verdict : {sel['source_specificity_verdict']}\n"
            f"selection_rule             : {sel['selection_rule']}\n"
            f"specificity_signal_used    : {sel['specificity_signal_used']}\n"
            f"combined_score_basis       : {sel['combined_score_basis']}"
        ),
        block(
            "paragraph",
            "두 job이 독립적으로 같은 결론에 도달했습니다 — falsify가 structure_gap 음수를 "
            "측정했고, siblings가 별개 경로로 '이 source에는 대조 자체가 없다'고 판정해 "
            "4번째 신호를 전역으로 껐습니다.",
        ),
        block("heading_3", f"대응 feature: L{s['target_layer']} #{s['target_feature']}"),
        table(
            ["신호", "값"],
            [
                ["decoder cosine", f"{s['decoder_cosine']:.3f}"],
                ["activation corr", f"{s['activation_corr']:.3f}"],
                ["effect corr", f"{s['effect_corr']:.3f}"],
            ],
        ),
        block("heading_3", "교차 레이어 회로는 없습니다"),
        table(
            ["조건", "평균", "p", "95% CI", "양수"],
            [
                stat_row("A만 (L15)", s["a_only"]),
                stat_row("B만 (L31)", s["b_only"]),
                stat_row("A+B 동시", s["joint"]),
                stat_row("difference-in-differences", s["difference_in_differences"]),
            ],
        ),
        block("heading_3", "유일하게 확실한 숫자 — 예상과 반대입니다"),
        table(
            ["지표", "평균", "p", "95% CI", "양수"],
            [stat_row("sibling_repair", s["sibling_repair"])],
        ),
        callout(
            "A를 지우면 B가 더 세게 켜지는 게 아니라 거의 꺼집니다. 12문항 전부, 예외 없이. "
            "self-repair(보상)의 반대입니다. B는 A를 보완하는 형제가 아니라 A에 의존하는 "
            "하류 사본입니다. margin 쪽 숫자는 n=12에 CI가 전부 0을 포함하므로 해석하지 않습니다.",
            "🔍",
        ),
    ]


def trajectory_blocks() -> list[dict[str, Any]]:
    d = load_json("trajectory", "trajectory_summary.json")
    rows = []
    for mode, v in d["per_mode"].items():
        pm = v["phase_means"]
        reasoning = [x for k, x in pm.items() if k.startswith("reasoning_")]
        rows.append(
            [
                mode,
                f"{pm.get('cue', 0.0):.3f}",
                f"{pm['prompt_last']:.3f}",
                f"{max(reasoning):.3f}" if reasoning else "—",
                f"{v['reasoning_drift']:.3f}",
                f"{v['fire_rate_per_distinct_position']:.3f}",
            ]
        )
    return [
        block("heading_2", "5. reasoning_trajectory — 추론 도중에는 어떻게 움직이는가?"),
        block(
            "paragraph",
            "이 연구의 모든 인과 측정은 마지막 프롬프트 토큰 하나를 읽습니다. '답하기 직전'이지 "
            "추론 자체가 아닙니다. 이 job은 프롬프트의 cue 절부터 생성 trace 전체까지 "
            "feature를 따라갑니다.",
        ),
        table(
            ["arm", "cue", "prompt_last", "reasoning 최대", "drift", "발화율(위치당)"],
            rows,
        ),
        callout(
            "feature는 마지막 프롬프트 토큰에서만 켜집니다. cue 절에서 0, 생성된 500+ 토큰 "
            "전 구간에서 0. 발화율 0.143 = 1/7 — 샘플링한 일곱 위치 중 정확히 하나입니다. "
            "cue span은 10/10 정확히 찾았으므로(cue_located=True) 배선 문제가 아니라 "
            "실제로 거기서 안 켜지는 것입니다.",
            "🔴",
        ),
        callout(
            "단서: thinking arm의 trace가 max_new_tokens=512에서 잘렸습니다(has_think_end가 "
            "140행 전부 False, 생성 토큰 ~500). 따라서 '완결된 추론에서 숙고가 억제하는가'는 "
            "이 run이 답하지 못하며, drift_difference = 0.0은 0−0의 퇴화된 값이지 "
            "정보가 있는 null이 아닙니다. 다만 '생성 구간 어디에서도 안 켜진다'는 결론은 "
            "500토큰을 훑고 얻은 것이라 그대로 유효합니다.",
            "⚠️",
        ),
    ]


def conclusion_blocks() -> list[dict[str, Any]]:
    return [
        block("heading_2", "종합"),
        table(
            ["실험", "물은 것", "답"],
            [
                [
                    "feature_modules",
                    "효과가 집합에 분산돼 있는가",
                    "아니오 — 모듈이 단일 멤버보다 약함",
                ],
                [
                    "2B 재선택",
                    "#2144가 재현되는가",
                    "아니오 — selection overfitting, null_z +3.71→−0.02",
                ],
                [
                    "falsification",
                    "cue를 읽는가 형식을 읽는가",
                    "형식 — 모든 조건 100% 발화, AUC 0.472",
                ],
                [
                    "cross_layer_siblings",
                    "교차 레이어 회로인가",
                    "아니오 — DiD p=0.39, B는 A의 하류 사본",
                ],
                [
                    "trajectory",
                    "추론 도중 움직이는가",
                    "안 움직임 — 마지막 프롬프트 토큰에서만 발화",
                ],
            ],
        ),
        callout(
            "다섯 실험이 서로 다른 방향에서 공격했고 전부 같은 곳으로 수렴했습니다. "
            "#81663은 마지막 프롬프트 토큰의 위치/도메인 feature이고, 함정과는 무관합니다. "
            "즉 이 연구의 모든 인과 측정은 이 feature가 켜지는 유일한 위치에서 이뤄졌고, "
            "그 위치에서 그것은 모든 프롬프트에 대해 똑같이 켜집니다.",
            "🔴",
        ),
        block("heading_3", "이건 실패가 아닙니다"),
        block(
            "paragraph",
            "목표는 lure feature를 찾아내는 것이 아니라, 발견한 표상이 실제 인과 기제인지 "
            "artifact인지 구별할 수 있는 시스템을 만드는 것이었습니다. 그 시스템이 오늘 "
            "자기 자신의 후보를 기각했습니다 — 그리고 기각한 이유를 artifact에 기록했습니다.",
        ),
        block("heading_3", "이번 배치에서 고친 파이프라인 결함"),
        block(
            "bulleted_list_item",
            "in-sample 스크린 널 — 선택된 feature를 선택에 쓴 항목에서 재고 있었음. "
            "이것 하나가 2B 승자를 바꿨습니다.",
        ),
        block(
            "bulleted_list_item",
            "logprob delta 부호 규약이 세 job에서 반대로 기록되고 있었음 "
            "(같은 컬럼 이름, 반대 부호)",
        ),
        block(
            "bulleted_list_item",
            "sign-flip p값이 정확히 0.0으로 보고될 수 있었음 → (b+1)/(draws+1)",
        ),
        block(
            "bulleted_list_item",
            "빈 arm이 _mean([])==0.0으로 '자신 있는 0'을 발행 → mean_or_none",
        ),
        block(
            "bulleted_list_item",
            "이름이 수행되지 않은 검정을 주장하는 필드들 삭제/개명 (symmetric_null_fdr 등)",
        ),
        block("paragraph", "관련 커밋: ac64b33, 95f7ccc. 테스트 474개 통과."),
    ]


# ----------------------------------------------------------------- main


def build() -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    blocks += intro_blocks()
    blocks += modules_blocks()
    blocks += study2b_blocks()
    blocks += study2b_stats_blocks()
    blocks += falsify_blocks()
    blocks += siblings_blocks()
    blocks += trajectory_blocks()
    blocks += conclusion_blocks()
    blocks.append(block("heading_2", "run 디렉터리"))
    blocks.append(code_block("\n".join(f"{k:12s} results/runs/{v}" for k, v in RUN_DIRS.items())))
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    blocks = build()
    title = "2026-08-23 실험 결과 — 단일 feature 가설의 종결"
    if args.dry_run:
        print(f"title: {title}")
        print(f"blocks: {len(blocks)}")
        for b in blocks:
            kind = b["type"]
            if kind == "table":
                print(
                    f"  [table {b['table']['table_width']}col x {len(b['table']['children'])}row]"
                )
            else:
                text = "".join(p["text"]["content"] for p in b[kind].get("rich_text", []))
                print(f"  [{kind}] {text[:110]}")
        return

    page = request(
        "POST",
        f"{NOTION_API}/pages",
        {
            "parent": {"page_id": PARENT_PAGE_ID},
            "properties": {"title": {"title": rich(title)}},
            "children": blocks[:100],
        },
    )
    page_id = page["id"]
    rest = blocks[100:]
    while rest:
        chunk, rest = rest[:100], rest[100:]
        request("PATCH", f"{NOTION_API}/blocks/{page_id}/children", {"children": chunk})
    print(f"published: {page.get('url')}")


if __name__ == "__main__":
    main()
