"""Build public and project-generated reasoning-lure datasets and normalize
them into the repository's uniform lure-case JSON schema.

Run all builders::

    uv run python scripts/build_datasets.py

Run a single source::

    uv run python scripts/build_datasets.py hagendorff

Every builder writes a self-describing JSON file into
``src/mindscopex_analysis/data`` that :mod:`mindscopex_analysis.lure_datasets`
can load. The JSON is committed to the repository so notebooks do not need
network access; re-run this script only when refreshing a source.

Schema (one object per file)::

    {
      "dataset_id": str,          # stable id, also the loader key
      "title": str,
      "description": str,
      "task_kind": "crt" | "semantic_illusion",
      "scoring": "logprob_margin" | "premise_rejection" | "binary_choice",
      "source": {authors, year, title, venue, doi, project_url,
                 download_url, source_sha256, license, license_note},
      "generated_by": "scripts/build_datasets.py",
      "n_cases": int,
      "family_counts": {family: count},
      "cases": [
        {case_id, pair_id?, template_id?, condition?, family,
         question, correct_answer, lure_answer,
         control_question?, reference_answer?, note?}
      ]
    }

``correct_answer`` / ``lure_answer`` are stored as bare surface forms; the
loader adds the leading space and the ``\\nAnswer:`` delimiter that the
logprob scorer expects. Semantic-illusion items use empty answer strings
(they are scored by premise rejection, not string margin) and keep the
authoritative correction in ``reference_answer``.
"""
# ruff: noqa: E501 -- this module transcribes verbatim benchmark items; long
# question/source strings are kept on one line to preserve the source wording.

from __future__ import annotations

import argparse
import ast
import difflib
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "src" / "mindscopex_analysis" / "data"

# Allow ``import mindscopex_analysis`` when run without an editable install.
sys.path.insert(0, str(REPO_ROOT / "src"))

SCHEMA_VERSION = 1
GENERATED_BY = "scripts/build_datasets.py"

_CASE_KEY_ORDER = (
    "case_id",
    "pair_id",
    "template_id",
    "condition",
    "family",
    "question",
    "correct_answer",
    "lure_answer",
    "control_question",
    "reference_answer",
    "rationale",
    "revision",
    "note",
)


# --------------------------------------------------------------------------
# shared normalized writer
# --------------------------------------------------------------------------
def _ordered_case(case: dict[str, Any]) -> dict[str, Any]:
    unknown = set(case) - set(_CASE_KEY_ORDER)
    if unknown:
        raise ValueError(f"case {case.get('case_id')!r} has unknown keys {sorted(unknown)}")
    return {key: case[key] for key in _CASE_KEY_ORDER if key in case and case[key] != ""}


def _validate_cases(cases: list[dict[str, Any]], *, scoring: str) -> None:
    seen: set[str] = set()
    for case in cases:
        cid = case.get("case_id")
        if not isinstance(cid, str) or not cid:
            raise ValueError(f"case is missing a string case_id: {case!r}")
        if cid in seen:
            raise ValueError(f"duplicate case_id {cid!r}")
        seen.add(cid)
        for field in ("family", "question"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"case {cid!r} has empty {field!r}")
        if scoring in {"binary_choice", "logprob_margin"}:
            correct = case.get("correct_answer", "")
            lure = case.get("lure_answer", "")
            if not correct.strip() or not lure.strip():
                raise ValueError(f"case {cid!r} needs non-empty correct/lure for {scoring}")
            if correct.strip().casefold() == lure.strip().casefold():
                raise ValueError(f"case {cid!r} has identical correct and lure answers")


def write_dataset(
    *,
    dataset_id: str,
    title: str,
    description: str,
    task_kind: str,
    scoring: str,
    source: dict[str, Any],
    cases: list[dict[str, Any]],
    schema_version: int = SCHEMA_VERSION,
) -> Path:
    """Validate and write one normalized dataset file; return its path."""

    if scoring not in {"binary_choice", "logprob_margin", "premise_rejection"}:
        raise ValueError(f"unknown scoring {scoring!r}")
    _validate_cases(cases, scoring=scoring)

    family_counts = dict(sorted(Counter(case["family"] for case in cases).items()))
    payload = {
        "dataset_id": dataset_id,
        "schema_version": int(schema_version),
        "title": title,
        "description": description,
        "task_kind": task_kind,
        "scoring": scoring,
        "source": source,
        "generated_by": GENERATED_BY,
        "n_cases": len(cases),
        "family_counts": family_counts,
        "cases": [_ordered_case(case) for case in cases],
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / f"{dataset_id}.json"
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] {dataset_id}: {len(cases)} cases -> {destination.relative_to(REPO_ROOT)}")
    for family, count in family_counts.items():
        print(f"       {family}: {count}")
    return destination


# --------------------------------------------------------------------------
# Hagendorff, Fabi & Kosinski (2023) — Nature Computational Science
# --------------------------------------------------------------------------
HAGENDORFF_SOURCE = {
    "authors": "Hagendorff, T., Fabi, S. & Kosinski, M.",
    "year": 2023,
    "title": (
        "Human-like intuitive behavior and reasoning biases emerged in large "
        "language models but disappeared in ChatGPT"
    ),
    "venue": "Nature Computational Science 3, 833-838",
    "doi": "10.1038/s43588-023-00527-x",
    "project_url": "https://osf.io/w5vhp/",
    "download_url": "https://osf.io/download/z6kmw/",
    "source_file": "LLMs_responses.py",
    "source_sha256": "cdf4617e8dec63546762cbe2b3cae6b6c7f640adfb6002bf5fc226f5871a4125",
    "license": "CC BY 4.0 (article and supplementary materials)",
    "license_note": (
        "The OSF project has no node-level license tag; the article and its "
        "supplementary materials are published under CC BY 4.0. Cite the paper "
        "and the OSF project when publishing results."
    ),
}

_HAGENDORFF_TYPES = (
    # (hostile var, control var, family, type label)
    ("crt1", "crt_not_hostile", "crt_difference", "CRT1 (price difference)"),
    ("crt2", "crt2_not_hostile", "crt_rate", "CRT2 (work rate)"),
    ("crt3", "crt3_not_hostile", "crt_growth", "CRT3 (exponential growth)"),
)


def _hagendorff_source_text() -> str:
    """Return the checksum-verified OSF source, downloading it if needed."""

    from mindscopex_analysis.datasets import download_hagendorff_crt150_source

    path = download_hagendorff_crt150_source()
    return path.read_text(encoding="utf-8")


def _literal_lists(source: str, names: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Parse the named top-level list literals without executing the module."""

    starts: dict[str, int] = {}
    for match in re.finditer(r"(?m)^([A-Za-z_]\w*)\s*=", source):
        starts.setdefault(match.group(1), match.start())
    ordered = sorted(starts.values())
    result: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        if name not in starts:
            raise ValueError(f"Hagendorff source is missing {name!r}")
        start = starts[name]
        end = min((offset for offset in ordered if offset > start), default=len(source))
        value = ast.literal_eval(ast.parse(source[start:end]).body[0].value)
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise TypeError(f"{name!r} is not a list of records")
        result[name] = value
    return result


def build_hagendorff_crt() -> Path:
    source = _hagendorff_source_text()
    needed = [name for pair in _HAGENDORFF_TYPES for name in pair[:2]]
    blocks = _literal_lists(source, needed)

    cases: list[dict[str, Any]] = []
    for hostile_var, control_var, family, label in _HAGENDORFF_TYPES:
        hostile = blocks[hostile_var]
        control_by_number = {row["number"]: row["task"].strip() for row in blocks[control_var]}
        if len(hostile) != 50:
            raise ValueError(f"expected 50 {hostile_var} items, got {len(hostile)}")
        for row in hostile:
            number = int(row["number"])
            correct = str(row["correct"]).strip()
            lure = str(row["intuitive"]).strip()
            control = control_by_number.get(number)
            if not (correct and lure and control):
                raise ValueError(f"{hostile_var} item {number} is incomplete")
            cases.append(
                {
                    "case_id": f"hagendorff_{family}_{number:03d}",
                    "family": family,
                    "question": row["task"].strip(),
                    "correct_answer": correct,
                    "lure_answer": lure,
                    "control_question": control,
                    "note": (
                        f"{label} item {number}. Matched control ({control_var}) "
                        f"removes the trap so the intuitive answer is correct."
                    ),
                }
            )

    return write_dataset(
        dataset_id="hagendorff_crt",
        title="Hagendorff et al. (2023) CRT-150 (hostile items + matched controls)",
        description=(
            "150 bespoke cognitive-reflection tasks (50 each of price-difference, "
            "work-rate, and exponential-growth types) designed to elicit an "
            "intuitive but wrong answer. Each item embeds its matched non-hostile "
            "control as control_question (same surface, trap removed)."
        ),
        task_kind="crt",
        scoring="logprob_margin",
        source=HAGENDORFF_SOURCE,
        cases=cases,
    )


def build_hagendorff_semantic_illusion() -> Path:
    source = _hagendorff_source_text()
    blocks = _literal_lists(source, ["si", "six_sanity"])
    sanity_by_number = {row["number"]: row["task"].strip() for row in blocks["six_sanity"]}

    cases: list[dict[str, Any]] = []
    for row in blocks["si"]:
        number = int(row["number"])
        correction = str(row.get("correct", "")).strip()
        cases.append(
            {
                "case_id": f"hagendorff_semantic_illusion_{number:03d}",
                "family": "semantic_illusion",
                "question": row["task"].strip(),
                "correct_answer": "",
                "lure_answer": "",
                "control_question": sanity_by_number.get(number, ""),
                "reference_answer": correction,
                "note": (
                    f"Semantic illusion item {number}. A correct response rejects the "
                    f"false premise; the intuitive failure answers inside the false frame. "
                    f"Control (six_sanity) asks the non-misleading version."
                ),
            }
        )

    return write_dataset(
        dataset_id="hagendorff_semantic_illusion",
        title="Hagendorff et al. (2023) semantic illusions",
        description=(
            "50 false-premise trivia questions ('Who is the dictator of South "
            "Korea?'). Scored by premise rejection, not string margin: the source "
            "gives a free-form correction (reference_answer) and a matched "
            "non-misleading control (control_question), but no short lure token."
        ),
        task_kind="semantic_illusion",
        scoring="premise_rejection",
        source=HAGENDORFF_SOURCE,
        cases=cases,
    )


# --------------------------------------------------------------------------
# small canonical CRT sets — verbatim, source-cited transcriptions
#
# These instruments are only a handful of items each and are not distributed as
# machine-readable files (they live in article bodies / supplementary PDFs), so
# the items are transcribed here with their source. Problem text follows the
# cited source; a unit cue ("Answer in X.") is appended to match the repo's
# LureCase answer convention, and answers are the source's correct / intuitive
# keys.
# --------------------------------------------------------------------------
def _numbered_cases(prefix: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"case_id": f"{prefix}_{index:03d}", **row} for index, row in enumerate(rows, start=1)]


CRT2_SOURCE = {
    "authors": "Thomson, K. S. & Oppenheimer, D. M.",
    "year": 2016,
    "title": "Investigating an alternate form of the cognitive reflection test",
    "venue": "Judgment and Decision Making 11(1), 99-113",
    "doi": "",
    "project_url": "https://journal.sjdm.org/15/151029/jdm151029.html",
    "download_url": "https://journal.sjdm.org/15/151029/jdm151029.pdf",
    "license": "CC BY 3.0",
    "license_note": (
        "The authors license the article under CC BY 3.0. The four items are "
        "transcribed verbatim from the article body."
    ),
}


def build_crt2() -> Path:
    rows = [
        {
            "family": "crt_verbal",
            "question": "If you're running a race and you pass the person in second place, what place are you in?",
            "correct_answer": "second",
            "lure_answer": "first",
            "note": "CRT-2 item 1 (race). Passing 2nd place leaves you in 2nd, not 1st.",
        },
        {
            "family": "crt_verbal",
            "question": "A farmer had 15 sheep and all but 8 died. How many are left?",
            "correct_answer": "8",
            "lure_answer": "7",
            "note": "CRT-2 item 2 (sheep). 'All but 8' means 8 remain.",
        },
        {
            "family": "crt_verbal",
            "question": "Emily's father has three daughters. The first two are named April and May. What is the third daughter's name?",
            "correct_answer": "Emily",
            "lure_answer": "June",
            "note": "CRT-2 item 3 (Emily). The third daughter is Emily herself.",
        },
        {
            "family": "crt_verbal",
            "question": "How many cubic feet of dirt are there in a hole that is 3 feet deep by 3 feet wide by 3 feet long?",
            "correct_answer": "none",
            "lure_answer": "27",
            "note": "CRT-2 item 4 (hole). A hole contains no dirt.",
        },
    ]
    return write_dataset(
        dataset_id="crt2",
        title="CRT-2 (Thomson & Oppenheimer 2016)",
        description=(
            "The 4-item alternate Cognitive Reflection Test, designed to rely less "
            "on numeracy than the original CRT and to be less familiar. Verbal "
            "insight traps with a salient intuitive lure."
        ),
        task_kind="crt",
        scoring="logprob_margin",
        source=CRT2_SOURCE,
        cases=_numbered_cases("crt2", rows),
    )


VERBAL_CRT_SOURCE = {
    "authors": "Sirota, M., Dewberry, C., Juanchich, M., Valus, L. & Marshall, A. C.",
    "year": 2021,
    "title": (
        "Measuring cognitive reflection without maths: Development and validation "
        "of the verbal cognitive reflection test"
    ),
    "venue": "Journal of Behavioral Decision Making 34(3), 322-343",
    "doi": "10.1002/bdm.2213",
    "project_url": "https://osf.io/xehbv/",
    "download_url": "https://osf.io/download/64x92/",
    "license": "CC BY 4.0",
    "license_note": (
        "OSF node licensed CC BY 4.0. The 10 items are transcribed verbatim from "
        "the Supplementary Materials PDF (p.6). Response CSVs code answers as "
        "1=correct, 2=intuitive lure, 3=other."
    ),
}


def build_verbal_crt() -> Path:
    rows = [
        {
            "family": "crt_verbal",
            "question": "Mary's father has 5 daughters but no sons - Nana, Nene, Nini, Nono. What is the fifth daughter's name probably?",
            "correct_answer": "Mary",
            "lure_answer": "Nunu",
            "note": "CRT-V item 1. The fifth daughter is Mary.",
        },
        {
            "family": "crt_verbal",
            "question": "If you were running a race, and you passed the person in 2nd place, what place would you be in now?",
            "correct_answer": "2nd",
            "lure_answer": "1st",
            "note": "CRT-V item 2.",
        },
        {
            "family": "crt_verbal",
            "question": "It's a stormy night and a plane takes off from JFK airport in New York. The storm worsens, and the plane crashes - half lands in the United States, the other half lands in Canada. In which country do you bury the survivors?",
            "correct_answer": "you do not bury survivors",
            "lure_answer": "the United States",
            "note": "CRT-V item 3. Survivors are not buried.",
        },
        {
            "family": "crt_verbal",
            "question": "A monkey, a squirrel, and a bird are racing to the top of a coconut tree. Who will get the banana first, the monkey, the squirrel, or the bird?",
            "correct_answer": "there is no banana on a coconut tree",
            "lure_answer": "the bird",
            "note": "CRT-V item 4. Coconut trees have no bananas.",
        },
        {
            "family": "crt_verbal",
            "question": "In a one-storey pink house, there was a pink person, a pink cat, a pink fish, a pink computer, a pink chair, a pink table, a pink telephone, a pink shower - everything was pink! What colour were the stairs probably?",
            "correct_answer": "there are no stairs in a one-storey house",
            "lure_answer": "pink",
            "note": "CRT-V item 5. A one-storey house has no stairs.",
        },
        {
            "family": "crt_verbal",
            "question": "How many of each animal did Moses put on the ark?",
            "correct_answer": "none",
            "lure_answer": "two",
            "note": "CRT-V item 6 (Moses illusion). Noah, not Moses, built the ark.",
        },
        {
            "family": "crt_verbal",
            "question": "The wind blows west. An electric train runs east. In which cardinal direction does the smoke from the locomotive blow?",
            "correct_answer": "an electric train produces no smoke",
            "lure_answer": "west",
            "note": "CRT-V item 7. Electric trains produce no smoke.",
        },
        {
            "family": "crt_verbal",
            "question": "If you have only one match and you walk into a dark room where there is an oil lamp, a newspaper and wood - which thing would you light first?",
            "correct_answer": "the match",
            "lure_answer": "the oil lamp",
            "note": "CRT-V item 8. You light the match first.",
        },
        {
            "family": "crt_verbal",
            "question": "Would it be ethical for a man to marry the sister of his widow?",
            "correct_answer": "it is not possible",
            "lure_answer": "no",
            "note": "CRT-V item 9. A man with a widow is dead, so he cannot marry.",
        },
        {
            "family": "crt_verbal",
            "question": 'Which sentence is correct: a) "the yolk of the egg are white" or b) "the yolk of the egg is white"?',
            "correct_answer": "neither, the yolk is yellow",
            "lure_answer": "b",
            "note": "CRT-V item 10. Egg yolks are yellow, not white.",
        },
    ]
    return write_dataset(
        dataset_id="verbal_crt",
        title="Verbal CRT / CRT-V (Sirota et al. 2021)",
        description=(
            "A 10-item non-mathematical Cognitive Reflection Test. Each item has a "
            "salient intuitive lure and a reflective correct answer, isolating "
            "cognitive reflection from numeracy. Answers are heterogeneous phrases; "
            "several items are premise-rejection style and can also be scored by a "
            "generation judge rather than logprob margin."
        ),
        task_kind="crt",
        scoring="logprob_margin",
        source=VERBAL_CRT_SOURCE,
        cases=_numbered_cases("verbal_crt", rows),
    )


CRT7_SOURCE = {
    "authors": "Frederick, S. (2005); Toplak, M. E., West, R. F. & Stanovich, K. E. (2014)",
    "year": 2014,
    "title": "Cognitive Reflection Test, 7-item form (Frederick's 3 originals + Toplak et al.'s 4-item expansion)",
    "venue": "J. Economic Perspectives 19(4), 25-42; Thinking & Reasoning 20(2), 147-168",
    "doi": "10.1080/13546783.2013.844729",
    "project_url": "https://doi.org/10.1257/089533005775196732",
    "license": "No open data license (journal articles)",
    "license_note": (
        "Stimuli quoted from Frederick (2005) and Toplak et al. (2014) under "
        "academic fair use; the same 7 items are also distributed (in GBP) in the "
        "CC-BY companion of Yax et al. (2024). The source articles carry no open "
        "data license, so treat this set as fair-use reference material."
    ),
}


def build_crt7_classic() -> Path:
    rows = [
        {
            "family": "crt_difference",
            "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Answer in cents.",
            "correct_answer": "5 cents",
            "lure_answer": "10 cents",
            "note": "CRT-7 item 1 (Frederick 2005, bat-and-ball).",
        },
        {
            "family": "crt_rate",
            "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Answer in minutes.",
            "correct_answer": "5 minutes",
            "lure_answer": "100 minutes",
            "note": "CRT-7 item 2 (Frederick 2005, machines/widgets).",
        },
        {
            "family": "crt_growth",
            "question": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake? Answer in days.",
            "correct_answer": "47 days",
            "lure_answer": "24 days",
            "note": "CRT-7 item 3 (Frederick 2005, lily pads).",
        },
        {
            "family": "crt_rate",
            "question": "If John can drink one barrel of water in 6 days, and Mary can drink one barrel of water in 12 days, how long would it take them to drink one barrel of water together? Answer in days.",
            "correct_answer": "4 days",
            "lure_answer": "9 days",
            "note": "CRT-7 item 4 (Toplak et al. 2014, shared-rate).",
        },
        {
            "family": "crt_counting",
            "question": "Jerry received both the 15th highest and the 15th lowest mark in the class. How many students are in the class?",
            "correct_answer": "29 students",
            "lure_answer": "30 students",
            "note": "CRT-7 item 5 (Toplak et al. 2014, class rank).",
        },
        {
            "family": "crt_arithmetic",
            "question": "A man buys a pig for $60, sells it for $70, buys it back for $80, and sells it finally for $90. How much has he made? Answer in dollars.",
            "correct_answer": "20 dollars",
            "lure_answer": "10 dollars",
            "note": "CRT-7 item 6 (Toplak et al. 2014, pig trading).",
        },
        {
            "family": "crt_percentage",
            "question": "Simon decided to invest $8,000 in the stock market one day early in 2008. Six months after he invested, on July 17, the stocks he had purchased were down 50%. Fortunately for Simon, from July 17 to October 17, the stocks he had purchased went up 75%. At this point, Simon has: (a) broken even in the stock market, (b) is ahead of where he began, or (c) has lost money.",
            "correct_answer": "c",
            "lure_answer": "b",
            "note": "CRT-7 item 7 (Toplak et al. 2014, Simon stocks). $8,000 -> $4,000 -> $7,000, so he lost money (c).",
        },
    ]
    return write_dataset(
        dataset_id="crt7_classic",
        title="Classic CRT-7 (Frederick 2005 + Toplak et al. 2014)",
        description=(
            "The canonical 7-item Cognitive Reflection Test: Frederick's three "
            "originals (bat-and-ball, machines, lily pads) plus Toplak et al.'s "
            "four-item numeric expansion. The standard reference instrument; note "
            "several items also appear in crt_pilot."
        ),
        task_kind="crt",
        scoring="logprob_margin",
        source=CRT7_SOURCE,
        cases=_numbered_cases("crt7", rows),
    )


YAX_SOURCE = {
    "authors": "Yax, N. et al.",
    "year": 2024,
    "title": "Studying and improving reasoning in humans and machines (novel CRT isomorphs)",
    "venue": "Communications Psychology",
    "doi": "10.1038/s44271-024-00091-8",
    "project_url": "https://github.com/hrl-team/ReasoningGPT",
    "license": "Article CC BY 4.0; companion code/data repo GPL-3.0",
    "license_note": (
        "Novel CRT isomorph stimuli transcribed from the open-access article. The "
        "companion GitHub repo (response matrix, .npy inputs) is GPL-3.0 and is NOT "
        "vendored here; only the published stimuli are included. The source also "
        "provides matched pure-math controls (Experiment='crt-math') and a "
        "solved-example condition that can be fetched from the repo if needed."
    ),
}


def build_yax_crt_isomorph() -> Path:
    rows = [
        {
            "family": "crt_difference",
            "question": "A scarf costs 210 euros more than a hat. The scarf and the hat cost 220 euros in total. How much does the hat cost? Answer in euros.",
            "correct_answer": "5 euros",
            "lure_answer": "10 euros",
            "note": "Yax new-CRT isomorph of bat-and-ball.",
        },
        {
            "family": "crt_rate",
            "question": "How long would it take 80 carpenters to repair 80 tables, if it takes 8 carpenters 8 hours to repair 8 tables? Answer in hours.",
            "correct_answer": "8 hours",
            "lure_answer": "80 hours",
            "note": "Yax new-CRT isomorph of machines/widgets.",
        },
        {
            "family": "crt_growth",
            "question": "An entire forest was consumed by a wildfire in 40 hours, with its size doubling every hour. How long did it take to burn 50% of the forest? Answer in hours.",
            "correct_answer": "39 hours",
            "lure_answer": "20 hours",
            "note": "Yax new-CRT isomorph of lily pads.",
        },
        {
            "family": "crt_rate",
            "question": "If Andrea can clean a house in 3 hours, and Alex can clean a house in 6 hours, how many hours would it take for them to clean a house together? Answer in hours.",
            "correct_answer": "2 hours",
            "lure_answer": "9 hours",
            "note": "Yax new-CRT isomorph of shared-rate.",
        },
        {
            "family": "crt_counting",
            "question": "A runner participates in a marathon and arrives both at the 100th highest and the 100th lowest position. How many participants are in the marathon?",
            "correct_answer": "199 participants",
            "lure_answer": "200 participants",
            "note": "Yax new-CRT isomorph of class rank.",
        },
        {
            "family": "crt_arithmetic",
            "question": "A woman buys a second-hand car for $1000, then sells it for $2000. Later she buys it back for $3000 and finally sells it for $4000. How much has she made? Answer in dollars.",
            "correct_answer": "2000 dollars",
            "lure_answer": "1000 dollars",
            "note": "Yax new-CRT isomorph of pig trading.",
        },
        {
            "family": "crt_percentage",
            "question": "Frank decided to invest $10,000 into bitcoin in January 2018. Four months after he invested, the bitcoin he had purchased went down 50%. In the subsequent eight months, the bitcoin he had purchased went up 80%. What is the value of Frank's bitcoin after one year? Answer in dollars.",
            "correct_answer": "9000 dollars",
            "lure_answer": "18000 dollars",
            "note": "Yax new-CRT isomorph of the stock-market item.",
        },
    ]
    return write_dataset(
        dataset_id="yax_crt_isomorph",
        title="Yax et al. (2024) novel CRT isomorphs",
        description=(
            "7 novel CRT isomorphs matched one-to-one to the classic CRT-7 "
            "structures but with fresh surface stories, built to reduce pretraining "
            "contamination. Pairs with crt7_classic for an original-vs-isomorph "
            "contrast; the source also ships matched pure-math controls."
        ),
        task_kind="crt",
        scoring="logprob_margin",
        source=YAX_SOURCE,
        cases=_numbered_cases("yax", rows),
    )


# --------------------------------------------------------------------------
# MindScopeX fresh synthetic CRT isomorph pilot (2026)
# --------------------------------------------------------------------------
CRT_FRESH_V1_SOURCE = {
    "authors": "MindScopeX project",
    "year": 2026,
    "title": "CRT Fresh Isomorphs v1",
    "venue": "Repository-generated synthetic benchmark",
    "doi": "",
    "project_url": "",
    "download_url": "",
    "license": "Apache-2.0 (repository-generated content)",
    "license_note": (
        "Deterministically generated from closed-form CRT templates in this builder. "
        "No model outputs were used to define the questions, correct answers, lure "
        "answers, or matched controls. Cite the repository version and generation date."
    ),
}

_FRESH_DIFFERENCE_PAIRS = (
    ("portable projector", "wireless presenter"),
    ("camping stove", "metal cup"),
    ("museum annual pass", "audio guide rental"),
    ("mechanical keyboard", "mouse pad"),
    ("bicycle helmet", "water bottle cage"),
    ("chef's knife", "vegetable peeler"),
    ("desk lamp", "cable organizer"),
    ("telescope", "tripod adapter"),
    ("hiking backpack", "rain cover"),
    ("board game", "card sleeve pack"),
)

_FRESH_RATE_PROCESSES = (
    ("3D printers", "print", "prototype shells"),
    ("laser cutters", "cut", "acrylic panels"),
    ("baristas", "prepare", "iced drinks"),
    ("packing robots", "seal", "shipping boxes"),
    ("document scanners", "scan", "folders"),
    ("textile looms", "weave", "scarves"),
    ("labeling machines", "label", "bottles"),
    ("engraving stations", "engrave", "nameplates"),
    ("test benches", "test", "circuit boards"),
    ("photo printers", "print", "photo books"),
)

_FRESH_GROWTH_SUBJECTS = (
    "duckweed patch on a pond",
    "mold culture on a plate",
    "blue algae patch in a tank",
    "digital tile pattern on a display",
    "ground-cover plant in a greenhouse bed",
    "bacterial colony on an agar tray",
    "floating fern patch in a reservoir",
    "simulated wildfire region on a map",
    "crystal pattern in a lab dish",
    "lichen patch on a test surface",
)


def _fresh_crt_cases() -> list[dict[str, Any]]:
    """Create a deterministic, closed-form-validated 30-item CRT pilot."""

    cases: list[dict[str, Any]] = []

    # If total = 2 * small + difference, the reflective answer is ``small``.
    # The subtraction lure is total - difference = 2 * small. In the control,
    # the expensive item's price is stated directly, so the same lure becomes
    # the correct answer.
    for index, (expensive, inexpensive) in enumerate(_FRESH_DIFFERENCE_PAIRS, start=1):
        small = 7 + 3 * (index - 1)
        difference = 26 + 5 * (index - 1)
        total = 2 * small + difference
        lure = total - difference
        control_expensive = difference
        control_answer = total - control_expensive
        assert lure == 2 * small
        assert control_answer == lure
        cases.append(
            {
                "case_id": f"crt_fresh_v1_difference_{index:03d}",
                "family": "crt_difference",
                "question": (
                    f"At a shop, the combined price of a {expensive} and a {inexpensive} "
                    f"is ${total}. The {expensive} costs ${difference} more than the "
                    f"{inexpensive}. What is the price of the {inexpensive}? Answer in dollars."
                ),
                "correct_answer": f"${small}",
                "lure_answer": f"${lure}",
                "control_question": (
                    f"At a shop, the combined price of a {expensive} and a {inexpensive} "
                    f"is ${total}. The {expensive} costs ${control_expensive}. What is the "
                    f"price of the {inexpensive}? Answer in dollars."
                ),
                "note": (
                    "template=difference; validation=closed_form; "
                    "control_answer_equals_lure=true"
                ),
            }
        )

    # ``base`` agents make ``base`` outputs in ``base`` minutes. Therefore each
    # agent makes one output in ``base`` minutes: target agents make target
    # outputs in base minutes. Keeping only base agents makes target outputs in
    # target minutes, which turns the lure into the matched-control answer.
    for index, (agents, verb, outputs) in enumerate(_FRESH_RATE_PROCESSES, start=1):
        base = 3 + index - 1
        target = 18 + 4 * (index - 1)
        correct = base
        lure = target
        control_answer = target
        assert control_answer == lure
        assert correct != lure
        cases.append(
            {
                "case_id": f"crt_fresh_v1_rate_{index:03d}",
                "family": "crt_rate",
                "question": (
                    f"If {base} {agents} can {verb} {base} {outputs} in {base} minutes, "
                    f"how many minutes would {target} {agents} need to {verb} "
                    f"{target} {outputs}?"
                ),
                "correct_answer": f"{correct} minutes",
                "lure_answer": f"{lure} minutes",
                "control_question": (
                    f"If {base} {agents} can {verb} {base} {outputs} in {base} minutes, "
                    f"how many minutes would the same {base} {agents} need to {verb} "
                    f"{target} {outputs}?"
                ),
                "note": (
                    "template=parallel_rate; validation=closed_form; "
                    "control_answer_equals_lure=true"
                ),
            }
        )

    # With doubling, half coverage occurs exactly one day before full coverage.
    # The matched control changes only the growth law to equal daily increments,
    # making half of the total time the correct answer.
    for index, subject in enumerate(_FRESH_GROWTH_SUBJECTS, start=1):
        full_day = 20 + 2 * (index - 1)
        correct = full_day - 1
        lure = full_day // 2
        control_answer = full_day // 2
        assert full_day % 2 == 0
        assert control_answer == lure
        assert correct != lure
        cases.append(
            {
                "case_id": f"crt_fresh_v1_growth_{index:03d}",
                "family": "crt_growth",
                "question": (
                    f"A {subject} doubles in covered area every day. It takes {full_day} "
                    "days to cover the entire available area. How many days does it take "
                    "to cover half of the area?"
                ),
                "correct_answer": f"{correct} days",
                "lure_answer": f"{lure} days",
                "control_question": (
                    f"A {subject} increases by the same amount of covered area every day. "
                    f"It takes {full_day} days to cover the entire available area. How many "
                    "days does it take to cover half of the area?"
                ),
                "note": (
                    "template=exponential_growth; validation=closed_form; "
                    "control_answer_equals_lure=true"
                ),
            }
        )

    if len(cases) != 30:
        raise AssertionError(f"crt_fresh_v1 expected 30 cases, got {len(cases)}")
    if len({case["question"] for case in cases}) != len(cases):
        raise AssertionError("crt_fresh_v1 contains duplicate hostile questions")
    if len({case["control_question"] for case in cases}) != len(cases):
        raise AssertionError("crt_fresh_v1 contains duplicate control questions")
    return cases


def build_crt_fresh_v1() -> Path:
    return write_dataset(
        dataset_id="crt_fresh_v1",
        title="MindScopeX CRT Fresh Isomorphs v1",
        description=(
            "A deterministic 30-item synthetic pilot with 10 fresh surface isomorphs "
            "for each of difference, parallel-rate, and exponential-growth CRT families. "
            "Every hostile item has a matched control where the hostile lure becomes the "
            "correct answer. Answers and control relations are verified by closed-form "
            "assertions during generation; the set is not filtered on model failures."
        ),
        task_kind="crt",
        scoring="logprob_margin",
        source=CRT_FRESH_V1_SOURCE,
        cases=_fresh_crt_cases(),
    )


# --------------------------------------------------------------------------
# MindScopeX fresh synthetic CRT isomorph expansion (2026)
# --------------------------------------------------------------------------
CRT_FRESH_V2_SOURCE = {
    **CRT_FRESH_V1_SOURCE,
    "title": "CRT Fresh Isomorphs v2",
    "license_note": (
        "A 150-item deterministic expansion generated from closed-form CRT templates. "
        "The set supersedes rather than supplements crt_fresh_v1 in evaluation; do not "
        "pool the two versions. No model failures were used for item inclusion."
    ),
}

_FRESH_V2_DIFFERENCE_PAIRS = (
    ("portable projector", "wireless presenter"),
    ("mechanical keyboard", "mouse pad"),
    ("noise-canceling headphones", "audio cable"),
    ("webcam", "privacy cover"),
    ("external monitor", "HDMI adapter"),
    ("camping stove", "metal cup"),
    ("hiking backpack", "rain cover"),
    ("bicycle helmet", "water bottle cage"),
    ("climbing harness", "chalk bag"),
    ("sleeping bag", "camping pillow"),
    ("chef's knife", "vegetable peeler"),
    ("stand mixer", "silicone spatula"),
    ("espresso machine", "milk pitcher"),
    ("cast-iron pan", "wooden spoon"),
    ("food processor", "measuring cup"),
    ("desk lamp", "cable organizer"),
    ("ergonomic chair", "footrest"),
    ("whiteboard", "marker set"),
    ("document scanner", "stapler"),
    ("filing cabinet", "label pack"),
    ("telescope", "tripod adapter"),
    ("board game", "card sleeve pack"),
    ("electric guitar", "pick pack"),
    ("sewing machine", "thread set"),
    ("model train engine", "track connector"),
    ("hard-shell suitcase", "luggage tag"),
    ("travel backpack", "passport holder"),
    ("camera bag", "lens cloth"),
    ("portable charger", "charging cable"),
    ("train pass", "seat reservation"),
    ("tennis racket", "ball tube"),
    ("yoga mat", "stretching strap"),
    ("football boots", "lace set"),
    ("ski helmet", "goggle case"),
    ("baseball glove", "practice ball"),
    ("digital piano", "sustain pedal"),
    ("studio microphone", "pop filter"),
    ("violin", "rosin block"),
    ("drum stool", "drum key"),
    ("guitar amplifier", "instrument cable"),
    ("precision scale", "sample tray"),
    ("microscope", "slide box"),
    ("centrifuge", "tube rack"),
    ("soldering station", "tip cleaner"),
    ("thermal camera", "protective pouch"),
    ("vacuum cleaner", "dusting brush"),
    ("air purifier", "filter cover"),
    ("floor lamp", "extension cord"),
    ("toolbox", "measuring tape"),
    ("electric drill", "bit set"),
)

_FRESH_V2_RATE_PROCESSES = (
    ("prototype printers", "print", "prototype shells"),
    ("laser cutters", "cut", "acrylic panels"),
    ("baristas", "prepare", "iced drinks"),
    ("packing robots", "seal", "shipping boxes"),
    ("document scanners", "scan", "folders"),
    ("textile looms", "weave", "scarves"),
    ("labeling machines", "label", "bottles"),
    ("engraving stations", "engrave", "nameplates"),
    ("test benches", "test", "circuit boards"),
    ("photo printers", "print", "photo books"),
    ("bakers", "decorate", "cupcakes"),
    ("editors", "proofread", "articles"),
    ("inspection drones", "inspect", "solar panels"),
    ("water pumps", "fill", "storage tanks"),
    ("sorting robots", "sort", "parcels"),
    ("ceramic kilns", "fire", "tile batches"),
    ("CNC mills", "shape", "metal brackets"),
    ("sewing stations", "stitch", "tote bags"),
    ("dishwashers", "wash", "serving trays"),
    ("charging docks", "charge", "handheld scanners"),
    ("lab technicians", "prepare", "sample vials"),
    ("binding machines", "bind", "manuals"),
    ("paint booths", "coat", "cabinet doors"),
    ("coffee roasters", "roast", "bean batches"),
    ("packaging lines", "wrap", "gift boxes"),
    ("quality inspectors", "inspect", "helmets"),
    ("copy machines", "copy", "booklets"),
    ("wood lathes", "turn", "table legs"),
    ("mixing stations", "mix", "paint cans"),
    ("recycling sorters", "sort", "material bins"),
    ("medical imagers", "scan", "test phantoms"),
    ("translation teams", "translate", "short notices"),
    ("greenhouse robots", "water", "plant rows"),
    ("mail clerks", "stamp", "envelopes"),
    ("screen-printing presses", "print", "shirts"),
    ("assembly cells", "assemble", "sensor modules"),
    ("data encoders", "encode", "archive files"),
    ("cutting tables", "cut", "fabric panels"),
    ("polishing machines", "polish", "glass discs"),
    ("filling nozzles", "fill", "sample tubes"),
    ("carton sealers", "seal", "cartons"),
    ("proofing ovens", "proof", "bread trays"),
    ("robot welders", "weld", "frame joints"),
    ("optical readers", "read", "answer sheets"),
    ("seed planters", "plant", "garden rows"),
    ("audio processors", "render", "sound clips"),
    ("map plotters", "print", "survey maps"),
    ("sterilizers", "sterilize", "instrument trays"),
    ("badge printers", "print", "visitor badges"),
    ("parcel lockers", "process", "pickup orders"),
)

_FRESH_V2_GROWTH_SUBJECTS = (
    "duckweed patch on a pond",
    "mold culture on a plate",
    "blue algae patch in a tank",
    "digital tile pattern on a display",
    "ground-cover plant in a greenhouse bed",
    "bacterial colony on an agar tray",
    "floating fern patch in a reservoir",
    "simulated wildfire region on a map",
    "crystal pattern in a lab dish",
    "lichen patch on a test surface",
    "yeast colony on a culture plate",
    "water-lily patch in a garden pool",
    "pixelated stain in an image simulation",
    "moss patch on a greenhouse wall",
    "coral model in a reef simulation",
    "fungal mat in a sealed container",
    "grass patch in an ecology model",
    "oil-film simulation on a water surface",
    "ivy patch on a training wall",
    "snow-cover region in a climate model",
    "cell colony in a microscopy dish",
    "foam patch in a mixing tank",
    "rust patch on a test panel",
    "biofilm patch in a flow chamber",
    "colored region in a diffusion display",
    "leaf-canopy patch in a growth simulation",
    "mineral deposit on a lab tile",
    "ice patch in a freezing experiment",
    "spore colony on a nutrient sheet",
    "shadow region in a graphics demo",
    "plankton patch in a marine model",
    "salt-crystal patch on an evaporation tray",
    "heat-affected region in a material simulation",
    "ink patch on absorbent paper",
    "clover patch in a field model",
    "condensation patch on a cooling plate",
    "lichen colony in an enclosure",
    "reaction front in a chemistry simulation",
    "root-mat patch in a growth chamber",
    "colored-cell region in a spreadsheet model",
    "microbe colony on a nutrient pad",
    "floating-leaf patch in a wetland model",
    "surface crack region in a stress simulation",
    "fermentation culture in a shallow tray",
    "pollen patch on a collection slide",
    "water stain on a test fabric",
    "coverage mask in a mapping program",
    "seedling patch in a nursery bed",
    "luminescent region in a sensor test",
    "paint-spread region in a coating simulation",
)


def _difference_wording(
    variant: int,
    expensive: str,
    inexpensive: str,
    total: int,
    difference: int,
    *,
    control: bool,
) -> str:
    relation = (
        f"The price of the {expensive} is ${difference}."
        if control
        else (
            f"The price of the {expensive} is ${difference} more than the price of "
            f"the {inexpensive}."
        )
    )
    stems = (
        f"The combined prices of the {expensive} and the {inexpensive} total ${total}.",
        f"A receipt lists the {expensive} and the {inexpensive} for ${total} in total.",
        f"Together, the {expensive} and the {inexpensive} cost ${total}.",
        f"Buying the {expensive} with the {inexpensive} costs ${total} altogether.",
        f"The total for the {expensive} and the {inexpensive} comes to ${total}.",
    )
    return (
        f"{stems[variant]} {relation} What is the price of the {inexpensive}? "
        "Answer in dollars."
    )


def _rate_wording(
    variant: int,
    agents: str,
    verb: str,
    outputs: str,
    base: int,
    target: int,
    *,
    control: bool,
) -> str:
    target_agents = f"the same {base} {agents}" if control else f"{target} {agents}"
    stems = (
        f"If {base} {agents} can {verb} {base} {outputs} in {base} minutes,",
        f"A group of {base} {agents} needs {base} minutes to {verb} {base} {outputs}.",
        f"In {base} minutes, {base} {agents} {verb} {base} {outputs}.",
        f"Suppose {base} {agents} {verb} {base} {outputs} during a {base}-minute run.",
        f"It takes {base} {agents} exactly {base} minutes to {verb} {base} {outputs}.",
    )
    question_word = "how" if stems[variant].endswith(",") else "How"
    return (
        f"{stems[variant]} {question_word} many minutes would {target_agents} need to "
        f"{verb} {target} {outputs}?"
    )


def _growth_wording(variant: int, subject: str, full_day: int, *, control: bool) -> str:
    article = "an" if subject[0].casefold() in "aeiou" else "a"
    if control:
        stems = (
            f"{article.title()} {subject} adds the same amount of covered area every day.",
            f"The area covered by {article} {subject} increases by the same amount every day.",
            f"In an observation, {article} {subject} expands by the same area every day.",
            f"Researchers track {article} {subject} whose covered area grows equally each day.",
            f"Consider {article} {subject} that increases its covered area equally each day.",
        )
    else:
        stems = (
            f"{article.title()} {subject} doubles its covered area every day.",
            f"The area covered by {article} {subject} doubles every day.",
            f"In an observation, the covered area of {article} {subject} doubles every day.",
            f"Researchers track {article} {subject} whose covered area doubles every day.",
            f"Consider {article} {subject} that doubles its covered area every day.",
        )
    return (
        f"{stems[variant]} It covers the entire available area after {full_day} days. "
        "After how many days does it cover half of that area?"
    )


def _normalized_question(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _validate_fresh_question_novelty(
    cases: list[dict[str, Any]],
    *,
    similarity_limit: float = 0.92,
) -> dict[str, Any]:
    """Reject exact/near copies of committed non-synthetic benchmark questions."""

    reference_rows: list[tuple[str, str]] = []
    for path in sorted(DATA_DIR.glob("*.json")):
        if path.stem.startswith("crt_fresh"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("cases", []):
            question = str(row.get("question", "")).strip()
            if question:
                reference_rows.append((str(row.get("case_id", path.stem)), question))

    reference_normalized = {
        _normalized_question(question): case_id for case_id, question in reference_rows
    }
    closest = {"ratio": 0.0, "case_id": "", "reference_case_id": ""}
    for case in cases:
        normalized = _normalized_question(case["question"])
        if normalized in reference_normalized:
            raise ValueError(
                f"{case['case_id']} exactly duplicates {reference_normalized[normalized]}"
            )
        for reference_case_id, reference_question in reference_rows:
            ratio = difflib.SequenceMatcher(
                None,
                normalized,
                _normalized_question(reference_question),
                autojunk=False,
            ).ratio()
            if ratio > closest["ratio"]:
                closest = {
                    "ratio": ratio,
                    "case_id": case["case_id"],
                    "reference_case_id": reference_case_id,
                }
            if ratio >= similarity_limit:
                raise ValueError(
                    f"{case['case_id']} is too similar to {reference_case_id}: {ratio:.3f}"
                )
    return closest


def _validate_fresh_surface_quality(cases: list[dict[str, Any]]) -> None:
    """Catch deterministic grammar/format defects before committing generated JSON."""

    invalid_patterns = {
        "lowercase sentence start": re.compile(r"\.\s+[a-z]"),
        "a before vowel": re.compile(r"\ba\s+[aeiou][a-z-]*\b", re.IGNORECASE),
        "an before consonant": re.compile(
            r"\ban\s+[b-df-hj-np-tv-z][a-z-]*\b",
            re.IGNORECASE,
        ),
        "double space": re.compile(r" {2,}"),
    }
    for case in cases:
        for field in ("question", "control_question"):
            text = case[field]
            for label, pattern in invalid_patterns.items():
                match = pattern.search(text)
                if match:
                    raise ValueError(
                        f"{case['case_id']} {field} has {label}: {match.group(0)!r}"
                    )


def _fresh_crt_v2_cases() -> list[dict[str, Any]]:
    """Create 150 closed-form-validated CRT cases with five wording templates."""

    if not (
        len(_FRESH_V2_DIFFERENCE_PAIRS)
        == len(_FRESH_V2_RATE_PROCESSES)
        == len(_FRESH_V2_GROWTH_SUBJECTS)
        == 50
    ):
        raise AssertionError("crt_fresh_v2 source banks must contain 50 entries per family")

    cases: list[dict[str, Any]] = []
    for index, (expensive, inexpensive) in enumerate(_FRESH_V2_DIFFERENCE_PAIRS, start=1):
        small = 5 + index
        difference = 24 + 3 * index
        total = 2 * small + difference
        lure = total - difference
        assert lure == 2 * small
        cases.append(
            {
                "case_id": f"crt_fresh_v2_difference_{index:03d}",
                "pair_id": f"crt_fresh_v2_difference_{index:03d}",
                "template_id": f"difference_wording_{(index - 1) % 5 + 1}",
                "condition": "hostile",
                "family": "crt_difference",
                "question": _difference_wording(
                    (index - 1) % 5,
                    expensive,
                    inexpensive,
                    total,
                    difference,
                    control=False,
                ),
                "correct_answer": f"${small}",
                "lure_answer": f"${lure}",
                "control_question": _difference_wording(
                    (index - 1) % 5,
                    expensive,
                    inexpensive,
                    total,
                    difference,
                    control=True,
                ),
                "note": (
                    "validation=closed_form; control_answer_equals_lure=true; "
                    f"parameters=small:{small},difference:{difference},total:{total}"
                ),
            }
        )

    for index, (agents, verb, outputs) in enumerate(_FRESH_V2_RATE_PROCESSES, start=1):
        base = 3 + (index - 1) % 12
        target = 20 + 3 * (index - 1)
        assert base != target
        cases.append(
            {
                "case_id": f"crt_fresh_v2_rate_{index:03d}",
                "pair_id": f"crt_fresh_v2_rate_{index:03d}",
                "template_id": f"rate_wording_{(index - 1) % 5 + 1}",
                "condition": "hostile",
                "family": "crt_rate",
                "question": _rate_wording(
                    (index - 1) % 5,
                    agents,
                    verb,
                    outputs,
                    base,
                    target,
                    control=False,
                ),
                "correct_answer": f"{base} minutes",
                "lure_answer": f"{target} minutes",
                "control_question": _rate_wording(
                    (index - 1) % 5,
                    agents,
                    verb,
                    outputs,
                    base,
                    target,
                    control=True,
                ),
                "note": (
                    "validation=closed_form; control_answer_equals_lure=true; "
                    f"parameters=base:{base},target:{target}"
                ),
            }
        )

    for index, subject in enumerate(_FRESH_V2_GROWTH_SUBJECTS, start=1):
        full_day = 20 + 2 * (index - 1)
        correct = full_day - 1
        lure = full_day // 2
        assert full_day % 2 == 0
        assert correct != lure
        cases.append(
            {
                "case_id": f"crt_fresh_v2_growth_{index:03d}",
                "pair_id": f"crt_fresh_v2_growth_{index:03d}",
                "template_id": f"growth_wording_{(index - 1) % 5 + 1}",
                "condition": "hostile",
                "family": "crt_growth",
                "question": _growth_wording(
                    (index - 1) % 5,
                    subject,
                    full_day,
                    control=False,
                ),
                "correct_answer": f"{correct} days",
                "lure_answer": f"{lure} days",
                "control_question": _growth_wording(
                    (index - 1) % 5,
                    subject,
                    full_day,
                    control=True,
                ),
                "note": (
                    "validation=closed_form; control_answer_equals_lure=true; "
                    f"parameters=full_day:{full_day}"
                ),
            }
        )

    if len(cases) != 150:
        raise AssertionError(f"crt_fresh_v2 expected 150 cases, got {len(cases)}")
    for field in ("case_id", "pair_id", "question", "control_question"):
        if len({case[field] for case in cases}) != len(cases):
            raise AssertionError(f"crt_fresh_v2 contains duplicate {field}")
    template_counts = Counter(case["template_id"] for case in cases)
    if set(template_counts.values()) != {10}:
        raise AssertionError(f"crt_fresh_v2 template imbalance: {dict(template_counts)}")
    _validate_fresh_surface_quality(cases)
    _validate_fresh_question_novelty(cases)
    return cases


def build_crt_fresh_v2() -> Path:
    return write_dataset(
        dataset_id="crt_fresh_v2",
        title="MindScopeX CRT Fresh Isomorphs v2",
        description=(
            "A 150-item deterministic synthetic core set with 50 independently named "
            "scenarios in each of difference, parallel-rate, and exponential-growth "
            "families. Five wording templates per family reduce single-template "
            "dependence. Every hostile item has a matched control where its lure becomes "
            "correct. Closed-form assertions and public-benchmark near-duplicate checks "
            "run at build time; model failures do not determine inclusion."
        ),
        task_kind="crt",
        scoring="logprob_margin",
        source=CRT_FRESH_V2_SOURCE,
        cases=_fresh_crt_v2_cases(),
        schema_version=2,
    )


# --------------------------------------------------------------------------
# registry + entrypoint
# --------------------------------------------------------------------------
BUILDERS: dict[str, Callable[[], Path]] = {
    "hagendorff_crt": build_hagendorff_crt,
    "hagendorff_semantic_illusion": build_hagendorff_semantic_illusion,
    "crt2": build_crt2,
    "verbal_crt": build_verbal_crt,
    "crt7_classic": build_crt7_classic,
    "yax_crt_isomorph": build_yax_crt_isomorph,
    "crt_fresh_v1": build_crt_fresh_v1,
    "crt_fresh_v2": build_crt_fresh_v2,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "builders",
        nargs="*",
        metavar="BUILDER",
        help=f"Dataset(s) to build (default: all). Choices: {', '.join(BUILDERS)}, all.",
    )
    args = parser.parse_args(argv)
    requested = args.builders or ["all"]
    if "all" in requested:
        selected = list(BUILDERS)
    else:
        unknown = [name for name in requested if name not in BUILDERS]
        if unknown:
            parser.error(
                f"unknown builder(s): {', '.join(unknown)}. Choices: {', '.join(BUILDERS)}, all."
            )
        selected = requested
    for name in selected:
        BUILDERS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
