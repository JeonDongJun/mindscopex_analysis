"""One-time build scripts that fetch public reasoning-lure datasets and
normalize them into the repository's uniform lure-case JSON schema.

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
      "scoring": "logprob_margin" | "premise_rejection",
      "source": {authors, year, title, venue, doi, project_url,
                 download_url, source_sha256, license, license_note},
      "generated_by": "scripts/build_datasets.py",
      "n_cases": int,
      "family_counts": {family: count},
      "cases": [
        {case_id, family, question, correct_answer, lure_answer,
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
    "family",
    "question",
    "correct_answer",
    "lure_answer",
    "control_question",
    "reference_answer",
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
        if scoring == "logprob_margin":
            correct = case.get("correct_answer", "")
            lure = case.get("lure_answer", "")
            if not correct.strip() or not lure.strip():
                raise ValueError(f"case {cid!r} needs non-empty correct/lure for logprob_margin")
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
) -> Path:
    """Validate and write one normalized dataset file; return its path."""

    if scoring not in {"logprob_margin", "premise_rejection"}:
        raise ValueError(f"unknown scoring {scoring!r}")
    _validate_cases(cases, scoring=scoring)

    family_counts = dict(sorted(Counter(case["family"] for case in cases).items()))
    payload = {
        "dataset_id": dataset_id,
        "schema_version": SCHEMA_VERSION,
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
# registry + entrypoint
# --------------------------------------------------------------------------
BUILDERS: dict[str, Callable[[], Path]] = {
    "hagendorff_crt": build_hagendorff_crt,
    "hagendorff_semantic_illusion": build_hagendorff_semantic_illusion,
    "crt2": build_crt2,
    "verbal_crt": build_verbal_crt,
    "crt7_classic": build_crt7_classic,
    "yax_crt_isomorph": build_yax_crt_isomorph,
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
