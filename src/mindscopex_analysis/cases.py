"""Prompt cases for lure-feature experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LureCase:
    """One prompt with a correct answer and a salient lure answer."""

    case_id: str
    family: str
    prompt: str
    correct_answer: str
    lure_answer: str
    control_prompt: str = ""
    note: str = ""
    pair_id: str = ""
    template_id: str = ""
    condition: str = "hostile"


def _answer_prompt(text: str) -> str:
    return text.strip() + "\nAnswer:"


PILOT_CRT_DATASET_ID = "mindscopex_crt_pilot_v1"
_PILOT_TRANSFER_CASE_IDS = (
    "bat_ball_original",
    "machines_widgets",
    "lily_pads",
    "printers_pages",
)


def _required_text(row: dict[str, Any], field: str, *, case_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Pilot CRT case {case_number} has invalid {field!r}")
    return value.strip()


def load_pilot_crt_cases(path: str | Path | None = None) -> list[LureCase]:
    """Load and validate the repository's small JSON CRT pilot set."""

    if path is None:
        source = files("mindscopex_analysis").joinpath("data", "crt_pilot.json")
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError("Pilot CRT JSON root must be an object")
    if payload.get("dataset_id") != PILOT_CRT_DATASET_ID:
        raise ValueError(f"Unexpected pilot CRT dataset_id={payload.get('dataset_id')!r}")
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Pilot CRT JSON must contain a non-empty 'cases' list")

    cases: list[LureCase] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise TypeError(f"Pilot CRT case {index} must be an object")
        case_id = _required_text(row, "case_id", case_number=index)
        if case_id in seen_ids:
            raise ValueError(f"Duplicate pilot CRT case_id={case_id!r}")
        seen_ids.add(case_id)

        correct_answer = _required_text(row, "correct_answer", case_number=index)
        lure_answer = _required_text(row, "lure_answer", case_number=index)
        if correct_answer.casefold() == lure_answer.casefold():
            raise ValueError(f"Pilot CRT case {case_id!r} has identical correct and lure answers")

        control_question = row.get("control_question", "")
        if not isinstance(control_question, str):
            raise TypeError(f"Pilot CRT case {case_id!r} has invalid 'control_question'")
        note = row.get("note", "")
        if not isinstance(note, str):
            raise TypeError(f"Pilot CRT case {case_id!r} has invalid 'note'")

        cases.append(
            LureCase(
                case_id=case_id,
                family=_required_text(row, "family", case_number=index),
                prompt=_answer_prompt(_required_text(row, "question", case_number=index)),
                correct_answer=" " + correct_answer,
                lure_answer=" " + lure_answer,
                control_prompt=_answer_prompt(control_question) if control_question.strip() else "",
                note=note.strip(),
            )
        )
    return cases


def _pilot_case(case_id: str) -> LureCase:
    try:
        return next(case for case in load_pilot_crt_cases() if case.case_id == case_id)
    except StopIteration as exc:
        raise ValueError(f"Pilot CRT dataset is missing required case {case_id!r}") from exc


BAT_BALL_CASE = _pilot_case("bat_ball_original")


def bat_ball_paraphrases() -> list[LureCase]:
    """Prompt variants that preserve the same correct and lure answers."""

    return [
        BAT_BALL_CASE,
        LureCase(
            case_id="bat_ball_slow",
            family="crt_arithmetic",
            prompt=_answer_prompt(
                "Think carefully. A bat and a ball together cost $1.10. "
                "The bat costs exactly $1.00 more than the ball. "
                "What is the price of the ball in cents?"
            ),
            correct_answer=" 5 cents",
            lure_answer=" 10 cents",
            note="Adds a caution instruction.",
        ),
        LureCase(
            case_id="bat_ball_short",
            family="crt_arithmetic",
            prompt=_answer_prompt(
                "Bat + ball = $1.10. Bat = ball + $1.00. What does the ball cost, in cents?"
            ),
            correct_answer=" 5 cents",
            lure_answer=" 10 cents",
            note="Symbolic compact variant.",
        ),
        LureCase(
            case_id="bat_ball_korean",
            family="crt_arithmetic",
            prompt=_answer_prompt(
                "방망이와 공의 가격 합은 1.10달러입니다. "
                "방망이는 공보다 1.00달러 더 비쌉니다. "
                "공은 몇 센트인가요? 숫자와 cents로 답하세요."
            ),
            correct_answer=" 5 cents",
            lure_answer=" 10 cents",
            note="Korean wording with English answer format.",
        ),
        LureCase(
            case_id="book_toy_same_structure",
            family="crt_arithmetic",
            prompt=_answer_prompt(
                "A book and a toy cost $2.30 in total. "
                "The book costs $2.00 more than the toy. "
                "How much does the toy cost? Answer in cents."
            ),
            correct_answer=" 15 cents",
            lure_answer=" 30 cents",
            note="Same algebraic structure with a different lure value.",
        ),
    ]


def crt_transfer_cases() -> list[LureCase]:
    """Small set of CRT-like lure cases for transfer checks."""

    by_id = {case.case_id: case for case in load_pilot_crt_cases()}
    missing = set(_PILOT_TRANSFER_CASE_IDS) - set(by_id)
    if missing:
        raise ValueError(f"Pilot CRT dataset is missing transfer cases: {sorted(missing)}")
    return [by_id[case_id] for case_id in _PILOT_TRANSFER_CASE_IDS]


def crt_behavior_cases() -> list[LureCase]:
    """Broader CRT suite for model-level answer accuracy comparisons."""

    return load_pilot_crt_cases()


def semantic_lure_cases() -> list[LureCase]:
    """Semantic and logical lure cases for specificity checks."""

    return [
        LureCase(
            case_id="moses_ark",
            family="semantic_illusion",
            prompt=_answer_prompt(
                "How many animals of each kind did Moses take on the ark? "
                "Answer with a number or a short correction."
            ),
            correct_answer=" Noah",
            lure_answer=" two",
            control_prompt=_answer_prompt(
                "How many animals of each kind did Noah take on the ark? Answer with a number."
            ),
            note="Presupposition lure.",
        ),
        LureCase(
            case_id="widow_sister",
            family="semantic_illusion",
            prompt=_answer_prompt("Can a man marry his widow's sister? Answer yes or no."),
            correct_answer=" no",
            lure_answer=" yes",
            note="Impossible-premise lure.",
        ),
        LureCase(
            case_id="affirming_consequent",
            family="logic",
            prompt=_answer_prompt(
                "If it rains, the street gets wet. The street is wet. "
                "Therefore, did it rain? Answer yes or no."
            ),
            correct_answer=" no",
            lure_answer=" yes",
            note="Affirming the consequent.",
        ),
    ]


def bat_ball_answer_variants() -> list[tuple[str, str, str]]:
    """Alternative answer surface forms for tokenization sensitivity checks."""

    return [
        ("cents_words", " 5 cents", " 10 cents"),
        ("bare_numbers", " 5", " 10"),
        ("dollars_decimal", " $0.05", " $0.10"),
        ("sentence", " The ball costs 5 cents.", " The ball costs 10 cents."),
    ]
