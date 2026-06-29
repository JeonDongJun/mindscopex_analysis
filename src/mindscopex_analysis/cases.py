"""Prompt cases for lure-feature experiments."""

from __future__ import annotations

from dataclasses import dataclass


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


def _answer_prompt(text: str) -> str:
    return text.strip() + "\nAnswer:"


BAT_BALL_CASE = LureCase(
    case_id="bat_ball_original",
    family="crt_arithmetic",
    prompt=_answer_prompt(
        "A bat and a ball cost $1.10 in total. "
        "The bat costs $1.00 more than the ball. "
        "How much does the ball cost? Answer in cents."
    ),
    correct_answer=" 5 cents",
    lure_answer=" 10 cents",
    control_prompt=_answer_prompt(
        "A bat and a ball cost $1.10 in total. "
        "The bat costs $1.05. "
        "How much does the ball cost? Answer in cents."
    ),
    note="Canonical CRT item; 10 cents is the intuitive arithmetic lure.",
)


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

    return [
        BAT_BALL_CASE,
        LureCase(
            case_id="machines_widgets",
            family="crt_rate",
            prompt=_answer_prompt(
                "If it takes 5 machines 5 minutes to make 5 widgets, "
                "how long would it take 100 machines to make 100 widgets? "
                "Answer in minutes."
            ),
            correct_answer=" 5 minutes",
            lure_answer=" 100 minutes",
            control_prompt=_answer_prompt(
                "Each machine makes 1 widget in 5 minutes. "
                "How long would it take 100 machines to make 100 widgets? "
                "Answer in minutes."
            ),
            note="Rate/proportionality lure.",
        ),
        LureCase(
            case_id="lily_pads",
            family="crt_growth",
            prompt=_answer_prompt(
                "In a lake, a patch of lily pads doubles in size every day. "
                "If it takes 48 days to cover the whole lake, "
                "how long to cover half the lake?"
            ),
            correct_answer=" 47 days",
            lure_answer=" 24 days",
            control_prompt=_answer_prompt(
                "A patch of lily pads doubles every day. "
                "It covers half the lake on day 47 and the whole lake on day 48. "
                "On what day does it cover half the lake?"
            ),
            note="Exponential-growth lure.",
        ),
        LureCase(
            case_id="printers_pages",
            family="crt_rate",
            prompt=_answer_prompt(
                "If 3 printers print 3 pages in 3 minutes, "
                "how long would it take 9 printers to print 9 pages? "
                "Answer in minutes."
            ),
            correct_answer=" 3 minutes",
            lure_answer=" 9 minutes",
            note="Rate lure with smaller numbers.",
        ),
    ]


def crt_behavior_cases() -> list[LureCase]:
    """Broader CRT suite for model-level answer accuracy comparisons."""

    return [
        *crt_transfer_cases(),
        LureCase(
            case_id="race_second_place",
            family="crt_verbal",
            prompt=_answer_prompt(
                "You are running a race and pass the runner in second place. "
                "What place are you in now? Answer with first place, second place, and so on."
            ),
            correct_answer=" second place",
            lure_answer=" first place",
            note="Passing second place puts the runner in second, not first.",
        ),
        LureCase(
            case_id="sheep_all_but",
            family="crt_verbal",
            prompt=_answer_prompt(
                "A farmer has 15 sheep. All but 8 die. How many sheep remain? "
                "Answer as a number followed by sheep."
            ),
            correct_answer=" 8 sheep",
            lure_answer=" 7 sheep",
            note="The phrase 'all but 8' means that 8 remain.",
        ),
        LureCase(
            case_id="class_rank",
            family="crt_counting",
            prompt=_answer_prompt(
                "With no tied scores, a student is both the 15th highest and the 15th lowest "
                "scorer in a class. How many students are in the class? "
                "Answer as a number followed by students."
            ),
            correct_answer=" 29 students",
            lure_answer=" 30 students",
            note="The focal student is counted in both ranks, so 15 + 15 - 1 = 29.",
        ),
        LureCase(
            case_id="clock_strikes",
            family="crt_rate",
            prompt=_answer_prompt(
                "A clock takes 5 seconds from its first strike to its sixth strike. "
                "At the same rate, how many seconds pass from its first strike to its "
                "twelfth strike? Answer in seconds."
            ),
            correct_answer=" 11 seconds",
            lure_answer=" 10 seconds",
            note="Six strikes contain five intervals; twelve strikes contain eleven.",
        ),
        LureCase(
            case_id="discount_reversal",
            family="crt_percentage",
            prompt=_answer_prompt(
                "An item initially costs 100 dollars. Its price is reduced by 20 percent, "
                "then the reduced price is increased by 20 percent. What is the final price? "
                "Answer in dollars."
            ),
            correct_answer=" 96 dollars",
            lure_answer=" 100 dollars",
            note="Opposite percentage changes use different bases and do not cancel.",
        ),
    ]


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
