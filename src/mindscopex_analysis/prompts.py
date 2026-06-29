"""Shared prompt instructions for CRT generation and activation experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from mindscopex_analysis.cases import LureCase

CRT_FINAL_ANSWER_SYSTEM_PROMPT = (
    "Solve the problem, but make the final response shown to the user exactly one line "
    "containing only the requested answer and its requested unit. Do not include calculations, "
    "reasoning, explanations, labels, introductory text, restatements, or Markdown. When thinking "
    "mode is enabled, do not repeat any reasoning in the final response."
)


def prepend_final_answer_instruction(
    prompt: str,
    instruction: str = CRT_FINAL_ANSWER_SYSTEM_PROMPT,
) -> str:
    """Prefix a plain-text prompt for Base-model experiments."""

    instruction = instruction.strip()
    if not instruction:
        return prompt

    prefix = instruction + "\n\n"
    if prompt.startswith(prefix):
        return prompt
    return prefix + prompt


def instruct_lure_case(
    case: LureCase,
    instruction: str = CRT_FINAL_ANSWER_SYSTEM_PROMPT,
) -> LureCase:
    """Apply the shared answer instruction to a case and its matched control."""

    return replace(
        case,
        prompt=prepend_final_answer_instruction(case.prompt, instruction),
        control_prompt=(
            prepend_final_answer_instruction(case.control_prompt, instruction)
            if case.control_prompt
            else ""
        ),
    )


def instruct_lure_cases(
    cases: Sequence[LureCase],
    instruction: str = CRT_FINAL_ANSWER_SYSTEM_PROMPT,
) -> list[LureCase]:
    """Apply the shared answer instruction to several cases."""

    return [instruct_lure_case(case, instruction) for case in cases]
