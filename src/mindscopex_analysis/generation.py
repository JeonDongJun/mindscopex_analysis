"""Text-generation baselines for Qwen CRT experiments."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import torch

from mindscopex_analysis.cases import LureCase
from mindscopex_analysis.prompts import (
    CRT_FINAL_ANSWER_SYSTEM_PROMPT as CRT_FINAL_ANSWER_SYSTEM_PROMPT,
)
from mindscopex_analysis.qwen_scope import split_qwen_thinking

AnswerLabel = Literal["correct", "lure", "both", "other"]

_FINAL_EXPLANATION_PATTERN = re.compile(
    r"(?i)(?:\b(?:answer|because|therefore|thus|hence|since|calculation|reasoning)\b|"
    r"\b(?:costs?|takes?|equals?)\b|정답은|따라서|왜냐하면)"
)


@dataclass(frozen=True)
class QwenTextResponse:
    """One generated response and the metadata needed to reproduce it."""

    model_id: str
    case_id: str
    family: str
    enable_thinking: bool | None
    prompt: str
    correct_answer: str
    lure_answer: str
    raw_text: str
    thinking: str
    answer: str
    answer_label: AnswerLabel
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    seed: int
    hit_max_tokens: bool

    @property
    def mode(self) -> str:
        if self.enable_thinking is None:
            return "base_completion"
        return "thinking" if self.enable_thinking else "non_thinking"

    @property
    def has_thinking_block(self) -> bool:
        """Whether the generated tokens contain a complete Qwen thinking block."""

        start = self.raw_text.find("<think>")
        end = self.raw_text.find("</think>")
        return start >= 0 and end > start

    @property
    def thinking_protocol_issue(self) -> str | None:
        """Explain why generated tokens violate the selected thinking protocol."""

        if self.enable_thinking is None:
            return None

        open_count = self.raw_text.count("<think>")
        close_count = self.raw_text.count("</think>")
        if not self.enable_thinking:
            if open_count or close_count or self.reasoning_detected:
                return "unexpected_thinking_content"
            return None

        if not open_count and not close_count:
            return "missing_thinking_block"
        if not open_count:
            return "missing_think_open"
        if not close_count:
            return "truncated_before_think_close" if self.hit_max_tokens else "missing_think_close"
        if self.raw_text.find("</think>") < self.raw_text.find("<think>"):
            return "invalid_thinking_tag_order"
        if open_count != 1 or close_count != 1:
            return "multiple_thinking_blocks"
        if not self.reasoning_detected:
            return "empty_thinking_block"
        if not self.answer.strip():
            return "missing_final_answer"
        return None

    @property
    def reasoning_detected(self) -> bool:
        """Whether non-empty reasoning was parsed from generated tokens."""

        return bool(self.thinking.strip())

    @property
    def thinking_protocol_ok(self) -> bool | None:
        """Check generated-token use of Qwen's thinking protocol."""

        if self.enable_thinking is None:
            return None
        return self.thinking_protocol_issue is None

    @property
    def final_answer_format_issue(self) -> str | None:
        """Return a conservative reason when final output is not answer-only."""

        answer = self.answer.strip()
        if not answer:
            return "missing_final_answer"
        if "<think>" in answer or "</think>" in answer:
            return "thinking_tag_in_final"
        if "\n" in answer or "\r" in answer:
            return "multiline_final_answer"
        if _FINAL_EXPLANATION_PATTERN.search(answer):
            return "label_or_explanation"
        if len(answer) > 60 or len(answer.split()) > 5:
            return "final_answer_too_long"
        return None

    @property
    def final_answer_format_ok(self) -> bool:
        return self.final_answer_format_issue is None

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["mode"] = self.mode
        row["has_thinking_block"] = self.has_thinking_block
        row["reasoning_detected"] = self.reasoning_detected
        row["thinking_protocol_ok"] = self.thinking_protocol_ok
        row["thinking_protocol_issue"] = self.thinking_protocol_issue
        row["final_answer_format_ok"] = self.final_answer_format_ok
        row["final_answer_format_issue"] = self.final_answer_format_issue
        return row

    def summary_row(self) -> dict[str, Any]:
        """Return compact columns suitable for notebook display."""

        return {
            "model": self.model_id.rsplit("/", 1)[-1],
            "case": self.case_id,
            "mode": self.mode,
            "think_block": self.has_thinking_block,
            "reasoning_chars": len(self.thinking),
            "think_protocol_ok": self.thinking_protocol_ok,
            "protocol_issue": self.thinking_protocol_issue,
            "answer_only": self.final_answer_format_ok,
            "format_issue": self.final_answer_format_issue,
            "label": self.answer_label,
            "final_answer": self.answer,
            "output_tokens": self.output_tokens,
            "seconds": round(self.elapsed_seconds, 2),
            "truncated": self.hit_max_tokens,
        }


def qwen_recommended_sampling_kwargs(enable_thinking: bool | None) -> dict[str, Any]:
    """Return Qwen3 model-card sampling defaults for each reasoning mode."""

    if enable_thinking:
        return {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    return {"temperature": 0.7, "top_p": 0.8, "top_k": 20}


def _normalized_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def _answer_patterns(answer: str) -> tuple[str, ...]:
    normalized = _normalized_text(answer)
    patterns = [rf"(?<!\w){re.escape(normalized)}(?!\w)"]

    currency = re.fullmatch(r"\$\s*(\d+(?:\.\d+)?)", normalized)
    if currency:
        value = Decimal(currency.group(1))
        plain = format(value.normalize(), "f")
        if "." in plain:
            integer, fraction = plain.split(".", 1)
            number_pattern = rf"{re.escape(integer)}\.{re.escape(fraction)}0*"
        else:
            number_pattern = rf"{re.escape(plain)}(?:\.0+)?"
        patterns.extend(
            [
                rf"(?<!\w)\$\s*{number_pattern}(?![\d.])",
                rf"(?<!\w){number_pattern}\s*(?:dollars?|usd)(?!\w)",
            ]
        )

    cents = re.fullmatch(r"(\d+) cents?", normalized)
    if cents:
        value = int(cents.group(1))
        patterns.extend(
            [
                rf"(?<!\w){value}\s*(?:cent|cents|c)(?!\w)",
                rf"(?<!\w){value}\s*¢",
                rf"\$\s*{value / 100:.2f}(?!\d)",
            ]
        )

    quantity = re.fullmatch(
        r"(\d+) (seconds?|minutes?|hours?|days?|weeks?|months?|years?)",
        normalized,
    )
    if quantity:
        value, unit = quantity.groups()
        root = unit.rstrip("s")
        patterns.append(rf"(?<!\w){value}\s*{root}s?(?!\w)")

    ordinal_day = re.fullmatch(r"(\d+)(?:st|nd|rd|th) day", normalized)
    if ordinal_day:
        value = ordinal_day.group(1)
        patterns.extend(
            [
                rf"(?<!\w){value}(?:st|nd|rd|th)?\s+day(?!\w)",
                rf"(?<!\w)day\s+{value}(?!\w)",
            ]
        )

    return tuple(dict.fromkeys(patterns))


def text_contains_answer(text: str, expected_answer: str) -> bool:
    """Check common surface forms of an expected short answer."""

    normalized = _normalized_text(text)
    return any(re.search(pattern, normalized) for pattern in _answer_patterns(expected_answer))


def classify_lure_answer(text: str, case: LureCase) -> AnswerLabel:
    """Label final-answer text without treating hidden thinking as the answer."""

    has_correct = text_contains_answer(text, case.correct_answer)
    has_lure = text_contains_answer(text, case.lure_answer)
    if has_correct and has_lure:
        return "both"
    if has_correct:
        return "correct"
    if has_lure:
        return "lure"
    return "other"


def _input_device(model: Any) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, TypeError):
        return next(model.parameters()).device


def _prepare_inputs(
    tokenizer: Any,
    prompt: str,
    *,
    system_prompt: str,
    enable_thinking: bool | None,
    use_chat_template: bool,
) -> dict[str, torch.Tensor]:
    if not use_chat_template:
        return dict(tokenizer(prompt, return_tensors="pt"))

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking

    try:
        encoded = tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        # Older tokenizer templates use the textual Qwen3 soft switch.
        if enable_thinking is not None:
            suffix = " /think" if enable_thinking else " /no_think"
            messages[-1] = {"role": "user", "content": prompt + suffix}
        template_kwargs.pop("enable_thinking", None)
        encoded = tokenizer.apply_chat_template(messages, **template_kwargs)

    if isinstance(encoded, torch.Tensor):
        return {"input_ids": encoded, "attention_mask": torch.ones_like(encoded)}
    return dict(encoded)


def generate_qwen_text_response(
    model: Any,
    tokenizer: Any,
    case: LureCase,
    *,
    model_id: str,
    enable_thinking: bool | None = False,
    use_chat_template: bool = True,
    system_prompt: str = "",
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    seed: int = 42,
    generation_kwargs: dict[str, Any] | None = None,
) -> QwenTextResponse:
    """Generate one CRT response and separate reasoning from the final answer."""

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    inputs = _prepare_inputs(
        tokenizer,
        case.prompt,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
        use_chat_template=use_chat_template,
    )
    device = _input_device(model)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    input_tokens = int(inputs["input_ids"].shape[-1])

    kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
    }
    if do_sample:
        kwargs.update(qwen_recommended_sampling_kwargs(enable_thinking))
    if tokenizer.pad_token_id is not None:
        kwargs["pad_token_id"] = tokenizer.pad_token_id
    if generation_kwargs:
        kwargs.update(generation_kwargs)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()

    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    generated_ids = output[0, input_tokens:]
    output_tokens = int(generated_ids.numel())
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    thinking, answer = split_qwen_thinking(raw_text)

    eos_ids = tokenizer.eos_token_id
    if eos_ids is None:
        eos_set: set[int] = set()
    elif isinstance(eos_ids, int):
        eos_set = {eos_ids}
    else:
        eos_set = {int(item) for item in eos_ids}
    ended_with_eos = bool(output_tokens and int(generated_ids[-1]) in eos_set)

    return QwenTextResponse(
        model_id=model_id,
        case_id=case.case_id,
        family=case.family,
        enable_thinking=enable_thinking,
        prompt=case.prompt,
        correct_answer=case.correct_answer.strip(),
        lure_answer=case.lure_answer.strip(),
        raw_text=raw_text,
        thinking=thinking,
        answer=answer,
        answer_label=classify_lure_answer(answer, case),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_seconds=elapsed,
        seed=int(seed),
        hit_max_tokens=output_tokens >= max_new_tokens and not ended_with_eos,
    )


def generate_crt_response_suite(
    model: Any,
    tokenizer: Any,
    cases: Sequence[LureCase],
    *,
    model_id: str,
    thinking_modes: Sequence[bool | None] = (False, True),
    use_chat_template: bool = True,
    system_prompt: str = "",
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    seed: int = 42,
    generation_kwargs: dict[str, Any] | None = None,
) -> list[QwenTextResponse]:
    """Generate all case-by-mode responses for one already-loaded model."""

    results = []
    for case in cases:
        for enable_thinking in thinking_modes:
            results.append(
                generate_qwen_text_response(
                    model,
                    tokenizer,
                    case,
                    model_id=model_id,
                    enable_thinking=enable_thinking,
                    use_chat_template=use_chat_template,
                    system_prompt=system_prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    seed=seed,
                    generation_kwargs=generation_kwargs,
                )
            )
    return results


def summarize_crt_accuracy(
    responses: Sequence[QwenTextResponse],
) -> list[dict[str, Any]]:
    """Aggregate correct and incorrect CRT generations by model and mode."""

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for response in responses:
        model = response.model_id.rsplit("/", 1)[-1]
        key = (model, response.mode)
        row = grouped.setdefault(
            key,
            {
                "model": model,
                "mode": response.mode,
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "lure": 0,
                "both": 0,
                "other": 0,
                "format_failures": 0,
                "protocol_failures": 0,
            },
        )
        row["total"] += 1
        row[response.answer_label] += 1
        if response.answer_label != "correct":
            row["incorrect"] += 1
        if not response.final_answer_format_ok:
            row["format_failures"] += 1
        if response.thinking_protocol_ok is False:
            row["protocol_failures"] += 1

    for row in grouped.values():
        row["accuracy"] = row["correct"] / row["total"] if row["total"] else 0.0
    return list(grouped.values())


def save_qwen_text_responses(
    responses: Sequence[QwenTextResponse],
    path: str | Path,
) -> Path:
    """Save complete generated texts and metadata as UTF-8 JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [response.as_dict() for response in responses]
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination
