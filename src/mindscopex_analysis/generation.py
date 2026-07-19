"""Text-generation baselines for Qwen CRT experiments."""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from html import escape
from pathlib import Path
from typing import Any, Literal

import torch

from mindscopex_analysis.cases import LureCase
from mindscopex_analysis.prompts import (
    CRT_FINAL_ANSWER_SYSTEM_PROMPT as CRT_FINAL_ANSWER_SYSTEM_PROMPT,
)
from mindscopex_analysis.qwen_scope import split_qwen_thinking

AnswerLabel = Literal["correct", "lure", "both", "other"]
EvaluationLabel = Literal["correct", "lure", "hallucination"]

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
    generation_attempts: int = 1
    retry_reasons: tuple[str, ...] = ()
    attempt_history: tuple[dict[str, Any], ...] = ()

    @property
    def mode(self) -> str:
        if self.enable_thinking is None:
            return "base_completion"
        return "thinking" if self.enable_thinking else "non_thinking"

    @property
    def has_thinking_block(self) -> bool:
        """Whether the reconstructed assistant response has a complete thinking block."""

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

    @property
    def evaluation_label(self) -> EvaluationLabel:
        """Map final answers to the three headline categories used in reports."""

        if self.answer_label == "correct":
            return "correct"
        if self.answer_label == "lure":
            return "lure"
        return "hallucination"

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["mode"] = self.mode
        row["has_thinking_block"] = self.has_thinking_block
        row["reasoning_detected"] = self.reasoning_detected
        row["thinking_protocol_ok"] = self.thinking_protocol_ok
        row["thinking_protocol_issue"] = self.thinking_protocol_issue
        row["final_answer_format_ok"] = self.final_answer_format_ok
        row["final_answer_format_issue"] = self.final_answer_format_issue
        row["evaluation_label"] = self.evaluation_label
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
            "outcome": self.evaluation_label,
            "final_answer": self.answer,
            "attempts": self.generation_attempts,
            "retry_reasons": ", ".join(self.retry_reasons),
            "output_tokens": self.output_tokens,
            "seconds": round(self.elapsed_seconds, 2),
            "truncated": self.hit_max_tokens,
        }


def qwen_recommended_sampling_kwargs(enable_thinking: bool | None) -> dict[str, Any]:
    """Return Qwen3.5 text-task sampling defaults for each reasoning mode."""

    if enable_thinking:
        return {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    return {"temperature": 1.0, "top_p": 1.0, "top_k": 20}


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


PremiseVerdict = Literal["rejected", "accepted", "unclear"]

_PREMISE_REJECTION_CUES: tuple[str, ...] = (
    r"\bno\b",
    r"\bnot\b",
    r"n['o]t\b",
    r"\bnever\b",
    r"\bneither\b",
    r"\bnone\b",
    r"\bactually\b",
    r"\bin fact\b",
    r"\bincorrect\b",
    r"\bfalse\b",
    r"\bmistaken\b",
    r"\bmisconception\b",
    r"\bconfus",
    r"\bwrong\b",
    r"\bno such\b",
    r"\bdoes ?not exist\b",
    r"\bthere (?:is|are|was|were) no\b",
    r"\byou (?:probably )?mean\b",
    r"\btrick question\b",
    r"아니",
    r"없",
    r"않",
    r"잘못",
    r"사실",
)

_PREMISE_COMMON_WORDS = frozenset(
    {
        "that",
        "this",
        "with",
        "have",
        "from",
        "they",
        "there",
        "which",
        "were",
        "country",
        "during",
        "after",
        "before",
        "serves",
        "known",
        "called",
        "named",
        "actually",
    }
)


def classify_premise_rejection(
    text: str,
    *,
    reference_answer: str = "",
    question: str = "",
) -> PremiseVerdict:
    """Heuristically judge whether a response rejected a semantic illusion's false premise.

    Returns ``"rejected"`` (the correct behaviour: the response challenges the false
    premise), ``"accepted"`` (the intuitive failure: it answers inside the false
    frame), or ``"unclear"`` for empty text. This is a lexical baseline for the
    ``premise_rejection`` datasets such as ``hagendorff_semantic_illusion``; for
    rigorous scoring use an LLM judge. ``reference_answer`` (the authoritative
    correction stored on those cases) and ``question`` supply a secondary signal:
    a response is treated as a rejection if it surfaces a distinctive term from the
    correction that the question itself did not contain.
    """

    normalized = _normalized_text(text)
    if not normalized:
        return "unclear"
    if any(re.search(pattern, normalized) for pattern in _PREMISE_REJECTION_CUES):
        return "rejected"
    if reference_answer:
        question_terms = set(re.findall(r"[a-z]{4,}", _normalized_text(question)))
        reference_terms = set(re.findall(r"[a-z]{4,}", _normalized_text(reference_answer)))
        distinctive = reference_terms - question_terms - _PREMISE_COMMON_WORDS
        if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized) for term in distinctive):
            return "rejected"
    return "accepted"


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

    uses_multimodal_content = hasattr(tokenizer, "image_processor")

    def message_content(text: str) -> str | list[dict[str, str]]:
        if uses_multimodal_content:
            return [{"type": "text", "text": text}]
        return text

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": message_content(system_prompt)})
    messages.append({"role": "user", "content": message_content(prompt)})

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
        # Older Qwen tokenizer templates use the textual thinking soft switch.
        if enable_thinking is not None:
            suffix = " /think" if enable_thinking else " /no_think"
            messages[-1] = {"role": "user", "content": message_content(prompt + suffix)}
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
    token_backend = getattr(tokenizer, "tokenizer", tokenizer)
    if token_backend.pad_token_id is not None:
        kwargs["pad_token_id"] = token_backend.pad_token_id
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
    if enable_thinking is True and "<think>" not in raw_text:
        # Qwen3.5's chat template places the opening tag in the generation prompt,
        # so decoding only newly generated tokens starts inside the thinking block.
        raw_text = "<think>\n" + raw_text
    thinking, answer = split_qwen_thinking(raw_text)

    eos_ids = token_backend.eos_token_id
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


def response_retry_reason(
    response: QwenTextResponse,
    *,
    retry_protocol_issues: bool = True,
    retry_both: bool = True,
) -> str | None:
    """Return the issue that should trigger another stochastic generation."""

    if retry_protocol_issues and response.thinking_protocol_issue is not None:
        return f"protocol:{response.thinking_protocol_issue}"
    if retry_both and response.answer_label == "both":
        return "ambiguous_both"
    return None


def _attempt_record(
    response: QwenTextResponse,
    *,
    attempt_number: int,
    retry_reason: str | None,
    retried: bool,
) -> dict[str, Any]:
    return {
        "attempt": attempt_number,
        "seed": response.seed,
        "answer_label": response.answer_label,
        "evaluation_label": response.evaluation_label,
        "answer": response.answer,
        "raw_text": response.raw_text,
        "thinking_protocol_issue": response.thinking_protocol_issue,
        "final_answer_format_issue": response.final_answer_format_issue,
        "hit_max_tokens": response.hit_max_tokens,
        "retry_reason": retry_reason,
        "retried": retried,
    }


def generate_qwen_text_response_with_retries(
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
    max_retries: int = 0,
    retry_protocol_issues: bool = True,
    retry_both: bool = True,
    retry_seed_step: int = 1,
) -> QwenTextResponse:
    """Regenerate protocol failures and ambiguous answers while preserving an audit trail."""

    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if retry_seed_step < 1:
        raise ValueError("retry_seed_step must be positive")

    history: list[dict[str, Any]] = []
    retry_reasons: list[str] = []
    for attempt_index in range(max_retries + 1):
        response = generate_qwen_text_response(
            model,
            tokenizer,
            case,
            model_id=model_id,
            enable_thinking=enable_thinking,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            seed=seed + attempt_index * retry_seed_step,
            generation_kwargs=generation_kwargs,
        )
        retry_reason = response_retry_reason(
            response,
            retry_protocol_issues=retry_protocol_issues,
            retry_both=retry_both,
        )
        will_retry = retry_reason is not None and attempt_index < max_retries
        history.append(
            _attempt_record(
                response,
                attempt_number=attempt_index + 1,
                retry_reason=retry_reason,
                retried=will_retry,
            )
        )
        if not will_retry:
            return replace(
                response,
                generation_attempts=attempt_index + 1,
                retry_reasons=tuple(retry_reasons),
                attempt_history=tuple(history),
            )
        retry_reasons.append(retry_reason)

    raise RuntimeError("Generation retry loop ended unexpectedly")


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
    max_retries: int = 0,
    retry_protocol_issues: bool = True,
    retry_both: bool = True,
    retry_seed_step: int = 1,
    progress_callback: Callable[[int, int, QwenTextResponse], None] | None = None,
) -> list[QwenTextResponse]:
    """Generate all case-by-mode responses for one already-loaded model.

    ``progress_callback(done, total, response)`` is called after each response so
    long remote runs can stream progress and checkpoint partial results.
    """

    results = []
    total = len(cases) * len(thinking_modes)
    for case in cases:
        for enable_thinking in thinking_modes:
            response = generate_qwen_text_response_with_retries(
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
                max_retries=max_retries,
                retry_protocol_issues=retry_protocol_issues,
                retry_both=retry_both,
                retry_seed_step=retry_seed_step,
            )
            results.append(response)
            if progress_callback is not None:
                progress_callback(len(results), total, response)
    return results


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    radius = (
        z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _summarize_crt_accuracy(
    responses: Sequence[QwenTextResponse],
    *,
    by_family: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    for response in responses:
        model = response.model_id.rsplit("/", 1)[-1]
        key = (model, response.mode, response.family) if by_family else (model, response.mode)
        labels = {"model": model, "mode": response.mode}
        if by_family:
            labels["family"] = response.family
        row = grouped.setdefault(
            key,
            {
                **labels,
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "lure": 0,
                "hallucination": 0,
                "both": 0,
                "other": 0,
                "format_failures": 0,
                "protocol_failures": 0,
                "retried_responses": 0,
                "retry_attempts": 0,
            },
        )
        row["total"] += 1
        row[response.answer_label] += 1
        if response.evaluation_label == "hallucination":
            row["hallucination"] += 1
        if response.answer_label != "correct":
            row["incorrect"] += 1
        if not response.final_answer_format_ok:
            row["format_failures"] += 1
        if response.thinking_protocol_ok is False:
            row["protocol_failures"] += 1
        if response.generation_attempts > 1:
            row["retried_responses"] += 1
            row["retry_attempts"] += response.generation_attempts - 1

    for row in grouped.values():
        total = row["total"]
        row["accuracy"] = row["correct"] / total if total else 0.0
        row["lure_rate"] = row["lure"] / total if total else 0.0
        row["hallucination_rate"] = row["hallucination"] / total if total else 0.0
        row["accuracy_ci_low"], row["accuracy_ci_high"] = _wilson_interval(
            row["correct"],
            total,
        )
        row["lure_rate_ci_low"], row["lure_rate_ci_high"] = _wilson_interval(
            row["lure"],
            total,
        )
    return list(grouped.values())


def summarize_crt_accuracy(
    responses: Sequence[QwenTextResponse],
) -> list[dict[str, Any]]:
    """Aggregate CRT outcomes by model and reasoning mode."""

    return _summarize_crt_accuracy(responses, by_family=False)


def summarize_crt_accuracy_by_family(
    responses: Sequence[QwenTextResponse],
) -> list[dict[str, Any]]:
    """Aggregate CRT outcomes by model, reasoning mode, and task family."""

    return _summarize_crt_accuracy(responses, by_family=True)


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


def _markdown_cell(value: Any) -> str:
    text = escape(str(value), quote=False)
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def save_crt_markdown_report(
    responses: Sequence[QwenTextResponse],
    path: str | Path,
    *,
    dataset_name: str = "",
    dataset_reference: str = "",
) -> Path:
    """Write a readable experiment summary while retaining quality and retry diagnostics."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_crt_accuracy(responses)
    family_summary = summarize_crt_accuracy_by_family(responses)
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()

    lines = [
        "# CRT text-generation result summary",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Dataset: `{_markdown_cell(dataset_name or 'unspecified')}`",
        f"- Dataset reference: {_markdown_cell(dataset_reference or 'not provided')}",
        f"- Final responses: **{len(responses)}**",
        "",
        "## Headline results",
        "",
        "`Hallucination` is an operational bucket for final answers that are neither only the "
        "correct answer nor only the known lure. It includes unresolved `both` and `other` "
        "responses, so inspect the quality table before interpreting it as factual fabrication.",
        "",
        "| Model | Mode | N | Correct | Lure | Hallucination | Accuracy | Retried |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {model} | {mode} | {total} | {correct} | {lure} | {hallucination} | "
            "{accuracy:.1%} | {retried_responses} |".format(
                **{key: _markdown_cell(value) for key, value in row.items() if key != "accuracy"},
                accuracy=row["accuracy"],
            )
        )

    lines.extend(
        [
            "",
            "## Results by CRT family",
            "",
            "Wilson intervals below treat rows as independent Bernoulli observations. When the "
            "same items are repeated across multiple sampling seeds, use an item-clustered "
            "bootstrap for inferential intervals instead of interpreting the pooled Wilson "
            "interval as a confidence interval.",
            "",
            "| Model | Mode | Family | N | Correct | Lure | Other | Accuracy | "
            "Lure rate (Wilson 95% CI) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in family_summary:
        lines.append(
            f"| {_markdown_cell(row['model'])} | {_markdown_cell(row['mode'])} | "
            f"{_markdown_cell(row['family'])} | {row['total']} | {row['correct']} | "
            f"{row['lure']} | {row['hallucination']} | {row['accuracy']:.1%} | "
            f"{row['lure_rate']:.1%} "
            f"[{row['lure_rate_ci_low']:.1%}, {row['lure_rate_ci_high']:.1%}] |"
        )

    lines.extend(
        [
            "",
            "## Quality and retry diagnostics",
            "",
            "| Model | Mode | Format failures | Protocol failures | Retry attempts | "
            "Unresolved both | Other |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {_markdown_cell(row['model'])} | {_markdown_cell(row['mode'])} | "
            f"{row['format_failures']} | {row['protocol_failures']} | {row['retry_attempts']} | "
            f"{row['both']} | {row['other']} |"
        )

    review_rows = [
        response
        for response in responses
        if response.evaluation_label == "hallucination"
        or not response.final_answer_format_ok
        or response.thinking_protocol_ok is False
        or response.generation_attempts > 1
    ]
    lines.extend(
        [
            "",
            "## Responses requiring review",
            "",
            "This section includes hallucination/other outcomes, quality failures, and all retried "
            "responses so the final sample can be audited.",
            "",
        ]
    )
    if not review_rows:
        lines.append("No responses require review.")
    else:
        lines.extend(
            [
                "| Model | Mode | Case | Outcome | Raw label | Final answer | Attempts | "
                "Retry reasons | Protocol issue | Format issue |",
                "|---|---|---|---|---|---|---:|---|---|---|",
            ]
        )
        for response in review_rows:
            values = [
                response.model_id.rsplit("/", 1)[-1],
                response.mode,
                response.case_id,
                response.evaluation_label,
                response.answer_label,
                response.answer,
                response.generation_attempts,
                ", ".join(response.retry_reasons),
                response.thinking_protocol_issue or "",
                response.final_answer_format_issue or "",
            ]
            lines.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")

    retried_rows = [response for response in responses if response.generation_attempts > 1]
    lines.extend(["", "## Retry audit", ""])
    if not retried_rows:
        lines.append("No response was regenerated.")
    else:
        lines.extend(
            [
                "| Model | Mode | Case | Attempt | Seed | Raw label | Answer | "
                "Retry trigger | Retried |",
                "|---|---|---|---:|---:|---|---|---|---|",
            ]
        )
        for response in retried_rows:
            for attempt in response.attempt_history:
                values = [
                    response.model_id.rsplit("/", 1)[-1],
                    response.mode,
                    response.case_id,
                    attempt["attempt"],
                    attempt["seed"],
                    attempt["answer_label"],
                    attempt["answer"],
                    attempt["retry_reason"] or "",
                    attempt["retried"],
                ]
                lines.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")

    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
