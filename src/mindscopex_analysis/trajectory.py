"""Where to sample a feature along a generated reasoning trace.

The causal work reads one static vector at the last prompt token, which is the
representation *just before answering* -- not the reasoning itself. To ask whether
a lure representation rises when the cue is read and falls while the model
deliberates, the feature has to be sampled at several points along the sequence.

These helpers only pick indices and label them; capturing and encoding lives in
the job, so the slicing rules stay testable without a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPhase:
    """One sampled position, with the label that makes it comparable across items."""

    phase: str
    token_index: int
    fraction: float


def quantile_indices(start: int, end: int, phases: int) -> list[int]:
    """``phases`` evenly spaced indices covering ``[start, end)``.

    Quantiles rather than absolute offsets because reasoning traces differ wildly
    in length; item-relative positions are the only thing comparable across them.
    """

    if phases < 1:
        raise ValueError("phases must be positive")
    span = end - start
    if span <= 0:
        return []
    if phases == 1:
        return [start]
    return [start + min(span - 1, round(i * (span - 1) / (phases - 1))) for i in range(phases)]


def reasoning_phases(
    prompt_tokens: int,
    total_tokens: int,
    *,
    phases: int = 5,
    think_end: int | None = None,
) -> list[TokenPhase]:
    """Label the sampling points across a prompt plus its generated continuation.

    ``think_end`` is the absolute index just past ``</think>`` when the trace has a
    thinking block; the phase before it is deliberation and everything after is the
    committed answer. Without it the whole continuation is treated as one span,
    which is the no-thinking condition.
    """

    if prompt_tokens < 1:
        raise ValueError("prompt_tokens must be positive")
    if total_tokens < prompt_tokens:
        raise ValueError("total_tokens must include the prompt")

    out = [TokenPhase("prompt_last", prompt_tokens - 1, 0.0)]
    reasoning_end = think_end if think_end is not None else total_tokens
    reasoning_end = max(prompt_tokens, min(reasoning_end, total_tokens))

    for index in quantile_indices(prompt_tokens, reasoning_end, phases):
        span = max(reasoning_end - prompt_tokens - 1, 1)
        fraction = (index - prompt_tokens) / span
        out.append(TokenPhase(f"reasoning_{round(fraction * 100)}", index, fraction))

    if think_end is not None and total_tokens > reasoning_end:
        # The token the answer is actually emitted from: the last position whose
        # prediction is the first answer token.
        out.append(TokenPhase("pre_answer", reasoning_end - 1, 1.0))
    return out


def find_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> int | None:
    """First index where ``pattern`` occurs in ``sequence`` (None if absent)."""

    if not pattern or len(pattern) > len(sequence):
        return None
    window = len(pattern)
    for start in range(len(sequence) - window + 1):
        if list(sequence[start : start + window]) == list(pattern):
            return start
    return None


def cue_span(
    prompt_token_strings: Sequence[str],
    cue_text: str,
    *,
    min_overlap: int = 3,
) -> tuple[int, int] | None:
    """Locate the cue clause inside a tokenised prompt by string matching.

    Tokenisers split words unpredictably, so this walks the decoded pieces and
    matches on the concatenated text rather than on token ids. Returns the
    half-open ``[start, end)`` token range, or None when the cue is not found or
    matches too little to trust.
    """

    cue = "".join(cue_text.split()).lower()
    if len(cue) < min_overlap:
        return None
    joined = ""
    offsets: list[int] = []
    for piece in prompt_token_strings:
        offsets.append(len(joined))
        joined += "".join(piece.split()).lower()
    position = joined.find(cue)
    if position < 0:
        return None
    start = max(0, sum(1 for offset in offsets if offset <= position) - 1)
    end_char = position + len(cue)
    end = sum(1 for offset in offsets if offset < end_char)
    return (start, max(start + 1, end))
