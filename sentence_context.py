"""Offline sentence/context preparation for explicit Jev proofreading calls.

This module never calls a provider. Offsets are Python Unicode code-point offsets
into the original body; neighbor sentences are context, never extra targets.
"""
from __future__ import annotations

import re
from typing import Any

MAX_TARGETS_PER_BATCH = 4
MAX_BATCHES_PER_PASS = 12
_CLOSERS = frozenset("」』】）》〉〕］｝)）\"'”’")
_CONTINUATION_AFTER_QUOTED_SENTENCE = re.compile(
    r"^(?:と(?:いう|して|は|も|の|が|を|に|で|へ|から|まで|言|話|述|書|答|伝|説明|呼|思|考|感じ)|"
    r"(?:は|が|を|に|へ|で|と|も|の|や|から|まで|より|だけ|など|について|として))"
)


class CoverageLimitError(ValueError):
    """Raised rather than silently dropping targets beyond the configured cap."""

    def __init__(self, target_count: int):
        self.target_count = target_count
        self.batch_cap = MAX_BATCHES_PER_PASS
        self.target_cap = MAX_TARGETS_PER_BATCH * MAX_BATCHES_PER_PASS
        super().__init__(
            f"coverage_limit: {target_count} targets exceed {self.target_cap}; "
            "narrow the proofreading scope or explicitly raise a separate decision"
        )


def _sentence_spans(body: str) -> list[tuple[int, int]]:
    """Return trimmed sentence spans, retaining all non-whitespace source text."""
    spans: list[tuple[int, int]] = []
    start = 0
    length = len(body)
    i = 0
    while i < length:
        if body[i] not in "。！？!?":
            i += 1
            continue
        end = i + 1
        while end < length and body[end] in _CLOSERS:
            end += 1
        # Punctuation inside nested Japanese quotation marks usually belongs to
        # quoted dialogue that continues into the reporting clause. Do not split
        # there; a separate outer punctuation mark will close the sentence.
        prefix = body[start:i + 1]
        if prefix.count("「") > prefix.count("」") or prefix.count("『") > prefix.count("』"):
            i = end
            continue
        suffix = body[end:]
        if suffix and (
            _CONTINUATION_AFTER_QUOTED_SENTENCE.match(suffix)
            or body[end - 1] in _CLOSERS and re.match(r"\s*と", suffix)
        ):
            i = end
            continue
        left = start
        while left < i + 1 and body[left].isspace():
            left += 1
        right = end
        while right > left and body[right - 1].isspace():
            right -= 1
        if left < right:
            spans.append((left, right))
        start = end
        i = end
    left = start
    while left < length and body[left].isspace():
        left += 1
    right = length
    while right > left and body[right - 1].isspace():
        right -= 1
    if left < right:
        spans.append((left, right))
    return spans


def prepare_segments(
    body: str,
    *,
    purpose: str,
    context: str,
    reader: str = "",
    exclude_spans: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prepare bounded, sentence-target batches for the registered Jev tool.

    No heading or other content is excluded heuristically. A caller may pass
    exact ``exclude_spans`` (code-point ``start``/``end``) only when the source
    explicitly identifies those ranges as non-target headings. Spans must
    cover whole sentences; partial exclusions fail rather than lose prose.
    """
    if not isinstance(body, str) or not isinstance(purpose, str) or not isinstance(context, str) or not isinstance(reader, str):
        raise ValueError("body, purpose, context, and reader must be strings")
    spans = _sentence_spans(body)
    excluded: set[int] = set()
    for exclusion in exclude_spans or []:
        if not isinstance(exclusion, dict) or set(exclusion) != {"start", "end"}:
            raise ValueError("exclude_spans must contain exact start/end pairs")
        a, b = exclusion["start"], exclusion["end"]
        if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int) or not isinstance(b, int) or not 0 <= a < b <= len(body):
            raise ValueError("invalid explicit exclusion span")
        matches = [n for n, (s, e) in enumerate(spans) if s >= a and e <= b]
        if not matches or spans[matches[0]][0] != a or spans[matches[-1]][1] != b:
            raise ValueError("explicit exclusion must exactly cover complete sentence spans")
        excluded.update(matches)

    targets = [(n, span) for n, span in enumerate(spans) if n not in excluded]
    if len(targets) > MAX_TARGETS_PER_BATCH * MAX_BATCHES_PER_PASS:
        raise CoverageLimitError(len(targets))

    segments = []
    for n, (start, end) in targets:
        previous = body[slice(*spans[n - 1])] if n > 0 else ""
        following = body[slice(*spans[n + 1])] if n + 1 < len(spans) else ""
        segments.append({
            "id": f"sentence-{n + 1:04d}",
            "text": body[start:end],
            "previous_sentence": previous,
            "next_sentence": following,
            "source_start": start,
            "source_end": end,
        })
    batches = [segments[i:i + MAX_TARGETS_PER_BATCH] for i in range(0, len(segments), MAX_TARGETS_PER_BATCH)]
    return {
        "purpose": purpose,
        "context": context,
        "reader": reader,
        "target_count": len(segments),
        "batch_count": len(batches),
        "batches": [
            {"purpose": purpose, "context": context, "reader": reader,
             "segments": [{k: v for k, v in segment.items() if not k.startswith("source_")} for segment in batch],
             "source_spans": [{"id": segment["id"], "start": segment["source_start"], "end": segment["source_end"]} for segment in batch]}
            for batch in batches
        ],
    }
