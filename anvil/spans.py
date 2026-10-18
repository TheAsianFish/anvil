"""Entity spans: what an extraction teacher says and what the student is trained on.

A teacher answer is a list of `Mention`s: the entity's text copied from the input, its type and
the character offset the teacher claims it starts at. Language models copy text well and count
characters badly, so a mention's text is the evidence and its offset is a hint; `align.py`
turns mentions into verified `Span`s whose offsets index the input exactly.

Offsets are Python string indices (Unicode code points), half-open: `text[start:end]`.

The label store is task-agnostic: an extraction answer is stored as the canonical JSON of its
mentions in `Label.label`, so the cache, label files, replay and the store work unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Span:
    """A verified entity: `text[start:end]` is the mention, `type` one of the task's types."""

    start: int
    end: int
    type: str

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError(f"span offsets must satisfy 0 <= start < end, got {self}")

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end

    def text_in(self, text: str) -> str:
        return text[self.start : self.end]


@dataclass(frozen=True)
class Mention:
    """One entity as the teacher reported it, before verification against the input."""

    text: str
    type: str
    start: int


def encode_answer(mentions: Sequence[Mention]) -> str:
    """The canonical JSON stored as the label: sorted by offset, then text and type, so the
    same answer always has the same string (labels and recordings diff cleanly)."""
    rows = sorted(
        ({"start": m.start, "text": m.text, "type": m.type} for m in mentions),
        key=lambda r: (r["start"], r["text"], r["type"]),
    )
    return json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def decode_answer(label: str) -> list[Mention]:
    """Mentions from a stored label; `ValueError` when it is not an extraction answer."""
    try:
        rows = json.loads(label)
    except json.JSONDecodeError as exc:
        raise ValueError(f"extraction label is not JSON: {label[:80]!r}") from exc
    if not isinstance(rows, list):
        raise ValueError(f"extraction label must be a JSON list of mentions, got {label[:80]!r}")
    out: list[Mention] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"mention must be an object, got {row!r}")
        text, kind, start = row.get("text"), row.get("type"), row.get("start")
        if not isinstance(text, str) or not isinstance(kind, str):
            raise ValueError(f"mention needs string text and type, got {row!r}")
        if not isinstance(start, int) or isinstance(start, bool):
            raise ValueError(f"mention start must be an integer, got {row!r}")
        out.append(Mention(text=text, type=kind, start=start))
    return out


def spans_to_mentions(text: str, spans: Sequence[Span]) -> list[Mention]:
    """Mentions with exact offsets (gold labels, or a student's predictions as an answer)."""
    return [Mention(text=s.text_in(text), type=s.type, start=s.start) for s in spans]
