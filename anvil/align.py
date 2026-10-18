"""Alignment: turn teacher answers into training targets the student can learn.

Extraction labels pass two gates before training.

1. **Verify against the input** (`align_mentions`). A teacher mention carries the text it
   copied and the character offset it claims. The text is trusted, the offset is a hint: a
   language model reads tokens, not characters, so its offsets drift (off by one, counted in
   UTF-16 units past an emoji, counted from the wrong occurrence) while the copied text is
   usually exact. Each mention is matched to an occurrence of its text in the input, from the
   strictest rule to the loosest, and the rule that matched is recorded:

   - `exact`: the text sits at the claimed offset;
   - `shifted`: the text occurs elsewhere; the occurrence nearest the claimed offset wins
     (one already taken by an earlier mention is skipped, so two copies of a word map to two
     occurrences);
   - `case`: only a case-insensitive match exists ("Burger King" for "burger king");
   - `spacing`: only a match with different whitespace exists (a double space or a newline);
   - `normalized`: only a match after Unicode NFC normalisation exists (a precomposed "é"
     against "e" plus a combining accent);
   - `trimmed`: only after stripping punctuation and spaces from the mention's edges.

   A mention that matches nothing is dropped as `not_found`; one with empty text or an
   unknown type as `invalid`. Spans that overlap are resolved by keeping the longer (then the
   earlier, then the type listed first in the spec) and dropping the rest as `overlap`: BIO
   tags cannot represent nested or crossing entities. An identical repeat is a `duplicate`,
   dropped without counting against the teacher.

2. **Snap to the student's tokens** (`snap_to_tokens`). A span becomes `B-type` on the first
   token it touches and `I-type` on the rest. When a span boundary falls inside a token (the
   student's vocabulary splits words differently from the teacher), the span grows to the
   token boundary and is counted as `snapped`; it still trains. A span no token covers (cut
   off by `max_length`) is dropped as `no_token`, and one that would share a token with an
   earlier span as `token_conflict`.

`AlignmentReport` adds both gates up. Its unaligned rate is the share of mentions that cannot
be trained on (every drop except duplicates), and training refuses to start above
`MAX_UNALIGNED_RATE`: past that point the labels or the prompt are broken, and a student
trained on what is left would be scored against answers it never saw.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from anvil.errors import AlignmentError
from anvil.spans import Mention, Span

MAX_UNALIGNED_RATE = 0.05
"""Training refuses to start when more than this share of teacher mentions cannot be aligned."""

MATCH_RULES = ("exact", "shifted", "case", "spacing", "normalized", "trimmed")
"""How a mention was found in the input, strictest first."""

DROP_REASONS = ("not_found", "invalid", "overlap", "no_token", "token_conflict")
"""Why a mention was not trained on; every one counts toward the unaligned rate."""

_EDGE = " \t\n\r\f\v.,;:!?\"'()[]{}<>`*_-"
"""Characters stripped from a mention's edges by the `trimmed` rule."""

Offset = tuple[int, int]

OUTSIDE = "O"


def tag_names(types: Sequence[str]) -> list[str]:
    """The student's tag alphabet: `O`, then `B-` and `I-` per type, in spec order."""
    names = [OUTSIDE]
    for t in types:
        names += [f"B-{t}", f"I-{t}"]
    return names


@dataclass
class AlignmentReport:
    """What alignment did to the teacher's mentions. Counts add across inputs (`merge`)."""

    mentions: int = 0
    """Mentions the teacher returned, duplicates included."""
    matched: dict[str, int] = field(default_factory=lambda: dict.fromkeys(MATCH_RULES, 0))
    dropped: dict[str, int] = field(default_factory=lambda: dict.fromkeys(DROP_REASONS, 0))
    duplicates: int = 0
    snapped: int = 0
    """Aligned spans whose boundary moved to a token boundary (still trained on)."""
    examples: list[dict[str, object]] = field(default_factory=list)
    """Up to `MAX_EXAMPLES` dropped or repaired mentions, for the report."""

    MAX_EXAMPLES = 20

    @property
    def counted(self) -> int:
        """Mentions the unaligned rate is computed over (duplicates are not the teacher's
        error to count twice, and not a loss)."""
        return self.mentions - self.duplicates

    @property
    def unaligned(self) -> int:
        return sum(self.dropped.values())

    @property
    def unaligned_rate(self) -> float:
        return self.unaligned / self.counted if self.counted else 0.0

    @property
    def repaired(self) -> int:
        """Mentions found only by a rule looser than `exact`."""
        return sum(n for rule, n in self.matched.items() if rule != "exact")

    def note(self, input_id: str, outcome: str, mention: Mention, span: Span | None) -> None:
        if len(self.examples) < self.MAX_EXAMPLES:
            example: dict[str, object] = {
                "input_id": input_id,
                "outcome": outcome,
                "text": mention.text,
                "type": mention.type,
                "claimed_start": mention.start,
            }
            if span is not None:
                example["start"], example["end"] = span.start, span.end
            self.examples.append(example)

    def merge(self, other: AlignmentReport) -> None:
        self.mentions += other.mentions
        self.duplicates += other.duplicates
        self.snapped += other.snapped
        for k, v in other.matched.items():
            self.matched[k] = self.matched.get(k, 0) + v
        for k, v in other.dropped.items():
            self.dropped[k] = self.dropped.get(k, 0) + v
        room = self.MAX_EXAMPLES - len(self.examples)
        if room > 0:
            self.examples += other.examples[:room]

    def _consistent(self) -> None:
        """A report that counts mentions can drop each at most once. (A token-stage report
        alone counts none: its mentions were counted by the text stage it merges into.)"""
        if self.mentions and self.unaligned > self.counted:
            raise ValueError(
                f"alignment report counts {self.unaligned} drops for {self.counted} mentions;"
                " a stage's report was merged twice"
            )

    def as_dict(self) -> dict[str, object]:
        self._consistent()
        return {
            "mentions": self.mentions,
            "duplicates": self.duplicates,
            "matched": dict(self.matched),
            "repaired": self.repaired,
            "dropped": dict(self.dropped),
            "unaligned": self.unaligned,
            "unaligned_rate": self.unaligned_rate,
            "snapped": self.snapped,
            "examples": list(self.examples),
        }

    def check(self, limit: float = MAX_UNALIGNED_RATE) -> None:
        """Raise `AlignmentError` when the unaligned rate is above `limit`."""
        self._consistent()
        if self.unaligned_rate > limit:
            reasons = ", ".join(f"{k} {v}" for k, v in self.dropped.items() if v)
            raise AlignmentError(
                f"{self.unaligned} of {self.counted} teacher mentions"
                f" ({self.unaligned_rate:.1%}) could not be aligned ({reasons}), above the"
                f" {limit:.0%} limit; check the prompt and a few labels (the report lists"
                " examples), or raise max_length if most are no_token"
            )


def _occurrences(text: str, needle: str, rule: str) -> list[Offset]:
    """Every (start, end) in `text` matching `needle` under `rule`, left to right.

    Matches may overlap ("aa" in "aaa" is found at 0 and 1), since the claimed offset decides
    between them."""
    if rule == "exact" or rule == "shifted":
        pattern = re.compile(f"(?=({re.escape(needle)}))")
    elif rule == "case":
        pattern = re.compile(f"(?=({re.escape(needle)}))", re.IGNORECASE)
    elif rule == "spacing":
        words = needle.split()
        if not words:
            return []
        body = r"\s+".join(re.escape(w) for w in words)
        pattern = re.compile(f"(?=({body}))", re.IGNORECASE)
    else:
        raise ValueError(f"no regex rule {rule!r}")
    return [(m.start(1), m.end(1)) for m in pattern.finditer(text) if m.end(1) > m.start(1)]


def _normalized_occurrences(text: str, needle: str) -> list[Offset]:
    """Matches after NFC normalisation, as offsets into the original text.

    Normalisation can change lengths (a base letter plus a combining accent becomes one code
    point), so offsets cannot be mapped arithmetically. Inputs to a tagger are short, so every
    (start, end) window up to twice the needle's length is compared directly; only mentions
    no cheaper rule found get here."""
    target = unicodedata.normalize("NFC", needle).casefold()
    if not target:
        return []
    out: list[Offset] = []
    width = 2 * len(needle) + 2
    for i in range(len(text)):
        if unicodedata.combining(text[i]):
            continue  # a match never starts on a combining mark
        for j in range(i + 1, min(len(text), i + width) + 1):
            if j < len(text) and unicodedata.combining(text[j]):
                continue  # nor ends between a letter and its accent
            if unicodedata.normalize("NFC", text[i:j]).casefold() == target:
                out.append((i, j))
                break
    return out


def _whole_characters(text: str, offset: Offset) -> bool:
    """False when a match would start on a combining mark or stop just before one: "cafe"
    found inside a decomposed "café" would cut the accent off its letter."""
    start, end = offset
    if start < len(text) and unicodedata.combining(text[start]):
        return False
    return not (end < len(text) and unicodedata.combining(text[end]))


def _locate(
    text: str, mention: Mention, taken: set[Offset], rules: Sequence[str] = MATCH_RULES
) -> tuple[Offset, str] | None:
    """The occurrence of the mention's text that the teacher most likely meant, and the rule
    that found it. `taken` holds the occurrences earlier mentions of the same type claimed."""
    needle = mention.text
    end = mention.start + len(needle)
    claimed = (mention.start, end)
    if (
        mention.start >= 0
        and text[mention.start : end] == needle
        and claimed not in taken
        and _whole_characters(text, claimed)
    ):
        return claimed, "exact"
    attempts: list[tuple[str, str]] = [
        ("shifted", needle),
        ("case", needle),
        ("spacing", needle),
        ("normalized", needle),
    ]
    trimmed = needle.strip(_EDGE)
    if trimmed and trimmed != needle:
        attempts += [("trimmed", trimmed)]
    for rule, target in attempts:
        if rule not in rules:
            continue
        if rule == "trimmed":
            found = _occurrences(text, target, "case") or _normalized_occurrences(text, target)
        elif rule == "normalized":
            found = _normalized_occurrences(text, target)
        else:
            found = _occurrences(text, target, rule)
        free = [o for o in found if o not in taken and _whole_characters(text, o)]
        if free:
            # Nearest to the claimed offset; on a tie, the earlier occurrence.
            best = min(free, key=lambda o: (abs(o[0] - mention.start), o[0]))
            return best, rule
    return None


@dataclass(frozen=True)
class CharAlignment:
    """The verified spans of one input (sorted, non-overlapping) and what happened."""

    spans: list[Span]
    report: AlignmentReport


def align_mentions(
    text: str,
    mentions: Sequence[Mention],
    types: Sequence[str],
    input_id: str = "",
    rules: Sequence[str] = MATCH_RULES,
) -> CharAlignment:
    """Verify each mention against `text` and return non-overlapping spans with exact offsets.

    Mentions are located in the teacher's order, so with two copies of a word the first
    mention claims the occurrence nearest its offset and the second the next free one.
    `rules` restricts the matching rules (`("exact",)` is plain verification, the baseline
    the repair rules are measured against).
    """
    unknown = set(rules) - set(MATCH_RULES)
    if unknown or "exact" not in rules:
        raise ValueError(f"rules must include 'exact' and come from {MATCH_RULES}")
    report = AlignmentReport(mentions=len(mentions))
    rank = {t: i for i, t in enumerate(types)}
    # Occurrences already claimed, per type: a second mention of the same type moves on to
    # the next occurrence, while one of another type may claim the same text (and then loses
    # the overlap resolution below instead of being pushed somewhere it does not belong).
    taken: dict[str, set[Offset]] = {t: set() for t in types}
    located: list[tuple[Span, int]] = []
    seen: set[Mention] = set()
    for order, raw in enumerate(mentions):
        if raw.type not in rank or not raw.text.strip():
            report.dropped["invalid"] += 1
            report.note(input_id, "invalid", raw, None)
            continue
        # Edge whitespace is never part of an entity: a span kept with it would train the
        # student on a neighbouring token (byte-level BPE glues the space to the next word)
        # and could never be matched by a prediction, whose edges are stripped.
        lead = len(raw.text) - len(raw.text.lstrip())
        m = Mention(raw.text.strip(), raw.type, raw.start + lead)
        if m in seen:  # the same text, type and offset twice: one entity
            report.duplicates += 1
            continue
        seen.add(m)
        hit = _locate(text, m, taken[m.type], rules)
        if hit is None and _locate(text, m, set(), rules) is not None:
            # Found, but only where an earlier mention of the same type already sits: the
            # teacher listed one entity twice with different offsets, not a lost mention.
            report.duplicates += 1
            continue
        if hit is None:
            report.dropped["not_found"] += 1
            report.note(input_id, "not_found", m, None)
            continue
        (start, end), rule = hit
        span = Span(start, end, m.type)
        taken[m.type].add((start, end))
        report.matched[rule] += 1
        if rule != "exact":
            report.note(input_id, rule, m, span)
        located.append((span, order))

    # Longest first, then earliest, then spec order of the type, then teacher order.
    ranked = sorted(
        located, key=lambda so: (-(so[0].end - so[0].start), so[0].start, rank[so[0].type], so[1])
    )
    kept: list[Span] = []
    for span, order in ranked:
        if any(span.overlaps(k) for k in kept):
            report.dropped["overlap"] += 1
            report.note(input_id, "overlap", mentions[order], span)
            continue
        kept.append(span)
    return CharAlignment(sorted(kept), report)


@dataclass(frozen=True)
class TokenAlignment:
    """BIO tags for one input's tokens (`None` for special and padding tokens, which the loss
    ignores) and the spans as the student can represent them."""

    tags: list[str | None]
    spans: list[Span]
    report: AlignmentReport


def _is_real(offset: Offset) -> bool:
    return offset[1] > offset[0]


def _strip(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink [start, end) past whitespace at either edge (byte-level BPE tokens carry the
    space before a word)."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def snap_to_tokens(
    text: str,
    spans: Sequence[Span],
    offsets: Sequence[Offset],
    input_id: str = "",
) -> TokenAlignment:
    """BIO tags over the student's tokens. `offsets` is the tokenizer's offset mapping for
    `text` (special tokens are `(0, 0)`); `spans` must not overlap (`align_mentions` output).
    """
    # Counts only what this gate adds (drops and snaps): the mentions were counted when they
    # were aligned to the text, so merging the two reports does not count them twice.
    report = AlignmentReport()
    tags: list[str | None] = [OUTSIDE if _is_real(o) else None for o in offsets]
    owner: list[int] = [-1] * len(offsets)
    represented: list[Span] = []
    for n, span in enumerate(sorted(spans)):
        touching = [
            i
            for i, (s, e) in enumerate(offsets)
            if e > s and s < span.end and e > span.start and text[s:e].strip()
        ]
        # Every real token between the first and the last is inside the entity, including a
        # whitespace-only one (byte-level BPE makes a token of a double space): tagged O, it
        # would teach the student to split "pizza  hut" in two.
        covered = (
            [i for i in range(touching[0], touching[-1] + 1) if _is_real(offsets[i])]
            if touching
            else []
        )
        # A Mention is only for the example list; the span's own text is what it shows.
        as_mention = Mention(span.text_in(text), span.type, span.start)
        if not covered:
            report.dropped["no_token"] += 1
            report.note(input_id, "no_token", as_mention, span)
            continue
        if any(owner[i] != -1 for i in covered):
            report.dropped["token_conflict"] += 1
            report.note(input_id, "token_conflict", as_mention, span)
            continue
        for k, i in enumerate(covered):
            owner[i] = n
            tags[i] = f"{'B' if k == 0 else 'I'}-{span.type}"
        start, end = _strip(text, offsets[covered[0]][0], offsets[covered[-1]][1])
        snapped = Span(start, end, span.type)
        if (snapped.start, snapped.end) != (span.start, span.end):
            report.snapped += 1
            report.note(input_id, "snapped", as_mention, snapped)
        represented.append(snapped)
    return TokenAlignment(tags, represented, report)


def tags_to_spans(text: str, tags: Sequence[str | None], offsets: Sequence[Offset]) -> list[Span]:
    """Character spans from per-token BIO tags (the student's predictions).

    Chunks follow the CoNLL convention: `B-x` opens a chunk; `I-x` continues a chunk of type
    `x` and otherwise opens one (an `I` after `O` or after another type is read as a start
    rather than thrown away); `O` and special tokens close it. A chunk covers its first
    token's start to its last token's end, without edge whitespace.
    """
    if len(tags) != len(offsets):
        raise ValueError(f"{len(tags)} tags for {len(offsets)} tokens")
    spans: list[Span] = []
    current: tuple[str, int, int] | None = None

    def close() -> None:
        if current is not None:
            start, end = _strip(text, current[1], current[2])
            if end > start:
                spans.append(Span(start, end, current[0]))

    for tag, (s, e) in zip(tags, offsets, strict=True):
        if tag is None or e <= s or tag == OUTSIDE:
            close()
            current = None
            continue
        prefix, _, kind = tag.partition("-")
        if prefix == "I" and current is not None and current[0] == kind:
            current = (kind, current[1], e)
            continue
        close()
        current = (kind, s, e)
    close()
    return spans


def align_all(
    rows: Iterable[tuple[str, str, Sequence[Mention]]], types: Sequence[str]
) -> tuple[list[list[Span]], AlignmentReport]:
    """`align_mentions` over (input id, text, mentions) rows, with one merged report."""
    report = AlignmentReport()
    out: list[list[Span]] = []
    for input_id, text, mentions in rows:
        aligned = align_mentions(text, mentions, types, input_id)
        report.merge(aligned.report)
        out.append(aligned.spans)
    return out, report


def outcome_counts(report: AlignmentReport) -> Counter[str]:
    """Every outcome in one counter (matched rules, drops, duplicates), for logs."""
    counts: Counter[str] = Counter()
    counts.update({k: v for k, v in report.matched.items() if v})
    counts.update({k: v for k, v in report.dropped.items() if v})
    if report.duplicates:
        counts["duplicate"] = report.duplicates
    return counts
