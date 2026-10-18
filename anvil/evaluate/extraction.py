"""Agreement metrics for span extraction: the student's spans against the teacher's.

Each held-out input i has the teacher's spans T_i and the student's spans S_i (verified
`Span`s, half-open character offsets into the input text). Two views of agreement are
reported, because each hides something the other shows:

- **Entity level, exact match.** A student span is a true positive iff the identical
  (start, end, type) is among the teacher's spans of the same input. This is the strict
  CoNLL criterion: a span one character short is as wrong as a missing one, which is what a
  downstream consumer of the extractions sees. Micro precision, recall and F1 pool the
  counts over all inputs (and, per type, over one type). The error breakdown then says
  *how* the misses fail (boundary, type, both, or nothing found), which exact match alone
  hides.
- **Word level, tag agreement.** Every whitespace-delimited word gets a BIO tag from each
  rater and Cohen's kappa is taken over all words pooled. Kappa gives partial credit for a
  boundary off by a word and is chance-corrected, so it is comparable with the
  classification tasks' kappa; the fraction of words the teacher tags O is reported next to
  it because always answering O already reaches that accuracy (with kappa 0).

Spans are compared as sets: a span listed twice in one rater's list of one input counts
once.

Bootstrap intervals resample *inputs*, never words or entities (a cluster bootstrap): the
words and entities of one input share its topic, style and difficulty, so they are not
independent draws, and resampling them one by one would give intervals that are too narrow.
Every per-input statistic is reduced to a small vector of counts (tp, fp, fn; or the word
tag marginals), the bootstrap weight matrix W[r, i] counts how often input i is drawn in
resample r, and W @ counts gives every resample's pooled counts in one product. The
interval is the 95% percentile interval, as in `bootstrap.py`, with the same seeded
`resample_indices`, so entity F1 and token kappa share their resamples.

The all-empty conventions of the point estimates (F1 1.0 when neither rater found an entity,
kappa 1.0 over no words) are not applied to resamples. A resample that drew only inputs
without entities (or without words) carries no information about agreement; scoring it 1.0
would drag the upper bound to 1.0 whenever a few inputs hold all the entities. Such a
replicate is undefined (NaN) and the percentiles are taken over the defined ones only, so
the interval is conditional on a resample containing something to score. `Interval` keeps
`n_resamples` as requested; the number actually used can be fewer.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from anvil.evaluate.bootstrap import Interval, resample_indices
from anvil.evaluate.metrics import cohen_kappa
from anvil.evaluate.slices import Slice
from anvil.spans import Span

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

_WORD = re.compile(r"\S+")


# --- entity level -------------------------------------------------------------------------


def _check_lengths(**columns: Sequence[object]) -> int:
    lengths = {name: len(col) for name, col in columns.items()}
    if len(set(lengths.values())) > 1:
        detail = ", ".join(f"{name} {n}" for name, n in lengths.items())
        raise ValueError(f"one entry per input is needed, but lengths differ: {detail}")
    return next(iter(lengths.values()))


def _input_counts(
    teacher: Sequence[Sequence[Span]], student: Sequence[Sequence[Span]]
) -> IntArray:
    """Per-input (tp, fp, fn), shape (n, 3), over the deduplicated span sets."""
    n = _check_lengths(teacher=teacher, student=student)
    out = np.zeros((n, 3), dtype=np.int64)
    for i, (t, s) in enumerate(zip(teacher, student, strict=True)):
        ts, ss = set(t), set(s)
        tp = len(ts & ss)
        out[i] = (tp, len(ss) - tp, len(ts) - tp)
    return out


def _f1_from_counts(counts: IntArray) -> FloatArray:
    """Micro F1 for each row (tp, fp, fn) of `counts` (shape (..., 3)).

    F1 = 2 tp / (2 tp + fp + fn), the harmonic mean of precision and recall written so it is
    0 whenever tp is 0. When a row has no entities at all (tp = fp = fn = 0) both raters
    agree that there is nothing to extract, and F1 is 1.0 (see `entity_scores`).
    """
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    den = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 1.0)
    out: FloatArray = f1.astype(np.float64)
    return out


def _f1_replicates(counts: IntArray) -> FloatArray:
    """Micro F1 for each row (tp, fp, fn) as in `_f1_from_counts`, but NaN for a row with no
    entities at all: a bootstrap resample without entities is undefined, not perfect."""
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    den = 2 * tp + fp + fn
    out: FloatArray = np.where(den > 0, 2 * tp / np.maximum(den, 1), np.nan).astype(np.float64)
    return out


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


@dataclass(frozen=True)
class EntityScores:
    """Exact-match micro precision, recall and F1, with the counts they come from."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    """Student spans with no identical teacher span."""
    fn: int
    """Teacher spans with no identical student span."""


def _scores(tp: int, fp: int, fn: int) -> EntityScores:
    if tp == fp == fn == 0:
        return EntityScores(precision=1.0, recall=1.0, f1=1.0, tp=0, fp=0, fn=0)
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = float(_f1_from_counts(np.array([tp, fp, fn], dtype=np.int64)))
    return EntityScores(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


def entity_scores(
    teacher: Sequence[Sequence[Span]], student: Sequence[Sequence[Span]]
) -> EntityScores:
    """Micro precision, recall and F1 of the student's spans against the teacher's.

    precision = tp / (tp + fp), recall = tp / (tp + fn), F1 their harmonic mean; a ratio with
    an empty denominator is 0 and F1 is 0 whenever tp is 0. One exception: when neither
    rater produced any entity on any input, the student agrees perfectly with the teacher
    that there is nothing to extract, so precision, recall and F1 are all 1.0 (seqeval would
    report 0 with a warning, which would score a perfect copy of an empty teacher as total
    disagreement).
    """
    tp, fp, fn = (int(x) for x in _input_counts(teacher, student).sum(axis=0))
    return _scores(tp, fp, fn)


@dataclass(frozen=True)
class TypeStats:
    """One entity type's row of the per-type table; the teacher's spans are the reference."""

    type: str
    precision: float
    recall: float
    f1: float
    support: int
    """Distinct spans of this type in the teacher's answers."""
    predicted: int
    """Distinct spans of this type in the student's answers."""


def _check_types(
    teacher: Sequence[Sequence[Span]], student: Sequence[Sequence[Span]], types: Sequence[str]
) -> None:
    known = set(types)
    unknown = sorted({s.type for spans in (*teacher, *student) for s in spans} - known)
    if unknown:
        raise ValueError(
            f"span types outside the task's types: {unknown[:5]}; the teacher and the student"
            " must both use the types in the spec"
        )


def per_type(
    teacher: Sequence[Sequence[Span]], student: Sequence[Sequence[Span]], types: Sequence[str]
) -> list[TypeStats]:
    """Exact-match precision, recall and F1 per entity type, rows in `types` order.

    Each row is `entity_scores` restricted to the spans of one type, so the same conventions
    hold: a type neither rater used scores 1.0 (support and predicted are both 0, which is
    the signal to read it as "not evaluated"). A span whose type is not in `types` is a
    `ValueError`.
    """
    _check_lengths(teacher=teacher, student=student)
    _check_types(teacher, student, types)
    tp: Counter[str] = Counter()
    support: Counter[str] = Counter()
    predicted: Counter[str] = Counter()
    for t, s in zip(teacher, student, strict=True):
        ts, ss = set(t), set(s)
        tp.update(x.type for x in ts & ss)
        support.update(x.type for x in ts)
        predicted.update(x.type for x in ss)
    rows = []
    for kind in types:
        scores = _scores(tp[kind], predicted[kind] - tp[kind], support[kind] - tp[kind])
        rows.append(
            TypeStats(
                type=kind,
                precision=scores.precision,
                recall=scores.recall,
                f1=scores.f1,
                support=support[kind],
                predicted=predicted[kind],
            )
        )
    return rows


@dataclass(frozen=True)
class ErrorBreakdown:
    """Where the student fails, by teacher span (the first five) and by student span.

    The first five fields partition the teacher's distinct spans, so they sum to the
    teacher's entity count; `spurious` counts student spans instead.
    """

    correct: int
    """An identical student span exists."""
    wrong_type: int
    """A student span has the same start and end but another type."""
    wrong_boundary: int
    """A student span of the same type overlaps it with a different extent."""
    wrong_both: int
    """Only student spans of other types and different extents overlap it."""
    missed: int
    """No student span overlaps it."""
    spurious: int
    """Student spans that overlap no teacher span (false alarms)."""


def _categorise(t: Span, student: set[Span]) -> str:
    if t in student:
        return "correct"
    overlapping = [s for s in student if s.overlaps(t)]
    if any(s.start == t.start and s.end == t.end for s in overlapping):
        return "wrong_type"
    if any(s.type == t.type for s in overlapping):
        return "wrong_boundary"
    if overlapping:
        return "wrong_both"
    return "missed"


def error_breakdown(
    teacher: Sequence[Sequence[Span]], student: Sequence[Sequence[Span]]
) -> ErrorBreakdown:
    """Classify every teacher span by its best student counterpart in the same input.

    Each distinct teacher span gets exactly one category, the first that applies in the
    order correct > wrong_type > wrong_boundary > wrong_both > missed, so a teacher span that
    a student span matches exactly is correct even if other student spans also overlap it.
    The matching is not one-to-one: one long student span overlapping two teacher spans
    makes both of them boundary errors. A student span is spurious when it overlaps no
    teacher span at all; a student span that overlaps one but does not match it is already
    counted through that teacher span and is not spurious.
    """
    _check_lengths(teacher=teacher, student=student)
    counts: Counter[str] = Counter()
    spurious = 0
    for t, s in zip(teacher, student, strict=True):
        ts, ss = set(t), set(s)
        counts.update(_categorise(x, ss) for x in ts)
        spurious += sum(not any(x.overlaps(y) for y in ts) for x in ss)
    return ErrorBreakdown(
        correct=counts["correct"],
        wrong_type=counts["wrong_type"],
        wrong_boundary=counts["wrong_boundary"],
        wrong_both=counts["wrong_both"],
        missed=counts["missed"],
        spurious=spurious,
    )


# --- word level ---------------------------------------------------------------------------


def word_tags(text: str, spans: Sequence[Span]) -> list[str]:
    """One BIO tag per whitespace-delimited word of `text` (`re.finditer(r"\\S+")`).

    A word [ws, we) is tagged `B-<type>` when some span starts inside it
    (ws <= span.start < we), else `I-<type>` when it overlaps a span that started before it,
    else `O`. A span that starts mid-word therefore tags that whole word B. When several
    spans qualify for the chosen tag, the one with the smallest start wins, then the longer,
    then the type in sorted order, so the tag is a pure function of the span set. B takes
    precedence over I: a word in which a new span starts is tagged B even when an earlier
    span also covers it, because the start of an entity is the event a tagger must find.
    Two adjacent spans of the same type tag their first words B and B, never B and I, so a
    student that merges them is penalised on the second word.
    """
    for s in spans:
        if s.end > len(text):
            raise ValueError(f"span {s} ends past the end of a text of length {len(text)}")
    ordered = sorted(set(spans), key=lambda s: (s.start, -s.end, s.type))
    tags = []
    for word in _WORD.finditer(text):
        ws, we = word.start(), word.end()
        starting = next((s for s in ordered if ws <= s.start < we), None)
        if starting is not None:
            tags.append(f"B-{starting.type}")
            continue
        inside = next((s for s in ordered if s.start < ws < s.end), None)
        tags.append("O" if inside is None else f"I-{inside.type}")
    return tags


def _word_count(text: str) -> int:
    return sum(1 for _ in _WORD.finditer(text))


@dataclass(frozen=True)
class TagAgreement:
    """Word-level BIO tag agreement, all words of all inputs pooled."""

    n_words: int
    kappa: float
    accuracy: float
    all_o_accuracy: float
    """Accuracy of tagging every word O, i.e. the fraction of words the teacher tags O; the
    kappa of that baseline is 0."""


def _tags(
    texts: Sequence[str], teacher: Sequence[Sequence[Span]], student: Sequence[Sequence[Span]]
) -> tuple[list[list[str]], list[list[str]]]:
    _check_lengths(texts=texts, teacher=teacher, student=student)
    return (
        [word_tags(x, t) for x, t in zip(texts, teacher, strict=True)],
        [word_tags(x, s) for x, s in zip(texts, student, strict=True)],
    )


def _pooled(tags: list[list[str]]) -> list[str]:
    return [tag for row in tags for tag in row]


def tag_agreement(
    texts: Sequence[str], teacher: Sequence[Sequence[Span]], student: Sequence[Sequence[Span]]
) -> TagAgreement:
    """Cohen's kappa, accuracy and the all-O baseline over every word of every input.

    Kappa is `metrics.cohen_kappa` on the pooled tags, so a word in a long input weighs the
    same as a word in a short one. `ValueError` when the texts contain no word at all.
    """
    t_tags, s_tags = _tags(texts, teacher, student)
    a, b = _pooled(t_tags), _pooled(s_tags)
    if not a:
        raise ValueError("tag agreement needs at least one word in the texts")
    return TagAgreement(
        n_words=len(a),
        kappa=cohen_kappa(a, b),
        accuracy=sum(x == y for x, y in zip(a, b, strict=True)) / len(a),
        all_o_accuracy=sum(x == "O" for x in a) / len(a),
    )


def _tag_marginals(t_tags: list[list[str]], s_tags: list[list[str]]) -> IntArray:
    """Per-input kappa sufficient statistics, shape (n, 1 + 2L) over the tag alphabet of size L.

    Column 0 is the number of words both raters tag alike, then the teacher's L tag counts,
    then the student's. Kappa of pooled inputs needs only the trace and the marginals of the
    pooled confusion matrix, and those are sums of the per-input traces and marginals, so
    these 1 + 2L columns stand in for the full L * L confusion matrices (as in
    `bootstrap.kappa_replicates`).
    """
    labels = sorted(set(_pooled(t_tags)) | set(_pooled(s_tags)))
    index = {label: k for k, label in enumerate(labels)}
    width = len(labels)
    out = np.zeros((len(t_tags), 1 + 2 * width), dtype=np.int64)
    for i, (t, s) in enumerate(zip(t_tags, s_tags, strict=True)):
        a = np.fromiter((index[x] for x in t), dtype=np.int64, count=len(t))
        b = np.fromiter((index[x] for x in s), dtype=np.int64, count=len(s))
        out[i, 0] = int((a == b).sum())
        out[i, 1 : 1 + width] = np.bincount(a, minlength=width)
        out[i, 1 + width :] = np.bincount(b, minlength=width)
    return out


def _kappa_from_marginals(stats: IntArray) -> FloatArray:
    """Cohen's kappa for each row of pooled marginals (shape (R, 1 + 2L)).

    p_o = agree / n, p_e = sum_k rows_k * cols_k / n^2. As in `metrics.cohen_kappa`, p_e = 1
    (detected exactly on integers) gives 1.0 if the raters agree everywhere and 0.0
    otherwise. A resample that drew only inputs without words is undefined and gives NaN
    (see the module docstring).
    """
    width = (stats.shape[1] - 1) // 2
    agree = stats[:, 0]
    rows, cols = stats[:, 1 : 1 + width], stats[:, 1 + width :]
    n = rows.sum(axis=1)
    chance = (rows * cols).sum(axis=1)
    nn = n * n
    with np.errstate(divide="ignore", invalid="ignore"):
        p_o = agree / n
        p_e = chance / nn
        kappa = (p_o - p_e) / (1 - p_e)
    fallback = np.where(agree == n, 1.0, 0.0)
    kappa = np.where(chance == nn, fallback, kappa)
    out: FloatArray = np.where(n > 0, kappa, np.nan)
    return out


# --- bootstrap ----------------------------------------------------------------------------


def _weights(n: int, n_resamples: int, seed: int) -> IntArray:
    """W[r, i] = how many times input i is drawn in resample r, shape (n_resamples, n)."""
    idx = resample_indices(n, n_resamples, seed)
    offsets = (np.arange(n_resamples, dtype=np.int64) * n)[:, None]
    w: IntArray = np.bincount((offsets + idx).ravel(), minlength=n_resamples * n).reshape(
        n_resamples, n
    )
    return w


def _interval(
    estimate: float, values: FloatArray, confidence: float, n_resamples: int, what: str
) -> Interval:
    """Percentile interval over the defined (non-NaN) replicates; see the module docstring."""
    if np.isnan(values).all():
        raise ValueError(
            f"every bootstrap resample is undefined: no input has {what}, so there is no"
            " agreement to put an interval on; check that the teacher and student labels"
            " were loaded"
        )
    quantiles = [100 * (1 - confidence) / 2, 100 * (1 + confidence) / 2]
    low, high = np.nanpercentile(values, quantiles)
    return Interval(
        estimate=estimate,
        low=float(low),
        high=float(high),
        confidence=confidence,
        n_resamples=n_resamples,
    )


def _check_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


def entity_f1_ci(
    teacher: Sequence[Sequence[Span]],
    student: Sequence[Sequence[Span]],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for micro entity F1, resampling inputs.

    estimate is `entity_scores(...).f1` on all inputs; each resample pools the (tp, fp, fn)
    of the inputs it drew (with multiplicity) and recomputes F1 from the pooled counts. A
    resample with no entities from either rater is dropped (not scored 1.0), and if every
    resample is dropped, which happens when no input has an entity, this is a `ValueError`.
    """
    _check_confidence(confidence)
    counts = _input_counts(teacher, student)
    n = len(counts)
    values = _f1_replicates(_weights(n, n_resamples, seed) @ counts)
    estimate = float(_f1_from_counts(counts.sum(axis=0)))
    return _interval(estimate, values, confidence, n_resamples, "an entity")


def entity_f1_difference_ci(
    teacher: Sequence[Sequence[Span]],
    a: Sequence[Sequence[Span]],
    b: Sequence[Sequence[Span]],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Paired bootstrap interval for F1(teacher, a) - F1(teacher, b), resampling inputs.

    Both systems are scored on the same resampled inputs, so the interval is on the
    difference itself (see `bootstrap.bootstrap_difference_ci` for why that is sharper than
    comparing two separate intervals). With the same seed it uses the same resamples as
    `entity_f1_ci`. A resample in which either system's F1 is undefined (no entities from
    the teacher or that system) is dropped, and if every resample is dropped this is a
    `ValueError`.
    """
    _check_confidence(confidence)
    _check_lengths(teacher=teacher, a=a, b=b)
    counts_a, counts_b = _input_counts(teacher, a), _input_counts(teacher, b)
    w = _weights(len(counts_a), n_resamples, seed)
    values = _f1_replicates(w @ counts_a) - _f1_replicates(w @ counts_b)
    estimate = float(_f1_from_counts(counts_a.sum(axis=0)) - _f1_from_counts(counts_b.sum(axis=0)))
    return _interval(estimate, values, confidence, n_resamples, "an entity")


def token_kappa_ci(
    texts: Sequence[str],
    teacher: Sequence[Sequence[Span]],
    student: Sequence[Sequence[Span]],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for word-tag kappa, resampling inputs (not words).

    estimate is `tag_agreement(...).kappa`; each resample pools the tag marginals of the
    inputs it drew. A resample that drew only inputs without words is dropped.
    """
    _check_confidence(confidence)
    t_tags, s_tags = _tags(texts, teacher, student)
    pooled_t, pooled_s = _pooled(t_tags), _pooled(s_tags)
    if not pooled_t:
        raise ValueError("tag agreement needs at least one word in the texts")
    stats = _tag_marginals(t_tags, s_tags)
    values = _kappa_from_marginals(_weights(len(stats), n_resamples, seed) @ stats)
    estimate = cohen_kappa(pooled_t, pooled_s)
    return _interval(estimate, values, confidence, n_resamples, "a word")


# --- slices -------------------------------------------------------------------------------


def by_length(
    texts: Sequence[str],
    teacher: Sequence[Sequence[Span]],
    student: Sequence[Sequence[Span]],
    buckets: int = 4,
) -> list[Slice]:
    """Entity F1 by input length in words, shortest first; quartiles by default.

    Buckets are formed exactly as in `slices.by_length` (stable sort by word count, then
    np.array_split into min(buckets, n) equal-count chunks of inputs). Each `Slice` has the
    bucket's word-count range and number of inputs, and its `agreement` is the micro entity
    F1 of the inputs in the bucket (with the all-empty convention of `entity_scores`), the
    extraction analogue of the classification slice's accuracy.
    """
    n = _check_lengths(texts=texts, teacher=teacher, student=student)
    if buckets < 1:
        raise ValueError(f"buckets must be positive, got {buckets}")
    if n == 0:
        return []
    lengths = np.fromiter((_word_count(x) for x in texts), dtype=np.float64, count=n)
    counts = _input_counts(teacher, student)
    order = np.argsort(lengths, kind="stable")
    return [
        Slice(
            lo=float(lengths[chunk].min()),
            hi=float(lengths[chunk].max()),
            n=len(chunk),
            agreement=float(_f1_from_counts(counts[chunk].sum(axis=0))),
        )
        for chunk in np.array_split(order, min(buckets, n))
    ]


# --- reports ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionAgreement:
    """Student-vs-teacher extraction agreement on the held-out split; `asdict` of it is the
    eval report."""

    n: int
    """Held-out inputs."""
    teacher_entities: int
    """Distinct teacher spans over all inputs."""
    student_entities: int
    """Distinct student spans over all inputs."""
    entity: EntityScores
    entity_f1_ci: Interval
    per_type: list[TypeStats]
    errors: ErrorBreakdown
    token_kappa: float
    token_kappa_ci: Interval
    token_accuracy: float
    all_o_accuracy: float
    """Word accuracy of tagging everything O (its kappa is 0)."""
    by_length: list[Slice]


def extraction_agreement(
    texts: Sequence[str],
    teacher: Sequence[Sequence[Span]],
    student: Sequence[Sequence[Span]],
    types: Sequence[str],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> ExtractionAgreement:
    """The full extraction report: entity and word-level agreement, 95% CIs, per-type, errors
    and length slices. Both CIs use the same seed and so the same input resamples."""
    n = _check_lengths(texts=texts, teacher=teacher, student=student)
    if n == 0:
        raise ValueError("extraction agreement needs at least one input")
    entity = entity_scores(teacher, student)
    tags = tag_agreement(texts, teacher, student)
    return ExtractionAgreement(
        n=n,
        teacher_entities=entity.tp + entity.fn,
        student_entities=entity.tp + entity.fp,
        entity=entity,
        entity_f1_ci=entity_f1_ci(teacher, student, n_resamples=n_resamples, seed=seed),
        per_type=per_type(teacher, student, types),
        errors=error_breakdown(teacher, student),
        token_kappa=tags.kappa,
        token_kappa_ci=token_kappa_ci(texts, teacher, student, n_resamples=n_resamples, seed=seed),
        token_accuracy=tags.accuracy,
        all_o_accuracy=tags.all_o_accuracy,
        by_length=by_length(texts, teacher, student),
    )


@dataclass(frozen=True)
class ExtractionCeiling:
    """Teacher self-agreement on inputs it labelled twice (two prompt paraphrases)."""

    n: int
    entity_f1: float
    entity_f1_ci: Interval
    token_kappa: float
    token_kappa_ci: Interval


def extraction_self_agreement(
    texts_by_id: Mapping[str, str],
    first: Mapping[str, Sequence[Span]],
    second: Mapping[str, Sequence[Span]],
    n_resamples: int = 1000,
    seed: int = 0,
) -> ExtractionCeiling:
    """The parity ceiling for extraction: the teacher's agreement with itself, by input id.

    As with `metrics.self_agreement`, a student trained on one labelling cannot be expected
    to agree with the teacher more than a second labelling does. Only ids in both labellings
    count; the first labelling plays the teacher's role (F1 is symmetric in the two).
    """
    ids = sorted(first.keys() & second.keys())
    if not ids:
        raise ValueError("the two labellings share no input ids; label the same inputs twice")
    missing = [i for i in ids if i not in texts_by_id]
    if missing:
        raise ValueError(f"no text for input ids {missing[:5]}; pass the text of every id")
    texts = [texts_by_id[i] for i in ids]
    a = [first[i] for i in ids]
    b = [second[i] for i in ids]
    return ExtractionCeiling(
        n=len(ids),
        entity_f1=entity_scores(a, b).f1,
        entity_f1_ci=entity_f1_ci(a, b, n_resamples=n_resamples, seed=seed),
        token_kappa=tag_agreement(texts, a, b).kappa,
        token_kappa_ci=token_kappa_ci(texts, a, b, n_resamples=n_resamples, seed=seed),
    )
