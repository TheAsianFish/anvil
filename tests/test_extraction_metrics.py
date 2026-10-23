from __future__ import annotations

import json
import warnings
from dataclasses import asdict

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from seqeval.metrics import f1_score  # type: ignore[import-untyped]
from sklearn.metrics import cohen_kappa_score

from anvil.evaluate.bootstrap import Interval, resample_indices
from anvil.evaluate.extraction import (
    EntityScores,
    ErrorBreakdown,
    TypeStats,
    by_length,
    entity_f1_ci,
    entity_f1_difference_ci,
    entity_f1_statistic,
    entity_scores,
    error_breakdown,
    errors_by_input,
    extraction_agreement,
    extraction_self_agreement,
    per_type,
    tag_agreement,
    token_kappa_ci,
    word_tags,
)
from anvil.evaluate.slices import Slice
from anvil.spans import Span
from eval.data import bio_to_spans as gold_bio_to_spans

P, L = "PER", "LOC"


# --- entity scores ------------------------------------------------------------------------


def test_entity_scores_hand_case() -> None:
    teacher = [[Span(0, 4, P), Span(10, 15, L)], [Span(3, 6, P)]]
    student = [[Span(0, 4, P), Span(10, 14, L)], [Span(3, 6, L), Span(8, 9, P)]]
    # tp 1; fp: (10,14,L), (3,6,L), (8,9,P) = 3; fn: (10,15,L), (3,6,P) = 2.
    s = entity_scores(teacher, student)
    assert (s.tp, s.fp, s.fn) == (1, 3, 2)
    assert s.precision == pytest.approx(1 / 4)
    assert s.recall == pytest.approx(1 / 3)
    assert s.f1 == pytest.approx(2 * (1 / 4) * (1 / 3) / (1 / 4 + 1 / 3))


def test_matches_only_within_the_same_input() -> None:
    s = entity_scores([[Span(0, 2, P)], []], [[], [Span(0, 2, P)]])
    assert (s.tp, s.fp, s.fn, s.f1) == (0, 1, 1, 0.0)


def test_duplicates_count_once() -> None:
    s = entity_scores([[Span(0, 2, P), Span(0, 2, P)]], [[Span(0, 2, P)] * 3])
    assert s == EntityScores(precision=1.0, recall=1.0, f1=1.0, tp=1, fp=0, fn=0)


def test_all_empty_is_perfect_agreement() -> None:
    assert entity_scores([[], []], [[], []]) == EntityScores(1.0, 1.0, 1.0, 0, 0, 0)


def test_empty_student_or_empty_teacher_scores_zero() -> None:
    assert entity_scores([[Span(0, 1, P)]], [[]]) == EntityScores(0.0, 0.0, 0.0, 0, 0, 1)
    assert entity_scores([[]], [[Span(0, 1, P)]]) == EntityScores(0.0, 0.0, 0.0, 0, 1, 0)


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="lengths differ"):
        entity_scores([[]], [[], []])
    with pytest.raises(ValueError, match="lengths differ"):
        tag_agreement(["a"], [[]], [[], []])


def test_per_type_hand_case() -> None:
    teacher = [[Span(0, 4, P), Span(10, 15, L)], [Span(3, 6, P)]]
    student = [[Span(0, 4, P), Span(10, 14, L)], [Span(3, 6, P), Span(8, 9, P)]]
    rows = per_type(teacher, student, [P, L, "ORG"])
    assert rows == [
        TypeStats(type=P, precision=2 / 3, recall=1.0, f1=0.8, support=2,
                  predicted=3),
        TypeStats(type=L, precision=0.0, recall=0.0, f1=0.0, support=1, predicted=1),
        TypeStats(type="ORG", precision=1.0, recall=1.0, f1=1.0, support=0, predicted=0),
    ]  # fmt: skip


def test_per_type_counts_sum_to_overall() -> None:
    rng = np.random.default_rng(9)
    teacher, student = random_corpus(rng, 40)
    rows = per_type(teacher, student, [P, L])
    overall = entity_scores(teacher, student)
    # tp of a row is recall * support; round to undo the float division.
    assert sum(round(r.recall * r.support) for r in rows) == overall.tp
    assert sum(r.support for r in rows) == sum(len(set(t)) for t in teacher)
    assert sum(r.predicted for r in rows) == sum(len(set(s)) for s in student)
    assert sum(r.support for r in rows) == overall.tp + overall.fn


def test_per_type_rejects_unknown_types() -> None:
    with pytest.raises(ValueError, match="outside the task's types"):
        per_type([[Span(0, 1, "X")]], [[]], [P])


# --- error breakdown ----------------------------------------------------------------------


def _errors(teacher: list[Span], student: list[Span]) -> ErrorBreakdown:
    return error_breakdown([teacher], [student])


def test_each_error_category() -> None:
    t = Span(5, 10, P)
    zero = ErrorBreakdown(0, 0, 0, 0, 0, 0)
    assert _errors([t], [Span(5, 10, P)]) == ErrorBreakdown(1, 0, 0, 0, 0, 0)
    assert _errors([t], [Span(5, 10, L)]) == ErrorBreakdown(0, 1, 0, 0, 0, 0)
    assert _errors([t], [Span(6, 12, P)]) == ErrorBreakdown(0, 0, 1, 0, 0, 0)
    assert _errors([t], [Span(3, 7, L)]) == ErrorBreakdown(0, 0, 0, 1, 0, 0)
    assert _errors([t], [Span(10, 12, P)]) == ErrorBreakdown(0, 0, 0, 0, 1, 1)
    assert _errors([t], []) == ErrorBreakdown(0, 0, 0, 0, 1, 0)
    assert _errors([], [Span(0, 2, L)]) == ErrorBreakdown(0, 0, 0, 0, 0, 1)
    assert _errors([], []) == zero


def test_error_precedence() -> None:
    t = Span(5, 10, P)
    # Exact match wins over every other overlapping span.
    assert _errors([t], [t, Span(5, 10, L), Span(4, 8, P)]).correct == 1
    # Same extent with the wrong type beats a same-type boundary error.
    assert _errors([t], [Span(5, 10, L), Span(4, 8, P)]).wrong_type == 1
    # A same-type overlap beats an other-type overlap.
    assert _errors([t], [Span(3, 7, L), Span(8, 12, P)]).wrong_boundary == 1


def test_one_student_span_over_two_teacher_spans() -> None:
    e = _errors([Span(0, 3, P), Span(4, 7, P)], [Span(0, 7, P)])
    assert e == ErrorBreakdown(0, 0, 2, 0, 0, 0)


def test_error_categories_partition_teacher_spans() -> None:
    rng = np.random.default_rng(3)
    teacher, student = random_corpus(rng, 30)
    e = error_breakdown(teacher, student)
    total = sum(len(set(t)) for t in teacher)
    assert e.correct + e.wrong_type + e.wrong_boundary + e.wrong_both + e.missed == total
    assert e.correct == entity_scores(teacher, student).tp


def _sum_breakdowns(rows: list[ErrorBreakdown]) -> ErrorBreakdown:
    fields = [asdict(r) for r in rows]
    return ErrorBreakdown(**{k: sum(f[k] for f in fields) for k in asdict(rows[0])})


@pytest.mark.parametrize("seed", range(6))
def test_errors_by_input_rows_sum_to_the_breakdown(seed: int) -> None:
    rng = np.random.default_rng(seed)
    teacher, student = random_corpus(rng, 40, p_keep=0.5)
    rows = errors_by_input(teacher, student)
    assert len(rows) == 40
    assert _sum_breakdowns(rows) == error_breakdown(teacher, student)
    for t, s, row in zip(teacher, student, rows, strict=True):
        assert row == error_breakdown([t], [s])


def test_errors_by_input_hand_case_one_of_each_kind() -> None:
    t = Span(5, 10, P)
    teacher = [[t], [t], [t], [t], [], [t, Span(20, 25, L)], []]
    student = [
        [Span(6, 12, P)],  # wrong_boundary
        [Span(5, 10, L)],  # wrong_type
        [Span(3, 7, L)],  # wrong_both
        [],  # missed
        [Span(0, 2, L)],  # spurious
        [t, Span(30, 32, P)],  # one correct, one missed, one spurious
        [],  # nothing at all
    ]
    rows = errors_by_input(teacher, student)
    assert rows == [
        ErrorBreakdown(0, 0, 1, 0, 0, 0),
        ErrorBreakdown(0, 1, 0, 0, 0, 0),
        ErrorBreakdown(0, 0, 0, 1, 0, 0),
        ErrorBreakdown(0, 0, 0, 0, 1, 0),
        ErrorBreakdown(0, 0, 0, 0, 0, 1),
        ErrorBreakdown(1, 0, 0, 0, 1, 1),
        ErrorBreakdown(0, 0, 0, 0, 0, 0),
    ]
    assert _sum_breakdowns(rows) == error_breakdown(teacher, student)
    assert error_breakdown(teacher, student) == ErrorBreakdown(1, 1, 1, 1, 2, 2)


def test_errors_by_input_rejects_length_mismatch_and_takes_no_inputs() -> None:
    with pytest.raises(ValueError, match="lengths differ"):
        errors_by_input([[]], [[], []])
    assert errors_by_input([], []) == []


# --- word tags ----------------------------------------------------------------------------


def test_word_tags_basic_and_multiword() -> None:
    text = "Ada Lovelace met Babbage in London"
    spans = [Span(0, 12, P), Span(17, 24, P), Span(28, 34, L)]
    assert word_tags(text, spans) == ["B-PER", "I-PER", "O", "B-PER", "O", "B-LOC"]


def test_span_starting_mid_word_tags_the_word_b() -> None:
    # "x-Paris is": the span starts at "Paris" inside the first word and runs into the next.
    assert word_tags("x-Paris is", [Span(2, 10, L)]) == ["B-LOC", "I-LOC"]


def test_adjacent_spans_of_same_type_are_b_b() -> None:
    assert word_tags("Ada Bob Cy", [Span(0, 3, P), Span(4, 7, P)]) == ["B-PER", "B-PER", "O"]


def test_word_tag_ties() -> None:
    # Two spans start in the same word: the earlier start wins.
    assert word_tags("abcd", [Span(2, 4, L), Span(0, 2, P)]) == ["B-PER"]
    # Same start: the longer wins.
    assert word_tags("ab cd", [Span(0, 2, L), Span(0, 5, P)]) == ["B-PER", "I-PER"]
    # A span starting inside a word beats an earlier span covering it.
    assert word_tags("ab cd", [Span(0, 5, P), Span(3, 5, L)]) == ["B-PER", "B-LOC"]


def test_word_tags_whitespace_and_bounds() -> None:
    assert word_tags("  a\n\tb  ", []) == ["O", "O"]
    assert word_tags("", []) == []
    with pytest.raises(ValueError, match="past the end"):
        word_tags("ab", [Span(0, 3, P)])


def conlleval_bio(tags: list[str]) -> list[str]:
    """The tags as a chunker reads them: an I-x that does not continue a chunk of type x
    opens one, so it is written B-x."""
    out: list[str] = []
    for i, tag in enumerate(tags):
        prefix, _, kind = tag.partition("-")
        prev = tags[i - 1].partition("-")[2] if i and tags[i - 1] != "O" else None
        out.append(f"B-{kind}" if prefix == "I" and prev != kind else tag)
    return out


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.text(alphabet="abcxyz", min_size=1, max_size=5), min_size=1, max_size=12,
             unique=True),
    st.data(),
)  # fmt: skip
def test_word_tags_round_trip_gold_bio(words: list[str], data: st.DataObject) -> None:
    # The gold loader writes spans from BIO word tags; reading them back as word tags must
    # give the same tags (after conlleval's reading of a stray I as B).
    tags = data.draw(st.lists(st.sampled_from(TAGS), min_size=len(words), max_size=len(words)))
    text, spans = gold_bio_to_spans(words, tags)
    assert word_tags(text, spans) == conlleval_bio(tags)


def test_tag_agreement_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    texts, teacher, student = random_texts(rng, 40)
    got = tag_agreement(texts, teacher, student)
    a = [x for t, s in zip(texts, teacher, strict=True) for x in word_tags(t, s)]
    b = [x for t, s in zip(texts, student, strict=True) for x in word_tags(t, s)]
    assert got.n_words == len(a)
    assert got.kappa == pytest.approx(cohen_kappa_score(a, b))
    assert got.accuracy == pytest.approx(np.mean([x == y for x, y in zip(a, b, strict=True)]))
    assert got.all_o_accuracy == pytest.approx(np.mean([x == "O" for x in a]))


def test_tag_agreement_needs_words() -> None:
    with pytest.raises(ValueError, match="at least one word"):
        tag_agreement(["  "], [[]], [[]])


# --- seqeval oracle -----------------------------------------------------------------------

TAGS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]


def bio_to_spans(tags: list[str]) -> list[Span]:
    """conlleval chunking over single-character words at offsets 0, 2, 4, ...

    A chunk starts at B-x, or at I-x after O or after a tag of another type; it ends before
    O, before any B, and before a tag of another type.
    """
    spans: list[Span] = []
    start: int | None = None
    kind = ""
    for i, tag in enumerate([*tags, "O"]):
        prefix, _, t = tag.partition("-")
        continues = prefix == "I" and start is not None and t == kind
        if start is not None and not continues:
            spans.append(Span(2 * start, 2 * (i - 1) + 1, kind))
            start = None
        if prefix == "B" or (prefix == "I" and not continues):
            start, kind = i, t
    return spans


sentences = st.lists(st.lists(st.sampled_from(TAGS), min_size=1, max_size=12), min_size=1,
                     max_size=6)  # fmt: skip


@settings(max_examples=150, deadline=None)
@given(sentences, st.data())
def test_entity_f1_matches_seqeval(truth: list[list[str]], data: st.DataObject) -> None:
    pred = [data.draw(st.lists(st.sampled_from(TAGS), min_size=len(s), max_size=len(s)))
            for s in truth]  # fmt: skip
    teacher = [bio_to_spans(s) for s in truth]
    student = [bio_to_spans(s) for s in pred]
    ours = entity_scores(teacher, student)
    if ours.tp + ours.fp + ours.fn == 0:
        assert ours.f1 == 1.0  # our convention; seqeval reports 0 with a warning
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = f1_score(truth, pred)
    assert ours.f1 == pytest.approx(expected)


def test_bio_converter_follows_conlleval() -> None:
    assert bio_to_spans(["I-PER", "I-PER", "B-PER", "I-LOC", "O", "I-LOC"]) == [
        Span(0, 3, P),
        Span(4, 5, P),
        Span(6, 7, L),
        Span(10, 11, L),
    ]


# --- bootstrap ----------------------------------------------------------------------------


def random_corpus(
    rng: np.random.Generator, n: int, p_keep: float = 0.7
) -> tuple[list[list[Span]], list[list[Span]]]:
    """Teacher spans on a grid of 3-char words; the student keeps, shifts, retypes or drops."""
    teacher, student = [], []
    for _ in range(n):
        t, s = [], []
        for w in range(int(rng.integers(0, 8))):
            if rng.random() < 0.4:
                span = Span(4 * w, 4 * w + 3, P if rng.random() < 0.5 else L)
                t.append(span)
                u = rng.random()
                if u < p_keep:
                    s.append(span)
                elif u < p_keep + 0.1:
                    s.append(Span(span.start, span.end + 4, span.type))
                elif u < p_keep + 0.2:
                    s.append(Span(span.start, span.end, L if span.type == P else P))
            elif rng.random() < 0.05:
                s.append(Span(4 * w, 4 * w + 3, P))
        teacher.append(t)
        student.append(s)
    return teacher, student


def random_texts(
    rng: np.random.Generator, n: int, p_keep: float = 0.7
) -> tuple[list[str], list[list[Span]], list[list[Span]]]:
    teacher, student = random_corpus(rng, n, p_keep)
    texts = [" ".join(["abc"] * 10) for _ in range(n)]
    return texts, teacher, student


def test_ci_estimate_is_point_estimate_and_inside() -> None:
    rng = np.random.default_rng(1)
    texts, teacher, student = random_texts(rng, 60)
    f1 = entity_f1_ci(teacher, student, n_resamples=500, seed=4)
    assert f1.estimate == pytest.approx(entity_scores(teacher, student).f1)
    assert f1.low < f1.estimate < f1.high
    kappa = token_kappa_ci(texts, teacher, student, n_resamples=500, seed=4)
    assert kappa.estimate == pytest.approx(tag_agreement(texts, teacher, student).kappa)
    assert kappa.low < kappa.estimate < kappa.high
    assert (f1.confidence, f1.n_resamples, f1.method) == (0.95, 500, "percentile")


def test_ci_is_deterministic_given_seed() -> None:
    rng = np.random.default_rng(2)
    texts, teacher, student = random_texts(rng, 30)
    assert entity_f1_ci(teacher, student, seed=7) == entity_f1_ci(teacher, student, seed=7)
    assert token_kappa_ci(texts, teacher, student, seed=7) == token_kappa_ci(
        texts, teacher, student, seed=7
    )
    assert entity_f1_ci(teacher, student, seed=7) != entity_f1_ci(teacher, student, seed=8)


def test_kappa_replicates_match_direct_computation() -> None:
    # The marginal shortcut must equal kappa recomputed on the explicitly resampled inputs.
    from anvil.evaluate.bootstrap import resample_indices

    rng = np.random.default_rng(5)
    texts, teacher, student = random_texts(rng, 12)
    ci = token_kappa_ci(texts, teacher, student, n_resamples=50, seed=3)
    values = []
    for row in resample_indices(12, 50, 3):
        a = [x for i in row for x in word_tags(texts[i], teacher[i])]
        b = [x for i in row for x in word_tags(texts[i], student[i])]
        values.append(cohen_kappa_score(a, b) if len(set(a) | set(b)) > 1 else 1.0)
    low, high = np.percentile(values, [2.5, 97.5])
    assert (ci.low, ci.high) == pytest.approx((low, high))


def sparse_corpus() -> tuple[list[list[Span]], list[list[Span]], list[list[Span]]]:
    """Six inputs, entities only in two, so some resamples draw no entity at all."""
    teacher: list[list[Span]] = [[Span(0, 1, P), Span(2, 3, L), Span(4, 5, P)]] + [[]] * 5
    a: list[list[Span]] = [[Span(0, 1, P), Span(2, 3, P)], [], [], [Span(0, 1, L)], [], []]
    b: list[list[Span]] = [[Span(0, 1, P)], [], [], [], [], []]
    return teacher, a, b


def _oracle_f1(teacher: list[list[Span]], student: list[list[Span]], row: list[int]) -> float:
    tp = fp = fn = 0
    for i in row:
        t, s = set(teacher[i]), set(student[i])
        tp += len(t & s)
        fp += len(s - t)
        fn += len(t - s)
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")


def _corpora() -> list[tuple[list[list[Span]], list[list[Span]], list[list[Span]]]]:
    rng = np.random.default_rng(10)
    teacher, a = random_corpus(rng, 12)
    b = [s[: len(s) // 2] for s in a]
    return [(teacher, a, b), sparse_corpus()]


@pytest.mark.parametrize("corpus", _corpora(), ids=["random", "sparse"])
def test_f1_replicates_match_direct_computation(
    corpus: tuple[list[list[Span]], list[list[Span]], list[list[Span]]],
) -> None:
    # Per-resample micro F1 summed input by input over the same index draws; resamples with
    # no entity are left out of the percentiles.
    teacher, a, b = corpus
    n, r, seed = len(teacher), 200, 3
    rows = resample_indices(n, r, seed).tolist()
    f1 = [_oracle_f1(teacher, a, row) for row in rows]
    diff = [_oracle_f1(teacher, a, row) - _oracle_f1(teacher, b, row) for row in rows]
    q = [2.5, 97.5]
    expected_f1 = np.percentile([v for v in f1 if not np.isnan(v)], q)
    expected_diff = np.percentile([v for v in diff if not np.isnan(v)], q)
    ci = entity_f1_ci(teacher, a, n_resamples=r, seed=seed)
    d = entity_f1_difference_ci(teacher, a, b, n_resamples=r, seed=seed)
    assert (ci.low, ci.high) == pytest.approx(tuple(expected_f1), abs=1e-9)
    assert (d.low, d.high) == pytest.approx(tuple(expected_diff), abs=1e-9)
    if n == 6:
        assert any(np.isnan(v) for v in f1)  # the sparse corpus does exercise dropping


def test_empty_resamples_do_not_score_perfect() -> None:
    # Only the first input has an entity and the student misses it: F1 is 0 on every
    # resample that can be scored. The resamples that miss the first input have nothing to
    # score and must not count as perfect agreement.
    teacher: list[list[Span]] = [[Span(0, 1, P)], [], []]
    student: list[list[Span]] = [[], [], []]
    ci = entity_f1_ci(teacher, student, n_resamples=500, seed=0)
    assert (ci.estimate, ci.low, ci.high) == (0.0, 0.0, 0.0)
    d = entity_f1_difference_ci(teacher, student, student, n_resamples=500, seed=0)
    assert (d.estimate, d.low, d.high) == (0.0, 0.0, 0.0)
    # Same for words: one input has words, kappa there is 0 (estimate), the rest have none.
    texts = ["a b", " ", ""]
    kappa = token_kappa_ci(texts, teacher, student, n_resamples=500, seed=0)
    assert kappa.estimate == pytest.approx(0.0)
    assert (kappa.low, kappa.high) == (0.0, 0.0)
    # The point estimates keep their all-empty convention.
    assert entity_scores([[]], [[]]).f1 == 1.0


def test_ci_without_any_entity_raises() -> None:
    with pytest.raises(ValueError, match="every bootstrap resample is undefined"):
        entity_f1_ci([[], []], [[], []], n_resamples=50)
    with pytest.raises(ValueError, match="every bootstrap resample is undefined"):
        entity_f1_difference_ci([[], []], [[], []], [[], []], n_resamples=50)


def test_resampling_is_by_input() -> None:
    # 20 inputs; one carries all 20 entities and the student gets every one of them, the
    # other inputs carry nothing in the teacher's labels but one spurious student span each.
    # Word- or entity-level resampling would give a narrow interval around 20/39; resampling
    # inputs makes F1 swing with how often the one heavy input is drawn.
    teacher: list[list[Span]] = [[Span(2 * j, 2 * j + 1, P) for j in range(20)]]
    student: list[list[Span]] = [list(teacher[0])]
    teacher += [[] for _ in range(19)]
    student += [[Span(0, 1, P)] for _ in range(19)]
    ci = entity_f1_ci(teacher, student, n_resamples=2000, seed=0)
    assert ci.estimate == pytest.approx(40 / 59)
    assert ci.low == 0.0  # some resamples miss the heavy input entirely
    assert ci.high - ci.low > 0.6


def test_difference_of_identical_students_is_zero() -> None:
    rng = np.random.default_rng(6)
    teacher, student = random_corpus(rng, 25)
    d = entity_f1_difference_ci(teacher, student, student, n_resamples=200, seed=1)
    assert d == Interval(estimate=0.0, low=0.0, high=0.0, confidence=0.95, n_resamples=200)


def test_difference_separates_a_better_student() -> None:
    rng = np.random.default_rng(7)
    teacher, good = random_corpus(rng, 80, p_keep=0.8)
    bad: list[list[Span]] = [s[: len(s) // 2] for s in good]
    d = entity_f1_difference_ci(teacher, good, bad, n_resamples=500, seed=2)
    assert d.estimate == pytest.approx(
        entity_scores(teacher, good).f1 - entity_scores(teacher, bad).f1
    )
    assert 0 < d.low < d.estimate < d.high


def test_bootstrap_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="at least one item"):
        entity_f1_ci([], [])
    with pytest.raises(ValueError, match="confidence"):
        entity_f1_ci([[]], [[]], confidence=1.0)
    with pytest.raises(ValueError, match="lengths differ"):
        entity_f1_difference_ci([[]], [[]], [[], []])


# --- slices and reports -------------------------------------------------------------------


def test_by_length_hand_case() -> None:
    texts = ["a b c d", "a", "a b", "a b c", "a b c d e"]
    hit = [Span(0, 1, P)]
    teacher = [hit] * 5
    student = [hit, [], hit, [Span(0, 1, L)], hit]
    # Sorted by word count: 1 (miss), 2 (hit), 3 (wrong type), 4 (hit), 5 (hit) -> 2,1,1,1.
    assert by_length(texts, teacher, student) == [
        Slice(lo=1, hi=2, n=2, agreement=2 / 3),
        Slice(lo=3, hi=3, n=1, agreement=0.0),
        Slice(lo=4, hi=4, n=1, agreement=1.0),
        Slice(lo=5, hi=5, n=1, agreement=1.0),
    ]
    assert by_length([], [], []) == []


def test_extraction_agreement_report_round_trips_json() -> None:
    rng = np.random.default_rng(8)
    texts, teacher, student = random_texts(rng, 20)
    report = extraction_agreement(texts, teacher, student, [P, L], n_resamples=100, seed=0)
    assert report.n == 20
    assert report.entity == entity_scores(teacher, student)
    assert report.teacher_entities == sum(len(set(t)) for t in teacher)
    assert report.student_entities == sum(len(set(s)) for s in student)
    assert report.entity_f1_ci.estimate == pytest.approx(report.entity.f1)
    assert report.token_kappa == pytest.approx(report.token_kappa_ci.estimate)
    assert [r.type for r in report.per_type] == [P, L]
    assert sum(s.n for s in report.by_length) == 20
    payload = json.loads(json.dumps(asdict(report)))
    assert payload["entity"]["f1"] == report.entity.f1
    assert payload["errors"]["correct"] == report.errors.correct


def test_self_agreement_uses_shared_ids() -> None:
    texts = {"a": "Ada met Bob", "b": "in Rome", "c": "unused"}
    first = {"a": [Span(0, 3, P), Span(8, 11, P)], "b": [Span(3, 7, L)]}
    second = {"a": [Span(0, 3, P)], "b": [Span(3, 7, L)], "z": [Span(0, 1, P)]}
    ceiling = extraction_self_agreement(texts, first, second, n_resamples=50, seed=0)
    assert ceiling.n == 2
    assert ceiling.entity_f1 == pytest.approx(0.8)
    assert ceiling.entity_f1_ci.estimate == pytest.approx(0.8)
    tags_a = ["B-PER", "O", "B-PER", "O", "B-LOC"]
    tags_b = ["B-PER", "O", "O", "O", "B-LOC"]
    assert ceiling.token_kappa == pytest.approx(cohen_kappa_score(tags_a, tags_b))
    with pytest.raises(ValueError, match="share no input ids"):
        extraction_self_agreement(texts, {"a": []}, {"b": []})
    with pytest.raises(ValueError, match="no text"):
        extraction_self_agreement({}, {"a": []}, {"a": []})


# --- entity F1 as a cascade statistic -------------------------------------------------------


def _f1_statistic_corpora() -> list[tuple[list[list[Span]], list[list[Span]]]]:
    rng = np.random.default_rng(21)
    teacher, student = random_corpus(rng, 25)
    sparse_t, sparse_a, _ = sparse_corpus()
    return [(teacher, student), (sparse_t, sparse_a)]


@pytest.mark.parametrize("corpus", _f1_statistic_corpora(), ids=["random", "sparse"])
def test_entity_f1_statistic_prefix_is_entity_scores(
    corpus: tuple[list[list[Span]], list[list[Span]]],
) -> None:
    teacher, student = corpus
    stat = entity_f1_statistic(teacher, student)
    n = len(teacher)
    assert stat.n == n
    prefix = stat.prefix(np.arange(n, dtype=np.int64))
    expected = [entity_scores(teacher[: m + 1], student[: m + 1]).f1 for m in range(n)]
    assert prefix.tolist() == pytest.approx(expected, abs=1e-12)
    order = np.arange(n, dtype=np.int64)[::-1].copy()
    reverse = [
        entity_scores(
            [teacher[i] for i in order[: m + 1]], [student[i] for i in order[: m + 1]]
        ).f1
        for m in range(n)
    ]
    assert stat.prefix(order).tolist() == pytest.approx(reverse, abs=1e-12)


@pytest.mark.parametrize("corpus", _f1_statistic_corpora(), ids=["random", "sparse"])
def test_entity_f1_statistic_replicates_and_jackknife(
    corpus: tuple[list[list[Span]], list[list[Span]]],
) -> None:
    teacher, student = corpus
    stat = entity_f1_statistic(teacher, student)
    n = len(teacher)
    members = np.arange(n, dtype=np.int64)
    idx = resample_indices(n, 100, 4)
    reps = stat.replicates(members, idx)
    expected = np.array([_oracle_f1(teacher, student, row) for row in idx.tolist()])
    np.testing.assert_allclose(reps, expected, atol=1e-12, equal_nan=True)
    # The same draws give entity_f1_ci's interval.
    ci = entity_f1_ci(teacher, student, n_resamples=100, seed=4)
    defined = reps[~np.isnan(reps)]
    assert (ci.low, ci.high) == pytest.approx(tuple(np.percentile(defined, [2.5, 97.5])))
    # A subset of members, resampled with their own draws.
    sub = members[::2]
    sub_idx = resample_indices(sub.size, 30, 1)
    got = stat.replicates(sub, sub_idx)
    want = [_oracle_f1(teacher, student, [int(sub[j]) for j in row]) for row in sub_idx]
    np.testing.assert_allclose(got, np.array(want), atol=1e-12, equal_nan=True)

    jack = stat.jackknife(members)
    loo = [_oracle_f1(teacher, student, [j for j in range(n) if j != i]) for i in range(n)]
    np.testing.assert_allclose(jack, np.array(loo), atol=1e-12, equal_nan=True)


def test_entity_f1_statistic_conventions_when_nothing_to_score() -> None:
    # Point estimate 1.0 on an all-empty prefix (entity_scores), NaN for an all-empty
    # resample or leave-one-out set (entity_f1_ci).
    teacher: list[list[Span]] = [[], [Span(0, 1, P)], []]
    student: list[list[Span]] = [[], [], []]
    stat = entity_f1_statistic(teacher, student)
    assert stat.prefix(np.array([0, 2, 1], dtype=np.int64)).tolist() == [1.0, 1.0, 0.0]
    members = np.array([0, 2], dtype=np.int64)
    assert np.isnan(stat.replicates(members, resample_indices(2, 10, 0))).all()
    assert np.isnan(stat.jackknife(members)).all()
    jack = stat.jackknife(np.arange(3, dtype=np.int64))
    assert jack[0] == 0.0
    assert np.isnan(jack[1])
    assert jack[2] == 0.0


def test_entity_f1_statistic_dedups_and_checks_lengths() -> None:
    stat = entity_f1_statistic([[Span(0, 2, P), Span(0, 2, P)]], [[Span(0, 2, P)] * 3])
    assert stat.counts.tolist() == [[1, 0, 0]]
    with pytest.raises(ValueError, match="lengths differ"):
        entity_f1_statistic([[]], [])
