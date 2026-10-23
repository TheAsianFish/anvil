"""The generic threshold scan over a `PooledStatistic`: kappa through it reproduces the
classification cascade's old `choose_threshold` exactly, and counts statistics (entity F1)
agree with brute-force prefixes, resamples, jackknives and scans."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from anvil.calibrate import (
    BOUND_METHOD,
    STATISTIC_RULES,
    CountStatistic,
    KappaStatistic,
    Rule,
    StatisticChoice,
    ThresholdChoice,
    choose_statistic_threshold,
    choose_threshold,
    coverage_curve,
    statistic_curve,
)
from anvil.evaluate.bootstrap import bca_levels, lower_bounds, resample_indices
from anvil.evaluate.extraction import entity_f1_statistic, entity_scores
from anvil.spans import Span

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BOOTSTRAP_RULES: tuple[Rule, ...] = ("point", "lower", "bca")


# --- the reference: choose_threshold as it was before the generic scan ---------------------


def _old_choose_threshold(
    confidence: npt.ArrayLike,
    teacher: npt.ArrayLike,
    student: npt.ArrayLike,
    n_labels: int,
    target_kappa: float,
    *,
    rule: Rule,
    min_accepted: int,
    confidence_level: float,
    n_resamples: int,
    seed: int,
) -> ThresholdChoice | None:
    """The bootstrap and point branch of `choose_threshold` from main, verbatim in logic."""
    conf = np.asarray(confidence, dtype=np.float64)
    t = np.asarray(teacher).astype(np.int64)
    s = np.asarray(student).astype(np.int64)
    order = np.argsort(-conf, kind="stable").astype(np.int64)
    curve = coverage_curve(confidence, teacher, student, n_labels)
    t_sorted, s_sorted = t[order], s[order]
    for point in reversed(curve):
        if point.accepted < min_accepted or point.kappa < target_kappa:
            continue
        low: float | None = None
        if rule != "point":
            k = point.accepted
            method = BOUND_METHOD[rule]
            idx = resample_indices(k, n_resamples, seed)
            low = lower_bounds(
                t_sorted[:k], s_sorted[:k], n_labels, idx, confidence_level, (method,)
            )[method]
            if low < target_kappa:
                continue
        return ThresholdChoice(
            rule=rule,
            target_kappa=target_kappa,
            threshold=point.threshold,
            accepted=point.accepted,
            coverage=point.coverage,
            kappa=point.kappa,
            kappa_low=low,
            system_kappa=point.system_kappa,
        )
    return None


def _classify_case(
    seed: int, n: int, k: int, decimals: int
) -> tuple[FloatArray, IntArray, IntArray]:
    """Confidences rounded to `decimals` (heavy ties when small); the student agrees with the
    teacher more often the more confident it is."""
    rng = np.random.default_rng(seed)
    conf = np.round(rng.uniform(0.2, 1.0, n), decimals)
    teacher = rng.integers(0, k, n).astype(np.int64)
    wrong = rng.random(n) > conf
    student = np.where(wrong, rng.integers(0, k, n), teacher).astype(np.int64)
    return conf, teacher, student


CASES = list(
    itertools.product(
        (40, 120, 300),  # n
        (2, 4, 77),  # labels
        (1, 2),  # confidence decimals: 1 gives at most 9 distinct values
    )
)


@pytest.mark.parametrize(("n", "k", "decimals"), CASES)
def test_choose_threshold_matches_the_old_implementation(n: int, k: int, decimals: int) -> None:
    conf, teacher, student = _classify_case(1000 * n + 10 * k + decimals, n, k, decimals)
    found = 0
    for target, min_accepted, rule in itertools.product(
        (0.5, 0.7, 0.85, 0.95), (2, 10, 30), BOOTSTRAP_RULES
    ):
        level = 0.9 if min_accepted == 10 else 0.95
        seed = n + min_accepted
        new = choose_threshold(
            conf,
            teacher,
            student,
            k,
            target,
            rule=rule,
            min_accepted=min_accepted,
            confidence_level=level,
            n_resamples=200,
            seed=seed,
        )
        old = _old_choose_threshold(
            conf,
            teacher,
            student,
            k,
            target,
            rule=rule,
            min_accepted=min_accepted,
            confidence_level=level,
            n_resamples=200,
            seed=seed,
        )
        assert new == old, (target, min_accepted, rule)
        if new is not None and rule != "point":
            assert new.kappa_low is not None
            found += 1
    assert found > 0


def test_choose_threshold_matches_when_nothing_qualifies() -> None:
    conf, teacher, student = _classify_case(7, 60, 4, 2)
    for rule in BOOTSTRAP_RULES:
        # Too many accepted inputs required, and a target no kappa reaches.
        for target, min_accepted in ((0.5, 61), (1.01, 2)):
            new = choose_threshold(
                conf,
                teacher,
                student,
                4,
                target,
                rule=rule,
                min_accepted=min_accepted,
                n_resamples=100,
            )
            old = _old_choose_threshold(
                conf,
                teacher,
                student,
                4,
                target,
                rule=rule,
                min_accepted=min_accepted,
                confidence_level=0.95,
                n_resamples=100,
                seed=0,
            )
            assert new is None
            assert old is None


@pytest.mark.parametrize(("n", "k", "decimals"), [(40, 2, 1), (120, 4, 2), (300, 77, 2)])
@pytest.mark.parametrize("rule", BOOTSTRAP_RULES)
def test_statistic_chooser_on_kappa_is_choose_threshold(
    n: int, k: int, decimals: int, rule: Rule
) -> None:
    conf, teacher, student = _classify_case(n + k, n, k, decimals)
    stat = KappaStatistic(teacher, student, k)
    for target in (0.5, 0.75, 0.9):
        a = choose_threshold(
            conf,
            teacher,
            student,
            k,
            target,
            rule=rule,
            min_accepted=20,
            n_resamples=200,
            seed=3,
        )
        b = choose_statistic_threshold(
            conf, stat, target, rule=rule, min_accepted=20, n_resamples=200, seed=3
        )
        if a is None:
            assert b is None
            continue
        assert b is not None
        assert (b.rule, b.target, b.threshold, b.accepted, b.coverage) == (
            a.rule,
            a.target_kappa,
            a.threshold,
            a.accepted,
            a.coverage,
        )
        assert b.value == a.kappa
        assert b.low == a.kappa_low


@pytest.mark.parametrize(("n", "k", "decimals"), [(40, 2, 1), (120, 4, 2), (300, 77, 3)])
def test_kappa_statistic_prefix_and_curve_match_coverage_curve(
    n: int, k: int, decimals: int
) -> None:
    conf, teacher, student = _classify_case(5 * n + k, n, k, decimals)
    stat = KappaStatistic(teacher, student, k)
    assert stat.n == n
    curve = coverage_curve(conf, teacher, student, k)
    order = np.argsort(-conf, kind="stable").astype(np.int64)
    ends = np.array([p.accepted - 1 for p in curve], dtype=np.int64)
    assert stat.prefix(order)[ends].tolist() == [p.kappa for p in curve]
    points = statistic_curve(conf, stat)
    assert [(p.threshold, p.accepted, p.coverage, p.value) for p in points] == [
        (p.threshold, p.accepted, p.coverage, p.kappa) for p in curve
    ]


def test_kappa_statistic_validates() -> None:
    with pytest.raises(ValueError, match="n_labels"):
        KappaStatistic([0], [0], 0)
    with pytest.raises(ValueError, match="1-d"):
        KappaStatistic([], [], 2)
    with pytest.raises(ValueError, match="student must have shape"):
        KappaStatistic([0, 1], [0], 2)
    with pytest.raises(ValueError, match="lie in"):
        KappaStatistic([0, 2], [0, 1], 2)


# --- CountStatistic -----------------------------------------------------------------------


def _sum_value(c: IntArray) -> FloatArray:
    """A toy statistic: share of the first count in the total, NaN for an empty total."""
    total = c.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out: FloatArray = np.where(total > 0, c[..., 0] / np.maximum(total, 1), np.nan)
    return out


def _counts(seed: int, n: int, d: int) -> IntArray:
    rng = np.random.default_rng(seed)
    out: IntArray = rng.integers(0, 4, size=(n, d)).astype(np.int64)
    out[rng.random(n) < 0.3] = 0  # rows with nothing to score
    return out


@pytest.mark.parametrize(("n", "d"), [(1, 1), (7, 2), (50, 3)])
def test_count_statistic_against_brute_force(n: int, d: int) -> None:
    counts = _counts(n * d, n, d)
    stat = CountStatistic(counts, _sum_value, _sum_value)
    assert stat.n == n
    rng = np.random.default_rng(n)
    order = rng.permutation(n).astype(np.int64)
    prefix = stat.prefix(order)
    expected = [_sum_value(counts[order[: m + 1]].sum(axis=0)[None, :])[0] for m in range(n)]
    np.testing.assert_array_equal(prefix, np.array(expected))

    members = rng.choice(n, size=max(1, n // 2 + 1), replace=False).astype(np.int64)
    m = int(members.size)
    idx = resample_indices(m, 40, seed=11)
    reps = stat.replicates(members, idx)
    brute = [_sum_value(counts[members][row].sum(axis=0)[None, :])[0] for row in idx]
    np.testing.assert_array_equal(reps, np.array(brute))

    jack = stat.jackknife(members)
    brute_jack = [
        _sum_value(counts[np.delete(members, i)].sum(axis=0)[None, :])[0] for i in range(m)
    ]
    np.testing.assert_array_equal(jack, np.array(brute_jack))


def test_count_statistic_keeps_its_two_conventions_apart() -> None:
    # value says 1.0 on an empty total, replicate_value NaN, as for entity F1.
    def point(c: IntArray) -> FloatArray:
        out: FloatArray = np.where(c.sum(axis=-1) > 0, 0.5, 1.0)
        return out

    stat = CountStatistic(np.zeros((3, 2), dtype=np.int64), point, _sum_value)
    assert stat.prefix(np.arange(3, dtype=np.int64)).tolist() == [1.0, 1.0, 1.0]
    members = np.arange(3, dtype=np.int64)
    assert np.isnan(stat.replicates(members, resample_indices(3, 5, 0))).all()
    assert np.isnan(stat.jackknife(members)).all()


@pytest.mark.parametrize(
    ("counts", "match"),
    [
        (np.ones((3, 2), dtype=np.float64), "non-negative integers"),
        (np.array([[1, -1], [0, 0]]), "non-negative integers"),
        (np.ones(3, dtype=np.int64), "shape"),
        (np.ones((2, 2, 2), dtype=np.int64), "shape"),
        (np.ones((0, 3), dtype=np.int64), "shape"),
        (np.ones((3, 0), dtype=np.int64), "shape"),
    ],
)
def test_count_statistic_validates(counts: IntArray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CountStatistic(counts, _sum_value, _sum_value)


# --- the scan on entity F1 against a naive one ----------------------------------------------

P, L = "PER", "LOC"


def _span_case(seed: int, n: int) -> tuple[list[list[Span]], list[list[Span]], FloatArray]:
    """Sentences on a grid of words; the more confident the student, the more it keeps the
    teacher's spans. Confidences rounded to 2 decimals, so ties occur."""
    rng = np.random.default_rng(seed)
    conf = np.round(rng.uniform(0.3, 1.0, n), 2)
    teacher: list[list[Span]] = []
    student: list[list[Span]] = []
    for c in conf:
        t: list[Span] = []
        s: list[Span] = []
        for w in range(int(rng.integers(0, 5))):
            if rng.random() < 0.6:
                span = Span(4 * w, 4 * w + 3, P if rng.random() < 0.5 else L)
                t.append(span)
                u = rng.random()
                if u < c:
                    s.append(span)
                elif u < c + (1 - c) / 2:
                    s.append(Span(span.start, span.end + 4, span.type))
            elif rng.random() < 0.05:
                s.append(Span(4 * w, 4 * w + 3, P))
        teacher.append(t)
        student.append(s)
    return teacher, student, conf


def _f1_or_nan(teacher: list[list[Span]], student: list[list[Span]], rows: list[int]) -> float:
    tp = fp = fn = 0
    for i in rows:
        t, s = set(teacher[i]), set(student[i])
        tp, fp, fn = tp + len(t & s), fp + len(s - t), fn + len(t - s)
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else math.nan


def _naive_choice(
    teacher: list[list[Span]],
    student: list[list[Span]],
    conf: FloatArray,
    target: float,
    rule: Rule,
    min_accepted: int,
    confidence_level: float,
    n_resamples: int,
    seed: int,
) -> tuple[float, int, float, float | None] | None:
    """(threshold, accepted, value, low) the slow way."""
    n = len(conf)
    order = sorted(range(n), key=lambda i: -conf[i])  # sorted() is stable
    cuts = [m for m in range(1, n + 1) if m == n or conf[order[m]] != conf[order[m - 1]]]
    for accepted in reversed(cuts):
        members = order[:accepted]
        value = entity_scores([teacher[i] for i in members], [student[i] for i in members]).f1
        if accepted < min_accepted or value < target:
            continue
        threshold = float(conf[order[accepted - 1]])
        if rule == "point":
            return threshold, accepted, value, None
        idx = resample_indices(accepted, n_resamples, seed)
        reps = np.array([_f1_or_nan(teacher, student, [members[j] for j in row]) for row in idx])
        defined = reps[~np.isnan(reps)]
        if not defined.size:
            continue
        if rule == "lower":
            low = float(np.percentile(defined, 100 * (1 - confidence_level)))
        else:
            estimate = _f1_or_nan(teacher, student, members)
            jack = np.array(
                [
                    _f1_or_nan(teacher, student, members[:j] + members[j + 1 :])
                    for j in range(accepted)
                ]
            )
            if math.isnan(estimate) or np.isnan(jack).any():
                continue
            level = bca_levels(defined, estimate, jack, [1 - confidence_level])[0]
            low = float(np.percentile(defined, 100 * level))
        if low >= target:
            return threshold, accepted, value, low
    return None


@pytest.mark.parametrize(("seed", "n"), [(0, 45), (1, 90), (2, 150)])
@pytest.mark.parametrize("rule", BOOTSTRAP_RULES)
def test_statistic_chooser_on_entity_f1_matches_a_naive_scan(
    seed: int, n: int, rule: Rule
) -> None:
    teacher, student, conf = _span_case(seed, n)
    stat = entity_f1_statistic(teacher, student)
    found = 0
    for target, min_accepted, level in ((0.6, 10, 0.95), (0.8, 15, 0.9), (0.9, 5, 0.95)):
        got = choose_statistic_threshold(
            conf,
            stat,
            target,
            rule=rule,
            min_accepted=min_accepted,
            confidence_level=level,
            n_resamples=120,
            seed=seed,
        )
        want = _naive_choice(teacher, student, conf, target, rule, min_accepted, level, 120, seed)
        if want is None:
            assert got is None
            continue
        found += 1
        assert got is not None
        assert isinstance(got, StatisticChoice)
        threshold, accepted, value, low = want
        assert (got.rule, got.target, got.threshold, got.accepted) == (
            rule,
            target,
            threshold,
            accepted,
        )
        assert got.coverage == accepted / n
        assert got.value == pytest.approx(value, abs=1e-12)
        if low is None:
            assert got.low is None
        else:
            assert got.low == pytest.approx(low, abs=1e-12)
    if rule == "point":
        assert found > 0


def test_statistic_curve_on_entity_f1_matches_entity_scores() -> None:
    teacher, student, conf = _span_case(4, 60)
    points = statistic_curve(conf, entity_f1_statistic(teacher, student))
    assert points[-1].coverage == 1.0
    for p in points:
        keep = [i for i in range(60) if conf[i] >= p.threshold]
        assert len(keep) == p.accepted
        f1 = entity_scores([teacher[i] for i in keep], [student[i] for i in keep]).f1
        assert p.value == pytest.approx(f1, abs=1e-12)


def _kappa_inputs() -> tuple[FloatArray, KappaStatistic]:
    conf, teacher, student = _classify_case(3, 50, 3, 2)
    return conf, KappaStatistic(teacher, student, 3)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda c, s: choose_statistic_threshold(c, s, 0.5, rule="exact"), "unknown rule"),
        (lambda c, s: choose_statistic_threshold(c, s, 0.5, rule="wilson"), "unknown rule"),
        (lambda c, s: choose_statistic_threshold(c, s, 0.5, rule="nope"), "unknown rule"),  # type: ignore[arg-type]
        (lambda c, s: choose_statistic_threshold(c, s, math.nan, rule="point"), "finite"),
        (lambda c, s: choose_statistic_threshold(c, s, math.inf, rule="lower"), "finite"),
        (
            lambda c, s: choose_statistic_threshold(c, s, 0.5, rule="point", min_accepted=0),
            "min_accepted",
        ),
        (
            lambda c, s: choose_statistic_threshold(c, s, 0.5, rule="bca", min_accepted=1),
            "min_accepted >= 2",
        ),
        (
            lambda c, s: choose_statistic_threshold(c, s, 0.5, rule="lower", confidence_level=1.0),
            "confidence_level",
        ),
        (
            lambda c, s: choose_statistic_threshold(c, s, 0.5, rule="lower", confidence_level=0.0),
            "confidence_level",
        ),
        (
            lambda c, s: choose_statistic_threshold(c[:-1], s, 0.5, rule="point"),
            "one confidence per input",
        ),
        (
            lambda c, s: choose_statistic_threshold(
                np.where(np.arange(c.size) == 3, np.nan, c), s, 0.5, rule="point"
            ),
            "NaN",
        ),
        (lambda c, s: statistic_curve(c[:-1], s), "one confidence per input"),
        (lambda c, s: statistic_curve(np.empty(0), s), "at least one item"),
    ],
)
def test_statistic_chooser_rejects_bad_arguments(
    call: Callable[[FloatArray, KappaStatistic], object], match: str
) -> None:
    conf, stat = _kappa_inputs()
    with pytest.raises(ValueError, match=match):
        call(conf, stat)


def test_statistic_rules_exclude_the_count_rules_which_choose_threshold_keeps() -> None:
    assert STATISTIC_RULES == ("point", "lower", "bca")
    conf, teacher, student = _classify_case(9, 200, 2, 2)
    for rule in ("exact", "wilson"):
        choice = choose_threshold(conf, teacher, student, 2, 0.3, rule=rule, min_accepted=20)
        assert choice is not None
        assert choice.kappa_low is not None
        assert choice.kappa_low <= choice.kappa


def test_a_bound_that_cannot_be_computed_never_clears() -> None:
    # No sentence has an entity: F1 is 1.0 by convention as a point estimate, but every
    # resample and every jackknife value has nothing to score. Only "point" accepts.
    n = 40
    stat = entity_f1_statistic([[] for _ in range(n)], [[] for _ in range(n)])
    conf = np.linspace(0.1, 1.0, n)
    point = choose_statistic_threshold(conf, stat, 0.9, rule="point", n_resamples=50)
    assert point is not None
    assert (point.accepted, point.coverage, point.value, point.low) == (n, 1.0, 1.0, None)
    for rule in ("lower", "bca"):
        assert choose_statistic_threshold(conf, stat, 0.9, rule=rule, n_resamples=50) is None


def test_bca_skips_a_candidate_whose_jackknife_has_an_empty_set() -> None:
    # Only one of the accepted sentences has an entity, so leaving it out leaves nothing to
    # score: the BCa bound is undefined there and the cut cannot clear, while "lower" (every
    # defined resample perfect) does.
    n = 30
    teacher: list[list[Span]] = [[Span(0, 3, P)]] + [[] for _ in range(n - 1)]
    student = [list(t) for t in teacher]
    stat = entity_f1_statistic(teacher, student)
    conf = np.linspace(1.0, 0.5, n)
    lower = choose_statistic_threshold(
        conf, stat, 0.9, rule="lower", min_accepted=10, n_resamples=100
    )
    assert lower is not None
    assert (lower.coverage, lower.low) == (1.0, 1.0)
    bca = choose_statistic_threshold(conf, stat, 0.9, rule="bca", min_accepted=10, n_resamples=100)
    assert bca is None


def test_statistic_chooser_is_deterministic_in_the_seed() -> None:
    teacher, student, conf = _span_case(5, 120)
    stat = entity_f1_statistic(teacher, student)
    a = choose_statistic_threshold(conf, stat, 0.7, rule="lower", n_resamples=100, seed=1)
    b = choose_statistic_threshold(conf, stat, 0.7, rule="lower", n_resamples=100, seed=1)
    assert a == b
