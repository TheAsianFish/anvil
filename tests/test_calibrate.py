from __future__ import annotations

import itertools
import json
import math
from statistics import NormalDist

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]

from anvil.calibrate import (
    DEFAULT_RULE,
    CoveragePoint,
    calibrate,
    choose_threshold,
    coverage_curve,
    ece,
    fit_temperature,
    nll,
    reliability,
    serving_temperature,
    softmax,
)
from anvil.evaluate.bootstrap import kappa_replicates, lower_bounds, resample_indices
from anvil.evaluate.metrics import cohen_kappa

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def sampled(seed: int, n: int, k: int, scale: float) -> tuple[FloatArray, IntArray]:
    """Logits whose softmax is the true label distribution, multiplied by `scale`, and labels
    drawn from that distribution: the fitted temperature should come out near `scale`."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.5, size=(n, k))
    p = softmax(w)
    u = rng.random(n)[:, None]
    y = np.minimum((p.cumsum(axis=1) < u).sum(axis=1), k - 1).astype(np.int64)
    return scale * w, y


def codes(*values: int) -> IntArray:
    return np.array(values, dtype=np.int64)


# softmax and nll


def test_softmax_rows_sum_to_one_and_shift_invariant() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(0, 5, size=(20, 6))
    p = softmax(z)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(softmax(z + rng.normal(0, 100, size=(20, 1))), p)
    assert softmax(np.array([[1000.0, 0.0]]))[0, 0] == pytest.approx(1.0)


def test_softmax_temperature_semantics() -> None:
    z = np.array([[2.0, 1.0, 0.0]])
    assert np.allclose(softmax(z, 2.0), softmax(z / 2.0))
    assert softmax(z, 0.5)[0, 0] > softmax(z)[0, 0] > softmax(z, 4.0)[0, 0]
    assert np.allclose(softmax(z, 1e6), 1 / 3, atol=1e-5)
    with pytest.raises(ValueError, match="temperature"):
        softmax(z, 0.0)


def test_nll_matches_log_softmax() -> None:
    z, y = sampled(1, 50, 4, 1.0)
    expected = -np.mean(np.log(softmax(z, 1.7)[np.arange(50), y]))
    assert nll(z, y, 1.7) == pytest.approx(expected)


# fit_temperature


def test_fit_temperature_recovers_overconfidence_and_matches_scipy() -> None:
    z, y = sampled(2, 4000, 5, 3.0)
    t = fit_temperature(z, y)
    assert t == pytest.approx(3.0, rel=0.15)
    ref = minimize_scalar(
        lambda temp: nll(z, y, temp),
        bounds=(0.05, 20.0),
        method="bounded",
        options={"xatol": 1e-8},
    )
    assert t == pytest.approx(ref.x, rel=1e-4)
    best = nll(z, y, t)
    assert best <= nll(z, y, 1.0)
    assert best <= nll(z, y, t * 1.05)
    assert best <= nll(z, y, t * 0.95)


def test_fit_temperature_underconfident() -> None:
    z, y = sampled(3, 3000, 3, 0.5)
    assert fit_temperature(z, y) == pytest.approx(0.5, rel=0.15)


def test_fit_temperature_clamps() -> None:
    separable = np.array([[3.0, 0.0], [0.0, 3.0], [2.0, -1.0]])
    assert fit_temperature(separable, codes(0, 1, 0)) == 0.05
    assert fit_temperature(separable, codes(0, 1, 0), bounds=(0.5, 2.0)) == 0.5
    # Always wrong: the flattest distribution is best.
    assert fit_temperature(separable, codes(1, 0, 1)) == 20.0
    # Constant rows: T has no effect.
    assert fit_temperature(np.zeros((3, 2)), codes(0, 1, 0)) == 1.0


@pytest.mark.parametrize(
    ("logits", "targets", "message"),
    [
        (np.zeros(3), codes(0, 1, 0), "shape"),
        (np.zeros((0, 2)), np.zeros(0, dtype=np.int64), "shape"),
        (np.zeros((3, 2)), codes(0, 1), "shape"),
        (np.zeros((3, 2)), codes(0, 2, 0), r"\[0, 2\)"),
        (np.zeros((3, 2)), codes(0, -1, 0), r"\[0, 2\)"),
        (np.array([[0.0, math.nan], [0.0, 1.0]]), codes(0, 1), "NaN"),
        (np.zeros((2, 2)), np.array([0.0, 1.0]), "integer"),
    ],
)
def test_fit_temperature_rejects_bad_inputs(
    logits: FloatArray, targets: npt.NDArray[np.generic], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_temperature(logits, targets)


def test_fit_temperature_rejects_bad_bounds() -> None:
    with pytest.raises(ValueError, match="bounds"):
        fit_temperature(np.zeros((2, 2)), codes(0, 1), bounds=(2.0, 1.0))


# reliability and ece


def test_ece_hand_computed() -> None:
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    targets = codes(0, 1, 1, 0)
    # Two bins: (0, .5] is empty; (.5, 1] holds confidences .9 .8 .7 .6 (mean .75), 3/4 right.
    assert ece(probs, targets, n_bins=2) == pytest.approx(0.0)
    # Five bins: (.5,.6] {.6 right}, (.6,.8] {.7 right, .8 wrong}, (.8,1] {.9 right}.
    expected = 0.25 * 0.4 + 0.5 * abs(0.5 - 0.75) + 0.25 * 0.1
    assert ece(probs, targets, n_bins=5) == pytest.approx(expected)


def test_ece_small_on_calibrated_data() -> None:
    z, y = sampled(4, 20000, 4, 1.0)
    assert ece(softmax(z), y) < 0.02
    assert ece(softmax(z * 3), y) > 0.1


def test_reliability_bins_and_boundaries() -> None:
    probs = np.array([[1.0, 0.0], [0.5, 0.5], [0.2, 0.8], [0.6, 0.4]])
    bins = reliability(probs, codes(0, 0, 0, 0), n_bins=2)
    assert sum(b.n for b in bins) == 4
    assert [(b.low, b.high, b.n) for b in bins] == [(0.0, 0.5, 1), (0.5, 1.0, 3)]
    assert bins[0].confidence == 0.5  # exactly on an edge: the bin below
    assert bins[1].accuracy == pytest.approx(2 / 3)
    many = reliability(probs, codes(0, 0, 0, 0), n_bins=5)
    assert [b.n for b in many] == [0, 0, 2, 1, 1]  # .5 and .6 in (.4, .6], .8 in (.6, .8]
    assert many[0].confidence == 0.0
    assert many[0].accuracy == 0.0
    zero = reliability(np.array([[0.0, 0.0]]), codes(0), n_bins=3)
    assert zero[0].n == 1


def test_reliability_counts_sum_to_n() -> None:
    z, y = sampled(5, 777, 6, 2.0)
    assert sum(b.n for b in reliability(softmax(z), y)) == 777


# coverage_curve


def tied_case(seed: int, n: int, k: int) -> tuple[FloatArray, IntArray, IntArray]:
    rng = np.random.default_rng(seed)
    confidence = rng.integers(0, n // 3, size=n) / (n // 3)  # many ties
    teacher = rng.integers(0, k, size=n).astype(np.int64)
    flip = rng.random(n) < 0.6 * (1 - confidence)
    student = np.where(flip, rng.integers(0, k, size=n), teacher).astype(np.int64)
    return confidence, teacher, student


def brute(
    point: CoveragePoint, confidence: FloatArray, teacher: IntArray, student: IntArray
) -> tuple[int, float, float, float]:
    keep = confidence >= point.threshold
    t = [str(x) for x in teacher]
    s = [str(x) for x in student]
    acc_t = [x for x, m in zip(t, keep, strict=True) if m]
    acc_s = [x for x, m in zip(s, keep, strict=True) if m]
    cascade = [si if m else ti for ti, si, m in zip(t, s, keep, strict=True)]
    agreement = sum(a == b for a, b in zip(acc_t, acc_s, strict=True)) / len(acc_t)
    return int(keep.sum()), cohen_kappa(acc_t, acc_s), agreement, cohen_kappa(t, cascade)


def test_coverage_curve_matches_brute_force_with_ties() -> None:
    confidence, teacher, student = tied_case(6, 300, 4)
    curve = coverage_curve(confidence, teacher, student, 4)
    assert len(curve) == len(np.unique(confidence))
    assert [p.threshold for p in curve] == sorted(np.unique(confidence).tolist(), reverse=True)
    assert curve[-1].coverage == 1.0
    assert curve[-1].kappa == pytest.approx(
        cohen_kappa([str(x) for x in teacher], [str(x) for x in student])
    )
    assert curve[-1].system_kappa == pytest.approx(curve[-1].kappa)
    for point in curve:
        accepted, kappa, agreement, system = brute(point, confidence, teacher, student)
        assert point.accepted == accepted  # ties are never split
        assert point.coverage == accepted / 300
        assert point.kappa == pytest.approx(kappa, abs=1e-12)
        assert point.agreement == pytest.approx(agreement)
        assert point.system_kappa == pytest.approx(system, abs=1e-12)


def test_coverage_curve_degenerate_and_validation() -> None:
    curve = coverage_curve([0.9, 0.9], codes(1, 1), codes(1, 1), 2)
    assert len(curve) == 1
    assert curve[0].kappa == 1.0
    assert curve[0].system_kappa == 1.0
    assert coverage_curve([0.9, 0.5], codes(1, 1), codes(1, 0), 2)[0].kappa == 1.0
    with pytest.raises(ValueError, match="shape"):
        coverage_curve([0.9, 0.5], codes(1), codes(1, 0), 2)
    with pytest.raises(ValueError, match="NaN"):
        coverage_curve([math.nan, 0.5], codes(1, 1), codes(1, 0), 2)
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        coverage_curve([0.9, 0.5], codes(1, 2), codes(1, 0), 2)


@given(
    st.integers(1, 25).flatmap(
        lambda n: st.tuples(
            st.lists(st.integers(0, 4), min_size=n, max_size=n),
            st.lists(st.integers(0, 2), min_size=n, max_size=n),
            st.lists(st.integers(0, 2), min_size=n, max_size=n),
        )
    )
)
def test_coverage_curve_kappa_property(data: tuple[list[int], list[int], list[int]]) -> None:
    conf_levels, t_list, s_list = data
    confidence = np.array(conf_levels, dtype=np.float64) / 4
    teacher, student = np.array(t_list, dtype=np.int64), np.array(s_list, dtype=np.int64)
    curve = coverage_curve(confidence, teacher, student, 3)
    assert curve[-1].accepted == len(t_list)
    assert all(a.accepted < b.accepted for a, b in itertools.pairwise(curve))
    for point in curve:
        accepted, kappa, _, system = brute(point, confidence, teacher, student)
        assert point.accepted == accepted
        assert point.kappa == pytest.approx(kappa, abs=1e-12)
        assert point.system_kappa == pytest.approx(system, abs=1e-12)


# choose_threshold


def dip_case() -> tuple[FloatArray, IntArray, IntArray]:
    """Kappa on the accepted prefix is 1 for 40 items, dips over 15 errors, then recovers."""
    n = 200
    confidence = np.linspace(1.0, 0.01, n)
    teacher = (np.arange(n) % 3).astype(np.int64)
    student = teacher.copy()
    student[40:55] = (teacher[40:55] + 1) % 3
    return confidence, teacher, student


def oracle(curve: list[CoveragePoint], target: float, min_accepted: int) -> CoveragePoint | None:
    ok = [p for p in curve if p.accepted >= min_accepted and p.kappa >= target]
    return max(ok, key=lambda p: p.coverage) if ok else None


def test_point_rule_matches_oracle_on_non_monotone_curve() -> None:
    confidence, teacher, student = dip_case()
    curve = coverage_curve(confidence, teacher, student, 3)
    kappas = [p.kappa for p in curve]
    dip = int(np.argmin(kappas))
    assert kappas[dip] < kappas[-1] < kappas[0]  # non-monotone
    for target in (0.7, 0.8, kappas[-1], kappas[-1] + 1e-9, 0.95, 1.0):
        best = oracle(curve, target, 30)
        got = choose_threshold(confidence, teacher, student, 3, target, rule="point")
        assert (got is None) == (best is None)
        if got is not None and best is not None:
            assert (got.threshold, got.accepted, got.kappa) == (
                best.threshold,
                best.accepted,
                best.kappa,
            )
            assert got.kappa_low is None
            assert got.system_kappa == best.system_kappa


def test_lower_rule_never_exceeds_point_rule() -> None:
    for seed in range(4):
        confidence, teacher, student = tied_case(seed, 400, 3)
        for target in (0.5, 0.7, 0.85):
            point = choose_threshold(confidence, teacher, student, 3, target, rule="point")
            lower = choose_threshold(
                confidence, teacher, student, 3, target, n_resamples=300, seed=seed
            )
            if lower is not None:
                assert point is not None
                assert lower.coverage <= point.coverage
                assert lower.kappa_low is not None
                assert lower.kappa_low >= target
                assert lower.kappa >= target


def test_lower_bound_is_the_fifth_percentile_of_the_bootstrap_on_the_accepted_inputs() -> None:
    confidence, teacher, student = tied_case(11, 400, 3)
    choice = choose_threshold(
        confidence, teacher, student, 3, 0.7, rule="lower", n_resamples=300, seed=5
    )
    assert choice is not None
    keep = confidence >= choice.threshold
    assert int(keep.sum()) == choice.accepted
    order = np.argsort(-confidence, kind="stable")  # the accepted prefix, as the search sees it
    t_acc, s_acc = teacher[order][: choice.accepted], student[order][: choice.accepted]
    idx = resample_indices(choice.accepted, 300, 5)
    replicates = kappa_replicates(t_acc, s_acc, 3, idx)
    assert choice.kappa_low == float(np.percentile(replicates, 5))
    assert choice.kappa_low < choice.kappa < float(np.percentile(replicates, 95))


def test_bca_rule_takes_the_first_point_whose_bca_bound_clears_the_target() -> None:
    # Oracle: walk the curve from the largest coverage down and read the BCa bound of every
    # eligible point directly; the first that clears the target is the answer.
    for seed in range(3):
        confidence, teacher, student = tied_case(20 + seed, 300, 4)
        order = np.argsort(-confidence, kind="stable")
        t_sorted, s_sorted = teacher[order], student[order]
        for target in (0.5, 0.7):
            expected = None
            for point in reversed(coverage_curve(confidence, teacher, student, 4)):
                if point.accepted < 30 or point.kappa < target:
                    continue
                k = point.accepted
                idx = resample_indices(k, 200, seed)
                low = lower_bounds(t_sorted[:k], s_sorted[:k], 4, idx, 0.95, ("bca",))["bca"]
                if low >= target:
                    expected = (point.threshold, low)
                    break
            choice = choose_threshold(
                confidence, teacher, student, 4, target, rule="bca", n_resamples=200, seed=seed
            )
            if expected is None:
                assert choice is None
            else:
                assert choice is not None
                assert (choice.threshold, choice.kappa_low) == expected
                assert choice.rule == "bca"


def test_bca_rule_needs_two_answered_inputs() -> None:
    with pytest.raises(ValueError, match="min_accepted >= 2"):
        choose_threshold([0.9, 0.5], [0, 1], [0, 0], 2, 0.5, rule="bca", min_accepted=1)
    assert choose_threshold([0.9, 0.5], [0, 1], [0, 0], 2, 0.5, rule="lower", min_accepted=1)


def test_lower_rule_gives_up_coverage_the_point_rule_keeps() -> None:
    # The full curve's kappa clears the target by a hair: the point rule takes all 200 inputs,
    # the lower bound of 200 inputs does not reach it, so the lower rule must stop earlier.
    confidence, teacher, student = dip_case()
    full = coverage_curve(confidence, teacher, student, 3)[-1].kappa
    target = full - 1e-9
    point = choose_threshold(confidence, teacher, student, 3, target, rule="point")
    lower = choose_threshold(confidence, teacher, student, 3, target, n_resamples=500)
    assert point is not None
    assert point.coverage == 1.0
    assert lower is not None
    assert lower.coverage < 1.0
    assert lower.kappa_low is not None
    assert lower.kappa_low >= target


def test_choose_threshold_none_min_accepted_and_determinism() -> None:
    confidence, teacher, student = dip_case()
    assert choose_threshold(confidence, teacher, student, 3, 1.01, rule="point") is None
    # Only the first 40 are perfect: with min_accepted above 40, kappa 1 is unreachable.
    assert choose_threshold(confidence, teacher, student, 3, 1.0, rule="point").accepted == 40  # type: ignore[union-attr]
    assert (
        choose_threshold(confidence, teacher, student, 3, 1.0, rule="point", min_accepted=41)
        is None
    )
    a = choose_threshold(confidence, teacher, student, 3, 0.8, n_resamples=200, seed=7)
    b = choose_threshold(confidence, teacher, student, 3, 0.8, n_resamples=200, seed=7)
    assert a is not None
    assert a == b
    with pytest.raises(ValueError, match="rule"):
        choose_threshold(confidence, teacher, student, 3, 0.8, rule="mean")  # type: ignore[arg-type]


# calibrate


def test_calibrate_end_to_end() -> None:
    z, y = sampled(8, 1500, 4, 3.0)
    result = calibrate(z, y, target_kappa=0.6, rule="lower", n_resamples=200)
    assert result.temperature == pytest.approx(3.0, rel=0.2)
    assert result.ece_after < result.ece_before
    assert result.nll_after < result.nll_before
    assert result.rule == "lower"
    assert result.threshold is not None
    assert result.kappa_low is not None
    assert result.kappa_low >= 0.6
    restored = json.loads(json.dumps(result.as_dict()))
    assert restored == result.as_dict()
    assert restored["n"] == 1500


def test_calibrate_without_target_or_unreachable() -> None:
    z, y = sampled(9, 300, 3, 2.0)
    none = calibrate(z, y, target_kappa=None)
    assert none.rule is None
    assert none.threshold is None
    unreachable = calibrate(z, y, target_kappa=1.01, rule="point")
    assert unreachable.rule == "point"
    assert unreachable.threshold is None
    assert unreachable.coverage is None
    assert unreachable.kappa is None
    assert unreachable.kappa_low is None


def test_perfect_agreement_leaves_the_temperature_at_one() -> None:
    # Every argmax matches its target: the NLL has no minimum, so T is not rescaled, and an
    # unseen input keeps the confidence the student gives it instead of being pushed to 1.0.
    logits = np.array([[2.0, 0.5, 0.0], [0.1, 1.5, 0.3], [0.0, 0.2, 0.9], [1.0, 0.0, 0.2]])
    targets = np.array([0, 1, 2, 0], dtype=np.int64)
    assert fit_temperature(logits, targets) == 0.05
    assert serving_temperature(logits, targets) == (1.0, False)
    result = calibrate(logits, targets, target_kappa=None)
    assert result.temperature == 1.0
    assert result.temperature_identified is False
    assert result.ece_after == result.ece_before
    disagree = targets.copy()
    disagree[0] = 1
    t, identified = serving_temperature(logits, disagree)
    assert identified
    assert t == fit_temperature(logits, disagree)


def test_a_fit_that_runs_to_a_bound_is_not_identified_even_with_disagreements() -> None:
    # 300 confident correct inputs and one wrong input with nearly flat logits: the NLL still
    # falls as T shrinks, the fit clamps at the lower bound, and at T = 0.05 a near-uniform
    # (out-of-distribution) prediction would be served as confidence ~1.0.
    rng = np.random.default_rng(0)
    right = np.zeros((300, 5))
    right[np.arange(300), rng.integers(0, 5, 300)] = 6.0
    wrong = np.array([[0.001, 0.0, -0.05, -0.05, -0.05]])
    logits = np.concatenate([right, wrong])
    targets = np.concatenate([right.argmax(axis=1), [1]]).astype(np.int64)
    assert fit_temperature(logits, targets) == 0.05
    assert serving_temperature(logits, targets) == (1.0, False)
    flat = np.array([[0.2, 0.0, 0.0, 0.0, 0.0]])
    assert softmax(flat, 0.05)[0].max() > 0.5
    assert softmax(flat, 1.0)[0].max() < 0.25


# count rules: "exact" and "wilson"


def brute_tail(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1))


def brute_cp(k: int, n: int, confidence: float) -> float:
    if k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if brute_tail(k, n, mid) < 1 - confidence:
            lo = mid
        else:
            hi = mid
    return lo


def brute_wilson(k: int, n: int, confidence: float) -> float:
    z = NormalDist().inv_cdf(confidence)
    p_hat = k / n
    a, b, c = 1 + z * z / n, -(2 * p_hat + z * z / n), p_hat * p_hat
    return max((-b - math.sqrt(b * b - 4 * a * c)) / (2 * a), 0.0)


def brute_count_bound(
    point: CoveragePoint,
    confidence: FloatArray,
    teacher: IntArray,
    student: IntArray,
    n_labels: int,
    method: str,
    level: float,
) -> float:
    """The count bound on the inputs a threshold accepts, from their marginals by hand."""
    keep = confidence >= point.threshold
    t = [int(x) for x, m in zip(teacher, keep, strict=True) if m]
    s = [int(x) for x, m in zip(student, keep, strict=True) if m]
    n = len(t)
    agree = sum(a == b for a, b in zip(t, s, strict=True))
    chance = sum(t.count(j) * s.count(j) for j in range(n_labels))
    if chance == n * n:
        return 0.0
    low = brute_cp(agree, n, level) if method == "exact" else brute_wilson(agree, n, level)
    p_e = chance / (n * n)
    return (low - p_e) / (1 - p_e)


@pytest.mark.parametrize("rule", ["exact", "wilson"])
def test_count_rules_match_a_brute_force_scan(rule: str) -> None:
    for seed in range(2):
        confidence, teacher, student = tied_case(30 + seed, 210, 3)
        curve = coverage_curve(confidence, teacher, student, 3)
        lows = [brute_count_bound(p, confidence, teacher, student, 3, rule, 0.95) for p in curve]
        for target in (0.4, 0.6, 0.75, 0.9, 1.0):
            ok = [
                (p, low)
                for p, low in zip(curve, lows, strict=True)
                if p.accepted >= 30 and p.kappa >= target and low >= target
            ]
            choice = choose_threshold(
                confidence,
                teacher,
                student,
                3,
                target,
                rule=rule,  # type: ignore[arg-type]
            )
            if not ok:
                assert choice is None
                continue
            best, best_low = max(ok, key=lambda pl: pl[0].coverage)
            assert choice is not None
            assert choice.rule == rule
            assert (choice.threshold, choice.accepted, choice.coverage) == (
                best.threshold,
                best.accepted,
                best.coverage,
            )
            assert choice.kappa == best.kappa
            assert choice.system_kappa == best.system_kappa
            assert choice.kappa_low is not None
            assert choice.kappa_low == pytest.approx(best_low, abs=1e-9)
            assert target <= choice.kappa_low <= choice.kappa


def test_count_rules_never_exceed_the_point_rule_and_ignore_resampling() -> None:
    for seed in range(3):
        confidence, teacher, student = tied_case(seed, 300, 3)
        for target in (0.5, 0.7, 0.85):
            point = choose_threshold(confidence, teacher, student, 3, target, rule="point")
            for rule in ("exact", "wilson"):
                got = choose_threshold(
                    confidence,
                    teacher,
                    student,
                    3,
                    target,
                    rule=rule,
                )
                other = choose_threshold(
                    confidence,
                    teacher,
                    student,
                    3,
                    target,
                    rule=rule,
                    n_resamples=7,
                    seed=99,
                )
                assert got == other
                if got is not None:
                    assert point is not None
                    assert got.coverage <= point.coverage


def test_count_rules_on_perfect_agreement() -> None:
    # The first 40 inputs are perfect over 3 labels: point kappa 1, but the exact bound of 40
    # out of 40 is (0.05 ** (1 / 40) - p_e) / (1 - p_e), about 0.89. A target of 1.0 is out of
    # reach; 0.85 is met at that perfect prefix, where the full curve's bound (about 0.83)
    # misses it even though its point kappa (about 0.89) clears it.
    confidence, teacher, student = dip_case()
    assert choose_threshold(confidence, teacher, student, 3, 1.0, rule="exact") is None
    point = choose_threshold(confidence, teacher, student, 3, 0.85, rule="point")
    assert point is not None
    assert point.coverage == 1.0
    choice = choose_threshold(confidence, teacher, student, 3, 0.85, rule="exact")
    assert choice is not None
    assert choice.accepted == 40
    assert choice.kappa == 1.0
    p_e = sum((teacher[:40] == j).sum() ** 2 for j in range(3)) / 40**2
    assert choice.kappa_low == pytest.approx((0.05 ** (1 / 40) - p_e) / (1 - p_e), abs=1e-12)


def test_count_rules_bound_a_single_label_cascade_at_zero() -> None:
    # Every input has the same label from both raters: the point kappa takes the 1.0
    # convention, but p_e == 1 leaves no chance-corrected evidence, so the count bound is 0.
    same = np.zeros(50, dtype=np.int64)
    confidence = np.linspace(1.0, 0.0, 50)
    assert choose_threshold(confidence, same, same, 2, 0.5, rule="point") is not None
    for rule in ("exact", "wilson"):
        assert choose_threshold(confidence, same, same, 2, 0.5, rule=rule) is None
        assert choose_threshold(confidence, same, same, 2, 0.0, rule=rule) is not None


def test_count_rule_unknown_raises() -> None:
    confidence, teacher, student = dip_case()
    with pytest.raises(ValueError, match="unknown rule"):
        choose_threshold(confidence, teacher, student, 3, 0.8, rule="clopper")  # type: ignore[arg-type]


def test_calibrate_with_the_exact_rule() -> None:
    z, y = sampled(8, 600, 4, 3.0)
    result = calibrate(z, y, target_kappa=0.5, rule="exact")
    assert result.rule == "exact"
    assert result.threshold is not None
    assert result.kappa_low is not None
    assert result.kappa is not None
    assert 0.5 <= result.kappa_low <= result.kappa
    assert json.loads(json.dumps(result.as_dict()))["rule"] == "exact"


def test_the_default_rule_is_exact() -> None:
    confidence, teacher, student = tied_case(11, 400, 3)
    default = choose_threshold(confidence, teacher, student, 3, 0.7)
    assert default == choose_threshold(confidence, teacher, student, 3, 0.7, rule="exact")
    assert default is not None
    assert default.rule == DEFAULT_RULE == "exact"


def test_exact_rule_scales_to_a_large_held_out_set() -> None:
    # Every cut is eligible and most fail the bound: the scan must stay near linear.
    rng = np.random.default_rng(0)
    n = 20_000
    teacher = rng.integers(0, 4, size=n, dtype=np.int64)
    student = np.where(rng.random(n) < 0.95, teacher, rng.integers(0, 4, size=n))
    choice = choose_threshold(rng.random(n), teacher, student, 4, 0.95, rule="exact")
    assert choice is not None
    assert choice.kappa_low is not None
    assert choice.kappa_low >= 0.95
