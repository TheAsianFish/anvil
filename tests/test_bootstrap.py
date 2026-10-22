from __future__ import annotations

import math
import time
import warnings

import numpy as np
import numpy.typing as npt
import pytest
from scipy import special, stats  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, cohen_kappa_score

from anvil.evaluate.bootstrap import (
    Interval,
    accuracy_jackknife,
    accuracy_replicates,
    bca_levels,
    bootstrap_ci,
    bootstrap_difference_ci,
    encode,
    encode_all,
    endpoints,
    kappa_jackknife,
    kappa_replicates,
    lower_bounds,
    resample_indices,
)
from anvil.evaluate.metrics import cohen_kappa

Codes = npt.NDArray[np.int64]


def raters(
    rng: np.random.Generator, n: int, k: int, rho: float, skew: float = 1.0
) -> tuple[Codes, Codes]:
    """Rater b copies rater a with probability rho, else draws independently.

    Both raters share the label distribution p, so the population kappa is exactly rho:
    p_o = rho + (1 - rho) * sum p^2 and p_e = sum p^2.
    """
    weights = np.arange(1, k + 1, dtype=np.float64) ** -skew
    p = weights / weights.sum()
    a = rng.choice(k, size=n, p=p)
    other = rng.choice(k, size=n, p=p)
    b = np.where(rng.random(n) < rho, a, other)
    return a.astype(np.int64), b.astype(np.int64)


def sklearn_kappa(a: Codes, b: Codes) -> float:
    """sklearn's kappa with the documented p_e == 1 convention where sklearn returns nan."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn warns exactly in the p_e == 1 case
        value = float(cohen_kappa_score(a, b))
    if math.isnan(value):
        return 1.0 if bool(np.all(a == b)) else 0.0
    return value


@pytest.mark.parametrize(
    ("n", "k", "rho", "seed"),
    [(30, 3, 0.5, 0), (60, 5, 0.8, 1), (8, 2, 0.3, 2), (120, 10, 0.0, 3), (4, 4, 0.9, 4)],
)
def test_every_replicate_matches_sklearn(n: int, k: int, rho: float, seed: int) -> None:
    a, b = raters(np.random.default_rng(seed), n, k, rho, skew=2.0)
    idx = resample_indices(n, 300, seed)
    kappas = kappa_replicates(a, b, k, idx)
    accs = accuracy_replicates(a, b, k, idx)
    for row, kappa, acc in zip(idx, kappas, accs, strict=True):
        assert kappa == pytest.approx(sklearn_kappa(a[row], b[row]), abs=1e-12)
        assert acc == pytest.approx(float(accuracy_score(a[row], b[row])), abs=1e-12)


def test_degenerate_resamples() -> None:
    a = np.array([0, 0, 1, 2], dtype=np.int64)
    b = np.array([0, 1, 1, 2], dtype=np.int64)
    idx = np.array(
        [
            [0, 0, 0, 0],  # both raters constant on label 0: p_e == 1, sklearn gives nan
            [2, 2, 2, 2],  # both constant on label 1
            [1, 1, 1, 1],  # constant on different labels: p_e == 0, kappa 0
            [0, 0, 1, 1],  # a constant, b not: kappa 0
            [0, 2, 3, 1],  # the full sample
        ],
        dtype=np.int64,
    )
    kappas = kappa_replicates(a, b, 3, idx)
    assert kappas.tolist()[:4] == [1.0, 1.0, 0.0, 0.0]
    for row, kappa in zip(idx, kappas, strict=True):
        assert kappa == pytest.approx(sklearn_kappa(a[row], b[row]), abs=1e-12)
    assert kappas[4] == cohen_kappa(["0", "0", "1", "2"], ["0", "1", "1", "2"])


def test_interval_is_percentile_of_the_replicates() -> None:
    rng = np.random.default_rng(5)
    a, b = raters(rng, 200, 6, 0.7)
    sa, sb = [f"l{x}" for x in a], [f"l{x}" for x in b]
    for statistic, fn in (("kappa", kappa_replicates), ("accuracy", accuracy_replicates)):
        ci = bootstrap_ci(sa, sb, statistic, n_resamples=500, confidence=0.9, seed=11)  # type: ignore[arg-type]
        ca, cb, k = encode(sa, sb)
        values = fn(ca, cb, k, resample_indices(200, 500, 11))
        low, high = np.percentile(values, [5.0, 95.0])
        assert (ci.low, ci.high) == (low, high)
        assert (ci.confidence, ci.n_resamples, ci.method) == (0.9, 500, "percentile")
    kappa = bootstrap_ci(sa, sb, "kappa")
    assert kappa.estimate == cohen_kappa(sa, sb)
    assert bootstrap_ci(sa, sb, "accuracy").estimate == pytest.approx(float(np.mean(a == b)))


def test_deterministic_per_seed() -> None:
    a, b = raters(np.random.default_rng(6), 100, 4, 0.6)
    sa, sb = [str(x) for x in a], [str(x) for x in b]
    first = bootstrap_ci(sa, sb, "kappa", seed=3)
    assert bootstrap_ci(sa, sb, "kappa", seed=3) == first
    assert bootstrap_ci(sa, sb, "kappa", seed=4) != first
    assert isinstance(first, Interval)


def test_close_to_scipy_bootstrap() -> None:
    a, b = raters(np.random.default_rng(7), 300, 5, 0.6)
    ours = bootstrap_ci([str(x) for x in a], [str(x) for x in b], "kappa", n_resamples=5000)

    def kappa_stat(x: Codes, y: Codes, axis: int = -1) -> Codes:
        # Kappa over the last axis for any leading batch shape; the replicate test above
        # already ties this formula to sklearn, here only the resampling scheme is compared.
        xs, ys = np.moveaxis(x, axis, -1), np.moveaxis(y, axis, -1)
        n = xs.shape[-1]
        eye = np.eye(5, dtype=np.int64)
        p_o = (xs == ys).mean(axis=-1)
        p_e = (eye[xs].sum(axis=-2) * eye[ys].sum(axis=-2)).sum(axis=-1) / (n * n)
        out: Codes = (p_o - p_e) / (1 - p_e)
        return out

    res = stats.bootstrap(
        (a, b),
        kappa_stat,
        paired=True,
        vectorized=True,
        n_resamples=5000,
        method="percentile",
        confidence_level=0.95,
        random_state=np.random.default_rng(0),
    )
    # Monte Carlo noise at 5,000 resamples is about 0.001; 0.005 separates the percentile
    # method from scipy's basic (off by ~0.005 here) and BCa intervals, and a 90% interval.
    assert abs(ours.low - res.confidence_interval.low) < 0.005
    assert abs(ours.high - res.confidence_interval.high) < 0.005


def test_coverage_of_the_population_kappa() -> None:
    rng = np.random.default_rng(8)
    trials, covered = 200, 0
    for trial in range(trials):
        rho = 0.7
        a, b = raters(rng, 200, 5, rho)
        ci = bootstrap_ci(list(map(str, a)), list(map(str, b)), "kappa", seed=trial)
        covered += ci.low <= rho <= ci.high
    assert 0.90 <= covered / trials <= 0.99


def test_fast_enough_for_banking77() -> None:
    a, b = raters(np.random.default_rng(9), 500, 77, 0.8)
    sa, sb = [str(x) for x in a], [str(x) for x in b]
    start = time.perf_counter()
    bootstrap_ci(sa, sb, "kappa", n_resamples=1000)
    assert time.perf_counter() - start < 3.0  # milliseconds in practice; loose for slow CI


def test_bad_input_raises() -> None:
    with pytest.raises(ValueError, match="lengths differ"):
        bootstrap_ci(["a"], ["a", "b"], "kappa")
    with pytest.raises(ValueError, match="at least one"):
        bootstrap_ci([], [], "kappa")
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci(["a"], ["a"], "kappa", confidence=1.0)
    with pytest.raises(ValueError, match="n_resamples"):
        resample_indices(5, 0, 0)
    with pytest.raises(ValueError, match="unknown statistic"):
        bootstrap_ci(["a"], ["a"], "f1")  # type: ignore[arg-type]


def test_paired_difference_interval() -> None:
    rng = np.random.default_rng(10)
    ref, a = raters(rng, 300, 5, 0.8)
    _, b = raters(rng, 300, 5, 0.6)
    sr, sa, sb = ([f"l{x}" for x in v] for v in (ref, a, b))
    diff = bootstrap_difference_ci(sr, sa, sb, "kappa", n_resamples=400, seed=2)
    assert diff.estimate == pytest.approx(cohen_kappa(sr, sa) - cohen_kappa(sr, sb))
    # The same item resamples as the two marginal intervals, so the difference of the
    # per-resample kappas is exactly what is summarised.
    idx = resample_indices(300, 400, 2)
    ka = kappa_replicates(ref, a, 5, idx)
    kb = kappa_replicates(ref, b, 5, idx)
    low, high = np.percentile(ka - kb, [2.5, 97.5])
    assert (diff.low, diff.high) == (low, high)
    assert diff.low > 0  # rho 0.8 vs 0.6 on 300 items separates
    same = bootstrap_difference_ci(sr, sa, sa, "accuracy", n_resamples=50)
    assert (same.estimate, same.low, same.high) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="lengths differ"):
        bootstrap_difference_ci(sr, sa, sb[:-1], "kappa")


def test_paired_difference_is_sharper_than_overlap_on_paired_data() -> None:
    # Two students that differ on a handful of items, both at kappa ~0.7: the marginal
    # intervals overlap almost entirely, the paired interval on the difference excludes 0.
    rng = np.random.default_rng(12)
    ref, a = raters(rng, 400, 4, 0.7)
    b = a.copy()
    disagree = np.flatnonzero(a != ref)[:20]
    b[disagree] = ref[disagree]  # b fixes 20 of a's errors, nothing else changes
    sr, sa, sb = ([str(x) for x in v] for v in (ref, a, b))
    ca, cb = bootstrap_ci(sr, sa, "kappa"), bootstrap_ci(sr, sb, "kappa")
    assert ca.low < cb.high  # the marginal intervals overlap...
    assert cb.low < ca.high
    diff = bootstrap_difference_ci(sr, sb, sa, "kappa")
    assert diff.low > 0


def test_codes_outside_the_label_range_are_rejected() -> None:
    a = np.array([0, 1, 2], dtype=np.int64)
    b = np.array([0, 1, 1], dtype=np.int64)
    idx = np.arange(3, dtype=np.int64)[None, :]
    assert kappa_replicates(a, b, 3, idx)[0] == pytest.approx(sklearn_kappa(a, b))
    with pytest.raises(ValueError, match=r"codes must lie in \[0, 2\)"):
        kappa_replicates(a, b, 2, idx)
    with pytest.raises(ValueError, match="codes must lie"):
        accuracy_replicates(a, b - 1, 3, idx)


def _kappa_stat(k: int) -> object:
    def stat(x: Codes, y: Codes, axis: int = -1) -> Codes:
        xs, ys = np.moveaxis(x, axis, -1), np.moveaxis(y, axis, -1)
        n = xs.shape[-1]
        eye = np.eye(k, dtype=np.int64)
        p_o = (xs == ys).mean(axis=-1)
        p_e = (eye[xs].sum(axis=-2) * eye[ys].sum(axis=-2)).sum(axis=-1) / (n * n)
        out: Codes = (p_o - p_e) / (1 - p_e)
        return out

    return stat


@pytest.mark.parametrize(
    ("n", "k", "rho", "seed"),
    [(30, 3, 0.5, 0), (60, 77, 0.9, 1), (8, 2, 0.3, 2), (5, 4, 1.0, 3), (2, 2, 0.0, 4)],
)
def test_jackknife_matches_leaving_each_item_out(n: int, k: int, rho: float, seed: int) -> None:
    a, b = raters(np.random.default_rng(seed), n, k, rho)
    brute_kappa = [sklearn_kappa(np.delete(a, i), np.delete(b, i)) for i in range(n)]
    brute_acc = [float(np.mean(np.delete(a, i) == np.delete(b, i))) for i in range(n)]
    np.testing.assert_allclose(kappa_jackknife(a, b, k), brute_kappa, atol=1e-12)
    np.testing.assert_allclose(accuracy_jackknife(a, b, k), brute_acc, atol=1e-12)


def test_jackknife_degenerate_and_validation() -> None:
    # Every leave-one-out set of one repeated label pair has p_e == 1: kappa 1.0 when the
    # raters agree on it, 0.0 when they do not.
    same = np.zeros(4, dtype=np.int64)
    assert kappa_jackknife(same, same, 2).tolist() == [1.0] * 4
    a, b = np.array([0, 0, 1], dtype=np.int64), np.array([1, 1, 1], dtype=np.int64)
    assert kappa_jackknife(a, b, 2)[2] == 0.0  # {(0, 1), (0, 1)}: p_e = 0, p_o = 0
    with pytest.raises(ValueError, match="at least two"):
        kappa_jackknife(same[:1], same[:1], 2)
    with pytest.raises(ValueError, match="codes must lie"):
        kappa_jackknife(a, b + 1, 2)


def test_bca_levels_follow_the_textbook_formula() -> None:
    rng = np.random.default_rng(21)
    values = rng.normal(0.8, 0.05, 1000) ** 2  # skewed, like a bounded statistic
    jack = rng.normal(0.64, 0.01, 40) ** 3
    estimate = 0.62
    levels = [0.025, 0.05, 0.975]
    ours = bca_levels(values, estimate, jack, levels)
    # The same quantities through scipy's normal functions, written out (Efron and
    # Tibshirani eq. 14.9-14.15), with the mid-rank share scipy.stats.bootstrap uses.
    share = (np.sum(values < estimate) + np.sum(values <= estimate)) / (2 * values.size)
    z0 = special.ndtri(share)
    d = jack.mean() - jack
    a = np.sum(d**3) / (6 * np.sum(d**2) ** 1.5)
    z = z0 + special.ndtri(levels)
    np.testing.assert_allclose(ours, special.ndtr(z0 + z / (1 - a * z)), rtol=1e-12)


def test_bca_without_bias_or_acceleration_is_the_percentile_method() -> None:
    # Replicates symmetric about the estimate (z0 = 0) and a symmetric jackknife (a = 0).
    values = np.concatenate([np.linspace(0.5, 0.9, 500), np.linspace(0.9, 1.3, 500)])
    jack = np.array([0.8, 0.9, 1.0])
    np.testing.assert_allclose(bca_levels(values, 0.9, jack, [0.05, 0.95]), [0.05, 0.95])
    np.testing.assert_allclose(
        endpoints(values, [0.05], "bca", estimate=0.9, jackknife=jack),
        endpoints(values, [0.05]),
    )


def test_bca_edge_cases() -> None:
    values = np.full(1000, 0.7)
    # Every replicate above the estimate: the share is clipped to 1 / (2B), so z0 is finite.
    finite = bca_levels(values, 0.5, np.array([0.5, 0.5]), [0.05])
    assert 0.0 < finite[0] < 0.05
    # |a| <= 1/6 always, so 1 - a * (z0 + z_alpha) <= 0 needs |z0 + z_alpha| >= 6: one
    # leave-one-out value far from the rest gives a close to -1/6, and a level of 1e-12
    # (z_alpha = -7) leaves the formula's domain. The endpoint goes to its own extreme.
    jack = np.array([0.0] * 399 + [10.0])
    spread = np.linspace(0.0, 1.0, 1000)
    lo, mid, hi = bca_levels(spread, 0.5, jack, [1e-12, 0.05, 1 - 1e-12])
    assert lo == 0.0
    assert 0.0 < mid < 0.05  # a < 0 moves the lower level down, more conservative
    assert 0.0 < hi < 1.0
    with pytest.raises(ValueError, match=r"levels must lie in \(0, 1\)"):
        bca_levels(spread, 0.5, jack, [0.0])
    with pytest.raises(ValueError, match="at least two"):
        bca_levels(spread, 0.5, jack[:1], [0.05])
    with pytest.raises(ValueError, match="needs the estimate"):
        endpoints(spread, [0.05], "bca")
    with pytest.raises(ValueError, match="unknown method"):
        endpoints(spread, [0.05], "basic")  # type: ignore[arg-type]


@pytest.mark.parametrize(("n", "k", "rho"), [(300, 5, 0.6), (100, 77, 0.9), (150, 4, 0.95)])
def test_bca_interval_close_to_scipy(n: int, k: int, rho: float) -> None:
    a, b = raters(np.random.default_rng(22), n, k, rho)
    sa, sb = [str(x) for x in a], [str(x) for x in b]
    ours = bootstrap_ci(sa, sb, "kappa", n_resamples=5000, method="bca")
    assert ours.method == "bca"
    ac, bc, kk = encode(sa, sb)
    res = stats.bootstrap(
        (ac, bc),
        _kappa_stat(kk),
        paired=True,
        vectorized=True,
        n_resamples=5000,
        method="BCa",
        confidence_level=0.95,
        random_state=np.random.default_rng(0),
    )
    # Different resamples, the same method: Monte Carlo noise at 5,000 resamples is about
    # 0.002 here, and the BCa and percentile endpoints differ by more than 0.005 on (150, 4).
    assert abs(ours.low - res.confidence_interval.low) < 0.006
    assert abs(ours.high - res.confidence_interval.high) < 0.006


def test_lower_bounds_read_one_set_of_replicates() -> None:
    a, b = raters(np.random.default_rng(23), 200, 6, 0.85)
    idx = resample_indices(200, 1000, 4)
    values = kappa_replicates(a, b, 6, idx)
    bounds = lower_bounds(a, b, 6, idx, 0.95, ("percentile", "bca"))
    assert bounds["percentile"] == pytest.approx(float(np.percentile(values, 5)))
    estimate = float(kappa_replicates(a, b, 6, np.arange(200, dtype=np.int64)[None, :])[0])
    level = bca_levels(values, estimate, kappa_jackknife(a, b, 6), [0.05])[0]
    assert bounds["bca"] == pytest.approx(float(np.percentile(values, 100 * level)))
    acc = lower_bounds(a, b, 6, idx, 0.9, ("bca",), statistic="accuracy")["bca"]
    assert 0.0 < acc < float(np.mean(a == b))
    with pytest.raises(ValueError, match="confidence"):
        lower_bounds(a, b, 6, idx, 1.0)


def test_bca_difference_interval_uses_the_difference_of_the_jackknifes() -> None:
    rng = np.random.default_rng(24)
    ref, a = raters(rng, 250, 4, 0.85)
    _, b = raters(rng, 250, 4, 0.7)
    sr, sa, sb = ([str(x) for x in v] for v in (ref, a, b))
    diff = bootstrap_difference_ci(sr, sa, sb, "kappa", n_resamples=800, seed=1, method="bca")
    (rc, ac, bc), k = encode_all(sr, sa, sb)
    idx = resample_indices(250, 800, 1)
    values = kappa_replicates(rc, ac, k, idx) - kappa_replicates(rc, bc, k, idx)
    jack = kappa_jackknife(rc, ac, k) - kappa_jackknife(rc, bc, k)
    expected = endpoints(values, [0.025, 0.975], "bca", estimate=diff.estimate, jackknife=jack)
    assert (diff.low, diff.high) == (expected[0], expected[1])
    assert diff.method == "bca"
