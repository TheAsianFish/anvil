from __future__ import annotations

import math
from fractions import Fraction
from statistics import NormalDist

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st

from anvil.evaluate.binomial import (
    CountMethod,
    binomial_pmf,
    binomial_upper_tail,
    clopper_pearson_lower,
    kappa_bound_clears,
    kappa_lower_bound,
    kappa_lower_from_counts,
    wilson_lower,
)
from anvil.evaluate.metrics import cohen_kappa

IntArray = npt.NDArray[np.int64]


# Independent references: math.comb and plain sums, nothing from the module under test.


def pmf(j: int, n: int, p: float) -> float:
    return math.comb(n, j) * p**j * (1 - p) ** (n - j)


def tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), summed term by term."""
    return sum(pmf(j, n, p) for j in range(k, n + 1))


def exact_tail(k: int, n: int, p: Fraction) -> Fraction:
    return sum((math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1)), Fraction(0))


def cp_bisect(k: int, n: int, confidence: float) -> float:
    """The root in p of tail(k, n, p) = 1 - confidence by bisection, 0 at k == 0."""
    if k == 0:
        return 0.0
    alpha = 1 - confidence
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if tail(k, n, mid) < alpha:
            lo = mid
        else:
            hi = mid
    return lo


def wilson_quadratic(k: int, n: int, confidence: float) -> float:
    """Smaller root of (p_hat - p)^2 = z^2 p (1 - p) / n, i.e.
    (1 + z^2/n) p^2 - (2 p_hat + z^2/n) p + p_hat^2 = 0, clipped at 0."""
    z = NormalDist().inv_cdf(confidence)
    p_hat = k / n
    a = 1 + z * z / n
    b = -(2 * p_hat + z * z / n)
    c = p_hat * p_hat
    disc = b * b - 4 * a * c
    return max((-b - math.sqrt(disc)) / (2 * a), 0.0)


def coverage(bounds: npt.NDArray[np.float64], n: int, p: float) -> float:
    """P(bound(X) <= p) for X ~ Binomial(n, p), by enumerating every count."""
    return sum(pmf(k, n, p) for k in range(n + 1) if bounds[k] <= p)


# binomial_upper_tail


def test_upper_tail_matches_brute_force_on_a_grid_with_edges() -> None:
    for n in (1, 2, 7, 20, 60):
        for k in sorted({0, 1, n // 2, n - 1, n}):
            for p in (0.0, 1e-3, 0.1, 0.37, 0.5, 0.9, 0.999, 1.0):
                got = float(binomial_upper_tail(k, n, p)[0])
                assert got == pytest.approx(tail(k, n, p), abs=1e-12), (k, n, p)


def test_upper_tail_matches_exact_rationals() -> None:
    for n, k, p in ((10, 3, Fraction(1, 3)), (25, 20, Fraction(7, 8)), (40, 1, Fraction(1, 50))):
        assert float(binomial_upper_tail(k, n, float(p))[0]) == pytest.approx(
            float(exact_tail(k, n, p)), rel=1e-12
        )


def test_upper_tail_edges() -> None:
    assert binomial_upper_tail(0, 5, 0.0)[0] == 1.0
    assert binomial_upper_tail(0, 5, 0.3)[0] == 1.0
    assert binomial_upper_tail(3, 5, 0.0)[0] == 0.0
    assert binomial_upper_tail(5, 5, 1.0)[0] == 1.0
    assert binomial_upper_tail(5, 5, 0.5)[0] == pytest.approx(1 / 32)


def test_upper_tail_broadcasts_a_scalar_p() -> None:
    k = np.array([0, 1, 4, 9, 12], dtype=np.int64)
    n = np.array([3, 5, 9, 12, 12], dtype=np.int64)
    got = binomial_upper_tail(k, n, 0.6)
    assert got.shape == (5,)
    assert np.allclose(got, [tail(int(a), int(b), 0.6) for a, b in zip(k, n, strict=True)])
    # scalar n over an array of k, and an array of p
    ks = np.arange(0, 11, dtype=np.int64)
    ps = np.linspace(0.0, 1.0, 11)
    got2 = binomial_upper_tail(ks, 10, ps)
    assert np.allclose(got2, [tail(int(a), 10, float(p)) for a, p in zip(ks, ps, strict=True)])


def test_upper_tail_rejects_bad_p() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        binomial_upper_tail(1, 3, 1.5)
    with pytest.raises(ValueError, match="probabilities"):
        binomial_upper_tail(1, 3, math.nan)


# clopper_pearson_lower


@pytest.mark.parametrize("confidence", [0.9, 0.95, 0.99])
def test_clopper_pearson_is_the_root_of_the_tail(confidence: float) -> None:
    alpha = 1 - confidence
    for n in (1, 4, 17, 50, 200):
        for k in sorted({1, n // 3, n // 2, n - 1, n} - {0}):
            low = float(clopper_pearson_lower(k, n, confidence)[0])
            assert tail(k, n, low) == pytest.approx(alpha, abs=1e-9), (k, n)
            assert tail(k, n, max(low - 1e-6, 0.0)) < alpha
            assert tail(k, n, min(low + 1e-6, 1.0)) > alpha
            assert low == pytest.approx(cp_bisect(k, n, confidence), abs=1e-10)


def test_clopper_pearson_closed_forms() -> None:
    assert np.all(clopper_pearson_lower(np.zeros(4, dtype=np.int64), [1, 5, 9, 100]) == 0.0)
    for n in (1, 3, 10, 80):
        for c in (0.8, 0.95, 0.999):
            assert float(clopper_pearson_lower(n, n, c)[0]) == pytest.approx(
                (1 - c) ** (1 / n), abs=1e-12
            )


def test_clopper_pearson_matches_scipy_beta() -> None:
    stats = pytest.importorskip("scipy.stats")
    ns = np.array([3, 10, 10, 47, 120, 500], dtype=np.int64)
    ks = np.array([1, 4, 9, 40, 119, 480], dtype=np.int64)
    for confidence in (0.9, 0.95):
        alpha = 1 - confidence
        got = clopper_pearson_lower(ks, ns, confidence)
        expected = stats.beta.ppf(alpha, ks, ns - ks + 1)
        assert np.allclose(got, expected, atol=1e-10)


def test_clopper_pearson_increases_in_k() -> None:
    for n in (1, 6, 33, 150):
        bounds = clopper_pearson_lower(np.arange(n + 1, dtype=np.int64), n)
        assert np.all(np.diff(bounds) > 0)


@given(st.integers(1, 80).flatmap(lambda n: st.tuples(st.just(n), st.integers(0, n - 1))))
def test_clopper_pearson_increases_in_k_property(case: tuple[int, int]) -> None:
    n, k = case
    lo, hi = clopper_pearson_lower(np.array([k, k + 1], dtype=np.int64), n)
    assert lo < hi
    assert 0.0 <= lo <= k / n


@pytest.mark.parametrize("n", [5, 12, 30])
def test_clopper_pearson_covers_at_every_p(n: int) -> None:
    confidence = 0.95
    bounds = clopper_pearson_lower(np.arange(n + 1, dtype=np.int64), n, confidence)
    # The grid includes the bounds themselves, where coverage jumps.
    grid = np.unique(np.concatenate([np.linspace(1e-4, 1 - 1e-4, 400), bounds[1:]]))
    for p in grid:
        assert coverage(bounds, n, float(p)) >= confidence - 1e-12, (n, p)


@pytest.mark.parametrize("n", [5, 12, 30])
def test_wilson_is_close_on_average_but_not_exact(n: int) -> None:
    # Wilson is an approximation: averaged over p its coverage is near nominal, but at some p
    # it falls well short (an exact bound never does, see the test above).
    confidence = 0.95
    bounds = wilson_lower(np.arange(n + 1, dtype=np.int64), n, confidence)
    grid = np.linspace(1e-3, 1 - 1e-3, 500)
    cov = np.array([coverage(bounds, n, float(p)) for p in grid])
    assert abs(float(cov.mean()) - confidence) < 0.03
    assert float(cov.min()) < confidence - 0.05


# wilson_lower


@pytest.mark.parametrize("confidence", [0.8, 0.95, 0.99])
def test_wilson_is_the_lower_root_of_the_score_quadratic(confidence: float) -> None:
    z = NormalDist().inv_cdf(confidence)
    for n in (1, 2, 9, 40, 300):
        ks = np.arange(n + 1, dtype=np.int64)
        got = wilson_lower(ks, n, confidence)
        for k in range(n + 1):
            low = float(got[k])
            assert low == pytest.approx(wilson_quadratic(k, n, confidence), abs=1e-12)
            # k == 0 gives 0 only up to rounding (about 1e-17), as the docstring says.
            assert 0.0 <= low <= k / n + 1e-15
            if k > 0:  # it is a root: the score equation holds there
                p_hat = k / n
                assert (p_hat - low) ** 2 == pytest.approx(z * z * low * (1 - low) / n, abs=1e-12)
    assert np.allclose(wilson_lower(np.zeros(3, dtype=np.int64), [1, 10, 100]), 0.0, atol=1e-15)


# kappa_lower_from_counts


def test_kappa_lower_from_counts_hand_computed() -> None:
    # 20 items, 17 agreements; marginals rows (10, 6, 4), cols (9, 7, 4): chance = 90+42+16.
    n, agree, chance = 20, 17, 10 * 9 + 6 * 7 + 4 * 4
    p_e = chance / n**2
    for method, low in (
        ("exact", cp_bisect(agree, n, 0.95)),
        ("wilson", wilson_quadratic(agree, n, 0.95)),
    ):
        got = kappa_lower_from_counts(agree, chance, n, method)  # type: ignore[arg-type]
        assert float(got[0]) == pytest.approx((low - p_e) / (1 - p_e), abs=1e-10)


def test_kappa_lower_from_counts_vectorised() -> None:
    agree = np.array([5, 30, 99], dtype=np.int64)
    chance = np.array([10, 300, 2600], dtype=np.int64)
    n = np.array([6, 40, 100], dtype=np.int64)
    got = kappa_lower_from_counts(agree, chance, n, "exact", 0.9)
    for i in range(3):
        a, c, m = int(agree[i]), int(chance[i]), int(n[i])
        p_e = c / m**2
        assert float(got[i]) == pytest.approx((cp_bisect(a, m, 0.9) - p_e) / (1 - p_e), abs=1e-10)


def test_kappa_lower_from_counts_degenerate_chance_is_zero() -> None:
    for method in ("exact", "wilson"):
        got = kappa_lower_from_counts(10, 100, 10, method)
        assert float(got[0]) == 0.0


def test_kappa_lower_from_counts_validation() -> None:
    with pytest.raises(ValueError, match="successes"):
        kappa_lower_from_counts(11, 50, 10, "exact")
    with pytest.raises(ValueError, match="successes"):
        kappa_lower_from_counts(-1, 50, 10, "exact")
    with pytest.raises(ValueError, match="trials"):
        kappa_lower_from_counts(0, 0, 0, "exact")
    with pytest.raises(ValueError, match="integer"):
        kappa_lower_from_counts(5.0, 50, 10, "exact")
    with pytest.raises(ValueError, match="integer"):
        kappa_lower_from_counts(5, 50, 10.0, "exact")
    with pytest.raises(ValueError, match="integer"):
        kappa_lower_from_counts(5, 50.0, 10, "exact")
    with pytest.raises(ValueError, match="chance"):
        kappa_lower_from_counts(5, 101, 10, "exact")
    with pytest.raises(ValueError, match="chance"):
        kappa_lower_from_counts(5, -1, 10, "exact")
    with pytest.raises(ValueError, match="method"):
        kappa_lower_from_counts(5, 50, 10, "bca")  # type: ignore[arg-type]
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="confidence"):
            kappa_lower_from_counts(5, 50, 10, "exact", bad)


# kappa_lower_bound


def test_kappa_lower_bound_matches_counts_and_sits_below_the_point_kappa() -> None:
    rng = np.random.default_rng(3)
    for n, labels in ((25, 2), (80, 4), (300, 7)):
        a = rng.integers(0, labels, size=n).astype(np.int64)
        b = np.where(rng.random(n) < 0.8, a, rng.integers(0, labels, size=n)).astype(np.int64)
        rows = [int((a == j).sum()) for j in range(labels)]
        cols = [int((b == j).sum()) for j in range(labels)]
        agree = int((a == b).sum())
        chance = sum(r * c for r, c in zip(rows, cols, strict=True))
        point = cohen_kappa([str(x) for x in a], [str(x) for x in b])
        for method in ("exact", "wilson"):
            low = kappa_lower_bound(a, b, labels, method)
            counts = kappa_lower_from_counts(agree, chance, n, method)
            assert low == float(counts[0])
            assert low <= point


def test_kappa_lower_bound_validation() -> None:
    a: IntArray = np.array([0, 1, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="equal length"):
        kappa_lower_bound(a, a[:2], 2, "exact")
    with pytest.raises(ValueError, match="equal length"):
        kappa_lower_bound(a[:0], a[:0], 2, "exact")
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        kappa_lower_bound(a, np.array([0, 1, 2], dtype=np.int64), 2, "exact")


@pytest.mark.parametrize("n", [1, 7, 60])
def test_binomial_pmf_matches_brute_force(n: int) -> None:
    rates = [0.0, 0.03, 0.5, 0.97, 1.0]
    table = binomial_pmf(n, rates)
    assert table.shape == (len(rates), n + 1)
    for row, p in zip(table, rates, strict=True):
        assert row == pytest.approx([pmf(j, n, p) for j in range(n + 1)], abs=1e-12)
        assert row.sum() == pytest.approx(1.0, abs=1e-12)


def test_binomial_pmf_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="trials"):
        binomial_pmf(0, [0.5])
    with pytest.raises(ValueError, match="probabilities"):
        binomial_pmf(5, [1.5])


@pytest.mark.parametrize("method", ["exact", "wilson"])
def test_kappa_bound_clears_agrees_with_the_bound(method: CountMethod) -> None:
    rng = np.random.default_rng(3)
    n = rng.integers(1, 400, size=500)
    agree = (n * rng.uniform(0.5, 1.0, size=n.size)).astype(np.int64)
    chance = (n * n * rng.uniform(0.0, 1.0, size=n.size)).astype(np.int64)
    chance[:20] = n[:20] * n[:20]  # degenerate p_e == 1
    for target in (-0.1, 0.0, 0.5, 0.9, 0.95, 1.0):
        bound = kappa_lower_from_counts(agree, chance, n, method)
        clears = kappa_bound_clears(agree, chance, n, method, target)
        decided = np.abs(bound - target) > 1e-9  # leave out ties at the bisection's precision
        assert (clears[decided] == (bound[decided] >= target)).all()
