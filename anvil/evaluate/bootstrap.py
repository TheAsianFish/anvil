"""Bootstrap confidence intervals for agreement statistics: percentile and BCa.

The unit of resampling is the item: each resample draws n item indices with replacement and
keeps both raters' labels for each drawn item together (paired resampling), so the interval
reflects how much the statistic would move on another held-out set of the same size drawn
from the same population.

Two ways to turn the resampled statistics into an interval (Efron and Tibshirani, ch. 13-14):

- percentile: the 95% interval is the 2.5th and 97.5th percentiles of the replicates. Simple,
  but it assumes the replicates are centred on the estimate and spread symmetrically on some
  scale. Kappa is bounded at 1 and skewed near it, and the sample kappa is biased (its chance
  term is estimated from the same items), so the percentile interval can sit too high.
- BCa (bias-corrected and accelerated): the same replicates, read at adjusted percentiles.
  The bias correction z0 is the normal quantile of the share of replicates below the estimate
  (0 when they straddle it evenly); the acceleration a is the skewness of the jackknife
  (leave-one-out) values, which measures how fast the statistic's spread changes with its
  value. For level alpha the BCa endpoint is the percentile at

      Phi(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))

  which is alpha itself when z0 = a = 0. The jackknife of kappa is computed from the full
  sample's marginals in O(n + K) (dropping item i changes one row count, one column count and
  maybe the agreement), like the replicates, so BCa costs no more than the percentile method.

`eval/coverage.py` (`make eval-coverage`) measures how often each bound covers the population
kappa at the sample sizes the cascade's threshold search meets.

Everything is vectorised over resamples: 1,000 resamples of 500 items with 77 labels take a
few milliseconds, so the CI is cheap enough to report on every eval and every loop round.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np
import numpy.typing as npt

Statistic = Literal["kappa", "accuracy"]
Method = Literal["percentile", "bca"]
IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Interval:
    """A point estimate on the full sample and its bootstrap interval [low, high]."""

    estimate: float
    low: float
    high: float
    confidence: float
    n_resamples: int
    method: str = "percentile"


def resample_indices(n: int, n_resamples: int, seed: int) -> IntArray:
    """Item indices for every resample, shape (n_resamples, n), drawn uniformly with replacement.

    Drawing all indices up front from one seeded generator makes the interval a pure function of
    (inputs, n_resamples, seed), and lets kappa and accuracy share the same resamples.
    """
    if n < 1:
        raise ValueError("bootstrap needs at least one item")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_resamples, n), dtype=np.int64)


def _check(a_codes: IntArray, b_codes: IntArray, n_labels: int, idx: IntArray) -> None:
    if a_codes.shape != b_codes.shape or a_codes.ndim != 1:
        raise ValueError(
            f"rater codes must be 1-d and equal length: {a_codes.shape} vs {b_codes.shape}"
        )
    if idx.ndim != 2 or idx.shape[1] == 0:
        raise ValueError(f"idx must have shape (n_resamples, n) with n > 0, got {idx.shape}")
    if a_codes.size and (
        a_codes.min() < 0 or b_codes.min() < 0 or max(a_codes.max(), b_codes.max()) >= n_labels
    ):
        # A code outside [0, n_labels) would spill into the next resample's bincount row and
        # return a wrong kappa silently.
        raise ValueError(f"label codes must lie in [0, {n_labels})")


def accuracy_replicates(
    a_codes: IntArray, b_codes: IntArray, n_labels: int, idx: IntArray
) -> FloatArray:
    """Observed agreement p_o = (1/n) * sum_i [a_i == b_i] for every resample (row of `idx`).

    `n_labels` only bounds the codes; it keeps the signature interchangeable with
    `kappa_replicates`.
    """
    _check(a_codes, b_codes, n_labels, idx)
    out: FloatArray = np.count_nonzero((a_codes == b_codes)[idx], axis=1) / idx.shape[1]
    return out


def kappa_replicates(
    a_codes: IntArray, b_codes: IntArray, n_labels: int, idx: IntArray
) -> FloatArray:
    """Cohen's kappa for every resample (row of `idx`), all at once.

    Labels are integer codes in [0, n_labels). For resample r with n items, the per-resample
    confusion matrix C_r only enters kappa through its trace and its marginals:

        p_o = trace(C_r) / n
        p_e = sum_k rowsum_k(C_r) * colsum_k(C_r) / n^2
        kappa = (p_o - p_e) / (1 - p_e)

    so the marginals are counted directly with one np.bincount over r * K + code per rater
    (memory O(R * K) instead of O(R * K^2) for the full matrices). p_e is computed from integer
    counts, so p_e == 1 is detected exactly and handled as in `metrics.cohen_kappa`: kappa is
    1.0 when the raters also agree everywhere and 0.0 otherwise.
    """
    _check(a_codes, b_codes, n_labels, idx)
    r, n = idx.shape
    k = n_labels
    offsets = (np.arange(r, dtype=np.int64) * k)[:, None]
    a_r = a_codes[idx]
    a_r += offsets  # in place: the gathered copy is ours, and one temporary fewer is faster
    b_r = b_codes[idx]
    b_r += offsets
    rows = np.bincount(a_r.ravel(), minlength=r * k).reshape(r, k)
    cols = np.bincount(b_r.ravel(), minlength=r * k).reshape(r, k)
    agree = np.count_nonzero((a_codes == b_codes)[idx], axis=1)
    chance = (rows * cols).sum(axis=1)  # integer: n^2 * p_e
    p_o = agree / n
    p_e = chance / (n * n)
    degenerate = chance == n * n
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = (p_o - p_e) / (1 - p_e)
    fallback = np.where(agree == n, 1.0, 0.0)
    out: FloatArray = np.where(degenerate, fallback, kappa)
    return out


def _check_jackknife(a_codes: IntArray, b_codes: IntArray, n_labels: int) -> None:
    _check(a_codes, b_codes, n_labels, np.zeros((1, 1), dtype=np.int64))
    if a_codes.size < 2:
        raise ValueError("the jackknife needs at least two items")


def accuracy_jackknife(a_codes: IntArray, b_codes: IntArray, n_labels: int) -> FloatArray:
    """Leave-one-out observed agreement: entry i is p_o on every item but item i."""
    _check_jackknife(a_codes, b_codes, n_labels)
    hit = (a_codes == b_codes).astype(np.int64)
    out: FloatArray = (hit.sum() - hit) / (a_codes.size - 1)
    return out


def kappa_jackknife(a_codes: IntArray, b_codes: IntArray, n_labels: int) -> FloatArray:
    """Leave-one-out Cohen's kappa: entry i is kappa on every item but item i.

    With row counts r and column counts c of the full sample, S = sum_k r_k * c_k and A the
    number of agreements, dropping item i (codes a_i, b_i) leaves n - 1 items with

        A_i = A - [a_i == b_i]
        S_i = sum_k (r_k - [k == a_i]) * (c_k - [k == b_i]) = S - c[a_i] - r[b_i] + [a_i == b_i]

    so all n values come from the marginals in O(n + K), with the integer p_e == 1 handling of
    `kappa_replicates`.
    """
    _check_jackknife(a_codes, b_codes, n_labels)
    n = a_codes.size
    rows = np.bincount(a_codes, minlength=n_labels)
    cols = np.bincount(b_codes, minlength=n_labels)
    hit = (a_codes == b_codes).astype(np.int64)
    agree = hit.sum() - hit
    chance = int((rows * cols).sum()) - cols[a_codes] - rows[b_codes] + hit
    m = n - 1
    p_o = agree / m
    p_e = chance / (m * m)
    degenerate = chance == m * m
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = (p_o - p_e) / (1 - p_e)
    out: FloatArray = np.where(degenerate, np.where(agree == m, 1.0, 0.0), kappa)
    return out


_NORMAL = NormalDist()


def bca_levels(
    replicates: FloatArray,
    estimate: float,
    jackknife: FloatArray,
    levels: Sequence[float] | FloatArray,
) -> FloatArray:
    """The percentile levels (in [0, 1]) at which to read `replicates` for BCa endpoints.

    z0 = Phi^-1(share of replicates below the estimate, ties counted half), the mid-rank
    convention scipy.stats.bootstrap also uses: kappa replicates of a near-perfect student tie
    the estimate often (every resample of an all-agreeing set is exactly 1.0), and counting
    ties as "below" or "above" would push z0 to an infinity either way. The share is clipped
    to [1 / (2B), 1 - 1 / (2B)] for B replicates, so z0 stays finite when every replicate lies
    on one side. The acceleration is

        a = sum_i d_i^3 / (6 * (sum_i d_i^2)^1.5),  d_i = mean(jackknife) - jackknife_i,

    0 when the jackknife values are all equal. A level is alpha mapped through
    Phi(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))). |a| <= 1/6 always (|sum d^3| <=
    (sum d^2)^1.5, the l3 norm being at most the l2 norm), so the denominator can only reach 0
    when |z0 + z_alpha| >= 6, far beyond the levels a cascade uses; if it does, the formula
    has left its domain and the level goes to the extreme on its own side (0 for a lower
    endpoint, 1 for an upper one), the conservative reading.
    """
    values = np.asarray(replicates, dtype=np.float64)
    jack = np.asarray(jackknife, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("replicates must be 1-d and non-empty")
    if jack.ndim != 1 or jack.size < 2:
        raise ValueError("jackknife must be 1-d with at least two values")
    b = values.size
    share = (np.sum(values < estimate) + 0.5 * np.sum(values == estimate)) / b
    z0 = _NORMAL.inv_cdf(min(max(float(share), 0.5 / b), 1 - 0.5 / b))
    d = jack.mean() - jack
    ss = float(np.sum(d * d))
    accel = float(np.sum(d**3)) / (6 * ss**1.5) if ss > 0 else 0.0
    out = np.empty(len(levels), dtype=np.float64)
    for i, level in enumerate(levels):
        if not 0.0 < level < 1.0:
            raise ValueError(f"levels must lie in (0, 1), got {level}")
        z = z0 + _NORMAL.inv_cdf(float(level))
        denom = 1 - accel * z
        out[i] = _NORMAL.cdf(z0 + z / denom) if denom > 0 else (0.0 if level < 0.5 else 1.0)
    return out


_REPLICATES = {"kappa": kappa_replicates, "accuracy": accuracy_replicates}
_JACKKNIFE = {"kappa": kappa_jackknife, "accuracy": accuracy_jackknife}


def endpoints(
    replicates: FloatArray,
    levels: Sequence[float] | FloatArray,
    method: Method = "percentile",
    *,
    estimate: float | None = None,
    jackknife: FloatArray | None = None,
) -> FloatArray:
    """The replicates read at `levels` (fractions in (0, 1), np.percentile with linear
    interpolation): as given for "percentile", BCa-adjusted (`bca_levels`) for "bca", which
    needs the full-sample `estimate` and its `jackknife` values."""
    if method == "percentile":
        adjusted = np.asarray(levels, dtype=np.float64)
    elif method == "bca":
        if estimate is None or jackknife is None:
            raise ValueError("the BCa method needs the estimate and its jackknife values")
        adjusted = bca_levels(replicates, estimate, jackknife, levels)
    else:
        raise ValueError(f"unknown method {method!r}; use 'percentile' or 'bca'")
    out: FloatArray = np.percentile(replicates, 100 * adjusted)
    return out


def lower_bounds(
    a_codes: IntArray,
    b_codes: IntArray,
    n_labels: int,
    idx: IntArray,
    confidence: float = 0.95,
    methods: Sequence[Method] = ("percentile",),
    statistic: Statistic = "kappa",
) -> dict[Method, float]:
    """One-sided lower confidence bounds of `statistic` over the resamples `idx`, one per
    method: the replicates read at the (1 - confidence) level, as is or BCa-adjusted. The
    statement "the population value is at least this" holds with probability `confidence`
    when the method's coverage is nominal. The replicates are computed once for all methods.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    replicates = _REPLICATES[statistic]
    values = replicates(a_codes, b_codes, n_labels, idx)
    out: dict[Method, float] = {}
    for method in methods:
        estimate: float | None = None
        jack: FloatArray | None = None
        if method == "bca":
            full = np.arange(a_codes.size, dtype=np.int64)[None, :]
            estimate = float(replicates(a_codes, b_codes, n_labels, full)[0])
            jack = _JACKKNIFE[statistic](a_codes, b_codes, n_labels)
        bound = endpoints(values, [1 - confidence], method, estimate=estimate, jackknife=jack)
        out[method] = float(bound[0])
    return out


def encode(a: Sequence[str], b: Sequence[str]) -> tuple[IntArray, IntArray, int]:
    """Integer codes for both raters over the sorted union of their labels."""
    (a_codes, b_codes), k = encode_all(a, b)
    return a_codes, b_codes, k


def encode_all(*raters: Sequence[str]) -> tuple[list[IntArray], int]:
    """Integer codes for any number of equal-length raters over the sorted union of their
    labels, so replicates of different pairs can be compared code for code."""
    lengths = {len(r) for r in raters}
    if len(lengths) != 1:
        raise ValueError(f"rater lengths differ: {sorted(len(r) for r in raters)}")
    labels = sorted(set().union(*(set(r) for r in raters)))
    index = {label: i for i, label in enumerate(labels)}
    codes = [np.fromiter((index[x] for x in r), dtype=np.int64, count=len(r)) for r in raters]
    return codes, len(labels)


def bootstrap_ci(
    a: Sequence[str],
    b: Sequence[str],
    statistic: Statistic,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    method: Method = "percentile",
) -> Interval:
    """Bootstrap interval for `statistic` between raters `a` and `b`.

    estimate is the statistic on the full sample; low and high are the replicates read at the
    (1 - c) / 2 and (1 + c) / 2 levels (np.percentile, linear interpolation), BCa-adjusted when
    `method` is "bca".
    """
    if statistic not in _REPLICATES:
        raise ValueError(f"unknown statistic {statistic!r}; use 'kappa' or 'accuracy'")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    a_codes, b_codes, k = encode(a, b)
    n = len(a_codes)
    if n == 0:
        raise ValueError("bootstrap needs at least one item")
    replicates = _REPLICATES[statistic]
    estimate = float(replicates(a_codes, b_codes, k, np.arange(n, dtype=np.int64)[None, :])[0])
    values = replicates(a_codes, b_codes, k, resample_indices(n, n_resamples, seed))
    jack = _JACKKNIFE[statistic](a_codes, b_codes, k) if method == "bca" else None
    low, high = endpoints(
        values,
        [(1 - confidence) / 2, (1 + confidence) / 2],
        method,
        estimate=estimate,
        jackknife=jack,
    )
    return Interval(
        estimate=estimate,
        low=float(low),
        high=float(high),
        confidence=confidence,
        n_resamples=n_resamples,
        method=method,
    )


def bootstrap_difference_ci(
    reference: Sequence[str],
    a: Sequence[str],
    b: Sequence[str],
    statistic: Statistic,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    method: Method = "percentile",
) -> Interval:
    """Paired bootstrap interval for statistic(reference, a) - statistic(reference, b).

    Two students scored on the same held-out items are not independent samples, so comparing
    their separate intervals is only a loose test: overlapping intervals do not mean the two
    are tied. Here every resample draws one set of items and scores both students on it, so
    the interval is on the difference itself; it excludes 0 when the paired data separate the
    two students. Same seed as `bootstrap_ci`, so it uses the same item resamples. For BCa
    the jackknife of the difference is the difference of the two jackknifes.
    """
    if statistic not in _REPLICATES:
        raise ValueError(f"unknown statistic {statistic!r}; use 'kappa' or 'accuracy'")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    (ref_codes, a_codes, b_codes), k = encode_all(reference, a, b)
    n = len(ref_codes)
    if n == 0:
        raise ValueError("bootstrap needs at least one item")
    replicates = _REPLICATES[statistic]
    full = np.arange(n, dtype=np.int64)[None, :]
    estimate = float(
        replicates(ref_codes, a_codes, k, full)[0] - replicates(ref_codes, b_codes, k, full)[0]
    )
    idx = resample_indices(n, n_resamples, seed)
    values = replicates(ref_codes, a_codes, k, idx) - replicates(ref_codes, b_codes, k, idx)
    jack: FloatArray | None = None
    if method == "bca":
        jackknife = _JACKKNIFE[statistic]
        jack = jackknife(ref_codes, a_codes, k) - jackknife(ref_codes, b_codes, k)
    low, high = endpoints(
        values,
        [(1 - confidence) / 2, (1 + confidence) / 2],
        method,
        estimate=estimate,
        jackknife=jack,
    )
    return Interval(
        estimate=estimate,
        low=float(low),
        high=float(high),
        confidence=confidence,
        n_resamples=n_resamples,
        method=method,
    )
