"""Percentile bootstrap confidence intervals for agreement statistics.

The unit of resampling is the item: each resample draws n item indices with replacement and
keeps both raters' labels for each drawn item together (paired resampling), so the interval
reflects how much the statistic would move on another held-out set of the same size drawn
from the same population. The 95% percentile interval is the 2.5th and 97.5th percentiles of
the statistic over the resamples (Efron and Tibshirani, ch. 13). The percentile method is
simple and cheap but not bias-corrected: for a statistic bounded at 1 it sits high when the
estimate is near 1 or n is small, so coverage is below nominal there (the tests check
coverage at kappa 0.7 with 200 items); BCa would be the upgrade if the ceiling or a
near-perfect student needs a tight interval.

Everything is vectorised over resamples: 1,000 resamples of 500 items with 77 labels take a
few milliseconds, so the CI is cheap enough to report on every eval and every loop round.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

Statistic = Literal["kappa", "accuracy"]
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
    matches = (a_codes == b_codes)[idx]
    out: FloatArray = matches.sum(axis=1) / idx.shape[1]
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
    b_r = b_codes[idx]
    rows = np.bincount((offsets + a_r).ravel(), minlength=r * k).reshape(r, k)
    cols = np.bincount((offsets + b_r).ravel(), minlength=r * k).reshape(r, k)
    agree = (a_r == b_r).sum(axis=1)
    chance = (rows * cols).sum(axis=1)  # integer: n^2 * p_e
    p_o = agree / n
    p_e = chance / (n * n)
    degenerate = chance == n * n
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = (p_o - p_e) / (1 - p_e)
    fallback = np.where(agree == n, 1.0, 0.0)
    out: FloatArray = np.where(degenerate, fallback, kappa)
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


_REPLICATES = {"kappa": kappa_replicates, "accuracy": accuracy_replicates}


def bootstrap_ci(
    a: Sequence[str],
    b: Sequence[str],
    statistic: Statistic,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for `statistic` between raters `a` and `b`.

    estimate is the statistic on the full sample; low and high are the (1 - c) / 2 and
    (1 + c) / 2 quantiles of the resampled statistics (np.percentile, linear interpolation).
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
    low, high = np.percentile(values, [100 * (1 - confidence) / 2, 100 * (1 + confidence) / 2])
    return Interval(
        estimate=estimate,
        low=float(low),
        high=float(high),
        confidence=confidence,
        n_resamples=n_resamples,
    )


def bootstrap_difference_ci(
    reference: Sequence[str],
    a: Sequence[str],
    b: Sequence[str],
    statistic: Statistic,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Paired percentile bootstrap interval for statistic(reference, a) - statistic(reference, b).

    Two students scored on the same held-out items are not independent samples, so comparing
    their separate intervals is only a loose test: overlapping intervals do not mean the two
    are tied. Here every resample draws one set of items and scores both students on it, so
    the interval is on the difference itself; it excludes 0 when the paired data separate the
    two students. Same seed as `bootstrap_ci`, so it uses the same item resamples.
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
    low, high = np.percentile(values, [100 * (1 - confidence) / 2, 100 * (1 + confidence) / 2])
    return Interval(
        estimate=estimate,
        low=float(low),
        high=float(high),
        confidence=confidence,
        n_resamples=n_resamples,
    )
