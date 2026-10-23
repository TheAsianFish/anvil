"""Lower confidence bounds on kappa built on the agreement count, with no resampling.

On the m inputs a cascade answers, the student agrees with the teacher on A of them. If each
answered input agrees independently with probability p_o, A is Binomial(m, p_o), and a lower
bound on p_o comes straight from the binomial distribution:

- exact (Clopper and Pearson 1934): the p at which seeing A or more agreements is exactly as
  unlikely as the level alpha, i.e. the root of P(X >= A; m, p) = alpha. It inverts the
  binomial test itself, so its coverage is at least 1 - alpha at every p and every m (it is
  conservative because A is discrete: at most p it covers more than asked).
- wilson (Wilson 1927): the lower root of (p_hat - p)^2 = z^2 p (1 - p) / m, the normal
  approximation to that test with the variance at p rather than at p_hat. Close to nominal on
  average, but not guaranteed at every p.

Kappa is (p_o - p_e) / (1 - p_e), increasing in p_o, so a bound L on p_o maps to the bound
(L - p_e) / (1 - p_e) on kappa. The chance term p_e = sum_k r_k c_k / m^2 is computed from the
same m inputs and treated as known (the plug-in): its own sampling error is ignored, which
`eval/coverage.py` checks for directly (it would show as under-coverage at few labels and few
inputs, where p_e is largest and noisiest).

Everything is vectorised over a batch of (A, m) pairs: the binomial tail is a sum of
exp(log C(m, j) + j log p + (m - j) log(1 - p)) over j >= A with log C from a table of log
factorials, and the exact bound bisects on p for the whole batch at once (54 halvings put it
within 1e-16 of the root). The tail only sums the m - A + 1 terms above A, so a high-agreement
batch, the one a cascade cares about, is cheap.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Literal

import numpy as np
import numpy.typing as npt

CountMethod = Literal["exact", "wilson"]
IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

_BISECTION_STEPS = 54
_NORMAL = NormalDist()


def _log_factorials(n: int) -> FloatArray:
    out = np.zeros(n + 1, dtype=np.float64)
    if n > 0:
        out[1:] = np.cumsum(np.log(np.arange(1, n + 1, dtype=np.float64)))
    return out


def _counts(successes: npt.ArrayLike, trials: npt.ArrayLike) -> tuple[IntArray, IntArray]:
    k = np.atleast_1d(np.asarray(successes))
    n = np.atleast_1d(np.asarray(trials))
    if k.size and not np.issubdtype(k.dtype, np.integer):
        raise ValueError(f"successes must be integer counts, got dtype {k.dtype}")
    if n.size and not np.issubdtype(n.dtype, np.integer):
        raise ValueError(f"trials must be integer counts, got dtype {n.dtype}")
    k64, n64 = np.broadcast_arrays(k.astype(np.int64), n.astype(np.int64))
    if k64.ndim != 1:
        raise ValueError(f"successes and trials must be scalars or 1-d, got {k64.shape}")
    if k64.size and (n64.min() < 1 or k64.min() < 0 or (k64 > n64).any()):
        raise ValueError("counts must satisfy 0 <= successes <= trials and trials >= 1")
    return k64.copy(), n64.copy()


def _check_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


def binomial_upper_tail(
    successes: npt.ArrayLike, trials: npt.ArrayLike, p: npt.ArrayLike
) -> FloatArray:
    """P(X >= successes) for X ~ Binomial(trials, p), elementwise over the broadcast batch.

    p must lie in [0, 1]. The tail is 1 at successes == 0 and, for successes > 0, 0 at p == 0
    and 1 at p == 1.
    """
    k, n = _counts(successes, trials)
    prob = np.broadcast_to(np.asarray(p, dtype=np.float64), k.shape).astype(np.float64)
    if prob.size and (not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1):
        raise ValueError("p must be probabilities in [0, 1]")
    out = np.ones(k.shape, dtype=np.float64)
    inner = (k > 0) & (prob > 0) & (prob < 1)
    out[(k > 0) & (prob == 0)] = 0.0
    if inner.any():
        out[inner] = _tail(k[inner], n[inner], prob[inner])
    return out


def binomial_pmf(trials: int, p: npt.ArrayLike) -> FloatArray:
    """P(X = k) for X ~ Binomial(trials, p) at every k in 0..trials (columns) and every p in
    the 1-d `p` (rows), from log factorials; p must lie in [0, 1]."""
    if trials < 1:
        raise ValueError(f"trials must be positive, got {trials}")
    prob = np.atleast_1d(np.asarray(p, dtype=np.float64))
    if prob.ndim != 1 or not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1:
        raise ValueError("p must be a 1-d array of probabilities in [0, 1]")
    k = np.arange(trials + 1, dtype=np.float64)[None, :]
    lf = _log_factorials(trials)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_pmf = (
            lf[trials]
            - lf[None, :]
            - lf[::-1][None, :]
            + k * np.log(prob)[:, None]
            + (trials - k) * np.log1p(-prob)[:, None]
        )
    out: FloatArray = np.exp(log_pmf)
    # 0 * log(0) is 0 here: at p == 0 all the mass is on k == 0, at p == 1 on k == trials.
    out[prob == 0] = np.eye(1, trials + 1, 0)[0]
    out[prob == 1] = np.eye(1, trials + 1, trials)[0]
    return out


def _tail(k: IntArray, n: IntArray, p: FloatArray) -> FloatArray:
    """The tail for 0 < k <= n and 0 < p < 1: a padded (batch, n - k + 1) sum of the pmf."""
    width = int((n - k).max()) + 1
    j = k[:, None] + np.arange(width, dtype=np.int64)[None, :]
    valid = j <= n[:, None]
    j = np.minimum(j, n[:, None])
    lf = _log_factorials(int(n.max()))
    log_pmf = (
        lf[n][:, None]
        - lf[j]
        - lf[n[:, None] - j]
        + j * np.log(p)[:, None]
        + (n[:, None] - j) * np.log1p(-p)[:, None]
    )
    out: FloatArray = np.where(valid, np.exp(log_pmf), 0.0).sum(axis=1)
    return np.minimum(out, 1.0)


def clopper_pearson_lower(
    successes: npt.ArrayLike, trials: npt.ArrayLike, confidence: float = 0.95
) -> FloatArray:
    """One-sided exact lower bound on the success probability: the root in p of
    P(X >= successes; trials, p) = 1 - confidence, and 0 when successes == 0.

    The tail is continuous and strictly increasing in p for successes > 0 (0 at p = 0, 1 at
    p = 1), so the root is unique; bisection keeps it bracketed. At successes == trials it is
    (1 - confidence) ** (1 / trials) in closed form, which the bisection reproduces.
    """
    _check_confidence(confidence)
    k, n = _counts(successes, trials)
    alpha = 1.0 - confidence
    lo = np.zeros(k.shape, dtype=np.float64)
    hi = np.ones(k.shape, dtype=np.float64)
    live = k > 0
    if live.any():
        kl, nl = k[live], n[live]
        a, b = lo[live], hi[live]
        for _ in range(_BISECTION_STEPS):
            mid = (a + b) / 2
            below = _tail(kl, nl, mid) < alpha  # the root is above mid
            a = np.where(below, mid, a)
            b = np.where(below, b, mid)
        lo[live] = a
    return lo


def wilson_lower(
    successes: npt.ArrayLike, trials: npt.ArrayLike, confidence: float = 0.95
) -> FloatArray:
    """One-sided Wilson score lower bound with z = Phi^-1(confidence):

        (p_hat + z^2 / 2m - z * sqrt(p_hat (1 - p_hat) / m + z^2 / 4m^2)) / (1 + z^2 / m)

    and 0 when successes == 0 (where the formula gives 0 up to rounding)."""
    _check_confidence(confidence)
    k, n = _counts(successes, trials)
    z = _NORMAL.inv_cdf(confidence)
    m = n.astype(np.float64)
    p_hat = k / m
    centre = p_hat + z * z / (2 * m)
    spread = z * np.sqrt(p_hat * (1 - p_hat) / m + z * z / (4 * m * m))
    out: FloatArray = np.where(k == 0, 0.0, np.maximum((centre - spread) / (1 + z * z / m), 0.0))
    return out


COUNT_BOUNDS = {"exact": clopper_pearson_lower, "wilson": wilson_lower}


def kappa_lower_from_counts(
    agree: npt.ArrayLike,
    chance: npt.ArrayLike,
    n: npt.ArrayLike,
    method: CountMethod,
    confidence: float = 0.95,
) -> FloatArray:
    """Lower bound on kappa from the agreement count `agree` of `n` items and the plug-in
    chance term, given as the integer `chance` = n^2 * p_e = sum_k r_k * c_k of the two
    raters' label counts on those items: (L - p_e) / (1 - p_e) for the `method`'s bound L on
    p_o.

    When p_e == 1 (both raters give every item the same single label) kappa is 0/0 and has no
    chance-corrected evidence to bound; the bound is then 0.0, the chance level, whatever the
    point estimate's convention (1.0) says.
    """
    if method not in COUNT_BOUNDS:
        raise ValueError(f"unknown count method {method!r}; use 'exact' or 'wilson'")
    a, m = _counts(agree, n)
    c = np.broadcast_to(np.atleast_1d(np.asarray(chance)), a.shape)
    if c.size and not np.issubdtype(c.dtype, np.integer):
        raise ValueError(f"chance must be an integer count, got dtype {c.dtype}")
    c64 = c.astype(np.int64)
    if c64.size and (c64.min() < 0 or (c64 > m * m).any()):
        raise ValueError("chance must lie in [0, n^2]")
    low = COUNT_BOUNDS[method](a, m, confidence)
    mf = m.astype(np.float64)
    p_e = c64 / (mf * mf)
    degenerate = c64 == m * m
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = (low - p_e) / (1 - p_e)
    out: FloatArray = np.where(degenerate, 0.0, kappa)
    return out


def kappa_lower_bound(
    a_codes: IntArray,
    b_codes: IntArray,
    n_labels: int,
    method: CountMethod,
    confidence: float = 0.95,
) -> float:
    """`kappa_lower_from_counts` on two raters' integer label codes in [0, n_labels)."""
    if a_codes.shape != b_codes.shape or a_codes.ndim != 1 or a_codes.size < 1:
        raise ValueError(
            f"rater codes must be 1-d, non-empty and equal length: {a_codes.shape} vs"
            f" {b_codes.shape}"
        )
    if min(a_codes.min(), b_codes.min()) < 0 or max(a_codes.max(), b_codes.max()) >= n_labels:
        raise ValueError(f"label codes must lie in [0, {n_labels})")
    rows = np.bincount(a_codes, minlength=n_labels)
    cols = np.bincount(b_codes, minlength=n_labels)
    agree = int(np.count_nonzero(a_codes == b_codes))
    chance = int((rows * cols).sum())
    return float(kappa_lower_from_counts(agree, chance, a_codes.size, method, confidence)[0])
