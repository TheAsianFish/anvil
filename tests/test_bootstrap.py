from __future__ import annotations

import math
import time
import warnings

import numpy as np
import numpy.typing as npt
import pytest
from scipy import stats  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, cohen_kappa_score

from anvil.evaluate.bootstrap import (
    Interval,
    accuracy_replicates,
    bootstrap_ci,
    encode,
    kappa_replicates,
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
    assert abs(ours.low - res.confidence_interval.low) < 0.02
    assert abs(ours.high - res.confidence_interval.high) < 0.02


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
    assert time.perf_counter() - start < 1.0


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
