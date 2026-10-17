"""The acquisition strategies in `anvil.select`: the contract every strategy keeps, and an oracle
per strategy that pins down what it picks.

These tests are the specification for strategies that are declared but not written yet: a test
of an unwritten strategy is an expected failure (it must fail with `NotImplementedError`, nothing
else), and the marker switches off by itself once the strategy runs, so the same test then has
to pass.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from anvil.select import STRATEGIES, Pool, implemented, select_batch

FAST = settings(max_examples=50, deadline=None, derandomize=True, database=None)
"""Property tests: few examples, derandomized (the same examples every run), no example
database written to the working tree."""


def needs(*strategies: str) -> pytest.MarkDecorator:
    """Expected to fail with NotImplementedError while any of `strategies` is unwritten."""
    missing = [s for s in strategies if not implemented(s)]
    return pytest.mark.xfail(
        bool(missing),
        raises=NotImplementedError,
        strict=True,
        reason=f"not implemented yet: {', '.join(missing)}",
    )


EVERY = [pytest.param(s, marks=needs(s), id=s) for s in STRATEGIES]
"""Every strategy, each expected to fail until it is written."""
UNMARKED = [pytest.param(s, id=s) for s in STRATEGIES]
"""Every strategy, for behaviour `select_batch` settles before any strategy code runs."""


def random_pool(n: int, classes: int, seed: int, dim: int = 3) -> Pool:
    rng = np.random.default_rng(seed)
    return Pool(
        probs=rng.dirichlet(np.ones(classes), size=n),
        embeddings=rng.normal(size=(n, dim)),
    )


def uniform_pool(n: int, classes: int = 4, dim: int = 3) -> Pool:
    """Every row scores the same under every strategy: identical probabilities and embeddings."""
    return Pool(probs=np.full((n, classes), 1.0 / classes), embeddings=np.zeros((n, dim)))


def entropy(probs: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    p = np.clip(probs, 1e-12, 1.0)
    return np.asarray(-(probs * np.log(p)).sum(axis=1))


# The contract


def probe(strategy: str) -> None:
    """Fail at once while `strategy` is unwritten, instead of letting hypothesis spend seconds
    shrinking an expected failure."""
    select_batch(random_pool(4, 3, 0), 2, strategy, seed=0)


@pytest.mark.parametrize("strategy", EVERY)
def test_returns_min_k_n_distinct_indices_in_range_deterministically(strategy: str) -> None:
    probe(strategy)

    @FAST
    @given(
        n=st.integers(0, 30),
        classes=st.integers(2, 6),
        k=st.integers(-2, 40),
        pool_seed=st.integers(0, 2**16),
        seed=st.integers(0, 2**31),
    )
    def check(n: int, classes: int, k: int, pool_seed: int, seed: int) -> None:
        pool = random_pool(n, classes, pool_seed)
        picked = select_batch(pool, k, strategy, seed)
        assert len(picked) == max(0, min(k, n))
        assert len(set(picked)) == len(picked)
        assert all(isinstance(i, int) and 0 <= i < n for i in picked)
        assert select_batch(pool, k, strategy, seed) == picked

    check()


@pytest.mark.parametrize("strategy", UNMARKED)
@pytest.mark.parametrize("k", [0, -1, -10])
def test_non_positive_k_selects_nothing(strategy: str, k: int) -> None:
    assert select_batch(random_pool(10, 3, 0), k, strategy, seed=0) == []


@pytest.mark.parametrize("strategy", UNMARKED)
def test_empty_pool_selects_nothing(strategy: str) -> None:
    empty = Pool(probs=np.zeros((0, 3)), embeddings=np.zeros((0, 2)))
    assert select_batch(empty, 5, strategy, seed=0) == []


@pytest.mark.parametrize("strategy", EVERY)
def test_k_above_pool_size_selects_the_whole_pool(strategy: str) -> None:
    picked = select_batch(random_pool(7, 3, 1), 50, strategy, seed=0)
    assert sorted(picked) == list(range(7))


@pytest.mark.parametrize("strategy", EVERY)
def test_ties_are_broken_by_the_seed_not_by_position(strategy: str) -> None:
    pool = uniform_pool(20)
    picks = []
    for seed in range(10):
        picked = select_batch(pool, 5, strategy, seed)
        assert select_batch(pool, 5, strategy, seed) == picked
        assert len(set(picked)) == 5
        picks.append(frozenset(picked))
    assert len(set(picks)) > 1, "every seed picked the same tied rows"
    assert set().union(*picks) != set(range(5)), "ties went to the first rows of the pool"


@pytest.mark.parametrize("strategy", [p for p in EVERY if p.id in ("uncertainty", "margin")])
def test_permuting_the_pool_selects_the_same_items(strategy: str) -> None:
    probe(strategy)

    @FAST
    @given(
        n=st.integers(2, 25),
        classes=st.integers(2, 6),
        k=st.integers(1, 25),
        pool_seed=st.integers(0, 2**16),
        seed=st.integers(0, 2**31),
    )
    def check(n: int, classes: int, k: int, pool_seed: int, seed: int) -> None:
        # Dirichlet draws in float64 have distinct scores almost surely: no ties to break.
        pool = random_pool(n, classes, pool_seed)
        assert pool.embeddings is not None
        perm = np.random.default_rng(pool_seed + 1).permutation(n)
        permuted = Pool(probs=pool.probs[perm], embeddings=pool.embeddings[perm])
        picked = select_batch(pool, k, strategy, seed)
        picked_permuted = select_batch(permuted, k, strategy, seed)
        assert [int(perm[i]) for i in picked_permuted] == picked

    check()


# Oracles

ENTROPY_POOL = np.array(
    [
        [1.0, 0.0, 0.0],  # entropy 0
        [0.8, 0.1, 0.1],  # 0.639
        [1 / 3, 1 / 3, 1 / 3],  # 1.099, the most uncertain
        [0.5, 0.5, 0.0],  # 0.693
        [0.6, 0.3, 0.1],  # 0.898
    ]
)

MARGIN_POOL = np.array(
    [
        [0.90, 0.05, 0.05],  # margin 0.85
        [0.45, 0.44, 0.11],  # 0.01, the smallest
        [0.40, 0.30, 0.30],  # 0.10
        [0.60, 0.35, 0.05],  # 0.25
        [0.31, 0.36, 0.33],  # 0.03 (the top class need not be the first column)
    ]
)


@needs("uncertainty")
def test_uncertainty_picks_the_highest_entropy_rows_most_uncertain_first() -> None:
    assert np.argsort(-entropy(ENTROPY_POOL)).tolist()[:3] == [2, 4, 3]
    assert select_batch(Pool(ENTROPY_POOL), 3, "uncertainty", seed=0) == [2, 4, 3]
    assert select_batch(Pool(ENTROPY_POOL), 5, "uncertainty", seed=7) == [2, 4, 3, 1, 0]


@needs("margin")
def test_margin_picks_the_smallest_top_two_gaps_smallest_first() -> None:
    assert select_batch(Pool(MARGIN_POOL), 3, "margin", seed=0) == [1, 4, 2]
    assert select_batch(Pool(MARGIN_POOL), 5, "margin", seed=7) == [1, 4, 2, 3, 0]


@needs("uncertainty", "margin")
def test_uncertainty_and_margin_disagree_where_entropy_and_margin_do() -> None:
    classes = 10
    two_way = np.zeros(classes)
    two_way[:2] = 0.5  # margin 0, entropy ln 2 = 0.69
    spread = np.full(classes, 0.08)
    spread[0] = 0.28  # margin 0.20, entropy 2.2
    confident = np.full(classes, 0.01)
    confident[3] = 0.91
    probs = np.stack([confident, two_way, spread, confident])
    assert entropy(probs)[2] > entropy(probs)[1]
    pool = Pool(probs)
    assert select_batch(pool, 1, "uncertainty", seed=0) == [2]
    assert select_batch(pool, 1, "margin", seed=0) == [1]


def _near_uniform(rng: np.random.Generator, n: int, classes: int, peak: float) -> np.ndarray:
    """Rows that put `peak` on one random class and spread the rest, plus a little jitter."""
    probs = np.full((n, classes), (1 - peak) / (classes - 1))
    probs[np.arange(n), rng.integers(0, classes, size=n)] = peak
    probs += rng.uniform(0, 1e-3, size=probs.shape)
    return np.asarray(probs / probs.sum(axis=1, keepdims=True))


@needs("diverse")
@pytest.mark.parametrize("seed", range(5))
def test_diverse_takes_one_input_from_each_uncertain_cluster(seed: int) -> None:
    rng = np.random.default_rng(seed)
    classes, dim, size = 4, 5, 3
    # Two tight clusters of three highly uncertain inputs each. Cluster A is a touch more
    # uncertain, so a pure-uncertainty pick of two would take both from A. The clusters are
    # small so that a strategy which spreads only the few-times-k most uncertain candidates
    # (the contract's "among the most uncertain") still sees both of them.
    a = _near_uniform(rng, size, classes, peak=0.26)
    b = _near_uniform(rng, size, classes, peak=0.30)
    confident = _near_uniform(rng, 10, classes, peak=0.97)
    probs = np.concatenate([a, b, confident])
    centre_a, centre_b = np.full(dim, 5.0), np.full(dim, -5.0)
    embeddings = np.concatenate(
        [
            centre_a + rng.normal(scale=0.01, size=(size, dim)),
            centre_b + rng.normal(scale=0.01, size=(size, dim)),
            rng.normal(scale=2.0, size=(10, dim)),  # confident inputs elsewhere
        ]
    )
    order = rng.permutation(len(probs))  # no help from pool position
    picked = select_batch(
        Pool(probs=probs[order], embeddings=embeddings[order]), 2, "diverse", seed
    )
    clusters = sorted(min(int(order[i]) // size, 2) for i in picked)
    assert clusters == [0, 1]


@needs("diverse")
@pytest.mark.parametrize("seed", range(5))
def test_diverse_prefers_uncertain_inputs(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    n, classes, k = 60, 5, 3
    confident = rng.permutation(n)[:8]  # a clear minority, scattered among the rest
    probs = _near_uniform(rng, n, classes, peak=0.3)
    probs[confident] = _near_uniform(rng, len(confident), classes, peak=0.98)
    # Embeddings spread evenly over a grid, so coverage alone cannot single anything out.
    side = np.linspace(-1, 1, 10)
    grid = np.array([(x, y) for x in side for y in side[:6]])
    embeddings = grid[rng.permutation(len(grid))]
    picked = select_batch(Pool(probs=probs, embeddings=embeddings), k, "diverse", seed)
    assert len(picked) == k
    assert not set(picked) & {int(i) for i in confident}


def test_diverse_without_embeddings_is_an_error() -> None:
    with pytest.raises(ValueError, match="embeddings"):
        select_batch(Pool(probs=np.full((4, 2), 0.5)), 2, "diverse", seed=0)


def test_random_is_seeded_and_roughly_uniform() -> None:
    n, k, draws = 10, 3, 2000
    counts = np.zeros(n)
    pool = uniform_pool(n)
    for seed in range(draws):
        counts[select_batch(pool, k, "random", seed)] += 1
    expected = draws * k / n
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < 27.9  # the 0.999 quantile of chi-square with 9 degrees of freedom
    big = random_pool(100, 3, 0)
    assert select_batch(big, 10, "random", 0) == select_batch(big, 10, "random", 0)
    assert select_batch(big, 10, "random", 0) != select_batch(big, 10, "random", 1)


def test_random_is_implemented() -> None:
    assert implemented("random")


def test_unknown_strategy_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        select_batch(random_pool(3, 2, 0), 1, "entropy", seed=0)


@pytest.mark.parametrize(
    ("probs_shape", "embeddings_shape"),
    [((4,), None), ((2, 3, 4), None), ((4, 3), (5, 2)), ((4, 3), (4,)), ((4, 3), (4, 2, 1))],
)
def test_pool_rejects_bad_shapes(
    probs_shape: tuple[int, ...], embeddings_shape: tuple[int, ...] | None
) -> None:
    embeddings = None if embeddings_shape is None else np.zeros(embeddings_shape)
    with pytest.raises(ValueError, match="must be"):
        Pool(probs=np.zeros(probs_shape), embeddings=embeddings)
