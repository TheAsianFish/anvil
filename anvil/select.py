"""Acquisition strategies: which unlabelled pool inputs to send to the teacher next.

`select_batch(pool, k, strategy, seed)` is the one entry point. The pool is what the current
student sees of the unlabelled inputs: its class probabilities and, for `diverse`, an
embedding per input. Every strategy follows the same contract, which the tests check for each:

- it returns `min(k, len(pool))` distinct indices into the pool, most wanted first;
- the same (pool, k, strategy, seed) always gives the same indices;
- ties (an untrained student scores every input the same) are broken by the seed, never by
  pool position, so a pool sorted by class or by source cannot bias the pick.

Strategies:

- `random`: a seeded uniform draw. The reference curve every other strategy is judged against.
- `uncertainty`: the highest predictive entropy, `-sum(p * log p)` over the classes.
- `margin`: the smallest gap between the two most probable classes.
- `diverse`: uncertain inputs that also cover the pool, so a batch is not k copies of one
  confusing input: among the most uncertain candidates (a few times k of them, say 5k), pick
  inputs spread across the embedding space (for example one per cluster of the candidates'
  embeddings).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

Strategy = Literal["random", "uncertainty", "margin", "diverse"]
STRATEGIES: tuple[Strategy, ...] = get_args(Strategy)

FloatArray = npt.NDArray[np.floating]


@dataclass(frozen=True)
class Pool:
    """The unlabelled candidates as the current student sees them."""

    probs: FloatArray
    """Shape (n, classes): the student's class probabilities, each row summing to 1."""
    embeddings: FloatArray | None = None
    """Shape (n, d): one vector per input (the student's pooled hidden state); `diverse`
    needs it, the other strategies ignore it."""

    def __post_init__(self) -> None:
        if self.probs.ndim != 2:
            raise ValueError(f"probs must be 2-D (n, classes), got shape {self.probs.shape}")
        if self.embeddings is not None and (
            self.embeddings.ndim != 2 or self.embeddings.shape[0] != self.probs.shape[0]
        ):
            raise ValueError(
                f"embeddings must be (n, d) with n = {self.probs.shape[0]},"
                f" got shape {self.embeddings.shape}"
            )

    def __len__(self) -> int:
        return int(self.probs.shape[0])


def select_batch(pool: Pool, k: int, strategy: str, seed: int) -> list[int]:
    """Indices into `pool` of the next `k` inputs to label, most wanted first."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose one of {', '.join(STRATEGIES)}")
    k = min(k, len(pool))
    if k <= 0:
        return []
    rng = np.random.default_rng(seed)
    if strategy == "random":
        return _random(pool, k, rng)
    if strategy == "uncertainty":
        return _uncertainty(pool, k, rng)
    if strategy == "margin":
        return _margin(pool, k, rng)
    if pool.embeddings is None:
        raise ValueError("the diverse strategy needs pool embeddings")
    return _diverse(pool, k, rng)


def _random(pool: Pool, k: int, rng: np.random.Generator) -> list[int]:
    picked: list[int] = rng.choice(len(pool), size=k, replace=False).tolist()
    return picked


def _uncertainty(pool: Pool, k: int, rng: np.random.Generator) -> list[int]:
    raise NotImplementedError("the uncertainty strategy is not implemented yet")


def _margin(pool: Pool, k: int, rng: np.random.Generator) -> list[int]:
    raise NotImplementedError("the margin strategy is not implemented yet")


def _diverse(pool: Pool, k: int, rng: np.random.Generator) -> list[int]:
    raise NotImplementedError("the diverse strategy is not implemented yet")


def implemented(strategy: str) -> bool:
    """Whether `strategy` runs (a strategy that is declared but not written yet does not)."""
    rng = np.random.default_rng(0)
    probe = Pool(
        probs=rng.dirichlet(np.ones(3), size=8).astype(np.float32),
        embeddings=rng.normal(size=(8, 4)).astype(np.float32),
    )
    try:
        select_batch(probe, 2, strategy, seed=0)
    except NotImplementedError:
        return False
    return True
