"""Error slices: where the student disagrees with the teacher.

Items are sorted by a key (the student's confidence, or the input length) and cut into
equal-count buckets, and each bucket reports its key range and the student's agreement
(accuracy) with the teacher. Equal-count buckets keep every bucket's agreement estimated
from about the same number of items, which equal-width buckets on skewed confidences do not.

Bucket formation, exactly: sort item positions by key with a stable sort (ties keep their
input order), then split that order with np.array_split into min(buckets, n) consecutive
chunks, whose sizes differ by at most one (the first n % buckets chunks get the extra item).
Tied keys may therefore straddle a bucket boundary, so adjacent buckets can share an endpoint.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Slice:
    lo: float
    """Smallest key in the bucket."""
    hi: float
    """Largest key in the bucket."""
    n: int
    agreement: float
    """Fraction of the bucket where the student's label equals the teacher's."""


def _slices(
    teacher: Sequence[str], student: Sequence[str], keys: Sequence[float], buckets: int
) -> list[Slice]:
    if not len(teacher) == len(student) == len(keys):
        raise ValueError(
            f"lengths differ: teacher {len(teacher)}, student {len(student)}, keys {len(keys)}"
        )
    if buckets < 1:
        raise ValueError(f"buckets must be positive, got {buckets}")
    n = len(keys)
    if n == 0:
        return []
    values = np.asarray(keys, dtype=np.float64)
    agree = np.fromiter(
        (t == s for t, s in zip(teacher, student, strict=True)), dtype=np.bool_, count=n
    )
    order = np.argsort(values, kind="stable")
    out = []
    for chunk in np.array_split(order, min(buckets, n)):
        out.append(
            Slice(
                lo=float(values[chunk].min()),
                hi=float(values[chunk].max()),
                n=len(chunk),
                agreement=float(agree[chunk].mean()),
            )
        )
    return out


def by_confidence(
    teacher: Sequence[str],
    student: Sequence[str],
    confidence: Sequence[float],
    buckets: int = 10,
) -> list[Slice]:
    """Agreement by the student's confidence (its max class probability), lowest first.

    Deciles by default. A well-behaved student agrees more in higher buckets; that slope is
    what the calibration threshold later exploits.
    """
    return _slices(teacher, student, confidence, buckets)


def by_length(
    teacher: Sequence[str],
    student: Sequence[str],
    lengths: Sequence[int],
    buckets: int = 4,
) -> list[Slice]:
    """Agreement by input length (the caller passes word counts), shortest first; quartiles."""
    return _slices(teacher, student, [float(x) for x in lengths], buckets)
