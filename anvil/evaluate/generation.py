"""Agreement metrics for generation students: the student's output fields against the
teacher's.

A generate task's output is a dict of string fields (`{"sql": "SELECT ..."}`); a student
completion that could not be parsed into one is None. Agreement here is string agreement:
a field matches when both values are equal after `str.strip()` (models pad completions with
whitespace and newlines that carry no meaning), and an item is an exact match when every
field matches. An unparseable prediction matches nothing but stays in the denominator, so
a student cannot raise its rate by failing to answer the hard items. Task-specific
semantic metrics (execution match for SQL, in `sql.py`) are reported next to these.

Intervals are percentile bootstraps of a mean over items, with the same seeded
`resample_indices` as `bootstrap.py`, so for one seed every interval in the project draws
the same item resamples. `paired_difference_ci` scores two students on one shared set of
resampled items per replicate, as `bootstrap_difference_ci` does for classification.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from anvil.evaluate.bootstrap import Interval, resample_indices
from anvil.evaluate.slices import by_length

FloatArray = npt.NDArray[np.float64]


def _as_vector(values: Sequence[bool] | Sequence[float], name: str) -> FloatArray:
    out = np.asarray(values, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a flat sequence of numbers, got shape {out.shape}")
    if out.size == 0:
        raise ValueError(f"{name} is empty: a mean needs at least one item")
    return out


def _bootstrap_mean(
    values: FloatArray, n_resamples: int, confidence: float, seed: int
) -> Interval:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    replicates = values[resample_indices(len(values), n_resamples, seed)].mean(axis=1)
    low, high = np.percentile(replicates, [100 * (1 - confidence) / 2, 100 * (1 + confidence) / 2])
    return Interval(
        estimate=float(values.mean()),
        low=float(low),
        high=float(high),
        confidence=confidence,
        n_resamples=n_resamples,
    )


def mean_ci(
    values: Sequence[bool] | Sequence[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for the mean of `values` (a match rate when they are
    booleans). estimate is the plain mean of all values."""
    return _bootstrap_mean(_as_vector(values, "values"), n_resamples, confidence, seed)


def paired_difference_ci(
    a: Sequence[bool] | Sequence[float],
    b: Sequence[bool] | Sequence[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Paired percentile bootstrap interval for mean(a) - mean(b), where a[i] and b[i] score
    two students on the same item i.

    Every replicate draws one set of items and takes both means over it, which is the mean
    of the per-item differences a[i] - b[i] over that set; the interval is on the difference
    itself and excludes 0 when the paired data separate the two students.
    """
    va, vb = _as_vector(a, "a"), _as_vector(b, "b")
    if va.shape != vb.shape:
        raise ValueError(f"a and b must score the same items: {len(va)} vs {len(vb)} values")
    return _bootstrap_mean(va - vb, n_resamples, confidence, seed)


@dataclass(frozen=True)
class FieldMatch:
    name: str
    match_rate: float
    """Fraction of all n items (invalid predictions included) where this field matches."""
    n: int


@dataclass(frozen=True)
class LengthSlice:
    low: int
    """Shortest input (in words) in the bucket."""
    high: int
    """Longest input (in words) in the bucket."""
    n: int
    match_rate: float
    """Exact-match rate within the bucket."""


@dataclass(frozen=True)
class GenerationAgreement:
    n: int
    exact_match: Interval
    """Fraction of items where every field matches, with its bootstrap interval."""
    field_match: list[FieldMatch]
    valid: int
    """Predictions that parsed into fields."""
    invalid: int
    """Predictions that are None; each counts as a mismatch on every field."""
    by_length: list[LengthSlice]
    """Exact match by input length, shortest first; empty when no lengths were given."""


def _field_matches(prediction: Mapping[str, str] | None, reference: str, name: str) -> bool:
    if prediction is None:
        return False
    value = prediction.get(name)
    # A field the student left out, or filled with a non-string, is a mismatch, not an error:
    # malformed student output is exactly what this metric has to count.
    return isinstance(value, str) and value.strip() == reference.strip()


def generation_agreement(
    reference: Sequence[Mapping[str, str]],
    predicted: Sequence[Mapping[str, str] | None],
    fields: Sequence[str],
    lengths: Sequence[int] | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
) -> GenerationAgreement:
    """Exact and per-field agreement of `predicted` with `reference` on `fields`.

    `lengths` are the inputs' word counts, for the length slices (quartiles, formed exactly
    as in `slices.by_length`). A reference missing a field raises ValueError: the teacher's
    outputs were validated against the task schema before they were stored, so a gap there
    is a bug upstream, not disagreement.
    """
    n = len(reference)
    if n == 0:
        raise ValueError("generation agreement needs at least one item")
    if len(predicted) != n:
        raise ValueError(f"one prediction per item is needed: {n} references, {len(predicted)}")
    if lengths is not None and len(lengths) != n:
        raise ValueError(f"one length per item is needed: {n} references, {len(lengths)}")
    if not fields:
        raise ValueError("fields is empty: name the output fields to compare")
    for i, ref in enumerate(reference):
        missing = [f for f in fields if not isinstance(ref.get(f), str)]
        if missing:
            raise ValueError(f"reference {i} has no string value for fields {missing}")

    per_field = np.array(
        [
            [_field_matches(pred, ref[f], f) for f in fields]
            for ref, pred in zip(reference, predicted, strict=True)
        ],
        dtype=np.bool_,
    ).reshape(n, len(fields))
    exact = per_field.all(axis=1)
    invalid = sum(1 for p in predicted if p is None)

    slices: list[LengthSlice] = []
    if lengths is not None:
        # slices.by_length scores label agreement between two raters; encoding each item's
        # exact match as agree / disagree against a constant reuses its bucketing (stable
        # sort, equal-count chunks) unchanged, so these slices line up with the
        # classification and extraction reports.
        agree = ["match" if m else "mismatch" for m in exact]
        slices = [
            LengthSlice(low=int(s.lo), high=int(s.hi), n=s.n, match_rate=s.agreement)
            for s in by_length(["match"] * n, agree, lengths)
        ]

    return GenerationAgreement(
        n=n,
        exact_match=mean_ci(exact.tolist(), n_resamples=n_resamples, seed=seed),
        field_match=[
            FieldMatch(name=f, match_rate=float(per_field[:, j].mean()), n=n)
            for j, f in enumerate(fields)
        ],
        valid=n - invalid,
        invalid=invalid,
        by_length=slices,
    )
