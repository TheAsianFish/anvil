from __future__ import annotations

import numpy as np
import pytest

from anvil.evaluate.bootstrap import resample_indices
from anvil.evaluate.generation import (
    FieldMatch,
    generation_agreement,
    mean_ci,
    paired_difference_ci,
)


def test_mean_ci_matches_a_hand_computed_bootstrap() -> None:
    values = [True, False, True, True, False, True, True, False, True, True]
    interval = mean_ci(values, n_resamples=500, seed=7)
    v = np.asarray(values, dtype=np.float64)
    replicates = v[resample_indices(len(v), 500, 7)].mean(axis=1)
    low, high = np.percentile(replicates, [2.5, 97.5])
    assert interval.estimate == pytest.approx(0.7)
    assert interval.low == pytest.approx(low)
    assert interval.high == pytest.approx(high)
    assert interval.low <= interval.estimate <= interval.high
    assert (interval.confidence, interval.n_resamples) == (0.95, 500)


def test_mean_ci_estimate_is_the_plain_mean_and_is_seeded() -> None:
    values = [0.1, 0.4, 0.9, 0.3]
    assert mean_ci(values).estimate == pytest.approx(np.mean(values))
    assert mean_ci(values, seed=3) == mean_ci(values, seed=3)


@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5])
def test_mean_ci_rejects_bad_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        mean_ci([1.0], confidence=confidence)


def test_mean_ci_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        mean_ci([])


def test_paired_difference_sign_and_pairing() -> None:
    a = [True] * 8 + [False] * 2
    b = [True] * 4 + [False] * 6
    interval = paired_difference_ci(a, b, seed=1)
    assert interval.estimate == pytest.approx(0.4)
    assert interval.low > 0
    reverse = paired_difference_ci(b, a, seed=1)
    assert reverse.estimate == pytest.approx(-0.4)
    assert reverse.high == pytest.approx(-interval.low)


def test_paired_difference_of_identical_vectors_is_exactly_zero() -> None:
    a = [True, False, True, False, False]
    interval = paired_difference_ci(a, a)
    assert (interval.estimate, interval.low, interval.high) == (0.0, 0.0, 0.0)


def test_paired_difference_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="same items"):
        paired_difference_ci([1.0, 0.0], [1.0])


def test_generation_agreement_counts_and_rates() -> None:
    reference = [
        {"sql": "SELECT 1", "note": "a"},
        {"sql": "SELECT 2", "note": "b"},
        {"sql": "SELECT 3", "note": "c"},
        {"sql": "SELECT 4", "note": "d"},
    ]
    predicted: list[dict[str, str] | None] = [
        {"sql": "  SELECT 1\n", "note": "a "},  # whitespace only: a match
        {"sql": "SELECT 2", "note": "x"},  # one field wrong
        None,  # unparseable: a mismatch everywhere
        {"note": "d"},  # field missing: a mismatch on it
    ]
    result = generation_agreement(reference, predicted, ["sql", "note"], n_resamples=200)
    assert result.n == 4
    assert result.exact_match.estimate == pytest.approx(0.25)
    assert result.field_match == [
        FieldMatch(name="sql", match_rate=0.5, n=4),
        FieldMatch(name="note", match_rate=0.5, n=4),
    ]
    assert (result.valid, result.invalid) == (3, 1)
    assert result.by_length == []


def test_generation_agreement_length_slices() -> None:
    reference = [{"sql": str(i)} for i in range(8)]
    predicted: list[dict[str, str] | None] = [{"sql": str(i)} for i in range(4)] + [None] * 4
    lengths = [1, 2, 3, 4, 5, 6, 7, 8]
    result = generation_agreement(reference, predicted, ["sql"], lengths=lengths)
    assert [(s.low, s.high, s.n, s.match_rate) for s in result.by_length] == [
        (1, 2, 2, 1.0),
        (3, 4, 2, 1.0),
        (5, 6, 2, 0.0),
        (7, 8, 2, 0.0),
    ]


def test_generation_agreement_rejects_bad_input() -> None:
    ref = [{"sql": "SELECT 1"}]
    with pytest.raises(ValueError, match="at least one"):
        generation_agreement([], [], ["sql"])
    with pytest.raises(ValueError, match="one prediction per item"):
        generation_agreement(ref, [None, None], ["sql"])
    with pytest.raises(ValueError, match="one length per item"):
        generation_agreement(ref, [None], ["sql"], lengths=[1, 2])
    with pytest.raises(ValueError, match="fields is empty"):
        generation_agreement(ref, [None], [])
    with pytest.raises(ValueError, match="reference 0"):
        generation_agreement(ref, [None], ["sql", "missing"])
