from __future__ import annotations

import itertools

import pytest

from anvil.evaluate.slices import Slice, by_confidence, by_length


def test_confidence_deciles_hand_case() -> None:
    # 20 items, confidence 0.00..0.95; the student is wrong on the 6 least confident ones.
    conf = [i / 20 for i in range(20)][::-1]
    teacher = ["a"] * 20
    student = ["a"] * 14 + ["b"] * 6
    slices = by_confidence(teacher, student, conf)
    assert len(slices) == 10
    assert [s.n for s in slices] == [2] * 10
    assert [s.lo for s in slices] == pytest.approx([i / 10 for i in range(10)])
    assert [s.hi for s in slices] == pytest.approx([i / 10 + 0.05 for i in range(10)])
    assert [s.agreement for s in slices] == [0.0, 0.0, 0.0] + [1.0] * 7


def test_unequal_buckets_and_counts_sum_to_n() -> None:
    lengths = [5, 1, 9, 3, 7, 2, 8, 4, 6, 10, 11]
    teacher = ["x"] * 11
    student = ["x" if n % 2 else "y" for n in lengths]
    slices = by_length(teacher, student, lengths)
    # 11 items in 4 buckets: sizes 3, 3, 3, 2 over the sorted lengths 1..11.
    assert slices == [
        Slice(lo=1, hi=3, n=3, agreement=2 / 3),
        Slice(lo=4, hi=6, n=3, agreement=1 / 3),
        Slice(lo=7, hi=9, n=3, agreement=2 / 3),
        Slice(lo=10, hi=11, n=2, agreement=0.5),
    ]
    assert sum(s.n for s in slices) == 11
    assert all(a.hi <= b.lo for a, b in itertools.pairwise(slices))


def test_fewer_items_than_buckets() -> None:
    slices = by_confidence(["a", "b", "c"], ["a", "x", "c"], [0.9, 0.2, 0.5])
    assert slices == [
        Slice(lo=0.2, hi=0.2, n=1, agreement=0.0),
        Slice(lo=0.5, hi=0.5, n=1, agreement=1.0),
        Slice(lo=0.9, hi=0.9, n=1, agreement=1.0),
    ]
    assert by_confidence([], [], []) == []


def test_ties_keep_stable_order_across_a_boundary() -> None:
    # Four tied lengths split 2 / 2 in input order: the first two items land in bucket 0.
    teacher = ["a", "a", "a", "a"]
    student = ["a", "a", "b", "b"]
    slices = by_length(teacher, student, [3, 3, 3, 3], buckets=2)
    assert slices == [
        Slice(lo=3, hi=3, n=2, agreement=1.0),
        Slice(lo=3, hi=3, n=2, agreement=0.0),
    ]


def test_bad_input_raises() -> None:
    with pytest.raises(ValueError, match="lengths differ"):
        by_confidence(["a"], ["a"], [0.1, 0.2])
    with pytest.raises(ValueError, match="buckets must be positive"):
        by_length(["a"], ["a"], [1], buckets=0)
