from __future__ import annotations

import math
import random

import pytest
from sklearn.metrics import cohen_kappa_score

from anvil.evaluate.metrics import accuracy, agreement, cohen_kappa, confusion


def test_kappa_matches_sklearn() -> None:
    rng = random.Random(0)
    compared = 0
    for _ in range(200):
        n = rng.randint(5, 300)
        classes = [f"c{k}" for k in range(rng.randint(2, 6))]
        a = [rng.choice(classes) for _ in range(n)]
        # Correlate b with a half the time so kappa spans more than just ~0.
        b = [x if rng.random() < 0.5 else rng.choice(classes) for x in a]
        expected = float(cohen_kappa_score(a, b))
        if math.isnan(expected):
            continue
        assert cohen_kappa(a, b) == pytest.approx(expected, abs=1e-12)
        compared += 1
    assert compared >= 190


def test_kappa_perfect_and_chance() -> None:
    assert cohen_kappa(["a", "b", "a", "c"], ["a", "b", "a", "c"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)


def test_kappa_degenerate_single_label() -> None:
    # p_e == 1: both raters use one and the same label everywhere.
    assert cohen_kappa(["a"] * 5, ["a"] * 5) == 1.0
    # Two constant raters on different labels: p_e == 0, plain kappa formula gives 0.
    assert cohen_kappa(["a"] * 5, ["b"] * 5) == 0.0


def test_kappa_and_accuracy_reject_bad_input() -> None:
    with pytest.raises(ValueError, match="lengths differ"):
        cohen_kappa(["a"], ["a", "b"])
    with pytest.raises(ValueError, match="at least one"):
        cohen_kappa([], [])
    with pytest.raises(ValueError, match="equal-length"):
        accuracy(["a"], [])
    with pytest.raises(ValueError, match="non-empty"):
        accuracy([], [])


def test_accuracy() -> None:
    assert accuracy(["a", "b", "c", "a"], ["a", "b", "a", "b"]) == 0.5


def test_confusion_rows_are_teacher_columns_student() -> None:
    teacher = ["a", "a", "a", "b"]
    student = ["b", "b", "a", "b"]
    assert confusion(teacher, student, ["a", "b", "c"]) == [[1, 2, 0], [0, 1, 0], [0, 0, 0]]


def test_agreement_majority_baseline_from_train_labels() -> None:
    teacher = ["x", "y", "y", "y"]
    student = ["x", "y", "x", "y"]
    result = agreement(teacher, student, ["x", "y"], train_labels=["x", "x", "y"])
    assert result.n == 4
    assert result.majority_label == "x"
    assert result.majority_accuracy == 0.25
    assert result.accuracy == 0.75
    assert result.kappa == pytest.approx(cohen_kappa(teacher, student))
    assert result.confusion == [[1, 0], [1, 2]]
    assert agreement(teacher, student, ["y", "x"], train_labels=[]).majority_label == "y"
