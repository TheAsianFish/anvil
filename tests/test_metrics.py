from __future__ import annotations

import json
import math
import random
import warnings
from dataclasses import asdict

import pytest
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from anvil.evaluate.bootstrap import bootstrap_ci
from anvil.evaluate.metrics import (
    Confusion,
    accuracy,
    agreement,
    cohen_kappa,
    confusion,
    macro_f1,
    per_class,
    self_agreement,
    top_confusions,
)


def random_case(rng: random.Random) -> tuple[list[str], list[str], list[str]]:
    """Two correlated raters; b sometimes uses a label a never does, and vice versa."""
    n = rng.randint(1, 120)
    classes = [f"c{k}" for k in range(rng.randint(1, 8))]
    a = [rng.choice(classes) for _ in range(n)]
    b = [x if rng.random() < 0.6 else rng.choice([*classes, "extra"]) for x in a]
    return a, b, [*classes, "extra"]


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


def test_macro_f1_matches_sklearn() -> None:
    rng = random.Random(1)
    for _ in range(300):
        a, b, _ = random_case(rng)
        expected = float(f1_score(a, b, average="macro", zero_division=0))
        assert macro_f1(a, b) == pytest.approx(expected, abs=1e-12)


def test_macro_f1_hand_case() -> None:
    # x: p=1/2 r=1 f1=2/3; y: p=1 r=1/2 f1=2/3; z (student only): 0.
    assert macro_f1(["x", "y", "y"], ["x", "y", "x"]) == pytest.approx(2 / 3)
    assert macro_f1(["x", "x"], ["x", "z"]) == pytest.approx((2 / 3 + 0) / 2)
    with pytest.raises(ValueError, match="non-empty"):
        macro_f1([], [])


def test_per_class_matches_sklearn() -> None:
    rng = random.Random(2)
    for _ in range(200):
        a, b, labels = random_case(rng)
        rng.shuffle(labels)
        p, r, f, support = precision_recall_fscore_support(a, b, labels=labels, zero_division=0)
        rows = per_class(a, b, labels)
        assert [row.label for row in rows] == labels
        for i, row in enumerate(rows):
            assert row.precision == pytest.approx(p[i], abs=1e-12)
            assert row.recall == pytest.approx(r[i], abs=1e-12)
            assert row.f1 == pytest.approx(f[i], abs=1e-12)
            assert row.support == support[i]
            assert row.predicted == b.count(row.label)
            ta = [x == row.label for x in a]
            tb = [x == row.label for x in b]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # undefined (p_e == 1) cases are handled below
                expected = float(cohen_kappa_score(ta, tb))
            if row.support == 0 and row.predicted == 0:
                assert math.isnan(expected)
                assert row.kappa is None  # nothing to agree on
                continue
            if math.isnan(expected):  # both always True: p_e == 1 convention
                expected = 1.0 if ta == tb else 0.0
            assert row.kappa == pytest.approx(expected, abs=1e-12)


def test_confusion_matches_sklearn() -> None:
    rng = random.Random(3)
    for _ in range(100):
        a, b, labels = random_case(rng)
        assert confusion(a, b, labels) == confusion_matrix(a, b, labels=labels).tolist()


def test_top_confusions_hand_case() -> None:
    teacher = ["a", "a", "a", "b", "b", "c", "c", "c"]
    student = ["b", "b", "a", "a", "c", "a", "b", "c"]
    assert top_confusions(teacher, student) == [
        Confusion("a", "b", 2),
        Confusion("b", "a", 1),
        Confusion("b", "c", 1),
        Confusion("c", "a", 1),
        Confusion("c", "b", 1),
    ]
    assert top_confusions(teacher, student, k=2) == [
        Confusion("a", "b", 2),
        Confusion("b", "a", 1),
    ]
    assert top_confusions(["a"], ["a"]) == []


def test_agreement_report_fields_and_json() -> None:
    rng = random.Random(4)
    labels = ["p", "q", "r"]
    teacher = [rng.choice(labels) for _ in range(60)]
    student = [t if rng.random() < 0.8 else rng.choice(labels) for t in teacher]
    conf = [rng.random() for _ in teacher]
    lengths = [rng.randint(1, 40) for _ in teacher]
    result = agreement(
        teacher, student, labels, teacher, confidence=conf, lengths=lengths, n_resamples=200
    )
    assert result.kappa_ci == bootstrap_ci(teacher, student, "kappa", n_resamples=200)
    assert result.kappa_ci.estimate == pytest.approx(result.kappa)
    assert result.kappa_ci.low <= result.kappa <= result.kappa_ci.high
    assert result.accuracy_ci.estimate == pytest.approx(result.accuracy)
    assert result.accuracy_ci.n_resamples == 200
    assert result.macro_f1 == pytest.approx(macro_f1(teacher, student))
    assert [c.label for c in result.per_class] == labels
    assert result.top_confusions == top_confusions(teacher, student)
    assert len(result.by_confidence) == 10
    assert sum(s.n for s in result.by_confidence) == 60
    assert len(result.by_length) == 4
    payload = json.loads(json.dumps(asdict(result)))
    assert payload["kappa_ci"]["method"] == "percentile"
    assert payload["per_class"][0]["label"] == "p"

    bare = agreement(teacher, student, labels, [], n_resamples=50)
    assert (bare.by_confidence, bare.by_length) == ([], [])


def test_self_agreement_uses_the_id_intersection() -> None:
    first = {"1": "a", "2": "b", "3": "a", "4": "b", "only-first": "a"}
    second = {"1": "a", "2": "b", "3": "b", "4": "b", "only-second": "b"}
    ceiling = self_agreement(first, second, n_resamples=300, seed=7)
    a, b = ["a", "b", "a", "b"], ["a", "b", "b", "b"]
    assert ceiling.n == 4
    assert ceiling.accuracy == 0.75
    assert ceiling.kappa == pytest.approx(float(cohen_kappa_score(a, b)))
    assert ceiling.kappa_ci == bootstrap_ci(a, b, "kappa", n_resamples=300, seed=7)
    assert ceiling.kappa_ci.low <= ceiling.kappa <= ceiling.kappa_ci.high


def test_self_agreement_without_shared_ids_raises() -> None:
    with pytest.raises(ValueError, match="share no input ids"):
        self_agreement({"1": "a"}, {"2": "a"})


def test_a_label_outside_the_task_set_is_a_specific_error() -> None:
    with pytest.raises(ValueError, match=r"outside the task's label set.*zzz"):
        agreement(["a", "b"], ["a", "zzz"], ["a", "b"], ["a"])
