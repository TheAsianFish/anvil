"""Agreement metrics between the student and the teacher.

M1 has the point estimates; M2 adds the bootstrap CI, the teacher self-agreement ceiling and
error slices, verified against scikit-learn.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa: agreement corrected for the agreement two raters would reach by chance.

    kappa = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and p_e the agreement
    expected if both raters labelled independently with their own label frequencies. When
    p_e is 1 (both raters always use the same single label) kappa is undefined; this returns
    1.0 if they agree everywhere and 0.0 otherwise.
    """
    if len(a) != len(b):
        raise ValueError(f"rater lengths differ: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        raise ValueError("kappa needs at least one item")
    p_o = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    p_e = sum(ca[k] * cb[k] for k in ca.keys() & cb.keys()) / (n * n)
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def accuracy(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("accuracy needs two equal-length, non-empty sequences")
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def confusion(truth: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> list[list[int]]:
    """Rows are the teacher's label, columns the student's, in `labels` order."""
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(truth, pred, strict=True):
        matrix[index[t]][index[p]] += 1
    return matrix


@dataclass(frozen=True)
class Agreement:
    n: int
    kappa: float
    accuracy: float
    majority_label: str
    majority_accuracy: float
    """Accuracy of always predicting the most common training label (its kappa is 0)."""
    confusion: list[list[int]]


def agreement(
    teacher: Sequence[str],
    student: Sequence[str],
    labels: Sequence[str],
    train_labels: Sequence[str],
) -> Agreement:
    majority = Counter(train_labels).most_common(1)[0][0] if train_labels else labels[0]
    return Agreement(
        n=len(teacher),
        kappa=cohen_kappa(teacher, student),
        accuracy=accuracy(teacher, student),
        majority_label=majority,
        majority_accuracy=accuracy(teacher, [majority] * len(teacher)),
        confusion=confusion(teacher, student, labels),
    )
