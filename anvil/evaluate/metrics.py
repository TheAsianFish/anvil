"""Agreement metrics between the student and the teacher.

Point estimates (kappa, accuracy, macro F1, per-class stats, confusions) live here, together
with the `Agreement` report that adds bootstrap CIs (`bootstrap.py`) and error slices
(`slices.py`), and the teacher self-agreement ceiling. Every statistic is checked against
scikit-learn in the tests; the runtime code only needs numpy.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from anvil.evaluate.bootstrap import Interval, bootstrap_ci
from anvil.evaluate.slices import Slice, by_confidence, by_length


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
    unknown = sorted({x for x in (*truth, *pred) if x not in index})
    if unknown:
        raise ValueError(
            f"labels outside the task's label set: {unknown[:5]}; the teacher and the student"
            " must both use the labels in the spec"
        )
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(truth, pred, strict=True):
        matrix[index[t]][index[p]] += 1
    return matrix


def _f1(precision: float, recall: float) -> float:
    total = precision + recall
    return 2 * precision * recall / total if total else 0.0


def _ratio(num: int, den: int) -> float:
    """num / den with 0.0 for an empty denominator (scikit-learn's zero_division=0)."""
    return num / den if den else 0.0


@dataclass(frozen=True)
class ClassStats:
    """One label's row of the per-class table; the teacher's labels are the reference."""

    label: str
    support: int
    """Items the teacher gave this label."""
    predicted: int
    """Items the student gave this label."""
    precision: float
    recall: float
    f1: float
    kappa: float | None
    """One-vs-rest kappa: Cohen's kappa of the binary "is this label" ratings; None when
    neither rater used the label (nothing to agree on, where sklearn reports nan)."""


def per_class(
    truth: Sequence[str], pred: Sequence[str], labels: Sequence[str]
) -> list[ClassStats]:
    """Precision, recall, F1 and one-vs-rest kappa for each label, rows in `labels` order.

    With tp the items both gave the label: precision = tp / predicted, recall = tp / support,
    f1 = 2 * precision * recall / (precision + recall); any 0/0 is 0 (zero_division=0). The
    one-vs-rest kappa is None for a label neither rater used.
    """
    if len(truth) != len(pred):
        raise ValueError(f"rater lengths differ: {len(truth)} vs {len(pred)}")
    support, predicted = Counter(truth), Counter(pred)
    hits = Counter(t for t, p in zip(truth, pred, strict=True) if t == p)
    rows = []
    for label in labels:
        precision = _ratio(hits[label], predicted[label])
        recall = _ratio(hits[label], support[label])
        kappa = (
            cohen_kappa([str(t == label) for t in truth], [str(p == label) for p in pred])
            if support[label] or predicted[label]
            else None
        )
        rows.append(
            ClassStats(
                label=label,
                support=support[label],
                predicted=predicted[label],
                precision=precision,
                recall=recall,
                f1=_f1(precision, recall),
                kappa=kappa,
            )
        )
    return rows


def macro_f1(truth: Sequence[str], pred: Sequence[str]) -> float:
    """Unweighted mean of per-label F1 over the labels present in either sequence.

    This matches scikit-learn's f1_score(average="macro", zero_division=0) with labels=None:
    a label the student predicts but the teacher never uses counts with F1 0, while a label
    of the task that appears in neither sequence is left out of the mean rather than scored.
    Every present label weighs the same, so macro F1 is the imbalance-robust companion to
    accuracy.
    """
    if len(truth) != len(pred) or not truth:
        raise ValueError("macro F1 needs two equal-length, non-empty sequences")
    rows = per_class(truth, pred, sorted(set(truth) | set(pred)))
    return sum(r.f1 for r in rows) / len(rows)


@dataclass(frozen=True)
class Confusion:
    teacher: str
    student: str
    count: int


def top_confusions(truth: Sequence[str], pred: Sequence[str], k: int = 10) -> list[Confusion]:
    """The k most frequent (teacher label, student label) disagreements.

    Off-diagonal cells only, ordered by count descending, then teacher label, then student
    label, so ties are reported in a stable order.
    """
    cells = Counter((t, p) for t, p in zip(truth, pred, strict=True) if t != p)
    ordered = sorted(cells.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    return [Confusion(teacher=t, student=p, count=c) for (t, p), c in ordered[:k]]


@dataclass(frozen=True)
class Agreement:
    """Student-vs-teacher agreement on the held-out split; `asdict` of it is the eval report."""

    n: int
    kappa: float
    accuracy: float
    majority_label: str
    majority_accuracy: float
    """Accuracy of always predicting the most common training label (its kappa is 0)."""
    confusion: list[list[int]]
    kappa_ci: Interval
    accuracy_ci: Interval
    macro_f1: float
    per_class: list[ClassStats]
    top_confusions: list[Confusion]
    by_confidence: list[Slice]
    """Empty when no student confidences were given."""
    by_length: list[Slice]
    """Empty when no input lengths were given."""


def agreement(
    teacher: Sequence[str],
    student: Sequence[str],
    labels: Sequence[str],
    train_labels: Sequence[str],
    *,
    confidence: Sequence[float] | None = None,
    lengths: Sequence[int] | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
) -> Agreement:
    """The full agreement report: point estimates, 95% bootstrap CIs, per-class and slices.

    kappa and accuracy CIs use the same seed and so the same item resamples.
    """
    majority = Counter(train_labels).most_common(1)[0][0] if train_labels else labels[0]
    return Agreement(
        n=len(teacher),
        kappa=cohen_kappa(teacher, student),
        accuracy=accuracy(teacher, student),
        majority_label=majority,
        majority_accuracy=accuracy(teacher, [majority] * len(teacher)),
        confusion=confusion(teacher, student, labels),
        kappa_ci=bootstrap_ci(teacher, student, "kappa", n_resamples=n_resamples, seed=seed),
        accuracy_ci=bootstrap_ci(teacher, student, "accuracy", n_resamples=n_resamples, seed=seed),
        macro_f1=macro_f1(teacher, student),
        per_class=per_class(teacher, student, labels),
        top_confusions=top_confusions(teacher, student),
        by_confidence=[] if confidence is None else by_confidence(teacher, student, confidence),
        by_length=[] if lengths is None else by_length(teacher, student, lengths),
    )


@dataclass(frozen=True)
class Ceiling:
    """Teacher self-agreement on items it labelled twice (two prompt paraphrases)."""

    n: int
    kappa: float
    kappa_ci: Interval
    accuracy: float


def self_agreement(
    first: Mapping[str, str],
    second: Mapping[str, str],
    n_resamples: int = 1000,
    seed: int = 0,
) -> Ceiling:
    """The parity ceiling: how well the teacher agrees with itself, by input id.

    A student trained on one labelling cannot be expected to agree with the teacher more than
    a second labelling of the same inputs does, so the student's kappa is read against this
    one. Only ids present in both mappings count.
    """
    ids = sorted(first.keys() & second.keys())
    if not ids:
        raise ValueError("the two labellings share no input ids; label the same inputs twice")
    a = [first[i] for i in ids]
    b = [second[i] for i in ids]
    return Ceiling(
        n=len(ids),
        kappa=cohen_kappa(a, b),
        kappa_ci=bootstrap_ci(a, b, "kappa", n_resamples=n_resamples, seed=seed),
        accuracy=accuracy(a, b),
    )
