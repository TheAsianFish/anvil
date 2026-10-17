"""The student interface: a small model trained to reproduce the teacher's labels."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from anvil.spec import StudentSpec
    from anvil.train.loop import TrainResult


class Classifier(Protocol):
    labels: list[str]

    def predict_proba(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Class probabilities, shape (len(texts), len(labels)), rows sum to 1."""
        ...

    def save(self, path: Path) -> None: ...


class Student(Classifier, Protocol):
    """A classifier that can be trained and reloaded: what the pipeline and the studies need."""

    def fit(
        self,
        texts: Sequence[str],
        targets: Sequence[int],
        spec: StudentSpec,
        metrics_path: Path | None = None,
    ) -> TrainResult:
        """Train in place on (text, label index) pairs; label indices point into `labels`."""
        ...

    @classmethod
    def load(cls, path: Path) -> Self: ...
