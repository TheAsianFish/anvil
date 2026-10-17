"""The student interface: a small model trained to reproduce the teacher's labels."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt


class Classifier(Protocol):
    labels: list[str]

    def predict_proba(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Class probabilities, shape (len(texts), len(labels)), rows sum to 1."""
        ...

    def save(self, path: Path) -> None: ...
