"""The teacher interface and the records it produces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from anvil.errors import AnvilError


@dataclass(frozen=True)
class Item:
    """One unlabeled input."""

    id: str
    text: str


@dataclass(frozen=True)
class Label:
    """One teacher label with its provenance and what it cost to produce."""

    input_id: str
    label: str
    teacher_model: str
    prompt_hash: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    """Price of producing this label, whether or not this run paid it (see `cached`)."""
    cached: bool = False
    """True when the label came from the disk cache, so this run spent nothing on it."""


@dataclass(frozen=True)
class Failure:
    """An input the teacher could not label (refusal, truncated or invalid output).

    A refused or truncated response is still billed, so failures carry their cost too.
    """

    input_id: str
    reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class LabelBatch:
    labels: list[Label] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)

    def add(self, outcome: Label | Failure) -> None:
        if isinstance(outcome, Label):
            self.labels.append(outcome)
        else:
            self.failures.append(outcome)

    @property
    def spent_usd(self) -> float:
        """Dollars actually spent by this call: new labels plus billed failures."""
        return sum(lb.cost_usd for lb in self.labels if not lb.cached) + sum(
            f.cost_usd for f in self.failures
        )

    @property
    def cache_hits(self) -> int:
        return sum(1 for lb in self.labels if lb.cached)


class BatchAbortedError(AnvilError):
    """A batch stopped before every input was answered (a configuration error from the API,
    Ctrl-C, a crash). `batch` holds the answers that did complete, so the caller can persist
    what was paid for before re-raising `cause`."""

    def __init__(self, cause: BaseException, batch: LabelBatch) -> None:
        super().__init__(f"teacher batch aborted: {cause}")
        self.cause = cause
        self.batch = batch


class Teacher(Protocol):
    """Anything that can label a batch of inputs for a task."""

    @property
    def model_id(self) -> str:
        """Identifies the labelling model; part of the key labels are stored under."""
        ...

    def label(self, items: Sequence[Item]) -> LabelBatch: ...
