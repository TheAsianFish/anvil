"""Recorded labels: a teacher that answers from a JSONL file of labels made earlier.

An eval labels its inputs once with Claude and commits the labels; every later run replays
them through this teacher, so the student, the statistics and the report are re-run offline
and for free. The same class serves a dataset's gold labels (model id `gold`), which lets the
harness run end to end without a key, under a model id that can never be mistaken for Claude.

A row is either a label or a failure (a refusal, a truncated or an invalid answer, with what
it cost). Failures are recorded so that a replay sees the same refusals the live run did,
instead of mistaking a refused input for a hole in the recording.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from anvil.errors import ConfigError
from anvil.teacher.base import BatchAbortedError, Failure, Item, Label, LabelBatch, Teacher

GOLD = "gold"
"""Model id for a dataset's human labels used as the teacher."""

NOT_RECORDED = "no recorded label for this input"
"""`Failure.reason` for an input the recording has no row for (never labelled), as opposed to
a recorded failure, which keeps the teacher's own reason."""


@dataclass(frozen=True)
class RecordedLabel:
    """One recorded row: exactly one of `label` (an answer) and `reason` (a failure) is set."""

    input_id: str
    label: str | None
    teacher_model: str
    prompt_hash: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    reason: str | None = None
    compute_seconds: float | None = None
    """A local teacher's server seconds for this row; absent for Claude and gold rows."""

    def __post_init__(self) -> None:
        if (self.label is None) == (self.reason is None):
            raise ValueError("a recorded row needs exactly one of label and reason")
        for name in ("input_id", "label", "teacher_model", "prompt_hash", "reason"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
        seconds = self.compute_seconds
        if seconds is not None and (
            not isinstance(seconds, int | float) or isinstance(seconds, bool) or seconds < 0
        ):
            raise ValueError("compute_seconds must be a non-negative number")

    @property
    def failed(self) -> bool:
        return self.reason is not None


def write_label_file(
    path: Path,
    labels: Iterable[Label],
    failures: Iterable[tuple[Failure, str, str]] = (),
) -> int:
    """Write labels and failures as JSONL, sorted by input id so the file diffs cleanly.

    `failures` are `(failure, teacher model, prompt hash)`. A label for an input supersedes a
    failure for the same input (the label is the answer the run needs; the failure was an
    earlier attempt). Returns the row count.
    """
    rows: dict[str, RecordedLabel] = {}
    for f, model, prompt_hash in failures:
        rows[f.input_id] = RecordedLabel(
            f.input_id,
            None,
            model,
            prompt_hash,
            f.input_tokens,
            f.output_tokens,
            f.cost_usd,
            reason=f.reason,
            compute_seconds=f.compute_seconds,
        )
    for lb in labels:
        rows[lb.input_id] = RecordedLabel(
            lb.input_id,
            lb.label,
            lb.teacher_model,
            lb.prompt_hash,
            lb.input_tokens,
            lb.output_tokens,
            lb.cost_usd,
            compute_seconds=lb.compute_seconds,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for key in sorted(rows):
                # Unset fields (one of label/reason, the seconds of a remote teacher) are left
                # out, so Claude and gold rows look as they always did.
                obj = {k: v for k, v in asdict(rows[key]).items() if v is not None}
                fh.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return len(rows)


def read_label_file(path: Path) -> list[RecordedLabel]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ConfigError(f"label file {path} not found") from exc
    rows: list[RecordedLabel] = []
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("expected a JSON object")
            rows.append(RecordedLabel(**{"label": None, **obj}))
        except (ValueError, TypeError) as exc:  # JSONDecodeError is a ValueError
            raise ConfigError(
                f"{path}:{n} is not a recorded label or failure: {exc}; fix or delete the row"
                " (a deleted row is labelled again by the next --teacher claude run)"
            ) from exc
    return rows


class LabelFileTeacher:
    """Answers from recorded labels for one (teacher model, prompt hash) key.

    Labels come back with `cached=True` and their original cost, so a replay spends nothing
    while the store still knows what the labels cost to make. A recorded failure comes back as
    a `Failure` with its original reason, tokens and cost, also `cached=True`. An input with no
    row at all is a `Failure` with reason `NOT_RECORDED`, never a guess.
    """

    def __init__(self, rows: Sequence[RecordedLabel], model_id: str, prompt_hash: str) -> None:
        self.model_id = model_id
        self._prompt_hash = prompt_hash
        self._rows: dict[str, RecordedLabel] = {}
        for row in rows:
            if row.teacher_model != model_id or row.prompt_hash != prompt_hash:
                raise ConfigError(
                    f"recorded label for {row.input_id!r} is from ({row.teacher_model},"
                    f" {row.prompt_hash}), expected ({model_id}, {prompt_hash}); the prompt"
                    " changed since recording, so re-label with the teacher"
                )
            self._rows[row.input_id] = row

    @classmethod
    def from_file(cls, path: Path, model_id: str, prompt_hash: str) -> LabelFileTeacher:
        return cls(read_label_file(path), model_id, prompt_hash)

    @classmethod
    def gold(cls, gold: dict[str, str], prompt_hash: str) -> LabelFileTeacher:
        """A dataset's human labels as the teacher (free, deterministic, model id `gold`)."""
        rows = [RecordedLabel(i, lab, GOLD, prompt_hash) for i, lab in gold.items()]
        return cls(rows, GOLD, prompt_hash)

    def __len__(self) -> int:
        return len(self._rows)

    def label(self, items: Sequence[Item]) -> LabelBatch:
        batch = LabelBatch()
        for item in items:
            row = self._rows.get(item.id)
            if row is None:
                batch.failures.append(Failure(item.id, NOT_RECORDED))
                continue
            if row.label is None:
                batch.failures.append(
                    Failure(
                        item.id,
                        row.reason or "",
                        row.input_tokens,
                        row.output_tokens,
                        row.cost_usd,
                        cached=True,
                        compute_seconds=row.compute_seconds,
                    )
                )
                continue
            batch.labels.append(
                Label(
                    input_id=item.id,
                    label=row.label,
                    teacher_model=self.model_id,
                    prompt_hash=self._prompt_hash,
                    input_tokens=row.input_tokens,
                    output_tokens=row.output_tokens,
                    cost_usd=row.cost_usd,
                    cached=True,
                    compute_seconds=row.compute_seconds,
                )
            )
        return batch


class RecordedThenLiveTeacher:
    """Answers from recorded labels first and sends only the inputs without a row
    (`NOT_RECORDED`) to a live teacher under the same (model, prompt hash).

    A fresh checkout has the committed label file but neither the store nor the disk cache,
    so without this a run that still needs a few labels (say the paraphrase set for the
    ceiling) would buy every main label a second time. A recorded failure (a refusal) is
    returned as recorded, not asked again, so a recording stays one fixed set of answers;
    delete its row from the label file to pay for a retry.
    """

    def __init__(self, recorded: LabelFileTeacher, live: Teacher) -> None:
        if recorded.model_id != live.model_id:
            raise ConfigError(
                f"recorded labels are from {recorded.model_id}, the live teacher is"
                f" {live.model_id}; use the same teacher model or re-record"
            )
        self.model_id = live.model_id
        self._recorded = recorded
        self._live = live

    def label(self, items: Sequence[Item]) -> LabelBatch:
        batch = self._recorded.label(items)
        missing = {f.input_id for f in batch.failures if f.reason == NOT_RECORDED}
        todo = [it for it in items if it.id in missing]
        batch.failures = [f for f in batch.failures if f.reason != NOT_RECORDED]
        if not todo:
            return batch
        try:
            fresh = self._live.label(todo)
        except BatchAbortedError as exc:
            exc.batch.labels[:0] = batch.labels  # the recorded answers are kept as well
            exc.batch.failures[:0] = batch.failures
            raise
        batch.labels.extend(fresh.labels)
        batch.failures.extend(fresh.failures)
        return batch
