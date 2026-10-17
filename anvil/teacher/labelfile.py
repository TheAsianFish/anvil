"""Recorded labels: a teacher that answers from a JSONL file of labels made earlier.

An eval labels its inputs once with Claude and commits the labels; every later run replays
them through this teacher, so the student, the statistics and the report are re-run offline
and for free. The same class serves a dataset's gold labels (model id `gold`), which lets the
harness run end to end without a key, under a model id that can never be mistaken for Claude.
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


@dataclass(frozen=True)
class RecordedLabel:
    input_id: str
    label: str
    teacher_model: str
    prompt_hash: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def write_label_file(path: Path, labels: Iterable[Label]) -> int:
    """Write labels as JSONL, sorted by input id so the file diffs cleanly. Returns the count."""
    rows = sorted(
        (
            RecordedLabel(
                lb.input_id,
                lb.label,
                lb.teacher_model,
                lb.prompt_hash,
                lb.input_tokens,
                lb.output_tokens,
                lb.cost_usd,
            )
            for lb in labels
        ),
        key=lambda r: r.input_id,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
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
            rows.append(RecordedLabel(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ConfigError(f"{path}:{n} is not a recorded label: {exc}") from exc
    return rows


class LabelFileTeacher:
    """Answers from recorded labels for one (teacher model, prompt hash) key.

    Labels come back with `cached=True` and their original cost, so a replay spends nothing
    while the store still knows what the labels cost to make. An input without a recorded
    label is a `Failure`, never a guess.
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
                batch.failures.append(Failure(item.id, "no recorded label for this input"))
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
                )
            )
        return batch


class RecordedThenLiveTeacher:
    """Answers from recorded labels first and sends only the inputs without one to a live
    teacher under the same (model, prompt hash).

    A fresh checkout has the committed label file but neither the store nor the disk cache,
    so without this a run that still needs a few labels (say the paraphrase set for the
    ceiling) would buy every main label a second time.
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
        missing = {f.input_id for f in batch.failures}
        todo = [it for it in items if it.id in missing]
        batch.failures = []
        if not todo:
            return batch
        try:
            fresh = self._live.label(todo)
        except BatchAbortedError as exc:
            exc.batch.labels[:0] = batch.labels  # the recorded answers are kept as well
            raise
        batch.labels.extend(fresh.labels)
        batch.failures.extend(fresh.failures)
        return batch
