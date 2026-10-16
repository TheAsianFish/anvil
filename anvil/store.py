"""The label store: inputs, append-only teacher labels and fixed hash-based splits (SQLite).

Split membership is a pure function of (split seed, input id), so adding inputs, re-running a
command or labelling in a different order never moves an item between held-out and pool.
Labels are never updated or deleted (triggers enforce it); each one carries the teacher model
and prompt hash, and reads always ask for one (model, prompt hash) pair.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from anvil.errors import ConfigError, StoreError
from anvil.teacher.base import Item, Label

SCHEMA = """
CREATE TABLE IF NOT EXISTS inputs (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    added_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS labels (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    input_id TEXT NOT NULL REFERENCES inputs(id),
    label TEXT NOT NULL,
    teacher_model TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    cached INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS labels_key ON labels (teacher_model, prompt_hash, input_id);
CREATE TRIGGER IF NOT EXISTS labels_no_update BEFORE UPDATE ON labels
BEGIN SELECT RAISE(ABORT, 'labels are append-only'); END;
CREATE TRIGGER IF NOT EXISTS labels_no_delete BEFORE DELETE ON labels
BEGIN SELECT RAISE(ABORT, 'labels are append-only'); END;
"""


class Split(StrEnum):
    HELDOUT = "heldout"
    """Teacher-labelled evaluation set; never trained on."""
    POOL = "pool"
    """Everything else: labelled pool items are the training set."""


def split_of(input_id: str, seed: int, heldout_fraction: float) -> Split:
    digest = hashlib.sha256(f"anvil-split:{seed}:{input_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    return Split.HELDOUT if u < heldout_fraction else Split.POOL


def order_key(input_id: str, seed: int) -> str:
    """A stable pseudo-random order over inputs (the `random` selection baseline in M1)."""
    return hashlib.sha256(f"anvil-order:{seed}:{input_id}".encode()).hexdigest()


def text_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_inputs(path: Path) -> list[Item]:
    """Read a JSONL file of `{"text": ..., "id": optional}` objects.

    Inputs without an `id` get a content hash, so the same text always has the same id.
    """
    items: list[Item] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ConfigError(f"inputs file {path} not found") from exc
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path}:{n} is not valid JSON: {exc.msg}") from exc
        text = obj.get("text") if isinstance(obj, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ConfigError(f'{path}:{n} needs a non-empty "text" string')
        raw_id = obj.get("id")
        item_id = str(raw_id) if raw_id is not None else text_id(text)
        if item_id in seen:
            raise ConfigError(f"{path}:{n} repeats id {item_id!r}; ids must be unique")
        seen.add(item_id)
        items.append(Item(id=item_id, text=text))
    if not items:
        raise ConfigError(f"{path} has no inputs")
    return items


@dataclass(frozen=True)
class LabelStats:
    labels: int
    cost_usd: float
    spent_usd: float


class Store:
    def __init__(self, path: Path, split_seed: int, heldout_fraction: float) -> None:
        self.path = path
        self.split_seed = split_seed
        self.heldout_fraction = heldout_fraction
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._db:
            yield self._db

    def split(self, input_id: str) -> Split:
        return split_of(input_id, self.split_seed, self.heldout_fraction)

    def add_inputs(self, items: Iterable[Item]) -> int:
        """Insert new inputs; returns how many were new. A known id with new text is an error."""
        added = 0
        now = time.time()
        with self._tx() as db:
            for item in items:
                row = db.execute("SELECT text FROM inputs WHERE id = ?", (item.id,)).fetchone()
                if row is None:
                    db.execute(
                        "INSERT INTO inputs (id, text, added_at) VALUES (?, ?, ?)",
                        (item.id, item.text, now),
                    )
                    added += 1
                elif row[0] != item.text:
                    raise StoreError(
                        f"input {item.id!r} already exists with different text; give the new"
                        " text a new id (labels refer to ids)"
                    )
        return added

    def inputs(self, split: Split | None = None) -> list[Item]:
        rows = self._db.execute("SELECT id, text FROM inputs").fetchall()
        items = [Item(id=r[0], text=r[1]) for r in rows]
        if split is not None:
            items = [it for it in items if self.split(it.id) == split]
        return sorted(items, key=lambda it: order_key(it.id, self.split_seed))

    def append_labels(self, labels: Sequence[Label]) -> None:
        now = time.time()
        with self._tx() as db:
            db.executemany(
                "INSERT INTO labels (input_id, label, teacher_model, prompt_hash, input_tokens,"
                " output_tokens, cost_usd, cached, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (
                        lb.input_id,
                        lb.label,
                        lb.teacher_model,
                        lb.prompt_hash,
                        lb.input_tokens,
                        lb.output_tokens,
                        lb.cost_usd,
                        int(lb.cached),
                        now,
                    )
                    for lb in labels
                ],
            )

    def labels(self, teacher_model: str, prompt_hash: str) -> dict[str, str]:
        """The latest label per input for one (teacher model, prompt hash) key."""
        rows = self._db.execute(
            "SELECT input_id, label FROM labels WHERE teacher_model = ? AND prompt_hash = ?"
            " ORDER BY seq",
            (teacher_model, prompt_hash),
        ).fetchall()
        return {input_id: label for input_id, label in rows}

    def labelled(
        self, teacher_model: str, prompt_hash: str, split: Split
    ) -> list[tuple[Item, str]]:
        """Inputs of a split that have a label under this key, in stable order."""
        by_id = self.labels(teacher_model, prompt_hash)
        return [(it, by_id[it.id]) for it in self.inputs(split) if it.id in by_id]

    def stats(self, teacher_model: str, prompt_hash: str) -> LabelStats:
        row = self._db.execute(
            "SELECT COUNT(DISTINCT input_id), COALESCE(SUM(cost_usd), 0),"
            " COALESCE(SUM(CASE WHEN cached = 0 THEN cost_usd ELSE 0 END), 0)"
            " FROM labels WHERE teacher_model = ? AND prompt_hash = ?",
            (teacher_model, prompt_hash),
        ).fetchone()
        return LabelStats(labels=int(row[0]), cost_usd=float(row[1]), spent_usd=float(row[2]))
