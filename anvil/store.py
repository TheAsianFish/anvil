"""The label store: inputs, append-only teacher labels and fixed hash-based splits (SQLite).

Split membership is a pure function of (split seed, input text), so adding inputs, re-running a
command or labelling in a different order never moves an item between held-out and pool, and
two copies of the same text under different ids always land on the same side.
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
from anvil.teacher.base import Failure, Item, Label

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
CREATE TABLE IF NOT EXISTS failures (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    input_id TEXT NOT NULL REFERENCES inputs(id),
    reason TEXT NOT NULL,
    teacher_model TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS loops (
    loop_id TEXT PRIMARY KEY,
    config TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS loop_picks (
    loop_id TEXT NOT NULL REFERENCES loops(loop_id),
    round INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    input_id TEXT NOT NULL REFERENCES inputs(id),
    PRIMARY KEY (loop_id, input_id)
);
CREATE INDEX IF NOT EXISTS labels_key ON labels (teacher_model, prompt_hash, input_id);
CREATE TRIGGER IF NOT EXISTS labels_no_update BEFORE UPDATE ON labels
BEGIN SELECT RAISE(ABORT, 'labels are append-only'); END;
CREATE TRIGGER IF NOT EXISTS labels_no_delete BEFORE DELETE ON labels
BEGIN SELECT RAISE(ABORT, 'labels are append-only'); END;
CREATE TRIGGER IF NOT EXISTS failures_no_update BEFORE UPDATE ON failures
BEGIN SELECT RAISE(ABORT, 'failures are append-only'); END;
CREATE TRIGGER IF NOT EXISTS failures_no_delete BEFORE DELETE ON failures
BEGIN SELECT RAISE(ABORT, 'failures are append-only'); END;
CREATE TRIGGER IF NOT EXISTS loop_picks_no_update BEFORE UPDATE ON loop_picks
BEGIN SELECT RAISE(ABORT, 'loop picks are append-only'); END;
CREATE TRIGGER IF NOT EXISTS loop_picks_no_delete BEFORE DELETE ON loop_picks
BEGIN SELECT RAISE(ABORT, 'loop picks are append-only'); END;
"""


class Split(StrEnum):
    HELDOUT = "heldout"
    """Teacher-labelled evaluation set; never trained on."""
    POOL = "pool"
    """Everything else: labelled pool items are the training set."""


def split_of(text: str, seed: int, heldout_fraction: float) -> Split:
    """Held-out or pool, from a hash of the text (not the id, so duplicates cannot straddle)."""
    digest = hashlib.sha256(f"anvil-split:{seed}:{text_id(text)}".encode()).digest()
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
        if raw_id is not None and (isinstance(raw_id, bool) or not isinstance(raw_id, str | int)):
            raise ConfigError(f'{path}:{n} "id" must be a string or an integer')
        item_id = str(raw_id) if raw_id is not None else text_id(text)
        if not item_id:
            raise ConfigError(f'{path}:{n} "id" must not be empty')
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
    """Distinct inputs with a label under the key."""
    cost_usd: float
    """Price of producing those labels plus every billed failure (what the labels cost)."""
    spent_usd: float
    """What was actually paid: cache hits excluded, failures included."""


class Store:
    def __init__(self, path: Path, split_seed: int, heldout_fraction: float) -> None:
        self.path = path
        self.split_seed = split_seed
        self.heldout_fraction = heldout_fraction
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("PRAGMA foreign_keys = ON")  # per connection; labels need their input
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

    def split(self, item: Item) -> Split:
        return split_of(item.text, self.split_seed, self.heldout_fraction)

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
            items = [it for it in items if self.split(it) == split]
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

    def append_failures(
        self, failures: Sequence[Failure], teacher_model: str, prompt_hash: str
    ) -> None:
        """Failed teacher calls are kept for their cost and for diagnosing the prompt.

        Every live attempt is a row (each one was billed). A replayed failure (`cached`, from a
        label file) is skipped: it was billed once, when it was recorded, and its row in the
        label file is its record. A refused input never gets a label, so every run sends it
        again; storing each replay would add the recorded cost to `stats` once per run.
        """
        now = time.time()
        with self._tx() as db:
            db.executemany(
                "INSERT INTO failures (input_id, reason, teacher_model, prompt_hash,"
                " input_tokens, output_tokens, cost_usd, created_at) VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        f.input_id,
                        f.reason,
                        teacher_model,
                        prompt_hash,
                        f.input_tokens,
                        f.output_tokens,
                        f.cost_usd,
                        now,
                    )
                    for f in failures
                    if not f.cached
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

    def label_records(self, teacher_model: str, prompt_hash: str) -> list[Label]:
        """The latest label per input under one key, with its tokens and cost (for export)."""
        rows = self._db.execute(
            "SELECT input_id, label, input_tokens, output_tokens, cost_usd, cached FROM labels"
            " WHERE teacher_model = ? AND prompt_hash = ? ORDER BY seq",
            (teacher_model, prompt_hash),
        ).fetchall()
        latest = {
            r[0]: Label(r[0], r[1], teacher_model, prompt_hash, r[2], r[3], r[4], bool(r[5]))
            for r in rows
        }
        return [latest[k] for k in sorted(latest)]

    def teacher_models(self, prompt_hash: str) -> list[str]:
        """Every teacher model that has labelled inputs under this prompt hash."""
        rows = self._db.execute(
            "SELECT DISTINCT teacher_model FROM labels WHERE prompt_hash = ? ORDER BY 1",
            (prompt_hash,),
        ).fetchall()
        return [row[0] for row in rows]

    def labelled(
        self, teacher_model: str, prompt_hash: str, split: Split
    ) -> list[tuple[Item, str]]:
        """Inputs of a split that have a label under this key, in stable order."""
        by_id = self.labels(teacher_model, prompt_hash)
        return [(it, by_id[it.id]) for it in self.inputs(split) if it.id in by_id]

    def create_loop(self, loop_id: str, config: dict[str, object]) -> None:
        """Register an active-learning loop; its config is fixed from then on."""
        with self._tx() as db:
            try:
                db.execute(
                    "INSERT INTO loops (loop_id, config, created_at) VALUES (?, ?, ?)",
                    (loop_id, json.dumps(config, sort_keys=True), time.time()),
                )
            except sqlite3.IntegrityError:
                raise StoreError(
                    f"loop {loop_id!r} already exists; resume it or pick another id"
                ) from None

    def loop_config(self, loop_id: str) -> dict[str, object] | None:
        row = self._db.execute("SELECT config FROM loops WHERE loop_id = ?", (loop_id,)).fetchone()
        if row is None:
            return None
        config: dict[str, object] = json.loads(row[0])
        return config

    def add_picks(self, loop_id: str, round_no: int, input_ids: Sequence[str]) -> None:
        """Record which inputs a loop chose in a round, in the order they were ranked."""
        with self._tx() as db:
            db.executemany(
                "INSERT INTO loop_picks (loop_id, round, rank, input_id) VALUES (?, ?, ?, ?)",
                [(loop_id, round_no, rank, i) for rank, i in enumerate(input_ids)],
            )

    def picks(self, loop_id: str) -> list[tuple[int, str]]:
        """(round, input id) for every input a loop has picked, by round then rank."""
        rows = self._db.execute(
            "SELECT round, input_id FROM loop_picks WHERE loop_id = ? ORDER BY round, rank",
            (loop_id,),
        ).fetchall()
        return [(int(r[0]), str(r[1])) for r in rows]

    def cost_of(self, teacher_model: str, prompt_hash: str, input_ids: Iterable[str]) -> float:
        """What labelling these inputs cost under one key: their labels plus every billed
        failed attempt on them (a refusal is paid for even though it yields no label)."""
        wanted = set(input_ids)
        rows = self._db.execute(
            "SELECT input_id, cost_usd FROM labels WHERE teacher_model = ? AND prompt_hash = ?"
            " UNION ALL SELECT input_id, cost_usd FROM failures"
            " WHERE teacher_model = ? AND prompt_hash = ?",
            (teacher_model, prompt_hash, teacher_model, prompt_hash),
        ).fetchall()
        return sum(float(cost) for input_id, cost in rows if input_id in wanted)

    def stats(self, teacher_model: str, prompt_hash: str) -> LabelStats:
        row = self._db.execute(
            "SELECT COUNT(DISTINCT input_id), COALESCE(SUM(cost_usd), 0),"
            " COALESCE(SUM(CASE WHEN cached = 0 THEN cost_usd ELSE 0 END), 0)"
            " FROM labels WHERE teacher_model = ? AND prompt_hash = ?",
            (teacher_model, prompt_hash),
        ).fetchone()
        failed = self._db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM failures"
            " WHERE teacher_model = ? AND prompt_hash = ?",
            (teacher_model, prompt_hash),
        ).fetchone()
        return LabelStats(
            labels=int(row[0]),
            cost_usd=float(row[1]) + float(failed[0]),
            spent_usd=float(row[2]) + float(failed[0]),
        )

    def failure_records(self, teacher_model: str, prompt_hash: str) -> list[Failure]:
        """The latest failure per input under one key, for inputs that have no label under it
        (a later answer supersedes a refusal), sorted by input id (for export)."""
        rows = self._db.execute(
            "SELECT input_id, reason, input_tokens, output_tokens, cost_usd FROM failures"
            " WHERE teacher_model = ? AND prompt_hash = ? ORDER BY seq",
            (teacher_model, prompt_hash),
        ).fetchall()
        labelled = self.labels(teacher_model, prompt_hash)
        latest = {r[0]: Failure(r[0], r[1], r[2], r[3], r[4]) for r in rows}
        return [latest[k] for k in sorted(latest) if k not in labelled]
