"""`compute_seconds`: a local teacher's server time, carried through label files, the store
and batches; and the serve teacher selection for Ollama bundles."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from anvil.serve import SERVE_TEACHER_TIMEOUT_S, _teacher_for
from anvil.settings import Settings
from anvil.spec import TaskSpec
from anvil.store import Store
from anvil.teacher.base import Failure, Item, Label, LabelBatch
from anvil.teacher.labelfile import LabelFileTeacher, read_label_file, write_label_file
from anvil.teacher.ollama import HttpOllamaTransport, OllamaTeacher
from tests.test_teacher_ollama import MODEL, ollama_spec

OLD_SCHEMA = """
CREATE TABLE inputs (id TEXT PRIMARY KEY, text TEXT NOT NULL, added_at REAL NOT NULL);
CREATE TABLE labels (
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
CREATE TABLE failures (
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
"""


def test_label_file_round_trips_compute_seconds(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    labels = [
        Label("a", "bug", MODEL, "p1", 10, 2, 0.0, compute_seconds=1.25),
        Label("b", "other", MODEL, "p1", 10, 2, 0.0),
    ]
    failures = [(Failure("c", "not JSON", 5, 1, 0.0, compute_seconds=0.5), MODEL, "p1")]
    write_label_file(path, labels, failures)
    rows = {
        json.loads(line)["input_id"]: json.loads(line) for line in path.read_text().splitlines()
    }
    assert rows["a"]["compute_seconds"] == 1.25
    assert rows["c"]["compute_seconds"] == 0.5
    assert "compute_seconds" not in rows["b"]
    recorded = {r.input_id: r for r in read_label_file(path)}
    assert recorded["a"].compute_seconds == 1.25
    assert recorded["b"].compute_seconds is None
    assert recorded["c"].compute_seconds == 0.5
    batch = LabelFileTeacher(list(recorded.values()), MODEL, "p1").label(
        [Item("a", "x"), Item("b", "y"), Item("c", "z")]
    )
    assert [lb.compute_seconds for lb in batch.labels] == [1.25, None]
    assert all(lb.cached for lb in batch.labels)
    assert [f.compute_seconds for f in batch.failures] == [0.5]
    assert batch.compute_seconds == 0.0


def test_store_migrates_an_old_schema_database(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    db.executescript(OLD_SCHEMA)
    db.execute("INSERT INTO inputs VALUES ('i0', 'text 0', 0.0)")
    db.execute("INSERT INTO inputs VALUES ('i1', 'text 1', 0.0)")
    db.execute("INSERT INTO labels VALUES (NULL, 'i0', 'bug', 'm', 'p', 5, 1, 0.01, 0, 1.0)")
    db.execute("INSERT INTO failures VALUES (NULL, 'i1', 'refused', 'm', 'p', 5, 1, 0.01, 1.0)")
    db.commit()
    db.close()
    for _ in range(2):  # opening twice: the migration is idempotent
        with Store(path, split_seed=0, heldout_fraction=0.25) as store:
            (old,) = store.label_records("m", "p")
            assert (old.label, old.cost_usd, old.compute_seconds) == ("bug", 0.01, None)
            (old_failure,) = store.failure_records("m", "p")
            assert old_failure.compute_seconds is None
    with Store(path, split_seed=0, heldout_fraction=0.25) as store:
        store.append_labels([Label("i1", "bill", "m", "p", compute_seconds=2.5)])
        got = {lb.input_id: lb.compute_seconds for lb in store.label_records("m", "p")}
        assert got == {"i0": None, "i1": 2.5}


def test_store_returns_failure_seconds(tmp_path: Path) -> None:
    with Store(tmp_path / "s.sqlite", 0, 0.25) as store:
        store.add_inputs([Item("a", "x"), Item("b", "y")])
        store.append_failures(
            [Failure("a", "bad", 3, 1, 0.0, compute_seconds=0.75), Failure("b", "bad")],
            "m",
            "p",
        )
        got = {f.input_id: f.compute_seconds for f in store.failure_records("m", "p")}
        assert got == {"a": 0.75, "b": None}


def test_batch_compute_seconds_excludes_cached_outcomes() -> None:
    batch = LabelBatch()
    batch.add(Label("a", "x", "m", "p", compute_seconds=1.0))
    batch.add(Label("b", "x", "m", "p", compute_seconds=4.0, cached=True))
    batch.add(Label("c", "x", "m", "p"))
    batch.add(Failure("d", "bad", compute_seconds=0.5))
    batch.add(Failure("e", "bad", compute_seconds=8.0, cached=True))
    assert batch.compute_seconds == pytest.approx(1.5)
    assert LabelBatch().compute_seconds == 0.0


def test_serve_builds_the_ollama_teacher(toy_spec: TaskSpec, settings: Settings) -> None:
    spec = ollama_spec(toy_spec)
    for mode in ("ollama", "auto"):
        teacher = _teacher_for(mode, spec, settings, Path("bundle"))
        assert isinstance(teacher, OllamaTeacher)
        assert teacher.model_id == MODEL
        # A served request must not wait out the labelling timeout (600 s, two retries).
        transport = teacher._transport
        assert isinstance(transport, HttpOllamaTransport)
        assert (transport._timeout, transport._retries) == (SERVE_TEACHER_TIMEOUT_S, 0)
    # An Ollama spec wins over an explicit claude mode only through `ollama`/`auto`:
    # `claude` still asks for Claude, which needs credentials the test env lacks.
    assert _teacher_for("none", spec, settings, Path("bundle")) is None
