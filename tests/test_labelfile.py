from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from anvil.errors import ConfigError
from anvil.store import Store
from anvil.teacher.base import BatchAbortedError, Failure, Item, Label, LabelBatch
from anvil.teacher.labelfile import (
    GOLD,
    LabelFileTeacher,
    RecordedThenLiveTeacher,
    read_label_file,
    write_label_file,
)


def _label(i: str, lab: str, prompt: str = "p1", cost: float = 1e-4) -> Label:
    return Label(i, lab, "claude-haiku-5-5", prompt, 300, 5, cost)


def test_roundtrip_is_sorted_and_replays_as_cached(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    assert write_label_file(path, [_label("b", "x"), _label("a", "y")]) == 2
    rows = read_label_file(path)
    assert [r.input_id for r in rows] == ["a", "b"]
    teacher = LabelFileTeacher(rows, "claude-haiku-5-5", "p1")
    batch = teacher.label([Item("a", "text a"), Item("zz", "never recorded")])
    assert [lb.label for lb in batch.labels] == ["y"]
    assert batch.labels[0].cached
    assert batch.labels[0].cost_usd == pytest.approx(1e-4)
    assert batch.spent_usd == 0
    assert [f.input_id for f in batch.failures] == ["zz"]


def test_labels_from_another_prompt_are_refused(tmp_path: Path) -> None:
    rows = [r for r in read_label_file(_write(tmp_path, [_label("a", "x", prompt="old")]))]
    with pytest.raises(ConfigError, match="prompt changed"):
        LabelFileTeacher(rows, "claude-haiku-5-5", "new")


def _write(tmp_path: Path, labels: list[Label]) -> Path:
    path = tmp_path / "l.jsonl"
    write_label_file(path, labels)
    return path


def test_gold_teacher_is_free_and_named_gold() -> None:
    teacher = LabelFileTeacher.gold({"a": "x", "b": "y"}, "p1")
    batch = teacher.label([Item("b", "t")])
    assert teacher.model_id == GOLD
    assert batch.labels[0].label == "y"
    assert batch.labels[0].teacher_model == GOLD
    assert batch.labels[0].cost_usd == 0


def test_malformed_file_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"input_id": "a"}\n', encoding="utf-8")
    with pytest.raises(ConfigError, match=r"bad\.jsonl:1"):
        read_label_file(path)
    with pytest.raises(ConfigError, match="not found"):
        read_label_file(tmp_path / "missing.jsonl")


def test_store_exports_the_latest_label_per_input(tmp_path: Path) -> None:
    with Store(tmp_path / "s.sqlite", 0, 0.5) as store:
        store.add_inputs([Item("a", "alpha"), Item("b", "beta")])
        store.append_labels([_label("a", "x"), _label("b", "y"), _label("a", "z")])
        store.append_labels([_label("a", "other prompt", prompt="p2")])
        records = store.label_records("claude-haiku-5-5", "p1")
    assert [(r.input_id, r.label) for r in records] == [("a", "z"), ("b", "y")]
    assert records[0].input_tokens == 300


class _CountingTeacher:
    model_id = "claude-haiku-5-5"

    def __init__(self) -> None:
        self.asked: list[list[str]] = []

    def label(self, items: Sequence[Item]) -> LabelBatch:
        self.asked.append([it.id for it in items])
        batch = LabelBatch()
        for it in items:
            if it.id == "refused":
                batch.failures.append(Failure(it.id, "refusal"))
            else:
                batch.labels.append(_label(it.id, "live", cost=2e-4))
        return batch


def test_recorded_then_live_pays_only_for_missing_inputs(tmp_path: Path) -> None:
    recorded = LabelFileTeacher.from_file(
        _write(tmp_path, [_label("a", "x"), _label("b", "y")]), "claude-haiku-5-5", "p1"
    )
    live = _CountingTeacher()
    teacher = RecordedThenLiveTeacher(recorded, live)
    items = [Item("a", "ta"), Item("c", "tc"), Item("b", "tb"), Item("refused", "tr")]
    batch = teacher.label(items)
    assert live.asked == [["c", "refused"]]
    assert {lb.input_id: lb.label for lb in batch.labels} == {"a": "x", "b": "y", "c": "live"}
    assert [f.input_id for f in batch.failures] == ["refused"]
    assert batch.spent_usd == pytest.approx(2e-4)  # the recorded two were free
    assert teacher.label([Item("a", "ta")]).labels[0].cached
    assert live.asked == [["c", "refused"]]
    with pytest.raises(ConfigError, match="same teacher model"):
        RecordedThenLiveTeacher(LabelFileTeacher.gold({"a": "x"}, "p1"), live)


def test_recorded_then_live_keeps_recorded_answers_on_abort(tmp_path: Path) -> None:
    class Aborting:
        model_id = "claude-haiku-5-5"

        def label(self, items: Sequence[Item]) -> LabelBatch:
            partial = LabelBatch(labels=[_label(items[0].id, "live")])
            raise BatchAbortedError(KeyboardInterrupt(), partial)

    recorded = LabelFileTeacher.from_file(
        _write(tmp_path, [_label("a", "x")]), "claude-haiku-5-5", "p1"
    )
    teacher = RecordedThenLiveTeacher(recorded, Aborting())
    with pytest.raises(BatchAbortedError) as info:
        teacher.label([Item("a", "ta"), Item("c", "tc"), Item("d", "td")])
    assert [(lb.input_id, lb.label) for lb in info.value.batch.labels] == [
        ("a", "x"),
        ("c", "live"),
    ]
