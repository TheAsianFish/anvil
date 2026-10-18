from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from anvil.errors import ConfigError
from anvil.store import Store
from anvil.teacher.base import BatchAbortedError, Failure, Item, Label, LabelBatch
from anvil.teacher.labelfile import (
    GOLD,
    NOT_RECORDED,
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


def _refusal(i: str, cost: float = 0.005) -> Failure:
    return Failure(i, "refusal (cyber)", 1000, 50, cost)


def test_failure_rows_roundtrip_and_replay_as_free_failures(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    key = ("claude-haiku-5-5", "p1")
    assert write_label_file(path, [_label("a", "x")], [(_refusal("b"), *key)]) == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    assert '"reason"' not in lines[0]  # label rows keep their old shape
    assert '"label"' not in lines[1]
    rows = read_label_file(path)
    assert [(r.input_id, r.label, r.reason) for r in rows] == [
        ("a", "x", None),
        ("b", None, "refusal (cyber)"),
    ]
    assert rows[1].failed
    teacher = LabelFileTeacher(rows, *key)
    batch = teacher.label([Item("a", "ta"), Item("b", "tb"), Item("zz", "never recorded")])
    assert [lb.input_id for lb in batch.labels] == ["a"]
    assert batch.failures == [
        Failure("b", "refusal (cyber)", 1000, 50, 0.005, cached=True),
        Failure("zz", NOT_RECORDED),
    ]
    assert batch.spent_usd == 0  # the refusal was paid for when it was recorded


def test_a_label_supersedes_a_failure_for_the_same_input(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    key = ("claude-haiku-5-5", "p1")
    failures = [(_refusal("a"), *key), (_refusal("b"), *key)]
    assert write_label_file(path, [_label("a", "x")], failures) == 2
    assert [(r.input_id, r.label) for r in read_label_file(path)] == [("a", "x"), ("b", None)]


@pytest.mark.parametrize(
    "row",
    [
        '{"input_id": "a", "label": "x", "reason": "refusal", "teacher_model": "m",'
        ' "prompt_hash": "p"}',
        '{"input_id": "a", "teacher_model": "m", "prompt_hash": "p"}',
        '{"input_id": "a", "label": null, "reason": null, "teacher_model": "m",'
        ' "prompt_hash": "p"}',
        '{"input_id": "a", "label": 3, "teacher_model": "m", "prompt_hash": "p"}',
        '["a", "x"]',
    ],
)
def test_a_row_needs_exactly_one_of_label_and_reason(tmp_path: Path, row: str) -> None:
    path = tmp_path / "bad.jsonl"
    ok = '{"input_id": "b", "label": "x", "teacher_model": "m", "prompt_hash": "p"}'
    path.write_text(f"{ok}\n{row}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"bad\.jsonl:2"):
        read_label_file(path)


def test_recorded_then_live_keeps_recorded_refusals_and_asks_only_unrecorded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "l.jsonl"
    write_label_file(path, [_label("a", "x")], [(_refusal("b"), "claude-haiku-5-5", "p1")])
    live = _CountingTeacher()
    teacher = RecordedThenLiveTeacher(
        LabelFileTeacher.from_file(path, "claude-haiku-5-5", "p1"), live
    )
    batch = teacher.label([Item("a", "ta"), Item("b", "tb"), Item("c", "tc")])
    assert live.asked == [["c"]]  # the recorded refusal is not bought again
    assert {lb.input_id: lb.label for lb in batch.labels} == {"a": "x", "c": "live"}
    assert batch.failures == [Failure("b", "refusal (cyber)", 1000, 50, 0.005, cached=True)]
    assert batch.spent_usd == pytest.approx(2e-4)  # only the live label
    only_recorded = teacher.label([Item("b", "tb")])
    assert live.asked == [["c"]]
    assert [f.reason for f in only_recorded.failures] == ["refusal (cyber)"]


def test_store_exports_the_latest_failure_per_unlabelled_input(tmp_path: Path) -> None:
    with Store(tmp_path / "s.sqlite", 0, 0.5) as store:
        store.add_inputs([Item("a", "alpha"), Item("b", "beta"), Item("c", "gamma")])
        store.append_failures([_refusal("b", 0.01), _refusal("a")], "claude-haiku-5-5", "p1")
        store.append_failures(
            [Failure("b", "hit max_tokens", 10, 20, 0.02)], "claude-haiku-5-5", "p1"
        )
        store.append_failures([_refusal("c")], "claude-haiku-5-5", "p2")
        store.append_labels([_label("a", "x")])  # a later answer supersedes the refusal
        records = store.failure_records("claude-haiku-5-5", "p1")
        assert records == [Failure("b", "hit max_tokens", 10, 20, 0.02)]
        assert store.failure_records("claude-haiku-5-5", "p3") == []
        before = store.stats("claude-haiku-5-5", "p1")
        # A replayed refusal was billed when recorded: storing it on every replay would count
        # its cost again each run, so it is not stored.
        replayed = Failure("c", "refusal (cyber)", 1000, 50, 0.005, cached=True)
        for _ in range(3):
            store.append_failures([replayed], "claude-haiku-5-5", "p1")
        assert store.stats("claude-haiku-5-5", "p1") == before
        assert [f.input_id for f in store.failure_records("claude-haiku-5-5", "p1")] == ["b"]
