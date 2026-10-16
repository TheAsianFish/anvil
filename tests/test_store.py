from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from anvil.errors import ConfigError, StoreError
from anvil.store import Split, Store, read_inputs, split_of, text_id
from anvil.teacher.base import Item, Label


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "db" / "store.sqlite", split_seed=0, heldout_fraction=0.25) as s:
        yield s


def _items(n: int, start: int = 0) -> list[Item]:
    return [Item(f"i{k}", f"text {k}") for k in range(start, start + n)]


def _label(
    input_id: str,
    label: str,
    *,
    model: str = "m",
    ph: str = "p",
    cost_usd: float = 0.0,
    cached: bool = False,
) -> Label:
    return Label(input_id, label, model, ph, cost_usd=cost_usd, cached=cached)


def test_split_is_deterministic_and_matches_fraction() -> None:
    ids = [f"id-{k}" for k in range(5000)]
    first = [split_of(i, 0, 0.25) for i in ids]
    assert first == [split_of(i, 0, 0.25) for i in ids]
    frac = sum(s == Split.HELDOUT for s in first) / len(ids)
    assert abs(frac - 0.25) < 0.02
    assert [split_of(i, 1, 0.25) for i in ids] != first


def test_adding_inputs_never_moves_existing_ids(store: Store) -> None:
    store.add_inputs(_items(200))
    before = {it.id: store.split(it.id) for it in store.inputs()}
    heldout_before = [it.id for it in store.inputs(Split.HELDOUT)]
    store.add_inputs(_items(300, start=200))
    assert {i: store.split(i) for i in before} == before
    heldout_after = [it.id for it in store.inputs(Split.HELDOUT)]
    assert [i for i in heldout_after if i in before] == heldout_before


def test_inputs_order_is_stable_across_insert_order(tmp_path: Path) -> None:
    items = _items(50)
    with Store(tmp_path / "a.sqlite", 0, 0.25) as a, Store(tmp_path / "b.sqlite", 0, 0.25) as b:
        a.add_inputs(items)
        b.add_inputs(reversed(items))
        assert a.inputs() == b.inputs()
        assert a.inputs(Split.POOL) == b.inputs(Split.POOL)
        assert {it.id for it in a.inputs()} == {it.id for it in items}


def test_add_inputs_counts_new_and_is_idempotent(store: Store) -> None:
    assert store.add_inputs(_items(10)) == 10
    assert store.add_inputs(_items(10)) == 0
    assert store.add_inputs(_items(15)) == 5
    assert len(store.inputs()) == 15


def test_conflicting_text_for_existing_id(store: Store) -> None:
    store.add_inputs([Item("x", "one")])
    with pytest.raises(StoreError, match="different text"):
        store.add_inputs([Item("x", "two")])
    assert store.inputs() == [Item("x", "one")]


def test_labels_are_append_only(store: Store) -> None:
    store.add_inputs(_items(1))
    store.append_labels([_label("i0", "a")])
    with pytest.raises(sqlite3.DatabaseError, match="append-only"), store._db:
        store._db.execute("UPDATE labels SET label = 'b'")
    other = sqlite3.connect(store.path)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            other.execute("DELETE FROM labels")
    finally:
        other.close()
    assert store.labels("m", "p") == {"i0": "a"}


def test_labels_latest_wins_and_keys_are_isolated(store: Store) -> None:
    store.add_inputs(_items(2))
    store.append_labels([_label("i0", "a"), _label("i1", "a")])
    store.append_labels([_label("i0", "b")])
    store.append_labels([_label("i0", "c", model="other"), _label("i1", "c", ph="q")])
    assert store.labels("m", "p") == {"i0": "b", "i1": "a"}
    assert store.labels("other", "p") == {"i0": "c"}
    assert store.labels("m", "q") == {"i1": "c"}
    assert store.labels("other", "q") == {}


def test_labelled_returns_split_members_in_stable_order(store: Store) -> None:
    store.add_inputs(_items(40))
    store.append_labels([_label(it.id, "a") for it in _items(40)])
    pool = store.labelled("m", "p", Split.POOL)
    assert [it for it, _ in pool] == store.inputs(Split.POOL)
    assert all(store.split(it.id) == Split.POOL for it, _ in pool)


def test_stats_counts_inputs_and_separates_cost_from_spend(store: Store) -> None:
    store.add_inputs(_items(3))
    store.append_labels(
        [
            _label("i0", "a", cost_usd=0.01),
            _label("i1", "a", cost_usd=0.02, cached=True),
            _label("i0", "b", cost_usd=0.04),
            _label("i2", "a", model="other", cost_usd=1.0),
        ]
    )
    stats = store.stats("m", "p")
    assert stats.labels == 2
    assert stats.cost_usd == pytest.approx(0.07)
    assert stats.spent_usd == pytest.approx(0.05)
    empty = store.stats("none", "p")
    assert (empty.labels, empty.cost_usd, empty.spent_usd) == (0, 0.0, 0.0)


def _jsonl(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "inputs.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_read_inputs_ids_from_field_or_content_hash(tmp_path: Path) -> None:
    path = _jsonl(
        tmp_path,
        json.dumps({"id": "a", "text": "hello"}),
        "",
        json.dumps({"text": "world"}),
        json.dumps({"id": 7, "text": "seven"}),
    )
    items = read_inputs(path)
    assert items == [Item("a", "hello"), Item(text_id("world"), "world"), Item("7", "seven")]
    assert read_inputs(path) == items


@pytest.mark.parametrize(
    ("lines", "where", "what"),
    [
        (['{"text": "ok"}', "{nope"], ":2", "not valid JSON"),
        (['{"id": "a"}'], ":1", '"text"'),
        (['{"text": "   "}'], ":1", '"text"'),
        (['"just a string"'], ":1", '"text"'),
        (['{"id": "a", "text": "x"}', '{"id": "a", "text": "y"}'], ":2", "repeats id"),
        (['{"text": "same"}', '{"text": "same"}'], ":2", "repeats id"),
    ],
)
def test_read_inputs_errors_name_the_line(
    tmp_path: Path, lines: list[str], where: str, what: str
) -> None:
    path = _jsonl(tmp_path, *lines)
    with pytest.raises(ConfigError) as info:
        read_inputs(path)
    assert f"{path}{where}" in str(info.value)
    assert what in str(info.value)


def test_read_inputs_empty_and_missing(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no inputs"):
        read_inputs(_jsonl(tmp_path, "", "  "))
    with pytest.raises(ConfigError, match="not found"):
        read_inputs(tmp_path / "missing.jsonl")
