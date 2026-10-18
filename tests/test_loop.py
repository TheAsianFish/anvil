"""The active-learning loop (`anvil.loop`), its store tables, `min_steps` and the student's
embeddings, on a reduced copy of the toy task with the fake teacher and the tiny student."""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from collections import Counter
from collections.abc import Callable, Collection, Sequence
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

import anvil.loop as loop_mod
from anvil.cli import app
from anvil.errors import ConfigError, StoreError
from anvil.loop import (
    FRONTIER,
    STOP_DOLLARS,
    STOP_LABELS,
    STOP_POOL,
    STOP_TARGET,
    LoopConfig,
    LoopReport,
    read_frontier,
    round_seed,
    run_loop,
)
from anvil.pipeline import Workspace
from anvil.pipeline import fit_student as real_fit_student
from anvil.select import Pool
from anvil.select import select_batch as real_select_batch
from anvil.spec import StudentSpec, TaskSpec
from anvil.store import Split, Store, read_inputs
from anvil.students.encoder import EncoderClassifier
from anvil.teacher.base import Failure, Item, Label, LabelBatch
from anvil.teacher.fake import FakeTeacher
from anvil.train.classifier import train_classifier
from anvil.train.loop import TrainResult
from tests.conftest import TOY_INPUTS, TOY_SPEC

N_INPUTS = 160
BUDGETS = (30, 60, 90)
TIMING = ("train_seconds", "select_seconds", "spent_usd")

FitStudent = Callable[..., tuple[EncoderClassifier, TrainResult]]


# --- fixtures and helpers -------------------------------------------------------------------


def _spec_doc(**student: object) -> dict[str, object]:
    doc: dict[str, object] = yaml.safe_load(TOY_SPEC.read_text(encoding="utf-8"))
    overrides = {"epochs": 6, "max_length": 32, "validation_fraction": 0.0, **student}
    student_doc = doc["student"]
    assert isinstance(student_doc, dict)
    student_doc.update(overrides)
    return doc


def _spec(**student: object) -> TaskSpec:
    return TaskSpec.model_validate(_spec_doc(**student))


@pytest.fixture
def spec() -> TaskSpec:
    return _spec()


@pytest.fixture
def inputs(tmp_path: Path) -> Path:
    path = tmp_path / "inputs.jsonl"
    lines = TOY_INPUTS.read_text(encoding="utf-8").splitlines()[:N_INPUTS]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def ws(tmp_path: Path, spec: TaskSpec) -> Workspace:
    return Workspace.for_task(tmp_path / "home", spec)


def _config(**kwargs: object) -> LoopConfig:
    base: dict[str, object] = {"strategy": "random", "budgets": BUDGETS, "seed": 0}
    base.update(kwargs)
    return LoopConfig(**base)  # type: ignore[arg-type]


def _loop(
    spec: TaskSpec,
    ws: Workspace,
    inputs: Path,
    loop_id: str,
    config: LoopConfig | None = None,
    teacher: FakeTeacher | None = None,
) -> LoopReport:
    return run_loop(
        spec,
        ws,
        teacher or FakeTeacher(spec),
        config or _config(),
        loop_id=loop_id,
        inputs=inputs,
    )


def _picks(spec: TaskSpec, ws: Workspace, loop_id: str) -> list[tuple[int, str]]:
    with ws.open_store(spec) as store:
        return store.picks(loop_id)


def _ingest(spec: TaskSpec, ws: Workspace, inputs: Path) -> None:
    with ws.open_store(spec) as store:
        store.add_inputs(read_inputs(inputs))


def _heldout_ids(spec: TaskSpec, ws: Workspace) -> set[str]:
    with ws.open_store(spec) as store:
        return {it.id for it in store.inputs(Split.HELDOUT)}


def _pool_size(spec: TaskSpec, ws: Workspace) -> int:
    with ws.open_store(spec) as store:
        return len(store.inputs(Split.POOL))


def _untimed(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [{k: v for k, v in row.items() if k not in TIMING and k != "loop_id"} for row in rows]


def _least_confident(pool: Pool, k: int, strategy: str, seed: int) -> list[int]:
    """Test stand-in for a real strategy: lowest max-probability first (ties by index)."""
    if strategy == "random":
        return real_select_batch(pool, k, strategy, seed)
    order = np.argsort(pool.probs.max(axis=1), kind="stable")
    return [int(i) for i in order[: min(k, len(pool))]]


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the loop run `uncertainty` with the deterministic stand-in above."""
    monkeypatch.setattr(loop_mod, "select_batch", _least_confident)
    monkeypatch.setattr(loop_mod, "implemented", lambda strategy: True)


class CostlyTeacher(FakeTeacher):
    """The fake teacher, but every answer is billed (never cached), at `usd` per call."""

    def __init__(self, spec: TaskSpec, usd: float = 0.001) -> None:
        super().__init__(spec)
        self.usd = usd

    def label(self, items: Sequence[Item]) -> LabelBatch:
        batch = super().label(items)
        return LabelBatch(
            labels=[
                dataclasses.replace(lb, cost_usd=self.usd, cached=False) for lb in batch.labels
            ]
        )


class RefusingTeacher(FakeTeacher):
    """The fake teacher, but it refuses every input in `refuse` (and logs every request)."""

    def __init__(self, spec: TaskSpec, refuse: set[str]) -> None:
        super().__init__(spec)
        self.refuse = refuse
        self.requests: list[str] = []

    def label(self, items: Sequence[Item]) -> LabelBatch:
        self.requests += [it.id for it in items]
        batch = super().label([it for it in items if it.id not in self.refuse])
        batch.failures = [Failure(it.id, "refusal") for it in items if it.id in self.refuse]
        return batch


# --- the loop end to end --------------------------------------------------------------------


def test_loop_end_to_end(spec: TaskSpec, ws: Workspace, inputs: Path) -> None:
    report = _loop(spec, ws, inputs, "e2e")

    assert [row["round"] for row in report.rows] == [0, 1, 2]
    assert [row["picked"] for row in report.rows] == list(BUDGETS)
    assert [row["labels"] for row in report.rows] == list(BUDGETS)
    assert [row["budget"] for row in report.rows] == list(BUDGETS)
    assert [row["selected"] for row in report.rows] == [30, 30, 30]
    assert [row["failures"] for row in report.rows] == [0, 0, 0]
    assert [row["stop"] for row in report.rows] == [None, None, STOP_LABELS]
    assert report.stopped == STOP_LABELS
    assert report.rows[0]["selected_confidence"] is None  # round 0 is a uniform draw
    assert isinstance(report.rows[1]["selected_confidence"], float)
    for row in report.rows:
        assert row["heldout"] == len(_heldout_ids(spec, ws))
        kappa, low, high = row["kappa"], row["kappa_low"], row["kappa_high"]
        assert isinstance(kappa, float)
        assert isinstance(low, float)
        assert isinstance(high, float)
        assert -1.0 <= low <= kappa <= high <= 1.0

    assert report.frontier == ws.root / "loops" / "e2e" / FRONTIER
    assert read_frontier(report.frontier) == json.loads(json.dumps(report.rows))
    students = sorted(p.name for p in report.root.iterdir() if p.is_dir())
    assert students == ["round-2"]
    assert report.student == report.root / "round-2"
    student = EncoderClassifier.load(report.student)
    assert len(student.meta["train_labels"]) == BUDGETS[-1]  # type: ignore[arg-type]
    assert student.predict_proba(["the app crashes"]).shape == (1, len(spec.label_names))


def test_every_round_retrains_on_the_union_of_picks(
    spec: TaskSpec, ws: Workspace, inputs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trained: list[list[str]] = []

    def spy(*args: object, **kwargs: object) -> tuple[EncoderClassifier, TrainResult]:
        pairs = args[2]
        assert isinstance(pairs, list)
        trained.append([it.id for it, _ in pairs])
        fit: FitStudent = real_fit_student
        return fit(*args, **kwargs)

    monkeypatch.setattr(loop_mod, "fit_student", spy)
    report = _loop(spec, ws, inputs, "union")

    picks = _picks(spec, ws, "union")
    assert len(trained) == len(BUDGETS)
    for r, ids in enumerate(trained):
        assert sorted(ids) == sorted(i for rnd, i in picks if rnd <= r)
    assert report.student is not None
    meta = EncoderClassifier.load(report.student).meta
    assert isinstance(meta["train_labels"], list)
    assert len(meta["train_labels"]) == len(picks) == BUDGETS[-1]


def test_round_zero_depends_only_on_the_seed(
    spec: TaskSpec, ws: Workspace, inputs: Path, stand_in: None
) -> None:
    two = _config(budgets=(30, 60))
    _loop(spec, ws, inputs, "random-s0", two)
    other = _loop(spec, ws, inputs, "least-s0", dataclasses.replace(two, strategy="uncertainty"))
    _loop(spec, ws, inputs, "random-s1", dataclasses.replace(two, seed=1))

    def round_picks(loop_id: str, r: int) -> list[str]:
        return [i for rnd, i in _picks(spec, ws, loop_id) if rnd == r]

    assert round_picks("random-s0", 0) == round_picks("least-s0", 0)
    random_row, least_row = _untimed(_loop(spec, ws, inputs, "random-s0", two).rows)[0], None
    least_row = _untimed(other.rows)[0]
    random_row.pop("strategy"), least_row.pop("strategy")
    assert random_row == least_row  # the shared draw gives the shared round-0 student
    assert set(round_picks("random-s0", 0)) != set(round_picks("random-s1", 0))
    assert round_picks("random-s0", 1) != round_picks("least-s0", 1)

    # The stand-in picked the round-0 student's least confident inputs, and the row says so.
    confidence = other.rows[1]["selected_confidence"]
    assert isinstance(confidence, float)
    assert 0.0 < confidence <= 1.0


def test_picks_are_nested_disjoint_and_never_held_out(
    spec: TaskSpec, ws: Workspace, inputs: Path
) -> None:
    _loop(spec, ws, inputs, "nested")
    picks = _picks(spec, ws, "nested")
    ids = [i for _, i in picks]
    assert len(ids) == len(set(ids)) == BUDGETS[-1]
    assert [rnd for rnd, _ in picks] == sorted(rnd for rnd, _ in picks)
    for r, budget in enumerate(BUDGETS):
        assert sum(1 for rnd, _ in picks if rnd <= r) == budget
    assert not set(ids) & _heldout_ids(spec, ws)


# --- resume ---------------------------------------------------------------------------------


def test_interrupted_loop_resumes_to_the_same_rows(
    spec: TaskSpec, ws: Workspace, inputs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"failed": False}

    def flaky(*args: object, **kwargs: object) -> tuple[EncoderClassifier, TrainResult]:
        metrics = args[4]
        assert isinstance(metrics, Path)
        if metrics.name == "round-1.jsonl" and not calls["failed"]:
            calls["failed"] = True
            raise RuntimeError("interrupted")
        fit: FitStudent = real_fit_student
        return fit(*args, **kwargs)

    monkeypatch.setattr(loop_mod, "fit_student", flaky)
    with pytest.raises(RuntimeError, match="interrupted"):
        _loop(spec, ws, inputs, "resumed")
    assert len(read_frontier(ws.root / "loops" / "resumed" / FRONTIER)) == 1
    assert sum(1 for rnd, _ in _picks(spec, ws, "resumed") if rnd == 1) == 30

    resumed = _loop(spec, ws, inputs, "resumed")
    straight = _loop(spec, ws, inputs, "straight")
    assert resumed.stopped == straight.stopped == STOP_LABELS
    assert _untimed(resumed.rows) == _untimed(straight.rows)
    assert [i for _, i in _picks(spec, ws, "resumed")] == [
        i for _, i in _picks(spec, ws, "straight")
    ]


def test_picks_are_recorded_before_their_labels_are_bought(
    spec: TaskSpec, ws: Workspace, inputs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash while the teacher labels round 1 leaves its picks in the store; the resume
    labels exactly those picks and never selects again."""
    teacher = FakeTeacher(spec)
    real_label = teacher.label
    seen: list[int] = []

    def crashing(items: Sequence[Item]) -> LabelBatch:
        seen.append(len(items))
        if len(seen) == 3:  # held-out, round 0, then round 1
            raise RuntimeError("teacher down")
        return real_label(items)

    monkeypatch.setattr(teacher, "label", crashing)
    with pytest.raises(RuntimeError, match="teacher down"):
        _loop(spec, ws, inputs, "picked", teacher=teacher)
    picks = _picks(spec, ws, "picked")
    assert sum(1 for rnd, _ in picks if rnd == 1) == 30
    with ws.open_store(spec) as store:
        pool_labels = store.labelled(teacher.model_id, spec.prompt_hash, Split.POOL)
    assert len(pool_labels) == 30  # round 0 only: round 1 was picked, not labelled

    selections: list[int] = []

    def counted(pool: Pool, k: int, strategy: str, seed: int) -> list[int]:
        selections.append(seed)
        return real_select_batch(pool, k, strategy, seed)

    monkeypatch.setattr(loop_mod, "select_batch", counted)
    resumed = _loop(spec, ws, inputs, "picked", teacher=FakeTeacher(spec))
    assert selections == [round_seed(0, 2)]  # round 1 reused its picks; only round 2 selected
    assert [i for _, i in _picks(spec, ws, "picked")][:60] == [i for _, i in picks]
    monkeypatch.setattr(loop_mod, "select_batch", real_select_batch)
    straight = _loop(spec, ws, inputs, "straight")
    assert _untimed(resumed.rows) == _untimed(straight.rows)


def test_resume_lost_previous_student_is_a_store_error(
    spec: TaskSpec, ws: Workspace, inputs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(*args: object, **kwargs: object) -> tuple[EncoderClassifier, TrainResult]:
        metrics = args[4]
        assert isinstance(metrics, Path)
        if metrics.name == "round-1.jsonl":
            raise RuntimeError("interrupted")
        fit: FitStudent = real_fit_student
        return fit(*args, **kwargs)

    monkeypatch.setattr(loop_mod, "fit_student", interrupted)
    with pytest.raises(RuntimeError):
        _loop(spec, ws, inputs, "lost")
    import shutil

    shutil.rmtree(ws.root / "loops" / "lost" / "round-0")
    monkeypatch.setattr(loop_mod, "fit_student", real_fit_student)
    with pytest.raises(StoreError, match="lost the round 0 student"):
        _loop(spec, ws, inputs, "lost")


def test_finished_loop_returns_without_retraining(
    spec: TaskSpec, ws: Workspace, inputs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _loop(spec, ws, inputs, "done", _config(budgets=(30, 60)))

    def boom(*args: object, **kwargs: object) -> tuple[EncoderClassifier, TrainResult]:
        raise AssertionError("a finished loop retrained")

    monkeypatch.setattr(loop_mod, "fit_student", boom)
    again = _loop(spec, ws, inputs, "done", _config(budgets=(30, 60)))
    assert again.rows == json.loads(json.dumps(first.rows))
    assert again.stopped == first.stopped == STOP_LABELS
    assert again.student == first.student


@pytest.mark.parametrize(
    ("change", "setting"),
    [({"seed": 1}, "seed"), ({"budgets": (30, 90)}, "budgets"), ({"max_usd": 1.0}, "max_usd")],
)
def test_resume_with_other_settings_is_a_config_error(
    spec: TaskSpec,
    ws: Workspace,
    inputs: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, object],
    setting: str,
) -> None:
    def interrupted(*args: object, **kwargs: object) -> tuple[EncoderClassifier, TrainResult]:
        raise RuntimeError("interrupted")

    monkeypatch.setattr(loop_mod, "fit_student", interrupted)
    with pytest.raises(RuntimeError):
        _loop(spec, ws, inputs, "fixed", _config(budgets=(30, 60)))
    with pytest.raises(ConfigError, match=setting):
        _loop(spec, ws, inputs, "fixed", _config(**{"budgets": (30, 60), **change}))


# --- stopping rules -------------------------------------------------------------------------


def test_dollar_budget_truncates_the_last_round_and_stops(
    spec: TaskSpec, ws: Workspace, inputs: Path
) -> None:
    teacher = CostlyTeacher(spec, usd=0.001)
    _ingest(spec, ws, inputs)
    heldout = len(_heldout_ids(spec, ws))
    max_usd = 0.001 * (heldout + 45)  # round 0 (30) fits, round 1 only half of its 30
    report = _loop(spec, ws, inputs, "dollars", _config(max_usd=max_usd), teacher)

    assert [row["picked"] for row in report.rows] == [30, 45]
    assert [row["labels"] for row in report.rows] == [30, 45]
    assert report.rows[-1]["stop"] == STOP_DOLLARS
    assert report.stopped == STOP_DOLLARS
    for row in report.rows:
        pool_cost, heldout_cost = row["label_cost_usd"], row["heldout_cost_usd"]
        assert isinstance(pool_cost, float)
        assert isinstance(heldout_cost, float)
        assert pool_cost + heldout_cost <= max_usd + 1e-9
    assert report.rows[-1]["heldout_cost_usd"] == pytest.approx(0.001 * heldout)
    with ws.open_store(spec) as store:
        assert store.stats(teacher.model_id, spec.prompt_hash).cost_usd <= max_usd + 1e-9


class ReplayTeacher(FakeTeacher):
    """A recording: every answer costs `usd` but is `cached` (paid for when it was made), and
    the inputs in `refuse` were refused then, at the same price."""

    def __init__(self, spec: TaskSpec, refuse: Collection[str] = (), usd: float = 0.001) -> None:
        super().__init__(spec)
        self.refuse = refuse
        self.usd = usd

    def label(self, items: Sequence[Item]) -> LabelBatch:
        batch = super().label([it for it in items if it.id not in self.refuse])
        return LabelBatch(
            labels=[
                dataclasses.replace(lb, cost_usd=self.usd, cached=True) for lb in batch.labels
            ],
            failures=[
                Failure(it.id, "refusal", cost_usd=self.usd, cached=True)
                for it in items
                if it.id in self.refuse
            ],
        )


def test_dollar_budget_counts_replayed_label_cost_not_spend(
    spec: TaskSpec, ws: Workspace, inputs: Path
) -> None:
    """A replay stops where the live run did, although it spends nothing."""
    _ingest(spec, ws, inputs)
    heldout = len(_heldout_ids(spec, ws))
    config = _config(max_usd=0.001 * (heldout + 45))
    report = _loop(spec, ws, inputs, "replay", config, ReplayTeacher(spec))
    assert [row["picked"] for row in report.rows] == [30, 45]
    assert report.stopped == STOP_DOLLARS
    assert [row["spent_usd"] for row in report.rows] == [0.0, 0.0]


def test_resumed_replay_keeps_the_cost_of_recorded_refusals(
    spec: TaskSpec, ws: Workspace, inputs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded refusal is never stored (`Store.append_failures`), so after a crash its cost
    is known again only once the resumed loop re-labels its earlier picks; a resume that
    forgot it would buy more labels than the budget allows."""
    _ingest(spec, ws, inputs)
    with ws.open_store(spec) as store:
        pool = [it.id for it in store.inputs(Split.POOL)]
    refuse = set(pool[::3])
    heldout = len(_heldout_ids(spec, ws))
    config = _config(max_usd=0.001 * (heldout + 45))
    straight = _loop(spec, ws, inputs, "straight", config, ReplayTeacher(spec, refuse))
    assert [row["picked"] for row in straight.rows] == [30, 45]

    calls = {"select": 0}

    def crash_at_round_one(pool: Pool, k: int, strategy: str, seed: int) -> list[int]:
        calls["select"] += 1
        if calls["select"] == 2:
            raise RuntimeError("interrupted")
        return real_select_batch(pool, k, strategy, seed)

    monkeypatch.setattr(loop_mod, "select_batch", crash_at_round_one)
    with pytest.raises(RuntimeError, match="interrupted"):
        _loop(spec, ws, inputs, "resumed", config, ReplayTeacher(spec, refuse))
    monkeypatch.setattr(loop_mod, "select_batch", real_select_batch)
    resumed = _loop(spec, ws, inputs, "resumed", config, ReplayTeacher(spec, refuse))
    assert [row["picked"] for row in resumed.rows] == [30, 45]
    assert _untimed(resumed.rows) == _untimed(straight.rows)


def test_dollar_truncation_of_the_last_round_is_the_dollar_stop(
    spec: TaskSpec, ws: Workspace, inputs: Path
) -> None:
    _ingest(spec, ws, inputs)
    heldout = len(_heldout_ids(spec, ws))
    config = _config(budgets=(30, 60), max_usd=0.001 * (heldout + 45))
    report = _loop(spec, ws, inputs, "cut", config, CostlyTeacher(spec, usd=0.001))
    assert [row["picked"] for row in report.rows] == [30, 45]
    assert report.rows[-1]["budget"] == 60
    assert report.stopped == report.rows[-1]["stop"] == STOP_DOLLARS


def test_free_failures_are_asked_again_next_run(
    spec: TaskSpec, ws: Workspace, inputs: Path
) -> None:
    """A failure that cost nothing (a dropped connection) settles nothing: the next run of
    the loop asks the teacher again, as `anvil label` does."""
    _ingest(spec, ws, inputs)
    with ws.open_store(spec) as store:
        pool = [it.id for it in store.inputs(Split.POOL)]
    flaky = RefusingTeacher(spec, refuse=set(pool))  # every pool input fails, for free
    with pytest.raises(StoreError, match="labelled none of the picked inputs"):
        _loop(spec, ws, inputs, "flaky", _config(budgets=(30,)), flaky)
    healed = RefusingTeacher(spec, refuse=set())
    report = _loop(spec, ws, inputs, "flaky", _config(budgets=(30,)), healed)
    assert report.rows[0]["labels"] == 30
    assert len(healed.requests) == 30  # the 30 picks, asked again; the held-out set was fine


def test_dollar_budget_below_the_first_round_is_a_store_error(
    spec: TaskSpec, ws: Workspace, inputs: Path
) -> None:
    _ingest(spec, ws, inputs)
    heldout = len(_heldout_ids(spec, ws))
    config = _config(max_usd=0.001 * heldout + 0.0005)
    with pytest.raises(StoreError, match="dollar budget before the first round"):
        _loop(spec, ws, inputs, "broke", config, CostlyTeacher(spec, usd=0.001))
    assert _picks(spec, ws, "broke") == []


def test_target_kappa_stops_after_round_zero(spec: TaskSpec, ws: Workspace, inputs: Path) -> None:
    trained = _spec(epochs=8)
    report = _loop(trained, ws, inputs, "target", _config(budgets=(60, 90), target_kappa=0.01))
    assert len(report.rows) == 1
    low = report.rows[0]["kappa_low"]
    assert isinstance(low, float)
    assert low >= 0.01
    assert report.rows[0]["stop"] == report.stopped == STOP_TARGET


def test_budgets_beyond_the_pool_stop_at_the_pool(
    spec: TaskSpec, ws: Workspace, inputs: Path
) -> None:
    report = _loop(spec, ws, inputs, "big", _config(budgets=(60, 1000, 2000)))
    pool = _pool_size(spec, ws)
    assert pool < 1000
    assert [row["picked"] for row in report.rows] == [60, pool]
    assert report.stopped == report.rows[-1]["stop"] == STOP_POOL
    assert len(_picks(spec, ws, "big")) == pool


def test_teacher_refusals_are_counted_and_never_picked_again(
    spec: TaskSpec, ws: Workspace, inputs: Path
) -> None:
    _ingest(spec, ws, inputs)
    with ws.open_store(spec) as store:
        pool_ids = [it.id for it in store.inputs(Split.POOL)]
    refuse = set(pool_ids[::3])
    teacher = RefusingTeacher(spec, refuse)
    report = _loop(spec, ws, inputs, "refused", _config(), teacher)

    picks = _picks(spec, ws, "refused")
    for r, row in enumerate(report.rows):
        so_far = [i for rnd, i in picks if rnd <= r]
        refused = sum(1 for i in so_far if i in refuse)
        assert refused > 0
        assert row["failures"] == refused
        assert row["picked"] == len(so_far) == BUDGETS[r]
        assert row["labels"] == len(so_far) - refused
    ids = [i for _, i in picks]
    assert len(ids) == len(set(ids))  # a refused pick is not picked again
    # Every input, refused or not, reaches the teacher exactly once across all rounds.
    asked = Counter(teacher.requests)
    assert set(asked) == set(ids) | _heldout_ids(spec, ws)
    assert set(asked.values()) == {1}
    assert any(i in refuse for i in ids)
    assert report.student is not None
    meta = EncoderClassifier.load(report.student).meta
    assert isinstance(meta["train_labels"], list)
    assert len(meta["train_labels"]) == report.rows[-1]["labels"]


# --- configuration --------------------------------------------------------------------------


def test_unimplemented_strategy_fails_before_any_work(
    spec: TaskSpec, ws: Workspace, inputs: Path
) -> None:
    with pytest.raises(ConfigError, match="--strategy random"):
        _loop(spec, ws, inputs, "nope", _config(strategy="uncertainty"))
    assert not (ws.root / "loops").exists()
    if ws.db.exists():
        with ws.open_store(spec) as store:
            assert store.labels(FakeTeacher.model_id_for(spec), spec.prompt_hash) == {}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"strategy": "psychic"}, "unknown strategy"),
        ({"budgets": ()}, "positive"),
        ({"budgets": (0, 10)}, "positive"),
        ({"budgets": (10, 10)}, "increase strictly"),
        ({"budgets": (20, 10)}, "increase strictly"),
        ({"max_usd": 0.0}, "dollar budget"),
        ({"target_kappa": 0.0}, "target kappa"),
        ({"target_kappa": 1.5}, "target kappa"),
        ({"heldout_max": 0}, "heldout_max"),
    ],
)
def test_bad_loop_config(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        _config(**kwargs)


def test_round_seed_is_stable_and_distinct() -> None:
    assert round_seed(0, 0) == round_seed(0, 0)
    seeds = {round_seed(s, r) for s in range(3) for r in range(5)}
    assert len(seeds) == 15
    assert all(0 <= s < 2**32 for s in seeds)


# --- min_steps ------------------------------------------------------------------------------


def _tiny_set() -> tuple[list[str], list[int]]:
    words = ["refund invoice", "crash error", "wish add", "password login"]
    texts = [f"{words[k % 4]} please {k}" for k in range(16)]
    return texts, [k % 4 for k in range(16)]


@pytest.mark.parametrize(("min_steps", "epochs"), [(0, 2), (5, 3), (20, 10)])
def test_min_steps_raises_epochs_on_a_tiny_set(min_steps: int, epochs: int) -> None:
    texts, targets = _tiny_set()
    spec = StudentSpec(
        model="tiny", max_length=16, epochs=2, batch_size=8, validation_fraction=0.0,
        min_steps=min_steps,
    )  # fmt: skip
    student = EncoderClassifier.build(spec, ["billing", "bug", "feature", "account"], texts)
    result = train_classifier(student, texts, targets, spec)
    assert result.epochs == epochs
    assert result.steps == epochs * 2  # 16 examples in batches of 8


# --- predict_with_embeddings ----------------------------------------------------------------


def test_predict_with_embeddings() -> None:
    corpus = ["the app crashes", "refund my invoice", "please add dark mode", "reset password"]
    spec = StudentSpec(model="tiny", max_length=64, seed=0)
    student = EncoderClassifier.build(spec, ["a", "b", "c"], corpus)
    probs, vectors = student.predict_with_embeddings(corpus, batch_size=3)
    assert np.allclose(probs, student.predict_proba(corpus), atol=1e-6)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    hidden = int(student.model.config.hidden_size)
    assert vectors.shape == (len(corpus), hidden)
    assert probs.dtype == vectors.dtype == np.float32

    short = "the app crashes"
    long = " ".join(corpus * 4)
    _, alone = student.predict_with_embeddings([short])
    _, padded = student.predict_with_embeddings([short, long])
    assert np.allclose(alone[0], padded[0], atol=1e-5)
    assert not np.allclose(padded[0], padded[1], atol=1e-3)

    empty_probs, empty_vectors = student.predict_with_embeddings([])
    assert empty_probs.shape == (0, 3)
    assert empty_vectors.shape == (0, hidden)


# --- store ----------------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "db" / "store.sqlite", split_seed=0, heldout_fraction=0.25)
    s.add_inputs([Item(f"i{k}", f"text {k}") for k in range(6)])
    return s


def test_picks_order_by_round_then_rank(store: Store) -> None:
    with store:
        store.create_loop("L", {"strategy": "random"})
        store.add_picks("L", 1, ["i5", "i0"])
        store.add_picks("L", 0, ["i3", "i1"])
        assert store.picks("L") == [(0, "i3"), (0, "i1"), (1, "i5"), (1, "i0")]
        assert store.picks("other") == []
        assert store.loop_config("L") == {"strategy": "random"}
        assert store.loop_config("other") is None


def test_loop_picks_are_append_only(store: Store) -> None:
    with store:
        store.create_loop("L", {})
        store.add_picks("L", 0, ["i1", "i2"])
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            store._db.execute("UPDATE loop_picks SET round = 3")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            store._db.execute("DELETE FROM loop_picks")
        with pytest.raises(sqlite3.IntegrityError):
            store.add_picks("L", 1, ["i1"])  # an input is picked at most once per loop
        assert store.picks("L") == [(0, "i1"), (0, "i2")]


def test_create_loop_twice_is_a_store_error(store: Store) -> None:
    with store:
        store.create_loop("L", {"seed": 0})
        with pytest.raises(StoreError, match="already exists"):
            store.create_loop("L", {"seed": 1})
        assert store.loop_config("L") == {"seed": 0}


def test_cost_of_sums_labels_and_failures_for_the_given_ids(store: Store) -> None:
    with store:
        store.append_labels(
            [
                Label("i0", "a", "m", "p", cost_usd=0.01),
                Label("i1", "a", "m", "p", cost_usd=0.02),
                Label("i2", "a", "other-model", "p", cost_usd=0.5),
            ]
        )
        store.append_failures(
            [Failure("i1", "refusal", cost_usd=0.003), Failure("i3", "refusal", cost_usd=0.004)],
            "m",
            "p",
        )
        assert store.cost_of("m", "p", ["i0", "i1", "i3"]) == pytest.approx(0.037)
        assert store.cost_of("m", "p", ["i0"]) == pytest.approx(0.01)
        assert store.cost_of("m", "p", ["i2", "i4"]) == 0.0
        assert store.cost_of("m", "p", []) == 0.0
        assert store.cost_of("other-model", "p", ["i2"]) == pytest.approx(0.5)


# --- CLI ------------------------------------------------------------------------------------


@pytest.fixture
def cli_spec(tmp_path: Path) -> Path:
    path = tmp_path / "anvil.yaml"
    path.write_text(yaml.safe_dump(_spec_doc()), encoding="utf-8")
    return path


def test_cli_loop(cli_spec: Path, inputs: Path, tmp_path: Path) -> None:
    args = ["loop", "-s", str(cli_spec), "-i", str(inputs), "--teacher", "fake"]
    result = CliRunner().invoke(app, [*args, "--budgets", "30,60", "--id", "cli"])
    assert result.exit_code == 0, result.output
    assert "loop cli (random, seed 0)" in result.stdout
    assert "round  labels  kappa" in result.stdout
    lines = [line.split() for line in result.stdout.splitlines()]
    assert [line[:2] for line in lines if line[:1] in (["0"], ["1"])] == [
        ["0", "30"],
        ["1", "60"],
    ]
    assert "stopped: label budget" in result.stdout
    assert (tmp_path / "anvil-home" / "ticket-triage" / "loops" / "cli" / FRONTIER).exists()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--strategy", "uncertainty"], "--strategy random"),
        (["--budgets", "10,5"], "increase strictly"),
    ],
)
def test_cli_loop_config_errors(
    cli_spec: Path, inputs: Path, extra: list[str], message: str
) -> None:
    args = ["loop", "-s", str(cli_spec), "-i", str(inputs), "--teacher", "fake", *extra]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2, result.output
    assert message in result.output
