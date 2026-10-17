"""End-to-end: the CLI and the pipeline on a reduced copy of the toy task, fully offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from transformers import AutoTokenizer
from typer.testing import CliRunner

from anvil import __version__
from anvil.cli import app
from anvil.errors import ConfigError, StoreError
from anvil.pipeline import Workspace, evaluate, label, train
from anvil.spec import TaskSpec, load_spec
from anvil.store import Split, read_inputs, split_of
from anvil.teacher import ClaudeTeacher, DiskCache, Failure
from anvil.teacher.fake import FakeTeacher
from tests.conftest import TOY_INPUTS, TOY_SPEC, FakeTransport, message

N_INPUTS = 160


@pytest.fixture
def toy_copy(tmp_path: Path) -> tuple[Path, Path]:
    doc = yaml.safe_load(TOY_SPEC.read_text(encoding="utf-8"))
    doc["student"]["epochs"] = 3
    spec = tmp_path / "anvil.yaml"
    spec.write_text(yaml.safe_dump(doc), encoding="utf-8")
    inputs = tmp_path / "inputs.jsonl"
    lines = TOY_INPUTS.read_text(encoding="utf-8").splitlines()[:N_INPUTS]
    inputs.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return spec, inputs


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "home"
    monkeypatch.setenv("ANVIL_HOME", str(path))
    return path


def _heldout_count(inputs: Path) -> int:
    return sum(split_of(it.text, 0, 0.25) == Split.HELDOUT for it in read_inputs(inputs))


def test_label_train_eval(toy_copy: tuple[Path, Path], home: Path) -> None:
    spec, inputs = toy_copy
    runner = CliRunner()
    label_args = ["label", "-s", str(spec), "-i", str(inputs), "-n", "80", "--teacher", "fake"]
    first = runner.invoke(app, label_args)
    assert first.exit_code == 0, first.output
    heldout = _heldout_count(inputs)
    assert f"labelled {heldout} held-out + 80 pool" in first.stdout

    trained = runner.invoke(app, ["train", "-s", str(spec), "--teacher", "fake"])
    assert trained.exit_code == 0, trained.output
    assert "on 80 fake-rules-" in trained.stdout
    assert " labels: 3 epochs" in trained.stdout

    evaluated = runner.invoke(app, ["eval", "-s", str(spec)])
    assert evaluated.exit_code == 0, evaluated.output
    assert "kappa" in evaluated.stdout
    (report_path,) = (home / "ticket-triage" / "reports").glob("eval-*.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["n"] == heldout
    assert -1.0 <= report["kappa"] <= 1.0
    assert report["teacher_model"] == FakeTeacher.model_id_for(load_spec(spec))
    assert report["train_examples"] == 80

    again = runner.invoke(app, label_args)
    assert again.exit_code == 0, again.output
    assert "labelled 0 held-out + " in again.stdout
    assert f"{N_INPUTS} labels in store" in again.stdout


def test_label_without_key_is_a_config_error(toy_copy: tuple[Path, Path], home: Path) -> None:
    spec, inputs = toy_copy
    result = CliRunner().invoke(app, ["label", "-s", str(spec), "-i", str(inputs)])
    assert result.exit_code == 2
    assert "ANTHROPIC_API_KEY" in result.stderr


def test_train_before_label_fails(toy_copy: tuple[Path, Path], home: Path) -> None:
    result = CliRunner().invoke(app, ["train", "-s", str(toy_copy[0]), "--teacher", "fake"])
    assert result.exit_code == 1
    assert "anvil label" in result.stderr


def test_bad_spec_path(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["eval", "-s", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 2
    assert "not found" in result.stderr


def test_version() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_eval_after_prompt_edit_fails(toy_copy: tuple[Path, Path], home: Path) -> None:
    spec, inputs = toy_copy
    runner = CliRunner()
    args = ["-s", str(spec), "--teacher", "fake"]
    assert runner.invoke(app, ["label", "-i", str(inputs), "-n", "20", *args]).exit_code == 0
    assert runner.invoke(app, ["train", *args]).exit_code == 0
    doc = yaml.safe_load(spec.read_text(encoding="utf-8"))
    doc["instructions"] += "Prefer bug when unsure.\n"
    spec.write_text(yaml.safe_dump(doc), encoding="utf-8")
    result = runner.invoke(app, ["eval", "-s", str(spec)])
    assert result.exit_code == 1
    assert "relabel and retrain" in result.stderr


def test_eval_after_split_change_fails(toy_copy: tuple[Path, Path], home: Path) -> None:
    spec, inputs = toy_copy
    runner = CliRunner()
    args = ["-s", str(spec), "--teacher", "fake"]
    assert runner.invoke(app, ["label", "-i", str(inputs), "-n", "20", *args]).exit_code == 0
    assert runner.invoke(app, ["train", *args]).exit_code == 0
    doc = yaml.safe_load(spec.read_text(encoding="utf-8"))
    doc["splits"]["heldout_fraction"] = 0.75
    spec.write_text(yaml.safe_dump(doc), encoding="utf-8")
    result = runner.invoke(app, ["eval", "-s", str(spec)])
    assert result.exit_code == 1
    assert "retrain before evaluating" in result.stderr


def test_pipeline_library(toy_copy: tuple[Path, Path], tmp_path: Path) -> None:
    spec_path, inputs = toy_copy
    spec = load_spec(spec_path)
    ws = Workspace.for_task(tmp_path / "lib", spec)
    teacher = FakeTeacher(spec)

    first = label(spec, ws, teacher, inputs, pool_n=30, heldout_max=10)
    assert (first.inputs_added, first.heldout_labelled, first.pool_labelled) == (N_INPUTS, 10, 30)
    assert first.total_labels == 40
    assert first.spent_usd == 0.0
    second = label(spec, ws, teacher, None, pool_n=10, heldout_max=15)
    assert (second.inputs_added, second.heldout_labelled, second.pool_labelled) == (0, 5, 10)
    third = label(spec, ws, teacher, inputs, pool_n=0, heldout_max=15)
    assert (third.inputs_added, third.heldout_labelled, third.total_labels) == (0, 0, 55)

    trained = train(spec, ws, teacher.model_id)
    assert trained.result.examples == 40
    assert ws.latest_student() == trained.path
    meta = json.loads((trained.path / "anvil_student.json").read_text(encoding="utf-8"))
    assert meta["prompt_hash"] == spec.prompt_hash
    assert len(meta["train_labels"]) == 40

    report = evaluate(spec, ws)
    assert report.agreement.n == 15
    assert report.report_path.exists()
    assert report.student == trained.run_id


def test_train_names_the_teachers_that_have_labels(
    toy_copy: tuple[Path, Path], home: Path
) -> None:
    spec_path, inputs = toy_copy
    runner = CliRunner()
    args = ["-s", str(spec_path), "--teacher", "fake"]
    assert runner.invoke(app, ["label", "-i", str(inputs), "-n", "20", *args]).exit_code == 0
    result = runner.invoke(app, ["train", "-s", str(spec_path)])
    assert result.exit_code == 1
    assert "no pool labels from claude-opus-5-5" in result.stderr
    assert "labels from fake-rules-" in result.stderr
    assert "--teacher" in result.stderr


def test_eval_refuses_a_student_from_another_task(toy_copy: tuple[Path, Path], home: Path) -> None:
    spec_path, inputs = toy_copy
    runner = CliRunner()
    args = ["-s", str(spec_path), "--teacher", "fake"]
    assert runner.invoke(app, ["label", "-i", str(inputs), "-n", "20", *args]).exit_code == 0
    assert runner.invoke(app, ["train", *args]).exit_code == 0
    doc = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    doc["name"] = "other-task"
    spec_path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    student = home / "ticket-triage" / "students" / "LATEST"
    student_dir = home / "ticket-triage" / "students" / student.read_text().strip()
    result = runner.invoke(app, ["eval", "-s", str(spec_path), "--student", str(student_dir)])
    assert result.exit_code == 1
    assert "trained for task 'ticket-triage'" in result.stderr


def test_eval_before_train_fails(toy_copy: tuple[Path, Path], home: Path) -> None:
    result = CliRunner().invoke(app, ["eval", "-s", str(toy_copy[0])])
    assert result.exit_code == 1
    assert "anvil train" in result.stderr


def test_eval_without_heldout_labels_fails(toy_copy: tuple[Path, Path], tmp_path: Path) -> None:
    spec_path, inputs = toy_copy
    spec = load_spec(spec_path)
    ws = Workspace.for_task(tmp_path / "lib", spec)
    label(spec, ws, FakeTeacher(spec), inputs, pool_n=20, heldout_max=0)
    train(spec, ws, FakeTeacher(spec).model_id)
    with pytest.raises(StoreError, match="held-out"):
        evaluate(spec, ws)


def test_teacher_failures_are_persisted_billed_and_retried(
    toy_copy: tuple[Path, Path], tmp_path: Path
) -> None:
    spec_path, inputs = toy_copy
    spec = load_spec(spec_path)
    ws = Workspace.for_task(tmp_path / "lib", spec)
    with ws.open_store(spec) as store:
        store.add_inputs(read_inputs(inputs))
        first = store.inputs(Split.POOL)[0]
    transport = FakeTransport()
    refusal = {**message(""), "stop_reason": "refusal", "stop_details": {"category": "cyber"}}
    transport.program(first.text, refusal)
    teacher = ClaudeTeacher(spec, transport, cache=DiskCache(tmp_path / "cache"), workers=2)

    report = label(spec, ws, teacher, None, pool_n=3, heldout_max=0)
    assert report.failures == [Failure(first.id, "refusal (cyber)", 1000, 50, 0.005)]
    assert report.pool_labelled == 2
    assert report.spent_usd == pytest.approx(0.015)
    with ws.open_store(spec) as store:
        stats = store.stats(teacher.model_id, spec.prompt_hash)
        assert stats.labels == 2
        assert stats.spent_usd == pytest.approx(0.015)
        assert first.id not in store.labels(teacher.model_id, spec.prompt_hash)
    assert report.total_label_cost_usd == pytest.approx(0.015)

    # The refused input is first in line again and, now answered, gets its label.
    del transport.responses[first.text]
    again = label(spec, ws, teacher, None, pool_n=3, heldout_max=0)
    assert again.failures == []
    assert again.pool_labelled == 3
    with ws.open_store(spec) as store:
        assert first.id in store.labels(teacher.model_id, spec.prompt_hash)
        # 2 + 3 labels and the one billed refusal, at $0.005 each.
        assert store.stats(teacher.model_id, spec.prompt_hash).spent_usd == pytest.approx(0.030)


def test_cli_label_reports_failures_and_exits_1(
    toy_copy: tuple[Path, Path], home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path, inputs = toy_copy
    truncated = message('{"lab', stop_reason="max_tokens")
    teacher = ClaudeTeacher(load_spec(spec_path), FakeTransport(truncated), workers=2)
    monkeypatch.setattr("anvil.teacher.build_teacher", lambda *a, **k: teacher)
    result = CliRunner().invoke(
        app, ["label", "-s", str(spec_path), "-i", str(inputs), "-n", "5", "--heldout-max", "2"]
    )
    assert result.exit_code == 1
    assert "labelled 0 held-out + 0 pool inputs with claude-opus-5-5" in result.stdout
    assert "$0.0350 spent" in result.stdout
    assert "max_tokens" in result.stderr
    assert "7 inputs failed; re-run to retry them" in result.stderr


def test_aborted_label_run_keeps_paid_answers(
    toy_copy: tuple[Path, Path], home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path, inputs = toy_copy
    spec = load_spec(spec_path)
    ws = Workspace.for_task(home, spec)
    with ws.open_store(spec) as store:
        store.add_inputs(read_inputs(inputs))
        second = store.inputs(Split.POOL)[1]
    transport = FakeTransport()
    transport.errors[second.text] = ConfigError("the API key cannot use claude-opus-5-5")
    teacher = ClaudeTeacher(spec, transport, cache=DiskCache(home / "cache"), workers=1)
    monkeypatch.setattr("anvil.teacher.build_teacher", lambda *a, **k: teacher)
    result = CliRunner().invoke(
        app, ["label", "-s", str(spec_path), "-n", "4", "--heldout-max", "0"]
    )
    assert result.exit_code == 2
    assert "cannot use claude-opus-5-5" in result.stderr
    with ws.open_store(spec) as store:
        stats = store.stats("claude-opus-5-5", spec.prompt_hash)
    # Every input answered (and billed) before the run stopped is in the store: the first pool
    # input for sure, plus any the worker picked up before the cancellation landed.
    answered = len(transport.calls) - 1
    assert answered >= 1
    assert stats.labels == answered
    assert stats.spent_usd == pytest.approx(0.005 * answered)


def _texts_on(split: Split, n: int, prefix: str) -> list[str]:
    found: list[str] = []
    k = 0
    while len(found) < n:
        text = f"{prefix} ticket number {k} crash"
        if split_of(text, 0, 0.25) == split:
            found.append(text)
        k += 1
    return found


def test_tiny_vocabulary_never_sees_heldout_text(small_spec: TaskSpec, tmp_path: Path) -> None:
    heldout = _texts_on(Split.HELDOUT, 2, "zebra")
    pool = _texts_on(Split.POOL, 6, "yak")
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text("".join(json.dumps({"text": t}) + "\n" for t in heldout + pool))
    ws = Workspace.for_task(tmp_path / "lib", small_spec)
    label(small_spec, ws, FakeTeacher(small_spec), inputs, pool_n=6)
    trained = train(small_spec, ws, FakeTeacher.model_id_for(small_spec))
    vocab = AutoTokenizer.from_pretrained(trained.path).get_vocab()
    assert "yak" in vocab
    assert "zebra" not in vocab
