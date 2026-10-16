"""End-to-end: the CLI and the pipeline on a reduced copy of the toy task, fully offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from anvil import __version__
from anvil.cli import app
from anvil.pipeline import Workspace, evaluate, label, train
from anvil.spec import load_spec
from anvil.store import Split, read_inputs, split_of
from anvil.teacher.fake import FakeTeacher
from tests.conftest import TOY_INPUTS, TOY_SPEC

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
    assert "on 80 fake-rules labels: 3 epochs" in trained.stdout

    evaluated = runner.invoke(app, ["eval", "-s", str(spec)])
    assert evaluated.exit_code == 0, evaluated.output
    assert "kappa" in evaluated.stdout
    (report_path,) = (home / "ticket-triage" / "reports").glob("eval-*.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["n"] == heldout
    assert -1.0 <= report["kappa"] <= 1.0
    assert report["teacher_model"] == "fake-rules"
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
