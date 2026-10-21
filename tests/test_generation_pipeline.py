"""End-to-end generation: label -> train -> eval -> merged export on the toy SQL task, offline."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from typer.testing import CliRunner

from anvil.cli import app
from anvil.errors import AlignmentError, ConfigError, StoreError
from anvil.merge import MERGED_META, load_merged
from anvil.pipeline import (
    TrainReport,
    Workspace,
    evaluate,
    export_bundle,
    export_merged,
    label,
    train,
)
from anvil.spec import TaskSpec, load_spec
from anvil.students.generator import ADAPTER_DIR, BASE_DIR, META_FILE, LoraGenerator
from anvil.teacher.fake import FakeTeacher
from anvil.train import generator as train_gen
from anvil.train.loop import LAST
from tests.conftest import ROOT, TOY_SPEC

TOY_DIR = ROOT / "examples" / "toy-generate"
TOY_INPUTS = TOY_DIR / "inputs.jsonl"
POOL = 80
HELDOUT = 16


def _spec(**student: object) -> TaskSpec:
    spec = load_spec(TOY_DIR / "anvil.yaml")
    update: dict[str, object] = {"epochs": 2, **student}
    return spec.model_copy(update={"student": spec.student.model_copy(update=update)})


@dataclass(frozen=True)
class Trained:
    spec: TaskSpec
    ws: Workspace
    teacher_model: str
    report: TrainReport


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Trained:
    """One labelled and trained toy run shared by the read-only tests below."""
    spec = _spec()
    ws = Workspace.for_task(tmp_path_factory.mktemp("gen-home"), spec)
    teacher = FakeTeacher(spec)
    labelled = label(spec, ws, teacher, TOY_INPUTS, POOL, heldout_max=HELDOUT)
    assert not labelled.failures
    assert (labelled.heldout_labelled, labelled.pool_labelled) == (HELDOUT, POOL)
    return Trained(spec, ws, teacher.model_id, train(spec, ws, teacher.model_id))


def test_toy_generate_spec_loads() -> None:
    spec = load_spec(TOY_DIR / "anvil.yaml")
    assert spec.kind == "generate"
    assert spec.output_names == ["sql"]
    assert spec.student.model == "tiny"


def test_train_saves_a_generator_run(trained: Trained) -> None:
    r = trained.report
    assert r.chat is not None
    assert r.chat.examples == POOL
    assert r.chat.unusable == 0
    assert r.alignment is None
    assert r.result.examples + r.result.validation_examples == POOL
    assert (r.path / ADAPTER_DIR).is_dir()
    assert (r.path / BASE_DIR).is_dir()
    assert (r.path / META_FILE).exists()
    assert (r.path / "metrics.jsonl").read_text(encoding="utf-8").strip()
    assert not (r.path / "checkpoint").exists()
    assert trained.ws.latest_student() == r.path
    meta = LoraGenerator.load_meta(r.path)
    assert meta["task"] == trained.spec.name
    assert meta["kind"] == "generate"
    assert meta["prompt_hash"] == trained.spec.prompt_hash
    assert meta["teacher_model"] == trained.teacher_model
    assert meta["train_examples"] == POOL
    assert meta["chat"] == r.chat.as_dict()


def test_evaluate_reports_exact_match_and_predictions(trained: Trained) -> None:
    r = evaluate(trained.spec, trained.ws, n_resamples=50)
    assert r.student == trained.report.run_id
    assert r.agreement.n == HELDOUT
    report = json.loads(r.report_path.read_text(encoding="utf-8"))
    assert report["n"] == HELDOUT
    em = report["exact_match"]
    assert 0.0 <= em["low"] <= em["estimate"] <= em["high"] <= 1.0
    assert em["n_resamples"] == 50
    assert [f["name"] for f in report["field_match"]] == ["sql"]
    assert report["valid"] + report["invalid"] == HELDOUT
    preds = report["predictions"]
    assert len(preds["input_id"]) == len(preds["completion"]) == HELDOUT
    assert all(isinstance(c, str) for c in preds["completion"])
    assert report["train_examples"] == POOL
    assert report["teacher_model"] == trained.teacher_model


def test_evaluate_refuses_a_student_for_another_task(trained: Trained) -> None:
    other = trained.spec.model_copy(update={"name": "other-sql"})
    with pytest.raises(StoreError, match="trained for task 'toy-sql'"):
        evaluate(other, trained.ws, trained.report.path)


def test_evaluate_refuses_a_student_for_another_prompt(trained: Trained) -> None:
    edited = trained.spec.model_copy(
        update={"instructions": trained.spec.instructions + "Prefer COUNT.\n"}
    )
    assert edited.prompt_hash != trained.spec.prompt_hash
    with pytest.raises(StoreError, match="relabel and retrain"):
        evaluate(edited, trained.ws, trained.report.path)


def test_evaluate_refuses_a_student_for_other_splits(trained: Trained) -> None:
    spec = trained.spec
    moved = spec.model_copy(
        update={"splits": spec.splits.model_copy(update={"heldout_fraction": 0.75})}
    )
    with pytest.raises(StoreError, match="retrain before evaluating"):
        evaluate(moved, trained.ws, trained.report.path)


def test_export_merged_keeps_the_answers(trained: Trained) -> None:
    m = export_merged(trained.spec, trained.ws, probes=8)
    expected = trained.ws.root / "exports" / f"{trained.report.run_id}-merged"
    assert m.out == expected
    meta = json.loads((expected / MERGED_META).read_text(encoding="utf-8"))
    assert meta["parity"]["passed"] is True
    assert meta["parity"]["probes"] == 8
    assert meta["student"] == trained.report.run_id
    assert meta["spec"]["name"] == trained.spec.name
    texts = [
        json.loads(line)["text"] for line in TOY_INPUTS.read_text(encoding="utf-8").splitlines()
    ][:12]
    student = LoraGenerator.load(trained.report.path)
    assert load_merged(expected).generate(texts) == student.generate(texts)


def test_export_merged_refuses_a_classify_spec(tmp_path: Path) -> None:
    spec = load_spec(TOY_SPEC)
    with pytest.raises(ConfigError, match="merged export is for generate tasks"):
        export_merged(spec, Workspace.for_task(tmp_path, spec))


def test_onnx_export_refuses_a_generate_spec(trained: Trained) -> None:
    with pytest.raises(ConfigError, match="export_merged") as info:
        export_bundle(trained.spec, trained.ws)
    assert info.value.exit_code == 2


def test_answers_too_long_for_max_length_refuse_and_leave_no_run(tmp_path: Path) -> None:
    spec = _spec(max_length=64)
    ws = Workspace.for_task(tmp_path, spec)
    teacher = FakeTeacher(spec)
    label(spec, ws, teacher, TOY_INPUTS, 40, heldout_max=4)
    with pytest.raises(AlignmentError, match="max_length") as info:
        train(spec, ws, teacher.model_id)
    assert info.value.exit_code == 2
    assert not [p for p in ws.students.iterdir() if p.is_dir()]
    assert not (ws.students / "LATEST").exists()


class SimulatedCrashError(Exception):
    pass


def test_train_resume_continues_an_interrupted_generator_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No validation slice: the crash count below is then training micro-batches only.
    spec = _spec(validation_fraction=0.0)
    ws = Workspace.for_task(tmp_path, spec)
    teacher = FakeTeacher(spec)
    label(spec, ws, teacher, TOY_INPUTS, 40, heldout_max=4)
    reference = train(spec, ws, teacher.model_id)

    original = train_gen.collate
    calls = 0

    def failing(
        examples: Sequence[Any], pad_id: int, device: torch.device
    ) -> dict[str, torch.Tensor]:  # Any: anvil.chat.Example
        nonlocal calls
        calls += 1
        if calls == 7:  # epoch 2 (5 micro-batches an epoch), after the first checkpoint
            raise SimulatedCrashError
        return original(examples, pad_id, device)

    monkeypatch.setattr(train_gen, "collate", failing)
    with pytest.raises(SimulatedCrashError):
        train(spec, ws, teacher.model_id)
    monkeypatch.undo()
    (interrupted,) = [p for p in ws.students.iterdir() if p.is_dir() and p != reference.path]
    assert (interrupted / "checkpoint" / LAST).exists()
    assert ws.latest_student() == reference.path  # a crashed run is never the latest

    resumed = train(spec, ws, teacher.model_id, resume=interrupted.name)
    assert resumed.path == interrupted
    assert resumed.chat is not None
    assert resumed.result.resumed_at_step is not None
    assert resumed.result.epoch_losses == reference.result.epoch_losses
    assert not (interrupted / "checkpoint").exists()
    assert ws.latest_student() == interrupted
    a = LoraGenerator.load(reference.path).model.named_parameters()
    b = dict(LoraGenerator.load(interrupted).model.named_parameters())
    for name, value in a:
        if "lora_" in name:
            assert torch.equal(value, b[name]), name


# --- CLI ------------------------------------------------------------------------------


@pytest.fixture
def cli_copy(tmp_path: Path) -> tuple[Path, Path]:
    doc = yaml.safe_load((TOY_DIR / "anvil.yaml").read_text(encoding="utf-8"))
    doc["student"]["epochs"] = 1
    spec = tmp_path / "anvil.yaml"
    spec.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    inputs = tmp_path / "inputs.jsonl"
    lines = TOY_INPUTS.read_text(encoding="utf-8").splitlines()[:100]
    inputs.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return spec, inputs


def test_cli_label_train_eval_export(cli_copy: tuple[Path, Path]) -> None:
    spec, inputs = cli_copy
    runner = CliRunner()
    args = ["-s", str(spec), "--teacher", "fake"]
    labelled = runner.invoke(app, ["label", "-i", str(inputs), "-n", "60", *args])
    assert labelled.exit_code == 0, labelled.output
    trained = runner.invoke(app, ["train", *args])
    assert trained.exit_code == 0, trained.output
    assert "aligned 60 answers to the chat template: 0 dropped" in trained.stdout
    assert "validation score (exact match, loss as tie-break)" in trained.stdout
    evaluated = runner.invoke(app, ["eval", "-s", str(spec)])
    assert evaluated.exit_code == 0, evaluated.output
    assert "exact match" in evaluated.stdout
    assert "(per field: sql " in evaluated.stdout
    exported = runner.invoke(app, ["export", "-s", str(spec), "--probes", "4"])
    assert exported.exit_code == 0, exported.output
    assert "merged: passed parity vs the adapter on 4 held-out inputs" in exported.stdout
    assert "4/4 identical completions" in exported.stdout
    assert "a plain transformers checkpoint" in exported.stdout

    int8 = runner.invoke(app, ["export", "-s", str(spec), "--int8"])
    assert int8.exit_code == 2
    assert "--int8 is for ONNX classifier bundles" in int8.stderr
    calibrated = runner.invoke(app, ["calibrate", "-s", str(spec)])
    assert calibrated.exit_code == 1
    assert "calibration is for classify students" in calibrated.stderr


def test_cli_loop_refuses_a_generate_task(cli_copy: tuple[Path, Path]) -> None:
    spec, _ = cli_copy
    result = CliRunner().invoke(app, ["loop", "-s", str(spec), "--teacher", "fake"])
    assert result.exit_code == 2
    assert "runs classify tasks" in result.stderr
