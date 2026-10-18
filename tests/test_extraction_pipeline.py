"""End-to-end extraction: label -> train -> eval on the toy restaurant task, fully offline."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from anvil.cli import app
from anvil.errors import AlignmentError, ConfigError
from anvil.extraction import answer_spans
from anvil.loop import LoopConfig, run_loop
from anvil.pipeline import Workspace, evaluate, label, train
from anvil.spans import Mention, Span, encode_answer
from anvil.spec import TaskSpec, load_spec
from anvil.students.tagger import EncoderTagger
from anvil.teacher.base import Item, Label, LabelBatch
from anvil.teacher.fake import FakeTeacher
from tests.conftest import ROOT

TOY_DIR = ROOT / "examples" / "toy-extract"
N_INPUTS = 200


@pytest.fixture
def toy_copy(tmp_path: Path) -> tuple[Path, Path]:
    doc = yaml.safe_load((TOY_DIR / "anvil.yaml").read_text(encoding="utf-8"))
    doc["student"]["epochs"] = 6
    spec = tmp_path / "anvil.yaml"
    spec.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    inputs = tmp_path / "inputs.jsonl"
    lines = (TOY_DIR / "inputs.jsonl").read_text(encoding="utf-8").splitlines()[:N_INPUTS]
    inputs.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return spec, inputs


def test_toy_extract_spec_loads() -> None:
    spec = load_spec(TOY_DIR / "anvil.yaml")
    assert spec.kind == "extract"
    assert "Restaurant_Name" in spec.label_names


def test_cli_label_train_eval(toy_copy: tuple[Path, Path]) -> None:
    spec, inputs = toy_copy
    runner = CliRunner()
    labelled = runner.invoke(
        app, ["label", "-s", str(spec), "-i", str(inputs), "-n", "150", "--teacher", "fake"]
    )
    assert labelled.exit_code == 0, labelled.output
    trained = runner.invoke(app, ["train", "-s", str(spec), "--teacher", "fake"])
    assert trained.exit_code == 0, trained.output
    assert "entity F1" in trained.stdout
    assert "repaired (shifted" in trained.stdout
    evaluated = runner.invoke(app, ["eval", "-s", str(spec)])
    assert evaluated.exit_code == 0, evaluated.output
    assert "entity F1" in evaluated.stdout
    assert "token kappa" in evaluated.stdout
    report_path = Path(evaluated.stdout.rsplit("report ", 1)[1].strip())
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # The fake teacher shifts a fifth of its offsets by one; alignment repairs them all.
    assert report["heldout_alignment"]["matched"]["shifted"] > 0
    assert report["heldout_alignment"]["unaligned"] == 0
    assert report["entity"]["f1"] > 0.5  # a templated task the tiny tagger learns quickly


def test_trained_tagger_keeps_its_alignment_report(toy_copy: tuple[Path, Path]) -> None:
    spec_path, inputs = toy_copy
    spec = load_spec(spec_path)
    ws = Workspace.for_task(Path(spec_path.parent / "home"), spec)
    label(spec, ws, FakeTeacher(spec), inputs, 120)
    trained = train(spec, ws, FakeTeacher.model_id_for(spec))
    tagger = EncoderTagger.load(trained.path)
    alignment = tagger.meta["alignment"]
    assert isinstance(alignment, dict)
    assert alignment["mentions"] > 0
    assert alignment["matched"]["shifted"] > 0
    assert alignment["unaligned_rate"] == 0.0
    assert tagger.meta["kind"] == "extract"
    result = evaluate(spec, ws, trained.path, n_resamples=50)
    assert result.agreement.n > 0


class _WrongTeacher:
    """Answers with mentions that are not in the input: what a broken prompt produces."""

    def __init__(self, spec: TaskSpec, bad_every: int) -> None:
        self.model_id = "wrong-teacher"
        self._spec = spec
        self._fake = FakeTeacher(spec)
        self._bad_every = bad_every

    def label(self, items: Sequence[Item]) -> LabelBatch:
        batch = LabelBatch()
        for n, item in enumerate(items):
            mentions = self._fake.mentions(item)
            if n % self._bad_every == 0:
                mentions = [*mentions, Mention("a dish nobody asked for", "Dish", 0)]
            batch.labels.append(
                Label(item.id, encode_answer(mentions), self.model_id, self._spec.prompt_hash)
            )
        return batch


def test_training_refuses_when_too_many_mentions_do_not_align(
    toy_copy: tuple[Path, Path],
) -> None:
    spec_path, inputs = toy_copy
    spec = load_spec(spec_path)
    ws = Workspace.for_task(Path(spec_path.parent / "home"), spec)
    label(spec, ws, _WrongTeacher(spec, bad_every=2), inputs, 120)
    with pytest.raises(AlignmentError, match="could not be aligned") as info:
        train(spec, ws, "wrong-teacher")
    assert info.value.exit_code == 2
    assert not list((ws.students).glob("*/config.json"))  # nothing was trained or saved


def test_cli_maps_the_alignment_gate_to_exit_code_2(
    toy_copy: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path, inputs = toy_copy
    spec = load_spec(spec_path)
    home = spec_path.parent / "cli-home"
    monkeypatch.setenv("ANVIL_HOME", str(home))
    label(spec, Workspace.for_task(home, spec), _WrongTeacher(spec, 2), inputs, 120)
    monkeypatch.setattr("anvil.cli._teacher_model", lambda task, teacher: "wrong-teacher")
    result = CliRunner().invoke(app, ["train", "-s", str(spec_path)])
    assert result.exit_code == 2
    assert "could not be aligned" in result.output


def test_answer_spans_rejects_a_classify_label() -> None:
    spec = load_spec(TOY_DIR / "anvil.yaml")
    with pytest.raises(Exception, match="not an extraction answer"):
        answer_spans(spec, [(Item("1", "cheap pizza"), "billing")])


def test_answer_spans_aligns_stored_answers() -> None:
    spec = load_spec(TOY_DIR / "anvil.yaml")
    stored = encode_answer([Mention("ramen", "Dish", 12)])  # true start 13
    spans, report = answer_spans(spec, [(Item("1", "where is the ramen"), stored)])
    assert spans == [[Span(13, 18, "Dish")]]
    assert report.matched["shifted"] == 1


def test_loop_refuses_an_extract_task(toy_copy: tuple[Path, Path]) -> None:
    spec_path, _ = toy_copy
    spec = load_spec(spec_path)
    ws = Workspace.for_task(spec_path.parent / "home", spec)
    with pytest.raises(ConfigError, match="runs classify tasks"):
        run_loop(spec, ws, FakeTeacher(spec), LoopConfig(strategy="random", budgets=(10,)))


def test_the_gate_runs_before_the_first_training_step(
    toy_copy: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path, inputs = toy_copy
    spec = load_spec(spec_path)
    ws = Workspace.for_task(Path(spec_path.parent / "home"), spec)
    label(spec, ws, _WrongTeacher(spec, bad_every=2), inputs, 120)

    def no_training(*args: object, **kwargs: object) -> None:
        raise AssertionError("fit was called although the gate should have refused first")

    monkeypatch.setattr(EncoderTagger, "fit", no_training)
    with pytest.raises(AlignmentError):
        train(spec, ws, "wrong-teacher")
    assert not list(ws.students.iterdir())  # the refused run leaves no directory behind


def test_a_few_bad_mentions_pass_the_gate_and_train(toy_copy: tuple[Path, Path]) -> None:
    spec_path, inputs = toy_copy
    spec = load_spec(spec_path)
    ws = Workspace.for_task(Path(spec_path.parent / "home"), spec)
    label(spec, ws, _WrongTeacher(spec, bad_every=25), inputs, 120)  # about 2% unaligned
    report = train(spec, ws, "wrong-teacher")
    assert report.alignment is not None
    assert 0 < report.alignment.unaligned_rate <= 0.05
    assert (report.path / "config.json").exists()
