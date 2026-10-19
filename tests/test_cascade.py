"""End-to-end cascade on the toy task: label, train, calibrate, export, serve with escalation.

Everything runs offline: the fake teacher labels the toy tickets, a tiny student trains on
them, `anvil calibrate` and `anvil export` write the calibration and the ONNX bundle, and the
server built from the bundle escalates low-confidence inputs to the same fake teacher.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from anvil.cli import app
from anvil.errors import ConfigError, StoreError
from anvil.pipeline import (
    CALIBRATION_FILE,
    Workspace,
    calibrate_student,
    export_bundle,
    label,
    read_calibration,
    train,
)
from anvil.runtime import BUNDLE_META, OnnxStudent
from anvil.serve import load_app
from anvil.settings import Settings
from anvil.spec import TaskSpec, load_spec
from anvil.store import Split
from anvil.teacher.base import Item
from anvil.teacher.fake import FakeTeacher
from tests.conftest import TOY_INPUTS, TOY_SPEC


@dataclass(frozen=True)
class Toy:
    spec: TaskSpec
    spec_path: Path
    ws: Workspace
    student: Path
    bundle: Path


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Toy:
    root = tmp_path_factory.mktemp("cascade")
    doc = yaml.safe_load(TOY_SPEC.read_text(encoding="utf-8"))
    doc["student"]["epochs"] = 4
    spec_path = root / "anvil.yaml"
    spec_path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    spec = load_spec(spec_path)
    ws = Workspace.for_task(root / "home", spec)
    label(spec, ws, FakeTeacher(spec), TOY_INPUTS, 200)
    trained = train(spec, ws, FakeTeacher.model_id_for(spec))
    calibrate_student(spec, ws, trained.path, target_kappa=0.8, rule="point", n_resamples=200)
    report = export_bundle(spec, ws, trained.path)
    return Toy(spec, spec_path, ws, trained.path, report.bundle)


def _heldout_texts(toy: Toy) -> list[str]:
    with toy.ws.open_store(toy.spec) as store:
        return [it.text for it in store.inputs(Split.HELDOUT)]


def test_calibration_is_written_next_to_the_student_and_into_the_bundle(toy: Toy) -> None:
    calibration = read_calibration(toy.student)
    assert calibration is not None
    assert calibration["teacher_model"] == FakeTeacher.model_id_for(toy.spec)
    meta = json.loads((toy.bundle / BUNDLE_META).read_text(encoding="utf-8"))
    assert meta["temperature"] == calibration["temperature"]
    assert meta["threshold"] == calibration["threshold"]
    assert meta["spec"]["name"] == toy.spec.name
    assert TaskSpec.model_validate(meta["spec"]) == toy.spec


def test_bundle_matches_the_student_with_the_calibrated_temperature(toy: Toy) -> None:
    from anvil.calibrate import softmax
    from anvil.students.encoder import EncoderClassifier

    texts = _heldout_texts(toy)[:20]
    onnx = OnnxStudent.load(toy.bundle)
    torch_logits = EncoderClassifier.load(toy.student).predict_logits(texts)
    expected = softmax(torch_logits, onnx.temperature)
    np.testing.assert_allclose(onnx.predict_proba(texts), expected, atol=1e-4)


def test_server_escalates_exactly_the_inputs_under_the_threshold(toy: Toy) -> None:
    texts = _heldout_texts(toy)[:40]
    probs = OnnxStudent.load(toy.bundle).predict_proba(texts)
    confidence = probs.max(axis=1)
    # A threshold between the observed confidences, so some inputs fall on each side.
    threshold = float(np.median(confidence))
    client = TestClient(load_app(toy.bundle, teacher="fake", threshold=threshold))
    rid = "cascade-test"
    body = client.post("/predict", json={"texts": texts}, headers={"X-Request-ID": rid}).json()
    # The fake teacher's answer for each input under the item id the server gives it (its
    # noise is keyed by the id), computed independently of the server.
    fake = FakeTeacher(toy.spec)
    expected = {
        lb.input_id: lb.label
        for lb in fake.label([Item(f"{rid}:{i}", t) for i, t in enumerate(texts)]).labels
    }
    escalated = 0
    for text, conf, p in zip(texts, confidence, body["predictions"], strict=True):
        below = float(conf) < threshold
        assert p["below_threshold"] is below
        assert p["escalated"] is below
        if below:
            escalated += 1
            assert p["source"] == "teacher"
            assert p["label"] == expected[f"{rid}:{texts.index(text)}"]
        else:
            assert p["source"] == "student"
            assert p["label"] == p["student_label"]
    assert 0 < escalated < len(texts)
    metrics = client.get("/metrics").text
    assert f'anvil_inputs_total{{source="teacher"}} {escalated}' in metrics
    assert f"anvil_escalations_total {escalated}" in metrics


def test_bundle_threshold_is_the_default(toy: Toy) -> None:
    client = TestClient(load_app(toy.bundle, teacher="none"))
    health = client.get("/healthz").json()
    assert health["threshold"] == read_calibration(toy.student)["threshold"]  # type: ignore[index]
    assert health["teacher"] is None


def test_cli_calibrate_and_export(toy: Toy, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANVIL_HOME", str(toy.ws.root.parent))
    runner = CliRunner()
    calibrated = runner.invoke(
        app, ["calibrate", "-s", str(toy.spec_path), "--target-kappa", "0.5", "--rule", "point"]
    )
    assert calibrated.exit_code == 0, calibrated.output
    assert "temperature" in calibrated.stdout
    assert "threshold" in calibrated.stdout
    assert (toy.student / CALIBRATION_FILE).exists()
    out = toy.ws.root / "cli-bundle"
    exported = runner.invoke(app, ["export", "-s", str(toy.spec_path), "--out", str(out)])
    assert exported.exit_code == 0, exported.output
    assert "fp32: passed parity vs pytorch" in exported.stdout
    assert (out / BUNDLE_META).exists()


def test_cli_calibrate_reports_an_unreachable_target(
    toy: Toy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANVIL_HOME", str(toy.ws.root.parent))
    result = CliRunner().invoke(
        app, ["calibrate", "-s", str(toy.spec_path), "--target-kappa", "1.01"]
    )
    assert result.exit_code == 0, result.output
    assert "escalate everything" in result.stdout


def test_calibrate_and_export_refuse_a_changed_prompt(toy: Toy) -> None:
    edited = toy.spec.model_copy(update={"instructions": "Something else entirely."})
    with pytest.raises(StoreError, match="relabel and retrain"):
        calibrate_student(edited, toy.ws, toy.student)
    with pytest.raises(StoreError, match="relabel and retrain"):
        export_bundle(edited, toy.ws, toy.student, toy.ws.root / "x")


def test_export_refuses_extract_tasks(toy: Toy, tmp_path: Path) -> None:
    extract = TaskSpec.model_validate(
        {"name": "slots", "kind": "extract", "instructions": "Find slots.", "labels": ["dish"]}
    )
    with pytest.raises(ConfigError, match="classify"):
        export_bundle(extract, toy.ws, toy.student, tmp_path / "b")


def test_serve_command_needs_a_bundle(tmp_path: Path) -> None:
    settings = Settings.from_env()
    with pytest.raises(ConfigError, match="anvil export"):
        load_app(tmp_path / "missing", settings=settings)
