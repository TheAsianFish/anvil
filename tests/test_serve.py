"""The serving app: escalation, fallbacks, validation, request ids, health, metrics, loading."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from anvil import log
from anvil.cli import app as cli_app
from anvil.errors import ConfigError, TeacherError
from anvil.serve import ServedModel, TeacherMode, Variant, create_app, load_app
from anvil.settings import Settings
from anvil.spec import TaskSpec
from anvil.teacher.base import BatchAbortedError, Failure, Item, Label, LabelBatch
from anvil.teacher.fake import FakeTeacher
from tests.conftest import ROOT

LABELS = ["bug", "billing", "other"]
SURE = "the app is fine"  # 0.9 other
CRASH = "it may crash"  # 0.5 bug; FakeTeacher says bug
REFUND = "maybe a refund"  # 0.45 bug; FakeTeacher says billing
VAGUE = "hmm"  # 0.4 billing; FakeTeacher says other
PROBS = {
    SURE: [0.05, 0.05, 0.9],
    CRASH: [0.5, 0.3, 0.2],
    REFUND: [0.45, 0.35, 0.2],
    VAGUE: [0.3, 0.4, 0.3],
}


@dataclass
class StubModel:
    threshold: float | None = 0.6
    labels: list[str] = field(default_factory=lambda: list(LABELS))
    temperature: float = 1.0
    variant: str = "fp32"
    meta: dict[str, object] = field(default_factory=dict)

    def predict_proba(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        return np.array([PROBS.get(t, [1.0, 0.0, 0.0]) for t in texts], dtype=np.float32)


class RecordingTeacher:
    """Wraps a teacher and records each call's items."""

    def __init__(self, inner: FakeTeacher) -> None:
        self.inner = inner
        self.model_id = inner.model_id
        self.calls: list[list[Item]] = []

    def label(self, items: Sequence[Item]) -> LabelBatch:
        self.calls.append(list(items))
        return self.inner.label(items)


class ScriptedTeacher:
    """Answers with a function of the items, or raises."""

    model_id = "scripted"

    def __init__(self, answer: Exception | dict[str, Label | Failure]) -> None:
        self.answer = answer
        self.calls = 0

    def label(self, items: Sequence[Item]) -> LabelBatch:
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        batch = LabelBatch()
        for item in items:
            outcome = self.answer.get(item.text)
            if outcome is not None:
                batch.add(_with_id(outcome, item.id))
        return batch


def _with_id(outcome: Label | Failure, input_id: str) -> Label | Failure:
    if isinstance(outcome, Label):
        return Label(input_id, outcome.label, "scripted", "h", cost_usd=outcome.cost_usd)
    return Failure(input_id, outcome.reason, cost_usd=outcome.cost_usd)


def _label(name: str, cost: float = 0.0) -> Label:
    return Label("", name, "scripted", "h", cost_usd=cost)


@pytest.fixture
def fake(small_spec: TaskSpec) -> RecordingTeacher:
    return RecordingTeacher(FakeTeacher(small_spec))


def _predict(client: TestClient, texts: list[str]) -> list[dict[str, object]]:
    r = client.post("/predict", json={"texts": texts})
    assert r.status_code == 200, r.text
    preds: list[dict[str, object]] = r.json()["predictions"]
    return preds


def _metrics(client: TestClient) -> dict[str, float]:
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain; version=0.0.4")
    out: dict[str, float] = {}
    sample = re.compile(r'^([a-z_]+)(\{[a-z]+="[^"]*"\})? (\S+)$')
    for line in r.text.splitlines():
        if line.startswith("#"):
            assert re.match(r"^# (HELP|TYPE) [a-z_]+ ", line), line
            continue
        m = sample.match(line)
        assert m, f"not a Prometheus sample line: {line!r}"
        out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
    return out


def test_all_confident_never_calls_teacher(fake: RecordingTeacher) -> None:
    client = TestClient(create_app(StubModel(), fake))
    preds = _predict(client, [SURE, SURE])
    assert fake.calls == []
    assert [p["source"] for p in preds] == ["student", "student"]
    assert preds[0] == {
        "label": "other",
        "confidence": pytest.approx(0.9),
        "source": "student",
        "student_label": "other",
        "below_threshold": False,
        "escalated": False,
        "error": None,
    }


def test_mixed_batch_escalates_only_unsure_inputs_in_one_call(fake: RecordingTeacher) -> None:
    client = TestClient(create_app(StubModel(), fake))
    r = client.post("/predict", json={"texts": [CRASH, SURE, REFUND, VAGUE]})
    body = r.json()
    preds = body["predictions"]
    assert len(fake.calls) == 1
    rid = body["request_id"]
    assert [(i.id, i.text) for i in fake.calls[0]] == [
        (f"{rid}:0", CRASH),
        (f"{rid}:2", REFUND),
        (f"{rid}:3", VAGUE),
    ]
    assert [p["label"] for p in preds] == ["bug", "other", "billing", "other"]
    assert [p["student_label"] for p in preds] == ["bug", "other", "bug", "billing"]
    assert [p["source"] for p in preds] == ["teacher", "student", "teacher", "teacher"]
    assert [p["escalated"] for p in preds] == [True, False, True, True]
    assert [p["below_threshold"] for p in preds] == [True, False, True, True]
    assert all(p["error"] is None for p in preds)
    assert body["model"] == {"variant": "fp32", "threshold": 0.6}


def test_threshold_argument_overrides_model(fake: RecordingTeacher) -> None:
    client = TestClient(create_app(StubModel(threshold=0.6), fake, threshold=0.42))
    preds = _predict(client, [CRASH, REFUND, VAGUE])
    assert [p["escalated"] for p in preds] == [False, False, True]
    assert client.get("/healthz").json()["threshold"] == 0.42


def test_no_threshold_escalates_nothing(fake: RecordingTeacher) -> None:
    client = TestClient(create_app(StubModel(threshold=None), fake))
    preds = _predict(client, [CRASH, VAGUE])
    assert fake.calls == []
    assert all(p["source"] == "student" for p in preds)
    assert not any(p["escalated"] or p["below_threshold"] for p in preds)


def test_unreachable_calibration_target_escalates_everything(fake: RecordingTeacher) -> None:
    # The calibration asked for a kappa no threshold reached: the student is not trusted at
    # any coverage, so even a confident input goes to the teacher.
    model = StubModel(threshold=None, meta={"calibration": {"target_kappa": 0.95}})
    client = TestClient(create_app(model, fake))
    preds = _predict(client, [SURE, CRASH])
    assert all(p["escalated"] and p["below_threshold"] for p in preds)
    assert len(fake.calls) == 1
    assert client.get("/healthz").json()["escalate_all"] is True
    # An explicit threshold still wins.
    explicit = TestClient(create_app(model, fake, threshold=0.5))
    assert explicit.get("/healthz").json()["escalate_all"] is False


def test_unreachable_calibration_target_without_a_teacher_refuses_to_serve() -> None:
    # Escalating everything needs someone to escalate to; without a teacher the server would
    # answer every input with a student the calibration said cannot be trusted at any coverage.
    model = StubModel(threshold=None, meta={"calibration": {"target_kappa": 0.95}})
    with pytest.raises(ConfigError, match="no teacher is configured"):
        create_app(model, None)
    # An explicit threshold is the operator's override.
    client = TestClient(create_app(model, None, threshold=0.5))
    assert client.get("/healthz").json()["escalate_all"] is False


def test_load_app_auto_without_credentials_refuses_an_unreachable_bundle(
    tmp_path: Path, small_spec: TaskSpec, settings: Settings
) -> None:
    def load(bundle: Path, variant: Variant, threads: int | None) -> ServedModel:
        meta = {**_spec_meta(small_spec), "calibration": {"target_kappa": 0.9}}
        return StubModel(threshold=None, meta=meta)

    with pytest.raises(ConfigError, match="no teacher is configured"):
        load_app(tmp_path, teacher="auto", settings=settings, loader=load)
    assert TestClient(load_app(tmp_path, teacher="fake", settings=settings, loader=load))


def test_the_input_that_set_the_threshold_is_answered_despite_float32_rounding(
    fake: RecordingTeacher,
) -> None:
    # The threshold is a float64 confidence; the served one went through float32.
    served = float(np.float32(0.9))
    exact = served + 1e-8  # rounds to `served` in float32
    assert float(np.float32(exact)) == served < exact
    client = TestClient(create_app(StubModel(threshold=exact), fake))
    PROBS["edge case"] = [served, 1 - served, 0.0]
    try:
        preds = _predict(client, ["edge case"])
    finally:
        del PROBS["edge case"]
    assert preds[0]["escalated"] is False


def test_no_teacher_answers_with_student(small_spec: TaskSpec) -> None:
    client = TestClient(create_app(StubModel(), None))
    preds = _predict(client, [CRASH, SURE])
    assert [p["label"] for p in preds] == ["bug", "other"]
    assert [p["below_threshold"] for p in preds] == [True, False]
    assert not any(p["escalated"] for p in preds)
    assert all(p["error"] is None and p["source"] == "student" for p in preds)
    assert client.get("/healthz").json()["teacher"] is None


def test_failure_for_one_item_falls_back_for_that_item_only() -> None:
    t = ScriptedTeacher({CRASH: Failure("", "refused"), VAGUE: _label("other")})
    client = TestClient(create_app(StubModel(), t))
    preds = _predict(client, [CRASH, VAGUE])
    assert preds[0]["source"] == "student"
    assert preds[0]["label"] == "bug"
    assert preds[0]["escalated"] is True
    assert preds[0]["error"] == "teacher failed: refused"
    assert preds[1]["source"] == "teacher"
    assert preds[1]["label"] == "other"
    assert preds[1]["error"] is None


@pytest.mark.parametrize(
    "exc",
    [
        TeacherError("api down"),
        RuntimeError("boom"),
        BatchAbortedError(KeyboardInterrupt(), LabelBatch()),
    ],
)
def test_teacher_exception_falls_back_for_all(exc: Exception) -> None:
    t = ScriptedTeacher(exc)
    client = TestClient(create_app(StubModel(), t))
    preds = _predict(client, [CRASH, SURE, VAGUE])
    assert [p["label"] for p in preds] == ["bug", "other", "billing"]
    assert all(p["source"] == "student" for p in preds)
    assert [p["escalated"] for p in preds] == [True, False, True]
    errors = [p["error"] for p in preds]
    assert errors[1] is None
    assert errors[0] == errors[2]
    assert isinstance(errors[0], str)
    assert type(exc).__name__ in errors[0] or "aborted" in errors[0]
    assert "api down" not in errors[0]
    assert "boom" not in errors[0]
    m = _metrics(client)
    assert m["anvil_escalations_total"] == 2
    assert m["anvil_escalation_failures_total"] == 2
    assert m['anvil_inputs_total{source="student"}'] == 3


def test_aborted_batch_keeps_completed_answers() -> None:
    class Aborting:
        model_id = "aborting"

        def label(self, items: Sequence[Item]) -> LabelBatch:
            done = LabelBatch(labels=[Label(items[0].id, "billing", "a", "h", cost_usd=0.5)])
            raise BatchAbortedError(TeacherError("bad request"), done)

    client = TestClient(create_app(StubModel(), Aborting()))
    preds = _predict(client, [CRASH, VAGUE])
    assert (preds[0]["source"], preds[0]["label"]) == ("teacher", "billing")
    assert preds[1]["source"] == "student"
    assert preds[1]["error"] == "teacher batch aborted: TeacherError"
    assert _metrics(client)["anvil_teacher_spent_usd_total"] == 0.5


def test_teacher_label_outside_label_set_falls_back() -> None:
    t = ScriptedTeacher({CRASH: _label("spam")})
    client = TestClient(create_app(StubModel(), t))
    (pred,) = _predict(client, [CRASH])
    assert pred["source"] == "student"
    assert pred["label"] == "bug"
    assert pred["escalated"] is True
    assert "spam" in str(pred["error"])


@pytest.mark.parametrize(
    "texts",
    [[], ["x"] * 4, ["x" * 11]],
    ids=["empty", "too-many", "too-long"],
)
def test_validation(texts: list[str]) -> None:
    client = TestClient(create_app(StubModel(), None, max_batch=3, max_chars=10))
    assert client.post("/predict", json={"texts": texts}).status_code == 422


def test_body_over_the_byte_limit_is_rejected_before_validation() -> None:
    calls: list[int] = []

    class Counting(StubModel):
        def predict_proba(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
            calls.append(len(texts))
            return super().predict_proba(texts)

    client = TestClient(create_app(Counting(), None, max_body_bytes=64))
    assert client.post("/predict", json={"texts": [SURE]}).status_code == 200
    r = client.post("/predict", json={"texts": ["x" * 100]})
    assert r.status_code == 413
    assert "64 bytes" in r.json()["detail"]
    r = client.post("/predict", content=b"{}", headers={"transfer-encoding": "chunked"})
    assert r.status_code == 411
    assert calls == [1]
    # The default limit has room for max_batch texts of max_chars four-byte characters.
    wide = TestClient(create_app(StubModel(), None, max_batch=2, max_chars=4))
    assert wide.post("/predict", json={"texts": ["\U0001f600" * 4] * 2}).status_code == 200


def test_bad_body_shape_is_422() -> None:
    client = TestClient(create_app(StubModel(), None))
    assert client.post("/predict", json={"text": "x"}).status_code == 422
    assert client.post("/predict", json={"texts": [1]}).status_code == 422


def test_request_id_echoed_or_replaced() -> None:
    client = TestClient(create_app(StubModel(), None))
    r = client.post("/predict", json={"texts": [SURE]}, headers={"X-Request-ID": "abc.1-_Z"})
    assert r.headers["x-request-id"] == "abc.1-_Z"
    assert r.json()["request_id"] == "abc.1-_Z"
    for bad in ("has space", "x" * 129, "semi;colon"):
        r = client.post("/predict", json={"texts": [SURE]}, headers={"X-Request-ID": bad})
        rid = r.json()["request_id"]
        assert re.fullmatch(r"[0-9a-f]{32}", rid)
        assert r.headers["x-request-id"] == rid
    r = client.post("/predict", json={"texts": [SURE]})
    assert re.fullmatch(r"[0-9a-f]{32}", r.json()["request_id"])


def test_healthz(fake: RecordingTeacher) -> None:
    client = TestClient(create_app(StubModel(variant="int8"), fake))
    assert client.get("/healthz").json() == {
        "status": "ok",
        "variant": "int8",
        "threshold": 0.6,
        "labels": 3,
        "teacher": fake.model_id,
        "escalate_all": False,
    }


def test_metrics_after_known_sequence(fake: RecordingTeacher) -> None:
    client = TestClient(create_app(StubModel(), fake))
    _predict(client, [SURE])
    _predict(client, [CRASH, SURE, VAGUE])
    _predict(client, [REFUND])
    client.post("/predict", json={"texts": []})  # rejected: not counted
    m = _metrics(client)
    assert m["anvil_requests_total"] == 3
    assert m['anvil_inputs_total{source="student"}'] == 2
    assert m['anvil_inputs_total{source="teacher"}'] == 3
    assert m["anvil_escalations_total"] == 3
    assert m["anvil_escalation_failures_total"] == 0
    assert m["anvil_teacher_spent_usd_total"] == 0
    assert m["anvil_request_seconds_count"] == 3
    assert m['anvil_request_seconds_bucket{le="+Inf"}'] == 3
    buckets = [v for k, v in m.items() if k.startswith("anvil_request_seconds_bucket")]
    assert len(buckets) == 12
    assert buckets == sorted(buckets)
    assert m["anvil_request_seconds_sum"] > 0


def test_spent_dollars_summed_from_batches() -> None:
    t = ScriptedTeacher(
        {CRASH: _label("bug", 0.01), VAGUE: Failure("", "truncated", cost_usd=0.02)}
    )
    client = TestClient(create_app(StubModel(), t))
    _predict(client, [CRASH, VAGUE])
    _predict(client, [CRASH])
    assert _metrics(client)["anvil_teacher_spent_usd_total"] == pytest.approx(0.04)


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self.setFormatter(log.JsonFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


@pytest.fixture
def captured() -> Iterator[_Capture]:
    log.configure("INFO")
    handler = _Capture()
    logger = logging.getLogger("anvil")
    logger.addHandler(handler)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel("WARNING")


def test_predict_log_line_has_no_input_text(captured: _Capture) -> None:
    t = ScriptedTeacher(RuntimeError(f"failed on {CRASH}"))
    client = TestClient(create_app(StubModel(), t))
    _predict(client, [CRASH, SURE])
    events = [json.loads(line) for line in captured.lines]
    (line,) = [e for e in events if e["event"] == "predict"]
    assert line["n"] == 2
    assert line["escalated"] == 1
    assert line["teacher_answered"] == 0
    assert line["teacher_failed"] == 1
    assert line["threshold"] == 0.6
    assert line["latency_ms"] >= 0
    assert len(line["request_id"]) == 32
    for text in (CRASH, SURE):
        assert all(text not in raw for raw in captured.lines)


def test_create_app_rejects_threshold_out_of_range() -> None:
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ConfigError, match="threshold"):
            create_app(StubModel(), None, threshold=bad)
    with pytest.raises(ConfigError, match="threshold"):
        create_app(StubModel(threshold=2.0), None)


def _spec_meta(spec: TaskSpec) -> dict[str, object]:
    return {"spec": spec.model_dump(mode="json"), "student": {"task": spec.name}}


def test_load_app_with_fake_teacher(
    tmp_path: Path, small_spec: TaskSpec, settings: Settings
) -> None:
    calls: list[tuple[Path, str, int | None]] = []

    def load(bundle: Path, variant: Variant, threads: int | None) -> ServedModel:
        calls.append((bundle, variant, threads))
        return StubModel(meta=_spec_meta(small_spec), variant=variant)

    app = load_app(
        tmp_path, variant="int8", teacher="fake", settings=settings, threads=3, loader=load
    )
    assert calls == [(tmp_path, "int8", 3)]
    client = TestClient(app)
    assert client.get("/healthz").json()["teacher"] == FakeTeacher.model_id_for(small_spec)
    preds = _predict(client, [REFUND])
    assert (preds[0]["source"], preds[0]["label"]) == ("teacher", "billing")


def test_load_app_threads_default_from_settings(
    tmp_path: Path, small_spec: TaskSpec, settings: Settings
) -> None:
    calls: list[int | None] = []

    def load(bundle: Path, variant: Variant, threads: int | None) -> ServedModel:
        calls.append(threads)
        return StubModel(meta=_spec_meta(small_spec))

    s = Settings(settings.home, 1, None, "WARNING", serve_threads=5)
    load_app(tmp_path, teacher="none", settings=s, loader=load)
    assert calls == [5]


def test_load_app_auto_without_credentials_disables_escalation(
    tmp_path: Path, small_spec: TaskSpec, settings: Settings
) -> None:
    def load(bundle: Path, variant: Variant, threads: int | None) -> ServedModel:
        return StubModel(meta=_spec_meta(small_spec))

    client = TestClient(load_app(tmp_path, teacher="auto", settings=settings, loader=load))
    assert client.get("/healthz").json()["teacher"] is None
    assert _predict(client, [CRASH])[0]["escalated"] is False
    with pytest.raises(ConfigError):
        load_app(tmp_path, teacher="claude", settings=settings, loader=load)


def test_load_app_without_spec(tmp_path: Path, settings: Settings) -> None:
    def load(bundle: Path, variant: Variant, threads: int | None) -> ServedModel:
        return StubModel(meta={"spec": None})

    off: tuple[TeacherMode, ...] = ("auto", "none")
    for mode in off:
        app = load_app(tmp_path, teacher=mode, settings=settings, loader=load)
        assert TestClient(app).get("/healthz").json()["teacher"] is None
    needs_spec: tuple[TeacherMode, ...] = ("fake", "claude")
    for mode in needs_spec:
        with pytest.raises(ConfigError, match="no task spec"):
            load_app(tmp_path, teacher=mode, settings=settings, loader=load)


def test_load_app_fake_needs_fake_section(
    tmp_path: Path, small_spec: TaskSpec, settings: Settings
) -> None:
    spec = small_spec.model_copy(update={"fake": None})

    def load(bundle: Path, variant: Variant, threads: int | None) -> ServedModel:
        return StubModel(meta=_spec_meta(spec))

    with pytest.raises(ConfigError, match="fake"):
        load_app(tmp_path, teacher="fake", settings=settings, loader=load)


def test_load_app_threshold_out_of_range(
    tmp_path: Path, small_spec: TaskSpec, settings: Settings
) -> None:
    def load(bundle: Path, variant: Variant, threads: int | None) -> ServedModel:
        return StubModel(meta=_spec_meta(small_spec))

    with pytest.raises(ConfigError, match="threshold"):
        load_app(tmp_path, teacher="none", threshold=1.2, settings=settings, loader=load)


def test_load_app_missing_bundle(tmp_path: Path, settings: Settings) -> None:
    with pytest.raises(ConfigError, match="anvil export"):
        load_app(tmp_path / "nope", settings=settings)


def test_cli_serve_help() -> None:
    r = CliRunner().invoke(cli_app, ["serve", "--help"], env={"NO_COLOR": "1", "COLUMNS": "120"})
    assert r.exit_code == 0
    text = re.sub(r"\x1b\[[0-9;]*m", "", r.output)  # rich may still style the help
    for opt in ("--host", "--port", "--variant", "--threshold", "--teacher"):
        assert opt in text


def test_cli_serve_missing_bundle(tmp_path: Path) -> None:
    r = CliRunner().invoke(cli_app, ["serve", str(tmp_path / "nope")])
    assert r.exit_code == 2
    lines = r.stderr.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error: bundle")


def test_serve_and_cli_import_without_torch() -> None:
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['transformers'] = None\n"
        "import anvil.serve, anvil.cli\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr
