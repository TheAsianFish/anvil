"""Replays real Claude responses committed under tests/recordings/ (recorded by
`make record` from test_live.py). Catches parsing regressions and prompt drift offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anvil.spec import TaskSpec
from anvil.teacher import ClaudeTeacher, Item, ReplayTransport
from anvil.teacher.pricing import Usage, cost_usd
from anvil.teacher.transport import RecordingTransport
from tests.conftest import FakeTransport, message
from tests.test_live import live_spec

RECORDINGS = Path(__file__).parent / "recordings" / "ticket-triage"


def _check_recordings(directory: Path, spec: TaskSpec) -> None:
    transport = ReplayTransport(directory)
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        request, response = data["request"], data["response"]
        model = request["model"]
        teacher_spec = spec.model_copy(
            update={"teacher": spec.teacher.model_copy(update={"model": model})}
        )
        teacher = ClaudeTeacher(teacher_spec, transport, workers=1)
        item = Item(path.stem, request["messages"][0]["content"])
        assert teacher.request_params(item) == request, (
            f"{path.name}: the prompt changed since recording; re-record with `make record`"
        )
        batch = teacher.label([item])
        if batch.failures:
            assert batch.failures[0].reason
            continue
        (lb,) = batch.labels
        assert lb.label in spec.label_names
        assert lb.teacher_model == model
        expected = cost_usd(model, Usage.from_response(response.get("usage") or {}))
        assert lb.cost_usd == pytest.approx(expected)
        assert lb.cost_usd > 0


def test_replay_committed_recordings() -> None:
    if not list(RECORDINGS.glob("*.json")):
        pytest.skip("no recordings yet; record with `make record` (needs ANTHROPIC_API_KEY)")
    _check_recordings(RECORDINGS, live_spec())


def test_replay_check_on_fresh_recordings(tmp_path: Path) -> None:
    """The same check on recordings made here, so it runs before real ones are committed."""
    spec = live_spec()
    fake = FakeTransport(message('{"label": "billing"}', model="claude-haiku-5-5"))
    teacher = ClaudeTeacher(spec, RecordingTransport(fake, tmp_path))
    items = [Item("a", "I was charged twice"), Item("b", "Refund please")]
    assert len(teacher.label(items).labels) == 2
    _check_recordings(tmp_path, spec)
