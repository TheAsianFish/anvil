"""Calls the real Claude API (a fraction of a cent). Run with `make test-live`; `make record`
also writes the responses to tests/recordings/ for test_replay.py."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from anvil.settings import Settings
from anvil.spec import TaskSpec, load_spec
from anvil.store import read_inputs
from anvil.teacher import build_teacher
from tests.conftest import TOY_INPUTS, TOY_SPEC

LIVE_MODEL = "claude-haiku-5-5"


def live_spec() -> TaskSpec:
    """The toy spec with the cheap teacher model; recordings are made with exactly this."""
    spec = load_spec(TOY_SPEC)
    teacher = spec.teacher.model_copy(update={"model": LIVE_MODEL, "provider": "claude"})
    return spec.model_copy(update={"teacher": teacher})


@pytest.mark.live
def test_claude_labels_toy_inputs(tmp_path: Path) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set; live tests call the real Claude API")
    spec = live_spec()
    settings = dataclasses.replace(Settings.from_env(), home=tmp_path)
    items = read_inputs(TOY_INPUTS)[:8]

    batch = build_teacher(spec, settings).label(items)
    assert len(batch.labels) + len(batch.failures) == 8
    assert all(f.reason for f in batch.failures)
    assert all(lb.label in spec.label_names for lb in batch.labels)
    assert all(lb.teacher_model == LIVE_MODEL for lb in batch.labels)
    assert 0 < batch.spent_usd < 0.01

    # Failures are not cached, so the rerun covers only the inputs that got a label.
    labelled = {lb.input_id for lb in batch.labels}
    again = build_teacher(spec, settings).label([it for it in items if it.id in labelled])
    assert again.cache_hits == len(batch.labels)
    assert again.spent_usd == 0
    assert [lb.label for lb in again.labels] == [lb.label for lb in batch.labels]
