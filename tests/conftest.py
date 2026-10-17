"""Shared fixtures: network blocking, an isolated ANVIL_HOME, specs and a fake transport."""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from anvil.settings import Settings
from anvil.spec import TaskSpec, load_spec

ROOT = Path(__file__).resolve().parent.parent
TOY_DIR = ROOT / "examples" / "toy"
TOY_SPEC = TOY_DIR / "anvil.yaml"
TOY_INPUTS = TOY_DIR / "inputs.jsonl"

# Any: Messages API bodies are arbitrary JSON (same alias as anvil.teacher.transport).
JSONDict = dict[str, Any]


def _no_network(*args: object, **kwargs: object) -> None:
    raise RuntimeError("network access in an offline test; mark it `live` or mock it")


@pytest.fixture(autouse=True)
def _offline(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("live") is None:
        monkeypatch.setattr(socket.socket, "connect", _no_network)
        home = request.getfixturevalue("tmp_path") / "anvil-home"
        monkeypatch.setenv("ANVIL_HOME", str(home))
        monkeypatch.setenv("ANVIL_LOG_LEVEL", "WARNING")
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANVIL_RECORD_DIR"):
            monkeypatch.delenv(name, raising=False)
        # An empty, explicitly selected SDK config dir, so a developer's `ant auth login`
        # profile can never leak into an offline test.
        monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(home / "anthropic-config"))


@pytest.fixture
def toy_spec() -> TaskSpec:
    return load_spec(TOY_SPEC)


@pytest.fixture
def small_spec() -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "name": "small",
            "kind": "classify",
            "instructions": "Classify the ticket.",
            "labels": [
                {"name": "bug", "description": "something is broken"},
                {"name": "billing", "description": "money matters"},
                "other",
            ],
            "teacher": {"model": "claude-opus-5-5", "effort": "low", "max_tokens": 1024},
            "student": {"model": "tiny", "max_length": 32, "epochs": 2, "seed": 0},
            "fake": {"rules": {"bug": ["crash"], "billing": ["refund"]}, "default": "other"},
        }
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        home=tmp_path / "home",
        teacher_workers=2,
        record_dir=None,
        log_level="WARNING",
    )


def message(
    text: str = '{"label": "bug"}',
    *,
    stop_reason: str = "end_turn",
    input_tokens: int = 1000,
    output_tokens: int = 50,
    model: str = "claude-opus-5-5",
) -> JSONDict:
    """A Messages API response body as `Message.to_dict()` returns it."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


class FakeTransport:
    """Records every request; answers with `message()`, or a response or an exception
    programmed per input text."""

    def __init__(self, default: JSONDict | None = None) -> None:
        self.default = default if default is not None else message()
        self.responses: dict[str, JSONDict] = {}
        self.errors: dict[str, Exception] = {}
        self.delays: dict[str, float] = {}
        self.calls: list[JSONDict] = []
        self._lock = threading.Lock()

    def program(self, text: str, response: JSONDict) -> None:
        self.responses[text] = response

    def __call__(self, params: JSONDict) -> JSONDict:
        with self._lock:
            self.calls.append(params)
        text = params["messages"][0]["content"]
        if text in self.delays:
            time.sleep(self.delays[text])
        if text in self.errors:
            raise self.errors[text]
        return self.responses.get(text, self.default)


@pytest.fixture
def fake_transport() -> FakeTransport:
    return FakeTransport()
