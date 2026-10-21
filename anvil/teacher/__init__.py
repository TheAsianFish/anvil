"""Teachers: the models whose answers the student learns to reproduce."""

from __future__ import annotations

from typing import Literal

from anvil.errors import ConfigError
from anvil.settings import Settings
from anvil.spec import TaskSpec
from anvil.teacher.base import BatchAbortedError, Failure, Item, Label, LabelBatch, Teacher
from anvil.teacher.cache import DiskCache
from anvil.teacher.claude import ClaudeTeacher
from anvil.teacher.fake import FakeTeacher
from anvil.teacher.ollama import HttpOllamaTransport, OllamaTeacher, OllamaTransport
from anvil.teacher.transport import (
    AnthropicTransport,
    RecordingTransport,
    ReplayTransport,
    Transport,
)

__all__ = [
    "AnthropicTransport",
    "BatchAbortedError",
    "ClaudeTeacher",
    "DiskCache",
    "Failure",
    "FakeTeacher",
    "HttpOllamaTransport",
    "Item",
    "Label",
    "LabelBatch",
    "OllamaTeacher",
    "OllamaTransport",
    "RecordingTransport",
    "ReplayTransport",
    "Teacher",
    "Transport",
    "build_teacher",
]


def build_teacher(
    spec: TaskSpec,
    settings: Settings,
    provider: Literal["claude", "ollama", "fake"] | None = None,
    transport: Transport | None = None,
    ollama: OllamaTransport | None = None,
) -> Teacher:
    """The teacher the spec asks for; `provider` overrides `teacher.provider` (CLI flag).

    Both model-backed teachers share one disk cache: a key hashes the whole request,
    model included, so their entries never collide."""
    chosen = provider or spec.teacher.provider
    if chosen == "fake":
        return FakeTeacher(spec)
    if chosen == "ollama":
        if spec.teacher.provider != "ollama":
            raise ConfigError(
                f"--teacher ollama needs a local model in the spec, not {spec.teacher.model!r};"
                " set teacher.provider: ollama and teacher.model (e.g. qwen2.5:3b)"
            )
        return OllamaTeacher(
            spec,
            ollama or HttpOllamaTransport(settings.ollama_url),
            cache=DiskCache(settings.home / "cache" / "teacher"),
        )
    if transport is None:
        transport = AnthropicTransport()  # raises ConfigError when the SDK finds no credentials
        if settings.record_dir is not None:
            transport = RecordingTransport(transport, settings.record_dir)
    cache = DiskCache(settings.home / "cache" / "teacher")
    return ClaudeTeacher(spec, transport, cache=cache, workers=settings.teacher_workers)
