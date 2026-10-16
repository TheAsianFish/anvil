"""Teachers: the models whose answers the student learns to reproduce."""

from __future__ import annotations

from typing import Literal

from anvil.settings import Settings
from anvil.spec import TaskSpec
from anvil.teacher.base import BatchAbortedError, Failure, Item, Label, LabelBatch, Teacher
from anvil.teacher.cache import DiskCache
from anvil.teacher.claude import ClaudeTeacher
from anvil.teacher.fake import FakeTeacher
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
    "Item",
    "Label",
    "LabelBatch",
    "RecordingTransport",
    "ReplayTransport",
    "Teacher",
    "Transport",
    "build_teacher",
]


def build_teacher(
    spec: TaskSpec,
    settings: Settings,
    provider: Literal["claude", "fake"] | None = None,
    transport: Transport | None = None,
) -> Teacher:
    """The teacher the spec asks for; `provider` overrides `teacher.provider` (CLI flag)."""
    chosen = provider or spec.teacher.provider
    if chosen == "fake":
        return FakeTeacher(spec)
    if transport is None:
        transport = AnthropicTransport()  # raises ConfigError when the SDK finds no credentials
        if settings.record_dir is not None:
            transport = RecordingTransport(transport, settings.record_dir)
    cache = DiskCache(settings.home / "cache" / "teacher")
    return ClaudeTeacher(spec, transport, cache=cache, workers=settings.teacher_workers)
