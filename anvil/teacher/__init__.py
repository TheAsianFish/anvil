"""Teachers: the models whose answers the student learns to reproduce."""

from __future__ import annotations

from typing import Literal

from anvil.errors import ConfigError
from anvil.settings import Settings
from anvil.spec import TaskSpec
from anvil.teacher.base import Failure, Item, Label, LabelBatch, Teacher
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
    "ClaudeTeacher",
    "DiskCache",
    "Failure",
    "FakeTeacher",
    "Item",
    "Label",
    "LabelBatch",
    "ReplayTransport",
    "Teacher",
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
        if not settings.has_api_key:
            raise ConfigError(
                "the Claude teacher needs ANTHROPIC_API_KEY; set it, or pass --teacher fake to"
                " run offline with the spec's keyword rules"
            )
        transport = AnthropicTransport()
        if settings.record_dir is not None:
            transport = RecordingTransport(transport, settings.record_dir)
    cache = DiskCache(settings.home / "cache" / "teacher")
    return ClaudeTeacher(spec, transport, cache=cache, workers=settings.teacher_workers)
