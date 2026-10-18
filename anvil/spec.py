"""The task spec (`anvil.yaml`): what to distil, which teacher, which student."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from anvil.errors import ConfigError

_LABEL_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-]*")

PROMPT_FORMAT_VERSION = 1
"""Bump when the way a spec is rendered into the teacher request changes (the system prompt
layout in `teacher/claude.py` or the output schema), so labels made under the old rendering
are never mixed with new ones."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LabelDef(_Strict):
    name: str
    description: str = ""

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        if not _LABEL_RE.fullmatch(v):
            raise ValueError(f"label {v!r} must be letters, digits, '_' or '-'")
        return v


class TeacherSpec(_Strict):
    provider: Literal["claude", "fake"] = "claude"
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high"] = "low"
    max_tokens: int = Field(default=4096, ge=512, le=16000)
    """Thinking tokens count toward this and Opus 5.5 always thinks; unused tokens are free."""


class StudentSpec(_Strict):
    model: str = "tiny"
    """`tiny` builds a small random-init ModernBERT for tests and the demo; anything else is a
    Hugging Face model id (default for real runs: `answerdotai/ModernBERT-base`)."""
    max_length: int = Field(default=64, ge=8, le=8192)
    epochs: int = Field(default=8, ge=1)
    min_steps: int = Field(default=0, ge=0)
    """Raise `epochs` until training runs at least this many optimizer steps: with 100 labels
    a fixed epoch count is a handful of updates, which would under-train the small end of a
    label-count curve. Early stopping still cuts the extra epochs when they do not help."""
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=5e-5, gt=0)
    seed: int = 0
    grad_accum: int = Field(default=1, ge=1)
    """Micro-batches per optimizer step; the effective batch is `batch_size * grad_accum`."""
    warmup_ratio: float = Field(default=0.1, ge=0, le=0.5)
    """Fraction of optimizer steps with linear warmup from 0, then linear decay to 0."""
    weight_decay: float = Field(default=0.01, ge=0)
    """AdamW weight decay, not applied to biases and norm weights."""
    max_grad_norm: float | None = Field(default=1.0, gt=0)
    """Clip the global gradient norm to this; null disables clipping."""
    amp: Literal["auto", "off", "bf16", "fp16"] = "auto"
    """Mixed precision: `auto` is bf16 or fp16 on CUDA and off on CPU."""
    validation_fraction: float = Field(default=0.1, ge=0, lt=0.5)
    """Share of the training labels held back to pick the best epoch (0 disables)."""
    patience: int = Field(default=2, ge=1)
    """Epochs without a better validation kappa before training stops early."""


class SplitSpec(_Strict):
    heldout_fraction: float = Field(default=0.25, gt=0, lt=1)
    seed: int = 0


class FakeTeacherSpec(_Strict):
    """Keyword rules for the offline `FakeTeacher` (demo and tests only).

    For `classify`, the first label with a keyword present wins (else `default`), and `noise`
    flips that share of labels. For `extract`, every keyword occurrence is a mention of its
    type, and `noise` shifts that share of mentions' offsets by one character, the error a
    real teacher makes most, so alignment has something to repair."""

    rules: dict[str, list[str]] = Field(default_factory=dict)
    default: str | None = None
    noise: float = Field(default=0.0, ge=0, le=1)
    seed: int = 0


class TaskSpec(_Strict):
    name: str
    kind: Literal["classify", "extract", "generate"]
    instructions: str
    labels: list[LabelDef] = Field(default_factory=list)
    """Classes for `classify`; entity types for `extract`."""
    teacher: TeacherSpec = TeacherSpec()
    student: StudentSpec = StudentSpec()
    splits: SplitSpec = SplitSpec()
    fake: FakeTeacherSpec | None = None

    @field_validator("labels", mode="before")
    @classmethod
    def _labels_from_strings(cls, v: object) -> object:
        if isinstance(v, list):
            return [{"name": x} if isinstance(x, str) else x for x in v]
        return v

    @field_validator("name")
    @classmethod
    def _valid_task_name(cls, v: str) -> str:
        if not _LABEL_RE.fullmatch(v):
            raise ValueError("task name must be letters, digits, '_' or '-' (it names a folder)")
        return v

    @model_validator(mode="after")
    def _check_kind(self) -> TaskSpec:
        if self.kind == "generate":
            raise ValueError(
                "kind 'generate' is not implemented yet (it lands in M6); use classify or extract"
            )
        names = [label.name for label in self.labels]
        if self.kind == "classify" and len(names) < 2:
            raise ValueError("a classify task needs at least two labels")
        if self.kind == "extract" and not names:
            raise ValueError("an extract task needs at least one entity type under `labels`")
        if len(set(names)) != len(names):
            raise ValueError("label names must be unique")
        if self.fake is not None:
            known = set(self.fake.rules)
            if self.kind == "classify":
                known |= {self.fake.default or names[0]}
            elif self.fake.default is not None:
                raise ValueError("fake.default has no meaning for an extract task; remove it")
            unknown = known - set(names)
            if unknown:
                raise ValueError(f"fake rules name unknown labels: {sorted(unknown)}")
        return self

    @property
    def label_names(self) -> list[str]:
        return [label.name for label in self.labels]

    @property
    def prompt_hash(self) -> str:
        """Hash of everything that changes the teacher's answers, except the model id.

        Labels are keyed by (teacher model, prompt hash), so editing the instructions, the label
        set or the teacher's effort produces new labels instead of silently mixing old and new
        ones. `max_tokens` is left out: it decides whether an answer is truncated (a failure),
        not which label a completed answer has. `PROMPT_FORMAT_VERSION` is in, so a change to
        how the spec is rendered into the request also starts a new key.
        """
        payload = {
            "format": PROMPT_FORMAT_VERSION,
            "kind": self.kind,
            "instructions": self.instructions.strip(),
            "labels": [label.model_dump() for label in self.labels],
            "effort": self.teacher.effort,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def load_spec(path: Path) -> TaskSpec:
    """Read and validate `anvil.yaml`, turning every failure into a `ConfigError`."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(
            f"spec file {path} not found; pass --spec or run from its folder"
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must be a mapping with name, kind, instructions and labels")
    try:
        return TaskSpec.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigError(f"{path} is not a valid task spec: {problems}") from exc
