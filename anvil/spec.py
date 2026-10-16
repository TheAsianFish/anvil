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

_LABEL_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_\-]*$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LabelDef(_Strict):
    name: str
    description: str = ""

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        if not _LABEL_RE.match(v):
            raise ValueError(f"label {v!r} must be letters, digits, '_' or '-'")
        return v


class TeacherSpec(_Strict):
    provider: Literal["claude", "fake"] = "claude"
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high"] = "low"
    max_tokens: int = Field(default=1024, ge=64, le=16000)


class StudentSpec(_Strict):
    model: str = "tiny"
    """`tiny` builds a small random-init ModernBERT for tests and the demo; anything else is a
    Hugging Face model id (default for real runs: `answerdotai/ModernBERT-base`)."""
    max_length: int = Field(default=64, ge=8, le=8192)
    epochs: int = Field(default=8, ge=1)
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=5e-5, gt=0)
    seed: int = 0


class SplitSpec(_Strict):
    heldout_fraction: float = Field(default=0.25, gt=0, lt=1)
    seed: int = 0


class FakeTeacherSpec(_Strict):
    """Keyword rules for the offline `FakeTeacher` (demo and tests only)."""

    rules: dict[str, list[str]] = Field(default_factory=dict)
    default: str | None = None
    noise: float = Field(default=0.0, ge=0, le=1)
    seed: int = 0


class TaskSpec(_Strict):
    name: str
    kind: Literal["classify", "extract", "generate"]
    instructions: str
    labels: list[LabelDef] = Field(default_factory=list)
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
        if not _LABEL_RE.match(v):
            raise ValueError("task name must be letters, digits, '_' or '-' (it names a folder)")
        return v

    @model_validator(mode="after")
    def _check_kind(self) -> TaskSpec:
        if self.kind != "classify":
            raise ValueError(
                f"kind {self.kind!r} is not implemented yet (extract lands in M4, generate in M6);"
                " use kind: classify"
            )
        names = [label.name for label in self.labels]
        if len(names) < 2:
            raise ValueError("a classify task needs at least two labels")
        if len(set(names)) != len(names):
            raise ValueError("label names must be unique")
        if self.fake is not None:
            unknown = (set(self.fake.rules) | {self.fake.default or names[0]}) - set(names)
            if unknown:
                raise ValueError(f"fake rules name unknown labels: {sorted(unknown)}")
        return self

    @property
    def label_names(self) -> list[str]:
        return [label.name for label in self.labels]

    @property
    def prompt_hash(self) -> str:
        """Hash of everything that changes what the teacher is asked (not which teacher).

        Labels are keyed by (teacher model, prompt hash), so editing the instructions or the
        label set produces new labels instead of silently mixing old and new ones.
        """
        payload = {
            "kind": self.kind,
            "instructions": self.instructions.strip(),
            "labels": [label.model_dump() for label in self.labels],
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
