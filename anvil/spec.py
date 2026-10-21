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
    provider: Literal["claude", "ollama", "fake"] = "claude"
    model: str = "claude-opus-5-5"
    """A Claude model id, or for `ollama` a local model tag such as `qwen2.5:3b`."""
    effort: Literal["low", "medium", "high"] = "low"
    """Claude's effort setting; a local model has none, so `ollama` ignores it."""
    max_tokens: int = Field(default=4096, ge=512, le=16000)
    """Thinking tokens count toward this and Opus 5.5 always thinks; unused tokens are free."""
    digest: str | None = None
    """`ollama` only: the sha256 of the model's weights (`ollama list` shows its first 12
    characters). A tag such as `qwen2.5:3b` can be re-pushed with other weights, so a pinned
    digest makes the teacher refuse a server whose model has changed."""

    @field_validator("digest")
    @classmethod
    def _hex_digest(cls, v: str | None) -> str | None:
        if v is not None and not re.fullmatch(r"(sha256:)?[0-9a-f]{64}", v):
            raise ValueError("digest must be 64 hex characters (the model's sha256)")
        return v.removeprefix("sha256:") if v is not None else None

    @model_validator(mode="after")
    def _digest_is_local(self) -> TeacherSpec:
        if self.digest is not None and self.provider != "ollama":
            raise ValueError("teacher.digest pins a local model; it needs provider: ollama")
        return self


DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
"""Every linear layer of a Qwen2 (or Llama-style) decoder block: adapting attention alone
leaves the MLPs, which hold most of the weights, frozen."""


class LoraSpec(_Strict):
    """LoRA settings of a `generate` student: the adapter is `B @ A` of rank `r`, scaled by
    `alpha / r`, on each named linear layer; the base weights stay frozen."""

    r: int = Field(default=16, ge=1, le=256)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: list[str] = Field(default_factory=lambda: list(DEFAULT_LORA_TARGETS))

    @field_validator("target_modules")
    @classmethod
    def _some_targets(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("LoRA needs at least one target module")
        return v


class StudentSpec(_Strict):
    model: str = "tiny"
    """`tiny` builds a small random-init model for tests and the demo (a ModernBERT for
    classify and extract, a Qwen2 decoder for generate); anything else is a Hugging Face model
    id (defaults for real runs: `answerdotai/ModernBERT-base`, `Qwen/Qwen2.5-0.5B-Instruct`)."""
    max_length: int = Field(default=64, ge=8, le=8192)
    """Tokens per input for the encoders; for a generator, the whole training sequence (chat
    prompt plus answer), and an example longer than this is left out of training, not cut."""
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
    lora: LoraSpec = LoraSpec()
    """Generate students only: the adapter trained on the frozen decoder."""
    max_new_tokens: int = Field(default=128, ge=1, le=4096)
    """Generate students only: the longest answer decoding may produce."""


class SplitSpec(_Strict):
    heldout_fraction: float = Field(default=0.25, gt=0, lt=1)
    seed: int = 0


class Rewrite(_Strict):
    """A `generate` rule of the fake teacher: when `pattern` matches the input (a case
    insensitive `re.search`), each output field is its template with the match's groups
    expanded (`\\1`, `\\g<name>`)."""

    pattern: str
    output: dict[str, str]

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"pattern {v!r} is not a regular expression: {exc}") from None
        return v


class FakeTeacherSpec(_Strict):
    """Keyword rules for the offline `FakeTeacher` (demo and tests only).

    For `classify`, the first label with a keyword present wins (else `default`), and `noise`
    flips that share of labels. For `extract`, every keyword occurrence is a mention of its
    type, and `noise` shifts that share of mentions' offsets by one character, the error a
    real teacher makes most, so alignment has something to repair. For `generate`, the first
    of `rewrites` whose pattern matches writes the answer, and an input none matches fails
    (as a refusal would)."""

    rules: dict[str, list[str]] = Field(default_factory=dict)
    rewrites: list[Rewrite] = Field(default_factory=list)
    default: str | None = None
    noise: float = Field(default=0.0, ge=0, le=1)
    seed: int = 0


class TaskSpec(_Strict):
    name: str
    kind: Literal["classify", "extract", "generate"]
    instructions: str
    labels: list[LabelDef] = Field(default_factory=list)
    """Classes for `classify`; entity types for `extract`."""
    outputs: list[LabelDef] = Field(default_factory=list)
    """Fields of a `generate` answer (each a string), in the order the teacher writes them."""
    teacher: TeacherSpec = TeacherSpec()
    student: StudentSpec = StudentSpec()
    splits: SplitSpec = SplitSpec()
    fake: FakeTeacherSpec | None = None

    @field_validator("labels", "outputs", mode="before")
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
            return self._check_generate()
        if self.outputs:
            raise ValueError(f"`outputs` is for generate tasks; a {self.kind} task uses `labels`")
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
            if self.fake.rewrites:
                raise ValueError("fake.rewrites are for generate tasks; use fake.rules")
        return self

    def _check_generate(self) -> TaskSpec:
        if self.labels:
            raise ValueError("a generate task lists its answer fields under `outputs`, not labels")
        names = self.output_names
        if not names:
            raise ValueError("a generate task needs at least one field under `outputs`")
        if len(set(names)) != len(names):
            raise ValueError("output field names must be unique")
        if self.fake is not None:
            if self.fake.rules or self.fake.default is not None:
                raise ValueError("a generate task's fake teacher uses fake.rewrites, not rules")
            for rw in self.fake.rewrites:
                if sorted(rw.output) != sorted(names):
                    raise ValueError(
                        f"fake rewrite {rw.pattern!r} writes fields {sorted(rw.output)}, the"
                        f" task's outputs are {sorted(names)}"
                    )
        return self

    @property
    def output_names(self) -> list[str]:
        return [field.name for field in self.outputs]

    @property
    def label_names(self) -> list[str]:
        return [label.name for label in self.labels]

    @property
    def prompt_hash(self) -> str:
        """Hash of everything that changes the teacher's answers, except the model tag.

        Labels are keyed by (teacher model, prompt hash), so editing the instructions, the label
        set or the teacher's effort produces new labels instead of silently mixing old and new
        ones. `max_tokens` is left out: it decides whether an answer is truncated (a failure),
        not which label a completed answer has. `PROMPT_FORMAT_VERSION` is in, so a change to
        how the spec is rendered into the request also starts a new key. A pinned
        `teacher.digest` is in too: a local model's tag can be re-pushed with other weights, and
        those are another teacher whose labels must not be served or topped up as this one's.
        """
        payload: dict[str, object] = {
            "format": PROMPT_FORMAT_VERSION,
            "kind": self.kind,
            "instructions": self.instructions.strip(),
            "labels": [label.model_dump() for label in self.labels],
            "effort": self.teacher.effort,
        }
        if self.kind == "generate":  # only here, so classify and extract keys stay as they were
            payload["outputs"] = [field.model_dump() for field in self.outputs]
        if self.teacher.digest is not None:  # only when pinned, so every other key is unchanged
            payload["digest"] = self.teacher.digest
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
