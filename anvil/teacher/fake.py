"""A deterministic rule-based teacher for tests and the offline demo. Never used for numbers."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from anvil.errors import ConfigError
from anvil.spec import TaskSpec
from anvil.teacher.base import Item, Label, LabelBatch


def _unit(*parts: object) -> float:
    """A stable number in [0, 1) from its arguments."""
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class FakeTeacher:
    """Labels by keyword rules from the spec's `fake` section, with optional seeded noise.

    The first label (in spec order) with a keyword present wins; otherwise `default`. With
    `noise > 0`, that fraction of inputs (chosen by a hash of the input id, so the same
    inputs every run) gets a different label, which makes the teacher imperfect the way
    a real one is.
    """

    model_id = "fake-rules"

    def __init__(self, spec: TaskSpec) -> None:
        if spec.fake is None:
            raise ConfigError(
                "--teacher fake needs a `fake:` section with keyword rules in the spec"
            )
        self._spec = spec
        self._fake = spec.fake
        self._patterns = [
            (label, re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE))
            for label in spec.label_names
            if (words := self._fake.rules.get(label))
        ]
        self._default = self._fake.default or spec.label_names[0]

    def rule_label(self, text: str) -> str:
        for label, pattern in self._patterns:
            if pattern.search(text):
                return label
        return self._default

    def label(self, items: Sequence[Item]) -> LabelBatch:
        names = self._spec.label_names
        batch = LabelBatch()
        for item in items:
            value = self.rule_label(item.text)
            if self._fake.noise and _unit(self._fake.seed, "noise", item.id) < self._fake.noise:
                others = [n for n in names if n != value]
                value = others[int(_unit(self._fake.seed, "flip", item.id) * len(others))]
            batch.labels.append(
                Label(
                    input_id=item.id,
                    label=value,
                    teacher_model=self.model_id,
                    prompt_hash=self._spec.prompt_hash,
                )
            )
        return batch
