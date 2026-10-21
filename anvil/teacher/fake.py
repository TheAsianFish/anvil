"""A deterministic rule-based teacher for tests and the offline demo. Never used for numbers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence

from anvil.errors import ConfigError
from anvil.outputs import encode_output
from anvil.spans import Mention, encode_answer
from anvil.spec import TaskSpec
from anvil.teacher.base import Failure, Item, Label, LabelBatch


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

    def __init__(self, spec: TaskSpec) -> None:
        if spec.fake is None:
            raise ConfigError(
                "--teacher fake needs a `fake:` section with keyword rules in the spec"
            )
        self._spec = spec
        self._fake = spec.fake
        self.model_id = self.model_id_for(spec)
        self._patterns = [
            (label, re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE))
            for label in spec.label_names
            if (words := self._fake.rules.get(label))
        ]
        self._default = self._fake.default or (spec.label_names or [""])[0]
        self._rewrites = [
            (re.compile(rw.pattern, re.IGNORECASE), rw.output) for rw in self._fake.rewrites
        ]

    @staticmethod
    def model_id_for(spec: TaskSpec) -> str:
        """`fake-rules-<hash of the fake section>`: labels are keyed by teacher model, and the
        rules, default, noise and seed are this teacher's whole behaviour, so editing them
        must start a new key just as a prompt edit does for Claude."""
        if spec.fake is None:
            raise ConfigError(
                "--teacher fake needs a `fake:` section with keyword rules in the spec"
            )
        blob = json.dumps(spec.fake.model_dump(), sort_keys=True, separators=(",", ":"))
        return "fake-rules-" + hashlib.sha256(blob.encode()).hexdigest()[:8]

    def rule_label(self, text: str) -> str:
        for label, pattern in self._patterns:
            if pattern.search(text):
                return label
        return self._default

    def mentions(self, item: Item) -> list[Mention]:
        """Extraction: every keyword occurrence is a mention of its type (the longest wins
        where two overlap), and with `noise` that share of mentions, chosen by a hash of the
        input id and offset, claims a start one character off."""
        found: list[tuple[int, int, str]] = []
        for label, pattern in self._patterns:
            found += [(m.start(), m.end(), label) for m in pattern.finditer(item.text)]
        found.sort(key=lambda f: (f[0] - f[1], f[0]))
        kept: list[tuple[int, int, str]] = []
        for start, end, label in found:
            if all(end <= k[0] or start >= k[1] for k in kept):
                kept.append((start, end, label))
        out: list[Mention] = []
        for start, end, label in sorted(kept):
            claimed = start
            if (
                self._fake.noise
                and _unit(self._fake.seed, "shift", item.id, start) < self._fake.noise
            ):
                down = _unit(self._fake.seed, "dir", item.id, start) < 0.5 and start > 0
                claimed = start - 1 if down else start + 1  # never a no-op at offset 0
            out.append(Mention(item.text[start:end], label, claimed))
        return out

    def rewrite(self, text: str) -> str | None:
        """Generate: the stored answer of the first rewrite whose pattern matches, or None."""
        for pattern, output in self._rewrites:
            match = pattern.search(text)
            if match is not None:
                fields = {name: match.expand(template) for name, template in output.items()}
                return encode_output(fields, self._spec.output_names)
        return None

    def label(self, items: Sequence[Item]) -> LabelBatch:
        if self._spec.kind == "generate":
            batch = LabelBatch()
            for item in items:
                value = self.rewrite(item.text)
                batch.add(
                    Failure(item.id, "no fake rewrite matches the input")
                    if value is None
                    else Label(
                        input_id=item.id,
                        label=value,
                        teacher_model=self.model_id,
                        prompt_hash=self._spec.prompt_hash,
                    )
                )
            return batch
        if self._spec.kind == "extract":
            return LabelBatch(
                labels=[
                    Label(
                        input_id=item.id,
                        label=encode_answer(self.mentions(item)),
                        teacher_model=self.model_id,
                        prompt_hash=self._spec.prompt_hash,
                    )
                    for item in items
                ]
            )
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
