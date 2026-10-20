"""Answers of a `generate` task: a JSON object of string fields, stored in a canonical form.

The teacher returns the fields through structured outputs; the store keeps them as one JSON
string with the fields in the spec's order, so the same answer is always the same string and
label files diff cleanly. How a student is trained to write an answer is a separate choice
(`anvil.chat.render_target`): the stored form is the teacher's, not the student's.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence


def encode_output(fields: Mapping[str, str], names: Sequence[str]) -> str:
    """The canonical stored form of an answer: its `names` fields in that order, compact JSON.

    `ValueError` when a field is missing or not a string, or an unknown field is present."""
    unknown = sorted(set(fields) - set(names))
    if unknown:
        raise ValueError(f"answer has fields {unknown} the task does not define")
    out: dict[str, str] = {}
    for name in names:
        value = fields.get(name)
        if not isinstance(value, str):
            raise ValueError(f"answer field {name!r} must be a string, got {value!r:.80}")
        out[name] = value
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def decode_output(label: str, names: Sequence[str]) -> dict[str, str]:
    """The fields of a stored answer; `ValueError` when it is not an answer to this task."""
    try:
        obj = json.loads(label)
    except json.JSONDecodeError as exc:
        raise ValueError(f"generate label is not JSON: {label[:80]!r}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"generate label must be a JSON object, got {label[:80]!r}")
    encode_output(obj, names)  # the same checks the teacher's answer passed
    return {name: str(obj[name]) for name in names}
