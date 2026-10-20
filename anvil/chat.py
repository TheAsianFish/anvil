"""Chat-template alignment: turning (input, teacher answer) into a decoder's training sequence.

A generation student is an instruction-tuned decoder, so it is trained on exactly the text it
will see when it answers: its own chat template around a system message (the task's
instructions) and a user message (the input), followed by the answer and the token that
ends the assistant's turn. Only the answer and that end token are scored; the prompt tokens
get the ignored label id, so the loss teaches what to say, not how to repeat the question.

Three things are checked rather than assumed, because each one fails silently otherwise:

- the prompt the student generates from is a token-for-token prefix of the training
  sequence (a template or tokenizer that merges across the boundary would train the student
  on a prompt it never sees at inference);
- the answer ends with the end-of-turn token, and nothing the template adds after it (Qwen
  writes a newline) is trained, since generation stops at that token;
- the answer's tokens decode back to the answer (a template that trims or rewrites the
  content would teach a different string from the teacher's).

An example whose sequence is longer than `max_length` is left out, never cut: a truncated
answer would teach the student to stop mid-query. So is one whose answer has more tokens than
the student may generate (`max_new_tokens`): it would train fine and could never be answered
in full, which caps exact match silently. Like span alignment, the losses are counted, and
training refuses to start when they pass `MAX_DROPPED_RATE`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from anvil.errors import AlignmentError, StudentError
from anvil.outputs import encode_output

IGNORE = -100
"""Label id of tokens the loss skips (the cross-entropy's `ignore_index`)."""

MAX_DROPPED_RATE = 0.05
"""Training refuses to start when more than this share of examples cannot be used."""

DROP_REASONS = ("too_long", "answer_too_long", "not_prefix", "not_roundtrip")


def messages(system: str, text: str, answer: str | None = None) -> list[dict[str, str]]:
    """The conversation of one example; with `answer`, including the assistant's turn."""
    out = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    if answer is not None:
        out.append({"role": "assistant", "content": answer})
    return out


def render_target(fields: Mapping[str, str], names: Sequence[str]) -> str:
    """What the student is trained to write for an answer.

    A one-field answer is its bare value: wrapping a SQL query in JSON would make the student
    escape every double-quoted column name, a skill the task does not need, and spend tokens
    on the wrapper. Several fields are the canonical JSON object (`anvil.outputs`).

    Surrounding whitespace is dropped from a bare value: a leading newline would merge with
    the template's own newline into one token (Qwen's `\n\n`), so the generation prompt
    would no longer be a prefix of the training sequence, and answers are compared stripped
    anyway (`parse_completion`)."""
    if len(names) == 1:
        return fields[names[0]].strip()
    return encode_output(fields, names)


def parse_completion(text: str, names: Sequence[str]) -> dict[str, str] | None:
    """A student completion read back into fields; None when it is not a valid answer (a
    multi-field completion that is not the JSON object of the task's fields)."""
    if len(names) == 1:
        return {names[0]: text.strip()}
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or sorted(obj) != sorted(names):
        return None
    if not all(isinstance(obj[n], str) for n in names):
        return None
    return {n: str(obj[n]) for n in names}


@dataclass(frozen=True)
class Example:
    """One training sequence: token ids and per-token labels (`IGNORE` on the prompt)."""

    input_ids: list[int]
    labels: list[int]

    @property
    def answer_tokens(self) -> int:
        return sum(1 for x in self.labels if x != IGNORE)


@dataclass
class ChatReport:
    """What aligning a set of examples to the chat template cost."""

    examples: int = 0
    dropped: dict[str, int] = field(default_factory=lambda: dict.fromkeys(DROP_REASONS, 0))
    longest: int = 0
    """Tokens in the longest full sequence, kept or not (what `max_length` must hold)."""
    longest_answer: int = 0
    """Tokens in the longest answer with its end token (what `max_new_tokens` must hold)."""
    dropped_ids: list[str] = field(default_factory=list)

    @property
    def unusable(self) -> int:
        return sum(self.dropped.values())

    @property
    def dropped_rate(self) -> float:
        return self.unusable / self.examples if self.examples else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "examples": self.examples,
            "dropped": dict(self.dropped),
            "dropped_rate": self.dropped_rate,
            "longest": self.longest,
            "longest_answer": self.longest_answer,
            "dropped_ids": self.dropped_ids[:20],
        }

    def check(self, max_rate: float = MAX_DROPPED_RATE) -> None:
        """Raise `AlignmentError` when too many examples were dropped to train on the rest."""
        if self.examples and self.dropped_rate > max_rate:
            reasons = ", ".join(f"{k} {v}" for k, v in self.dropped.items() if v)
            hints = []
            if self.dropped["too_long"]:
                hints.append(
                    f"raise student.max_length (the longest sequence is {self.longest} tokens)"
                    " or shorten the inputs"
                )
            if self.dropped["answer_too_long"]:
                hints.append(
                    "raise student.max_new_tokens (the longest answer is"
                    f" {self.longest_answer} tokens with its end token)"
                )
            if self.dropped["not_prefix"]:
                hints.append(
                    "the template or tokenizer merges the end of the prompt with the start of"
                    " these answers, so the student would train on a prompt it never sees;"
                    " use a tokenizer whose template tokenizes the two consistently"
                )
            if self.dropped["not_roundtrip"]:
                hints.append(
                    "use a model whose chat template keeps the answer text as given (it"
                    " trims or rewrites what the teacher wrote)"
                )
            raise AlignmentError(
                f"{self.unusable} of {self.examples} examples ({self.dropped_rate:.1%}) cannot"
                f" be trained on ({reasons}), above the {max_rate:.0%} limit; " + "; ".join(hints)
            )


def _ids(encoded: Any) -> list[int]:  # Any: a BatchEncoding, a dict or a list of ids
    """Token ids from `apply_chat_template(tokenize=True)`, which returns a `BatchEncoding`
    in transformers 5 and a plain list in earlier versions."""
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    return [int(x) for x in ids]


def prompt_ids(tokenizer: Any, system: str, text: str) -> list[int]:  # Any: a HF tokenizer
    """The tokens the student generates from: the conversation up to the assistant's turn."""
    return _ids(
        tokenizer.apply_chat_template(
            messages(system, text), add_generation_prompt=True, tokenize=True
        )
    )


def end_of_turn_id(tokenizer: Any) -> int:  # Any: a HF tokenizer
    """The token the chat template closes an assistant turn with (Qwen's `<|im_end|>`): the
    first special token after a probe answer. Generation stops on it."""
    probe = "x"
    prompt = prompt_ids(tokenizer, "s", "u")
    full = _ids(tokenizer.apply_chat_template(messages("s", "u", probe), tokenize=True))
    special = set(tokenizer.all_special_ids)
    for token in full[len(prompt) :]:
        if token in special:
            return token
    raise StudentError(
        "the chat template does not close the assistant's turn with a special token, so"
        " generation would never stop; use an instruction-tuned model with a chat template"
    )


def build_example(
    tokenizer: Any,  # Any: a HF tokenizer with a chat template
    system: str,
    text: str,
    target: str,
    max_length: int,
    end_id: int,
    max_answer: int | None = None,
) -> Example | str:
    """The training sequence of one example, or the reason it cannot be used (one of
    `DROP_REASONS`). `max_answer` is the most tokens an answer may have with its end token
    (the student's `max_new_tokens`); None does not check it. Raises `StudentError` when the
    template itself is unusable (it closes the assistant's turn without the end token): that
    is not a property of one example, and every other example would fail the same way."""
    return _build(tokenizer, system, text, target, max_length, end_id, max_answer)[0]


def _build(
    tokenizer: Any,  # Any: a HF tokenizer with a chat template
    system: str,
    text: str,
    target: str,
    max_length: int,
    end_id: int,
    max_answer: int | None,
) -> tuple[Example | str, int, int]:
    """`build_example`, and the lengths of the full sequence and of the answer with its end
    token, whether or not the example is kept."""
    prompt = prompt_ids(tokenizer, system, text)
    full = _ids(tokenizer.apply_chat_template(messages(system, text, target), tokenize=True))
    answer = full[len(prompt) :]
    if end_id not in answer:
        raise StudentError(
            "the templated answer does not end with the end-of-turn token; the chat"
            " template closes assistant turns differently from the probe it was checked with"
        )
    answer = answer[: answer.index(end_id) + 1]  # what follows the end token is never generated
    length = len(prompt) + len(answer)
    if full[: len(prompt)] != prompt:
        # The tokenizer merged the end of the prompt with the start of this answer (a
        # template's trailing newline with an answer's leading one), so the sequence does
        # not start with the prompt the student generates from. One answer's property, not
        # the template's: it is dropped and counted, and the gate reports a template that
        # does it to every answer.
        return "not_prefix", length, len(answer)
    if length > max_length:
        return "too_long", length, len(answer)
    if max_answer is not None and len(answer) > max_answer:
        return "answer_too_long", length, len(answer)
    if tokenizer.decode(answer[:-1]) != target:
        return "not_roundtrip", length, len(answer)
    return Example(prompt + answer, [IGNORE] * len(prompt) + answer), length, len(answer)


def build_examples(
    tokenizer: Any,  # Any: a HF tokenizer with a chat template
    system: str,
    texts: Sequence[str],
    targets: Sequence[str],
    max_length: int,
    end_id: int,
    input_ids: Sequence[str] | None = None,
    max_answer: int | None = None,
) -> tuple[list[Example | None], ChatReport]:
    """`build_example` over a dataset: one entry per text (None where dropped) and the report."""
    if len(texts) != len(targets):
        raise StudentError(f"{len(texts)} texts but {len(targets)} targets")
    report = ChatReport()
    out: list[Example | None] = []
    for i, (text, target) in enumerate(zip(texts, targets, strict=True)):
        report.examples += 1
        built, length, answer = _build(
            tokenizer, system, text, target, max_length, end_id, max_answer
        )
        report.longest = max(report.longest, length)
        report.longest_answer = max(report.longest_answer, answer)
        if isinstance(built, str):
            report.dropped[built] += 1
            report.dropped_ids.append(input_ids[i] if input_ids is not None else str(i))
            out.append(None)
            continue
        out.append(built)
    return out, report
