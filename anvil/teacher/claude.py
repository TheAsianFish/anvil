"""Claude as the teacher: one structured-output request per input, cached and costed."""

from __future__ import annotations

from anvil.spec import TaskSpec
from anvil.teacher.base import Failure, Item, Label
from anvil.teacher.cache import DiskCache
from anvil.teacher.pricing import Usage, cost_usd, price_for
from anvil.teacher.requests import RequestTeacher, decode_answer
from anvil.teacher.transport import JSONDict, Transport

EXTRACT_RULES = """Return every entity mention in the input. For each one give:
- text: the mention copied exactly from the input, character for character;
- type: its entity type;
- start: the 0-based character offset where the mention begins in the input.
Mentions must not overlap. If the input has no entities, return an empty list."""


GENERATE_RULES = "Answer with a JSON object that gives every field."


def system_prompt(spec: TaskSpec) -> str:
    heading = {"extract": "Entity types:", "generate": "Fields:"}.get(spec.kind, "Labels:")
    lines = [spec.instructions.strip(), "", heading]
    for label in spec.outputs if spec.kind == "generate" else spec.labels:
        lines.append(
            f"- {label.name}: {label.description}" if label.description else f"- {label.name}"
        )
    if spec.kind == "extract":
        lines += ["", EXTRACT_RULES]
    elif spec.kind == "generate":
        lines += ["", GENERATE_RULES]
    else:
        lines += ["", "Answer with the single best label for the input."]
    return "\n".join(lines)


def output_schema(spec: TaskSpec) -> JSONDict:
    if spec.kind == "generate":
        return {
            "type": "object",
            "properties": {name: {"type": "string"} for name in spec.output_names},
            "required": spec.output_names,
            "additionalProperties": False,
        }
    if spec.kind == "extract":
        # The text is what alignment trusts; the offset only says which occurrence is meant.
        # No `minimum` on start: structured outputs do not support numeric bounds, and a
        # negative offset is simply repaired or dropped by alignment.
        mention = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "type": {"type": "string", "enum": spec.label_names},
                "start": {"type": "integer"},
            },
            "required": ["text", "type", "start"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {"entities": {"type": "array", "items": mention}},
            "required": ["entities"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": spec.label_names}},
        "required": ["label"],
        "additionalProperties": False,
    }


class ClaudeTeacher(RequestTeacher):
    """Labels inputs with Claude through structured outputs.

    Each input is its own request so the cache is per input: adding inputs or re-running a
    round only pays for inputs never seen with this exact prompt, model and settings.
    Refusals, truncation and schema violations become `Failure`s rather than guessed labels;
    no server-side model fallback is used, because a label from a different model would mix
    two teachers under one key and corrupt the agreement measurement.
    """

    def __init__(
        self,
        spec: TaskSpec,
        transport: Transport,
        cache: DiskCache | None = None,
        workers: int = 4,
    ) -> None:
        price_for(spec.teacher.model)  # fail fast on a model we cannot account for
        super().__init__(spec, cache, workers)
        self._transport = transport
        self._system = system_prompt(spec)
        self._schema = output_schema(spec)

    def request_params(self, item: Item) -> JSONDict:
        t = self._spec.teacher
        return {
            "model": t.model,
            "max_tokens": t.max_tokens,
            "system": self._system,
            "messages": [{"role": "user", "content": item.text}],
            "output_config": {
                "effort": t.effort,
                "format": {"type": "json_schema", "schema": self._schema},
            },
        }

    def _send(self, params: JSONDict) -> JSONDict:
        return self._transport(params)

    def _parse(self, item: Item, response: JSONDict, cached: bool) -> Label | Failure:
        usage = Usage.from_response(response.get("usage") or {})
        cost = cost_usd(self.model_id, usage)

        def fail(reason: str) -> Failure:
            return Failure(item.id, reason, usage.input_tokens, usage.output_tokens, cost)

        stop = response.get("stop_reason")
        if stop == "refusal":
            details = response.get("stop_details") or {}
            return fail(f"refusal ({details.get('category') or 'no category'})")
        if stop == "max_tokens":
            return fail(
                "hit max_tokens before answering (thinking counts); raise teacher.max_tokens"
            )
        text = "".join(
            block.get("text", "")
            for block in response.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        try:
            value = decode_answer(self._spec, text)
        except ValueError as exc:
            return fail(str(exc))
        return Label(
            input_id=item.id,
            label=value,
            teacher_model=self.model_id,
            prompt_hash=self._spec.prompt_hash,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost,
            cached=cached,
        )
