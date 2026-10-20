"""Claude as the teacher: one structured-output request per input, cached and costed."""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, wait

from anvil.errors import TeacherError
from anvil.log import get_logger
from anvil.outputs import encode_output
from anvil.spans import Mention, encode_answer
from anvil.spec import TaskSpec
from anvil.teacher.base import BatchAbortedError, Failure, Item, Label, LabelBatch
from anvil.teacher.cache import DiskCache, request_key
from anvil.teacher.pricing import Usage, cost_usd, price_for
from anvil.teacher.transport import JSONDict, Transport

log = get_logger(__name__)


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


class ClaudeTeacher:
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
        self._spec = spec
        self._transport = transport
        self._cache = cache
        self._workers = workers
        self._system = system_prompt(spec)
        self._schema = output_schema(spec)

    @property
    def model_id(self) -> str:
        return self._spec.teacher.model

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

    def label(self, items: Sequence[Item]) -> LabelBatch:
        """Label every item, in input order.

        Items with identical requests (the same text under different ids) are sent once and the
        answer is shared, so a duplicate never costs a second call. If a request fails in a way
        that stops the run (a configuration error, Ctrl-C), the answers that did complete are
        handed back inside `BatchAbortedError` so the caller can persist what was paid for.
        """
        batch = LabelBatch()
        if not items:
            return batch
        params = [self.request_params(item) for item in items]
        keys = [request_key(p) for p in params]
        first: dict[str, int] = {}
        for i, key in enumerate(keys):
            first.setdefault(key, i)
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futures = {key: pool.submit(self._fetch, params[i], key) for key, i in first.items()}
            try:
                for i, item in enumerate(items):
                    fetched = futures[keys[i]].result()
                    batch.add(self._outcome(item, fetched, keys[i], shared=first[keys[i]] != i))
            except BaseException as exc:
                for future in futures.values():
                    future.cancel()
                wait(futures.values())
                partial = LabelBatch()
                for i, item in enumerate(items):
                    future = futures[keys[i]]
                    if future.cancelled() or future.exception() is not None:
                        continue
                    outcome = self._outcome(item, future.result(), keys[i], first[keys[i]] != i)
                    partial.add(outcome)
                raise BatchAbortedError(exc, partial) from exc
        log.info(
            "teacher batch",
            extra={
                "model": self.model_id,
                "labelled": len(batch.labels),
                "failed": len(batch.failures),
                "cache_hits": batch.cache_hits,
                "spent_usd": round(batch.spent_usd, 6),
            },
        )
        return batch

    def _fetch(self, params: JSONDict, key: str) -> tuple[JSONDict, bool] | TeacherError:
        """The response for one request and whether it came from the cache. A per-request API
        failure (5xx after retries, a dropped connection, an oversized input) is returned, not
        raised, so it fails one input; configuration errors propagate and stop the run."""
        cached = self._cache.get(key) if self._cache else None
        if cached is not None:
            return cached, True
        try:
            return self._transport(params), False
        except TeacherError as exc:
            return exc

    def _outcome(
        self, item: Item, fetched: tuple[JSONDict, bool] | TeacherError, key: str, shared: bool
    ) -> Label | Failure:
        if isinstance(fetched, TeacherError):
            return Failure(item.id, str(fetched))
        response, cached = fetched
        outcome = self._parse(item, response, cached=cached or shared)
        if shared and isinstance(outcome, Failure):
            # The one request was billed on the first item that shares it.
            outcome = Failure(item.id, outcome.reason)
        # Only valid answers are cached, so a refusal or truncation is retried next run.
        if not cached and not shared and self._cache is not None and isinstance(outcome, Label):
            self._cache.put(key, response)
        return outcome

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
            answer = json.loads(text)
        except json.JSONDecodeError:
            return fail(f"output is not JSON: {text[:80]!r}")
        if self._spec.kind == "extract":
            mentions = self._mentions(answer)
            if isinstance(mentions, str):
                return fail(mentions)
            value = encode_answer(mentions)
        elif self._spec.kind == "generate":
            if not isinstance(answer, dict):
                return fail(f"output is not a JSON object: {text[:80]!r}")
            try:
                value = encode_output(answer, self._spec.output_names)
            except ValueError as exc:
                return fail(str(exc))
        else:
            choice = answer.get("label") if isinstance(answer, dict) else None
            if not isinstance(choice, str) or choice not in self._spec.label_names:
                return fail(f"label {choice!r} is not in the label set")
            value = choice
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

    def _mentions(self, answer: object) -> list[Mention] | str:
        """The mentions of an extraction answer, or why it is invalid. Offsets are not checked
        against the input here: that is alignment's job, and it records what it repaired."""
        rows = answer.get("entities") if isinstance(answer, dict) else None
        if not isinstance(rows, list):
            return "output has no `entities` list"
        out: list[Mention] = []
        for row in rows:
            if not isinstance(row, dict):
                return f"entity is not an object: {row!r:.80}"
            text, kind, start = row.get("text"), row.get("type"), row.get("start")
            if not isinstance(text, str) or kind not in self._spec.label_names:
                return f"entity has no text or an unknown type: {row!r:.80}"
            if not isinstance(start, int) or isinstance(start, bool):
                return f"entity start is not an integer: {row!r:.80}"
            out.append(Mention(text, str(kind), start))
        return out
