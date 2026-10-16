"""Claude as the teacher: one structured-output request per input, cached and costed."""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from anvil.errors import TeacherError
from anvil.log import get_logger
from anvil.spec import TaskSpec
from anvil.teacher.base import Failure, Item, Label, LabelBatch
from anvil.teacher.cache import DiskCache, request_key
from anvil.teacher.pricing import Usage, cost_usd, price_for
from anvil.teacher.transport import JSONDict, Transport

log = get_logger(__name__)


def system_prompt(spec: TaskSpec) -> str:
    lines = [spec.instructions.strip(), "", "Labels:"]
    for label in spec.labels:
        lines.append(
            f"- {label.name}: {label.description}" if label.description else f"- {label.name}"
        )
    lines += ["", "Answer with the single best label for the input."]
    return "\n".join(lines)


def output_schema(spec: TaskSpec) -> JSONDict:
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
        batch = LabelBatch()
        if not items:
            return batch
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            for result in pool.map(self._label_one, items):
                if isinstance(result, Label):
                    batch.labels.append(result)
                else:
                    batch.failures.append(result)
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

    def _label_one(self, item: Item) -> Label | Failure:
        params = self.request_params(item)
        key = request_key(params)
        cached = self._cache.get(key) if self._cache else None
        if cached is not None:
            response = cached
        else:
            try:
                response = self._transport(params)
            except TeacherError as exc:
                # One input's API failure (5xx after retries, a dropped connection) must not
                # throw away the rest of the batch; configuration errors still propagate.
                return Failure(item.id, str(exc))
        outcome = self._parse(item, response, cached=cached is not None)
        # Only valid answers are cached, so a refusal or truncation is retried next run.
        if cached is None and self._cache is not None and isinstance(outcome, Label):
            self._cache.put(key, response)
        return outcome

    def _parse(self, item: Item, response: JSONDict, cached: bool) -> Label | Failure:
        stop = response.get("stop_reason")
        if stop == "refusal":
            details = response.get("stop_details") or {}
            return Failure(item.id, f"refusal ({details.get('category') or 'no category'})")
        if stop == "max_tokens":
            return Failure(item.id, "hit max_tokens before answering; raise teacher.max_tokens")
        text = "".join(
            block.get("text", "")
            for block in response.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        try:
            answer = json.loads(text)
        except json.JSONDecodeError:
            return Failure(item.id, f"output is not JSON: {text[:80]!r}")
        value = answer.get("label") if isinstance(answer, dict) else None
        if value not in self._spec.label_names:
            return Failure(item.id, f"label {value!r} is not in the label set")
        usage = Usage.from_response(response.get("usage") or {})
        return Label(
            input_id=item.id,
            label=value,
            teacher_model=self.model_id,
            prompt_hash=self._spec.prompt_hash,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost_usd(self.model_id, usage),
            cached=cached,
        )
