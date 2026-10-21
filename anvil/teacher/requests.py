"""What every model-backed teacher shares: one request per input, a per-input disk cache,
duplicate requests sent once, and answers decoded against the task's schema.

`ClaudeTeacher` and `OllamaTeacher` differ only in how a request is written, sent and read
back; the batching, caching and answer validation here are the same for both, so labels from
either teacher mean the same thing and land in the same store and label files.
"""

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
from anvil.teacher.transport import JSONDict

log = get_logger(__name__)


def decode_answer(spec: TaskSpec, text: str) -> str:
    """The stored label for a teacher's JSON answer `text`: the class name, the mentions'
    canonical JSON or the output fields' canonical JSON. Raises `ValueError` with the reason
    when the answer does not fit the task's schema."""
    try:
        answer = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"output is not JSON: {text[:80]!r}") from None
    if spec.kind == "extract":
        return encode_answer(_mentions(spec, answer))
    if spec.kind == "generate":
        if not isinstance(answer, dict):
            raise ValueError(f"output is not a JSON object: {text[:80]!r}")
        return encode_output(answer, spec.output_names)
    choice = answer.get("label") if isinstance(answer, dict) else None
    if not isinstance(choice, str) or choice not in spec.label_names:
        raise ValueError(f"label {choice!r} is not in the label set")
    return choice


def _mentions(spec: TaskSpec, answer: object) -> list[Mention]:
    """The mentions of an extraction answer. Offsets are not checked against the input here:
    that is alignment's job, and it records what it repaired."""
    rows = answer.get("entities") if isinstance(answer, dict) else None
    if not isinstance(rows, list):
        raise ValueError("output has no `entities` list")
    out: list[Mention] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"entity is not an object: {row!r:.80}")
        text, kind, start = row.get("text"), row.get("type"), row.get("start")
        if not isinstance(text, str) or kind not in spec.label_names:
            raise ValueError(f"entity has no text or an unknown type: {row!r:.80}")
        if not isinstance(start, int) or isinstance(start, bool):
            raise ValueError(f"entity start is not an integer: {row!r:.80}")
        out.append(Mention(text, str(kind), start))
    return out


class RequestTeacher:
    """Labels a batch by sending one request per distinct input, through a disk cache.

    Subclasses write the request (`request_params`), send it (`_send`) and turn a response into
    a label or a failure (`_parse`). Items with identical requests (the same text under
    different ids) are sent once and share the answer. If a request fails in a way that stops
    the run (a configuration error, Ctrl-C), the answers that did complete are handed back
    inside `BatchAbortedError` so the caller can persist what was paid for. Only valid answers
    are cached, so a refusal or a truncated answer is asked again next run.
    """

    def __init__(self, spec: TaskSpec, cache: DiskCache | None, workers: int) -> None:
        self._spec = spec
        self._cache = cache
        self._workers = workers

    @property
    def model_id(self) -> str:
        return self._spec.teacher.model

    def request_params(self, item: Item) -> JSONDict:
        raise NotImplementedError

    def cache_key(self, params: JSONDict) -> str:
        """The disk-cache key of a request: by default the hash of its exact parameters."""
        return request_key(params)

    def _send(self, params: JSONDict) -> JSONDict:
        raise NotImplementedError

    def _parse(self, item: Item, response: JSONDict, cached: bool) -> Label | Failure:
        raise NotImplementedError

    def label(self, items: Sequence[Item]) -> LabelBatch:
        """Label every item, in input order."""
        batch = LabelBatch()
        if not items:
            return batch
        params = [self.request_params(item) for item in items]
        keys = [self.cache_key(p) for p in params]
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
        """The response for one request and whether it came from the cache. A per-request
        failure (a server error after retries, a dropped connection, an oversized input) is
        returned, not raised, so it fails one input; configuration errors propagate and stop
        the run."""
        cached = self._cache.get(key) if self._cache else None
        if cached is not None:
            return cached, True
        try:
            return self._send(params), False
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
        if not cached and not shared and self._cache is not None and isinstance(outcome, Label):
            self._cache.put(key, response)
        return outcome
