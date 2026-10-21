"""A local open model as the teacher, served by Ollama: free, offline, and fallible.

The request carries the same system prompt and JSON schema as the Claude teacher (the schema
goes in Ollama's `format`, which constrains decoding to it), and the answer goes through the
same decoding and validation, so a label means the same thing whichever teacher made it and
lands in the same store, disk cache and label files. What differs is the accounting: a label
costs $0, and the server's own time for the request (`total_duration`) is recorded as
`compute_seconds` instead; tokens are Ollama's counts (`prompt_eval_count`, `eval_count`).

A model tag such as `qwen2.5:3b` names whatever weights were last pushed under it, so the
teacher reads the weights' digest from the server and puts it in the disk-cache key; with
`teacher.digest` pinned in the spec it refuses a server whose model has other weights.
Decoding is greedy (temperature 0, a fixed seed) and requests go one at a time by default:
the server batches concurrent requests, and a batch changes the arithmetic enough to flip a
close label, so one worker is what makes a re-run give the same labels.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Protocol

from anvil.errors import ConfigError, TeacherError
from anvil.spec import TaskSpec
from anvil.teacher.base import Failure, Item, Label
from anvil.teacher.cache import DiskCache, request_key
from anvil.teacher.claude import output_schema, system_prompt
from anvil.teacher.requests import RequestTeacher, decode_answer
from anvil.teacher.transport import JSONDict

NUM_CTX = 4096
"""Context window per request. Ollama silently drops the start of a prompt longer than this,
so it is set well above the longest label prompt (Banking77's 77 labels are about 500
tokens), and a prompt that reaches it is a failure rather than a label of a cut prompt."""

SEED = 0

RETRIES = 2
"""Extra attempts after a transient failure (a 5xx, a timeout, a dropped connection). A
failure that survives them is recorded against the input and replayed as a failure, so one
slow request on a busy CPU must not cost the input for good."""


class OllamaTransport(Protocol):
    """How requests reach an Ollama server (a real one, or a scripted one in tests)."""

    def __call__(self, body: JSONDict) -> JSONDict:
        """`POST /api/chat` with `body`; the parsed response."""
        ...

    def digest(self, model: str) -> str:
        """The sha256 (64 hex characters) of `model`'s weights on the server."""
        ...


def _no_server(url: str, exc: object) -> ConfigError:
    return ConfigError(
        f"no Ollama server at {url} ({exc}); start one with `ollama serve`, or point"
        " ANVIL_OLLAMA_URL at it"
    )


class _TransientError(Exception):
    """A failure worth retrying; becomes a `TeacherError` once the retries are spent."""


class HttpOllamaTransport:
    """Talks to Ollama's HTTP API with the standard library (no SDK needed)."""

    def __init__(self, url: str, timeout_s: float = 600.0, backoff_s: float = 1.0) -> None:
        if not url.startswith(("http://", "https://")):
            raise ConfigError(f"ANVIL_OLLAMA_URL={url!r} must start with http:// or https://")
        self._url = url.rstrip("/")
        self._timeout = timeout_s
        self._backoff_s = backoff_s

    def _request(self, method: str, path: str, body: JSONDict | None = None) -> JSONDict:
        for attempt in range(RETRIES + 1):
            try:
                return self._request_once(method, path, body)
            except _TransientError as exc:
                if attempt == RETRIES:
                    raise TeacherError(str(exc)) from exc
                time.sleep(self._backoff_s * 2**attempt)
        raise AssertionError("unreachable")

    def _request_once(self, method: str, path: str, body: JSONDict | None) -> JSONDict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            f"{self._url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            model = (body or {}).get("model")
            if exc.code == 404:
                raise ConfigError(
                    f"the Ollama server has no model {model!r} ({detail}); run"
                    f" `ollama pull {model}`"
                ) from exc
            if exc.code == 400:
                raise ConfigError(f"Ollama rejected the request: {detail}") from exc
            if exc.code >= 500:
                raise _TransientError(f"Ollama error {exc.code}: {detail}") from exc
            raise TeacherError(f"Ollama error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, ConnectionRefusedError):
                raise _no_server(self._url, exc.reason) from exc
            raise _TransientError(f"could not reach Ollama at {self._url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise _TransientError(f"Ollama did not answer within {self._timeout:.0f} s") from exc
        except (http.client.HTTPException, ConnectionError) as exc:
            raise _TransientError(f"the connection to Ollama broke: {exc!r}") from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TeacherError(f"Ollama sent a response that is not JSON: {raw[:80]!r}") from exc
        if not isinstance(parsed, dict):
            raise TeacherError(f"Ollama sent a response that is not an object: {raw[:80]!r}")
        return parsed

    def __call__(self, body: JSONDict) -> JSONDict:
        return self._request("POST", "/api/chat", body)

    def digest(self, model: str) -> str:
        try:
            tags = self._request("GET", "/api/tags")
        except TeacherError as exc:
            raise _no_server(self._url, exc) from exc
        wanted = {model, f"{model}:latest"} if ":" not in model else {model}
        for entry in tags.get("models") or []:
            if isinstance(entry, dict) and entry.get("name") in wanted:
                value = entry.get("digest")
                if isinstance(value, str):
                    return value.removeprefix("sha256:")
        raise ConfigError(f"the Ollama server has no model {model!r}; run `ollama pull {model}`")


class OllamaTeacher(RequestTeacher):
    """Labels inputs with a local model through Ollama's structured outputs."""

    def __init__(
        self,
        spec: TaskSpec,
        transport: OllamaTransport,
        cache: DiskCache | None = None,
        workers: int = 1,
    ) -> None:
        super().__init__(spec, cache, workers)
        self._transport = transport
        self._system = system_prompt(spec)
        self._schema = output_schema(spec)
        self._digest: str | None = None
        self._lock = threading.Lock()

    @property
    def digest(self) -> str:
        """The weights' digest: the pinned one when the spec has it (checked against the
        server before the first request is sent), else the server's."""
        pinned = self._spec.teacher.digest
        if pinned is not None:
            return pinned
        return self._server_digest()

    def _server_digest(self) -> str:
        with self._lock:
            if self._digest is None:
                self._digest = self._transport.digest(self.model_id)
            return self._digest

    def request_params(self, item: Item) -> JSONDict:
        t = self._spec.teacher
        return {
            "model": t.model,
            "messages": [
                {"role": "system", "content": self._system},
                {"role": "user", "content": item.text},
            ],
            "format": self._schema,
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": SEED,
                "num_ctx": NUM_CTX,
                "num_predict": t.max_tokens,
            },
        }

    def cache_key(self, params: JSONDict) -> str:
        # The tag alone does not name the weights; the digest does.
        return request_key({**params, "digest": self.digest})

    def _send(self, params: JSONDict) -> JSONDict:
        pinned = self._spec.teacher.digest
        if pinned is not None:
            served = self._server_digest()
            if served != pinned:
                raise ConfigError(
                    f"the Ollama server's {self.model_id} has digest {served[:12]}, the spec"
                    f" pins {pinned[:12]}; these are other weights, so their labels would mix"
                    " two teachers. Pull the pinned model or update teacher.digest (and"
                    " re-label)"
                )
        return self._transport(params)

    def _parse(self, item: Item, response: JSONDict, cached: bool) -> Label | Failure:
        def count(key: str) -> int:
            value = response.get(key)
            return value if isinstance(value, int) and not isinstance(value, bool) else 0

        input_tokens, output_tokens = count("prompt_eval_count"), count("eval_count")
        seconds = count("total_duration") / 1e9

        def fail(reason: str) -> Failure:
            return Failure(
                item.id, reason, input_tokens, output_tokens, 0.0, compute_seconds=seconds
            )

        if response.get("error"):
            return fail(f"Ollama error: {str(response['error'])[:120]}")
        if response.get("done_reason") == "length":
            return fail("hit num_predict before answering; raise teacher.max_tokens")
        if input_tokens >= NUM_CTX:
            return fail(f"the prompt filled the {NUM_CTX}-token context, so Ollama cut it")
        message = response.get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str):
            return fail("response has no message content")
        try:
            value = decode_answer(self._spec, text)
        except ValueError as exc:
            return fail(str(exc))
        return Label(
            input_id=item.id,
            label=value,
            teacher_model=self.model_id,
            prompt_hash=self._spec.prompt_hash,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,
            cached=cached,
            compute_seconds=seconds,
        )
