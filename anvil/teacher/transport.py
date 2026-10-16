"""How a teacher request reaches a model: the real API, a recorder, or a replay of recordings.

A transport takes Messages API request parameters and returns the response as a plain dict
(`Message.to_dict()`), so the same parsing code runs on live responses and on recordings.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol

from anvil.errors import ConfigError, TeacherError

# Any: request and response bodies are arbitrary JSON as defined by the Messages API.
JSONDict = dict[str, Any]


class Transport(Protocol):
    def __call__(self, params: JSONDict) -> JSONDict: ...


class AnthropicTransport:
    """Calls the Messages API through the official SDK (retries 429/5xx twice by default)."""

    def __init__(self, timeout_s: float = 60.0, max_retries: int = 3) -> None:
        import anthropic

        self._client = anthropic.Anthropic(timeout=timeout_s, max_retries=max_retries)

    def __call__(self, params: JSONDict) -> JSONDict:
        import anthropic

        try:
            message = self._client.messages.create(**params)
        except anthropic.AuthenticationError as exc:
            raise ConfigError(
                "the Anthropic API rejected the key; check ANTHROPIC_API_KEY"
            ) from exc
        except anthropic.PermissionDeniedError as exc:
            raise ConfigError(f"the API key cannot use {params.get('model')}: {exc}") from exc
        except anthropic.NotFoundError as exc:
            raise ConfigError(f"unknown model {params.get('model')!r}; fix teacher.model") from exc
        except anthropic.BadRequestError as exc:
            # A 400 is the same for every input (bad model settings, schema, effort), so it
            # stops the run instead of failing each input and suggesting a retry.
            raise ConfigError(
                f"the API rejected the request; check the teacher spec: {exc}"
            ) from exc
        except anthropic.APIStatusError as exc:
            raise TeacherError(
                f"API error {exc.status_code} after retries (request id {exc.request_id})"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise TeacherError(f"could not reach the Anthropic API: {exc}") from exc
        except TypeError as exc:
            if "authentication" in str(exc):
                raise ConfigError(
                    "no Anthropic credentials; set ANTHROPIC_API_KEY or use --teacher fake"
                ) from exc
            raise
        result: JSONDict = message.to_dict()
        return result


class RecordingTransport:
    """Wraps a transport and writes each request/response pair as a replayable recording."""

    def __init__(self, inner: Transport, directory: Path) -> None:
        self._inner = inner
        self._dir = directory

    def __call__(self, params: JSONDict) -> JSONDict:
        from anvil.teacher.cache import request_key

        response = self._inner(params)
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{request_key(params)[:16]}.json"
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"request": params, "response": response}, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
        return response


class ReplayTransport:
    """Serves responses from recordings; an unrecorded request is an error, never a guess."""

    def __init__(self, directory: Path) -> None:
        from anvil.teacher.cache import request_key

        self._responses: dict[str, JSONDict] = {}
        for path in sorted(directory.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            self._responses[request_key(data["request"])] = data["response"]
        if not self._responses:
            raise ConfigError(f"no recordings in {directory}")

    def __len__(self) -> int:
        return len(self._responses)

    def __call__(self, params: JSONDict) -> JSONDict:
        from anvil.teacher.cache import request_key

        key = request_key(params)
        try:
            return self._responses[key]
        except KeyError:
            raise TeacherError(
                f"no recording for request {key[:16]}; the prompt or inputs changed since"
                " recording, re-record with `make record` (needs ANTHROPIC_API_KEY)"
            ) from None
