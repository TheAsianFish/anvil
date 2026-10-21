"""The Ollama teacher: request shape, accounting, failures, caching, digest pinning, and the
HTTP transport against a loopback server."""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from pydantic import ValidationError

from anvil.errors import ConfigError, TeacherError
from anvil.settings import Settings
from anvil.spans import Mention
from anvil.spans import decode_answer as decode_mentions
from anvil.spec import TaskSpec, TeacherSpec
from anvil.teacher import build_teacher
from anvil.teacher.base import BatchAbortedError, Failure, Item
from anvil.teacher.cache import DiskCache
from anvil.teacher.claude import ClaudeTeacher, output_schema, system_prompt
from anvil.teacher.ollama import NUM_CTX, SEED, HttpOllamaTransport, OllamaTeacher
from tests.conftest import FakeTransport, JSONDict

# Captured at import, before the autouse fixture blocks sockets: the loopback tests restore it.
_REAL_CONNECT = socket.socket.connect

ITEM = Item("t1", "The app crashes on launch.")
DIGEST = "ab" * 32
OTHER_DIGEST = "cd" * 32
MODEL = "qwen2.5:3b"


def chat_response(
    content: str | None = '{"label": "bug"}',
    *,
    done_reason: str = "stop",
    prompt_eval_count: int = 300,
    eval_count: int = 12,
    total_duration: int = 1_500_000_000,
) -> JSONDict:
    """An Ollama /api/chat response body, shaped like a real one."""
    out: JSONDict = {
        "model": MODEL,
        "created_at": "2026-01-01T00:00:00Z",
        "done": True,
        "done_reason": done_reason,
        "total_duration": total_duration,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_cached_count": 0,
        "eval_count": eval_count,
    }
    if content is not None:
        out["message"] = {"role": "assistant", "content": content}
    return out


class ScriptedOllama:
    """Records every request body; answers per user text, else with `default`."""

    def __init__(self, digest: str = DIGEST, default: JSONDict | None = None) -> None:
        self.digest_value = digest
        self.default = default if default is not None else chat_response()
        self.responses: dict[str, JSONDict] = {}
        self.errors: dict[str, Exception] = {}
        self.calls: list[JSONDict] = []
        self.digest_calls: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, body: JSONDict) -> JSONDict:
        with self._lock:
            self.calls.append(body)
        text = body["messages"][1]["content"]
        if text in self.errors:
            raise self.errors[text]
        return self.responses.get(text, self.default)

    def digest(self, model: str) -> str:
        with self._lock:
            self.digest_calls.append(model)
        return self.digest_value


def ollama_spec(base: TaskSpec, digest: str | None = None) -> TaskSpec:
    data = base.model_dump(mode="json")
    data["teacher"] = {"provider": "ollama", "model": MODEL, "max_tokens": 512}
    if digest is not None:
        data["teacher"]["digest"] = digest
    return TaskSpec.model_validate(data)


@pytest.fixture
def spec(toy_spec: TaskSpec) -> TaskSpec:
    return ollama_spec(toy_spec)


@pytest.fixture
def server() -> ScriptedOllama:
    return ScriptedOllama()


def test_request_body_shape(toy_spec: TaskSpec, spec: TaskSpec, server: ScriptedOllama) -> None:
    body = OllamaTeacher(spec, server).request_params(ITEM)
    assert body == {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt(spec)},
            {"role": "user", "content": ITEM.text},
        ],
        "format": output_schema(spec),
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": SEED,
            "num_ctx": NUM_CTX,
            "num_predict": spec.teacher.max_tokens,
        },
    }
    assert body["format"]["properties"]["label"]["enum"] == spec.label_names
    claude_params = ClaudeTeacher(toy_spec, FakeTransport()).request_params(ITEM)
    assert body["format"] == claude_params["output_config"]["format"]["schema"]


def test_valid_answer_becomes_a_free_label(spec: TaskSpec, server: ScriptedOllama) -> None:
    batch = OllamaTeacher(spec, server).label([ITEM])
    assert batch.failures == []
    (lb,) = batch.labels
    assert (lb.input_id, lb.label, lb.teacher_model) == ("t1", "bug", MODEL)
    assert lb.prompt_hash == spec.prompt_hash
    assert lb.cost_usd == 0.0
    assert (lb.input_tokens, lb.output_tokens) == (300, 12)
    assert lb.compute_seconds == pytest.approx(1.5)
    assert not lb.cached
    assert batch.spent_usd == 0.0
    assert batch.compute_seconds == pytest.approx(1.5)
    assert len(server.calls) == 1


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (chat_response('{"label": "spam"}'), "'spam' is not in the label set"),
        (chat_response("Sure, it is a bug."), "not JSON"),
        (chat_response('{"lab', done_reason="length"), "num_predict"),
        (chat_response(prompt_eval_count=NUM_CTX), "context"),
        (chat_response(None), "no message content"),
        ({**chat_response(), "error": "model runner crashed"}, "model runner crashed"),
    ],
)
def test_bad_outputs_become_failures_with_compute_seconds(
    spec: TaskSpec, server: ScriptedOllama, response: JSONDict, reason: str
) -> None:
    server.responses[ITEM.text] = response
    batch = OllamaTeacher(spec, server).label([ITEM, Item("t2", "fine")])
    assert [lb.input_id for lb in batch.labels] == ["t2"]
    (failure,) = batch.failures
    assert failure.input_id == "t1"
    assert reason in failure.reason
    assert failure.cost_usd == 0.0
    assert failure.compute_seconds == pytest.approx(1.5)
    assert batch.compute_seconds == pytest.approx(3.0)
    assert batch.spent_usd == 0.0


def test_cache_makes_reruns_free_and_keeps_seconds(
    spec: TaskSpec, server: ScriptedOllama, tmp_path: Path
) -> None:
    items = [Item(f"c{i}", f"ticket {i}") for i in range(3)]
    cache = DiskCache(tmp_path / "cache")
    first = OllamaTeacher(spec, server, cache=cache).label(items)
    assert len(server.calls) == 3
    second = OllamaTeacher(spec, server, cache=cache).label(items)
    assert len(server.calls) == 3
    assert all(lb.cached for lb in second.labels)
    assert second.cache_hits == 3
    assert [lb.label for lb in second.labels] == [lb.label for lb in first.labels]
    assert [lb.compute_seconds for lb in second.labels] == [1.5] * 3
    assert second.compute_seconds == 0.0
    assert first.compute_seconds == pytest.approx(4.5)


def test_a_different_server_digest_misses_the_cache(
    spec: TaskSpec, server: ScriptedOllama, tmp_path: Path
) -> None:
    cache = DiskCache(tmp_path)
    OllamaTeacher(spec, server, cache=cache).label([ITEM])
    other = ScriptedOllama(digest=OTHER_DIGEST)
    batch = OllamaTeacher(spec, other, cache=cache).label([ITEM])
    assert len(other.calls) == 1
    assert batch.cache_hits == 0
    again = ScriptedOllama(digest=DIGEST)
    assert OllamaTeacher(spec, again, cache=cache).label([ITEM]).cache_hits == 1
    assert again.calls == []


def test_unpinned_digest_is_read_once_per_teacher(spec: TaskSpec, server: ScriptedOllama) -> None:
    teacher = OllamaTeacher(spec, server, cache=None)
    teacher.label([Item(f"i{k}", f"text {k}") for k in range(4)])
    assert server.digest_calls == [MODEL]


def test_pinned_digest_that_matches_works(toy_spec: TaskSpec, server: ScriptedOllama) -> None:
    pinned = ollama_spec(toy_spec, digest=f"sha256:{DIGEST}")
    assert pinned.teacher.digest == DIGEST
    batch = OllamaTeacher(pinned, server).label([ITEM])
    assert [lb.label for lb in batch.labels] == ["bug"]


def test_pinned_digest_mismatch_aborts_the_batch(toy_spec: TaskSpec) -> None:
    pinned = ollama_spec(toy_spec, digest=DIGEST)
    server = ScriptedOllama(digest=OTHER_DIGEST)
    with pytest.raises(BatchAbortedError) as info:
        OllamaTeacher(pinned, server).label([ITEM, Item("t2", "other")])
    assert isinstance(info.value.cause, ConfigError)
    assert DIGEST[:12] in str(info.value.cause)
    assert OTHER_DIGEST[:12] in str(info.value.cause)
    assert info.value.batch.labels == []
    assert server.calls == []


def test_pinned_digest_with_a_full_cache_never_asks_the_server(
    toy_spec: TaskSpec, tmp_path: Path
) -> None:
    pinned = ollama_spec(toy_spec, digest=DIGEST)
    cache = DiskCache(tmp_path)
    OllamaTeacher(pinned, ScriptedOllama(), cache=cache).label([ITEM])
    offline = ScriptedOllama(digest=OTHER_DIGEST)
    batch = OllamaTeacher(pinned, offline, cache=cache).label([ITEM])
    assert batch.cache_hits == 1
    assert offline.digest_calls == []
    assert offline.calls == []


def test_identical_texts_send_one_request(spec: TaskSpec, server: ScriptedOllama) -> None:
    items = [Item("x1", "same"), Item("y", "other"), Item("x2", "same")]
    batch = OllamaTeacher(spec, server).label(items)
    assert len(server.calls) == 2
    assert [(lb.input_id, lb.cached) for lb in batch.labels] == [
        ("x1", False),
        ("y", False),
        ("x2", True),
    ]
    assert batch.compute_seconds == pytest.approx(3.0)


def test_transport_error_fails_only_that_input(spec: TaskSpec, server: ScriptedOllama) -> None:
    server.errors[ITEM.text] = TeacherError("Ollama error 500: boom")
    batch = OllamaTeacher(spec, server).label([Item("a", "first"), ITEM, Item("b", "last")])
    assert [lb.input_id for lb in batch.labels] == ["a", "b"]
    assert batch.failures == [Failure("t1", "Ollama error 500: boom")]


def test_extract_answer_is_decoded_like_claude(toy_spec: TaskSpec) -> None:
    base = TaskSpec.model_validate(
        {
            "name": "slots",
            "kind": "extract",
            "instructions": "Find the slots.",
            "labels": [{"name": "Dish", "description": "a food"}, "Location"],
        }
    )
    spec = ollama_spec(base)
    answer = json.dumps({"entities": [{"text": "pizza", "type": "Dish", "start": 6}]})
    item = Item("1", "cheap pizza downtown")
    local = OllamaTeacher(spec, ScriptedOllama(default=chat_response(answer))).label([item])
    assert decode_mentions(local.labels[0].label) == [Mention("pizza", "Dish", 6)]
    bad = json.dumps({"entities": [{"text": "pizza", "type": "Pasta", "start": 6}]})
    failed = OllamaTeacher(spec, ScriptedOllama(default=chat_response(bad))).label([item])
    assert "unknown type" in failed.failures[0].reason


def test_generate_answer_is_decoded_like_claude() -> None:
    base = TaskSpec.model_validate(
        {
            "name": "sql",
            "kind": "generate",
            "instructions": "Write the query.",
            "outputs": [{"name": "sql", "description": "the query"}, "explanation"],
        }
    )
    spec = ollama_spec(base)
    text = '{"explanation": "all rows", "sql": "SELECT 1"}'
    (lb,) = OllamaTeacher(spec, ScriptedOllama(default=chat_response(text))).label([ITEM]).labels
    assert lb.label == '{"sql":"SELECT 1","explanation":"all rows"}'
    failed = OllamaTeacher(spec, ScriptedOllama(default=chat_response('["x"]'))).label([ITEM])
    assert "not a JSON object" in failed.failures[0].reason


# --- HttpOllamaTransport against a loopback server ---


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def _reply(self) -> None:
        state: _State = self.server.state  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        state.requests.append((self.command, self.path, body))
        status, payload = state.routes[(self.command, self.path)]
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = _reply  # noqa: N815
    do_POST = _reply  # noqa: N815


class _State:
    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], tuple[int, object]] = {}
        self.requests: list[tuple[str, str, bytes]] = []


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[_State, str]]:
    monkeypatch.setattr(socket.socket, "connect", _REAL_CONNECT)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    state = _State()
    httpd.state = state  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_http_chat_round_trip(http: tuple[_State, str]) -> None:
    state, url = http
    state.routes[("POST", "/api/chat")] = (200, chat_response())
    body: JSONDict = {"model": MODEL, "messages": [{"role": "user", "content": "héllo"}]}
    assert HttpOllamaTransport(url + "/")(body) == chat_response()
    ((method, path, raw),) = state.requests
    assert (method, path) == ("POST", "/api/chat")
    assert json.loads(raw) == body


def test_http_404_says_to_pull_the_model(http: tuple[_State, str]) -> None:
    state, url = http
    state.routes[("POST", "/api/chat")] = (404, {"error": "model 'qwen2.5:3b' not found"})
    with pytest.raises(ConfigError, match=r"ollama pull qwen2\.5:3b"):
        HttpOllamaTransport(url)({"model": MODEL})


def test_http_500_is_a_per_input_teacher_error(http: tuple[_State, str]) -> None:
    state, url = http
    state.routes[("POST", "/api/chat")] = (500, {"error": "out of memory"})
    with pytest.raises(TeacherError, match="500") as info:
        HttpOllamaTransport(url)({"model": MODEL})
    assert not isinstance(info.value, ConfigError)


def test_http_non_json_response_is_a_teacher_error(http: tuple[_State, str]) -> None:
    state, url = http
    state.routes[("POST", "/api/chat")] = (200, b"<html>")
    with pytest.raises(TeacherError, match="not JSON"):
        HttpOllamaTransport(url)({"model": MODEL})


@pytest.mark.parametrize(
    ("model", "name", "digest_field"),
    [
        (MODEL, MODEL, f"sha256:{DIGEST}"),
        (MODEL, MODEL, DIGEST),
        ("llama3", "llama3:latest", f"sha256:{DIGEST}"),
        ("llama3:latest", "llama3:latest", DIGEST),
    ],
)
def test_http_digest_reads_tags(
    http: tuple[_State, str], model: str, name: str, digest_field: str
) -> None:
    state, url = http
    state.routes[("GET", "/api/tags")] = (
        200,
        {
            "models": [
                {"name": "other:1b", "digest": OTHER_DIGEST},
                {"name": name, "digest": digest_field},
            ]
        },
    )
    assert HttpOllamaTransport(url).digest(model) == DIGEST


def test_http_digest_of_a_missing_model(http: tuple[_State, str]) -> None:
    state, url = http
    state.routes[("GET", "/api/tags")] = (200, {"models": [{"name": "a:1b", "digest": DIGEST}]})
    with pytest.raises(ConfigError, match=r"ollama pull qwen2\.5:3b"):
        HttpOllamaTransport(url).digest(MODEL)


def test_connection_refused_says_to_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _REAL_CONNECT)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    transport = HttpOllamaTransport(f"http://127.0.0.1:{port}")
    with pytest.raises(ConfigError, match="ollama serve"):
        transport({"model": MODEL})
    with pytest.raises(ConfigError, match="ollama serve"):
        transport.digest(MODEL)


@pytest.mark.parametrize("url", ["127.0.0.1:11434", "ftp://host", "localhost"])
def test_non_http_url_is_a_config_error(url: str) -> None:
    with pytest.raises(ConfigError, match="http"):
        HttpOllamaTransport(url)


# --- build_teacher, TeacherSpec, Settings ---


def test_build_teacher_ollama(spec: TaskSpec, settings: Settings, server: ScriptedOllama) -> None:
    teacher = build_teacher(spec, settings, provider="ollama", ollama=server)
    assert isinstance(teacher, OllamaTeacher)
    assert teacher.model_id == MODEL
    assert build_teacher(spec, settings, ollama=server).model_id == MODEL
    batch = teacher.label([ITEM])
    assert [lb.label for lb in batch.labels] == ["bug"]
    assert teacher.label([ITEM]).cache_hits == 1  # the built teacher caches under ANVIL_HOME


def test_build_teacher_ollama_needs_an_ollama_spec(
    toy_spec: TaskSpec, settings: Settings, server: ScriptedOllama
) -> None:
    with pytest.raises(ConfigError, match="provider: ollama"):
        build_teacher(toy_spec, settings, provider="ollama", ollama=server)


def test_teacher_spec_digest_validation() -> None:
    ok = TeacherSpec.model_validate({"provider": "ollama", "model": MODEL, "digest": DIGEST})
    assert ok.digest == DIGEST
    prefixed = TeacherSpec.model_validate(
        {"provider": "ollama", "model": MODEL, "digest": f"sha256:{DIGEST}"}
    )
    assert prefixed.digest == DIGEST
    assert TeacherSpec.model_validate({"provider": "ollama", "model": MODEL}).digest is None
    for bad in ("abc123", "g" * 64, DIGEST[:-1], DIGEST.upper(), ""):
        with pytest.raises(ValidationError, match="64 hex"):
            TeacherSpec.model_validate({"provider": "ollama", "model": MODEL, "digest": bad})
    with pytest.raises(ValidationError, match="provider: ollama"):
        TeacherSpec.model_validate({"provider": "claude", "model": "x", "digest": DIGEST})


def test_settings_ollama_url() -> None:
    assert Settings.from_env({}).ollama_url == "http://127.0.0.1:11434"
    assert Settings.from_env({"ANVIL_OLLAMA_URL": ""}).ollama_url == "http://127.0.0.1:11434"
    got = Settings.from_env({"ANVIL_OLLAMA_URL": "http://gpu-box:11434/"}).ollama_url
    assert got == "http://gpu-box:11434"
