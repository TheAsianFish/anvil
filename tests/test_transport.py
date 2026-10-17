from __future__ import annotations

import json
from pathlib import Path

import pytest

from anvil.errors import ConfigError, TeacherError
from anvil.settings import Settings
from anvil.spec import TaskSpec
from anvil.teacher import ClaudeTeacher, FakeTeacher, Item, build_teacher
from anvil.teacher.cache import DiskCache, request_key
from anvil.teacher.transport import RecordingTransport, ReplayTransport
from tests.conftest import FakeTransport, message

PARAMS = {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}]}


def test_recording_then_replay(tmp_path: Path, fake_transport: FakeTransport) -> None:
    recorder = RecordingTransport(fake_transport, tmp_path / "rec")
    assert recorder(PARAMS) == message()
    (path,) = (tmp_path / "rec").glob("*.json")
    assert path.name == f"{request_key(PARAMS)[:16]}.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "request": PARAMS,
        "response": message(),
    }
    assert not list((tmp_path / "rec").glob("*.tmp"))
    replay = ReplayTransport(tmp_path / "rec")
    assert len(replay) == 1
    assert replay(PARAMS) == message()


def test_unrecorded_request(tmp_path: Path, fake_transport: FakeTransport) -> None:
    RecordingTransport(fake_transport, tmp_path)(PARAMS)
    other = {**PARAMS, "max_tokens": 5}
    with pytest.raises(TeacherError, match="re-record"):
        ReplayTransport(tmp_path)(other)


def test_empty_recordings_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no recordings"):
        ReplayTransport(tmp_path)


@pytest.mark.parametrize("content", ['{"id": "msg", "conte', "[1, 2]", ""])
def test_invalid_cache_file_is_a_miss(tmp_path: Path, content: str) -> None:
    cache = DiskCache(tmp_path)
    key = request_key(PARAMS)
    cache.put(key, {"ok": True})
    assert cache.get(key) == {"ok": True}
    (tmp_path / key[:2] / f"{key}.json").write_text(content, encoding="utf-8")
    assert cache.get(key) is None
    assert cache.get("0" * 64) is None


def test_recording_identical_requests_from_threads(
    tmp_path: Path, fake_transport: FakeTransport
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    recorder = RecordingTransport(fake_transport, tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(recorder, [PARAMS] * 32))
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_request_key_ignores_dict_key_order() -> None:
    reordered = {"messages": [{"content": "hi", "role": "user"}], "model": "claude-opus-5-5"}
    assert request_key(reordered) == request_key(PARAMS)
    assert request_key({**PARAMS, "model": "claude-haiku-5-5"}) != request_key(PARAMS)


def test_build_fake_teacher(toy_spec: TaskSpec, settings: Settings) -> None:
    assert isinstance(build_teacher(toy_spec, settings, provider="fake"), FakeTeacher)
    fake_spec = toy_spec.model_copy(
        update={"teacher": toy_spec.teacher.model_copy(update={"provider": "fake"})}
    )
    assert isinstance(build_teacher(fake_spec, settings), FakeTeacher)


def test_build_claude_without_credentials(toy_spec: TaskSpec, settings: Settings) -> None:
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY") as info:
        build_teacher(toy_spec, settings)
    assert "--teacher fake" in str(info.value)


def test_build_claude_with_profile_credentials(
    toy_spec: TaskSpec, settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An `ant auth login` profile counts as credentials: the SDK resolves it, not Anvil."""
    config = tmp_path / "anthropic-config"
    (config / "configs").mkdir(parents=True)
    (config / "credentials").mkdir()
    profile = config / "configs" / "default.json"
    profile.write_text(json.dumps({"authentication": {"type": "user_oauth"}}), encoding="utf-8")
    token = config / "credentials" / "default.json"
    token.write_text(
        json.dumps({"access_token": "test-token", "expires_at": 4102444800}), encoding="utf-8"
    )
    token.chmod(0o600)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(config))
    assert isinstance(build_teacher(toy_spec, settings), ClaudeTeacher)


def test_build_claude_with_explicit_transport(
    toy_spec: TaskSpec, settings: Settings, fake_transport: FakeTransport
) -> None:
    teacher = build_teacher(toy_spec, settings, transport=fake_transport)
    assert isinstance(teacher, ClaudeTeacher)
    batch = teacher.label([Item("a", "crash")])
    assert [lb.label for lb in batch.labels] == ["bug"]
    assert list((settings.home / "cache" / "teacher").rglob("*.json"))


def _sdk_error(name: str, status: int, text: str = "boom") -> Exception:
    import anthropic
    import httpx2 as httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    if name == "APIConnectionError":
        return anthropic.APIConnectionError(request=request)
    if name == "TypeError":
        return TypeError("Could not resolve authentication method. Expected one of api_key ...")
    cls = getattr(anthropic, name)
    response = httpx.Response(status, request=request, json={"error": {"message": text}})
    return cls(text, response=response, body={"error": {"message": text}})  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    ("name", "status", "text", "error", "fragment"),
    [
        ("AuthenticationError", 401, "bad key", ConfigError, "ANTHROPIC_API_KEY"),
        ("PermissionDeniedError", 403, "no access", ConfigError, "cannot use claude-opus-5-5"),
        ("NotFoundError", 404, "no model", ConfigError, "unknown model"),
        ("BadRequestError", 400, "bad effort", ConfigError, "check the teacher spec"),
        (
            "BadRequestError",
            400,
            "prompt is too long: 213456 tokens > 200000 maximum",
            TeacherError,
            "too long",
        ),
        ("InternalServerError", 500, "down", TeacherError, "API error 500"),
        ("RateLimitError", 429, "slow down", TeacherError, "API error 429"),
        ("APIConnectionError", 0, "", TeacherError, "could not reach"),
        ("TypeError", 0, "", ConfigError, "--teacher fake"),
    ],
)
def test_sdk_errors_map_to_anvil_errors(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    status: int,
    text: str,
    error: type[Exception],
    fragment: str,
) -> None:
    from anvil.teacher.transport import AnthropicTransport

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    transport = AnthropicTransport()
    exc = _sdk_error(name, status, text)

    def reject(**kwargs: object) -> None:
        raise exc

    monkeypatch.setattr(transport._client.messages, "create", reject)
    with pytest.raises(error, match=fragment):
        transport(PARAMS)
