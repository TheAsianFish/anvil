from __future__ import annotations

from pathlib import Path

import pytest

from anvil.errors import ConfigError
from anvil.settings import Settings


def test_defaults() -> None:
    s = Settings.from_env({})
    assert s == Settings(
        home=Path(".anvil"),
        teacher_workers=4,
        record_dir=None,
        log_level="INFO",
    )


def test_env_parsing() -> None:
    s = Settings.from_env(
        {
            "ANVIL_HOME": "/tmp/h",
            "ANVIL_TEACHER_WORKERS": "9",
            "ANVIL_RECORD_DIR": "/tmp/rec",
            "ANVIL_LOG_LEVEL": "debug",
        }
    )
    assert s.home == Path("/tmp/h")
    assert s.teacher_workers == 9
    assert s.record_dir == Path("/tmp/rec")
    assert s.log_level == "DEBUG"
    assert Settings.from_env({"ANVIL_TEACHER_WORKERS": ""}).teacher_workers == 4


def test_serve_threads() -> None:
    assert Settings.from_env({}).serve_threads is None
    assert Settings.from_env({"ANVIL_SERVE_THREADS": ""}).serve_threads is None
    assert Settings.from_env({"ANVIL_SERVE_THREADS": "2"}).serve_threads == 2
    for bad in ("two", "0"):
        with pytest.raises(ConfigError, match="ANVIL_SERVE_THREADS"):
            Settings.from_env({"ANVIL_SERVE_THREADS": bad})


@pytest.mark.parametrize("value", ["four", "0"])
def test_bad_worker_count(value: str) -> None:
    with pytest.raises(ConfigError, match="ANVIL_TEACHER_WORKERS"):
        Settings.from_env({"ANVIL_TEACHER_WORKERS": value})


def test_bad_log_level() -> None:
    with pytest.raises(ConfigError, match="ANVIL_LOG_LEVEL"):
        Settings.from_env({"ANVIL_LOG_LEVEL": "LOUD"})


def test_reads_process_environment_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANVIL_TEACHER_WORKERS", "3")
    assert Settings.from_env().teacher_workers == 3


def test_hf_token_is_read_and_kept_out_of_the_repr() -> None:
    assert Settings.from_env({}).hf_token is None
    assert Settings.from_env({"HF_TOKEN": ""}).hf_token is None
    s = Settings.from_env({"HF_TOKEN": "hf_secret123"})
    assert s.hf_token == "hf_secret123"
    assert "hf_secret123" not in repr(s)
