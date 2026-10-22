"""Configuration from `ANVIL_*` environment variables, read in this one place."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from anvil.errors import ConfigError


def _int(env: dict[str, str], name: str, default: int, minimum: int = 1) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not an integer; unset it or give a number") from exc
    if value < minimum:
        raise ConfigError(f"{name}={value} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    """Runtime settings. The one secret is `hf_token`, kept out of the repr so it never reaches
    a log line; the anthropic SDK resolves its own credentials (`ANTHROPIC_API_KEY`,
    `ANTHROPIC_AUTH_TOKEN` or an `ant auth login` profile)."""

    home: Path
    """Working directory for the store, the teacher cache and trained students."""
    teacher_workers: int
    """Concurrent teacher requests."""
    record_dir: Path | None
    """When set, every real teacher response is also written here as a replayable recording."""
    log_level: str
    serve_threads: int | None = None
    """ONNX Runtime intra-op threads for `anvil serve`; None leaves ONNX Runtime's default."""
    ollama_url: str = "http://127.0.0.1:11434"
    """`ANVIL_OLLAMA_URL`: the Ollama server a `provider: ollama` teacher talks to."""
    hf_token: str | None = field(default=None, repr=False)
    """`HF_TOKEN`: write access to the Hugging Face Hub, for `anvil push`."""

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        e = dict(os.environ) if env is None else env
        record = e.get("ANVIL_RECORD_DIR", "")
        level = e.get("ANVIL_LOG_LEVEL", "INFO").upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ConfigError(f"ANVIL_LOG_LEVEL={level!r}; use DEBUG, INFO, WARNING or ERROR")
        return cls(
            home=Path(e.get("ANVIL_HOME", ".anvil")),
            teacher_workers=_int(e, "ANVIL_TEACHER_WORKERS", 4),
            record_dir=Path(record) if record else None,
            log_level=level,
            serve_threads=_int(e, "ANVIL_SERVE_THREADS", 1)
            if e.get("ANVIL_SERVE_THREADS")
            else None,
            ollama_url=(e.get("ANVIL_OLLAMA_URL") or "http://127.0.0.1:11434").rstrip("/"),
            hf_token=e.get("HF_TOKEN") or None,
        )
