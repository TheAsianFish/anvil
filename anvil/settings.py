"""Configuration from `ANVIL_*` environment variables, read in this one place."""

from __future__ import annotations

import os
from dataclasses import dataclass
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
    """Runtime settings. Nothing here is a secret; the API key stays in the SDK's hands."""

    home: Path
    """Working directory for the store, the teacher cache and trained students."""
    teacher_workers: int
    """Concurrent teacher requests."""
    record_dir: Path | None
    """When set, every real teacher response is also written here as a replayable recording."""
    log_level: str
    has_api_key: bool

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
            has_api_key=bool(e.get("ANTHROPIC_API_KEY") or e.get("ANTHROPIC_AUTH_TOKEN")),
        )
