"""Content-addressed disk cache for teacher responses.

The key is a hash of the exact request parameters (model, system prompt, input, output schema,
effort and max_tokens), so re-running with the same prompt, model and inputs costs nothing, and
any change to one of them misses the cache instead of returning a stale answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from anvil.teacher.transport import JSONDict


def request_key(params: JSONDict) -> str:
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> JSONDict | None:
        path = self._path(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, UnicodeDecodeError):
            # A torn write from a killed process; treat as a miss and overwrite later.
            return None
        return data if isinstance(data, dict) else None

    def put(self, key: str, value: JSONDict) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(value, fh, ensure_ascii=False)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
