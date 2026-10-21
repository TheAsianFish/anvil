"""The Colab notebook is valid nbformat 4, committed without outputs, and calls real modules."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "anvil_gpu.ipynb"


def _notebook() -> dict[str, Any]:  # Any: free-form notebook JSON
    data: dict[str, Any] = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return data


def _source(cell: dict[str, Any]) -> str:  # Any: free-form notebook JSON
    source = cell["source"]
    return source if isinstance(source, str) else "".join(source)


def test_notebook_is_nbformat_4() -> None:
    nb = _notebook()
    assert nb["nbformat"] == 4
    assert isinstance(nb["metadata"], dict)
    assert nb["cells"]
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], str | list)


def test_notebook_is_committed_clean() -> None:
    # Outputs would commit one machine's run (and possibly a printed secret) into the repo.
    for cell in _notebook()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_notebook_runs_modules_that_exist() -> None:
    modules = {
        m
        for cell in _notebook()["cells"]
        for m in re.findall(r"python -m eval\.(\w+)", _source(cell))
    }
    assert "generate" in modules
    for module in modules:
        assert (ROOT / "eval" / f"{module}.py").is_file(), f"eval/{module}.py is missing"
