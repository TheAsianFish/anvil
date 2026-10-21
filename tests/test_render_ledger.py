"""The spend ledger in RESULTS.md: one row per committed run, read from each `meta.json`."""

from __future__ import annotations

import json
from pathlib import Path

from eval import render


def _write(root: Path, name: str, meta: dict[str, object], report: dict[str, object]) -> None:
    run = root / name
    run.mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if report:
        (run / "report.json").write_text(json.dumps(report), encoding="utf-8")


def test_ledger_has_a_row_per_run_and_a_total(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "classify-toy-claude",
        {
            "command": "python -m eval.classify toy",
            "git_sha": "abcdef1234567",
            "git_dirty": False,
            "teacher_model": "claude-haiku-5-5",
            "wall_seconds": 3725.0,
            "spent_usd": 1.25,
        },
        {},
    )
    # The M1 spike predates the renderer: other field names, no report.
    _write(
        tmp_path,
        "spike-onnx-export",
        {
            "command": "uv run python spike.py",
            "git_sha": "1234567abcdef",
            "wall_time_s": 310.0,
            "dollars_spent": 0.0,
        },
        {},
    )
    lines = render.render_ledger(tmp_path).splitlines()
    assert lines[2] == (
        "| `classify-toy-claude` | eval-classify | claude-haiku-5-5"
        " | `python -m eval.classify toy` | `abcdef1` | 1h 02m 05s | $1.25 |"
    )
    assert lines[3] == (
        "| `spike-onnx-export` | spike-onnx | n/a | `uv run python spike.py` | `1234567`"
        " | 5m 10s | $0.00 |"
    )
    assert lines[4] == "| **Total (2 runs)** | | | | | 1h 07m 15s | $1.25 |"


def test_ledger_writes_n_a_for_a_field_a_run_did_not_record(tmp_path: Path) -> None:
    _write(tmp_path, "export-toy-gold", {"git_sha": "abcdef1234567"}, {"spent_usd": 0.5})
    _write(
        tmp_path,
        "bench-toy-gold",
        {"command": "python -m eval.bench"},
        {"teacher": {"model": "gold"}},
    )
    lines = render.render_ledger(tmp_path).splitlines()
    assert (
        lines[2] == "| `bench-toy-gold` | bench | gold | `python -m eval.bench` | ? | n/a | n/a |"
    )
    # The report's spend stands in for a meta without one; nothing is guessed for the rest.
    assert lines[3] == "| `export-toy-gold` | eval-export | n/a | n/a | `abcdef1` | n/a | $0.50 |"
    assert lines[4] == "| **Total (2 runs)** | | | | | 0m 00s | $0.50 |"


def test_ledger_skips_smoke_runs_and_handles_an_empty_directory(tmp_path: Path) -> None:
    assert "No runs yet" in render.render_ledger(tmp_path)
    _write(tmp_path, "classify-toy-smoke", {"wall_seconds": 1.0}, {})
    assert "No runs yet" in render.render_ledger(tmp_path)


def test_committed_ledger_is_in_results_md_and_current() -> None:
    text = render.RESULTS_MD.read_text(encoding="utf-8")
    assert "<!-- BEGIN generated:ledger -->" in text
    assert render.render(text, {"ledger": render.render_ledger(render.RESULTS_DIR)}) == text
