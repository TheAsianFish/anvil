"""The bench harness: timing arithmetic, batching, teacher cost and a real served bundle."""

from __future__ import annotations

import _socket
import json
import shutil
from pathlib import Path

import pytest

from anvil.runtime import BUNDLE_META
from eval import bench
from eval.bench import _timings, bench_http, teacher_cost, time_calls
from tests.test_export import exported


def test_timings_are_exact_on_known_seconds() -> None:
    t = _timings([0.1, 0.2, 0.3, 0.4, 0.5], 4)
    assert t["calls"] == 5
    assert t["batch"] == 4
    assert t["p50_ms"] == pytest.approx(300.0)
    assert t["p95_ms"] == pytest.approx(480.0)
    assert t["mean_ms"] == pytest.approx(300.0)
    assert t["inputs_per_s"] == pytest.approx(5 * 4 / 1.5)


def test_timings_of_zero_time_do_not_divide_by_zero() -> None:
    assert _timings([0.0, 0.0], 1)["inputs_per_s"] == 0.0


def test_time_calls_runs_warmup_then_calls_on_cycling_batches() -> None:
    seen: list[list[str]] = []
    texts = ["a", "b", "c"]
    result = time_calls(lambda b: seen.append(b), texts, batch=2, calls=3, warmup=2)
    assert seen == [["a", "b"], ["c", "a"], ["b", "c"], ["a", "b"], ["c", "a"]]
    assert result["calls"] == 3
    assert result["batch"] == 2


def write_labels(root: Path, rows: list[dict[str, object]]) -> None:
    (root / "ds").mkdir(parents=True)
    lines = "\n".join(json.dumps(r) for r in rows) + "\n\n"
    (root / "ds" / "claude-haiku-x.jsonl").write_text(lines, encoding="utf-8")


def test_teacher_cost_averages_recorded_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_labels(
        tmp_path,
        [
            {"label": "a", "cost_usd": 0.001, "input_tokens": 100, "output_tokens": 10},
            {"label": "b", "cost_usd": 0.003, "input_tokens": 300, "output_tokens": 30},
            {"reason": "refusal", "cost_usd": 0.002, "input_tokens": 200, "output_tokens": 20},
        ],
    )
    monkeypatch.setattr(bench, "LABELS_DIR", tmp_path)
    cost = teacher_cost("ds")
    assert cost is not None
    assert cost["labels"] == 2
    assert cost["failures"] == 1  # billed like the others, so it is in the means
    assert cost["usd_per_1k"] == pytest.approx(2.0)
    assert cost["input_tokens_mean"] == pytest.approx(200.0)
    assert cost["output_tokens_mean"] == pytest.approx(20.0)


def test_teacher_cost_is_none_without_claude_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bench, "LABELS_DIR", tmp_path)
    assert teacher_cost("ds") is None
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "gold-x.jsonl").write_text("{}\n", encoding="utf-8")
    assert teacher_cost("ds") is None


def test_bench_http_against_a_served_bundle(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The shared bundle carries a stub task spec; `serve --teacher none` needs none at all.
    bundle = tmp_path / "bundle"
    shutil.copytree(exported(tmp_path_factory).bundle, bundle)
    meta = json.loads((bundle / BUNDLE_META).read_text(encoding="utf-8"))
    meta["spec"] = None
    (bundle / BUNDLE_META).write_text(json.dumps(meta), encoding="utf-8")
    real_connect = _socket.socket.connect

    def loopback_only(self: _socket.socket, address: object) -> None:
        if not (isinstance(address, tuple) and address[0] == "127.0.0.1"):
            raise RuntimeError(f"non-loopback connect in an offline test: {address!r}")
        real_connect(self, address)

    # The autouse offline fixture replaced socket.connect; allow loopback only.
    monkeypatch.setattr("socket.socket.connect", loopback_only)
    texts = ["refund my invoice", "the app crashed", "please add dark mode", "ok"]
    out = bench_http(bundle, "fp32", texts, threads=1, calls=8, warmup=1)
    assert set(out) == {"batch1", "batch32", "concurrent4_batch1"}
    for key in out:
        stats = out[key]
        assert isinstance(stats, dict)
        assert stats["inputs_per_s"] > 0
    concurrent = out["concurrent4_batch1"]
    assert isinstance(concurrent, dict)
    assert concurrent["clients"] == 4
    assert concurrent["calls"] == 16  # 4 clients x calls // 2


def test_render_bench() -> None:
    from eval.render import render_bench

    assert "make bench" in render_bench([])
    stats = {"p50_ms": 1.25, "p95_ms": 2.5, "inputs_per_s": 400.0, "calls": 10}
    report = {
        "dataset": "toy",
        "student_model": "org/tiny",
        "threads": 4,
        "inputs": 50,
        "mean_words": 12.0,
        "engines": {
            "onnx-fp32": {"batch1": stats, "batch32": stats},
            "http-fp32": {"batch1": stats, "batch32": stats, "concurrent4_batch1": stats},
        },
        "cost": {
            "vcpu_hour_usd": 0.05,
            "onnx-fp32": {
                "core_seconds_per_1k": {"batch1": 10.0, "batch32": 12.5},
                "cheapest": "batch1",
                "usd_per_1k": 0.000139,
            },
        },
        "teacher": {"model": "gold", "recorded_cost": None},
        "calibration": {"threshold": 0.9, "target_kappa": 0.9, "rule": "lower", "coverage": 0.8},
    }
    text = render_bench([("bench-toy-gold", report, {"command": "c", "git_sha": "abc1234"})])
    assert (
        "| ONNX Runtime fp32 (in process) | 1.2 / 2.5 | 400 | 1.2 / 2.5 | 400 | 10.0 / 12.5 |"
        in text
    )
    assert "Four concurrent HTTP clients" in text
    assert "under To measure" in text
    assert "calls the teacher for 20.0% of traffic" in text
