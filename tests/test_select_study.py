"""`make eval-select` on a tiny configuration: the study runs, writes its report, and renders."""

from __future__ import annotations

import json
import time
from pathlib import Path

from anvil.select import implemented
from eval import render, select_study
from eval.select_study import StudyConfig

TINY = StudyConfig(
    seeds=2,
    pool_size=200,
    test_size=100,
    copies=5,
    initial=10,
    batch=10,
    max_labels=50,
)


def test_tiny_study_writes_a_complete_report_quickly(tmp_path: Path) -> None:
    started = time.perf_counter()
    report = select_study.run_study(TINY)
    select_study.write(report, TINY, tmp_path, "python -m eval.select_study", 1.0)
    assert time.perf_counter() - started < 5

    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert written == json.loads(json.dumps(report))
    assert {"config", "data", "student", "settings", "spent_usd", "wall_seconds"} <= set(written)
    assert written["data"]["data_seed"] == TINY.data_seed
    assert meta["spent_usd"] == 0
    assert meta["seeds"] == [0, 1]
    assert {"git_sha", "git_dirty", "cpus", "versions", "wall_seconds"} <= set(meta)
    assert set(meta["versions"]) == {"numpy", "scikit-learn"}

    assert set(written["settings"]) == {"blobs", "blobs-duplicated"}
    duplicated = written["settings"]["blobs-duplicated"]
    assert duplicated["pool_size"] == 200
    assert duplicated["distinct_pool_points"] == 40
    for setting in written["settings"].values():
        assert 0 < setting["full_pool_accuracy"] <= 1
        assert set(setting["strategies"]) == set(TINY.strategies)
        for strategy, result in setting["strategies"].items():
            if not implemented(strategy):
                assert result == {"status": "not implemented"}
                continue
            assert result["status"] == "ok"
            assert result["label_counts"] == [10, 20, 30, 40, 50]
            assert set(result["curves"]) == {"0", "1"}
            assert all(len(c) == 5 for c in result["curves"].values())
            assert len(result["accuracy"]["mean"]) == 5
            for target in result["labels_to_target"].values():
                assert set(target) == {"target", "per_seed", "mean", "min", "max", "never"}


def test_same_seed_gives_the_same_curves() -> None:
    data = select_study.make_data(TINY)["blobs"]
    first = select_study.curve(TINY, data, "random", seed=3)
    assert first == select_study.curve(TINY, data, "random", seed=3)
    assert select_study.initial_set(TINY, 200, 3) == select_study.initial_set(TINY, 200, 3)
    assert select_study.initial_set(TINY, 200, 3) != select_study.initial_set(TINY, 200, 4)


def test_labels_to_target_is_the_first_count_reaching_it() -> None:
    counts = [10, 20, 30, 40]
    assert select_study.labels_to(counts, [0.5, 0.95, 0.9, 0.97], 0.95) == 20
    assert select_study.labels_to(counts, [0.5, 0.6, 0.7, 0.8], 0.95) is None


def test_student_covers_classes_it_has_not_seen() -> None:
    data = select_study.make_data(TINY)["blobs"]
    one_class = data.pool_y == data.pool_y[0]
    student = select_study.Student(TINY.classes).fit(
        data.pool_x[one_class][:3], data.pool_y[one_class][:3]
    )
    probs = student.predict_proba(data.test_x[:4])
    assert probs.shape == (4, TINY.classes)
    assert (probs[:, data.pool_y[0]] == 1).all()
    some = data.pool_y < 3
    partial = select_study.Student(TINY.classes).fit(data.pool_x[some], data.pool_y[some])
    out = partial.predict_proba(data.test_x[:4])
    assert out.shape == (4, TINY.classes)
    assert (out[:, 3:] == 0).all()
    assert abs(out.sum(axis=1) - 1).max() < 1e-9


def _report(strategies: dict[str, object]) -> dict[str, object]:
    ok = {
        "status": "ok",
        "label_counts": [20, 50, 100],
        "accuracy": {"mean": [0.8, 0.93, 0.962], "min": [0, 0, 0], "max": [1, 1, 1]},
        "labels_to_target": {
            "accuracy": {"target": 0.95, "mean": 98.5, "min": 60, "max": 150, "never": 1},
            "relative": {"target": 0.924, "mean": 52.0, "min": 40, "max": 70, "never": 0},
        },
    }
    return {
        "config": {"max_labels": 400, "seeds": 20},
        "data": {"data_seed": 0},
        "settings": {
            "blobs": {
                "pool_size": 2000,
                "distinct_pool_points": 2000,
                "test_size": 1000,
                "full_pool_accuracy": 0.973,
                "targets": {"accuracy": 0.95, "relative": 0.9244},
                "strategies": {name: ok if s == "ok" else s for name, s in strategies.items()},
            }
        },
        "spent_usd": 0.0,
    }


def test_renderer_shows_measured_and_unwritten_strategies() -> None:
    report = _report({"random": "ok", "diverse": {"status": "not implemented"}})
    meta = {"command": "python -m eval.select_study", "git_sha": "abcdef123", "cpus": 4}
    block = render.render_select((report, meta))
    assert "**blobs**" in block
    assert "0.973" in block
    random_row = next(line for line in block.splitlines() if line.startswith("| random |"))
    assert "| 98.5 [60-150] | 1 of 20 |" in random_row
    assert "| 0.930 | 0.962 | n/a | n/a |" in random_row
    diverse_row = next(line for line in block.splitlines() if line.startswith("| diverse |"))
    assert "not implemented yet" in diverse_row
    assert "make eval-select" in diverse_row
    header = next(line for line in block.splitlines() if line.startswith("| Strategy"))
    assert header.count("|") == random_row.count("|") == diverse_row.count("|")
    assert "No selection study yet" in render.render_select(None)


def test_render_replaces_the_select_block() -> None:
    text = "before\n<!-- BEGIN generated:select -->\nold\n<!-- END generated:select -->\nafter\n"
    out = render.render(text, {"select": "new\n"})
    assert out == (
        "before\n<!-- BEGIN generated:select -->\nnew\n<!-- END generated:select -->\nafter\n"
    )


def test_select_block_is_built_from_the_results_directory(tmp_path: Path) -> None:
    assert "No selection study yet" in render.blocks_for(tmp_path)["select"]
    out = tmp_path / render.SELECT_DIR
    select_study.write(_report({"random": "ok"}), TINY, out, "cmd", 1.0)
    assert "| random | 98.5 [60-150]" in render.blocks_for(tmp_path)["select"]
