"""The `make eval-frontier` harness offline: toy tickets stand in for a dataset, the gold labels
are the teacher and the student is the tiny random-init encoder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from anvil import loop as loop_module
from anvil.errors import ConfigError
from anvil.settings import Settings
from anvil.spec import StudentSpec
from anvil.store import read_inputs
from anvil.teacher.base import Item, Label
from anvil.teacher.fake import FakeTeacher
from anvil.teacher.labelfile import (
    NOT_RECORDED,
    LabelFileTeacher,
    RecordedLabel,
    write_label_file,
)
from eval import classify, frontier, render
from eval.classify import EvalConfig
from eval.data import Sample, sample
from eval.frontier import (
    CompleteRecording,
    FrontierConfig,
    load_frontier_config,
    resolve_kind,
    run_dataset,
    summarise,
)
from tests.conftest import TOY_INPUTS, TOY_SPEC

PROVENANCE: dict[str, object] = {"git_sha": "abc1234def", "git_dirty": False}


def _toy() -> tuple[EvalConfig, Sample]:
    raw_spec = yaml.safe_load(TOY_SPEC.read_text(encoding="utf-8"))
    raw_spec["teacher"]["model"] = "claude-haiku-5-5"
    raw = {
        "dataset": {"name": "toy", "repo": "local/toy", "revision": "0" * 40},
        "sample": {"size": 150, "pool_labels": 110, "heldout_labels": 40, "seed": 0},
        "ceiling": {"n": 0, "instructions": "Pick the support queue for this ticket."},
        "spec": raw_spec,
    }
    config = EvalConfig.model_validate(raw)
    fake = config.spec.fake
    assert fake is not None
    rules = FakeTeacher(
        config.spec.model_copy(update={"fake": fake.model_copy(update={"noise": 0.0})})
    )
    texts = [it.text for it in read_inputs(TOY_INPUTS)]
    return config, sample("toy", texts, [rules.rule_label(t) for t in texts], n=150, seed=0)


def _frontier(**update: object) -> FrontierConfig:
    base: dict[str, object] = {
        "datasets": ["toy"],
        "strategies": ["random", "uncertainty"],
        "seeds": [0, 1],
        "budgets": [20, 40],
        "targets": [0.2, 0.99],
        "heldout_labels": 30,
        "student": StudentSpec(
            model="tiny", max_length=24, epochs=2, batch_size=8, learning_rate=2e-3
        ),
        "n_resamples": 50,
    }
    return FrontierConfig.model_validate({**base, **update})


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        home=tmp_path / "home", teacher_workers=1, record_dir=None, log_level="WARNING"
    )


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_dataset_end_to_end_with_gold_and_resumes_finished_loops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, data = _toy()
    fc = _frontier()
    out = tmp_path / "results" / "frontier-toy-gold"
    settings = _settings(tmp_path)
    report = run_dataset("toy", fc, data, config, "gold", out, settings, "cmd", PROVENANCE)

    assert report["status"] == {"random": "measured", "uncertainty": "not implemented"}
    rows = _jsonl(out / "frontier.jsonl")
    keys = [(r["strategy"], r["seed"], r["round"]) for r in rows]
    assert keys == [("random", 0, 0), ("random", 0, 1), ("random", 1, 0), ("random", 1, 1)]
    assert [r["picked"] for r in rows] == [20, 40, 20, 40]
    assert all(r["teacher_model"] == "gold" and r["student_model"] == "tiny" for r in rows)
    assert all(r["heldout"] == 30 for r in rows)

    on_disk = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(report))
    assert on_disk["teacher"]["kind"] == "gold"
    assert on_disk["teacher"]["model"] == "gold"
    assert on_disk["dataset"]["sampled"] == 150
    assert on_disk["dataset"]["pool"] + on_disk["dataset"]["heldout_inputs"] == 150
    assert set(on_disk["strategies"]) == {"random"}
    random = on_disk["strategies"]["random"]
    assert [p["budget"] for p in random["points"]] == [20, 40]
    for p in random["points"]:
        assert p["seeds"] == 2
        assert p["kappa_min"] <= p["kappa_mean"] <= p["kappa_max"]
        assert p["kappa_low_min"] <= p["kappa_min"]
        assert p["labels_mean"] == p["budget"]  # gold labels every pick
        assert p["label_cost_usd_mean"] == 0
    assert set(random["labels_to_target"]) == {"0.2", "0.99"}
    for reach in random["labels_to_target"].values():
        assert set(reach) == {"mean_kappa", "every_seed_ci"}
    assert on_disk["spent_usd"] == 0

    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["command"] == "cmd"
    assert meta["git_sha"] == "abc1234def"
    assert meta["seeds"] == [0, 1]
    assert meta["spent_usd"] == 0
    assert meta["teacher_model"] == "gold"
    assert meta["student_model"] == "tiny"

    # The report has the shape the renderer reads.
    table = render.render_frontier([("frontier-toy-gold", on_disk, meta)])
    assert "| random | " in table
    assert "| uncertainty | not implemented yet | not implemented yet |" in table

    # A second run finds every loop finished: nothing is trained again, the rows are the same.
    def no_training(*args: object, **kwargs: object) -> object:
        raise AssertionError("a finished loop was trained again")

    monkeypatch.setattr(loop_module, "fit_student", no_training)
    again = run_dataset("toy", fc, data, config, "gold", out, settings, "cmd", PROVENANCE)
    assert _jsonl(out / "frontier.jsonl") == rows
    assert again["strategies"] == report["strategies"]


def test_other_loop_settings_in_the_same_workspace_start_new_loops(tmp_path: Path) -> None:
    """A smoke run with other budgets (or another teacher) shares the workspace of the full
    run, since the sample is the same; its loops must not collide with the full run's."""
    config, data = _toy()
    settings = _settings(tmp_path)
    small = _frontier(strategies=["random"], seeds=[0], budgets=[20])
    first = run_dataset(
        "toy", small, data, config, "gold", tmp_path / "a", settings, "", PROVENANCE
    )
    other = _frontier(strategies=["random"], seeds=[0], budgets=[24])
    second = run_dataset(
        "toy", other, data, config, "gold", tmp_path / "b", settings, "", PROVENANCE
    )
    assert [r["picked"] for r in _jsonl(tmp_path / "a" / "frontier.jsonl")] == [20]
    assert [r["picked"] for r in _jsonl(tmp_path / "b" / "frontier.jsonl")] == [24]
    assert first["status"] == second["status"] == {"random": "measured"}


def _row(strategy: str, seed: int, budget: int, kappa: float, low: float) -> dict[str, object]:
    return {
        "strategy": strategy,
        "seed": seed,
        "budget": budget,
        "labels": budget,
        "kappa": kappa,
        "kappa_low": low,
        "label_cost_usd": budget * 0.001,
        "classes_covered": 4,
        "train_seconds": 2.0,
    }


def test_summarise_reads_targets_by_mean_and_by_every_seed_interval() -> None:
    fc = _frontier(budgets=[100, 200, 400], targets=[0.8, 0.95])
    rows = [
        _row("random", 0, 100, 0.85, 0.80),
        _row("random", 1, 100, 0.77, 0.70),
        _row("random", 0, 200, 0.90, 0.82),
        _row("random", 1, 200, 0.86, 0.81),
        # Seed 0 stopped early (say the pool ran out): 400 has one seed only.
        _row("random", 1, 400, 0.99, 0.97),
    ]
    out = summarise(rows, fc)
    assert set(out) == {"random"}  # a strategy without rows is left out
    points = out["random"]["points"]
    assert isinstance(points, list)
    first = points[0]
    assert first["budget"] == 100
    assert first["seeds"] == 2
    assert first["kappa_mean"] == pytest.approx(0.81)
    assert (first["kappa_min"], first["kappa_max"], first["kappa_low_min"]) == (0.77, 0.85, 0.70)
    assert first["label_cost_usd_mean"] == pytest.approx(0.1)
    assert [p["seeds"] for p in points] == [2, 2, 1]
    # 0.8: the mean reaches it at 100, every seed's interval only at 200. 0.95 is reached
    # at 400 by the one seed that got there, which is not a result over the seeds.
    assert out["random"]["labels_to_target"] == {
        "0.8": {"mean_kappa": 100, "every_seed_ci": 200},
        "0.95": {"mean_kappa": None, "every_seed_ci": None},
    }


def _recorded() -> LabelFileTeacher:
    return LabelFileTeacher(
        [
            RecordedLabel("a", "bug", "m", "p", cost_usd=0.01),
            RecordedLabel("b", None, "m", "p", cost_usd=0.02, reason="refusal (cyber)"),
        ],
        "m",
        "p",
    )


def test_complete_recording_passes_refusals_and_stops_on_holes(tmp_path: Path) -> None:
    teacher = CompleteRecording(_recorded(), tmp_path / "labels.jsonl")
    assert teacher.model_id == "m"
    batch = teacher.label([Item("a", "x"), Item("b", "y")])
    assert [lb.input_id for lb in batch.labels] == ["a"]
    assert [(f.input_id, f.reason) for f in batch.failures] == [("b", "refusal (cyber)")]
    assert NOT_RECORDED != "refusal (cyber)"
    with pytest.raises(ConfigError, match=r"1 inputs have no recorded label.*incomplete"):
        teacher.label([Item("a", "x"), Item("c", "z")])


def test_resolve_kind_prefers_recorded_labels_then_a_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _toy()
    monkeypatch.setattr(classify, "LABELS_DIR", tmp_path / "labels")
    assert resolve_kind("gold", config) == "gold"
    with pytest.raises(ConfigError, match="--teacher gold"):
        resolve_kind("auto", config)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert resolve_kind("auto", config) == "claude"
    spec = config.spec
    path = classify.label_file("toy", spec.teacher.model, spec.prompt_hash)
    write_label_file(
        path, [Label("toy-1", "bug", spec.teacher.model, spec.prompt_hash, 1, 1, 0.0)]
    )
    assert resolve_kind("auto", config) == "replay"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert resolve_kind("auto", config) == "replay"


def test_committed_frontier_config_and_unknown_strategies(tmp_path: Path) -> None:
    fc = load_frontier_config()
    assert len(fc.budgets) == 5
    assert len(fc.seeds) == 3
    assert set(fc.strategies) == {"random", "uncertainty", "margin", "diverse"}
    assert fc.budgets == sorted(fc.budgets)
    for name in fc.datasets:
        assert (classify.CONFIG_DIR / f"{name}.yaml").exists()

    raw = yaml.safe_load(frontier.FRONTIER_CONFIG.read_text(encoding="utf-8"))
    raw["strategies"] = ["random", "bald"]
    bad = tmp_path / "frontier.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="bald"):
        load_frontier_config(bad)
    with pytest.raises(ConfigError, match="not found"):
        load_frontier_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    ("flags", "suffix"),
    [([], ""), (["--seeds", "0"], "-smoke"), (["--budgets", "20,40"], "-smoke"),
     (["--strategies", "random"], "-smoke"), (["--student", "tiny", "--epochs", "1"], "-smoke"),
     (["--size", "100"], "-smoke"), (["--resamples", "10"], "-smoke")],
)  # fmt: skip
def test_main_sends_override_runs_to_a_smoke_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
    suffix: str,
) -> None:
    config, data = _toy()
    monkeypatch.setattr(frontier, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(frontier, "load_config", lambda path: config)
    monkeypatch.setattr(frontier, "load_rows", lambda ref: ([], []))
    monkeypatch.setattr(frontier, "sample", lambda *args, **kwargs: data)
    monkeypatch.setattr(frontier, "git_provenance", lambda: PROVENANCE)
    seen: list[tuple[FrontierConfig, EvalConfig, str, Path]] = []

    def fake_run(
        name: str,
        fc: FrontierConfig,
        data: Sample,
        config: EvalConfig,
        kind: str,
        out: Path,
        *rest: object,
    ) -> dict[str, object]:
        seen.append((fc, config, kind, out))
        point = {"budget": 1600, "kappa_mean": 0.9}
        return {"dataset": {"name": name}, "strategies": {"random": {"points": [point]}}}

    monkeypatch.setattr(frontier, "run_dataset", fake_run)
    assert frontier.main(["toy", "--teacher", "gold", *flags]) == 0
    fc, used, kind, out = seen[0]
    assert len(seen) == 1
    assert kind == "gold"
    assert out == tmp_path / "results" / f"frontier-toy-gold{suffix}"
    printed = capsys.readouterr().out.splitlines()
    assert printed == [f"toy: random 0.900 at 1600; raw output {out}"]
    committed = load_frontier_config()
    assert fc.seeds == ([0] if "--seeds" in flags else committed.seeds)
    assert fc.budgets == ([20, 40] if "--budgets" in flags else committed.budgets)
    assert fc.strategies == (["random"] if "--strategies" in flags else committed.strategies)
    assert fc.n_resamples == (10 if "--resamples" in flags else committed.n_resamples)
    if "--student" in flags:
        assert (fc.student.model, fc.student.epochs) == ("tiny", 1)
        assert fc.student.max_length == committed.student.max_length
    else:
        assert fc.student == committed.student
    if "--size" in flags:
        sizes = (used.sample.size, used.sample.pool_labels, used.sample.heldout_labels)
        assert sizes == (100, 73, 27)


def test_main_refuses_out_without_overrides(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        frontier.main(["ag_news", "--out", "/tmp/frontier-x"])
    assert exc.value.code == 2
    assert "--out is for a smoke run of one dataset" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        frontier.main(["ag_news", "banking77", "--seeds", "0", "--out", "/tmp/frontier-x"])


def _report(kind: str = "gold") -> dict[str, object]:
    def point(budget: int, mean: float) -> dict[str, object]:
        return {
            "budget": budget,
            "seeds": 3,
            "labels_mean": budget,
            "kappa_mean": mean,
            "kappa_min": mean - 0.01,
            "kappa_max": mean + 0.02,
            "kappa_low_min": mean - 0.05,
            "label_cost_usd_mean": budget * 0.0005,
            "classes_covered_mean": 4.0,
            "train_seconds_mean": 12.0,
        }

    return {
        "dataset": {
            "name": "ag_news",
            "repo": "fancyzhx/ag_news",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4",
            "sampled": 6000,
            "sample_sha256": "f" * 64,
            "pool": 4500,
            "heldout_inputs": 477,
        },
        "teacher": {"kind": kind, "model": "gold" if kind == "gold" else "claude-haiku-5-5"},
        "student": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        "settings": {
            "seeds": [0, 1, 2],
            "budgets": [100, 200],
            "targets": [0.8, 0.9],
            "heldout_labels": 500,
        },
        "status": {"random": "measured", "uncertainty": "not implemented"},
        "strategies": {
            "random": {
                "points": [point(100, 0.70), point(200, 0.85)],
                "labels_to_target": {
                    "0.8": {"mean_kappa": 200, "every_seed_ci": None},
                    "0.9": {"mean_kappa": None, "every_seed_ci": None},
                },
            }
        },
        "spent_usd": 0.0,
    }


def test_render_frontier_table() -> None:
    meta = {"command": "python -m eval.frontier", "git_sha": "abc1234def", "wall_seconds": 600}
    table = render.render_frontier([("frontier-ag_news-gold", _report(), meta)])
    assert "#### ag_news, teacher `gold`, student `all-MiniLM-L6-v2`" in table
    assert "| Strategy | 100 labels | 200 labels | Labels to kappa 0.8" in table
    assert "| random | 0.700 [0.690, 0.720] | 0.850 [0.840, 0.870] | 200 / > 200" in table
    assert "| > 200 / > 200 | $0.10 |" in table
    assert "| uncertainty | not implemented yet | not implemented yet | n/a | n/a | n/a |" in table
    assert "`python -m eval.frontier` at `abc1234`" in table
    assert "wall 10 min" in table
    assert "gold labels" in table
    assert "4.0 at 100, 4.0 at 200" in table
    assert "477 held-out inputs scored every round" in table  # the split's share, not the cap

    claude = render.render_frontier([("frontier-ag_news-claude", _report("claude"), meta)])
    assert "gold labels" not in claude

    text = "a\n<!-- BEGIN generated:frontier -->\nstale body\n<!-- END generated:frontier -->\nb\n"
    out = render.render(text, {"frontier": table})
    assert "stale body" not in out
    assert "| random | 0.700" in out
    assert out.endswith("<!-- END generated:frontier -->\nb\n")

    none = render.render_frontier([])
    assert "run `make eval-frontier`" in none
    assert "run `make eval-frontier`" in render.render(text, {"frontier": none})


def test_load_frontiers_skips_smoke_runs(tmp_path: Path) -> None:
    for name in ("frontier-ag_news-gold", "frontier-ag_news-gold-smoke"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "report.json").write_text(json.dumps(_report()), encoding="utf-8")
    (tmp_path / "frontier-ag_news-gold" / "meta.json").write_text(
        json.dumps({"git_sha": "abc"}), encoding="utf-8"
    )
    runs = render.load_frontiers(tmp_path)
    assert [(name, meta) for name, _, meta in runs] == [
        ("frontier-ag_news-gold", {"git_sha": "abc"})
    ]
    assert "| random | 0.700" in render.blocks_for(tmp_path)["frontier"]
