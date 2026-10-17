"""The `make eval-classify` harness end to end, offline: toy inputs stand in for a dataset."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import yaml

from anvil.errors import ConfigError
from anvil.evaluate.metrics import cohen_kappa
from anvil.settings import Settings
from anvil.store import Split, read_inputs
from anvil.teacher.base import Label
from anvil.teacher.fake import FakeTeacher
from anvil.teacher.labelfile import RecordedLabel, read_label_file, write_label_file
from eval import classify, render
from eval.classify import EvalConfig, load_config, resolve_teacher_kind
from eval.data import Sample, label_name, sample
from tests.conftest import TOY_INPUTS, TOY_SPEC


def test_label_names_are_normalised_to_the_spec_alphabet() -> None:
    assert label_name("Sci/Tech") == "sci_tech"
    assert label_name("reverted_card_payment?") == "reverted_card_payment"
    assert label_name("Refund_not_showing_up") == "refund_not_showing_up"


@pytest.mark.parametrize(("name", "n_labels"), [("ag_news", 4), ("banking77", 77)])
def test_shipped_configs_are_valid(name: str, n_labels: int) -> None:
    config = load_config(classify.CONFIG_DIR / f"{name}.yaml")
    assert len(config.spec.labels) == n_labels
    assert config.spec.teacher.model == "claude-haiku-5-5"
    assert config.paraphrase_spec().prompt_hash != config.spec.prompt_hash
    assert config.sample.pool_labels + config.sample.heldout_labels == config.sample.size


def test_sample_is_seeded_distinct_and_traceable() -> None:
    texts = ["a", "b", "b", " ", "c", "d", "e"]
    labels = ["x", "y", "y", "x", "x", "y", "x"]
    first = sample("ds", texts, labels, n=4, seed=3)
    assert first == sample("ds", texts, labels, n=4, seed=3)
    assert first.sha256 == sample("ds", texts, labels, n=4, seed=3).sha256
    assert len({it.text for it in first.items}) == 4
    for it in first.items:
        assert texts[int(it.id.split("-")[1])] == it.text
        assert first.gold[it.id] == labels[int(it.id.split("-")[1])]
    assert sample("ds", texts, labels, n=4, seed=4).sha256 != first.sha256


def _toy_config(tmp_path: Path) -> tuple[EvalConfig, Sample]:
    raw_spec = yaml.safe_load(TOY_SPEC.read_text(encoding="utf-8"))
    raw_spec["teacher"]["model"] = "claude-haiku-5-5"
    raw_spec["student"].update({"epochs": 3, "max_length": 24})
    raw = {
        "dataset": {"name": "toy", "repo": "local/toy", "revision": "0" * 40},
        "sample": {"size": 240, "pool_labels": 160, "heldout_labels": 60, "seed": 0},
        "ceiling": {"n": 40, "instructions": "Pick the support queue for this ticket."},
        "spec": raw_spec,
    }
    config = EvalConfig.model_validate(raw)
    items = read_inputs(TOY_INPUTS)
    gold_teacher = FakeTeacher(
        config.spec.model_copy(
            update={"fake": config.spec.fake.model_copy(update={"noise": 0.0})}  # type: ignore[union-attr]
        )
    )
    texts = [it.text for it in items]
    labels = [gold_teacher.rule_label(t) for t in texts]
    return config, sample("toy", texts, labels, n=240, seed=0)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        home=tmp_path / "home", teacher_workers=1, record_dir=None, log_level="WARNING"
    )


def test_gold_run_writes_a_complete_report_and_renders(tmp_path: Path) -> None:
    config, data = _toy_config(tmp_path)
    out = tmp_path / "results" / "classify-toy-gold"
    report = classify.run(config, data, "gold", out, _settings(tmp_path), n_resamples=200)

    on_disk = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(report))
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert {"git_sha", "seed", "wall_seconds", "spent_usd", "versions"} <= set(meta)
    assert (out / "train-metrics.jsonl").exists()

    ag = on_disk["agreement"]
    teacher = on_disk["teacher"]
    assert teacher["model"] == "gold"
    assert teacher["label_cost_usd"] == 0
    assert teacher["heldout_labels"] == ag["n"] <= 60
    assert ag["kappa_ci"]["low"] <= ag["kappa"] <= ag["kappa_ci"]["high"]
    # The headline CI follows the run's resamples, like every other interval in the report.
    assert ag["kappa_ci"]["n_resamples"] == 200
    assert "skipped" in on_disk["ceiling"]
    assert "teacher_vs_gold" not in on_disk["gold"]
    # With gold as the teacher, student-vs-gold is the headline number again.
    assert on_disk["gold"]["student_vs_gold"]["kappa"] == pytest.approx(ag["kappa"])
    assert "kappa" in on_disk["baselines"]["tfidf_logreg"]

    table = render.render_classify(render.load_runs(tmp_path / "results"))
    assert "| toy | gold | tiny |" in table
    assert "not a parity-with-Claude number" in table


def _as_labels(rows: list[RecordedLabel]) -> list[Label]:
    return [
        Label(r.input_id, r.label, r.teacher_model, r.prompt_hash, 300, 5, 3.25e-5) for r in rows
    ]


def _flipped(config: EvalConfig, data: Sample, k: int, offset: int) -> list[RecordedLabel]:
    """A "recorded" teacher: gold with every k-th answer (from `offset`) moved to the next
    label; offset 0 records the main prompt, any other the paraphrase."""
    spec, para = config.spec, config.paraphrase_spec()
    names = spec.label_names
    rows = []
    for i, it in enumerate(data.items):
        lab = data.gold[it.id]
        if i % k == offset:
            lab = names[(names.index(lab) + 1) % len(names)]
        prompt = spec.prompt_hash if offset == 0 else para.prompt_hash
        rows.append(RecordedLabel(it.id, lab, spec.teacher.model, prompt, 300, 5, 3.25e-5))
    return rows


def test_replay_run_measures_the_ceiling_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, data = _toy_config(tmp_path)
    monkeypatch.setattr(classify, "LABELS_DIR", tmp_path / "labels")
    spec, para = config.spec, config.paraphrase_spec()
    model = spec.teacher.model
    with pytest.raises(ConfigError, match="--teacher gold"):
        resolve_teacher_kind("auto", config)

    main_rows, para_rows = _flipped(config, data, 7, 0), _flipped(config, data, 5, 1)
    write_label_file(classify.label_file("toy", model, spec.prompt_hash), _as_labels(main_rows))
    write_label_file(classify.label_file("toy", model, para.prompt_hash), _as_labels(para_rows))
    assert resolve_teacher_kind("auto", config) == "replay"

    settings = _settings(tmp_path)
    out = tmp_path / "results" / "classify-toy-haiku"
    report = classify.run(config, data, "auto", out, settings, n_resamples=200)
    teacher = report["teacher"]
    assert isinstance(teacher, dict)
    assert teacher["kind"] == "replay"
    assert teacher["model"] == model
    assert teacher["label_cost_usd"] > 0  # what the labels cost when they were recorded
    assert report["spent_usd"] == 0  # a replay pays nothing

    # The expected numbers, straight from the constructed flips on the same items.
    ws = classify.workspace(settings, spec, data)
    with ws.open_store(spec) as store:
        heldout = [it.id for it, _ in store.labelled(model, spec.prompt_hash, Split.HELDOUT)]
    main = {r.input_id: r.label for r in main_rows}
    second = {r.input_id: r.label for r in para_rows}
    subset = heldout[: config.ceiling.n]
    ceiling = report["ceiling"]
    assert isinstance(ceiling, dict)
    assert ceiling["n"] == 40
    assert ceiling["kappa"] == pytest.approx(
        cohen_kappa([main[i] for i in subset], [second[i] for i in subset])
    )
    gold = report["gold"]
    assert isinstance(gold, dict)
    assert gold["teacher_vs_gold"]["kappa"] == pytest.approx(
        cohen_kappa([data.gold[i] for i in heldout], [main[i] for i in heldout])
    )
    assert 0 < ceiling["kappa"] < 1
    assert 0 < gold["teacher_vs_gold"]["kappa"] < 1

    # A second run reuses the stored paraphrase labels instead of appending them again.
    again = classify.run(config, data, "auto", out, settings, n_resamples=200)
    assert again["ceiling"] == ceiling
    # ...and tops the pool up to the configured total rather than labelling 160 more.
    assert again["teacher"]["pool_labels"] == teacher["pool_labels"] == 160  # type: ignore[index]
    with sqlite3.connect(ws.db) as db:
        rows = db.execute(
            "SELECT COUNT(*) FROM labels WHERE prompt_hash = ?", (para.prompt_hash,)
        ).fetchone()[0]
    assert rows == config.ceiling.n


def test_auto_replays_only_when_the_paraphrase_labels_are_recorded_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, data = _toy_config(tmp_path)
    monkeypatch.setattr(classify, "LABELS_DIR", tmp_path / "labels")
    spec, para = config.spec, config.paraphrase_spec()
    model = spec.teacher.model
    write_label_file(
        classify.label_file("toy", model, spec.prompt_hash),
        _as_labels(_flipped(config, data, 7, 0)),
    )
    with pytest.raises(ConfigError, match=para.prompt_hash):
        resolve_teacher_kind("auto", config)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert resolve_teacher_kind("auto", config) == "claude"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    # Without a ceiling, the main labels alone are enough to replay.
    no_ceiling = config.model_copy(update={"ceiling": config.ceiling.model_copy(update={"n": 0})})
    assert resolve_teacher_kind("auto", no_ceiling) == "replay"


def test_paraphrase_labels_missing_the_subset_are_a_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, data = _toy_config(tmp_path)
    monkeypatch.setattr(classify, "LABELS_DIR", tmp_path / "labels")
    spec, para = config.spec, config.paraphrase_spec()
    model = spec.teacher.model
    write_label_file(
        classify.label_file("toy", model, spec.prompt_hash),
        _as_labels(_flipped(config, data, 7, 0)),
    )
    write_label_file(classify.label_file("toy", model, para.prompt_hash), [])
    out = tmp_path / "results" / "classify-toy-haiku"
    with pytest.raises(ConfigError, match="re-label with --teacher claude"):
        classify.run(config, data, "replay", out, _settings(tmp_path), n_resamples=50)


def test_recording_merges_into_the_existing_label_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, data = _toy_config(tmp_path)
    rows = _as_labels(_flipped(config, data, 7, 0))
    path = tmp_path / "labels" / "toy" / "haiku.jsonl"
    assert classify.record_labels(path, rows) == len(rows)
    # A later, smaller run relabels a few inputs: those rows change, the rest are kept.
    small = [
        Label(lb.input_id, "other", lb.teacher_model, lb.prompt_hash, 1, 1, 1e-6)
        for lb in rows[:10]
    ]
    assert classify.record_labels(path, small) == len(rows)
    recorded = read_label_file(path)
    assert [r.input_id for r in recorded] == sorted(lb.input_id for lb in rows)
    by_id = {r.input_id: r.label for r in recorded}
    assert all(by_id[lb.input_id] == "other" for lb in small)
    assert all(by_id[lb.input_id] == lb.label for lb in rows[10:])


@pytest.mark.parametrize(
    ("flags", "suffix"),
    [([], ""), (["--size", "100"], "-smoke"), (["--student", "tiny"], "-smoke"),
     (["--epochs", "1"], "-smoke")],
)  # fmt: skip
def test_main_sends_override_runs_to_a_smoke_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
    suffix: str,
) -> None:
    config, data = _toy_config(tmp_path)
    monkeypatch.setattr(classify, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(classify, "load_config", lambda path: config)
    monkeypatch.setattr(classify, "load_rows", lambda ref: ([], []))
    monkeypatch.setattr(classify, "sample", lambda *args, **kwargs: data)
    monkeypatch.setattr(classify, "git_provenance", lambda: {"git_sha": "abc", "git_dirty": False})
    calls: list[tuple[Path, object]] = []

    def fake_run(*args: object, provenance: object = None) -> dict[str, object]:
        out = args[3]
        assert isinstance(out, Path)
        calls.append((out, provenance))
        ci = {"low": 0.5, "high": 0.7}
        return {
            "dataset": {"name": "toy"},
            "agreement": {"kappa": 0.6, "kappa_ci": ci, "n": 60},
            "teacher": {"model": "gold", "pool_labels": 160},
        }

    monkeypatch.setattr(classify, "run", fake_run)
    assert classify.main(["toy", "--teacher", "gold", *flags]) == 0
    assert calls == [
        (
            tmp_path / "results" / f"classify-toy-gold{suffix}",
            {"git_sha": "abc", "git_dirty": False},
        )
    ]
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 1
    assert printed[0].startswith("toy: gold -> student: kappa 0.600 [0.500, 0.700] on 60")


def test_render_skips_smoke_runs(tmp_path: Path) -> None:
    config, data = _toy_config(tmp_path)
    out = tmp_path / "results" / "classify-toy-gold-smoke"
    classify.run(config, data, "gold", out, _settings(tmp_path), n_resamples=50)
    assert render.load_runs(tmp_path / "results") == []
    (out.parent / "classify-toy-gold").mkdir()
    for name in ("report.json", "meta.json"):
        (out / name).rename(out.parent / "classify-toy-gold" / name)
    assert [name for name, _, _ in render.load_runs(tmp_path / "results")] == ["classify-toy-gold"]


def test_render_replaces_only_marked_blocks() -> None:
    text = (
        "intro\n<!-- BEGIN generated:classify -->\nold\n<!-- END generated:classify -->\n"
        "<!-- BEGIN generated:other -->\nkeep\n<!-- END generated:other -->\noutro\n"
    )
    out = render.render(text, {"classify": "new table\n"})
    assert "new table\n<!-- END generated:classify -->" in out
    assert "keep" in out
    assert "old" not in out
    assert render.render(out, {"classify": "new table\n"}) == out


def test_results_md_is_rendered_from_the_committed_raw_output() -> None:
    assert render.main(["--check"]) == 0
