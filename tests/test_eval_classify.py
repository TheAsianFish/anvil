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
from anvil.spec import TaskSpec
from anvil.store import Split, Store, read_inputs
from anvil.teacher.base import Failure, Item, Label, LabelBatch
from anvil.teacher.fake import FakeTeacher
from anvil.teacher.labelfile import read_label_file, write_label_file
from eval import classify, render
from eval.classify import EvalConfig, load_config, resolve_teacher_kind
from eval.data import Sample, label_name, near_duplicates, sample
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


def test_near_duplicates_count_word_overlap_not_exact_copies() -> None:
    pool = ["Consumer Prices Calm Down in November (AP)", "Stocks fall as oil rises", ""]
    heldout = [
        "Consumer prices calm down in November",  # 6 of 7 words: a near-copy
        "Oil rises as stocks fall",  # the same words in another order: Jaccard 1
        "Rates steady as markets wait",  # no overlap
        "",
    ]
    assert near_duplicates(heldout, pool) == 2
    assert near_duplicates(heldout, pool, threshold=1.0) == 1
    assert near_duplicates([], pool) == 0


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
    assert teacher["failures"] == 0
    assert teacher["pool_labels"] == 160
    tfidf = on_disk["baselines"]["tfidf_logreg"]
    assert "kappa" in tfidf
    # The student-vs-TF-IDF comparison is paired on the same held-out items and reproducible
    # from the per-item predictions the report keeps.
    pred = on_disk["predictions"]
    assert len(pred["input_id"]) == ag["n"]
    assert pred["teacher"] == pred["gold"]  # gold is the teacher here
    assert cohen_kappa(pred["teacher"], pred["student"]) == pytest.approx(ag["kappa"])
    assert cohen_kappa(pred["teacher"], pred["tfidf"]) == pytest.approx(tfidf["kappa"])
    diff = tfidf["student_minus_tfidf_kappa"]
    assert diff["estimate"] == pytest.approx(ag["kappa"] - tfidf["kappa"])
    assert diff["low"] <= diff["estimate"] <= diff["high"]
    assert diff["n_resamples"] == 200
    assert all(0 < c <= 1 for c in pred["student_confidence"])

    table = render.render_classify(render.load_runs(tmp_path / "results"))
    assert "| toy | gold | tiny | 160 |" in table
    assert f"{diff['estimate']:+.3f} [" in table
    assert "learns from 160 clean labels" in table
    assert on_disk["dataset"]["near_duplicate_heldout"] >= 0
    assert (
        f"- Near-duplicates: {on_disk['dataset']['near_duplicate_heldout']} of {ag['n']}" in table
    )
    assert "not a parity-with-Claude number" in table


def _as_labels(rows: list[Label]) -> list[Label]:
    return [
        Label(r.input_id, r.label, r.teacher_model, r.prompt_hash, 300, 5, 3.25e-5) for r in rows
    ]


def _flipped(config: EvalConfig, data: Sample, k: int, offset: int) -> list[Label]:
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
        rows.append(Label(it.id, lab, spec.teacher.model, prompt, 300, 5, 3.25e-5))
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


def test_partial_recorded_labels_are_refused_not_scored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, data = _toy_config(tmp_path)
    monkeypatch.setattr(classify, "LABELS_DIR", tmp_path / "labels")
    spec, para = config.spec, config.paraphrase_spec()
    model = spec.teacher.model
    # A recording from a smaller (smoke) run covers only part of the sample.
    write_label_file(
        classify.label_file("toy", model, spec.prompt_hash),
        _as_labels(_flipped(config, data, 7, 0)[:60]),
    )
    write_label_file(
        classify.label_file("toy", model, para.prompt_hash),
        _as_labels(_flipped(config, data, 5, 1)),
    )
    assert resolve_teacher_kind("auto", config) == "replay"
    out = tmp_path / "results" / "classify-toy-haiku"
    with pytest.raises(ConfigError, match=r"no recorded label.*incomplete"):
        classify.run(config, data, "auto", out, _settings(tmp_path), n_resamples=50)
    assert not out.exists()


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
    with pytest.raises(ConfigError, match="re-label with the live teacher"):
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
     (["--epochs", "1"], "-smoke"), (["--seed", "2"], "-seed2"), (["--seed", "0"], ""),
     (["--seed", "2", "--size", "100"], "-smoke"), (["--resamples", "10"], "-smoke")],
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
    seen: list[EvalConfig] = []

    def fake_run(
        *args: object, provenance: object = None, label_only: bool = False
    ) -> dict[str, object]:
        out = args[3]
        assert isinstance(out, Path)
        assert isinstance(args[0], EvalConfig)
        seen.append(args[0])
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
    # The overrides reach the run, not just the directory name.
    used = seen[0]
    assert used.spec.student.seed == (2 if "--seed" in flags and "2" in flags else 0)
    if "--size" in flags:
        assert (used.sample.size, used.sample.pool_labels, used.sample.heldout_labels) == (
            100, 67, 25,
        )  # fmt: skip
    if "--epochs" in flags:
        assert used.spec.student.epochs == 1
    if "--student" in flags:
        assert used.spec.student.model == "tiny"


def test_main_refuses_out_with_several_datasets(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        classify.main(["ag_news", "banking77", "--out", "/tmp/x"])
    assert "--out takes one dataset" in capsys.readouterr().err


def test_render_skips_smoke_runs(tmp_path: Path) -> None:
    config, data = _toy_config(tmp_path)
    out = tmp_path / "results" / "classify-toy-gold-smoke"
    classify.run(config, data, "gold", out, _settings(tmp_path), n_resamples=50)
    assert render.load_runs(tmp_path / "results") == []
    (out.parent / "classify-toy-gold").mkdir()
    for name in ("report.json", "meta.json"):
        (out / name).rename(out.parent / "classify-toy-gold" / name)
    assert [name for name, _, _ in render.load_runs(tmp_path / "results")] == ["classify-toy-gold"]


def _seed_report(seed: int, kappa: float, low: float, high: float) -> dict[str, object]:
    return {
        "dataset": {"name": "toy"},
        "teacher": {"model": "gold"},
        "student": {"model": "org/enc", "settings": {"seed": seed}},
        "agreement": {"kappa": kappa, "kappa_ci": {"low": low, "high": high}},
    }


def test_seed_spread_groups_repeats_with_the_canonical_run() -> None:
    runs: list[tuple[str, render.Report, render.Report]] = [
        ("classify-toy-gold", _seed_report(0, 0.80, 0.75, 0.85), {}),
        ("classify-toy-gold-seed2", _seed_report(2, 0.70, 0.66, 0.74), {}),
        ("classify-toy-gold-seed1", _seed_report(1, 0.90, 0.86, 0.94), {}),
        ("classify-other-gold", _seed_report(0, 0.5, 0.4, 0.6), {}),
    ]
    table = render.render_seed_spread(runs)
    assert (
        "| toy | gold | enc | 3 | 0: 0.800, 1: 0.900, 2: 0.700 | 0.800 | 0.200 | 0.087 |" in table
    )
    assert "other" not in table  # a dataset without repeats has no spread to show
    assert "No seed repeats" in render.render_seed_spread(runs[3:])


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


class _LiveStub:
    """Stands in for `ClaudeTeacher` in the paid path: gold with flips, billed per label and
    never cached, under the live model id. `calls` records every (prompt hash, input id) it is
    asked for, so a test can see exactly what a run would have paid for."""

    COST = 3.25e-5

    def __init__(
        self, config: EvalConfig, data: Sample, spec: TaskSpec, calls: list[tuple[str, str]]
    ) -> None:
        main = spec.prompt_hash == config.spec.prompt_hash
        rows = _flipped(config, data, 7, 0) if main else _flipped(config, data, 5, 1)
        self.model_id = spec.teacher.model
        self._rows = {r.input_id: r for r in rows}
        self._calls = calls

    def label(self, items: list[Item]) -> LabelBatch:
        batch = LabelBatch()
        for it in items:
            r = self._rows[it.id]
            self._calls.append((r.prompt_hash, it.id))
            batch.labels.append(
                Label(it.id, r.label, r.teacher_model, r.prompt_hash, 300, 5, self.COST)
            )
        return batch


def test_claude_run_pays_only_for_unrecorded_labels_and_records_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, data = _toy_config(tmp_path)
    monkeypatch.setattr(classify, "LABELS_DIR", tmp_path / "labels")
    spec, para = config.spec, config.paraphrase_spec()
    model = spec.teacher.model
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        classify,
        "build_teacher",
        lambda s, settings, provider=None: _LiveStub(config, data, s, calls),
    )
    # A committed recording from an earlier, smaller run: a fresh checkout has this file but
    # no store and no disk cache, and must not buy these 60 labels a second time.
    main_rows = _flipped(config, data, 7, 0)
    recorded = {r.input_id for r in main_rows[:60]}
    write_label_file(
        classify.label_file("toy", model, spec.prompt_hash), _as_labels(main_rows[:60])
    )

    settings = _settings(tmp_path)
    out = tmp_path / "results" / "classify-toy-haiku"
    report = classify.run(config, data, "claude", out, settings, n_resamples=100)
    teacher = report["teacher"]
    assert isinstance(teacher, dict)
    assert teacher["kind"] == "claude"
    assert teacher["pool_labels"] == 160
    ws = classify.workspace(settings, spec, data)
    with ws.open_store(spec) as store:
        labelled = set(store.labels(model, spec.prompt_hash))
    bought_main = {i for h, i in calls if h == spec.prompt_hash}
    bought_para = [i for h, i in calls if h == para.prompt_hash]
    assert bought_main == labelled - recorded  # only the inputs the file did not cover
    assert len(bought_para) == config.ceiling.n
    assert report["spent_usd"] == pytest.approx((len(bought_main) + 40) * _LiveStub.COST)
    assert teacher["label_cost_usd"] == pytest.approx(len(labelled) * _LiveStub.COST)
    # Both prompts' labels are recorded, merged with the rows that were already there.
    main_file = read_label_file(classify.label_file("toy", model, spec.prompt_hash))
    assert {r.input_id for r in main_file} == labelled | recorded
    assert all(r.teacher_model == model and r.cost_usd == _LiveStub.COST for r in main_file)
    para_file = read_label_file(classify.label_file("toy", model, para.prompt_hash))
    assert sorted(r.input_id for r in para_file) == sorted(bought_para)
    assert resolve_teacher_kind("auto", config) == "replay"

    # A fresh checkout (new home: no store, no cache) now replays the whole run for free.
    bought_before = len(calls)
    again = classify.run(config, data, "auto", out, _settings(tmp_path / "again"), 100)
    assert len(calls) == bought_before
    assert again["spent_usd"] == 0
    assert again["teacher"]["kind"] == "replay"  # type: ignore[index]
    assert again["teacher"]["label_cost_usd"] == teacher["label_cost_usd"]  # type: ignore[index]
    assert again["ceiling"] == report["ceiling"]
    assert again["agreement"]["kappa"] == report["agreement"]["kappa"]  # type: ignore[index]


def test_main_refuses_a_smoke_run_pointed_at_a_results_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(classify, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(classify, "run", lambda *a, **k: pytest.fail("must not run"))
    target = tmp_path / "results" / "classify-ag_news-gold"
    with pytest.raises(SystemExit):
        classify.main(["ag_news", "--teacher", "gold", "--size", "100", "--out", str(target)])
    assert "-smoke" in capsys.readouterr().err


def test_results_md_check_fails_when_a_rendered_number_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = tmp_path / "RESULTS.md"
    text = render.RESULTS_MD.read_text(encoding="utf-8")
    assert "0.844 [0.805, 0.882]" in text
    stale.write_text(
        text.replace("0.844 [0.805, 0.882]", "0.944 [0.905, 0.982]"), encoding="utf-8"
    )
    monkeypatch.setattr(render, "RESULTS_MD", stale)
    assert render.main(["--check"]) == 1


def test_recording_merges_failures_and_a_label_always_wins(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "toy" / "haiku.jsonl"
    key = ("claude-haiku-5-5", "p1")

    def lab(i: str, label: str = "x") -> Label:
        return Label(i, label, *key, 300, 5, 1e-4)

    def refusal(i: str, reason: str = "refusal (cyber)") -> Failure:
        return Failure(i, reason, 1000, 50, 0.005)

    write_label_file(path, [lab("a")], [(refusal("b"), *key)])
    # The store has a newer refusal for an input the file has a label for (the label stays),
    # a retry that failed differently (it replaces the old failure) and two new inputs.
    n = classify.record_labels(
        path, [lab("c")], [refusal("a"), refusal("b", "hit max_tokens"), refusal("d")], key
    )
    assert n == 4
    rows = {r.input_id: (r.label, r.reason) for r in read_label_file(path)}
    assert rows == {
        "a": ("x", None),
        "b": (None, "hit max_tokens"),
        "c": ("x", None),
        "d": (None, "refusal (cyber)"),
    }
    # A later answer for a refused input replaces its failure row.
    assert classify.record_labels(path, [lab("b", "y")]) == 4
    assert {r.input_id: r.label for r in read_label_file(path)}["b"] == "y"
    with pytest.raises(ValueError, match="teacher model, prompt hash"):
        classify.record_labels(path, [], [refusal("e")])


class _RefusingStub(_LiveStub):
    """`_LiveStub` that refuses (and bills) the given (prompt hash, input id) pairs."""

    def __init__(
        self,
        config: EvalConfig,
        data: Sample,
        spec: TaskSpec,
        calls: list[tuple[str, str]],
        refuse: set[tuple[str, str]],
    ) -> None:
        super().__init__(config, data, spec, calls)
        self._refuse = refuse

    def label(self, items: list[Item]) -> LabelBatch:
        batch = super().label(items)
        refused = [lb for lb in batch.labels if (lb.prompt_hash, lb.input_id) in self._refuse]
        for lb in refused:
            batch.labels.remove(lb)
            batch.failures.append(Failure(lb.input_id, "refusal (cyber)", 300, 5, self.COST))
        return batch


def test_recorded_refusals_replay_faithfully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, data = _toy_config(tmp_path)
    monkeypatch.setattr(classify, "LABELS_DIR", tmp_path / "labels")
    spec, para = config.spec, config.paraphrase_spec()
    model = spec.teacher.model
    with Store(tmp_path / "probe.sqlite", spec.splits.seed, spec.splits.heldout_fraction) as s:
        s.add_inputs(data.items)
        heldout_order = [it.id for it in s.inputs(Split.HELDOUT)]
        pool_order = [it.id for it in s.inputs(Split.POOL)]
    # The teacher refuses one held-out and one pool input under the main prompt, and one
    # held-out input of the ceiling subset under the paraphrase.
    refuse = {
        (spec.prompt_hash, heldout_order[3]),
        (spec.prompt_hash, pool_order[5]),
        (para.prompt_hash, heldout_order[7]),
    }
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        classify,
        "build_teacher",
        lambda s, settings, provider=None: _RefusingStub(config, data, s, calls, refuse),
    )
    settings = _settings(tmp_path)
    out = tmp_path / "results" / "classify-toy-haiku"
    live = classify.run(config, data, "claude", out, settings, n_resamples=100)
    teacher = live["teacher"]
    assert isinstance(teacher, dict)
    assert teacher["failures"] == 3
    asked = min(config.sample.heldout_labels, len(heldout_order))
    assert teacher["heldout_labels"] == asked - 1  # one refused
    assert teacher["pool_labels"] == 159
    assert live["spent_usd"] == pytest.approx(len(calls) * _LiveStub.COST)
    pred = live["predictions"]
    assert isinstance(pred, dict)
    assert heldout_order[3] not in pred["input_id"]
    main_file = read_label_file(classify.label_file("toy", model, spec.prompt_hash))
    assert {r.input_id: r.reason for r in main_file if r.failed} == {
        heldout_order[3]: "refusal (cyber)",
        pool_order[5]: "refusal (cyber)",
    }
    para_file = read_label_file(classify.label_file("toy", model, para.prompt_hash))
    assert [r.input_id for r in para_file if r.failed] == [heldout_order[7]]

    # A fresh checkout replays the run, refusals included, for free and to the same numbers.
    assert resolve_teacher_kind("auto", config) == "replay"
    bought = len(calls)
    fresh = _settings(tmp_path / "again")
    again = classify.run(config, data, "auto", out, fresh, n_resamples=100)
    assert len(calls) == bought
    assert again["spent_usd"] == 0
    assert again["predictions"] == live["predictions"]  # the same held-out set and labels
    assert again["agreement"] == live["agreement"]
    assert again["ceiling"] == live["ceiling"]
    for k in ("failures", "heldout_labels", "pool_labels", "label_cost_usd"):
        assert again["teacher"][k] == teacher[k]  # type: ignore[index]

    # Re-running the replay (and the live run, which now has the recording) asks nothing new,
    # stores no replayed failures and keeps the cost where it was.
    ws = classify.workspace(fresh, spec, data)
    third = classify.run(config, data, "auto", out, fresh, n_resamples=100)
    rerun = classify.run(config, data, "claude", out, settings, n_resamples=100)
    assert len(calls) == bought
    for report in (third, rerun):
        assert report["predictions"] == live["predictions"]
        assert report["spent_usd"] == 0
        for k in ("failures", "pool_labels", "label_cost_usd"):
            assert report["teacher"][k] == teacher[k]  # type: ignore[index]
    with sqlite3.connect(ws.db) as db:
        assert db.execute("SELECT COUNT(*) FROM failures").fetchone()[0] == 0


def test_override_validates_the_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = load_config(classify.CONFIG_DIR / "ag_news.yaml")
    with pytest.raises(ConfigError, match="epochs"):
        classify._override(config, None, -3, None)
    with pytest.raises(ConfigError, match="size"):
        classify._override(config, None, None, -5)

    def unreachable(*a: object, **k: object) -> None:
        raise AssertionError("the dataset was loaded for an invalid override")

    monkeypatch.setattr(classify, "load_rows", unreachable)
    assert classify.main(["ag_news", "--teacher", "gold", "--epochs", "-3"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: invalid override")
    assert "Traceback" not in err
