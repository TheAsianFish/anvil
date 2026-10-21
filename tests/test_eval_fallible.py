"""The fallible-teacher additions to the eval harness: the local (Ollama) teacher in
`eval.classify`, its error and ceiling statistics, and the `eval.noise` control, offline."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from anvil.evaluate.metrics import cohen_kappa
from anvil.settings import Settings
from anvil.spec import StudentSpec
from anvil.store import Split
from anvil.teacher.base import Failure, Label
from anvil.teacher.cache import DiskCache
from anvil.teacher.labelfile import read_label_file, write_label_file
from anvil.teacher.ollama import OllamaTeacher
from eval import classify, noise, render
from eval.classify import EvalConfig, OllamaCfg
from eval.data import Sample
from tests.test_eval_classify import _toy_config
from tests.test_teacher_ollama import DIGEST, MODEL, ScriptedOllama, chat_response

JSONDict = dict[str, object]


def _settings(tmp_path: Path, name: str = "home") -> Settings:
    return Settings(home=tmp_path / name, teacher_workers=1, record_dir=None, log_level="WARNING")


def _local_toy(tmp_path: Path) -> tuple[EvalConfig, Sample]:
    """The toy eval with an `ollama` section, converted to the local teacher."""
    config, data = _toy_config(tmp_path)
    config = config.model_copy(update={"ollama": OllamaCfg(model=MODEL, digest=DIGEST)})
    return config.with_local_teacher(), data


# --- eval.classify: file names and the local teacher config ---------------------------------


def test_file_safe_only_rewrites_unportable_characters() -> None:
    assert classify.file_safe("qwen2.5:3b") == "qwen2.5-3b"
    assert classify.file_safe("claude-haiku-5-5") == "claude-haiku-5-5"
    assert classify.file_safe("org/model:tag") == "org-model-tag"


def test_label_file_and_results_dir_use_the_file_safe_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(classify, "LABELS_DIR", tmp_path / "labels")
    path = classify.label_file("toy", "qwen2.5:3b", "abc123")
    assert path == tmp_path / "labels" / "toy" / "qwen2.5-3b-abc123.jsonl"
    assert classify.results_dir("toy", "qwen2.5:3b").name == "classify-toy-qwen2.5-3b"
    assert classify.results_dir("toy", "qwen2.5:3b", smoke=True).name.startswith(
        "classify-toy-qwen2.5-3b"
    )
    assert ":" not in classify.results_dir("toy", "qwen2.5:3b", seed=1).name
    assert classify.results_dir("toy", "claude-haiku-5-5").name == "classify-toy-claude-haiku-5-5"


def test_with_local_teacher_keeps_the_prompt_and_swaps_the_model(tmp_path: Path) -> None:
    claude, _ = _toy_config(tmp_path)
    claude = claude.model_copy(update={"ollama": OllamaCfg(model=MODEL, digest=DIGEST)})
    local = claude.with_local_teacher()
    t = local.spec.teacher
    assert (t.provider, t.model, t.digest) == ("ollama", MODEL, DIGEST)
    assert t.max_tokens == claude.spec.teacher.max_tokens
    assert local.spec.prompt_hash == claude.spec.prompt_hash  # the labels compare
    para = local.paraphrase_spec()
    assert para.teacher.model == MODEL
    assert para.teacher.provider == "ollama"
    assert para.prompt_hash == claude.paraphrase_spec().prompt_hash
    assert claude.spec.teacher.provider == "claude"  # the original is untouched


@pytest.mark.parametrize("name", ["banking77", "ag_news"])
def test_shipped_configs_pin_the_local_model(name: str) -> None:
    config = classify.load_config(classify.CONFIG_DIR / f"{name}.yaml")
    assert config.ollama.model == "qwen2.5:3b"
    digest = config.ollama.digest
    assert digest is not None
    assert len(digest) == 64
    int(digest, 16)  # hex
    local = config.with_local_teacher()
    assert local.spec.teacher.digest == digest
    assert local.spec.prompt_hash == config.spec.prompt_hash


# --- eval.classify: statistics --------------------------------------------------------------


def test_teacher_errors_counts_and_paired_difference() -> None:
    gold = ["a", "a", "a", "b", "b", "b", "c", "c"]
    teacher = ["a", "b", "b", "b", "c", "b", "c", "a"]
    student = ["a", "b", "a", "b", "c", "a", "b", "b"]
    out = classify.teacher_errors(gold, teacher, student, 100, 0)

    assert out["on_teacher_errors"] == {"n": 4, "copied": 2, "corrected": 1, "other": 1}
    assert out["on_teacher_correct"] == {"n": 4, "broken": 2}

    diff = out["student_minus_teacher_vs_gold_kappa"]
    assert isinstance(diff, dict)
    expected = cohen_kappa(gold, student) - cohen_kappa(gold, teacher)
    assert diff["estimate"] == pytest.approx(expected)
    assert diff["low"] <= diff["estimate"] <= diff["high"]
    acc = out["student_minus_teacher_vs_gold_accuracy"]
    assert isinstance(acc, dict)
    assert acc["estimate"] == pytest.approx(3 / 8 - 4 / 8)

    confusions = out["confusions"]
    assert isinstance(confusions, dict)
    assert confusions["teacher"] == [
        {"gold": "a", "label": "b", "teacher": 2, "student": 1},
        {"gold": "b", "label": "c", "teacher": 1, "student": 1},
        {"gold": "c", "label": "a", "teacher": 1, "student": 0},
    ]
    assert confusions["student"][0] == {"gold": "c", "label": "b", "student": 2}


def test_teacher_errors_of_a_perfect_teacher_has_no_error_rows() -> None:
    gold = ["a", "b", "a", "b"]
    out = classify.teacher_errors(gold, gold, ["a", "a", "a", "b"], 50, 0)
    assert out["on_teacher_errors"] == {"n": 0, "copied": 0, "corrected": 0, "other": 0}
    assert out["on_teacher_correct"] == {"n": 4, "broken": 1}
    confusions = out["confusions"]
    assert isinstance(confusions, dict)
    assert confusions["teacher"] == []


def test_on_ceiling_items_splits_stable_from_unstable() -> None:
    first = {"1": "a", "2": "a", "3": "b", "4": "b"}
    repeat = {"1": "a", "2": "b", "3": "b", "4": "a", "5": "a"}
    student = {"1": "a", "2": "a", "3": "a", "4": "b", "6": "b"}
    out = classify.on_ceiling_items(first, repeat, student, 50, 0)
    assert out["stable"] == {"n": 2, "agree": 1, "rate": 0.5}
    assert out["unstable"] == {"n": 2, "agree": 2, "rate": 1.0}
    shared = out["student_vs_teacher"]
    assert isinstance(shared, dict)
    assert shared["n"] == 4  # ids 5 and 6 are not on all three
    assert shared["kappa"] == pytest.approx(
        cohen_kappa(["a", "a", "b", "b"], ["a", "a", "a", "b"])
    )


def test_on_ceiling_items_with_no_unstable_items_has_no_rate() -> None:
    same = {"1": "a", "2": "b", "3": "a"}
    out = classify.on_ceiling_items(same, dict(same), {"1": "a", "2": "b", "3": "b"}, 20, 0)
    assert out["unstable"] == {"n": 0, "agree": 0, "rate": None}
    assert out["stable"] == {"n": 3, "agree": 2, "rate": pytest.approx(2 / 3)}


def test_compute_seconds_summarises_local_rows_and_includes_failures() -> None:
    def lab(i: str, s: float | None) -> Label:
        return Label(i, "x", MODEL, "h", compute_seconds=s)

    rows: list[Label | Failure] = [
        lab("1", 1.0),
        lab("2", 2.0),
        lab("3", 4.0),
        Failure("4", "refused", compute_seconds=0.5),
    ]
    assert classify.compute_seconds(rows) == {"total": 7.5, "median_per_label": 2.0, "labels": 4}
    assert classify.compute_seconds([lab("1", None), lab("2", None)]) == {}
    assert classify.compute_seconds([]) == {}
    # Rows without seconds do not count towards the labels.
    assert classify.compute_seconds([lab("1", 3.0), lab("2", None)])["labels"] == 1


# --- eval.classify.run with a local teacher -------------------------------------------------

TOTAL_DURATION = 2_000_000_000  # two seconds of server time per answer


class GoldFlips(ScriptedOllama):
    """Answers every user text with its gold label, moving every k-th item (by sample index)
    to the next label; the paraphrase prompt flips a different set."""

    def __init__(self, config: EvalConfig, data: Sample) -> None:
        super().__init__()
        self.names = config.spec.label_names
        self.instructions = config.ceiling.instructions
        self.index = {it.text: i for i, it in enumerate(data.items)}
        self.gold = {it.text: data.gold[it.id] for it in data.items}

    def answer(self, system: str, text: str) -> str:
        offset, k = (1, 5) if self.instructions in system else (0, 7)
        lab = self.gold[text]
        if self.index[text] % k == offset:
            lab = self.names[(self.names.index(lab) + 1) % len(self.names)]
        return lab

    def __call__(self, body: JSONDict) -> JSONDict:
        messages = body["messages"]
        assert isinstance(messages, list)
        super().__call__(body)
        label = self.answer(messages[0]["content"], messages[1]["content"])
        return chat_response(json.dumps({"label": label}), total_duration=TOTAL_DURATION)


class NoServer(ScriptedOllama):
    def __call__(self, body: JSONDict) -> JSONDict:
        raise AssertionError("the server was asked for a label that is recorded")


def _patch_teacher(
    monkeypatch: pytest.MonkeyPatch, transport: ScriptedOllama, cache_dir: Path
) -> None:
    monkeypatch.setattr(
        classify,
        "build_teacher",
        lambda s, settings, provider=None: OllamaTeacher(s, transport, cache=DiskCache(cache_dir)),
    )


def test_local_teacher_run_records_replays_and_needs_no_server_once_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, data = _local_toy(tmp_path)
    spec, para = config.spec, config.paraphrase_spec()
    monkeypatch.setattr(classify, "LABELS_DIR", tmp_path / "labels")
    _patch_teacher(monkeypatch, GoldFlips(config, data), tmp_path / "cache")

    out = tmp_path / "results" / "classify-toy-qwen2.5-3b"
    report = classify.run(
        config,
        data,
        "ollama",
        out,
        _settings(tmp_path),
        n_resamples=100,
        provenance={"git_sha": "x", "git_dirty": False},
    )
    main_file = tmp_path / "labels" / "toy" / f"qwen2.5-3b-{spec.prompt_hash}.jsonl"
    para_file = tmp_path / "labels" / "toy" / f"qwen2.5-3b-{para.prompt_hash}.jsonl"
    assert main_file.exists()
    assert para_file.exists()
    rows = read_label_file(main_file)
    assert all(r.compute_seconds == pytest.approx(2.0) and r.cost_usd == 0 for r in rows)
    assert all(r.teacher_model == MODEL for r in rows)
    assert len(read_label_file(para_file)) == config.ceiling.n

    teacher = report["teacher"]
    assert isinstance(teacher, dict)
    assert teacher["kind"] == "ollama"
    assert teacher["provider"] == "ollama"
    assert teacher["digest"] == DIGEST
    assert teacher["effort"] is None
    assert teacher["model"] == MODEL
    assert teacher["label_cost_usd"] == 0
    secs = teacher["compute_seconds"]
    assert isinstance(secs, dict)
    assert secs["median_per_label"] == pytest.approx(2.0)
    assert secs["total"] == pytest.approx(2.0 * secs["labels"])
    assert secs["labels"] == len(rows)
    assert len(rows) == teacher["pool_labels"] + teacher["heldout_labels"]
    assert secs["paraphrase_total"] == pytest.approx(2.0 * config.ceiling.n)
    assert report["spent_usd"] == 0
    gold = report["gold"]
    ceiling = report["ceiling"]
    assert isinstance(gold, dict)
    assert isinstance(ceiling, dict)
    assert "errors" in gold
    assert "teacher_vs_gold" in gold
    assert "on_same_items" in ceiling
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["teacher_digest"] == DIGEST
    assert meta["teacher_model"] == MODEL
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["teacher"]["digest"]

    # The run renders as a parity row, a details block and a fallible-teacher row.
    runs = render.load_runs(tmp_path / "results")
    details = render.render_classify(runs)
    assert "| toy | qwen2.5:3b | tiny |" in details
    assert f"weights sha256 `{DIGEST[:12]}`" in details
    assert "Student minus teacher, agreement with gold on the same" in details
    assert "held-out inputs the teacher got wrong, the student gave the teacher's label" in details
    assert f"On the ceiling's {config.ceiling.n} inputs: teacher vs itself" in details
    assert "not a parity-with-Claude number" not in details
    fallible = render.render_fallible(runs)
    errors = gold["errors"]
    assert isinstance(errors, dict)
    assert render._diff(errors["student_minus_teacher_vs_gold_kappa"]) in fallible
    assert "| toy | qwen2.5:3b | tiny |" in fallible
    assert "No run with a model as the teacher" in render.render_fallible([])

    # A fresh checkout: new home, no cache, a server that fails on any request.
    monkeypatch.setattr(
        classify,
        "build_teacher",
        lambda s, settings, provider=None: OllamaTeacher(s, NoServer(), cache=None),
    )
    again = classify.run(
        config,
        data,
        "ollama",
        tmp_path / "results2",
        _settings(tmp_path, "home2"),
        n_resamples=100,
        provenance={"git_sha": "x", "git_dirty": False},
    )
    assert again["spent_usd"] == 0
    a1, a2 = report["agreement"], again["agreement"]
    assert isinstance(a1, dict)
    assert isinstance(a2, dict)
    assert a2["kappa"] == pytest.approx(a1["kappa"])
    assert a2["n"] == a1["n"]
    t2 = again["teacher"]
    assert isinstance(t2, dict)
    assert t2["provider"] == "ollama"
    assert t2["pool_labels"] == teacher["pool_labels"]


def test_label_only_records_labels_and_writes_no_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, data = _local_toy(tmp_path)
    monkeypatch.setattr(classify, "LABELS_DIR", tmp_path / "labels")
    _patch_teacher(monkeypatch, GoldFlips(config, data), tmp_path / "cache")
    out = tmp_path / "results" / "classify-toy-qwen2.5-3b"
    report = classify.run(
        config,
        data,
        "ollama",
        out,
        _settings(tmp_path),
        n_resamples=50,
        provenance={"git_sha": "x", "git_dirty": False},
        label_only=True,
    )
    assert report["label_only"] is True
    teacher = report["teacher"]
    assert isinstance(teacher, dict)
    assert teacher["pool_labels"] == 160
    assert 0 < teacher["heldout_labels"] <= 60
    assert teacher["failures"] == 0
    assert not (out / "report.json").exists()
    assert classify.label_file("toy", MODEL, config.spec.prompt_hash).exists()
    assert "recorded under eval/labels/" in classify.summary_line(report, out)


# --- eval.noise: the synthetic corruptions --------------------------------------------------

CLASSES = ["a", "b", "c", "d"]


def test_corrupt_uniform_changes_exactly_n_labels_to_other_classes() -> None:
    gold = [CLASSES[i % 4] for i in range(60)]
    out = noise.corrupt_uniform(gold, CLASSES, 17, seed=3)
    changed = [(g, o) for g, o in zip(gold, out, strict=True) if g != o]
    assert len(changed) == 17
    assert all(o in CLASSES and o != g for g, o in changed)
    assert out == noise.corrupt_uniform(gold, CLASSES, 17, seed=3)
    assert out != noise.corrupt_uniform(gold, CLASSES, 17, seed=4)
    assert noise.corrupt_uniform(gold, CLASSES, 0, seed=3) == gold
    assert all(
        g != o for g, o in zip(gold, noise.corrupt_uniform(gold, CLASSES, 60, 1), strict=True)
    )


@pytest.mark.parametrize("n_wrong", [-1, 61])
def test_corrupt_uniform_rejects_an_impossible_count(n_wrong: int) -> None:
    with pytest.raises(ValueError, match="cannot corrupt"):
        noise.corrupt_uniform(["a"] * 60, CLASSES, n_wrong, 0)


_labels = st.lists(st.sampled_from(CLASSES), min_size=1, max_size=60)


@st.composite
def _gold_and_teacher(draw: st.DrawFn) -> tuple[list[str], list[str]]:
    gold = draw(_labels)
    teacher = [draw(st.sampled_from(CLASSES)) for _ in gold]
    return gold, teacher


@settings(max_examples=40, deadline=None)
@given(pair=_gold_and_teacher(), seed=st.integers(0, 5))
def test_corrupt_pairs_reproduces_the_teachers_confusions(
    pair: tuple[list[str], list[str]], seed: int
) -> None:
    gold, teacher = pair
    out = noise.corrupt_pairs(gold, teacher, seed)
    assert len(out) == len(gold)
    assert set(out) <= set(CLASSES)

    def cells(labels: list[str]) -> Counter[tuple[str, str]]:
        return Counter((g, x) for g, x in zip(gold, labels, strict=True) if g != x)

    assert cells(out) == cells(teacher)
    for c in CLASSES:
        wrong_out = Counter(x for g, x in zip(gold, out, strict=True) if g == c and x != c)
        wrong_teacher = Counter(x for g, x in zip(gold, teacher, strict=True) if g == c and x != c)
        assert wrong_out == wrong_teacher
    assert out == noise.corrupt_pairs(gold, teacher, seed)


def test_corrupt_pairs_moves_errors_to_other_inputs_and_varies_with_the_seed() -> None:
    gold = ["a"] * 30 + ["b"] * 30
    teacher = ["b"] * 5 + ["a"] * 25 + ["a"] * 4 + ["b"] * 26
    outs = {tuple(noise.corrupt_pairs(gold, teacher, s)) for s in range(5)}
    assert len(outs) > 1


def test_corrupt_pairs_rejects_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        noise.corrupt_pairs(["a", "b"], ["a"], 0)


# --- eval.noise.summarise -------------------------------------------------------------------


def _run(
    condition: str,
    seed: int,
    k_gold: float,
    k_teacher: float,
    copied: int,
    n: int,
    predicted: list[str],
) -> dict[str, object]:
    return {
        "condition": condition,
        "seed": seed,
        "vs_gold": {"kappa": k_gold},
        "vs_teacher": {"kappa": k_teacher},
        "errors": {"on_teacher_errors": {"n": n, "copied": copied}},
        "predicted": predicted,
    }


def test_summarise_aggregates_per_condition_and_pairs_per_seed() -> None:
    gold = ["a", "b", "a", "b", "a", "b", "a", "b"]
    teach0 = ["a", "b", "a", "b", "a", "b", "b", "a"]
    unif0 = ["a", "b", "b", "a", "a", "b", "b", "a"]
    teach1 = ["a", "b", "a", "b", "b", "a", "b", "a"]
    unif1 = ["a", "b", "a", "b", "a", "b", "a", "a"]
    runs = [
        _run("teacher", 0, 0.5, 0.9, 1, 2, teach0),
        _run("teacher", 1, 0.3, 0.7, 0, 4, teach1),
        _run("uniform", 0, 0.1, 0.2, 0, 0, unif0),
        _run("uniform", 1, 0.6, 0.4, 2, 4, unif1),
    ]
    out = noise.summarise(runs, gold, 50)
    teacher = out["teacher"]
    assert isinstance(teacher, dict)
    assert teacher["seeds"] == 2
    assert teacher["kappa_vs_gold"] == {"mean": pytest.approx(0.4), "min": 0.3, "max": 0.5}
    assert teacher["kappa_vs_teacher"] == {"mean": pytest.approx(0.8), "min": 0.7, "max": 0.9}
    assert teacher["copied_rate"] == {"mean": pytest.approx(0.25), "min": 0.0, "max": 0.5}
    uniform = out["uniform"]
    assert isinstance(uniform, dict)
    # No error rows: the copied rate is 0 rather than a division by zero.
    assert uniform["copied_rate"] == {"mean": pytest.approx(0.25), "min": 0.0, "max": 0.5}

    paired = out["paired_vs_gold"]
    assert isinstance(paired, dict)
    assert set(paired) == {"teacher_minus_uniform"}  # no pair or gold runs
    cells = paired["teacher_minus_uniform"]
    assert [c["seed"] for c in cells] == [0, 1]
    for cell, a, b in zip(cells, (teach0, teach1), (unif0, unif1), strict=True):
        assert cell["estimate"] == pytest.approx(cohen_kappa(gold, a) - cohen_kappa(gold, b))
        assert cell["low"] <= cell["high"]
        assert cell["n_resamples"] == 50


def test_summarise_skips_seeds_without_a_matching_condition() -> None:
    gold = ["a", "b", "a", "b"]
    pred = ["a", "b", "b", "b"]
    runs = [
        _run("teacher", 0, 0.5, 0.5, 0, 0, pred),
        _run("teacher", 1, 0.5, 0.5, 0, 0, pred),
        _run("gold", 1, 1.0, 1.0, 0, 0, gold),
    ]
    paired = noise.summarise(runs, gold, 20)["paired_vs_gold"]
    assert isinstance(paired, dict)
    assert list(paired) == ["teacher_minus_gold"]
    assert [c["seed"] for c in paired["teacher_minus_gold"]] == [1]


# --- eval.noise.run_dataset end to end ------------------------------------------------------


def test_run_dataset_trains_each_condition_once_and_reuses_students(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, data = _local_toy(tmp_path)
    spec = config.spec
    labels_dir = tmp_path / "labels"
    monkeypatch.setattr(classify, "LABELS_DIR", labels_dir)
    k = 7
    names = spec.label_names
    rows = []
    for i, it in enumerate(data.items):
        lab = data.gold[it.id]
        if i % k == 0:
            lab = names[(names.index(lab) + 1) % len(names)]
        rows.append(Label(it.id, lab, MODEL, spec.prompt_hash, 300, 5, 0.0, compute_seconds=1.0))
    write_label_file(classify.label_file("toy", MODEL, spec.prompt_hash), rows)

    monkeypatch.setattr(
        classify,
        "build_teacher",
        lambda s, settings, provider=None: OllamaTeacher(s, NoServer(), cache=None),
    )
    student = StudentSpec(
        model="tiny", epochs=1, min_steps=0, validation_fraction=0.0, max_length=24
    )
    monkeypatch.setattr(noise, "load_config", lambda path: config)
    monkeypatch.setattr(noise, "load_rows", lambda ref: ([], []))
    monkeypatch.setattr(noise, "sample", lambda *a, **kw: data)
    monkeypatch.setattr(noise, "load_frontier_config", lambda: SimpleNamespace(student=student))

    settings = _settings(tmp_path)
    conditions = ("gold", "teacher", "uniform", "pair")
    report = noise.run_dataset("toy", settings, (0,), conditions, 50)

    # The flips on the pool, counted straight from the recorded labels.
    ws = classify.workspace(settings, spec, data)
    with ws.open_store(spec) as store:
        pool_ids = [it.id for it, _ in store.labelled(MODEL, spec.prompt_hash, Split.POOL)]
    flipped = {
        r.input_id for r, it in zip(rows, data.items, strict=True) if r.label != data.gold[it.id]
    }
    expected_wrong = sum(i in flipped for i in pool_ids)
    assert 0 < expected_wrong < len(pool_ids)

    pool = report["pool"]
    assert isinstance(pool, dict)
    assert pool["n"] == len(pool_ids) == 160
    assert pool["teacher_wrong"] == expected_wrong
    assert pool["teacher_error_rate"] == pytest.approx(expected_wrong / 160)
    teacher = report["teacher"]
    assert isinstance(teacher, dict)
    assert teacher["model"] == MODEL
    assert teacher["digest"] == DIGEST
    heldout_n = len(report["heldout"]["input_id"])  # type: ignore[index]
    runs = report["runs"]
    assert isinstance(runs, list)
    assert [r["condition"] for r in runs] == list(conditions)
    for r in runs:
        want = 0 if r["condition"] == "gold" else expected_wrong
        assert r["train_labels_wrong"] == want
        assert r["seed"] == 0
        assert len(r["predicted"]) == heldout_n
        assert -1.0 <= r["vs_gold"]["kappa"] <= 1.0
        assert r["errors"]["on_teacher_errors"]["n"] >= 0
    assert len({r["student_run"] for r in runs}) == 4
    summary = report["summary"]
    assert isinstance(summary, dict)
    assert set(summary) == {*conditions, "paired_vs_gold"}
    paired = summary["paired_vs_gold"]
    assert isinstance(paired, dict)
    assert set(paired) == {"teacher_minus_uniform", "teacher_minus_pair", "teacher_minus_gold"}
    assert report["seeds"] == [0]
    heldout = report["heldout"]
    assert isinstance(heldout, dict)
    assert heldout["n"] == heldout_n
    assert noise.summary_line(report, tmp_path).startswith("toy: student kappa vs gold")
    text = render.render_noise([("noise-toy-qwen2.5-3b", report, {"command": "cmd"})])
    assert f"the teacher got {expected_wrong} (" in text
    for name in ("gold labels", "the teacher's labels", "uniform noise", "teacher's confusions"):
        assert name in text
    assert "the student trained on the teacher's labels minus the one trained on gold" in text
    assert "No noise-control run yet" in render.render_noise([])

    # A second run finds the trained students and does not train again.
    def no_train(*args: object, **kwargs: object) -> object:
        raise AssertionError("a student that is already trained was trained again")

    monkeypatch.setattr(noise, "train", no_train)
    second = noise.run_dataset("toy", settings, (0,), conditions, 50)
    second_runs = second["runs"]
    assert isinstance(second_runs, list)
    assert [r["student_run"] for r in second_runs] == [r["student_run"] for r in runs]
    assert [r["predicted"] for r in second_runs] == [r["predicted"] for r in runs]


def test_results_dir_names_are_file_safe() -> None:
    assert noise.results_dir("toy", "qwen2.5:3b", False).name == "noise-toy-qwen2.5-3b"
    assert noise.results_dir("toy", "qwen2.5:3b", True).name.startswith("noise-toy-qwen2.5-3b")
