"""The extraction cascade study on synthetic committed predictions: the report it writes, its
checks against the source report, error detection, the CLI entry and the RESULTS.md table.
No model and no dataset."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from anvil.errors import ConfigError
from anvil.evaluate.extraction import entity_scores, error_breakdown
from anvil.spans import Span
from eval import extract, extract_cascade, render
from eval.extract_cascade import (
    KINDS,
    RULES,
    TARGETS,
    Predictions,
    check_against,
    detection,
    exact_interval,
    load_predictions,
)
from eval.shift import auroc

Report = dict[str, Any]  # parsed JSON
PROVENANCE: dict[str, object] = {"git_sha": "abc", "git_dirty": False}
TYPES = ("Cuisine", "Location", "Amenity")


def _sentence(
    rng: np.random.Generator, confidence: float
) -> tuple[list[list[object]], list[list[object]]]:
    """Teacher spans on a grid of 6-char words; the student copies them, with a chance of an
    error falling as the confidence rises."""
    teacher: list[list[object]] = []
    student: list[list[object]] = []
    for w in range(int(rng.integers(1, 4))):
        start, end = 8 * w, 8 * w + 6
        kind = str(TYPES[int(rng.integers(0, len(TYPES)))])
        teacher.append([start, end, kind])
        if rng.random() >= 1.5 * (1 - confidence) ** 2:
            student.append([start, end, kind])
            continue
        u = rng.random()
        if u < 0.35:
            student.append([start, end - 2, kind])  # wrong boundary
        elif u < 0.6:
            student.append([start, end, TYPES[(TYPES.index(kind) + 1) % len(TYPES)]])
        elif u < 0.75:
            other = TYPES[(TYPES.index(kind) + 1) % len(TYPES)]
            student.append([start + 1, end, other])  # wrong both
        elif u < 0.9:
            pass  # missed
        else:
            student.append([start, end, kind])
            student.append([200 + 8 * w, 204 + 8 * w, kind])  # spurious
    return teacher, student


def synthetic_report(n: int = 100, seed: int = 0) -> Report:
    rng = np.random.default_rng(seed)
    confidence = np.round(rng.uniform(0.3, 1.0, n), 4)
    teacher, student = [], []
    for c in confidence:
        t, s = _sentence(rng, float(c))
        teacher.append(t)
        student.append(s)
    pred = load_predictions(
        {
            "predictions": {
                "input_id": [f"toy-{i}" for i in range(n)],
                "teacher": teacher,
                "student": student,
                "student_confidence": confidence.tolist(),
            }
        }
    )
    scores = entity_scores(pred.teacher, pred.student)
    return {
        "dataset": {"name": "toyset"},
        "teacher": {"model": "gold"},
        "student": {"model": "answerdotai/ModernBERT-base", "run": "run-1"},
        "agreement": {
            "errors": asdict(error_breakdown(pred.teacher, pred.student)),
            "entity": {"f1": scores.f1, "tp": scores.tp, "fp": scores.fp, "fn": scores.fn},
        },
        "predictions": {
            "input_id": pred.ids,
            "teacher": teacher,
            "student": student,
            "student_confidence": confidence.tolist(),
        },
    }


def _write(tmp_path: Path, report: Report, name: str = "report.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _run(path: Path, out: Path) -> dict[str, object]:
    return extract_cascade.run(
        path,
        out,
        n_splits=5,
        n_resamples=100,
        seed=0,
        command="x",
        provenance=PROVENANCE,
    )


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Report, Report]:
    """One run of the study shared by the tests that read it: (source path, report, meta)."""
    tmp = tmp_path_factory.mktemp("cascade")
    path = _write(tmp, synthetic_report())
    out = tmp / "out"
    _run(path, out)
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    return path, report, meta


def _add(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: a.get(k, 0) + b.get(k, 0) for k in a.keys() | b.keys()}


def test_run_writes_report_and_meta(study: tuple[Path, Report, Report]) -> None:
    path, report, meta = study
    source = synthetic_report()
    assert report["n"] == 100
    assert report["errors"] == source["agreement"]["errors"]
    assert report["f1"] == source["agreement"]["entity"]["f1"]
    assert report["targets"] == list(TARGETS)
    assert report["rules"] == list(RULES)
    assert report["source"]["dataset"] == "toyset"
    assert report["source"]["teacher_model"] == "gold"
    assert report["source"]["report"] == str(path)
    assert meta["command"] == "x"
    assert (meta["git_sha"], meta["git_dirty"]) == ("abc", False)
    assert (meta["seed"], meta["n_resamples"], meta["splits"]) == (0, 100, 5)
    assert meta["spent_usd"] == 0.0
    assert "wall_seconds" in meta
    assert "finished_at" in meta
    assert report["curve"][0]["level"] == 1.0
    assert report["curve"][0]["coverage"] == 1.0


def test_thresholds_split_the_errors_and_bound_rules_cover_less(
    study: tuple[Path, Report, Report],
) -> None:
    _, report, _ = study
    found = 0
    for target in TARGETS:
        by_rule = report["thresholds"][str(target)]
        assert set(by_rule) == set(RULES)
        for rule in RULES:
            c = by_rule[rule]
            if c is None:
                continue
            found += 1
            assert c["value"] >= target
            assert c["accepted"] >= extract_cascade.MIN_ACCEPTED
            above, below = c["above"], c["below"]
            assert above["sentences"] + below["sentences"] == report["n"]
            assert above["sentences"] == c["accepted"]
            assert above["f1"] == pytest.approx(c["value"], abs=1e-12)
            assert _add(above["errors"], below.get("errors", {})) == report["errors"]
            if rule == "point":
                assert c["low"] is None
            else:
                assert c["low"] >= target
            # The cascade F1 counts the escalated sentences as perfect.
            assert c["system_f1"] >= report["f1"]
        point = by_rule["point"]
        for rule in ("lower", "bca"):
            if by_rule[rule] is not None:
                assert point is not None
                assert by_rule[rule]["coverage"] <= point["coverage"]
    assert found == len(TARGETS) * len(RULES)  # the synthetic data supports every cut
    # A higher target never serves a larger coverage under the same rule.
    for rule in RULES:
        lo, hi = (report["thresholds"][str(t)][rule] for t in TARGETS)
        if hi is not None:
            assert lo is not None
            assert hi["coverage"] <= lo["coverage"]


def test_split_study_fields(study: tuple[Path, Report, Report]) -> None:
    _, report, _ = study
    ss = report["split_study"]
    assert (ss["splits"], ss["fit_size"], ss["test_size"]) == (5, 50, 50)
    assert (ss["seed"], ss["n_resamples"], ss["min_accepted"]) == (0, 100, 30)
    assert ss["targets"]["0.9"]["point"]["found_splits"] > 0
    for target in TARGETS:
        for rule in RULES:
            e = ss["targets"][str(target)][rule]
            assert e["found_splits"] == round(e["found"] * 5)
            assert 0 <= e["met_splits"] <= e["found_splits"]
            assert e["missed_share"] == (e["found_splits"] - e["met_splits"]) / 5
            if e["found_splits"]:
                assert e["met"] == e["met_splits"] / e["found_splits"]
                low, high = e["met_interval"]
                assert 0.0 <= low <= e["met"] <= high <= 1.0
            else:
                assert e["met"] is None
                assert e["met_interval"] is None


def test_run_is_deterministic_and_report_has_no_timings(tmp_path: Path) -> None:
    path = _write(tmp_path, synthetic_report(seed=3))
    _run(path, tmp_path / "a")
    _run(path, tmp_path / "b")
    a = (tmp_path / "a" / "report.json").read_text(encoding="utf-8")
    b = (tmp_path / "b" / "report.json").read_text(encoding="utf-8")
    assert a == b
    for word in ("wall", "seconds", "finished", "time"):
        assert word not in a


def test_exact_interval() -> None:
    low, high = exact_interval(5, 5)
    assert high == 1.0
    assert low == pytest.approx(0.025 ** (1 / 5))
    low, high = exact_interval(0, 5)
    assert low == 0.0
    assert high == pytest.approx(1 - 0.025 ** (1 / 5))


# --- checks on the source report ----------------------------------------------------------


def test_check_against_catches_a_stale_report(tmp_path: Path) -> None:
    report = synthetic_report(n=70)
    report["agreement"]["errors"]["missed"] += 1
    with pytest.raises(ConfigError, match="out of date"):
        _run(_write(tmp_path, report), tmp_path / "out")
    report = synthetic_report(n=70)
    report["agreement"]["entity"]["f1"] += 0.01
    with pytest.raises(ConfigError, match="F1"):
        _run(_write(tmp_path, report), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_check_against_passes_and_skips() -> None:
    analysis = {"errors": {"correct": 1}, "f1": 0.5}
    check_against({"agreement": {"errors": {"correct": 1}, "entity": {"f1": 0.5}}}, analysis)
    check_against({}, analysis)  # no agreement block: nothing to check against
    check_against({"agreement": {"entity": {"f1": None}}}, analysis)
    with pytest.raises(ConfigError, match="error counts"):
        check_against({"agreement": {"errors": {"correct": 2}}}, analysis)
    with pytest.raises(ConfigError, match="F1"):
        check_against({"agreement": {"entity": {"f1": 0.6}}}, analysis)


def _block() -> dict[str, Any]:
    return {
        "input_id": ["a", "b"],
        "teacher": [[[0, 3, "X"]], []],
        "student": [[[0, 3, "X"]], [[4, 6, "X"]]],
        "student_confidence": [0.9, 0.5],
    }


def test_load_predictions_reads_a_block() -> None:
    pred = load_predictions({"predictions": _block()})
    assert pred.ids == ["a", "b"]
    assert pred.teacher == [[Span(0, 3, "X")], []]
    assert pred.student == [[Span(0, 3, "X")], [Span(4, 6, "X")]]
    assert pred.confidence.tolist() == [0.9, 0.5]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"predictions": None}, "no predictions block"),
        ({"drop": "student_confidence"}, "lack student_confidence"),
        ({"drop": "teacher"}, "lack teacher"),
        ({"set": ("student_confidence", [0.9])}, "one teacher list"),
        ({"set": ("input_id", ["a"])}, "one teacher list"),
        ({"set": ("student", [[[0, 3, "X"]]])}, "one teacher list"),
        ({"set": ("input_id", [])}, "one teacher list"),
        ({"set": ("teacher", [[[0, 3]], []])}, "malformed span"),
        ({"set": ("student", [[["a", 3, "X"]], []])}, "malformed span"),
        ({"set": ("teacher", "nope")}, "must be a list"),
        ({"set": ("student_confidence", [0.9, float("nan")])}, "NaN"),
    ],
)
def test_load_predictions_rejects(change: dict[str, Any], match: str) -> None:
    block: dict[str, Any] = _block()
    report: dict[str, Any] = {"predictions": block}
    if "predictions" in change:
        report = {}
    elif "drop" in change:
        del block[change["drop"]]
    else:
        key, value = change["set"]
        block[key] = value
        if key == "input_id" and not value:
            block.update(teacher=[], student=[], student_confidence=[])
    with pytest.raises(ConfigError, match=match):
        load_predictions(report)


# --- detection ----------------------------------------------------------------------------


def test_detection_on_a_known_ranking() -> None:
    # 10 clean sentences at confidence 0.60..0.96, 5 boundary errors below all of them, 5 type
    # errors interleaved with the clean ones.
    t = Span(0, 5, "A")
    teacher: list[list[Span]] = []
    student: list[list[Span]] = []
    conf: list[float] = []
    for i in range(10):
        teacher.append([t])
        student.append([t])
        conf.append(0.6 + 0.04 * i)
    for i in range(5):
        teacher.append([t])
        student.append([Span(0, 4, "A")])
        conf.append(0.1 + 0.01 * i)
    for i in range(5):
        teacher.append([t])
        student.append([Span(0, 5, "B")])
        conf.append(0.62 + 0.08 * i)
    pred = Predictions([str(i) for i in range(20)], teacher, student, np.array(conf))
    out = detection(pred, n_resamples=200, seed=0)
    assert out["clean_sentences"] == 10
    assert out["boundary_and_type"] == 0
    clean: Any = out["clean"]
    boundary = clean["by_kind"]["wrong_boundary"]
    assert boundary["sentences"] == 5
    assert boundary["auroc"]["estimate"] == 1.0
    assert boundary["auroc"]["low"] == 1.0
    score = -np.array(conf)
    type_auroc = auroc(score[15:], score[:10])
    assert 0.0 < type_auroc < 1.0
    assert clean["by_kind"]["wrong_type"]["auroc"]["estimate"] == type_auroc
    diff = clean["boundary_minus_type"]
    assert diff["estimate"] == pytest.approx(1.0 - type_auroc)
    assert diff["low"] <= diff["estimate"] <= diff["high"]
    # Kinds with no sentence carry only their count.
    for kind in ("wrong_both", "missed", "spurious"):
        assert clean["by_kind"][kind] == {"sentences": 0}
    assert set(clean["by_kind"]) == {"any_error", *KINDS}
    rest: Any = out["rest"]
    assert "any_error" not in rest["by_kind"]
    assert set(rest["by_kind"]) == set(KINDS)
    # Against every other sentence the boundary errors still rank lowest.
    assert rest["by_kind"]["wrong_boundary"]["auroc"]["estimate"] == 1.0
    assert "boundary_minus_type" in rest


def test_detection_without_type_errors_has_no_difference() -> None:
    t = Span(0, 5, "A")
    teacher = [[t] for _ in range(6)]
    student = [[t], [t], [t], [Span(0, 4, "A")], [t], [Span(0, 4, "A")]]
    pred = Predictions([str(i) for i in range(6)], teacher, student, np.linspace(0.9, 0.4, 6))
    out: Any = detection(pred, n_resamples=50, seed=0)
    assert "boundary_minus_type" not in out["clean"]
    assert out["clean"]["by_kind"]["wrong_type"] == {"sentences": 0}


def test_split_study_needs_enough_sentences(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="at least 60 sentences"):
        _run(_write(tmp_path, synthetic_report(n=50)), tmp_path / "out")


# --- the CLI ------------------------------------------------------------------------------


@pytest.fixture
def no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extract, "git_provenance", lambda: dict(PROVENANCE))


@pytest.mark.usefixtures("no_git")
def test_main_runs_the_study_on_predictions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, synthetic_report(n=70))
    out = tmp_path / "x-smoke"
    argv = ["--predictions", str(path), "--out", str(out), "--splits", "3", "--resamples", "50"]
    assert extract.main(argv) == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert report["split_study"]["splits"] == 3
    assert meta["n_resamples"] == 50
    assert meta["git_sha"] == "abc"
    assert meta["command"] == "python -m eval.extract " + " ".join(argv)
    assert "toyset cascade (lower)" in capsys.readouterr().out


@pytest.mark.usefixtures("no_git")
def test_main_refuses_a_smoke_run_into_a_results_name(tmp_path: Path) -> None:
    path = _write(tmp_path, synthetic_report(n=70))
    argv = ["--predictions", str(path), "--out", str(tmp_path / "x"), "--splits", "2"]
    assert extract.main([*argv, "--resamples", "40"]) == 2
    assert not (tmp_path / "x").exists()
    assert not (tmp_path / "x-smoke").exists()


@pytest.mark.usefixtures("no_git")
def test_main_rejects_a_report_that_is_not_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    argv = ["--predictions", str(bad), "--out", str(tmp_path / "j-smoke"), "--splits", "2"]
    assert extract.main(argv) == 2


@pytest.mark.usefixtures("no_git")
def test_main_rejects_bad_invocations(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path, synthetic_report(n=70))
    with pytest.raises(SystemExit) as exc:
        extract.main(["--predictions", str(path), "mit_restaurant"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        extract.main([])
    assert exc.value.code == 2
    capsys.readouterr()
    missing = tmp_path / "missing.json"
    argv = ["--predictions", str(missing), "--out", str(tmp_path / "m-smoke"), "--splits", "2"]
    assert extract.main(argv) == 2
    assert "no report" in capsys.readouterr().err
    stale = synthetic_report(n=70)
    stale["agreement"]["errors"]["missed"] += 1
    bad = _write(tmp_path, stale, "stale.json")
    argv = ["--predictions", str(bad), "--out", str(tmp_path / "s-smoke"), "--splits", "2"]
    assert extract.main([*argv, "--resamples", "20"]) == 2
    assert "out of date" in capsys.readouterr().err


@pytest.mark.usefixtures("no_git")
@pytest.mark.parametrize("override", [["--splits", "0"], ["--resamples", "0"]])
def test_main_rejects_non_positive_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], override: list[str]
) -> None:
    path = _write(tmp_path, synthetic_report(n=70))
    argv = ["--predictions", str(path), "--out", str(tmp_path / "z-smoke"), *override]
    assert extract.main(argv) == 2
    assert "must be positive" in capsys.readouterr().err
    assert not (tmp_path / "z-smoke").exists()


# --- RESULTS.md -----------------------------------------------------------------------------


def test_render_extract_cascade(study: tuple[Path, Report, Report]) -> None:
    _, report, meta = study
    text = render.render_extract_cascade([("cascade-extract-toyset-gold", report, meta)])
    assert "toyset" in text
    assert "eval/results/cascade-extract-toyset-gold/" in text
    assert "| Target F1 | Rule | Threshold | Sentences answered | F1 on answered" in text
    assert "| Target F1 | Rule | Threshold found | Unseen half: kept the target" in text
    assert text.count("\n| 0.9 |") >= len(RULES)
    assert render.render_extract_cascade([]).startswith("No extraction cascade study yet")


def test_render_extract_cascade_reads_runs_from_disk(tmp_path: Path) -> None:
    path = _write(tmp_path, synthetic_report(n=70))
    _run(path, tmp_path / "results" / "cascade-extract-toyset-gold")
    _run(path, tmp_path / "results" / "cascade-extract-toyset-gold-smoke")
    runs = render.load_runs(tmp_path / "results", prefix="cascade-extract-")
    assert [name for name, _, _ in runs] == ["cascade-extract-toyset-gold"]
    assert "Threshold found" in render.render_extract_cascade(runs)
