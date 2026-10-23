"""The calibration harness on synthetic logits: report shape, the --logits path, rendering."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from anvil.calibrate import softmax
from eval import calibration
from eval.calibration import COVERAGE_GRID, METHODS, N_BINS, TARGETS, analyse, curve_at, main
from eval.render import render_calibration

N, K, SPLITS, RESAMPLES = 300, 5, 20, 200


def synthetic() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Overconfident logits: calibrated ones scaled by 3, so the fitted temperature is ~3."""
    rng = np.random.default_rng(0)
    calibrated = rng.normal(size=(N, K))
    calibrated[np.arange(N), rng.integers(0, K, N)] += 4.0
    probs = np.exp(calibrated - calibrated.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    teacher = np.array([rng.choice(K, p=p) for p in probs], dtype=np.int64)
    return 3 * calibrated, teacher


@cache
def report() -> dict[str, Any]:
    z, y = synthetic()
    return analyse(z, y, SPLITS, RESAMPLES, seed=0)


# Any: the report is nested JSON read back by key; there is no useful static type.
def section(r: dict[str, Any], key: str) -> dict[str, Any]:
    value = r[key]
    assert isinstance(value, dict)
    return value


def test_report_shape() -> None:
    r = report()
    assert r["n"] == N
    assert r["labels"] == K
    assert isinstance(r["kappa"], float)
    full = section(r, "full")
    assert set(full) == {"0.9", "0.95"}
    rel = section(r, "reliability")
    for side in ("before", "after"):
        bins = rel[side]
        assert isinstance(bins, list)
        assert len(bins) == N_BINS
        assert sum(b["n"] for b in bins) == N
    curve = r["curve"]
    assert isinstance(curve, list)
    assert curve
    for row in curve:
        assert row["level"] in COVERAGE_GRID
        assert row["coverage"] <= row["level"] + 1e-9
    study = section(r, "split_study")
    assert study["fit_size"] + study["test_size"] == N
    assert study["splits"] == SPLITS
    targets = section(study, "targets")
    assert set(targets) == {str(t) for t in TARGETS}
    for by_method in targets.values():
        assert set(by_method) == {name for name, _, _ in METHODS}


def test_temperature_is_fitted_near_three_and_calibration_helps() -> None:
    r = report()
    served = section(section(r, "full"), "0.9")
    assert 2.0 < served["temperature"] < 4.5
    study = section(r, "split_study")
    assert section(study, "ece_after")["median"] < section(study, "ece_before")["median"]
    assert served["ece_after"] < served["ece_before"]


def test_lower_rule_answers_no_more_than_point_rule() -> None:
    targets = section(section(report(), "split_study"), "targets")
    for by_method in targets.values():
        point, lower = by_method["point"], by_method["lower"]
        if "coverage" in point and "coverage" in lower:
            assert lower["coverage"]["median"] <= point["coverage"]["median"] + 1e-12


def test_count_rules_are_served_and_never_answer_more_than_point() -> None:
    r = report()
    by_rule = section(r, "by_rule")
    assert set(by_rule) == {"point", "lower", "bca", "exact", "wilson"}
    for target in ("0.9", "0.95"):
        point = by_rule["point"][target]
        for rule in ("exact", "wilson"):
            row = by_rule[rule][target]
            if row["threshold"] is None:
                continue
            assert row["kappa_low"] <= row["kappa"]
            assert row["kappa_low"] >= float(target)
            assert row["coverage"] <= point["coverage"]


def test_analyse_is_deterministic() -> None:
    z, y = synthetic()
    assert analyse(z, y, 5, 50, seed=3) == analyse(z, y, 5, 50, seed=3)


def test_curve_ends_at_full_coverage() -> None:
    z, y = synthetic()
    rows = curve_at(z, y, 3.0)
    assert rows[0]["level"] == 1.0
    assert rows[0]["coverage"] == 1.0


def write_logits(directory: Path, source: dict[str, str]) -> Path:
    z, y = synthetic()
    directory.mkdir(parents=True)
    npz = directory / "logits.npz"
    np.savez_compressed(
        npz,
        logits=z.astype(np.float32),
        teacher=y,
        input_ids=np.array([f"i{k}" for k in range(N)]),
        labels=np.array([f"l{k}" for k in range(K)]),
        teacher_model=np.array("gold-x"),
    )
    (directory / "report.json").write_text(json.dumps({"source": source}), encoding="utf-8")
    return npz


def test_main_from_logits_writes_a_smoke_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {"dataset": "banking77", "teacher_kind": "gold", "teacher_model": "gold-x"}
    npz = write_logits(tmp_path / "in", source)
    results = tmp_path / "results"
    monkeypatch.setattr(calibration, "RESULTS_DIR", results)
    code = main(["banking77", "--logits", str(npz), "--splits", "5", "--resamples", "50"])
    assert code == 0
    out = results / "calibration-banking77-gold-x-smoke"
    written = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert written["source"] == source
    assert written["n"] == N
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["splits"] == 5
    assert meta["from_logits"] == str(npz)
    assert meta["spent_usd"] == 0.0


def test_main_without_previous_report_is_a_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    npz = write_logits(tmp_path / "in", {})
    (npz.parent / "report.json").unlink()
    monkeypatch.setattr(calibration, "RESULTS_DIR", tmp_path / "results")
    assert main(["banking77", "--logits", str(npz), "--splits", "2", "--resamples", "10"]) == 2


def test_main_logits_takes_one_dataset(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["banking77", "ag_news", "--logits", str(tmp_path / "x.npz")])
    assert exc.value.code == 2


def test_render_calibration_without_runs() -> None:
    assert "No calibration run yet" in render_calibration([])


def test_render_calibration_of_a_report() -> None:
    source = {
        "dataset": "banking77",
        "teacher_model": "gold-x",
        "student_model": "tiny",
        "student_run": "run-1",
        "teacher_kind": "gold",
    }
    text = render_calibration(
        [("calibration-banking77-gold-x", {"source": source, **report()}, {"command": "cmd"})]
    )
    assert "banking77" in text
    assert "| Dataset | Teacher | Student |" in text
    assert "95% percentile lower bound, calibrated" in text
    assert "The teacher here is the dataset's gold labels" in text


def test_find_student_refuses_two_students_without_recorded_settings(tmp_path: Path) -> None:
    import json

    from anvil.errors import ConfigError
    from anvil.pipeline import Workspace
    from anvil.spec import TaskSpec
    from anvil.students.encoder import META_FILE
    from eval.student import find_student

    spec = TaskSpec.model_validate(
        {"name": "t", "kind": "classify", "instructions": "x", "labels": ["a", "b"]}
    )
    ws = Workspace(tmp_path / "ws")
    base = {
        "teacher_model": "gold",
        "prompt_hash": spec.prompt_hash,
        "student_model": spec.student.model,
        "splits": spec.splits.model_dump(),
        "train_labels": ["a", "b"],
    }

    def save(run: str, **extra: object) -> None:
        (ws.students / run).mkdir(parents=True)
        (ws.students / run / META_FILE).write_text(json.dumps({**base, **extra}), "utf-8")

    save("20260101-old")
    assert find_student(ws, spec, "gold", 2) == ws.students / "20260101-old"
    save("20260102-seed1")
    with pytest.raises(ConfigError, match="record no seed"):
        find_student(ws, spec, "gold", 2)
    other = spec.student.model_copy(update={"seed": 1}).model_dump()
    save("20260103-new-seed1", student_spec=other)
    save("20260104-new-seed0", student_spec=spec.student.model_dump())
    assert find_student(ws, spec, "gold", 2) == ws.students / "20260104-new-seed0"


def _repeat(y: npt.NDArray[np.int64], asked: int, flips: int) -> npt.NDArray[np.int64]:
    """A second labelling of the first `asked` inputs with every `flips`-th one changed."""
    repeat = np.full(y.size, -1, dtype=np.int64)
    repeat[:asked] = y[:asked]
    repeat[:asked:flips] = (y[:asked:flips] + 1) % K
    return repeat


def test_ceiling_on_accepted_inputs_matches_a_direct_count() -> None:
    z, y = synthetic()
    repeat = _repeat(y, 120, 4)
    t = 1.5
    out = calibration.ceiling_on_accepted(z, y, repeat, t, {"0.9": 0.6, "0.95": None}, 100, 0)
    assert out["asked"] == 120
    conf = softmax(z, t).max(axis=1)
    by_level = out["by_level"]
    assert isinstance(by_level, list)
    assert [row["level"] for row in by_level] == list(COVERAGE_GRID)
    full = by_level[0]  # level 1.0 accepts every input, so it covers the whole ceiling set
    assert full["coverage"] == 1.0
    assert full["n"] == 120
    expected = calibration._kappa(y[:120], repeat[:120], K)
    assert full["ceiling"]["kappa"] == pytest.approx(expected)
    assert full["ceiling"]["low"] <= expected <= full["ceiling"]["high"]
    for row in by_level:
        if "ceiling" in row:
            assert row["n"] >= calibration.MIN_CEILING_ITEMS
    served = out["served"]
    assert isinstance(served, dict)
    assert served["0.95"] is None
    accepted = conf >= 0.6
    assert served["0.9"]["n"] == int(accepted[:120].sum())
    assert served["0.9"]["coverage"] == pytest.approx(float(accepted.mean()))


def test_analyse_reports_the_ceiling_only_with_a_second_labelling() -> None:
    z, y = synthetic()
    assert "ceiling" not in report()
    with_repeat = analyse(z, y, 2, 50, seed=0, repeat=_repeat(y, 100, 5))
    ceiling = section(with_repeat, "ceiling")
    assert set(section(ceiling, "served")) == {str(t) for t in TARGETS}
    nobody = analyse(z, y, 2, 50, seed=0, repeat=np.full(N, -1, dtype=np.int64))
    assert "ceiling" not in nobody


def test_render_calibration_shows_the_ceiling_on_accepted_inputs() -> None:
    z, y = synthetic()
    r = analyse(z, y, 2, 50, seed=0, repeat=_repeat(y, 100, 5))
    source = {
        "dataset": "banking77",
        "teacher_model": "qwen2.5:3b",
        "student_model": "tiny",
        "student_run": "run-1",
        "teacher_kind": "ollama",
    }
    text = render_calibration([("calibration-banking77-qwen2.5-3b", {"source": source, **r}, {})])
    assert "teacher `qwen2.5:3b`" in text
    assert "The ceiling on the inputs the cascade answers: of the 100 held-out" in text
    assert "| top 100.0% | 100.0% | 100 |" in text
    assert "served at kappa 0.9" in text
    assert "gold labels" not in text


def test_main_from_logits_replays_the_second_labelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {"dataset": "banking77", "teacher_kind": "ollama", "teacher_model": "qwen2.5:3b"}
    npz = write_logits(tmp_path / "in", source)
    saved = dict(np.load(npz))
    _, y = synthetic()
    saved["repeat"] = _repeat(y, 100, 5)
    saved["teacher_model"] = np.array("qwen2.5:3b")
    np.savez_compressed(npz, **saved)
    monkeypatch.setattr(calibration, "RESULTS_DIR", tmp_path / "results")
    code = main(["banking77", "--logits", str(npz), "--splits", "2", "--resamples", "20"])
    assert code == 0
    out = tmp_path / "results" / "calibration-banking77-qwen2.5-3b-smoke" / "report.json"
    assert json.loads(out.read_text(encoding="utf-8"))["ceiling"]["asked"] == 100


# --- the fallible-teacher cascade against gold ------------------------------------------------


def test_against_gold_scores_the_cascade_and_the_teacher_on_the_same_inputs() -> None:
    # 200 inputs, 4 labels. The student copies the teacher everywhere; the teacher is wrong on
    # the 40 inputs where the student is least confident, so the confident half is all correct.
    n, k = 200, 4
    rng = np.random.default_rng(0)
    gold = rng.integers(0, k, size=n).astype(np.int64)
    teacher = gold.copy()
    margin = np.linspace(4.0, 0.2, n)  # most confident first
    teacher[-40:] = (gold[-40:] + 1) % k
    z = np.zeros((n, k))
    z[np.arange(n), teacher] = margin
    p = softmax(z, 1.0).max(axis=1)
    half = float(np.sort(p)[::-1][n // 2 - 1])
    out: Any = calibration.against_gold(z, teacher, gold, 1.0, {"0.9": half, "0.95": None}, 200, 0)

    assert out["teacher_wrong"] == 40
    served = out["served"]
    assert served["0.95"] is None
    row = served["0.9"]
    assert row["n"] == 100
    assert row["teacher_wrong"] == 0
    assert row["student_vs_teacher"] == pytest.approx(1.0)
    assert row["student_vs_gold"]["kappa"] == pytest.approx(1.0)
    assert row["teacher_vs_gold"]["kappa"] == pytest.approx(1.0)
    # The student is the teacher on the answered inputs, so the cascade is the teacher alone.
    assert row["cascade_vs_gold"]["kappa"] == pytest.approx(out["teacher_vs_gold"]["kappa"])
    assert row["cascade_minus_teacher"] == {"estimate": 0.0, "low": 0.0, "high": 0.0}
    assert [r["level"] for r in out["by_level"]] == list(calibration.GOLD_LEVELS)
    assert out["by_level"][0]["n"] == 180
    # Confidence falls exactly where the teacher goes wrong: a perfect detector either way.
    auc = out["teacher_error_auroc"]
    assert auc["max_softmax"]["auroc"] == pytest.approx(1.0)
    assert auc["teacher_label_probability"]["auroc"] == pytest.approx(1.0)


def test_gold_codes_follow_the_classify_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anvil.errors import ConfigError

    monkeypatch.setattr(calibration, "RESULTS_DIR", tmp_path)
    folder = tmp_path / "classify-toy-qwen2.5-3b"
    folder.mkdir()
    predictions = {"input_id": ["x", "y", "z"], "gold": ["b", "a", "b"]}
    (folder / "report.json").write_text(json.dumps({"predictions": predictions}), "utf-8")
    codes = calibration.gold_codes("toy", "qwen2.5:3b", ["z", "x", "y"], ["a", "b"])
    assert codes is not None
    assert codes.tolist() == [1, 1, 0]
    with pytest.raises(ConfigError, match="no gold label"):
        calibration.gold_codes("toy", "qwen2.5:3b", ["w"], ["a", "b"])
    assert calibration.gold_codes("other", "qwen2.5:3b", ["x"], ["a", "b"]) is None
